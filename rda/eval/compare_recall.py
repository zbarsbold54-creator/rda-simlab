"""
Model A vs Model B anomaly-recall harness.

Compares recall_gap = mean_loss(matching_control_span) - mean_loss(anomaly_span)
for a baseline model (A) and a RDA model (B).

Run:  python eval/compare_recall.py
"""
import os
import sys
import json
import math
import statistics
import importlib.util
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint

# REUSE the generator's sentence utilities; do not reimplement.
_spec = importlib.util.spec_from_file_location(
    "build_anomaly_set", os.path.join(REPO, "eval", "build_anomaly_set.py"))
_bas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bas)
split_sentences = _bas.split_sentences
sentences_text = _bas.sentences_text

EVAL_PATH = os.path.join(REPO, "eval", "eval_anomaly_set.jsonl")
MAX_LEN = 1024
ANOMALY_TYPES = ["factual_contradiction", "topic_break", "token_noise"]


# -----------------------------------------------------------------------------#
# Loading
# -----------------------------------------------------------------------------#
def load_model(tag, base_dir, device, rda=False):
    ckpt_dir = os.path.join(base_dir, tag)
    metas = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("meta_"))
    meta_path = os.path.join(ckpt_dir, metas[-1])
    meta = json.load(open(meta_path))
    # GPTConfig is a dataclass; the checkpoint stores it as a plain dict.
    config = GPTConfig(**meta["model_config"])
    model = GPT(config)
    model.init_weights()
    if rda:
        from nanochat.rda import GPTWithRDA
        model = GPTWithRDA(model, debug_steps=0)

    # load_checkpoint RETURNS (model_data, optim_data, meta) -- it does NOT
    # apply the weights to any model. The previous version of this function
    # called it for its side effect and threw the return value away, so
    # `model` kept its random init and both A and B were scored as untrained
    # networks. Every recall number produced that way is invalid.
    step = int(meta["step"])
    model_data, _, _ = load_checkpoint(ckpt_dir, step, torch.device(device))
    if torch.device(device).type in {"cpu", "mps"}:
        # CPU/MPS inference runs in fp32; the checkpoint may hold bf16.
        model_data = {k: (v.float() if v.dtype == torch.bfloat16 else v)
                      for k, v in model_data.items()}
    # Strip the torch.compile "_orig_mod." prefix if present.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    missing, unexpected = model.load_state_dict(model_data, strict=False)
    # strict=False is required because the wrapper/CPython can report keys the
    # freshly-built module does not declare. Anything ACTUALLY missing means a
    # trained tensor was silently left at random init, which is precisely the
    # bug this is fixing, so it must be fatal rather than a warning.
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {tag} step {step} did not load cleanly: "
            f"missing={missing} unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    return model, meta


# -----------------------------------------------------------------------------#
# Span -> token mapping
# -----------------------------------------------------------------------------#
def token_span(tok, text, char_start, char_end):
    """Map a char span to a token index range via PREFIX TOKENIZATION.

    Method (stated explicitly, as required): tiktoken/RustBPETokenizer exposes
    no offset mapping, so we tokenize the prefix text up to char_start and up to
    char_end separately and take the difference in token counts. BPE is
    context-dependent, so this is an approximation at a span edge; it is exact
    whenever the boundary falls on a token boundary, which is the case for the
    sentence spans used here because the anomaly sentence is inserted with
    surrounding whitespace.
    """
    pre_s = tok.encode(text[:char_start], prepend=None)
    pre_e = tok.encode(text[:char_end], prepend=None)
    return len(pre_s), len(pre_e)


def _unwrap(out):
    """GPT.forward returns either a tensor or a (tensor, hidden) tuple."""
    if isinstance(out, tuple):
        return out[0]
    return out


def per_token_loss(model, tok, text, device):
    """Mean per-token CE loss over the whole text, returned per token."""
    bos = tok.get_bos_token_id()
    ids = torch.tensor([tok.encode(text, prepend=bos)], dtype=torch.long, device=device)
    # Truncate to the model's context window. A bare forward asserts
    # T <= rotary cache, and documents here are far longer than 1024 tokens.
    if ids.size(1) > MAX_LEN:
        ids = ids[:, :MAX_LEN]
    with torch.no_grad():
        if hasattr(model, "state"):
            # RDA wrapper: must use the same FiLM path as training, else we
            # would be evaluating a model that never actually runs.
            logits = _unwrap(_rda_logits(model, ids))
        else:
            logits = _unwrap(model.forward(ids, None, return_hidden=True))
    logits = logits.float()
    lp = torch.log_softmax(logits[0, :-1, :], dim=-1)
    tgt = ids[0, 1:]
    nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return nll  # length T-1, nll[i] predicts token i+1


def _rda_logits(model, ids):
    """Run the RDA wrapper and return (logits, hidden) with FiLM applied."""
    _loss, hidden = model.base.forward(ids, None, return_hidden=True)
    E = model.state.init_state(ids.size(0), ids.device, dtype=torch.float32)
    hidden_mod = model.state.modulate(hidden, E)
    lg = model.base.lm_head(hidden_mod)
    lg = lg[..., :model.base.config.vocab_size].float()
    lg = 15 * torch.tanh(lg / 15)
    return _unwrap(lg), hidden_mod


# -----------------------------------------------------------------------------#
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.expanduser(r"~/.cache/nanochat/base_checkpoints")
    tok = get_tokenizer()

    print("loading models ...")
    model_a, meta_a = load_model("baseline-fwedu-d8", base_dir, device, rda=False)
    model_b, meta_b = load_model("model-b-primal", base_dir, device, rda=True)
    print(f"  A step={meta_a['step']} val_bpb={meta_a['val_bpb']:.6f}")
    print(f"  B step={meta_b['step']} val_bpb={meta_b['val_bpb']:.6f}")

    rows = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")]
    anom = {r["doc_id"]: r for r in rows if r["condition"] == "anomalous"}
    ctrl = {r["doc_id"]: r for r in rows if r["condition"] == "control"}

    skipped = []
    per_doc = []
    for doc_id in sorted(anom):
        a = anom[doc_id]
        c = ctrl[doc_id]
        atext, ctext = a["text"], c["text"]
        a_start, a_end = a["anomaly_char_start"], a["anomaly_char_end"]
        idx = a["anomaly_sentence_index"]

        ctrl_sents = split_sentences(ctext)
        if not (0 <= idx < len(ctrl_sents)):
            skipped.append({"doc_id": doc_id, "reason": "index out of range",
                            "idx": idx, "n_ctrl_sents": len(ctrl_sents)})
            continue
        c_start, c_end = ctrl_sents[idx]

        # Map BOTH spans to token indices on the FULL text, then check they fit
        # inside the model's context window. char offsets are NOT comparable to
        # a token count, so the context check must happen after mapping.
        ta0, ta1 = token_span(tok, atext, a_start, a_end)
        tc0, tc1 = token_span(tok, ctext, c_start, c_end)
        # With BOS prepended, tokens[1..] are scored, so usable loss positions
        # are 0 .. n_tokens-2.
        if ta1 > MAX_LEN - 1 or tc1 > MAX_LEN - 1:
            skipped.append({"doc_id": doc_id, "reason": "span beyond context",
                            "a_tok": [ta0, ta1], "c_tok": [tc0, tc1]})
            continue

        nll_a = per_token_loss(model_a, tok, atext, device)
        nll_b = per_token_loss(model_b, tok, atext, device)
        nll_ca = per_token_loss(model_a, tok, ctext, device)
        nll_cb = per_token_loss(model_b, tok, ctext, device)

        if not (0 <= ta0 < ta1 <= nll_a.numel()) or not (0 <= tc0 < tc1 <= nll_ca.numel()):
            skipped.append({"doc_id": doc_id, "reason": "token span invalid"})
            continue

        def mean(nll, s, e):
            return float(nll[s:e].mean())

        per_doc.append({
            "doc_id": doc_id,
            "anomaly_type": a["anomaly_type"],
            "A": mean(nll_ca, tc0, tc1) - mean(nll_a, ta0, ta1),
            "B": mean(nll_cb, tc0, tc1) - mean(nll_b, ta0, ta1),
            "n_tok_anom": ta1 - ta0,
            "n_tok_ctrl": tc1 - tc0,
        })

    print(f"\ndocs used={len(per_doc)}  skipped={len(skipped)}")

    def agg(sel, key):
        v = [d[key] for d in per_doc if sel(d)]
        if not v:
            return None, None, 0
        m = sum(v) / len(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        return m, sd, len(v)

    print("\n%-26s %10s %10s %10s %10s %10s %6s" %
          ("group", "A_mean", "A_std", "B_mean", "B_std", "B-A", "n"))
    rowsout = []
    for t in ANOMALY_TYPES + ["OVERALL"]:
        sel = (lambda d, t=t: d["anomaly_type"] == t) if t != "OVERALL" else (lambda d: True)
        am, asd, n = agg(sel, "A")
        bm, bsd, _ = agg(sel, "B")
        if am is None:
            continue
        rowsout.append({"group": t, "A_mean": am, "A_std": asd, "B_mean": bm,
                        "B_std": bsd, "B_minus_A": bm - am, "n": n})
        print("%-26s %10.4f %10.4f %10.4f %10.4f %10.4f %6d" %
              (t, am, asd, bm, bsd, bm - am, n))

    out_path = os.path.join(REPO, "eval", "recall_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"per_doc": per_doc, "skipped": skipped,
                   "summary": rowsout,
                   "A": {"step": meta_a["step"], "val_bpb": meta_a["val_bpb"]},
                   "B": {"step": meta_b["step"], "val_bpb": meta_b["val_bpb"]},
                   "film_log": meta_b.get("rda_film_log")}, f, indent=2)
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
