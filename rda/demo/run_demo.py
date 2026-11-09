"""
Interactive demo for the trained Model B (RDA) checkpoint.

READ-ONLY. Trains nothing; loads an existing checkpoint and prints the
per-token affective state trajectory.

Usage
  python demo/run_demo.py --example 1     # built-in example
  python demo/run_demo.py --text "..."    # one-off
  python demo/run_demo.py                 # interactive loop
  add --gpu to run on CUDA (default: CPU, so a demo never reserves the
  training GPU)

What the columns mean
  E_t      the 8-dim regulatory drive state AFTER the token at that position, i.e. the
           state the recurrence has settled into having read up to and
           including that token. This is the quantity the summary analyses.
  surprisal -log p(actual next token | prefix), i.e. how surprised the model
           was by the real next token in your text. For free-form text this
           is the honest, well-defined quantity; it is NOT a perplexity of
           the model's own generation unless you feed it its own output.

The "*" marker flags a token where any dimension moved more than --jump from
the previous token -- the "something happened here" signal.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import get_base_dir

from nanochat.rda import GPTWithRDA, DRIVE_DIMS, N_DRIVES

MAX_LEN = 1024

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "reference_state.json")


def load_reference():
    """Settled final state for a long plain neutral paragraph.

    Generated once by demo/make_reference.py. If the file is absent the demo
    still runs, but the comparison section is skipped with a note.
    """
    if not os.path.isfile(REF_PATH):
        return None
    with open(REF_PATH, encoding="utf-8") as f:
        return json.load(f)

EXAMPLES = {
    1: (
        "Plain expository prose (should sit close to the reference).",
        "A pendulum consists of a weighted bob suspended from a fixed point by "
        "a thin rod or string. When the bob is displaced from its resting "
        "position and released, it swings back and forth through an arc, and "
        "the time taken by a complete swing depends on the length of the "
        "suspension and not on the amplitude of the swing or on the mass of "
        "the bob. This independence is somewhat surprising, since a heavier "
        "bob exerts more force on the support and one might expect that to "
        "change the period. The reason it does not is that the restoring "
        "force is itself proportional to the mass, so the mass cancels out of "
        "the equation of motion entirely. What remains is a simple relationship "
        "between the period and the square root of the length, which was "
        "established by Huygens using a pendulum of adjustable length formed "
        "from two rods pivoted at their ends. The practical consequence was "
        "enormous. For the first time it became possible to measure time "
        "intervals accurately enough to compare with astronomical observations, "
        "and so to relate terrestrial timekeeping to the apparent motion of "
        "the sun and stars. Before this, mechanical clocks had to be "
        "recalibrated frequently against astronomical observations because "
        "their rates drifted. A pendulum clock was far more stable, and the "
        "accuracy of navigation and of the determination of longitude improved "
        "accordingly. The escapement mechanism that releases the pendulum at "
        "each extreme swing was refined over the following century, and the "
        "resulting clocks served as the standard timekeepers for scientific "
        "work well into the nineteenth century, until quartz oscillators "
        "provided a far higher rate stability and eventually atomic clocks "
        "made the pendulum obsolete for the most demanding applications. "
        "Nevertheless the pendulum remains in common use in elementary "
        "laboratory instruction because it is simple to construct, easy to "
        "observe directly, and illustrates several general principles at once. "
        "It demonstrates the relation between period and length, shows the "
        "conservation of energy in a system where potential and kinetic "
        "energy trade places continuously, and provides an accessible example "
        "of resonance when a driving force is applied at a matching "
        "frequency. It is also worth noting that the ideal small angle "
        "assumption underlying the simple formula is genuinely restrictive. "
        "For larger amplitudes the period does increase, and the correction "
        "can be written as a convergent series in powers of the angle squared. "
        "At ten degrees the error is already of order half a percent, which is "
        "far too large for precise work but adequate for a demonstration.",
    ),
    2: (
        "Narrative text with a topic break and a factual contradiction, "
        "styled like eval_anomaly_set.jsonl's anomaly types.",
        "The lighthouse had been automated in 1994, the same year the local "
        "council replaced the original Fresnel lens with a sealed acrylic "
        "panel, and the keeper's cottage was sold to a family from the city "
        "who used it as a holiday home. "
        "Qin Shi Huang died in 210 BC during a tour of the western regions, "
        "with the imperial court riding on wooden carts drawn by horses."
        " The Great Wall had already been completed under the rule of his "
        "father, and he declined to repair the existing walls, choosing "
        "instead to build entirely new sections further to the north. "
        "After his death the work was continued by his successor, and the "
        "structure eventually took on a form recognisable to later dynasties. "
        "Historians have long argued about how much of the surviving masonry "
        "dates from the original period, since the wall was repeatedly "
        "rebuilt and patched over many centuries. What is clear is that the "
        "project was begun under severe time pressure and that the labour was "
        "conscripted from the surrounding population, which suffered "
        "disproportionately as a result. The accounts describe villages "
        "emptied and entire communities redirected to the northern frontier "
        "for periods of years at a time. Those who returned often found their "
        "fields gone, and a substantial share did not return at all. Modern "
        "excavations at several points along the line have turned up "
        "household goods sealed inside the construction fill, including "
        "cooking implements and storage vessels that would ordinarily have "
        "been removed before a family vacated a site. Whether these indicate "
        "deliberate burial as part of a ritual practice or simply that the "
        "workers could not carry everything with them is a matter of ongoing "
        "debate, with the current scholarly consensus favouring the more "
        "mundane explanation. A further complication is that the term used in "
        "the earliest sources does not correspond precisely to any modern "
        "structure, and translating it as wall is a convention adopted by "
        "later commentators whose own political circumstances coloured the "
        "wording they chose. "
        "The lighthouse itself continued in service for another thirty years "
        "before the light was finally extinguished, and the tower was "
        "decommissioned shortly afterwards.",
    ),
    3: (
        "A long ordinary paragraph, deliberately near the reference register.",
        "Bread making is mostly chemistry that happens slowly. Flour, water, "
        "salt and a living culture of yeast and bacteria are mixed into a "
        "dough, and from that point the mixture is on its own schedule. The "
        "yeast cells consume the sugars that starch enzymes have already "
        "broken down, and the carbon dioxide they produce is trapped by an "
        "elastic network of gluten proteins formed when flour was hydrated. "
        "Strength comes from that network rather than from any single "
        "ingredient, which is why the same flour can produce a stiff crust "
        "or a slack one depending on how much water was added and how long "
        "the dough was allowed to develop. Temperature matters just as much: "
        "a warm dough ferments quickly and can sour before the gluten has "
        "had time to organise, while a cool dough develops slowly and tastes "
        "sweeter for it. Bakers learn to read the dough by touch as much as "
        "by clock, pressing a knuckle into the surface to see whether it "
        "springs back, folding and turning the mass every half hour or so to "
        "even out the fermentation. Longer fermentation tends to improve both "
        "the texture of the crumb and the depth of flavour, since the "
        "bacteria responsible for sourness are given more time to act, but it "
        "demands a schedule that fits around the work of the day. Overnight "
        "colds are popular partly for this reason and partly because they "
        "shift the labour into a period when someone is not otherwise "
        "occupied with the stove. "
        "The shaping step matters more than novices expect. A well risen mass "
        "will degas itself if handled roughly, and the gas that escapes cannot "
        "be replaced, so a torn surface or a careless degassing leaves the "
        "loaf with a dense layer that no amount of later proofing will fix. "
        "Gentle handling, a short rest before final shaping, and a banneton "
        "that supports the dough as it rises all contribute to an even "
        "distribution of gas. The scoring on top serves a comparable purpose "
        "steeply and the second is not, so the setting differs by perhaps a "
        "third. Bakers argue at length about the angle and depth of that "
        "cut, and while the argument is unresolved the practical difference "
        "between a well scored and a badly scored loaf is plainly visible. "
        "The oven itself is the last variable and the least forgiving of "
        "mistakes. A domestic oven heats unevenly, and a loaf placed near the "
        "element browns before the interior has set, while the same loaf "
        "placed on a middle shelf may need a further ten minutes. Steam at "
        "the start of the bake keeps the crust supple so the loaf can expand, "
        "and once that expansion is complete the surface dries and begins to "
        "crisp. "
        "A rough thermometer in the middle of the loaf is more reliable than "
        "any clock, since the internal temperature continues to rise after "
        "the crust has darkened and the crumb may still be underdone. Cool it "
        "on a rack rather than a board, because steam escaping from a loaf "
        "sitting on an impermeable surface will soften the crust underneath "
        "and leave the bottom damp. Recipes are best treated as starting "
        "points to be adjusted to the flour, the kitchen and the season "
        "rather than as instructions to be followed exactly.",
    ),
}


def build_model(tag, step, device):
    """Load the trained RDA checkpoint.

    NOTE: load_checkpoint RETURNS the state dict; it does not apply it. You
    must load_state_dict yourself. compare_recall.py's loader discards the
    return value and never loads, so it evaluates randomly initialised
    weights -- this function does it correctly.
    """
    ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", tag)
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(
            f"no checkpoint dir at {ckpt_dir}\n"
            f"train one first, e.g.\n"
            f"  python -m scripts.base_train --model-tag={tag} --RDA "
            f"--depth=8 --aspect-ratio=96 --head-dim=128 --max-seq-len=1024 "
            f"--window-pattern=L --device-batch-size=4 --total-batch-size=32768 "
            f"--num-iterations=600 --eval-every=50 --eval-tokens=524288 "
            f"--core-metric-every=-1 --sample-every=-1")
    metas = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("meta_"))
    if not metas:
        raise SystemExit(f"no meta_*.json in {ckpt_dir}")
    meta = json.load(open(os.path.join(ckpt_dir, metas[-1])))
    real_step = int(meta.get("step", 600))
    if step is not None and step != real_step:
        # fall back to whatever exists
        step = real_step
    model_data, _, _ = load_checkpoint(ckpt_dir, step, device)
    if device.type in {"cpu", "mps"}:
        model_data = {k: (v.float() if v.dtype == torch.bfloat16 else v)
                      for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    config = GPTConfig(**meta["model_config"])
    base = GPT(config)
    base.init_weights()
    model = GPTWithRDA(base, debug_steps=0)
    missing, unexpected = model.load_state_dict(model_data, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(device)
    model.eval()  # eval mode: no dropout, and _log_debug stays quiet
    return model, real_step, meta


def run_text(model, tok, text, device, jump, max_rows):
    bos = tok.get_bos_token_id()
    ids = tok.encode(text, prepend=bos)
    truncated = False
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
        truncated = True

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        # Two calls, because GPTWithRDA.forward returns EITHER a bare
        # (B,T) loss tensor (when loss_reduction="none") OR the full dict
        # (when loss_reduction is anything else). We need both the per-token
        # loss and E_next_seq, so take the dict path and get the per-token
        # losses out of the FiLM-modulated logits ourselves, exactly as
        # training computes them.
        out = model(x, x)
        E_seq = out["E_next_seq"]              # (1, T-1, 8)
        logits = _logits_for_surprisal(model, x)   # (1, T, V), FiLM-applied

    # Per-token CE loss and surprisal of the ACTUAL next token, both from the
    # same logits, so they are consistent with each other.
    lp = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
    tgt = x[0, 1:]
    surprisal = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)   # (T-1,)
    per_tok_loss = surprisal                                # identical: CE = -log p

    return {
        "ids": ids, "loss": per_tok_loss, "E": E_seq[0],
        "surprisal": surprisal,
        "truncated": truncated, "n_show": min(len(ids), max_rows),
    }


def _logits_for_surprisal(model, x):
    """FiLM-conditioned logits, matching the training path.

    IMPORTANT: in eval mode forward() with targets=None still runs modulate,
    but to be explicit and independent of that branch we rebuild the
    FiLM-modulated logits here using the same recipe as GPTWithRDA.
    """
    with torch.no_grad():
        loss, hidden = model.base.forward(x, None, return_hidden=True)
        E = model.state.init_state(x.size(0), x.device, dtype=torch.float32)
        hidden_mod = model.state.modulate(hidden, E)
        lg = model.base.lm_head(hidden_mod)
        lg = lg[..., :model.base.config.vocab_size].float()
        lg = 15 * torch.tanh(lg / 15)
    return lg


def fmt(tok, tid):
    try:
        s = tok.decode_single_token_bytes(tid)
        return s.decode("utf-8", errors="replace")
    except Exception:
        return "?"


def _print_table(tok, res, jump, max_rows, use_color, ref):
    ids, E, sur = res["ids"], res["E"], res["surprisal"]
    n = res["n_show"]
    head = (f"{'#':>4}  {'token':<14} "
            + " ".join(f"{d[:5]:>5}" for d in DRIVE_DIMS)
            + f" {'surprisal':>10}  ")
    print(head)
    print("-" * len(head))

    prev = None
    for i in range(n):
        tid = ids[i]
        ttext = fmt(tok, tid).replace("\n", "\\n").replace("\r", "\\r")
        if ttext == "":
            ttext = " "
        # state after this token; the last token has no E entry
        if i < E.shape[0]:
            e = E[i].tolist()
        else:
            e = [float("nan")] * N_DRIVES
        s = sur[i].item() if i < sur.shape[0] else float("nan")

        mark = " "
        if prev is not None:
            # NOTE: this is a POSITION marker and is only kept as a light
            # visual aid. It is not a content signal -- E_t relaxes along
            # position, so a jump here mostly reflects the geometric ramp.
            # The meaningful comparison is the settled state vs reference,
            # printed after the table.
            dmax = max(abs(e[k] - prev[k]) for k in range(N_DRIVES))
            if dmax > jump:
                mark = "*"
        if use_color and mark == "*":
            mark = "\033[1;33m*\033[0m"

        print(f"{i:>4}  {repr(ttext)[:14]:<14} "
              + " ".join(f"{v:>+5.2f}" for v in e)
              + f" {s:>10.3f}  {mark}")
        if not any(v != v for v in e):
            prev = e
    if n < len(ids):
        print(f"      ... {len(ids) - n} more tokens not shown (--max-rows)")

    # ---- summary: settled state vs the neutral reference -------------------
    # Per-token jump detection is deliberately NOT used. update() runs T-1
    # times inside one forward pass, so E_t relaxes geometrically along
    # POSITION (E_t ~= 1 - 0.95^t): the trajectory is a near-monotone ramp
    # with ~0.05 per-token movement that barely depends on content, so a
    # fixed jump threshold either never fires or fires on the ramp's early
    # positions regardless of what the text says. Comparing the SETTLED final
    # state against a fixed neutral reference is the meaningful comparison.
    print()
    print(f"  Settled final state vs neutral reference")
    if ref is None:
        print("  (no reference_state.json -- run demo/make_reference.py)")
        return
    print(f"  (reference = {ref['n_tokens']}-token neutral expository paragraph)")
    print()
    print(f"    {'dim':<20} {'this text':>10} {'reference':>10} {'delta':>9}")
    print("    " + "-" * 51)
    final = res["E"][-1].tolist()
    deltas = []
    for k, d in enumerate(DRIVE_DIMS):
        dv = final[k] - ref["state"][k]
        deltas.append((abs(dv), d, dv))
        print(f"    {d:<20} {final[k]:>+10.4f} {ref['state'][k]:>+10.4f} {dv:>+9.4f}")
    deltas.sort(reverse=True)
    top = deltas[:3]
    print()
    print("    most divergent dimensions: "
          + ", ".join(f"{d} ({dv:+.3f})" for _, d, dv in top))
    # L2 distance as a single scalar summary.
    l2 = sum(dv * dv for _, _, dv in deltas) ** 0.5
    print(f"    L2 distance from reference: {l2:.4f}")
    if len(ids) < 100:
        print(f"    note: only {len(ids)} tokens -- the state is still inside its")
        print("          ~20-100 token transient and has not settled. Use --text")
        print("          with a longer passage, or --example N.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--example", type=int, choices=sorted(EXAMPLES), default=None)
    ap.add_argument("--tag", type=str, default="model-b-primal")
    ap.add_argument("--step", type=int, default=600)
    ap.add_argument("--jump", type=float, default=0.15,
                    help="mark a token if any dim moves more than this")
    ap.add_argument("--max-rows", type=int, default=200)
    ap.add_argument("--gpu", action="store_true", help="use CUDA instead of CPU")
    args = ap.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")
    use_color = sys.stdout.isatty() and os.name != "nt"

    tok = get_tokenizer()
    model, step, meta = build_model(args.tag, args.step, device)
    ref = load_reference()
    print(f"loaded {args.tag} step {step} on {device}  "
          f"val_bpb={meta.get('val_bpb', float('nan')):.6f}  "
          f"n_drives={N_DRIVES}")
    if ref is None:
        print("NOTE: no reference_state.json found -- run "
              "`python demo/make_reference.py` to enable the settled-state "
              "comparison.")
    if device.type == "cpu":
        print("(CPU mode: slower, but does not reserve the training GPU)")
    print()

    def show(text, title=None):
        if title:
            print(f"--- {title} ---")
        res = run_text(model, tok, text, device, args.jump, args.max_rows)
        if res["truncated"]:
            print(f"[truncated to {MAX_LEN} tokens]\n")
        _print_table(tok, res, args.jump, args.max_rows, use_color, ref)
        print()

    if args.text is not None:
        show(args.text)
        return

    if args.example is not None:
        desc, text = EXAMPLES[args.example]
        print(f"example {args.example}: {desc}")
        print(f"input: {text[:120]}{'...' if len(text) > 120 else ''}\n")
        show(text)
        return

    # interactive loop
    print("interactive mode -- type text, or 'exit' to quit.\n")
    while True:
        try:
            text = input("text> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = text.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            break
        show(text)


if __name__ == "__main__":
    main()
