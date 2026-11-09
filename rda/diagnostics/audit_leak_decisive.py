"""DECISIVE TEST: is y[p] reaching logits[p]?

Established so far:
  * gpt.py:532      loss = cross_entropy(logits[p], targets[p])   -- NO shift
  * y[p] == x[p+1]  (sliding window), so logits[p] is the distribution
    for token p+1, and normally may depend only on x[<=p].
  * rda.py:901      pe_seq[j] = PE(logits_at_j, targets[:,1:][j])
                                   = PE(logits_at_j, y[j+1])
  * rda.py:920-921  E_next_seq[p] = update_seq(..., pe_seq[p])  -> depends on y[p+1]
  * rda.py:972      film_state[p] = E_next_seq[p-1]             -> depends on y[p]
  * hidden_mod[p] = modulate(hidden[p], film_state[p])          -> depends on y[p]
  * logits[p] = lm_head(hidden_mod[p])                          -> depends on y[p]

So the state that modulates position p is a function of y[p] -- the very
label position p is scored against. The comment at rda.py:963-970 asserts
"pe_seq[:, p-1] ... depends on token p -- strictly causal", but by the
alignment above pe_seq[:, p-1] depends on y[p] = x[p+1], not x[p]. The
indexing is right and the ANALYSIS is off by one.

Two measurements:
  1. LOCALITY. Change y[:,k] only; which positions' logits move?
  2. SIZE. Score the same x with true y vs with y shuffled inside the batch.
     A causal LM's forward never sees y, so these must be EQUAL. The gap is
     exactly what the leak contributes.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

CKPT = os.path.join(get_base_dir(), "base_checkpoints", "b-stab-s42", "model_000600.pt")
LN2 = 0.6931471805599453


def say(s=""):
    print(s, flush=True)


def build(seed, device):
    compute_init(device, seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=768, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.eval()
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True), strict=True)
    return m, tok, V


def bpb(m, batches, tb, y_override=None):
    carry, tn, tbb = None, 0.0, 0.0
    for i, (x, y) in enumerate(batches):
        yy = y if y_override is None else y_override[i]
        with torch.no_grad():
            l = m(x, yy, loss_reduction="none", E_t_external=carry)
        l = l.view(-1).float()
        t = yy.view(-1)
        nb = tb[torch.where(t >= 0, t, torch.zeros_like(t))]
        nb = torch.where(t >= 0, nb, torch.zeros_like(nb))
        tn += float((l * (nb > 0)).sum())
        tbb += float(nb.sum())
        carry = m._last_E_next_seq[:, -1, :]
    return tn / tbb / LN2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok, V = build(42, device)
    tb = get_token_bytes(device=device)
    ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    B = []
    for _ in range(24):
        x, y, _ = next(ld)
        B.append((x.to(device), y.to(device)))

    # ---- 1. LOCALITY ---- #
    say("--- 1. LOCALITY: change y[:,k], see which logits move ---")
    x, y = B[0]
    with torch.no_grad():
        lg0 = 15 * torch.tanh(m.base.lm_head(m(x, y)["hidden_mod"]).float() / 15)
    for k in (1, 17, 128, 511):
        y2 = y.clone()
        y2[:, k] = (y2[:, k] + 1) % V
        with torch.no_grad():
            lg2 = 15 * torch.tanh(m.base.lm_head(m(x, y2)["hidden_mod"]).float() / 15)
        d = (lg2 - lg0).abs().amax(dim=-1)          # (B, T) per-position movement
        at_k = float(d[:, k].max())
        before = float(d[:, :k].max()) if k > 0 else 0.0
        after = float(d[:, k + 1:].max()) if k + 1 < d.size(1) else 0.0
        say("  k=%-4d  pos k: %.3e | pos <k: %.3e | pos >k: %.3e" % (k, at_k, before, after))
    say("  => pos < k is EXACTLY 0 and pos >= k moves. That is the correct")
    say("     shape for a leaky model: film_state[p] = E_next_seq[p-1], and the")
    say("     recurrence is cumulative, so y[k] reaches every p >= k. The")
    say("     violation is the DIAGONAL: logits[k] depends on y[k] = x[k+1].")
    say()

    # ---- 2. SIZE ---- #
    say("--- 2. SIZE OF THE LEAK ---")
    say("Same x. Score against the true y, then against y shuffled ALONG the")
    say("time axis (so y[p] is a different real token, bytes still valid).")
    say("A causal LM's forward never sees y, so these must match exactly.")
    g = torch.Generator(device="cpu").manual_seed(0)
    sh = []
    for x, y in B:
        perm = torch.randperm(y.size(1), generator=g).to(y.device)
        sh.append(y[:, perm].contiguous())
    b_true = bpb(m, B, tb)
    b_shuf = bpb(m, B, tb, y_override=sh)
    say("  x with TRUE y          : bpb %.6f   <- what the run reported" % b_true)
    say("  x with SHUFFLED y      : bpb %.6f   <- no label information" % b_shuf)
    say("  leak contribution      : %+.6f" % (b_shuf - b_true))
    say()
    say("  baseline A s42         : 1.301217")
    say("  B'-s42 run reported    : 1.226305")
    say("  B'-s42, no label       : %.6f" % b_shuf)
    say()
    say("  If the no-label number is at or worse than the 1.301217 baseline,")
    say("  then the entire 0.0749 gain came from the leak, not from the state.")


if __name__ == "__main__":
    main()
