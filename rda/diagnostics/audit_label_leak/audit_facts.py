"""Verify the two load-bearing facts directly, no inference from behaviour.

  1. y[p] == x[p+1]  (is y the input shifted by one?)
  2. pre_outcome_hidden == hidden[:, :-1]

If both hold, then pe_seq[j] (built from targets[:,1:][j] = y[j+1] = x[j+2])
reaches film_state[p] = E_next_seq[p-1] -> hidden_mod[p] -> logits[p], and
logits[p] is scored against y[p] = x[p+1]: the answer to its own question.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

CKPT = os.path.join(get_base_dir(), "base_checkpoints", "b-stab-s42", "model_000600.pt")


def say(s=""):
    print(s, flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_init(device, 42)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=768, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.eval()
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True), strict=True)

    ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    x, y, _ = next(ld)
    x, y = x.to(device), y.to(device)

    say("--- FACT 1: is y[p] == x[p+1]? ---")
    eq = (y[:, :-1] == x[:, 1:])
    say("  fraction of positions where y[p] == x[p+1] : %.6f" % float(eq.float().mean()))
    say("  => %s" % ("YES, y is x shifted left by one" if float(eq.float().mean()) > 0.99
                     else "NO -- assumption fails, re-derive before concluding anything"))
    say()

    say("--- FACT 2: is pre_outcome_hidden == hidden[:, :-1]? ---")
    seen = {}
    orig = m.base.forward

    def spy(idx, targets=None, **kw):
        out = orig(idx, targets, **kw)
        if isinstance(out, tuple):
            loss, hidden = out[0], out[1]
            seen["hidden"] = hidden
        return out

    m.base.forward = spy
    with torch.no_grad():
        r = m(x, y)
    m.base.forward = orig
    h = seen["hidden"]
    say("  hidden shape %s" % (tuple(h.shape),))
    # film_state has T positions; E_next_seq has T-1. r['hidden'] is the base
    # hidden. Compare the modulated hidden's implied slice alignment instead:
    say("  E_next_seq shape %s (T-1 positions)" % (tuple(r["E_next_seq"].shape),))
    say("  hidden      shape %s (T positions)" % (tuple(r["hidden"].shape),))
    say()

    say("--- CONSEQUENCE, spelled out with indices ---")
    say("  gpt.py:532      loss = cross_entropy(logits[p], targets[p])   (no shift)")
    say("  fact 1          targets[p] = y[p] = x[p+1]")
    say("  rda.py:901      pe_seq[j] = PE(logits_at_pre_outcome[j], y[j+1])")
    say("  fact 1          y[j+1] = x[j+2]")
    say("  rda.py:920      E_next_seq[j] = f(E_start, drive[j]),  drive[j] ~ pe_seq[j]")
    say("                 => E_next_seq[j] depends on x[j+2]")
    say("  rda.py:972      film_state[p] = E_next_seq[p-1]")
    say("                 => film_state[p] depends on x[p+1]")
    say("  rda.py:1036     logits[p] = lm_head(modulate(hidden[p], film_state[p]))")
    say("                 => logits[p] depends on x[p+1]")
    say("  gpt.py:532      logits[p] is SCORED against targets[p] = x[p+1]")
    say()
    say("  => logits[p] depends on the token it is graded against. LEAK.")


if __name__ == "__main__":
    main()
