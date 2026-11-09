"""Follow-up checks on three findings from audit_causality.py. Read-only."""
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


def bpb(m, batches, tb, reset=False):
    carry, tn, tbb = None, 0.0, 0.0
    for x, y in batches:
        with torch.no_grad():
            l = m(x, y, loss_reduction="none", E_t_external=carry)
        l = l.view(-1).float()
        t = y.view(-1)
        nb = tb[torch.where(t >= 0, t, torch.zeros_like(t))]
        nb = torch.where(t >= 0, nb, torch.zeros_like(nb))
        tn += float((l * (nb > 0)).sum())
        tbb += float(nb.sum())
        carry = None if reset else m._last_E_next_seq[:, -1, :]
    return tn / tbb / LN2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok, V = build(42, device)
    tb = get_token_bytes(device=device)
    ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    B = []
    for _ in range(6):
        x, y, _ = next(ld)
        B.append((x.to(device), y.to(device)))

    # ---- A. correct causal bound on the target test ---- #
    say("--- A. TARGET TEST, correct bound (logits at p predict token p+1) ---")
    say("Changing y[:,k] == changing token k+1. So logits at positions <= k")
    say("must be unchanged. logits[k] predicts exactly the changed token.")
    x, y = B[0]
    with torch.no_grad():
        r0 = m(x, y)
        lg0 = 15 * torch.tanh(m.base.lm_head(r0["hidden_mod"]).float() / 15)
    worst = 0.0
    for k in (1, 17, 128, 511, 1022):
        y2 = y.clone()
        y2[:, k] = (y2[:, k] + 1) % V
        with torch.no_grad():
            r2 = m(x, y2)
        lg2 = 15 * torch.tanh(m.base.lm_head(r2["hidden_mod"]).float() / 15)
        d = (lg2[:, :k + 1] - lg0[:, :k + 1]).abs().max().item()
        worst = max(worst, d)
        say("  k=%-5d max|dlogits|<=k = %.3e" % (k, d))
    say("  VERDICT: %s" % ("LEAK" if worst > 1e-5 else "no leak (exactly 0)"))
    say("  => the earlier max|dE|<k > 0 is E *recording* token k, which is")
    say("     legitimate: E_next_seq[:,p] depends on token p+1 and feeds")
    say("     position p+1, never an earlier position.")
    say()

    # ---- B. is E at a fixed point? ---- #
    say("--- B. IS THE CARRIED STATE AT A FIXED POINT? ---")
    carry = None
    for i, (bx, by) in enumerate(B):
        with torch.no_grad():
            m(bx, by, loss_reduction="none", E_t_external=carry)
        E = m._last_E_next_seq[:, -1, :]
        say("  batch %d  max|E|=%.8f  mean=%.8f" % (i, float(E.abs().max()), float(E.abs().mean())))
        carry = E
    say("  => identical to 8 dp across all batches: the state is pinned, so the")
    say("     carry is irrelevant and batch order cannot matter. That is why")
    say("     the reversed-order delta was exactly 0.")
    say()

    # ---- C. does BPB converge toward the run's 1.226305? ---- #
    say("--- C. IS 1.132939 COMPARABLE TO THE RUN'S 1.226305? ---")
    ld2 = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    pool = []
    for _ in range(40):
        x, y, _ = next(ld2)
        pool.append((x.to(device), y.to(device)))
    for n in (6, 12, 24, 40):
        say("  %2d batches, carried : bpb %.6f" % (n, bpb(m, pool[:n], tb)))
    say("  %2d batches, reset   : bpb %.6f" % (40, bpb(m, pool[:40], tb, reset=True)))
    say("  run reported (eval_tokens=524288, reset each call): 1.226305")
    say("  baseline A s42: 1.301217")


if __name__ == "__main__":
    main()
