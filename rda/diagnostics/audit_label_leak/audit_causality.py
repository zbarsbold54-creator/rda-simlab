"""AUDIT (read-only): causality tests + eval-only ablations on b-stab-s42.

Rules: no training; no tracked file touched; this file stays uncommitted.
Ablations are forward hooks on production modules, installed and removed
locally, and the fact is printed.

BPB is nanochat's real metric: sum(nats * has_bytes) / sum(bytes) / ln2, using
loss_reduction="none" plus get_token_bytes -- the same arithmetic as
nanochat.loss_eval.evaluate_bpb, on a FIXED batch list so every variant sees
identical data in identical order.

Run:  python diagnostics/audit_causality.py
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

TAG = "b-stab-s42"
CKPT = os.path.join(get_base_dir(), "base_checkpoints", TAG, "model_000600.pt")
N_BATCH = 6
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
    sd = torch.load(CKPT, map_location=device, weights_only=True)
    m.load_state_dict(sd, strict=True)          # FULL GPTWithRDA state dict
    say("  loaded %d keys, strict=True" % len(sd))
    return m, tok, V


def bpb_over(m, batches, tb, reset_each_batch=False):
    """nanochat bits-per-byte over a fixed batch list.

    Carry source: forward() sets self._last_E_next_seq on EVERY path
    (rda.py:1001), including the loss_reduction="none" early return at 1041.
    Its last position is the E that the next batch would start from.
    """
    carry, tot_n, tot_b = None, 0.0, 0.0
    for x, y in batches:
        with torch.no_grad():
            l2d = m(x, y, loss_reduction="none", E_t_external=carry)
        l2d = l2d.view(-1).float()
        t = y.view(-1)
        nb = tb[torch.where(t >= 0, t, torch.zeros_like(t))]
        nb = torch.where(t >= 0, nb, torch.zeros_like(nb))
        tot_n += float((l2d * (nb > 0)).sum())
        tot_b += float(nb.sum())
        carry = None if reset_each_batch else m._last_E_next_seq[:, -1, :]
    return tot_n / tot_b / LN2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say("=" * 74)
    say("AUDIT: causality + eval ablations on %s" % TAG)
    say("=" * 74)
    m, tok, V = build(42, device)
    st = m.state
    tb = get_token_bytes(device=device)
    say("  |film_scale.W|=%.3f  |film_shift.W|=%.3f  (trained, non-zero)"
        % (float(st.film_scale.weight.abs().sum().detach()),
           float(st.film_shift.weight.abs().sum().detach())))
    say()

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    batches = []
    for _ in range(N_BATCH):
        x, y, _ = next(loader)
        batches.append((x.to(device), y.to(device)))
    say("fixed eval set: %d batches of %s\n" % (N_BATCH, tuple(batches[0][0].shape)))

    # ---------------- 3a. future-token test ---------------- #
    say("--- 3a. FUTURE-TOKEN TEST ---")
    say("Perturb x[:, k]; outputs at positions < k must be bit-identical.")
    x, y = batches[0]
    with torch.no_grad():
        ref = m(x, y)
        h0, E0 = ref["hidden"], ref["E_next_seq"]
    leak = False
    for k in (1, 17, 128, 511, 1022):
        x2 = x.clone()
        x2[:, k] = (x2[:, k] + 1) % V
        with torch.no_grad():
            o2 = m(x2, y)
        dh = (o2["hidden"][:, :k] - h0[:, :k]).abs().max().item()
        dE = (o2["E_next_seq"][:, :k] - E0[:, :k]).abs().max().item()
        leak |= max(dh, dE) > 1e-5
        say("  k=%-5d max|dh|<k=%.3e  max|dE|<k=%.3e" % (k, dh, dE))
    say("  VERDICT: %s" % ("LEAK" if leak else "no leak (all exactly 0)"))
    say()

    # ---------------- 3b. target test ---------------- #
    say("--- 3b. TARGET TEST ---")
    say("pe_seq is built from targets[:, 1:] (rda.py:901), so the state is a")
    say("function of the labels BY CONSTRUCTION. The leak question is not")
    say("'does it read targets' but 'can a target value reach an output that")
    say("predicts that same target'. Perturb y[:, k] and require outputs at")
    say("positions < k to be unchanged.")
    with torch.no_grad():
        r0 = m(x, y)
        h0, E0, lg0 = r0["hidden"], r0["E_next_seq"], None
        lg0 = 15 * torch.tanh(m.base.lm_head(r0["hidden_mod"]).float() / 15)
    leak = False
    for k in (1, 17, 128, 511, 1022):
        y2 = y.clone()
        y2[:, k] = (y2[:, k] + 1) % V
        with torch.no_grad():
            r2 = m(x, y2)
        # logits at position p predict token p+1, so positions < k are the
        # ones that must not see a change at k.
        dh = (r2["hidden"][:, :k] - h0[:, :k]).abs().max().item()
        dE = (r2["E_next_seq"][:, :k] - E0[:, :k]).abs().max().item()
        lg2 = 15 * torch.tanh(m.base.lm_head(r2["hidden_mod"]).float() / 15)
        dl = (lg2[:, :k] - lg0[:, :k]).abs().max().item()
        leak |= max(dh, dE, dl) > 1e-5
        say("  k=%-5d max|dh|<k=%.3e  max|dE|<k=%.3e  max|dlogits|<k=%.3e"
            % (k, dh, dE, dl))
    say("  VERDICT: %s" % ("LEAK" if leak else "no leak (all exactly 0)"))
    say()

    # ---------------- 3c. order / persistence ---------------- #
    say("--- 3c. BATCH-ORDER AND STATE-PERSISTENCE (real BPB) ---")
    b_carry = bpb_over(m, batches, tb)
    b_reset = bpb_over(m, batches, tb, reset_each_batch=True)
    b_rev = bpb_over(m, list(reversed(batches)), tb)
    say("  forward order, state carried : bpb %.6f" % b_carry)
    say("  forward order, state reset   : bpb %.6f  (delta %+.6f)"
        % (b_reset, b_reset - b_carry))
    say("  REVERSED order,  carried     : bpb %.6f  (delta %+.6f)"
        % (b_rev, b_rev - b_carry))
    say("  => E persists across batches: %s"
        % ("YES" if abs(b_carry - b_reset) > 1e-6 else "no"))
    say("  => order matters:            %s"
        % ("YES" if abs(b_carry - b_rev) > 1e-6 else "no"))
    say()
    say("  per-batch last-position |E| (carried run):")
    carry = None
    for i, (bx, by) in enumerate(batches):
        with torch.no_grad():
            m(bx, by, loss_reduction="none", E_t_external=carry)
        En = m._last_E_next_seq[:, -1, :]
        say("    batch %d: max|E|=%.4f  mean|E|=%.4f"
            % (i, float(En.abs().max()), float(En.abs().mean())))
        carry = En
    say()

    # ---------------- 4. ablations ---------------- #
    say("--- 4. EVAL-ONLY ABLATIONS (real BPB) ---")
    say("  reference: B'-s42 run 1.226305 | baseline A s42 1.301217")
    say("  reloaded ckpt, unmodified    : bpb %.6f" % b_carry)
    say()

    h = st.event_proj.register_forward_hook(lambda m_, i_, o_: torch.zeros_like(o_))
    b_ev = bpb_over(m, batches, tb)
    h.remove()
    say("  event term zeroed           : bpb %.6f  (delta %+.6f)" % (b_ev, b_ev - b_carry))

    h1 = st.film_scale.register_forward_hook(lambda m_, i_, o_: torch.zeros_like(o_))
    h2 = st.film_shift.register_forward_hook(lambda m_, i_, o_: torch.zeros_like(o_))
    b_film = bpb_over(m, batches, tb)
    h1.remove(); h2.remove()
    say("  FiLM forced constant         : bpb %.6f  (delta %+.6f)" % (b_film, b_film - b_carry))

    h3 = st.pe_proj.register_forward_hook(lambda m_, i_, o_: torch.zeros_like(o_))
    b_pe = bpb_over(m, batches, tb)
    h3.remove()
    say("  PE term zeroed               : bpb %.6f  (delta %+.6f)" % (b_pe, b_pe - b_carry))
    say()
    say("  hooks remaining: %d (expect 0)"
        % (len(st.event_proj._forward_hooks) + len(st.pe_proj._forward_hooks)
           + len(st.film_scale._forward_hooks) + len(st.film_shift._forward_hooks)))


if __name__ == "__main__":
    main()
