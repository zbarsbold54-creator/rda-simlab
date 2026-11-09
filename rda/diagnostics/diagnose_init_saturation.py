"""Diagnose the B' saturation at init. READ-ONLY: changes no code, uses no GPU.

Question: is the `threat` drive crossing |E| > 0.98 within ~10 steps caused by
the INITIALIZATION of the RDA modules -- specifically `pe_proj`, whose
weight/bias are drawn at PyTorch's DEFAULT init (kaiming_uniform on a
Linear(1, 8), i.e. bound ~1/sqrt(1) = 1.0) -- or by something else?

For each of seeds 42/43/44 this:
  1. builds the model exactly as base_train.py does,
  2. prints pe_proj.weight and pe_proj.bias per drive,
  3. runs ONE forward pass at init on a real batch,
  4. reports per-drive max |E| across positions.

The per-drive drive magnitude that actually matters is
    tanh(0.1 * event_proj(LN(h)) + pe_proj(pe / 10))
so the bias of pe_proj is a CONSTANT offset on every position -- a large
bias saturates the tanh immediately, with no data dependence at all.

CPU, batch 1. Does not modify any file.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

DEVICE = torch.device("cpu")
SEEDS = (42, 43, 44)


def build(seed):
    """Mirror base_train.py: compute_init(seed) then GPT(cfg).init_weights()
    then wrap in GPTWithRDA. The wrap happens AFTER the base model is
    initialized, so the RDA modules draw from whatever RNG state remains."""
    from nanochat.common import compute_init
    compute_init("cpu", seed)
    from nanochat.tokenizer import get_tokenizer as _gt
    tok = _gt()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=768, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0)
    return m, tok, V


def main():
    print("device=cpu  batch=1  (read-only diagnosis)\n")
    tok = get_tokenizer()

    for seed in SEEDS:
        print("=" * 70)
        print("SEED %d" % seed)
        # Reproduce the exact RNG consumption order of base_train.py by
        # building the model, then draw a batch from the same loader.
        from nanochat.common import compute_init
        compute_init("cpu", seed)
        V = tok.get_vocab_size()
        cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                        n_kv_head=6, n_embd=768, window_pattern="L")
        base = GPT(cfg)
        base.init_weights()
        m = GPTWithRDA(base, debug_steps=0)
        m = m.to(DEVICE)
        m.eval()

        print("\n  pe_proj  (Linear(1, %d)) -- this is the constant offset added"
              % len(DRIVE_DIMS))
        print("  to every position, so a large |bias| saturates tanh with no")
        print("  data dependence at all:")
        W = m.state.pe_proj.weight.detach().flatten()
        B = m.state.pe_proj.bias.detach()
        for i, name in enumerate(DRIVE_DIMS):
            print("    %-18s w=%+.4f  b=%+.4f  |b|=%.4f"
                  % (name, W[i].item(), B[i].item(), abs(B[i].item())))
        print("    max|w| = %.4f   max|b| = %.4f"
              % (W.abs().max().item(), B.abs().max().item()))

        # One real forward pass at init, on a real batch.
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok, 1, 1024, split="train", device=DEVICE)
        x, y, _ = next(loader)
        with torch.no_grad():
            out = m(x, y, E_t_external=None)
        E = out["E_next_seq"][0]                 # (T-1, n_drives)
        per_dim = E.abs().amax(dim=0)            # (n,)
        print("\n  step-0 loss: %.6f   (A seed 42 = 10.397894)"
              % out["loss"].item())
        print("\n  per-drive max |E| at init, across all %d positions:"
              % E.size(0))
        for i, name in enumerate(DRIVE_DIMS):
            v = per_dim[i].item()
            flag = "  <-- NEAR RAIL (>0.5)" if v > 0.5 else ""
            print("    %-18s max|E| = %.4f%s" % (name, v, flag))

        # Why: decompose the drive into its two terms.
        h = out["hidden"][:, :-1, :].to(m.state.event_proj.weight.dtype)
        ev = m.state.event_drive_scale * m.state.event_proj(m.state.h_norm(h))
        with torch.no_grad():
            logits = m.base.lm_head(out["hidden"][:, :-1, :])[
                ..., :m.base.config.vocab_size].float()
            logits = 15 * torch.tanh(logits / 15)
            pe_seq = m.state.prediction_error_from_logits(logits, y[:, 1:])
        # pe_proj is Linear(1, n_drives), so pe needs a trailing singleton.
        pv = m.state.pe_proj(
            (pe_seq / m.state.pe_reference_scale).unsqueeze(-1)
            .to(m.state.pe_proj.weight.dtype)
        )[0]                                  # (T-1, n_drives)
        print("\n  decomposition of the tanh ARGUMENT (max over positions):")
        print("    %-18s %10s %10s" % ("drive", "event", "pe_term"))
        for i, name in enumerate(DRIVE_DIMS):
            print("    %-18s %10.4f %10.4f"
                  % (name, ev[0, :, i].abs().amax().item(), pv[:, i].abs().max().item()))
        print()


if __name__ == "__main__":
    main()
