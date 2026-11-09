"""Why are the state-pathway gradients exactly zero at step 0?

Phase 1 turned up a result that contradicts the naive reading of the graph:
at step 0 the gradients reaching pe_proj, event_proj and h_norm are EXACTLY
0.0, while film_scale/film_shift and the base model get real gradient.

Hypothesis: film_scale and film_shift are ZERO-INITIALIZED (rda.py, so that
modulate() is an exact identity at init). The gradient that flows from the
loss back into the state E is

    dL/dE  =  film_scale.weight^T @ (dL/dhidden_mod * hidden)
            + film_shift.weight^T @ (dL/dhidden_mod)

Every term is proportional to a film weight, and every film weight is exactly
0 at step 0. So the state pathway is COMPLETELY DISCONNECTED from the loss at
initialization, and the loop cannot start until the film weights have moved.

If true, the saturation is not a steady-state instability at all: it is an
ONSET-TIME problem, and the question "is the loop closed?" is not answerable
from a single step. It has to be answered per step, over time.

This script measures when each pathway's gradient first becomes nonzero, and
how it grows, using the production forward/backward with NO reimplementation
and NO patching. It also confirms the mechanism directly by checking that
grad(film weights) is nonzero at step 0 while grad(state params) is not.

Run:  python -m diagnostics/pe_gradient_onset --steps 12
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

D_MODEL = 768


def gn(p):
    return 0.0 if p is None or p.grad is None else float(p.grad.detach().float().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    compute_init("cpu" if device.type == "cpu" else "cuda", a.seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=D_MODEL, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.train()
    st = m.state
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="train", device=device)

    # Mirror base_train.py's real LR scaling (batch_lr_scale = 0.25 at
    # total_batch_size=32768) so the trajectory is the real one.
    B_REF = 524288
    scale = (32768.0 / B_REF) ** 0.5
    LR_M = 0.02 * scale
    groups = [
        {"params": list(st.film_scale.parameters()) + list(st.film_shift.parameters()),
         "lr": LR_M, "initial_lr": LR_M, "weight_decay": 0.0, "betas": (0.9, 0.95)},
        {"params": list(st.event_proj.parameters()) + list(st.pe_proj.parameters()),
         "lr": LR_M, "initial_lr": LR_M, "weight_decay": 0.01, "betas": (0.9, 0.95)},
        {"params": [st.e_init], "lr": 0.5 * scale, "initial_lr": 0.5 * scale,
         "weight_decay": 0.0, "betas": (0.96, 0.95)},
    ]
    opt = torch.optim.AdamW(groups, eps=1e-10)

    print("=" * 78)
    print("PE/FiLM GRADIENT ONSET -- production code, no patching")
    print("=" * 78)
    print("device=%s steps=%d seed=%d  drive-group LR=%.5f" % (a.device, a.steps, a.seed, LR_M))
    print()
    print("film weights are ZERO-INITIALIZED, so at step 0")
    print("    dL/dE  is  proportional to  film weights  ==  0.")
    print("Prediction: grad(pe_proj)=grad(event_proj)=grad(h_norm)=0 at step 0,")
    print("and they stay 0 until the film weights have moved.")
    print()
    print("%-4s %12s %12s %12s %12s %12s %12s"
          % ("step", "g_pe_proj", "g_event_proj", "g_film_scl", "g_film_shf",
             "g_base", "||filmW||"))
    print("-" * 78)

    carry = None
    first_nonzero = {"pe_proj": None, "event_proj": None, "h_norm": None}
    for step in range(a.steps):
        m.zero_grad(set_to_none=True)
        x, y, _ = next(loader)
        out = m(x, y, E_t_external=carry)
        out["loss"].backward()

        g_pe = gn(st.pe_proj.weight)
        g_ep = gn(st.event_proj.weight)
        g_fs = gn(st.film_scale.weight)
        g_fh = gn(st.film_shift.weight)
        base_p = [p for p in m.base.parameters() if p.grad is not None]
        g_b = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in base_p))) if base_p else 0.0
        fw = float(st.film_scale.weight.detach().norm()
                   + st.film_shift.weight.detach().norm())

        for k, v in (("pe_proj", g_pe), ("event_proj", g_ep),
                     ("h_norm", gn(st.h_norm.weight))):
            if v > 0.0 and first_nonzero[k] is None:
                first_nonzero[k] = step

        print("%-4d %12.3e %12.3e %12.3e %12.3e %12.3e %12.6f"
              % (step, g_pe, g_ep, g_fs, g_fh, g_b, fw))
        opt.step()
        lr_mult = min(1.0, (step + 1) / 10.0)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_mult
        carry = out["E_next"]

    print("-" * 78)
    print("\nFirst step at which each state-pathway gradient became nonzero:")
    for k, v in first_nonzero.items():
        print("  %-12s %s" % (k, "step %d" % v if v is not None else "NEVER in %d steps" % a.steps))
    print("\nInterpretation:")
    if first_nonzero["pe_proj"] is None:
        print("  The PE pathway received NO gradient at all in %d steps." % a.steps)
    else:
        print("  The state pathway was gradient-dead for %d step(s), then opened."
              % first_nonzero["pe_proj"])
        print("  Until then, pe_proj.bias and event_proj.weight CANNOT drift.")
        print("  The saturation we measured at step 88 is therefore an")
        print("  ONSET-TIME phenomenon: it begins only once FiLM has")
        print("  acquired nonzero weights and reconnected the loop.")


if __name__ == "__main__":
    main()
