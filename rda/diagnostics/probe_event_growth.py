"""Phase 2 instrumentation: measure the event-term growth mechanism on a
REAL 30-step training trajectory.

WHY A SEPARATE SCRIPT. rda.py must not be modified yet (Phase 5 is the only
place a change is allowed), so this reproduces the exact update() arithmetic
in fp32 on the side, reading the real hidden states and real PE out of the
real model. Every number below is therefore measured, not inferred.

The question this answers: LayerNorm normalizes ||h||, but does the
PROJECTION output stay bounded? The Cauchy-Schwarz ceiling is
||w_row|| * ||h_norm||, and ||h_norm|| is pinned at sqrt(d) only if LayerNorm
has weight=1,bias=0 -- both of which are LEARNABLE and can drift. If
h_norm.weight grows during training, the ceiling grows with it, and
LayerNorm stops being the bound it was introduced to be.

Signals recorded per step (Phase 2 list):
   1/2  LN output stats: ||h_norm||, mean/std/min/max
   3/4  ||event_proj(h_norm)|| and per-dim event contribution
   5/6  ||pe_proj(pe)|| and per-dim PE contribution
   7    pre-tanh activation magnitude
   8    post-tanh / state E
   9/10 FiLM scale and shift magnitude
  11    grad norm of event_proj
  12    grad norm reaching the state update
  13    grad norm of base-model params
  14    event_proj parameter norm over time
Plus the crucial one: h_norm.weight and h_norm.bias norms, which decide
whether the LayerNorm ceiling itself is stationary.

Run (GPU):
  python diagnostics/probe_event_growth.py --steps 30 --seed 42
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

from nanochat.common import compute_init, get_base_dir, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

D_MODEL = 768
N_DRIVES = len(DRIVE_DIMS)


def build(seed, device, freeze_base=False, rda_only=False):
    compute_init("cpu" if device.type == "cpu" else "cuda", seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=D_MODEL, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0)
    m = m.to(device)
    m.train()
    return m, tok, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--freeze-base", action="store_true",
                    help="Control 2: disable base-model parameter updates")
    ap.add_argument("--out", type=str, default=None)
    # Phase 3 ablations, env-gated so the default path is untouched.
    ap.add_argument("--norm-event", action="store_true",
                    help="Control 3: RMS-normalize event_proj output")
    ap.add_argument("--bound-event", type=float, default=0.0,
                    help="Control 4: bound event contribution via tanh(k*x)/k")
    ap.add_argument("--decay", type=float, default=None,
                    help="Control 5: override decay")
    a = ap.parse_args()
    device = torch.device(a.device)

    m, tok, V = build(a.seed, device)
    st = m.state
    if a.decay is not None:
        st.decay = a.decay
        print("decay overridden to %.3f" % a.decay)

    # Ablations must change what the MODEL computes, not just what the probe
    # measures, otherwise the trajectory is unchanged. Monkey-patch drive_seq
    # so the real forward() uses the ablated event term. The drive's
    # recurrence, PE term, FiLM path and shapes are all untouched.
    #
    # CRITICAL: install this ONLY when an ablation is actually requested.
    # It is a full reimplementation of drive_seq, so leaving it in place when
    # no ablation is active silently shadows any fix made to the real
    # rda.py -- the probe would then report a stale trajectory and look
    # exactly like "the fix did nothing". Guarded on the flag set below.
    _ablate = a.norm_event or a.bound_event > 0
    if _ablate:
        def _patched(hs, pes):
            pe = pes
            if pe.dim() == 1:
                pe = pe.unsqueeze(-1)
            elif pe.dim() == 2:
                pe = pe.unsqueeze(-1)
            h = hs if hs.dim() == 3 else hs.unsqueeze(1)
            h = h.to(st.event_proj.weight.dtype)
            pe = pe.to(st.pe_proj.weight.dtype)
            raw = st.event_proj(st.h_norm(h))
            ev = st.event_drive_scale * raw
            if a.norm_event:
                ev = ev / (raw.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
            elif a.bound_event > 0:
                k = a.bound_event
                ev = k * torch.tanh(ev / k)
            return torch.tanh(ev + st.pe_proj(pe / st.pe_reference_scale))
        st.drive_seq = _patched
    else:
        print("no ablation flag set -> measuring the REAL drive_seq from rda.py")

    if a.freeze_base:
        for p in m.base.parameters():
            p.requires_grad_(False)
        for n_, p in m.named_parameters():
            if n_.startswith("predictor."):
                p.requires_grad_(False)

    # Mirror base_train.py's parameter groups AND its LR scaling exactly.
    # base_train.py applies batch_lr_scale = sqrt(total_batch/B_REF) = 0.25
    # for total_batch_size=32768, so the real drive-group LR is
    # 0.02 * 0.25 = 0.005, not 0.02. Using the unscaled value made the state
    # pathway move 4x too fast and reproduced nothing.
    B_REF = 524288
    scale = (32768.0 / B_REF) ** 0.5          # = 0.25, printed by the trainer
    LR_MATRIX = 0.02 * scale                  # -> 0.005
    LR_SCALAR = 0.5 * scale                   # -> 0.125
    print("batch_lr_scale = %.4f  ->  drive-group LR = %.5f" % (scale, LR_MATRIX))
    _state_m = list(st.film_scale.parameters()) + list(st.film_shift.parameters())
    _drive_m = list(st.event_proj.parameters()) + list(st.pe_proj.parameters())
    groups = [{"params": _state_m, "lr": LR_MATRIX, "initial_lr": LR_MATRIX,
               "weight_decay": 0.0, "betas": (0.9, 0.95)},
              {"params": _drive_m, "lr": LR_MATRIX, "initial_lr": LR_MATRIX,
               "weight_decay": 0.01, "betas": (0.9, 0.95)},
              {"params": [st.e_init], "lr": LR_SCALAR, "initial_lr": LR_SCALAR,
               "weight_decay": 0.0, "betas": (0.96, 0.95)}]
    opt = torch.optim.AdamW(groups, eps=1e-10)

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="train", device=device)

    print("device=%s steps=%d seed=%d freeze_base=%s d_model=%d"
          % (a.device, a.steps, a.seed, a.freeze_base, D_MODEL))
    print("LN weight norm at init = %.4f   LN bias norm at init = %.4f"
          % (st.h_norm.weight.norm().item(), st.h_norm.bias.norm().item()))
    print("Cauchy-Schwarz ceiling = ||event_proj row|| * ||h_norm||")
    print()

    rec = []
    carry = None
    lr_mult = 0.0
    for step in range(a.steps):
        x, y, _ = next(loader)
        out = m(x, y, E_t_external=carry)
        loss = out["loss"]
        loss.backward()

        # ---- gradient norms (signal 11/12/13) ----
        g_ep = st.event_proj.weight.grad
        g_bn = st.h_norm.weight.grad
        gn = lambda t: (0.0 if t is None else float(t.norm()))
        g_event_proj = gn(g_ep)
        g_hnorm = gn(g_bn)
        g_state_ep = gn(g_ep) + gn(st.pe_proj.weight.grad) + gn(g_bn)
        base_params = [p for p in m.base.parameters() if p.grad is not None]
        g_base = float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                                      for p in base_params))) if base_params else 0.0

        # ---- measure the drive decomposition on the REAL hidden states ----
        with torch.no_grad():
            h = out["hidden"][:, :-1, :].to(st.event_proj.weight.dtype)
            hn = st.h_norm(h)                              # (B,T-1,d)
            ev_raw = st.event_proj(hn)
            # Mirror the REAL rda.py computation exactly, so the measured event
            # term is the event term the model actually used. When no ablation
            # is active this must include the output RMS normalization that now
            # lives in RDAStateModule.drive_seq; otherwise the probe would
            # report the un-normalized magnitude and overstate the growth.
            if a.norm_event or a.bound_event > 0:
                ev = st.event_drive_scale * ev_raw
                if a.norm_event:
                    ev = ev / (ev_raw.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
                else:
                    k = a.bound_event
                    ev = k * torch.tanh(ev / k)
            else:
                ev = st.event_drive_scale * ev_raw / ev_raw.pow(2).mean(
                    dim=-1, keepdim=True).add(1e-6).sqrt()
            lg = m.base.lm_head(out["hidden"][:, :-1, :])[..., :V].float()
            lg = 15 * torch.tanh(lg / 15)
            pe_s = st.prediction_error_from_logits(lg, y[:, 1:])
            pv = st.pe_proj((pe_s / st.pe_reference_scale).unsqueeze(-1)
                            .to(st.pe_proj.weight.dtype))
            pre = ev + pv                                    # pre-tanh arg
            post = torch.tanh(pre)
            E = out["E_next_seq"]
            film_state = torch.cat(
                [carry.unsqueeze(1) if carry is not None
                 else st.init_state(x.size(0), x.device, torch.float32).unsqueeze(1),
                 E[:, :-1, :]], dim=1)
            fs = 1.0 + st.film_scale(film_state)
            fh = st.film_shift(film_state)
            ln_w = st.h_norm.weight
            ln_b = st.h_norm.bias
            row = dict(
                step=step,
                ln_out_rms=float(hn.pow(2).mean().sqrt()),
                ln_out_norm=float(hn.norm(dim=-1).mean()),
                ln_out_mean=float(hn.mean()), ln_out_std=float(hn.std()),
                ln_out_absmax=float(hn.abs().max()),
                ln_weight_norm=float(ln_w.norm()), ln_bias_norm=float(ln_b.norm()),
                ln_weight_absmax=float(ln_w.abs().max()),
                event_absmax=float(ev.abs().max()),
                event_perdim=[round(float(v), 5) for v in ev.abs().amax(dim=(0, 1))],
                pe_absmax=float(pv.abs().max()),
                pe_perdim=[round(float(v), 5) for v in pv.abs().amax(dim=(0, 1))],
                pretanh_absmax=float(pre.abs().max()),
                posttanh_absmax=float(post.abs().max()),
                E_absmax=float(E.abs().max()),
                E_final_absmax=float(out["E_next"].abs().max()),
                E_perdim=[round(float(v), 5) for v in out["E_next"].abs().max(dim=0).values],
                film_scale_absmax=float(fs.abs().max()),
                film_shift_absmax=float(fh.abs().max()),
                film_w_absmax=float(st.film_scale.weight.abs().max()),
                g_event_proj=g_event_proj, g_hnorm=g_hnorm,
                g_state_ep=g_state_ep, g_base=g_base,
                ep_w_norm=float(st.event_proj.weight.norm()),
                ep_b_norm=float(st.event_proj.bias.norm()),
                pe_w_norm=float(st.pe_proj.weight.norm()),
                pe_b_norm=float(st.pe_proj.bias.norm()),
                raw_h_rms=float(h.pow(2).mean().sqrt()),
                lr_applied=lr_mult,
                loss=float(loss.item()),
            )
            # Cauchy-Schwarz ceiling actually available at this step.
            row["cs_ceiling"] = float(
                st.event_proj.weight.norm(dim=1).max() * hn.norm(dim=-1).max())
            row["event_over_ceiling"] = row["event_absmax"] / row["cs_ceiling"]
            rec.append(row)

        # base_train's get_lr_multiplier: warmup over warmup_steps, then decay.
        # 40 warmup steps, so steps 0..39 are pure warmup and the drive group
        # LR is ~0.00125 at step 0 rising to 0.005 by step 40.
        warmup = 40
        if step < warmup:
            lr_mult = (step + 1) / warmup
        else:
            p = (step - warmup) / max(1, (a.steps - warmup))
            lr_mult = 1.0 - 0.95 * p
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_mult
        opt.step()
        opt.zero_grad(set_to_none=True)
        carry = out["E_next"].detach()

        if step in (0, 1, 4, 9, 14, 19, 24, a.steps - 1):
            r = rec[-1]
            print("step %3d | ln_out_rms=%.3f lnW=%.4f | ceiling=%.3f "
                  "event=%.4f (%.3f of ceil) | pe=%.4f | pretanh=%.4f | "
                  "posttanh=%.4f | maxE=%.4f | FiLM s=%.4f h=%.4f"
                  % (step, r["ln_out_rms"], r["ln_weight_norm"], r["cs_ceiling"],
                     r["event_absmax"], r["event_over_ceiling"], r["pe_absmax"],
                     r["pretanh_absmax"], r["posttanh_absmax"], r["E_absmax"],
                     r["film_scale_absmax"], r["film_shift_absmax"]))

    print()
    print("--- summary: what grew? ---")
    f, l = rec[0], rec[-1]
    for k, lab in (("ln_out_rms", "LN output rms"),
                   ("ln_weight_norm", "LN weight norm"),
                   ("ln_bias_norm", "LN bias norm"),
                   ("ln_weight_absmax", "LN weight |max|"),
                   ("event_absmax", "event |max|"),
                   ("pe_absmax", "pe |max|"),
                   ("pretanh_absmax", "pre-tanh |max|"),
                   ("posttanh_absmax", "post-tanh |max|"),
                   ("E_absmax", "state |E| |max|"),
                   ("cs_ceiling", "CS ceiling"),
                   ("event_over_ceiling", "event / ceiling"),
                   ("ep_w_norm", "event_proj ||W||"),
                   ("pe_b_norm", "pe_proj ||b||"),
                   ("film_scale_absmax", "FiLM scale |max|"),
                   ("raw_h_rms", "raw h rms"),
                   ("g_event_proj", "grad||event_proj||"),
                   ("g_state_ep", "grad||state ep||"),
                   ("g_base", "grad||base||")):
        print("  %-22s %10.4f -> %10.4f   (x%.2f)"
              % (lab, f[k], l[k], (l[k] / f[k]) if f[k] else float("nan")))
    print()
    print("  per-dim event contribution, first -> last:")
    for i, nm in enumerate(DRIVE_DIMS):
        print("    %-18s %7.4f -> %7.4f" % (nm, f["event_perdim"][i], l["event_perdim"][i]))
    print("  per-dim state |E| at last step:")
    for i, nm in enumerate(DRIVE_DIMS):
        print("    %-18s %7.4f" % (nm, l["E_perdim"][i]))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh_:
            json.dump(rec, fh_, indent=1)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
