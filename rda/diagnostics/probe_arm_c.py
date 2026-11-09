"""Arm-C smoke instrumentation: gradient norms, forward-call counts, drift.

NO TRAINING. Builds the model exactly as base_train.py does, runs a few real
forward/backward passes on real data, and reports:

  1. gradient norms of const_state_vec, film_scale.weight, film_shift.weight
     at each backward. The FIRST is zero BY DESIGN (FiLM weights are zero-init,
     so d/dW is proportional to c and d/dc is proportional to W, and W == 0).
     A zero at the SECOND or later is the saddle and is a hard stop.
  2. how many times update_seq / predict_seq / event_proj are called per step,
     so "the drive path still runs in C" is a COUNT, not an inference.
  3. whether event_proj / pe_proj / h_norm are bit-identical to their init
     after the steps (no update, no decay drift).

Everything is observation only: no model, gain or drive code is modified.
"""
import argparse
import os
import sys

# The repo root must be importable: this file lives in diagnostics/, and
# running it as a script puts THAT directory on sys.path, not the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.common import compute_init
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit)

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["Bprime", "C"], required=True)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

device = torch.device(a.device)
compute_init("cpu" if device.type == "cpu" else "cuda", a.seed)
tok = get_tokenizer()
V = tok.get_vocab_size()

base = GPT(GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                     n_kv_head=6, n_embd=768, window_pattern="L"))
base.init_weights()
const = (a.arm == "C")
model = GPTWithRDA(base, debug_steps=0, const_state=const,
                   const_state_seed=a.seed).to(device)
st = model.state

# ---- forward-call counters -----------------------------------------------
# event_proj is an nn.Module, so it cannot be attribute-wrapped (PyTorch
# rejects a non-Module assignment to a child module). It gets a forward HOOK,
# which is observation-only and returns nothing, so the value is unchanged.
# The rest are bound METHODS, so instance wrapping is the right tool.
CALLS = {"update_seq": 0, "predict_seq": 0, "event_proj": 0,
         "drive_seq": 0, "modulate": 0}
_orig = {}
_handles = []


def _wrap(name, fn):
    def inner(*x, **kw):
        CALLS[name] += 1
        return fn(*x, **kw)
    return inner


for name in ("update_seq", "drive_seq", "modulate"):
    obj = getattr(st, name)
    _orig[name] = obj
    setattr(st, name, _wrap(name, obj))
_orig["predict_seq"] = model.predictor.predict_seq
model.predictor.predict_seq = _wrap("predict_seq", _orig["predict_seq"])
_handles.append(st.event_proj.register_forward_hook(
    lambda mod, inp, out: CALLS.__setitem__("event_proj",
                                            CALLS["event_proj"] + 1)))

# init snapshot for the drift check
NAMES = ["event_proj", "pe_proj", "h_norm"]
INIT = {}
for nm in NAMES:
    for n, p in getattr(st, nm).named_parameters():
        INIT[nm + "." + n] = p.detach().clone()
if const:
    INIT_VEC = model.const_state_vec.detach().clone()
    print("const_state_vec init :", ["%+.6f" % v for v in INIT_VEC.tolist()])
else:
    INIT_VEC = None

loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tok, 4, 1024, split="train", device=device)

# The optimizer MUST actually move the FiLM weights, or this probe is vacuous:
# they are zero-init and d/d(const_state_vec) is proportional to them, so with a
# zero LR the constant's gradient is exactly 0.0 at EVERY step and check 3
# measures nothing but the zero init.
# Grad-free params (e_init, predictor.*, and in arm C the three drive
# projections) are excluded here ONLY because this probe passes
# weight_decay=0.0, so nothing can move them. That is NOT the training
# configuration: in the real run they are enrolled in a group with
# weight_decay=rda_drive_wd and decayed every step even with a zero grad
# (optim.py:75). So the drift result printed below is a property of THIS
# probe, not of arm C. See diagnostics/probe_arm_c_decay.py for the real
# mechanism.
_GRAD_FREE = {id(st.e_init)} | {id(p) for p in model.predictor.parameters()}
if const:
    _GRAD_FREE |= {id(p) for p in st.event_proj.parameters()}
    _GRAD_FREE |= {id(p) for p in st.pe_proj.parameters()}
    _GRAD_FREE |= {id(p) for p in st.h_norm.parameters()}
_STEP = [p for p in model.parameters() if id(p) not in _GRAD_FREE]
opt = torch.optim.AdamW(_STEP, lr=1e-3, betas=(0.9, 0.95), eps=1e-10,
                         weight_decay=0.0)

WATCH = ["const_state_vec", "film_scale.weight", "film_shift.weight"]
model.train()
print("")
print("=== ARM %s (const_state=%s) ===" % (a.arm, const))
print("backward-by-backward gradient norms")
print("%-5s %-24s %-24s %-24s" % ("step", "const_state_vec", "film_scale.w", "film_shift.w"))
zero_at_2plus = []
for step in range(a.steps):
    x, y, _ = next(loader)
    model.zero_grad(set_to_none=True)
    out = model(x, y, E_t_external=None)
    out["loss"].backward()
    row = []
    for w in WATCH:
        if w == "const_state_vec":
            p = model.const_state_vec
        elif w == "film_scale.weight":
            p = st.film_scale.weight
        else:
            p = st.film_shift.weight
        # p is None in B' (no constant). That is the arm's whole point, so a
        # missing param is NOT a saddle -- only a param that EXISTS with an
        # exactly-zero gradient counts.
        g = None if p is None else p.grad
        gn = 0.0 if g is None else float(g.detach().float().norm())
        row.append(gn)
        if g is not None and gn == 0.0 and step >= 1:
            zero_at_2plus.append((step + 1, w))
    print("%-5d %-24.10e %-24.10e %-24.10e" % (step + 1, row[0], row[1], row[2]))
    opt.step()
    print("        after opt.step(): film_scale.w absmax = %.6e  (was 0 at init)"
          % float(st.film_scale.weight.detach().abs().max()))

print("")
print("forward calls per step")
for k in ("drive_seq", "update_seq", "predict_seq", "event_proj", "modulate"):
    print("  %-12s %6d total  %.1f per step"
          % (k, CALLS[k], CALLS[k] / a.steps))

print("")
print("drive-module drift vs init (these params are grad-free, so even though")
print("the optimizer DID step above, they must not move at all)")
allsame = True
for nm in NAMES:
    for n, p in getattr(st, nm).named_parameters():
        k = nm + "." + n
        same = torch.equal(p.detach(), INIT[k])
        d = float((p.detach() - INIT[k]).abs().max())
        allsame = allsame and same
        print("  %-24s bit-identical=%-6s maxabsdiff=%.3e" % (k, same, d))
print("  ALL BIT-IDENTICAL:", allsame)

# restore the wrapped methods and remove the hook
for name in ("update_seq", "drive_seq", "modulate"):
    setattr(st, name, _orig[name])
model.predictor.predict_seq = _orig["predict_seq"]
for _h in _handles:
    _h.remove()

print("")
if INIT_VEC is not None:
    print("const_state_vec now  :", ["%+.6f" % v for v in
                                     model.const_state_vec.detach().tolist()])
print("SADDLE CHECK (zero at 2nd backward or later):",
      ("FAIL " + str(zero_at_2plus)) if zero_at_2plus else "PASS - none exactly 0.0")
print("CALLS_PER_STEP:", {k: CALLS[k] // a.steps for k in CALLS})
if zero_at_2plus:
    sys.exit(1)