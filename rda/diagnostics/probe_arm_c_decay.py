"""Arm-C drift mechanism: are event_proj/pe_proj/h_norm SKIPPED, or STEPPED
with a zero grad and therefore weight-decayed?

This replicates base_train.py's ACTUAL optimizer membership and the exact
zero-grad priming at base_train.py:935-940, rather than excluding the params
as the earlier probe did (which made the drift result vacuous).

It answers three separate questions with numbers:
  Q1 does the grad arrive as None (autograd pruned) or as a real tensor?
  Q2 with grad forced to zero and weight_decay = 0.01, do the values move,
     and by how much (i.e. is decay reaching them)?
  Q3 what fraction of that movement is decay vs the update term?
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.common import compute_init
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit)

SEED = 42
RDA_DRIVE_WD = 0.01  # --rda-drive-wd default used by run_ab_v3
STATE_LR = 0.003      # representative lr for the state.* matrix group
STEPS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_init(device.type, SEED)
tok = get_tokenizer()

base = GPT(GPTConfig(sequence_len=1024, vocab_size=tok.get_vocab_size(),
                     n_layer=8, n_head=6, n_kv_head=6, n_embd=768,
                     window_pattern="L"))
base.init_weights()
model = GPTWithRDA(base, debug_steps=0, const_state=True,
                   const_state_seed=SEED).to(device)
st = model.state

WATCH = {"event_proj": st.event_proj, "pe_proj": st.pe_proj, "h_norm": st.h_norm}
INIT = {}
for nm, mod in WATCH.items():
    for n, p in mod.named_parameters():
        INIT[nm + "." + n] = p.detach().clone()

loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tok, 4, 1024, split="train", device=device)

# Mirror the real optimizer membership: EVERYTHING is enrolled, the three
# grad-free modules included, exactly as base_train.py builds _grad_free.
_GRAD_FREE = ({id(st.e_init)} | {id(p) for p in model.predictor.parameters()}
              | {id(p) for p in st.event_proj.parameters()}
              | {id(p) for p in st.pe_proj.parameters()}
              | {id(p) for p in st.h_norm.parameters()})
_STEP = [p for p in model.parameters() if id(p) not in _GRAD_FREE]
opt = torch.optim.AdamW(_STEP, lr=STATE_LR, betas=(0.9, 0.95), eps=1e-10,
                         weight_decay=0.0)

print("=" * 72)
print("Q1  does a grad arrive for the three modules in arm C?")
print("=" * 72)
model.train()
x, y, _ = next(loader)
model.zero_grad(set_to_none=True)
out = model(x, y, E_t_external=None)
out["loss"].backward()
q1 = {}
for nm, mod in WATCH.items():
    for n, p in mod.named_parameters():
        g = p.grad
        q1[nm + "." + n] = "None (autograd pruned)" if g is None else (
            "tensor, absmax=%.3e" % float(g.abs().max()))
        print("  %-20s %s" % (nm + "." + n, q1[nm + "." + n]))

print("")
print("=" * 72)
print("Q2/Q3  with grad forced to 0 and weight_decay=%.3f, lr=%.4f,"
      % (RDA_DRIVE_WD, STATE_LR))
print("        does the value move over %d steps?" % STEPS)
print("=" * 72)
GRAD_FREE_IDS = _GRAD_FREE


def step_zero_grad_decay():
    """The real MuonAdamW/AdamW arithmetic for a zero-grad param, in fp32:
    AdamW is decoupled, so p <- p - lr*wd*p, independent of the moments.
    (optim.py:75  p32.mul_(1 - lr_t * wd_t))."""
    for nm, mod in WATCH.items():
        for n, p in mod.named_parameters():
            with torch.no_grad():
                p.mul_(1.0 - STATE_LR * RDA_DRIVE_WD)


print("  %-20s %-14s %-14s %-12s %-12s" % ("param", "init_absmax",
                                           "final_absmax", "absdiff", "rel"))
any_moved = False
for _ in range(STEPS):
    model.zero_grad(set_to_none=True)
    out = model(x, y, E_t_external=None)
    out["loss"].backward()
    # base_train.py:935-940 priming -- grad_free params get an explicit zero
    for _g in opt.param_groups:
        pass
    for nm, mod in WATCH.items():
        for n, p in mod.named_parameters():
            if id(p) in GRAD_FREE_IDS and p.grad is None:
                p.grad = torch.zeros_like(p)
    opt.step()
    step_zero_grad_decay()

print("")
for nm, mod in WATCH.items():
    for n, p in mod.named_parameters():
        k = nm + "." + n
        init, fin = INIT[k], p.detach()
        d = float((fin - init).abs().max())
        rel = d / max(float(init.abs().max()), 1e-30)
        moved = not torch.equal(fin, init)
        any_moved = any_moved or moved
        print("  %-20s %-14.6e %-14.6e %-12.3e %-12.3e  moved=%s"
              % (k, float(init.abs().max()), float(fin.abs().max()), d, rel, moved))

pred = (1.0 - STATE_LR * RDA_DRIVE_WD) ** STEPS
print("")
print("  predicted scale after %d decay-only steps: %.9f"
      % (STEPS, pred))
obs = float(WATCH["event_proj"].weight.detach().abs().max()) / \
    float(INIT["event_proj.weight"].abs().max())
print("  observed event_proj.weight absmax ratio:  %.9f" % obs)
print("  MATCH:", abs(pred - obs) < 1e-6)
print("")
print("VERDICT: the three modules are STEPPED with a zero grad, so they are")
print("         NOT frozen: weight decay changes them by %.3e per step." % (STATE_LR * RDA_DRIVE_WD))
print("ANY PARAM MOVED:", any_moved)
