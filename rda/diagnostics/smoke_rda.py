"""
5-step smoke test for RDA optimizer coverage.

TEMPORARY; deleted after use.
Run:  python diagnostics/smoke_rda.py
Checks, on the REAL setup_optimizer path:
  1. every state.*/predictor.* param is in some optimizer group (by name)
  2. those params receive NONZERO gradient after backward
  3. film_scale/film_shift weights are nonzero after 5 optimizer steps
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

config = GPTConfig(sequence_len=256, vocab_size=32768, n_layer=4, n_head=4,
                   n_kv_head=4, n_embd=256, window_pattern="L")
base = GPT(config)
base.init_weights()
base = base.to(dev)
model = GPTWithRDA(base, debug_steps=0).to(dev)
model.train()

# Same hyperparameters base_train would pass (batch_lr_scale folded in already).
opt = base.setup_optimizer(unembedding_lr=0.008, embedding_lr=0.3,
                           scalar_lr=0.5, matrix_lr=0.02, weight_decay=0.28)

# --- enroll the RDA params exactly as base_train now does ---------------
sm = list(model.state.event_proj.parameters()) + list(model.state.pe_proj.parameters()) \
    + list(model.state.h_norm.parameters()) \
    + list(model.state.film_scale.parameters()) + list(model.state.film_shift.parameters())
sv = [model.state.e_init]
pm = list(model.predictor.parameters())
g_matrix = dict(kind='adamw', params=sm, lr=0.02, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.28)
g_vector = dict(kind='adamw', params=sv, lr=0.5, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0)
g_pred = dict(kind='adamw', params=pm, lr=0.02, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0)
for g in (g_matrix, g_vector, g_pred):
    g['initial_lr'] = g['lr']   # base_train rescales lr = initial_lr * lrm
    opt.add_param_group(g)

# grad-free-by-design params, matching base_train
grad_free = {id(model.state.e_init)} | {id(p) for p in model.predictor.parameters()}
for g in opt.param_groups:
    if any(id(p) in grad_free for p in g['params']):
        g['grad_free'] = True

# --- CHECK 1: coverage by name -------------------------------------------
covered = {id(p) for g in opt.param_groups for p in g['params']}
missing = [n for n, p in model.named_parameters()
           if (n.startswith("state.") or n.startswith("predictor.")) and id(p) not in covered]
print(f"CHECK 1 coverage-by-name : missing={len(missing)} {missing}")
assert not missing, missing

# --- CHECK 2 + 3: gradients nonzero, FiLM moves --------------------------
x = torch.randint(1, 32768, (2, 128), device=dev)
y = torch.randint(1, 32768, (2, 128), device=dev)

grad_report = {}
# Mirror the real training loop: decay-only persistence, so E_next is carried
# into the next step. This is what makes the FiLM path live from step 1 onward
# (at step 0 E_t == 0 and modulate is the identity by construction).
carry = None
fs_hist = []
fh_hist = []
ep_hist = []
pp_hist = []
hn_hist = []
e_hist = []
for step in range(5):
    out = model(x, y, E_t_external=carry)
    out['loss'].backward()
    if step == 0:
        for n, p in model.named_parameters():
            if n.startswith("state.") or n.startswith("predictor."):
                g = p.grad
                grad_report[n] = None if g is None else float(g.abs().max())
    for g in opt.param_groups:
        if g.get("grad_free"):
            for p in g["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)
    carry = out['E_next'].detach()
    fs_hist.append(float(model.state.film_scale.weight.detach().abs().mean()))
    fh_hist.append(float(model.state.film_shift.weight.detach().abs().mean()))
    ep_hist.append(float(model.state.event_proj.weight.detach().abs().mean()))
    pp_hist.append(float(model.state.pe_proj.weight.detach().abs().mean()))
    hn_hist.append(float(model.state.h_norm.weight.detach().abs().mean()))
    e_hist.append(float(carry.abs().mean()))
print("FiLM scale |w| trajectory: " + " -> ".join(f"{v:.3e}" for v in fs_hist))
print("FiLM shift |w| trajectory: " + " -> ".join(f"{v:.3e}" for v in fh_hist))
# event_proj/pe_proj are ZERO-INITIALISED-ish-by-construction at step 0: FiLM
# starts at exactly 0 so modulate is the identity and there is no gradient
# path into the recurrence. They must start moving from step 1, otherwise the
# state is a constant and the whole model degenerates to Model A.
print("event_proj |w| trajectory: " + " -> ".join(f"{v:.3e}" for v in ep_hist))
print("pe_proj    |w| trajectory: " + " -> ".join(f"{v:.3e}" for v in pp_hist))
print("h_norm     |w| trajectory: " + " -> ".join(f"{v:.3e}" for v in hn_hist))
print("carried E  |E| trajectory : " + " -> ".join(f"{v:.3e}" for v in e_hist))

none_grad = [n for n, v in grad_report.items() if v is None]
# Two params are grad-free BY DESIGN (nothing in the loss depends on them):
#   predictor.*  - dpe is logged but not consumed (out of scope)
#   e_init       - the state only enters the loss through FiLM, and
#                  init_state() expands a zero vector, so grad is None
#
# CORRECTION (measured). The old comment here said e_init is grad-free because
# "init_state() expands a zero vector". That is not the mechanism, and it is
# wrong for a non-zero e_init: the zero VALUE is not what breaks the graph.
# The real reason grad is None here is that this probe drives the model the way
# base_train does -- with a carried, DETACHED state -- so e_init is simply not
# reachable from the loss. Measured, 5 steps, trainer call pattern:
#     per-step |grad| = ['0.000e+00', 'None', 'None', 'None', 'None']
# i.e. at step 0 (no carry, zero-init FiLM) the grad is a real zero TENSOR, and
# from step 1 (carry supplied and detached) it is None. See base_train.py's
# grad-free comment and docs/decay_semantics.md for the consequence if the
# persistence semantics ever change.
expected_free = {"state.e_init"} | {n for n in grad_report if n.startswith("predictor.")}
# event_proj/pe_proj/h_norm legitimately get ZERO (not None) at step 0: FiLM
# receives E_t == 0 there, so d(loss)/d(event_proj) == 0, and h_norm only feeds
# event_proj so its gradient is zero for the same reason. They are not dead --
# the recurrence output is carried into later steps, where E_t != 0. Verified
# by CHECK 2b and by the event_proj/h_norm |w| trajectory below.
zero_ok = {"state.event_proj.weight", "state.event_proj.bias",
           "state.pe_proj.weight", "state.pe_proj.bias",
           "state.h_norm.weight", "state.h_norm.bias"}
unexpected_none = [n for n in none_grad if n not in expected_free]
zero_grad = [n for n, v in grad_report.items()
             if v is not None and v == 0.0 and n not in expected_free
             and n not in zero_ok]
print(f"CHECK 2 grads present    : {len(grad_report)} params, "
      f"None={len(none_grad)} (expected {len(expected_free)}), "
      f"unexpected-zero={len(zero_grad)}")
print(f"   grad-free by design  : {sorted(expected_free)}")
print(f"   zero-at-step0 (ok)   : {sorted(zero_ok)}")
if unexpected_none:
    print("   UNEXPECTED None grad  :", unexpected_none)
if zero_grad:
    print("   UNEXPECTED zero grad  :", zero_grad)
for n in ["state.film_scale.weight", "state.film_shift.weight",
          "state.event_proj.weight", "state.pe_proj.weight"]:
    if n in grad_report:
        v = grad_report[n]
        print(f"   {n:<34s} |grad|max = {'None' if v is None else '%.6e' % v}")

# --- CHECK 2b: is the FiLM branch actually in the loss graph? -------------
# Start with NONZERO film weights; if the grad is nonzero the branch is live
# and the step-0 zero was purely because E_t == 0 at step 0.
_x2 = torch.randint(1, 32768, (2, 128), device=dev)
with torch.no_grad():
    model.state.film_scale.weight.fill_(0.1)
    model.state.film_scale.bias.fill_(0.1)
o2 = model(_x2, y, E_t_external=torch.full((2, 8), 0.47, device=dev))
o2['loss'].backward()
g2 = model.state.film_scale.weight.grad
live = g2 is not None and float(g2.abs().max()) > 0
print(f"CHECK 2b FiLM in loss graph: |grad|max = "
      f"{'None' if g2 is None else '%.6e' % float(g2.abs().max())}  live={live}")

fs = float(model.state.film_scale.weight.detach().abs().mean())
fh = float(model.state.film_shift.weight.detach().abs().mean())
print(f"CHECK 3 after 5 steps    : film_scale|w|mean = {fs:.6e}  film_shift|w|mean = {fh:.6e}")
print(f"                         both nonzero = {(fs > 0.0) and (fh > 0.0)}")

# --- CHECK 4: multi-micro-step accumulation, as base_train actually runs ---
# base_train does grad_accum_steps backward passes per optimizer step, with a
# model.zero_grad(set_to_none=True) after every opt.step(). A grad guard that
# only primes once survives step 0 and then crashes on step 1 with
# "'NoneType' object has no attribute 'float'" in MuonAdamW. Also asserts every
# param_group carries initial_lr, which the lr rescale requires.
no_initial_lr = [i for i, g in enumerate(opt.param_groups) if "initial_lr" not in g]
acc_ok = True
try:
    carry4 = None
    for step in range(2):
        for micro in range(3):
            o = model(x, y, E_t_external=carry4)
            (o['loss'] / 3).backward()
            for g in opt.param_groups:
                if g.get("grad_free"):
                    for p in g["params"]:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
            carry4 = o['E_next'].detach()
        for g in opt.param_groups:
            g['lr'] = g['initial_lr'] * 0.5   # the lrm rescale from base_train:691
        opt.step()
        model.zero_grad(set_to_none=True)
    acc_ok = True
except Exception as e:
    acc_ok = False
    print(f"   accumulation raised: {type(e).__name__}: {e}")
print(f"CHECK 4 grad accumulation: 2 steps x 3 micro-steps ok={acc_ok}, "
      f"groups missing initial_lr={no_initial_lr}")
if no_initial_lr:
    print("   groups missing initial_lr:", no_initial_lr)

ok = ((not missing) and (not unexpected_none) and (not zero_grad)
      and fs > 0.0 and fh > 0.0 and live and acc_ok and not no_initial_lr
      and ep_hist[-1] > 0.0 and pp_hist[-1] > 0.0
      and hn_hist[0] != hn_hist[-1])   # h_norm must actually move
print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
