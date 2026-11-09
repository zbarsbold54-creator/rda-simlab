"""
Task 4 (60 steps): saturation smoke test WITH the drive weight decay AND the
h_norm + event_drive_scale = 0.1 output bound active.

GATE -- BOTH checkpoints must pass (steps 30 and 60):
  (a) no dimension exceeds saturation fraction 0.99 at either checkpoint
  (b) the trend over steps 30->60 is FLAT or DECLINING for EVERY dimension

Rationale for the two-checkpoint gate: last round the 30-step test showed every
dimension sitting under 0.99 while still climbing, which passed the naive check
and then ratcheted to |E| = sqrt(8) by step ~130. So a sub-threshold climb is a
FAILURE, not a pass.

Saturation fraction per dimension = fraction of steps in which the batch-mean
|E[dim]| exceeded 0.99, i.e. the dimension sat at the rail.
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=60)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--device-batch-size", type=int, default=4)
p.add_argument("--max-seq-len", type=int, default=1024)
p.add_argument("--lr-scale", type=float, default=0.25)
p.add_argument("--drive-wd", type=float, default=0.01)
p.add_argument("--baseline", action="store_true",
               help="run with drive wd=0 to reproduce the saturated run")
args = p.parse_args()

device = torch.device(autodetect_device_type())
torch.manual_seed(1337)

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
model_dim = args.depth * 64
cfg = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size,
                n_layer=args.depth, n_head=max(1, model_dim // 128),
                n_kv_head=max(1, model_dim // 128), n_embd=model_dim,
                window_pattern="L")

base = GPT(cfg); base.init_weights(); base = base.to(device)
model = GPTWithRDA(base, debug_steps=0).to(device)
model.train()

matrix_lr = 0.02 * args.lr_scale
scalar_lr = 0.5 * args.lr_scale
drive_wd = 0.0 if args.baseline else args.drive_wd

film = (list(model.state.film_scale.parameters())
        + list(model.state.film_shift.parameters()))
drive = (list(model.state.event_proj.parameters())
         + list(model.state.pe_proj.parameters())
         + list(model.state.h_norm.parameters()))
svec = [model.state.e_init]
pred = list(model.predictor.parameters())

opt = base.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2,
                           matrix_lr=0.02, weight_decay=0.195865, scalar_lr=0.5)
for g, lr, w in ((film, matrix_lr, 0.0), (drive, matrix_lr, drive_wd),
                 (svec, scalar_lr, 0.0), (pred, matrix_lr, 0.0)):
    d = dict(kind='adamw', params=g, lr=lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=w)
    d["initial_lr"] = lr
    opt.add_param_group(d)
grad_free = {id(model.state.e_init)} | {id(q) for q in model.predictor.parameters()}
for g in opt.param_groups:
    if any(id(q) in grad_free for q in g["params"]):
        g["grad_free"] = True

loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device)
x, y, _ = next(loader)

N = len(DRIVE_DIMS)
carry = None
sat_count = [[0] * args.steps for _ in range(N)]   # sat_count[dim][step]
mean_abs  = [[0.0] * args.steps for _ in range(N)]
norm_hist, ev_hist, pe_hist, dw_hist = [], [], [], []
THRESH = 0.99   # |E| > 0.99 counts as saturated
CKPTS = (30, 60)
hist_by_ckpt = {}

print(f"COMPUTE_DTYPE={COMPUTE_DTYPE} d_model={model_dim} drive_wd={drive_wd}"
      f"{'  [BASELINE wd=0]' if args.baseline else ''}")
print()
hdr = f"{'step':>4} {'loss':>9} {'|E|':>7} {'sat_dims':>8} {'ev|.|':>8} {'pe|.|':>8} {'drive w':>9}"
print(hdr); print("-" * len(hdr))

for step in range(args.steps):
    cap = {}
    h1 = model.state.event_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__('ev', o.detach().abs().mean().item()))
    h2 = model.state.pe_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__('pe', o.detach().abs().mean().item()))
    out = model(x, y, E_t_external=carry)
    h1.remove(); h2.remove()

    E = out["E_next"].detach()                 # (B, n_drives)
    per_dim = E.abs().mean(dim=0)              # (n_drives,)
    n_sat = 0
    for d in range(N):
        v = float(per_dim[d])
        mean_abs[d][step] = v
        if v > THRESH:
            sat_count[d][step] = 1
            n_sat += 1
    norm_hist.append(E.norm(dim=-1).mean().item())
    ev_hist.append(cap['ev'] * model.state.event_drive_scale); pe_hist.append(cap['pe'])
    dw = float(model.state.event_proj.weight.detach().abs().mean())
    dw_hist.append(dw)
    print(f"{step:>4} {out['loss'].item():>9.4f} {norm_hist[-1]:>7.4f} "
          f"{n_sat:>4}/8 {cap['ev']:>8.4f} {cap['pe']:>8.4f} {dw:>9.5f}")

    out["loss"].backward()
    for g in opt.param_groups:
        if g.get("grad_free"):
            for q in g["params"]:
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
    opt.step()
    opt.zero_grad(set_to_none=True)
    carry = out["E_next"].detach()
    x, y, _ = next(loader)

    if (step + 1) in CKPTS:
        hist_by_ckpt[step + 1] = {
            'sat': [sum(sat_count[d][:step+1]) / (step+1) for d in range(N)],
            'mabs': [list(mean_abs[d][:step+1]) for d in range(N)],
            'n': step + 1,
        }

print()
print("=" * 78)
print(f"PER-DIMENSION SATURATION FRACTION (|E_dim| > {THRESH}) at BOTH checkpoints")
print("=" * 78)
for ck in CKPTS:
    h = hist_by_ckpt.get(ck)
    if h is None:
        print(f"\nstep {ck}: NOT REACHED (ran {args.steps} steps)")
        continue
    print(f"\n--- at step {ck} (over {h['n']} steps) ---")
    print(f"{'dim':<20} {'sat_frac':>9} {'mean|E|':>9} {'max|E|':>8}  verdict")
    print("-" * 58)
    for d, name in enumerate(DRIVE_DIMS):
        sat = h['sat'][d]
        m = sum(h['mabs'][d]) / h['n']
        mx = max(h['mabs'][d])
        print(f"{name:<20} {sat:>9.3f} {m:>9.4f} {mx:>8.4f}  "
              f"{'OVER' if sat > 0.99 else 'ok'}")

# ---- trend 30 -> 60 ----
h30, h60 = hist_by_ckpt.get(30), hist_by_ckpt.get(60)
print()
print("=" * 78)
print("TREND, steps 30 -> 60 (mean |E_dim| over the 10 steps before each checkpoint)")
print("=" * 78)
if h30 and h60:
    print(f"{'dim':<20} {'mean 21-30':>11} {'mean 51-60':>11} {'delta':>9}  verdict")
    print("-" * 62)
    trend_ok = True
    for d, name in enumerate(DRIVE_DIMS):
        w30 = h30['mabs'][d][-10:]
        w60 = h60['mabs'][d][-10:]
        a = sum(w30) / len(w30)
        b = sum(w60) / len(w60)
        delta = b - a
        flat = delta <= 0.02
        if not flat:
            trend_ok = False
        print(f"{name:<20} {a:>11.4f} {b:>11.4f} {delta:>+9.4f}  "
              f"{'flat/declining' if flat else 'CLIMBING'}")
else:
    trend_ok = False
    print("  (cannot evaluate: run 60 steps)")

sat_ok = all(hist_by_ckpt[c]['sat'][d] <= 0.99
             for c in CKPTS if c in hist_by_ckpt
             for d in range(N))
print()
print(f"|E| trajectory (every 5): {' -> '.join(f'{v:.3f}' for v in norm_hist[::5])}")
print(f"event_proj EFF |.|     : {' -> '.join(f'{v:.3f}' for v in ev_hist[::5])}")
print(f"pe_proj        |.|     : {' -> '.join(f'{v:.3f}' for v in pe_hist[::5])}")
print()
print(f"GATE (a) sat_frac <= 0.99 at steps {list(CKPTS)} : {'PASS' if sat_ok else 'FAIL'}")
print(f"GATE (b) trend 30->60 flat/declining all dims   : {'PASS' if trend_ok else 'FAIL'}")
ok = sat_ok and trend_ok
print()
print("SMOKE TEST:", "PASS" if ok else "FAIL", "--",
      "proceed to full run" if ok else "DO NOT proceed; diagnose further")
sys.exit(0 if ok else 1)
