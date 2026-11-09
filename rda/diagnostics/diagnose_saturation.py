"""
200-step saturation diagnostic with an abort-on-saturation tripwire.

MEASUREMENT ONLY. No changes to nanochat/rda.py.

TRIPWIRE: if any drive dimension has |E| > 0.98 for 10 CONSECUTIVE steps,
abort immediately and report which dimension(s) tripped and at what step.

WHY THE FIXED-POINT ESTIMATE MATTERS. The recurrence is

    E_next = decay * E_t + (1 - decay) * tanh(event_drive + pe_drive)

For a locally constant drive d, the fixed point is E* = d, and the gap
    E_t - d
decays geometrically by `decay` per update. Critically, `update()` runs
T-1 = 1023 times inside a single forward pass, so that gap is multiplied by
0.95^1023 ~ 1e-22 within ONE training step. The carried state across training
steps is therefore numerically irrelevant: each step's E_t is set almost
entirely by that step's own final-position drive.

So the two hypotheses separate cleanly:
  BENIGN   E_t tracks its own current drive d closely (small gap) and both
           rise together as the base model learns.
  BROKEN   the gap stays wide or grows, meaning E_t is moving for some reason
           other than simple convergence onto its own drive.

This script reports the gap so the distinction is measured, not assumed.
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
p.add_argument("--steps", type=int, default=200)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--device-batch-size", type=int, default=4)
p.add_argument("--max-seq-len", type=int, default=1024)
p.add_argument("--lr-scale", type=float, default=0.25)
p.add_argument("--drive-wd", type=float, default=0.01)
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
film = (list(model.state.film_scale.parameters())
        + list(model.state.film_shift.parameters()))
drive = (list(model.state.event_proj.parameters())
         + list(model.state.pe_proj.parameters())
         + list(model.state.h_norm.parameters()))
svec = [model.state.e_init]
pred = list(model.predictor.parameters())

opt = base.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2,
                           matrix_lr=0.02, weight_decay=0.195865, scalar_lr=0.5)
for g, lr, w in ((film, matrix_lr, 0.0), (drive, matrix_lr, args.drive_wd),
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
TRIP = 0.98
CONSEC = 10
CKPTS = (30, 60, 90, 120, 150, 200)
SAT_FRAC = 0.99

# consec[d] = consecutive steps dim d has been above the rail
consec = [0] * N
hist = {}     # ckpt -> dict
carry = None
tripped_at = None

print(f"COMPUTE_DTYPE={COMPUTE_DTYPE} d_model={model_dim} steps={args.steps} "
      f"drive_wd={args.drive_wd} event_scale={model.state.event_drive_scale}")
print(f"TRIPWIRE: |E_dim| > {TRIP} for {CONSEC} consecutive steps -> abort")
print(f"decay={model.state.decay}  decay^(T-1) = "
      f"{model.state.decay ** (args.max_seq_len - 1):.3e}  (gap decays within one step)")
print()
print(f"{'step':>4} {'loss':>9} {'|E|':>7} {'sat':>5} {'ev_eff':>8} {'pe':>7} "
      f"{'max consec':>10}")
print("-" * 66)

for step in range(args.steps):
    cap = {}
    # These fire once per sequence position (T-1 times); the LAST call is the
    # one that produced E_next, which is the drive the state is heading toward.
    h1 = model.state.event_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__('ev', o.detach()))
    h2 = model.state.pe_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__('pe', o.detach()))
    out = model(x, y, E_t_external=carry)
    h1.remove(); h2.remove()

    E = out["E_next"].detach()                       # (B, N)
    per_dim_mean = E.mean(dim=0)                     # (N,) signed, batch-mean
    per_dim_absmax = E.abs().max(dim=0).values       # (N,) worst in batch

    ev_eff = model.state.event_drive_scale * cap['ev']   # (B, N) last position
    drive_last = torch.tanh(ev_eff + cap['pe'])          # (B, N) fixed point E*
    d_mean = drive_last.mean(dim=0)                      # (N,)

    n_sat = int((per_dim_absmax > TRIP).sum())
    for dd in range(N):
        consec[dd] = consec[dd] + 1 if float(per_dim_absmax[dd]) > TRIP else 0

    print(f"{step:>4} {out['loss'].item():>9.4f} "
          f"{float(E.norm(dim=-1).mean()):>7.4f} {n_sat:>3}/8 "
          f"{float(ev_eff.abs().mean()):>8.4f} {float(cap['pe'].abs().mean()):>7.4f} "
          f"{max(consec):>10}")

    if (step + 1) in CKPTS:
        hist[step + 1] = dict(
            E=per_dim_mean.tolist(), Eabs=E.abs().mean(dim=0).tolist(),
            Emax=per_dim_absmax.tolist(), d=d_mean.tolist(),
            gap=(per_dim_mean - d_mean).tolist(), loss=out["loss"].item())

    # ---- TRIPWIRE ----
    if max(consec) >= CONSEC:
        bad = [DRIVE_DIMS[dd] for dd in range(N) if consec[dd] >= CONSEC]
        tripped_at = step + 1
        print()
        print("!" * 66)
        print(f"TRIPWIRE FIRED at step {tripped_at}: {', '.join(bad)}")
        print(f"  |E_dim| > {TRIP} for {CONSEC} consecutive steps")
        for dd in range(N):
            if consec[dd] >= CONSEC:
                print(f"  {DRIVE_DIMS[dd]:<20} |E|max={float(per_dim_absmax[dd]):.4f} "
                      f"E={float(per_dim_mean[dd]):+.4f} d={float(d_mean[dd]):+.4f}")
        print("!" * 66)
        break

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

# ---------------------------------------------------------------- report
print()
print("=" * 86)
print(f"PER-DIMENSION E_t AND ITS FIXED-POINT ESTIMATE  d = tanh(event_drive + pe_drive)")
print("=" * 86)
for ck in CKPTS:
    h = hist.get(ck)
    if h is None:
        continue
    print(f"\n--- step {ck}   (loss {h['loss']:.4f}) ---")
    print(f"{'dim':<20} {'E_t':>9} {'|E_t|max':>9} {'d=E*':>9} "
          f"{'gap E-d':>9} {'|gap|':>8}  flattened?")
    print("-" * 76)
    for dd, name in enumerate(DRIVE_DIMS):
        flat = "YES" if h['Eabs'][dd] < TRIP else "no"
        print(f"{name:<20} {h['E'][dd]:>+9.4f} {h['Emax'][dd]:>9.4f} "
              f"{h['d'][dd]:>+9.4f} {h['gap'][dd]:>+9.4f} "
              f"{abs(h['gap'][dd]):>8.4f}  {flat}")

print()
print("=" * 86)
print("GAP TRAJECTORY  (|E_t - d| -- should SHRINK if E_t converges onto its own drive)")
print("=" * 86)
cks = [ck for ck in CKPTS if ck in hist]
print(f"{'dim':<20} " + " ".join(f"{ck:>9}" for ck in cks) + "   monotone-shrinking?")
print("-" * (22 + 10 * len(cks) + 22))
mono_all = True
for dd, name in enumerate(DRIVE_DIMS):
    g = [abs(hist[ck]['gap'][dd]) for ck in cks]
    mono = all(g[i+1] <= g[i] + 1e-9 for i in range(len(g)-1))
    if not mono:
        mono_all = False
    print(f"{name:<20} " + " ".join(f"{v:>9.4f}" for v in g) +
          f"   {'YES' if mono else 'no'}")
print()

n_over = sum(1 for ck in cks for dd in range(N) if hist[ck]['Eabs'][dd] > SAT_FRAC)
print(f"TRIPWIRE FIRED              : {'YES at step %d' % tripped_at if tripped_at else 'NO'}")
print(f"dims > 0.99 at any ckpt     : {n_over}")
print(f"all gaps shrink monotonically: {'YES' if mono_all else 'no'}")
final = cks[-1]
flat_count = sum(1 for dd in range(N) if hist[final]['Eabs'][dd] < TRIP)
print(f"dims flattened < {TRIP} at step {final} : {flat_count}/8")
print()
ok = (tripped_at is None) and n_over == 0 and flat_count == 8
print("DIAGNOSTIC:", "BENIGN" if ok else "ATTENTION REQUIRED", "--",
      "proceed to full 600-step run" if ok else "do NOT relaunch; discuss first")
sys.exit(0 if ok else 1)
