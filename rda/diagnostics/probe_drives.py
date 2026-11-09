"""
Task 2: instrument event_proj(h) and pe_proj(pe/10) SEPARATELY.

The saturated run showed E_next pinned at sqrt(8) = 2.8284 from step ~129
onward. The recurrence's drive is

    drive = tanh( event_proj(hidden) + pe_proj(pe / 10) )
    E_next = decay * E_t + (1 - decay) * drive

so the drive terms are the only way a dimension can reach a rail. This script
reports each term's mean absolute output magnitude per step, so we can tell
WHICH term is crossing the pinning threshold and whether one dominates.

Runs the real training path (real data, real config, real optimizer) for 30
steps. Nothing is saved; this is measurement only.
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=30)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--device-batch-size", type=int, default=4)
p.add_argument("--max-seq-len", type=int, default=1024)
p.add_argument("--grad-accum", type=int, default=8)
p.add_argument("--total-batch-size", type=int, default=32768)
p.add_argument("--lr-scale", type=float, default=0.25, help="batch-size LR scale used by base_train")
args = p.parse_args()

device_type = autodetect_device_type()
device = torch.device(device_type)

torch.manual_seed(1337)
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
bos_token_id = tokenizer.get_bos_token_id()

model_dim = args.depth * 64
cfg = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size,
                n_layer=args.depth, n_head=max(1, model_dim // 128),
                n_kv_head=max(1, model_dim // 128), n_embd=model_dim,
                window_pattern="L")

base = GPT(cfg)
base.init_weights()
base = base.to(device)
model = GPTWithRDA(base, debug_steps=0).to(device)
model.train()

# Mirrors base_train's LR scheme for the three RDA groups.
matrix_lr = 0.02 * args.lr_scale
scalar_lr = 0.5 * args.lr_scale
wd = 0.195865
sm = (list(model.state.event_proj.parameters())
      + list(model.state.pe_proj.parameters())
      + list(model.state.film_scale.parameters())
      + list(model.state.film_shift.parameters()))
sv = [model.state.e_init]
pm = list(model.predictor.parameters())
opt = base.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2,
                           matrix_lr=0.02, weight_decay=wd, scalar_lr=0.5)
for g, lr, w in ((sm, matrix_lr, wd), (sv, scalar_lr, 0.0), (pm, matrix_lr, 0.0)):
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

state = model.state
carry = None
print(f"COMPUTE_DTYPE={COMPUTE_DTYPE}  d_model={model_dim}  "
      f"pe_reference_scale={state.pe_reference_scale}  decay={state.decay}")
print()
hdr = (f"{'step':>4} {'loss':>9} {'|E_t|':>8} {'ev_raw':>9} {'ev_eff':>9} "
       f"{'pe|.|':>9} {'sum|.|':>9} {'pe_raw':>9} {'dom':>5} {'tanh(sum)':>9}")
print(hdr)
print("-" * len(hdr))
print("# ev_raw = event_proj output as hooked (POST-LayerNorm, PRE-scale)")
print(f"# ev_eff = the term actually entering tanh = {0.1} * ev_raw")
print()

rows = []
for step in range(args.steps):
    # Per-term capture. event_proj/pe_proj are called inside update(); we hook
    # them so we see the RAW pre-tanh outputs, not their sum.
    cap = {}
    orig_ev, orig_pe = state.event_proj, state.pe_proj

    def ev_hook(_mod, _inp, out):
        cap['ev'] = out.detach()
        return out

    def pe_hook(_mod, _inp, out):
        cap['pe'] = out.detach()
        return out

    h1 = orig_ev.register_forward_hook(ev_hook)
    h2 = orig_pe.register_forward_hook(pe_hook)
    out = model(x, y, E_t_external=carry)
    h1.remove(); h2.remove()

    # SHAPES. The production path is the VECTORIZED drive_seq, which calls
    # event_proj ONCE on the whole (B, T, d) batch -- so the hook fires ONCE and
    # the captured tensors are (B, T, n_drives), not (B, n_drives).
    #
    # The previous comment here ("update() runs once per sequence position, so
    # the hook fired T-1 times ... Aggregate over the last position") described
    # the SEQUENTIAL implementation and had not been updated. It was wrong: only
    # the LAST invocation was kept, and the shapes were mislabelled.
    ev = cap['ev']            # (B, T, n_drives)
    pe = cap['pe']            # (B, T, n_drives)
    # The hook captures event_proj's RAW output. What actually enters tanh is
    # the RMS-NORMALIZED value (rda.py normalizes ev over the drive axis in
    # drive_seq), so the normalization has to be applied here or the probe
    # reports a quantity that never reaches the state.
    ev_n = ev / ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    # Reduce over batch AND positions, with explicit dims. A bare .mean() would
    # also reduce over the drive axis and collapse the 8 components together,
    # which is exactly the quantity this probe exists to keep separate.
    ev_abs = ev_n.abs().mean(dim=(0, 1))             # (n_drives,)
    pe_abs = pe.abs().mean(dim=(0, 1))                # (n_drives,)
    ev_eff = ev_abs * state.event_drive_scale         # what enters tanh
    # THE TANH ARGUMENT, summed per drive BEFORE reducing.
    #
    # The old line was
    #     sum_abs = (ev_eff + pe).abs().mean().item() if False else ev_eff + pe_abs
    # which had three defects:
    #   1. `if False` permanently disabled the only branch that was correct;
    #   2. that branch was itself broken -- ev_eff and pe had already been
    #      reduced to python floats by .item(), so it mixed a SCALAR with a
    #      (B,T,n) tensor and silently broadcast rather than erroring;
    #   3. the LIVE branch computed ev_eff + pe_abs, a sum of two INDEPENDENT
    #      absolute means. That is not the magnitude of the tanh argument and
    #      does not equal it even when the two terms share a sign, because
    #      |a|+|b| != |a+b| and the per-position cancellation is lost. The
    #      printed math.tanh(sum_abs) column was therefore not the drive.
    # Sum the per-drive magnitudes first, then reduce: verified against the
    # real pre-tanh argument to |diff| < 1e-5 when the terms share a sign.
    sum_abs = (ev_eff + pe_abs).mean().item()         # scalar
    # raw NLL behind the scaled pe
    pe_raw = pe_abs.mean().item() * state.pe_reference_scale

    loss = out["loss"].item()
    et_norm = out["E_next"].norm(dim=-1).mean().item()

    # Which term dominates the tanh argument? Compare the magnitudes that
    # actually enter it, per drive and then on average.
    dom = "ev" if ev_eff.mean().item() > pe_abs.mean().item() else "pe"
    rows.append((step, loss, et_norm, ev_abs.mean().item(), ev_eff.mean().item(),
                 pe_abs.mean().item(), sum_abs, pe_raw, dom))

    print(f"{step:>4} {loss:>9.4f} {et_norm:>8.4f} {ev_abs.mean().item():>9.5f} "
          f"{ev_eff.mean().item():>9.5f} {pe_abs.mean().item():>9.5f} "
          f"{sum_abs:>9.5f} {pe_raw:>9.4f} {dom:>5} {math.tanh(sum_abs):>9.5f}")

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

print()
ev_m = [r[3] for r in rows]      # raw (post-LayerNorm, pre-scale)
ev_e = [r[4] for r in rows]      # effective, what enters tanh
pe_m = [r[5] for r in rows]
print(f"event_proj RAW  |.| mean {sum(ev_m)/len(ev_m):.5f}  "
      f"min={min(ev_m):.5f} max={max(ev_m):.5f}")
print(f"event_proj EFF  |.| mean {sum(ev_e)/len(ev_e):.5f}  "
      f"min={min(ev_e):.5f} max={max(ev_e):.5f}   (x{state.event_drive_scale})")
print(f"pe_proj    |.| mean {sum(pe_m)/len(pe_m):.5f}  "
      f"min={min(pe_m):.5f} max={max(pe_m):.5f}")
ratio = (sum(pe_m)/len(pe_m)) / max(1e-9, sum(ev_e)/len(ev_e))
print(f"ratio pe/event(EFF) : {ratio:.3f}")
n_pe_dom = sum(1 for r in rows if r[8] == 'pe')
n_over = sum(1 for v in ev_e if v > 2.6467)
print(f"pe_proj dominated the tanh arg on {n_pe_dom}/{len(rows)} steps "
      f"({100.0*n_pe_dom/len(rows):.0f}%)")
print(f"event_proj(EFF) past atanh(0.99)=2.6467 on {n_over}/{len(rows)} steps")
print()
print("PINNING REFERENCE: |tanh(x)| > 0.9  <=>  |x| > atanh(0.9) = %.4f" % math.atanh(0.9))
print("                   |tanh(x)| > 0.99 <=>  |x| > atanh(0.99) = %.4f" % math.atanh(0.99))
print("                   |tanh(x)| > 0.999<=>  |x| > atanh(0.999) = %.4f" % math.atanh(0.999))
