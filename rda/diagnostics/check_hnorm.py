"""
Sanity check the LayerNorm + 0.1 output scale in isolation, before spending
GPU time. Confirms:
  1. h_norm is genuinely scale-invariant (the whole point of step 1)
  2. the resulting event_term magnitude is bounded as predicted
  3. h_norm params are registered under state.* so coverage must catch them
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from nanochat.rda import RDAStateModule

d_model, n_drives = 768, 8
m = RDAStateModule(d_model, n_drives=n_drives)
m.eval()
W = m.event_proj.weight
print(f"event_proj.weight shape = {tuple(W.shape)}  std = {float(W.std()):.5f}")
print(f"event_drive_scale = {m.event_drive_scale}  (type {type(m.event_drive_scale).__name__})")
print()

torch.manual_seed(0)
print(f"{'hidden scale':>13} {'raw |event|':>12} {'normed |event|':>14} {'tanh(raw)':>10} {'tanh(normed)':>13}")
print("-" * 66)
for s in (0.1, 1.0, 10.0, 100.0, 1000.0):
    h = torch.randn(4, d_model) * s
    with torch.no_grad():
        raw = m.event_proj(h)
        normed = m.event_drive_scale * m.event_proj(m.h_norm(h))
    print(f"{s:>13.1f} {raw.abs().mean():>12.4f} {normed.abs().mean():>14.5f} "
          f"{math.tanh(raw.abs().mean().item()):>10.4f} "
          f"{math.tanh(normed.abs().mean().item()):>13.5f}")

print()
print("Scale invariance: normed |event| should be ~constant across the column above.")
print()
print("Worst case bound: for a normalized h (each component ~N(0,1) at worst,")
print("bounded by sqrt(d_model) by Cauchy-Schwarz),")
print(f"  |event_term| <= 0.1 * ||W_row||_2 * ||h_norm||_2")
wn = W.norm(dim=1)
print(f"  ||W_row||_2 max = {float(wn.max()):.4f}  ||h_norm||_2 ~ sqrt({d_model}) = {math.sqrt(d_model):.2f}")
print(f"  => |event_term| <~ 0.1 * {float(wn.max()):.4f} * {math.sqrt(d_model):.2f} "
      f"= {0.1*float(wn.max())*math.sqrt(d_model):.3f}")
print(f"  pinning threshold atanh(0.99) = {math.atanh(0.99):.4f}")
print()
print("Registered state.* parameters (these must all be in an optimizer group):")
for n, p in m.named_parameters():
    print(f"  {n:<24} {tuple(p.shape)}")
