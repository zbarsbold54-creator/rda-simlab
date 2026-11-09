"""Quantify the training-time FiLM causality issue. MEASUREMENT ONLY.

Claim under test: modulate() runs once per forward pass, after the full
per-position loop, using the loop's final rebound E_t -- so hidden states
at position 0 are modulated by a state derived from the LAST token.

Captures the state modulate() ACTUALLY receives via a forward hook.
An earlier attempt reconstructed it with init_state() -- a fixed zero
vector -- which made every measurement identically zero and proved nothing.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import autodetect_device_type
from nanochat.rda import GPTWithRDA

device = torch.device(autodetect_device_type())
torch.manual_seed(1337)
tok = get_tokenizer()
V = tok.get_vocab_size()

cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                n_kv_head=6, n_embd=768, window_pattern="L")
base = GPT(cfg); base.init_weights()
model = GPTWithRDA(base, debug_steps=0).to(device)
model.eval()

with torch.no_grad():                      # else modulate() is the identity
    for lay in (model.state.film_scale, model.state.film_shift):
        lay.weight.fill_(0.05)
        lay.bias.fill_(0.05)

loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tok, 4, 1024, split="train", device=device)
x, y, _ = next(loader)
T = x.size(1)


def run(xx):
    """Full forward, returning (FiLM-conditioned logits, E that modulate got).

    modulate() is a METHOD, not a submodule, so hook the parent module's
    forward and read the E_t argument off the call. We also recompute
    logits ourselves because forward(targets=...) returns a dict without
    "logits" (that field is only populated when targets is None).
    """
    cap = {}
    orig = model.state.modulate

    def spy(x_, E_):
        cap["E"] = E_.detach().clone()
        return orig(x_, E_)

    model.state.modulate = spy
    try:
        with torch.no_grad():
            out = model(xx, xx)
            _, hid = model.base.forward(xx, xx, return_hidden=True)
            hm = orig(hid, cap["E"])
            lg = model.base.lm_head(hm)[..., :V].float()
            lg = 15 * torch.tanh(lg / 15)
    finally:
        model.state.modulate = orig
    return lg, cap["E"], out["E_next_seq"][0]


lg1, E_used, Eseq = run(x)
LAST = Eseq.shape[0] - 1

print(f"T = {T};  update() loop visits positions 0..{LAST}")
print()
print("WHAT modulate() ACTUALLY RECEIVED (row 0):")
print("  " + " ".join(f"{v:+.4f}" for v in E_used[0]))
print()
print("PER-POSITION E_next_seq (row 0), vs the state modulate() used:")
for i in (0, 1, 2, 10, 100, 500, LAST):
    if i < Eseq.shape[0]:
        l2 = float((E_used[0] - Eseq[i]).norm())
        print(f"  pos {i:>4}: [{' '.join(f'{v:+.3f}' for v in Eseq[i])}]"
              f"   L2 vs E_used = {l2:.6f}")
print()
print(f"  E_used == E_next_seq[{LAST}] ? "
      f"{torch.allclose(E_used[0], Eseq[LAST], atol=1e-6)}")
print()
print("LOOKAHEAD: every position p is modulated by the state produced at")
print(f"position {LAST}, so p is influenced by tokens p+1..{T-1}.")
print(f"{'pos p':>7} {'lookahead':>10}")
for p in (0, 1, 2, 10, 100, 500, T - 1):
    print(f"{p:>7} {LAST - p:>10}")
print()
print("EMPIRICAL: perturb ONLY the last token, measure dlogit per position.")
print("Base GPT is causal, so any delta is attributable to RDA FiLM.")
x2 = x.clone()
x2[:, -1] = (x2[:, -1] + 1234) % V
lg2, E_used2, _ = run(x2)

d = (lg2 - lg1).abs()
print(f"{'pos p':>7} {'max|dlogit|':>13} {'mean|dlogit|':>14}")
for p in (0, 1, 2, 10, 100, 500, T - 1):
    print(f"{p:>7} {float(d[:, p, :].max()):>13.6e} "
          f"{float(d[:, p, :].mean()):>14.6e}")
print()
de = (E_used2 - E_used).abs()
print(f"modulating state changed when the last token changed? "
      f"{not torch.allclose(E_used, E_used2, atol=1e-6)}"
      f"   (max|dE| = {float(de.max()):.6e})")
print(f"dlogit at position 0            : {float(d[:, 0, :].max()):.6e}")
print()
print("Nonzero at p=0 => the last token reached position 0's logits via FiLM.")
print("Zero everywhere would mean the leak is absent OR FiLM is inert.")
