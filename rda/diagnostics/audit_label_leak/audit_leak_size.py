"""Size the label leak and test the PE-on/off asymmetry. Read-only.

GPT.forward scores cross_entropy(logits[p], targets[p]) -- NO shift. So the
logits at position p are the distribution for token y[p] itself.

rda.py:901 builds pe_seq from targets[:, 1:] and rda.py:972 modulates
position p with E_next_seq[p-1], which is built from pe_seq[:, p-1], i.e. from
targets[:, p] -- token p ITSELF. So the state that modulates position p has
already seen the label position p is scored against.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

CKPT = os.path.join(get_base_dir(), "base_checkpoints", "b-stab-s42", "model_000600.pt")
LN2 = 0.6931471805599453


def say(s=""):
    print(s, flush=True)


def build(seed, device):
    compute_init(device, seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=768, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.eval()
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True), strict=True)
    return m, tok, V


def bpb(m, batches, tb, reset=False, drop_pe=False):
    carry, tn, tbb = None, 0.0, 0.0
    for x, y in batches:
        with torch.no_grad():
            if drop_pe:
                # Supply targets=None so the forward takes its own zero-PE
                # branch, then score the returned logits ourselves.
                lg = m(x)
                l = torch.nn.functional.cross_entropy(
                    lg.view(-1, lg.size(-1)), y.view(-1),
                    ignore_index=-1, reduction="none").view(y.shape)
                E = m.last_state
            else:
                l = m(x, y, loss_reduction="none", E_t_external=carry)
                E = m._last_E_next_seq[:, -1, :]
        l = l.view(-1).float()
        t = y.view(-1)
        nb = tb[torch.where(t >= 0, t, torch.zeros_like(t))]
        nb = torch.where(t >= 0, nb, torch.zeros_like(nb))
        tn += float((l * (nb > 0)).sum())
        tbb += float(nb.sum())
        carry = None if reset else E
    return tn / tbb / LN2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok, V = build(42, device)
    tb = get_token_bytes(device=device)
    ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    B = []
    for _ in range(24):
        x, y, _ = next(ld)
        B.append((x.to(device), y.to(device)))

    say("--- D. HOW BIG IS THE LEAK? ---")
    say("Compare scoring the SAME model with the PE pathway ON (targets given)")
    say("against PE OFF (targets=None, the forward's own zero-PE branch).")
    say("If PE-ON is much better and PE-ON is also cheating, the gap is the")
    say("size of the cheat.")
    on = bpb(m, B, tb)
    off = bpb(m, B, tb, drop_pe=True)
    say("  PE ON  (targets given) : bpb %.6f" % on)
    say("  PE OFF (targets=None)  : bpb %.6f" % off)
    say("  gap                    : %+.6f" % (on - off))
    say()
    say("  baseline A s42 1.301217 | B'-s42 run 1.226305 | B'-s42 PE-OFF %.6f" % off)
    say()

    # ---- E. does the leak reach the BASE model's own path? ---- #
    say("--- E. IS THE LEAK SPECIFIC TO FiLM? ---")
    say("modulate() is the only consumer of film_state. Zeroing the FiLM")
    say("projections should remove the label's path to the logits.")
    st = m.state
    h1 = st.film_scale.register_forward_hook(lambda a, b, c: torch.zeros_like(c))
    h2 = st.film_shift.register_forward_hook(lambda a, b, c: torch.zeros_like(c))
    nofilm = bpb(m, B, tb)
    h1.remove(); h2.remove()
    say("  FiLM zeroed, PE ON      : bpb %.6f  (delta vs PE ON %+.6f)" % (nofilm, nofilm - on))
    say("  => if FiLM zeroed lands near the baseline, the entire gain rode in")
    say("     on the FiLM path, which is also the label's only path in.")
    say()

    # ---- F. gradient reach: does d(loss at p)/d(target p) flow via FiLM? ---- #
    say("--- F. WHERE THE LABEL ENTERS: pe_seq dependence ---")
    x, y = B[0]
    with torch.no_grad():
        r = m(x, y)
        E_pe = r["E_next_seq"]
    # Recompute E with PE forced to zero, same inputs, via a hook on pe_proj.
    hz = st.pe_proj.register_forward_hook(lambda a, b, c: torch.zeros_like(c))
    with torch.no_grad():
        r2 = m(x, y)
    hz.remove()
    say("  max|E_next_seq(PE on) - E_next_seq(PE off)| = %.4f"
        % float((E_pe - r2["E_next_seq"]).abs().max()))
    say("  => PE drives the state. The state is the label's only route to the")
    say("     logits, since FiLM is the state's sole consumer.")


if __name__ == "__main__":
    main()
