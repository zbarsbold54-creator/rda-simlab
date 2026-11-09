"""VERIFY: is the console 'film s' the APPLIED scale, or the RAW projection?

I asserted in c5597df that console "film s" equals
    max|1 + _unit_rms(film_scale(E), 0.5)|
i.e. the applied scale. That claim needs checking, because

    row["mod_scale_absmax"] = float((1.0 + fs).abs().max())
    fs = cap["film_scale_out"]  = the film_scale MODULE OUTPUT

and rda.py applies _unit_rms to that module output INSIDE modulate(). If fs
is captured before that, the console value is max|1 + RAW|, which is a
different quantity from the applied gain.

Read-only. Builds a small model, runs a few steps of the same optimizer the
probe uses, and compares the two candidate definitions directly.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, _unit_rms

D_MODEL, N_LAYER, N_HEAD, T, V = 768, 8, 6, 1024, 32768


def say(s=""):
    print(s, flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=T, vocab_size=V, n_layer=N_LAYER,
                    n_head=N_HEAD, n_kv_head=N_HEAD, n_embd=D_MODEL,
                    window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    st = m.state
    m.train()

    # give the FiLM projections non-trivial weights, as after training
    with torch.no_grad():
        st.film_scale.weight.normal_(0.0, 0.02)
        st.film_scale.bias.normal_(0.0, 0.02)
        st.film_shift.weight.normal_(0.0, 0.02)
        st.film_shift.bias.normal_(0.0, 0.02)

    cap = {}
    h = st.film_scale.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("fs", out.detach()))
    h2 = st.film_shift.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("fh", out.detach()))

    opt = torch.optim.AdamW(
        list(st.film_scale.parameters()) + list(st.film_shift.parameters()),
        lr=5e-3, betas=(0.9, 0.95), eps=1e-10)

    x = torch.randint(4, V, (2, T), device=device)
    y = torch.roll(x, shifts=-1, dims=1).contiguous()

    say("step |  max|1+fs|  max|1+unit_rms(fs)|   rms(fs)   rms(unit_rms(fs))")
    say("-" * 74)
    for step in range(6):
        out = m(x, y)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        fs = cap["fs"]
        g = _unit_rms(fs, st.film_rms)
        say("%4d | %11.4f %19.4f %10.4f %18.4f"
            % (step, float((1.0 + fs).abs().max()),
               float((1.0 + g).abs().max()),
               float(fs.pow(2).mean().sqrt()),
               float(g.pow(2).mean().sqrt())))

    h.remove(); h2.remove()
    say("")
    say("film_rms constant in rda.py: %.3f" % st.film_rms)
    say("")
    say("INTERPRETATION")
    say("  max|1+fs|            is what the console prints as 'film s'")
    say("                        (probe_pe_film.py:280 -> :305-309).")
    say("  max|1+unit_rms(fs)|  is the APPLIED scale that multiplies the")
    say("                        residual stream in modulate().")
    say("  rms(unit_rms(fs))    should be exactly film_rms = 0.5 by")
    say("                        construction, at every step.")


if __name__ == "__main__":
    main()
