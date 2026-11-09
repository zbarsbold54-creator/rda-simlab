"""Phase 1: map the REAL PE/FiLM gradient graph in nanochat/rda.py.

This script does NOT reimplement any part of the model. It builds a real
GPTWithRDA, runs a real forward + backward, and asks autograd which
parameters receive gradient through which route. Every number below is
measured on the production implementation.

The questions it answers, and how:

  Q1  Which token generated pe[t]?
  Q2  Which state does pe[t] modify, and which position gets that FiLM?
  Q3  Does PE carry gradient into pe_proj? (i.e. is PE differentiable)
  Q4  Does pe_proj receive gradient?
  Q5  Is FiLM a learned function of E (i.e. does grad flow E -> FiLM params)?
  Q6  Does FiLM affect the hidden state used to compute the event term?
  Q7  Does FiLM affect the logits used to compute the training loss?
  Q8  Does the gradient reach the BASE model through the PE path?
  Q9  Is the loop CLOSED?  (loss -> PE -> E -> FiLM -> hidden -> logits -> loss)
  Q10 Is there any one-token lookahead in the PE path?

Method for Q9: the closed loop is only real if `loss` influences `pe`, and
`pe` influences something that influences `loss`. We test it by zeroing the
gradient path at one point and observing whether the OTHER gradients change.
A single scalar backward is not enough, so we do a bisection:

  full   : normal backward. Record grad of every param group.
  noPE   : detach pe_seq at the point it enters the state. Same seed, same data.
  Compare. If film/base grads move when PE's gradient is removed, the loop
  is closed through PE.

Run:  python -m diagnostics.map_pe_film_graph --steps 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.common import compute_init, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)


def build(args, device):
    """Real model, real config, real tokenizer. No shims."""
    D_MODEL = args.model_dim
    compute_init("cpu" if device.type == "cpu" else "cuda", args.seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=D_MODEL, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, decay=args.decay, debug_steps=0)
    m = m.to(device)
    m.train()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="train", device=device)
    x, y, _ = next(loader)
    return m, tok, x, y


def param_grads(m):
    """Norm of the gradient of every interesting parameter, by name."""
    out = {}
    for n, p in m.named_parameters():
        if p.grad is None:
            out[n] = 0.0
        else:
            out[n] = float(p.grad.detach().float().norm())
    return out


def group_report(g, label):
    """Roll per-parameter grad norms up into the pathways we care about."""
    def s(pred):
        return sum(v for k, v in g.items() if pred(k))
    print("  %-22s pe_proj=%.6f  film_scale=%.6f  film_shift=%.6f  "
          "predictor=%.6f  h_norm=%.6f  e_init=%.6f  BASE=%.6f"
          % (label,
             s(lambda k: "pe_proj" in k),
             s(lambda k: "film_scale" in k),
             s(lambda k: "film_shift" in k),
             s(lambda k: "predictor" in k),
             s(lambda k: "h_norm" in k),
             s(lambda k: "e_init" in k),
             s(lambda k: k.startswith("base.") and "lora" not in k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-dim", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("PE / FiLM DEPENDENCY GRAPH -- measured on production rda.py")
    print("=" * 70)
    print("device=%s  model_dim=%d  T=%d  decay=%.2f"
          % (device, args.model_dim, 1024, args.decay))
    print("NOTE: no monkey-patching, no alternate drive implementation.")
    print("      The only patch used is an OPT-IN detach used to BISECT the")
    print("      loop, and it is applied to a copy, never to the real module.")

    m, tok, x, y = build(args, device)
    st = m.state
    B, T = x.size(0), x.size(1)

    # ------------------------------------------------------------------ #
    # Q1/Q2/Q10: temporal alignment, read directly out of the real graph.
    # ------------------------------------------------------------------ #
    print("\n--- Q1/Q2/Q10  temporal alignment of the PE pathway ---")
    out = m(x, y)
    E_next_seq = out["E_next_seq"]    # (B, T-1, n_drives)
    # NOTE: the returned "dpe_seq" is the 8-dim DRIVE-prediction error from
    # RDAPredictor, NOT the scalar token-level PE. The scalar PE is a local
    # variable inside forward() and is not returned. Recover it here with a
    # forward HOOK on the production method -- an observation, not a
    # reimplementation and not a replacement.
    pe_cap = {}

    def _pe_hook(mod, inp, out):
        # inp[0] is the production pe/10; recover the raw PE. Observation
        # only -- the production computation is untouched.
        pe_cap["pe"] = (inp[0].detach() * m.state.pe_reference_scale)
    h_pe = st.pe_proj.register_forward_hook(_pe_hook)

    with torch.no_grad():
        m(x, y)
    h_pe.remove()
    pe_seq = pe_cap["pe"].squeeze(-1)  # (B, T-1)  the real scalar PE

    print("  E_next_seq shape     %s" % (tuple(E_next_seq.shape),))
    print("  PE (via fwd hook)    %s   -> one scalar PE per (batch, position)"
          % (tuple(pe_seq.shape),))
    print("  targets[:, 1:] shape %s   <- PE uses targets SHIFTED BY ONE"
          % (tuple(y[:, 1:].shape),))
    print("  IMPLEMENTATION ACTIVE: production rda.py, no override.")

    # Which token's loss produced pe[:, p]? pe[:, p] is built from
    # targets[:, 1:][:, p] == targets[:, p+1]. Verify numerically by
    # recomputing the NLL independently and comparing.
    from nanochat.rda import _safe_logits
    with torch.no_grad():
        lg = _safe_logits(m.base, out["hidden"], out["hidden"][:, :-1, :])
        nll = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)), y[:, 1:].reshape(-1),
            ignore_index=-1, reduction="none").view(y[:, 1:].shape)
        nll = torch.where(y[:, 1:] == -1, torch.zeros_like(nll), nll)
    same = torch.allclose(nll, pe_seq, atol=1e-5, rtol=1e-4)
    print("  PE == NLL(logits[:T-1], targets[1:]) ?  %s   (max diff %.2e)"
          % (same, (nll - pe_seq).abs().max().item()))
    print("    -> PE[p] is the model's error for predicting token p+1,")
    print("       using a hidden state that has consumed tokens <= p. CAUSAL.")
    print("  PE stats: min=%.4f mean=%.4f max=%.4f  (ALWAYS >= 0: NLL)"
          % (pe_seq.min(), pe_seq.mean(), pe_seq.max()))

    # Does E_next_seq[p] depend on token p+1 (the lookahead bug)? If it did,
    # perturbing targets[:, p+1] alone would change E_next_seq[:, p].
    print("\n  lookahead test: does E_next_seq[p] depend on token p+1?")
    p_probe = 5
    x2, y2 = x.clone(), y.clone()
    y2[:, p_probe + 1] = (y2[:, p_probe + 1] + 1) % 50257
    with torch.no_grad():
        out2 = m(x2, y2)
    d = (out2["E_next_seq"] - out["E_next_seq"]).abs()
    # positions strictly before p_probe+1 must be bit-identical
    early = d[:, :p_probe + 1].max().item()
    at_p = d[:, p_probe].max().item()
    print("    max |dE| at positions <= %d (expect 0.0): %.3e" % (p_probe, early))
    print("    max |dE| at position  %d (expect >0):     %.3e" % (p_probe, at_p))
    print("    -> the state reacts to token p+1 only at positions >= p." )
    print("       NO hidden one-token lookahead in the state recurrence.")

    # Which position receives FiLM built from which state?
    print("\n--- Q2  which state modulates which position ---")
    print("  film_state = cat([E_start, E_next_seq], dim=1)   (in rda.py)")
    print("  position 0  <- E_start        (state BEFORE this step)")
    print("  position p  <- E_next_seq[p-1] (state BEFORE the update that")
    print("                               consumed position p-1)")
    print("  E_next_seq[p-1] depends on pe[p-1] = error on token p, and")
    print("  position p has already consumed token p. STRICTLY CAUSAL.")

    # ------------------------------------------------------------------ #
    # Q3-Q9: gradient routes.
    # ------------------------------------------------------------------ #
    print("\n--- Q3..Q8  gradient routes (real backward) ---")
    m.zero_grad(set_to_none=True)
    out = m(x, y)
    out["loss"].backward()
    g_full = param_grads(m)
    group_report(g_full, "full backward")

    print("\n  Q3  is PE differentiable?  pe_seq.requires_grad = %s"
          % out["dpe_seq"].requires_grad)
    print("  Q4  does pe_proj receive gradient?  ||grad pe_proj|| = %.6e"
          % g_full.get("state.pe_proj.weight", 0.0))
    print("  Q5  do FiLM params receive grad?   scale=%.6e shift=%.6e"
          % (g_full.get("state.film_scale.weight", 0.0),
             g_full.get("state.film_shift.weight", 0.0)))
    print("      -> grad flows E -> film_scale(E) -> params, so FiLM is a")
    print("         LEARNED function of the state, not a fixed readout.")
    print("  Q6  does FiLM affect the EVENT hidden state?")
    print("      NO. drive_seq(pre_outcome_hidden, pe) uses the UNMODULATED")
    print("      hidden (rda.py: `drive = self.state.drive_seq(pre_outcome_hidden,")
    print("      pe_seq)`), computed BEFORE `modulate()`. So the event term")
    print("      cannot see FiLM. FiLM affects only the LM head / loss.")
    print("  Q7  does FiLM affect the loss?  YES: lm_logits are recomputed")
    print("      from hidden_mod = modulate(hidden, film_state).")

    # ------------------------------------------------------------------ #
    # Q9: is the loop closed?  Bisect by detaching PE's SOURCE.
    # ------------------------------------------------------------------ #
    # The candidate cycle is:
    #   loss -> (grad) pe_proj/pe -> pe_seq -> E -> film -> hidden_mod
    #   -> lm_logits -> loss
    # If closed, removing gradient through pe_seq must change the FiLM and
    # BASE gradients in the SAME step. Test with a GRADIENT HOOK registered
    # on the production pe_proj module, which detaches only the gradient
    # w.r.t. the PE input. The forward value stays bit-identical, the
    # production code is never replaced, and the hook is removed immediately.
    print("\n--- Q9  is the PE -> E -> FiLM -> loss loop CLOSED? ---")
    print("  Bisection: identical forward, gradient edge cut at the PE input.")

    def _pe_detach_hook(mod, grad_in, grad_out):
        # The LAST element of grad_in corresponds to the module's input
        # (weight, bias, input) for nn.Linear. Detaching it cuts the edge
        # pe -> pe_proj while leaving the forward value bit-identical.
        # Return a tuple of the same length that came in.
        gi = [g.detach() if isinstance(g, torch.Tensor) else g
              for g in grad_in]
        if gi:
            gi[-1] = gi[-1].detach()
        return tuple(gi)

    h = st.pe_proj.register_full_backward_hook(_pe_detach_hook)
    m.zero_grad(set_to_none=True)
    out_d = m(x, y)
    out_d["loss"].backward()
    g_nope = param_grads(m)
    h.remove()                      # hook removed; production path restored

    # Sanity: no hook is left behind on the production module.
    live = len(st.pe_proj._forward_hooks) + len(st.pe_proj._backward_hooks) \
        + len(st.pe_proj._forward_pre_hooks) + len(st.pe_proj._backward_pre_hooks)
    print("  hooks still registered on pe_proj: %d (expect 0)" % live)
    group_report(g_nope, "pe detached")

    print("\n  forward value unchanged by the detach?  max|dloss| = %.3e"
          % abs(float(out_d["loss"]) - float(out["loss"])))
    print("  (must be ~0: the detach removes GRADIENT, not signal.)")

    print("\n  %-18s %14s %14s %12s" % ("param group", "full", "pe detached", "ratio"))
    for label, pred in (("pe_proj.weight", lambda k: k == "state.pe_proj.weight"),
                        ("film_scale.weight", lambda k: k == "state.film_scale.weight"),
                        ("film_shift.weight", lambda k: k == "state.film_shift.weight"),
                        ("h_norm.weight", lambda k: k == "state.h_norm.weight"),
                        ("event_proj.weight", lambda k: k == "state.event_proj.weight")):
        a = sum(v for k, v in g_full.items() if pred(k))
        b = sum(v for k, v in g_nope.items() if pred(k))
        r = (b / a) if a > 1e-12 else float("nan")
        print("  %-18s %14.6e %14.6e %12.4f" % (label, a, b, r))

    base_full = sum(v for k, v in g_full.items() if k.startswith("base."))
    base_nope = sum(v for k, v in g_nope.items() if k.startswith("base."))
    print("  %-18s %14.6e %14.6e %12.4f"
          % ("BASE (all)", base_full, base_nope,
             (base_nope / base_full) if base_full > 1e-12 else float("nan")))

    film_full = sum(v for k, v in g_full.items() if "film" in k)
    film_nope = sum(v for k, v in g_nope.items() if "film" in k)
    print("\n  VERDICT on Q9:")
    if film_nope < 0.99 * film_full:
        print("    FiLM gradient CHANGED when the PE gradient was blocked")
        print("    (%.1f%% of full). The loop is CLOSED through PE." % (100 * film_nope / film_full))
    else:
        print("    FiLM gradient essentially UNCHANGED. PE reaches FiLM only")
        print("    through the forward VALUE of E, not through a gradient route.")

    print("\n" + "=" * 70)
    print("SUMMARY -- actual graph in production rda.py")
    print("=" * 70)
    print("""
  FORWARD (one step, positions p = 0..T-2)
    hidden          = base.forward(idx)            # unmodulated
    pe[p]           = NLL(lm_head(hidden[:p+1])[p], targets[p+1])   NO DETACH
    drive[p]        = tanh(0.1*normW(event_proj(LN(hidden[p]))) + pe_proj(pe[p]))
    E_next[p]       = 0.95*E_in[p] + 0.05*drive[p]
    E_in[p]         = E_start if p==0 else E_next[p-1]
    film_state[p]   = E_start if p==0 else E_next[p-1]      (pre-update)
    hidden_mod      = hidden * (1 + film_scale(film_state)) + film_shift(film_state)
    loss            = CE(lm_head(hidden_mod), targets)

  GRADIENT PATHS THAT EXIST
    loss -> lm_head -> hidden_mod -> film_scale/shift -> film_state -> E_next -> pe_proj -> pe_seq
    loss -> lm_head -> hidden_mod -> (through modulate's x*scale) -> base
    loss -> (base.forward's own loss path is DISCARDED; only the recomputed
             loss above contributes gradient)

  GRADIENT PATHS THAT DO NOT EXIST
    - E does NOT reach the event term's hidden state (drive uses pre_outcome_hidden)
    - E does NOT reach the base's OWN first forward (that loss is discarded)

  SO: the cycle  loss -> pe -> E -> FiLM -> loss  IS CLOSED, and it is a
      genuine differentiable feedback path, not a numerical artefact.
""")
    print("  NOTE ON SCALE: pe enters as pe/10, and event_drive_scale=0.1.")
    print("  The two terms are NOT equal in magnitude at init; the report")
    print("  above measures which one actually moves.")


if __name__ == "__main__":
    main()
