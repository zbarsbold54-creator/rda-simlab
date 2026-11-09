"""Phase 2/3: instrument the PE/FiLM runaway on PRODUCTION code.

Design rules (enforced, after the earlier shadowing incident):

  1. The normal measurement path calls the real RDAStateModule /
     GPTWithRDA methods. Nothing is reimplemented.
  2. Every intervention is a forward hook (ablation) or a requires_grad
     flag (freeze). Hooks are registered and removed explicitly.
  3. Every run PRINTS which implementation is active and whether any hook
     is installed, so a stale/shadowed run cannot masquerade as a result.
  4. Observation uses forward hooks to CAPTURE intermediates; a capture
     hook returns its input unchanged, so it cannot alter the trajectory.

Ablations (opt-in, off by default):
  --no-film      C2  FiLM disabled: hidden_mod == hidden (identity modulation)
  --freeze-base  C3  base LM frozen; state/FiLM still train
  --freeze-pe    C4  pe_proj.weight and .bias frozen
  --detach-pe    C5  PE keeps its numerical value; only the gradient
                     edge pe -> pe_proj is cut (gradient hook on the input)
  --film-cap F   C6  FiLM scale bounded by a normalized reparameterization

Run:  python -m diagnostics.probe_pe_film --steps 250 --out logs/pe_c1.json
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS, _unit_rms
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

# ---------------------------------------------------------------------
# Per-position threat-drive distribution capture (DIAGNOSTICS ONLY).
#
# WHY THIS EXISTS. E13-fixed-3000 left one question open: is the threat
# drive BIMODAL (two separated modes) or UNIMODAL near -1 with a thin
# positive tail? A mean of -0.97 with an absmax of +0.99 is consistent
# with both, and per-step summary scalars cannot tell them apart. Only the
# distribution can.
#
# Quantiles and tail fractions are recorded every step. A full histogram is
# recorded only at DIST_STEPS, because 100 bins x 3000 steps is large and
# the distribution's SHAPE changes slowly; the quantiles carry the
# per-step detail.
#
# The histogram range [-1.2, 1.2] deliberately EXCEEDS the tanh range
# (-1, 1) so that mass piling up against a bound appears inside the bins
# rather than being silently clipped. Under/overflow counts are recorded
# separately for the same reason.
#
# Written to its own file (--dist-out) so the existing per-step records
# are byte-for-byte unchanged.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ALL-8-DIMENSION PRE-ACTIVATION LOGGING (DIAGNOSTICS ONLY).
#
# Same decomposition as the threat-only block above, applied to every
# dimension, plus three additions that the threat-only version could not
# supply:
#   * variance SHARES rather than only correlations (a correlation says
#     "moves together", a variance share says "who owns the variation");
#   * dL/d(drive_d) per position, which is the quantity that actually
#     tells you whether a dimension can still be trained BY the loss, as
#     opposed to merely still varying;
#   * pe_proj.weight[d,0] with its zero-crossing step.
#
# SIGNED THROUGHOUT. Every one of these is a signed quantity and is stored
# signed. The recurring error in this investigation has been reading an
# unsigned aggregate as though it had a sign, so the all-8 file is signed
# by construction and the identity check below is the guard.
#
# Written to its own file (--preact-all-out). The threat-only records are
# untouched.
# ---------------------------------------------------------------------
ALL_ROWS = []
ALL_INFO = {}
# ---------------------------------------------------------------------
# FILM COLUMN-WEIGHT / E-CONTRIBUTION LOGGING (DIAGNOSTICS ONLY).
#
# QUESTION. Is the threat channel acting as a CONSTANT OFFSET into FiLM?
# FiLM is the ONLY path from the state to the loss (rda.py:1079), so if
# threat contributes a large DC term and almost no variation, it is
# acting as a bias on the whole network rather than as a signal.
#
# The decomposition is exact and linear:
#     film_scale(E)[:, c] = sum_d film_scale.weight[c, d] * E[:, d]
# so the contribution of dimension d to output channel c is
#     W[c, d] * E[:, d]
# and because that is a product of a constant with a scalar per position,
# its across-position MEAN is the offset part and its across-position STD
# is the varying part. offset/varying = |mean|/std is therefore a direct,
# assumption-free measure of "constant vs signal" for that channel's share
# of the FiLM output. The ratio is reported SIGNED (mean is signed) and
# the abs value is given too, since a negative offset is still an offset.
#
# COLUMN, not row: film_scale is Linear(n_drives, d_model), so
# weight is (d_model, n_drives) and the column weight[:, d] is the vector
# that reads E_d. Using weight[d, :] would read the wrong thing entirely.
#
# Written to --film-out. No existing output changes.
# ---------------------------------------------------------------------
FILM_ROWS = []
# Reconstruction residuals: rebuild FiLM from sum-over-dims + bias and
# compare to the module output AND to the captured forward output.
RECON = []
RECON_TOL = 1e-4
FILM_STEPS = (0, 250, 500, 1000, 1500, 2000, 2500, 2999)
# Restored: the threat-drive distribution accumulators. These were removed
# by an over-broad edit anchor and the smoke test failed with
# NameError: DIST_ROWS. Declared here, next to the other accumulators.
DIST = {}
DIST_ROWS = []
PREACT_ALL_STEPS = (0, 500, 1000, 1500, 2000, 2999)
# Identity tolerance required by the task. If the decomposition fails to
# reconcile to this, the run is reporting an arithmetic error somewhere
# upstream and every downstream number is void.
IDENTITY_TOL = 1e-6
IDENTITY_WORST = {"residual": 0.0, "step": None, "dim": None}
# Set by install_drive_retain_grad(); the wrapped tensor is needed for
# dL/d(drive) which cannot be recomputed after the graph is freed.
LAST_DRIVE = {}

# ---------------------------------------------------------------------
# PER-DIMENSION ABLATION AT END OF RUN (DIAGNOSTICS ONLY).
#
# QUESTION. The state E is 8 numbers per position, and the previous entries
# established that threat is offset-dominated (|E_mean|/E_std = 436) with a
# near-frozen E_std. OFFSET-dominated is a statement about the STATE; it
# does not say whether the LOSS cares. A channel can be constant and still
# be doing real work, or constant and inert. Only ablation distinguishes
# them.
#
# Two ablations per dimension, because they remove different things:
#   (a) MEAN-ABLATE: E_d <- its across-position mean, per sequence.
#       Removes POSITION VARIATION, keeps the offset. Isolates "does the
#       variation matter".
#   (b) ZERO-ABLATE:  E_d <- 0.
#       Removes BOTH signal and offset. Isolates "does this channel matter
#       at all".
# The pair is what makes the interpretation possible:
#   (a) ~ 0 and (b) large  => the offset matters, the variation is unused
#   (a) ~ 0 and (b) ~ 0    => the channel is inert
#   (a) large              => the variation matters
#
# A baseline REPEAT is included as the noise floor, because the run is not
# bit-deterministic (GPU nondeterminism in the backward pass, documented in
# E13-fixed-3000) and without it "within noise" would have no definition.
#
# Applied by WRAPPING modulate, and restored in a finally block, so the
# model/gain/drive code is untouched and the wrapper cannot outlive the
# ablation even on an exception.
# ---------------------------------------------------------------------
ABL = {"enabled": False, "mode": None, "dims": None, "rows": []}


def set_ablation(mode, dims):
    ABL["enabled"] = True
    ABL["mode"] = mode
    ABL["dims"] = tuple(dims) if dims is not None else None

# Which modules consume the drive, and which consume the E state.
#
# Read off nanochat/rda.py rather than from memory, with line numbers, so
# the claim can be checked. Two distinct consumption paths exist and they
# are not equally important:
#
#   VIA THE DRIVE (directly, and via the E it produces):
#     rda.py:951  RDAStateModule.update_seq(E_start, drive)
#                  -- the recurrence. This is the ONLY path by which the
#                  drive reaches the state, so dL/d(drive) flows through
#                  here and through nothing else.
#     rda.py:1084 RDAStateModule.modulate(hidden, film_state)
#                  -- FiLM. Consumes the E state, NOT the drive directly.
#     rda.py:501-502  film_scale(E_t), film_shift(E_t)
#                  -- the two projections that turn E into the applied
#                     scale and shift.
#     rda.py:954  predictor.predict_seq(E_in, pre_outcome_hidden)
#                  -- the state predictor, consumes E.
#     rda.py:544,561  predictor.state_proj (Linear(8,8))
#                  -- consumes E.
#     rda.py:957  drive_prediction_error(E_next_seq, E_hat_seq)
#                  -- the dpe readout, consumes E (post-update).
#
# The FiLM path is the one that can feed the drive BACK into the model:
# modulate changes `hidden`, which changes the next step's event_proj
# input. That feedback loop exists and is why the E path matters, but the
# drive's own gradient path is only the recurrence + predictor + dpe.
RDA_CONSUMERS = {
    "drive_direct": [
        "RDAStateModule.update_seq (rda.py:951) -- the recurrence; the "
        "only direct path from drive into E",
    ],
    "E_state": [
        "RDAStateModule.modulate (rda.py:1084) via film_scale/film_shift "
        "(rda.py:501-502) -- FiLM on the residual stream",
        "RDAStatePredictor.predict_seq (rda.py:954) -> state_proj "
        "(nn.Linear(8,8), rda.py:544)",
        "drive_prediction_error (rda.py:957) -- dpe readout on "
        "E_next_seq",
    ],
    "feedback_note": "modulate() changes `hidden`, which is the input to "
                     "the next step's event_proj, so the E->FiLM->hidden "
                     "loop can in principle change the drive. Whether it "
                     "does is an inference, not measured here.",
}


def install_drive_retain_grad(st):
    """Wrap st.drive_seq so the drive tensor carries .grad after backward.

    WHY A WRAPPER. drive_seq is a METHOD, not a module, so no forward hook
    can see its output. The tensor is also created inside a graph that is
    freed the moment backward() returns, so it cannot be re-derived
    afterwards under no_grad -- a recomputation would be a fresh leaf with
    no history and therefore no .grad at all.

    retain_grad() is a pure bookkeeping call: it changes no value and no
    gradient, it only tells autograd to keep a reference so the tensor's
    .grad field gets populated. The returned tensor is the SAME tensor, so
    the forward pass arithmetic is bit-for-bit what it would have been.
    The alternative -- editing nanochat/rda.py to call retain_grad -- would
    be a drive-code change and is explicitly out of scope.
    """
    orig = st.drive_seq

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        if out.requires_grad:
            out.retain_grad()
        LAST_DRIVE["t"] = out
        return out

    st.drive_seq = wrapped
    st._probe_orig_drive_seq = orig
    return orig
# ---------------------------------------------------------------------
# THREAT PRE-ACTIVATION / GRADIENT / INFORMATION LOGGING (DIAGNOSTICS ONLY).
#
# WHY. E13-fixed-3000-dist established that the threat drive is UNIMODAL,
# entirely negative, and pressed against the LOWER bound: the fraction of
# positions with drive > +0.9 is exactly 0.0 at every step, and the whole
# distribution lies below zero from step 500 on. That says WHERE the drive
# ends up but not WHY. Three quantities are needed to tell the candidate
# mechanisms apart, and all three are logged here:
#
#   (1) the PRE-TANH argument, decomposed. If |arg| is large, tanh is
#       saturated and its derivative is ~0; the question is then which
#       additive term is responsible.
#   (2) the GRADIENT on pe_proj.bias[threat] and .weight[threat,:]. If the
#       bias is being pushed negative by a consistently-signed gradient
#       while its own contribution to the argument is already dominant,
#       the bias is a ratchet, not a response to input.
#   (3) the CORRELATION between the drive and each raw input across
#       positions. If the drive stops tracking its inputs while still
#       moving, it has become input-independent -- a constant, not a
#       function.
#
# Written to --preact-out, separate from every other output.
# ---------------------------------------------------------------------
PREACT_ROWS = []
GRAD_ROWS = []
INFO_CORR = {}
PREACT_STEPS = (0, 500, 1000, 1500, 2999)
GRAD_EVERY = True
QLEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
QNAME = ("p1", "p5", "p25", "p50", "p75", "p95", "p99")
# Range for the pre-tanh histogram. The pre-tanh argument is UNBOUNDED, so
# this is a display window only; under/overflow are counted so a reader can
# tell when mass has left the window rather than assuming it is small.
PRE_BINS = 100
PRE_LO = -6.0
PRE_HI = 2.0
DIST_STEPS = (0, 500, 1000, 1500, 2000, 2500, 2999)
DIST_BINS = 100
DIST_LO = -1.2
DIST_HI = 1.2

D_MODEL = 768
NAMES = list(DRIVE_DIMS)


def gn(t):
    return 0.0 if t is None else float(t.detach().float().norm())


def stats(t):
    t = t.detach().float()
    return (float(t.min()), float(t.max()), float(t.mean()),
            float(t.pow(2).mean().sqrt()))


def build(seed, device):
    compute_init("cpu" if device.type == "cpu" else "cuda", seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=D_MODEL, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.train()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="train", device=device)
    return m, tok, loader


def install_hooks(m, a):
    """Register capture / ablation hooks. Returns (handles, capture dict).

    Capture hooks RETURN THEIR INPUT UNCHANGED, so they are pure observers.
    Ablation hooks are only installed when explicitly requested.
    """
    st = m.state
    cap = {}
    H = []

    def cap_mod(name, key, xform=None):
        def f(mod, inp, out):
            v = out if xform is None else xform(out)
            cap[key] = v.detach()
            return out          # UNCHANGED -- observation cannot alter the run
        H.append(getattr(st, name).register_forward_hook(f))

    # ---- observations (always on, never alter the trajectory) ----
    cap_mod("pe_proj", "pe_proj_out")            # (B,T,8)  8-dim PE term
    cap_mod("film_scale", "film_scale_out")      # (B,T,d)  the ADDED scale
    cap_mod("film_shift", "film_shift_out")      # (B,T,d)  the added shift

    # The E tensor FiLM ACTUALLY receives.
    #
    # This must not be inferred from the probe's own E_seq. FiLM is called as
    # modulate(hidden, film_state) (rda.py:1084) and film_state is
    #     cat([E_start.unsqueeze(1), E_next_seq], dim=1)   (rda.py:1022)
    # i.e. (B, T, 8) with the state at position p being the state AFTER
    # consuming position p-1. The probe's E_seq is E_next_seq, which is
    # (B, T-1, 8) and is SHIFTED BY ONE relative to what FiLM sees. Using it
    # here would measure the wrong tensor by a one-position offset.
    #
    # modulate is a METHOD, not a submodule, so it has no
    # register_forward_hook (verified: nn.Linear.forward has no such
    # attribute). The method is therefore WRAPPAPPED on the instance, the
    # same trick used for drive_seq's retain_grad. The wrapper records the
    # input and returns the original output UNCHANGED, so the forward pass
    # is bit-for-bit what it would have been.
    _orig_modulate = st.modulate

    def _modulate_capturing(x, E_t, *a2, **kw2):
        # The FiLM INPUT, and a FORWARD-TIME snapshot of the weights the
        # module is about to use.
        #
        # WHY THE CLONES. The reconstruction check runs AFTER backward() and
        # AFTER opt.step() would have run, so `st.film_scale.weight` at check
        # time may no longer be the tensor the forward pass used. That is
        # hypothesis (A). Snapshotting here removes the question by
        # construction: if the residual survives against these clones, (A) is
        # ruled out rather than merely suspected.
        #
        # The clones are detached and are never fed back into the graph, so
        # they cannot alter the trajectory.
        cap["film_E_in"] = E_t.detach()
        cap["film_W_scale"] = st.film_scale.weight.detach().clone()
        cap["film_b_scale"] = st.film_scale.bias.detach().clone()
        cap["film_W_shift"] = st.film_shift.weight.detach().clone()
        cap["film_b_shift"] = st.film_shift.bias.detach().clone()

        # --- ablation of the FiLM input, applied ONLY when armed ---
        # Armed for the end-of-run ablation and disarmed immediately after,
        # so training is never affected. The mean is taken per (batch, dim)
        # over POSITIONS, which is the "per-sequence mean" the ablation needs;
        # the recorded value is that same per-sequence mean.
        _E_use = E_t
        if ABL["enabled"]:
            with torch.no_grad():
                _E_use = E_t.detach().clone()
                if ABL["mode"] == "zero":
                    _d = ABL["dims"]
                    if _d is None:
                        _E_use.zero_()
                    else:
                        for _i in _d:
                            _E_use[:, :, _i] = 0.0
                elif ABL["mode"] == "mean":
                    _d = ABL["dims"]
                    _mu = E_t.detach().mean(dim=1, keepdim=True)   # (B,1,8)
                    if _d is None:
                        _E_use = _mu.expand_as(_E_use).contiguous()
                    else:
                        for _i in _d:
                            _E_use[:, :, _i] = _mu[:, :, _i]
                if not ABL.get("recorded"):
                    ABL["recorded"] = True
                    ABL["state_used"] = {
                        "mode": ABL["mode"],
                        "dims": list(ABL["dims"]) if ABL["dims"] else "all",
                        "E_shape": list(E_t.shape),
                        "per_seq_mean": [[float(v) for v in row]
                                         for row in E_t.detach().mean(dim=1)],
                        "per_seq_absmax": [[float(v) for v in row]
                                           for row in E_t.detach().abs().max(
                                               dim=1).values],
                    }
                # Per-batch values for EVERY held-out batch, not just the
                # first. The single-batch snapshot above is what the report
                # quotes for threat/novelty, and one batch is not enough to
                # check it against: the batches are different data.
                _rec = ABL.setdefault("per_batch_state", [])
                if len(_rec) < 64:
                    _rec.append({
                        "per_seq_mean": [[float(v) for v in row]
                                         for row in E_t.detach().mean(dim=1)],
                        "per_seq_absmax": [[float(v) for v in row]
                                           for row in E_t.detach().abs().max(
                                               dim=1).values],
                    })

        # Environment snapshot taken INSIDE the forward call, so autocast or a
        # precision context set by the caller would be visible rather than
        # assumed absent.
        cap["film_env"] = {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "autocast_enabled": bool(torch.is_autocast_enabled()),
            "autocast_dtype": (str(torch.get_autocast_dtype("cuda"))
                               if hasattr(torch, "get_autocast_dtype") else None),
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            "capability": (list(torch.cuda.get_device_capability(0))
                           if torch.cuda.is_available() else None),
            "E_dtype": str(E_t.dtype),
            "W_dtype": str(st.film_scale.weight.dtype),
        }
        # PASS THE ABLATED TENSOR. `_E_use` is `E_t` unchanged unless an
        # ablation is armed; when armed it is the modified clone. Returning
        # `_orig_modulate(x, E_t, ...)` here instead would compute the
        # ablation and then throw it away, making every delta exactly 0.0 and
        # every dimension falsely reading as inert. When disarmed `_E_use is
        # E_t`, so the training forward is bit-for-bit unchanged.
        return _orig_modulate(x, _E_use, *a2, **kw2)

    st.modulate = _modulate_capturing
    st._probe_orig_modulate = _orig_modulate
    # the raw scalar PE entering pe_proj is its INPUT
    H.append(st.pe_proj.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("pe_raw",
                                              (inp[0].detach() *
                                               st.pe_reference_scale))))

    # ---- ablations (opt-in only) ----
    if a.no_film:
        # C2: FiLM has NO effect on the residual stream. The state is still
        # computed, still recurs, and is still measured -- only the two
        # modulation branches are zeroed, which makes modulate() the exact
        # identity at every step rather than only at init.
        H.append(st.film_scale.register_forward_hook(
            lambda mod, inp, out: torch.zeros_like(out)))
        H.append(st.film_shift.register_forward_hook(
            lambda mod, inp, out: torch.zeros_like(out)))

    if a.detach_pe:
        def cut(mod, gi, go):
            gi = [g.detach() if isinstance(g, torch.Tensor) else g for g in gi]
            if gi:
                gi[-1] = gi[-1].detach()
            return tuple(gi)
        H.append(st.pe_proj.register_full_backward_hook(cut))

    if a.film_norm > 0:
        # C6: bound the FiLM GAIN by normalizing each per-position modulation
        # vector to a fixed RMS, so only its DIRECTION is learned. Same
        # philosophy as the event-projection fix: a reparameterization that is
        # scale-invariant, not a clamp -- no dead zone, no gradient
        # flattening, sign and relative pattern fully preserved. At init the
        # film weights are exactly 0, so 0/eps = 0 and modulate() is still the
        # exact identity.
        c = a.film_norm
        def gain(mod, inp, out):
            return out / out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * c
        H.append(st.film_scale.register_forward_hook(gain))
        H.append(st.film_shift.register_forward_hook(gain))

    return [h for h in H if h is not None], cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-film", action="store_true", help="C2")
    ap.add_argument("--freeze-base", action="store_true", help="C3")
    ap.add_argument("--freeze-pe", action="store_true", help="C4")
    ap.add_argument("--detach-pe", action="store_true", help="C5")
    ap.add_argument("--film-norm", type=float, default=0.0,
                    help="C6: bound FiLM gain to this RMS (0 = off)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--log", default=None,
                    help="also write every console line here as UTF-8")
    ap.add_argument("--abort-steps", type=int, default=0,
                    help="abort if max|E| exceeds --abort-bound for this many "
                         "consecutive steps (same rule as base_train.py; 0 = never)")
    ap.add_argument("--abort-bound", type=float, default=0.98,
                    help="the |E| threshold for the abort rule. base_train.py "
                         "uses 0.98; 0.999 is used to run past the crossing so "
                         "the trajectory AFTER saturation can be observed. "
                         "Raising this changes NOTHING about the model -- the "
                         "abort is a post-hoc check on an already-computed row.")
    ap.add_argument("--dist-out", default=None,
                    help="write the per-position threat-drive distribution "
                         "(quantiles every step, histograms at selected steps) "
                         "here as JSON. Kept separate from --out so the "
                         "existing per-step records are unchanged.")
    ap.add_argument("--preact-out", default=None,
                    help="write the threat PRE-TANH decomposition, the "
                         "pe_proj gradients, and the input-correlation check "
                         "here as JSON. Separate from --out and --dist-out.")
    ap.add_argument("--preact-all-out", default=None,
                    help="write the SAME decomposition for ALL 8 dimensions, "
                         "signed, plus variance shares and dL/d(drive) "
                         "(via retain_grad). Separate file; existing outputs "
                         "are unchanged.")
    ap.add_argument("--film-out", default=None,
                    help="write the FiLM column weights and each channel's "
                         "offset/varying contribution to the FiLM output. "
                         "Separate file; existing outputs are unchanged.")
    ap.add_argument("--ablate", action="store_true",
                    help="after the final training step, measure the loss "
                         "under per-dimension FiLM-input ablations (mean and "
                         "zero) plus a baseline repeat as the noise floor. "
                         "No optimizer step; weights frozen.")
    ap.add_argument("--ablate-batches", type=int, default=8,
                    help="held-out batches used for every ablation condition "
                         "(same batches for all conditions)")
    ap.add_argument("--ablate-out", default=None,
                    help="write the end-of-run per-dimension ablation results "
                         "here as JSON. Defaults to <out>_ablation.json when "
                         "--ablate is given and this is omitted.")
    a = ap.parse_args()
    device = torch.device(a.device)

    # LOG HYGIENE. PowerShell's `>` writes UTF-16LE, and a naive UTF-8 read
    # of the result returns NOTHING -- silently, with no error. That cost a
    # full extraction attempt on logs/pe_1000.txt before it was noticed, and
    # it is the reason the per-step series for the earlier E13 runs had to
    # be recovered by decoding UTF-16 by hand. This tee writes UTF-8
    # directly so nothing downstream has to guess the encoding.
    _log = open(a.log, "w", encoding="utf-8", newline="\n") if a.log else None
    _real_print = print

    def say(line=""):
        _real_print(line, flush=True)
        if _log is not None:
            _log.write(line + "\n")
            _log.flush()

    say("=" * 78)
    say("PE / FiLM RUNAWAY PROBE  --  label=%s" % a.label)
    say("=" * 78)
    say("  json : %s" % a.out)
    say("  log  : %s (utf-8)" % a.log)
    abl = [n for n, f in (("C2 no-film", a.no_film), ("C3 freeze-base", a.freeze_base),
                         ("C4 freeze-pe", a.freeze_pe), ("C5 detach-pe", a.detach_pe),
                         ("C6 film-norm %g" % a.film_norm, a.film_norm > 0)) if f]
    say("device=%s steps=%d seed=%d" % (a.device, a.steps, a.seed))
    say("ACTIVE IMPLEMENTATION: production nanochat/rda.py")
    say("  drive_seq overridden?      NO (never overridden in this script)")
    say("  prediction_error override? NO")
    say("  ablations active:          %s" % (", ".join(abl) if abl else "NONE (reference)"))
    say("  observation hooks are pure: they return their input unchanged.")
    say()

    m, tok, loader = build(a.seed, device)
    st = m.state

    if a.freeze_base:
        for p in m.base.parameters():
            p.requires_grad_(False)
    if a.freeze_pe:
        st.pe_proj.weight.requires_grad_(False)
        st.pe_proj.bias.requires_grad_(False)

    handles, cap = install_hooks(m, a)
    live = (len(st.pe_proj._forward_hooks) + len(st.pe_proj._backward_hooks) +
            len(st.film_scale._forward_hooks) + len(st.film_shift._forward_hooks))
    say("hooks installed (observation + ablation): %d" % live)

    # retain_grad on the drive tensor, needed for dL/d(drive_d) in the
    # all-8 diagnostics. Installed unconditionally because it is a pure
    # bookkeeping call -- it changes no value and no gradient -- and having
    # it always on means the flag cannot change what a run measures. It is
    # reported so a reader knows the graph retained one extra tensor.
    _orig_drive_seq = install_drive_retain_grad(st)
    if a.preact_all_out:
        say("drive retain_grad wrapper installed (dL/d(drive) enabled)")

    B_REF = 524288
    scale = (32768.0 / B_REF) ** 0.5
    LR_M = 0.02 * scale
    # The BASE model must be in the optimizer, exactly as base_train.py does it
    # (it calls model.setup_optimizer(matrix_lr=0.02*batch_lr_scale, ...)).
    # An earlier version of this probe omitted the base groups entirely, which
    # made "C3 freeze-base" identical to the reference BY CONSTRUCTION and
    # therefore meaningless. Base training is not optional here: the FiLM
    # runaway acts THROUGH the base model, so a frozen base cannot express it.
    # base_train.py uses the combined MuonAdamW optimizer and then APPENDS the
    # RDA groups to optimizer.param_groups. Do exactly that, so the base model
    # is trained by the real optimizer rather than a stand-in.
    optimizer = m.base.setup_optimizer(
        unembedding_lr=0.008 * scale,
        embedding_lr=0.3 * scale,
        scalar_lr=0.5 * scale,
        matrix_lr=0.02 * scale,
        weight_decay=0.0,
    )
    rda_groups = [
        dict(kind="adamw",
             params=list(st.film_scale.parameters()) + list(st.film_shift.parameters()),
             lr=0.02 * scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw",
             params=list(st.event_proj.parameters()) + list(st.pe_proj.parameters()),
             lr=0.02 * scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", params=[st.e_init],
             lr=0.5 * scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for g in rda_groups:
        g["initial_lr"] = g["lr"]
        optimizer.add_param_group(g)
    opt = optimizer
    n_state_p = sum(len(g["params"]) for g in rda_groups)
    say("base optimizer: %s with %d groups; appended %d state tensors"
        % (type(opt).__name__, len(opt.param_groups), n_state_p))
    say("drive-group LR=%.5f  (base_train.py batch_lr_scale)" % LR_M)
    say()

    rec = []
    carry = None
    _over = 0
    abort_fired = False
    for step in range(a.steps):
        # NOT set_to_none=True: MuonAdamW's fused kernels dereference every
        # grad unconditionally, so a None grad is a hard crash. Zeroing (rather
        # than clearing) is also what base_train.py does. Frozen parameters keep
        # a ZERO grad, so the kernels see a tensor and the update is exactly
        # a no-op for them -- the same effect as excluding them, without
        # mutating the optimizer's group structure between ablations.
        m.zero_grad(set_to_none=False)
        x, y, _ = next(loader)
        out = m(x, y, E_t_external=carry)
        loss = out["loss"]
        loss.backward()

        E_seq = out["E_next_seq"].detach()
        base_p = [p for p in m.base.parameters() if p.grad is not None]
        g_base = float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                                      for p in base_p))) if base_p else 0.0

        with torch.no_grad():
            fs = cap["film_scale_out"]          # the term ADDED to 1.0
            fh = cap["film_shift_out"]
            pe_raw = cap["pe_raw"].squeeze(-1) if cap["pe_raw"].dim() == 3 else cap["pe_raw"]
            pe_out = cap["pe_proj_out"]
            h = out["hidden"].detach()
            hm = out["hidden_mod"].detach()

            # per-position state change
            dE = (E_seq[:, 1:] - E_seq[:, :-1]).abs().max().item() if E_seq.size(1) > 1 else 0.0

            row = {
                "step": step,
                "loss": float(loss.detach()),
                # --- PE pathway ---
                "pe_min": float(pe_raw.min()), "pe_mean": float(pe_raw.mean()),
                "pe_max": float(pe_raw.max()),
                "pe_out_absmax": float(pe_out.abs().max()),
                "pe_proj_w_norm": gn(st.pe_proj.weight),
                "pe_proj_b_norm": gn(st.pe_proj.bias),
                "pe_perdim": [float(v) for v in pe_out.abs().max(dim=0).values.max(dim=0).values],
                "g_pe_proj": gn(st.pe_proj.weight.grad),
                # --- FiLM pathway ---
                "film_s_min": float(fs.min()), "film_s_max": float(fs.max()),
                "film_s_mean": float(fs.mean()), "film_s_rms": float(fs.pow(2).mean().sqrt()),
                "film_h_min": float(fh.min()), "film_h_max": float(fh.max()),
                "film_h_mean": float(fh.mean()), "film_h_rms": float(fh.pow(2).mean().sqrt()),
                "film_w_norm": gn(st.film_scale.weight) + gn(st.film_shift.weight),
                "g_film_scale": gn(st.film_scale.weight.grad),
                "g_film_shift": gn(st.film_shift.weight.grad),
                # --- event pathway (for continuity with the earlier study) ---
                "ep_w_norm": gn(st.event_proj.weight),
                "g_event_proj": gn(st.event_proj.weight.grad),
                "g_h_norm": gn(st.h_norm.weight.grad),
                # --- state ---
                "E_absmax": float(E_seq.abs().max()),
                "E_perdim": [float(v) for v in E_seq.abs().max(dim=0).values.max(dim=0).values],
                "E_mean": float(E_seq.mean()), "E_rms": float(E_seq.pow(2).mean().sqrt()),
                "dE_max": dE,
                # --- model ---
                "hidden_rms": float(h.pow(2).mean().sqrt()),
                "hidden_mod_rms": float(hm.pow(2).mean().sqrt()),
                "g_base": g_base,
            }
            # FiLM's effect on scale: 1 + film_s
            row["mod_scale_absmax"] = float((1.0 + fs).abs().max())
            row["mod_scale_min"] = float((1.0 + fs).min())

            # APPLIED gain -- the quantity that actually multiplies the
            # residual stream. The two lines above are built from the RAW
            # film_scale output, because the forward hook sees the module
            # output and _unit_rms is applied afterwards, inside modulate().
            # Measured directly (diagnostics/verify_film_s_definition.py):
            # max|1+fs| and max|1+_unit_rms(fs)| differ by roughly 2x, so
            # reporting only the raw one misreports the gain by that factor.
            # _unit_rms cannot be reconstructed from the aggregates stored
            # above -- it needs the per-position channel vector -- which is
            # why it is computed here rather than derived downstream.
            g_scale = _unit_rms(fs, st.film_rms)
            g_shift = _unit_rms(fh, st.film_rms)
            row["applied_scale_absmax"] = float((1.0 + g_scale).abs().max())
            row["applied_scale_min"] = float((1.0 + g_scale).min())
            row["applied_scale_rms"] = float((1.0 + g_scale).pow(2).mean().sqrt())
            row["applied_shift_absmax"] = float(g_shift.abs().max())
            row["applied_shift_rms"] = float(g_shift.pow(2).mean().sqrt())
            # The per-position RMS of the applied term is pinned at film_rms
            # by construction. Recorded so a reader can confirm the identity
            # instead of taking it on faith.
            row["applied_gain_rms_check"] = float(g_scale.pow(2).mean().sqrt())

            # ----------------------------------------------------------------
            # THREAT-DIMENSION DECOMPOSITION (diagnostics only).
            #
            # `threat` is DRIVE_DIMS[2]. Its drive at each position is
            #     drive_threat = tanh(event_term[2] + pe_proj(pe/10)[2])
            # so the two contributions can be separated and logged, together
            # with the tanh output BEFORE the convex combination updates the
            # state. The question this answers: is threat saturating because
            # its tanh ARGUMENT is being driven up (by the PE projection's
            # learned bias acting on a growing PE, or by the event term), or
            # because the recurrence is accumulating into a pinned state?
            #
            # Reproduces rda.py's drive_seq() for this one dimension:
            #   h    = hidden[:, :-1, :]              (= pre_outcome_hidden)
            #   ev   = RMS(event_proj(h_norm(h)), 0.5-equivalent normalization)
            #   arg  = event_drive_scale * ev + pe_proj(pe/10)
            #   drv  = tanh(arg)
            # pe/10 is recovered from the captured pe_raw, which the hook
            # already multiplied by pe_reference_scale, so pe/10 == pe_raw/100.
            #
            # Nothing here alters the computation: these are reads of
            # already-computed tensors, all under no_grad.
            # ----------------------------------------------------------------
            _T = DRIVE_DIMS.index("threat")
            with torch.no_grad():
                # drive_seq() casts hidden_seq to the event_proj weight dtype
                # before h_norm; mirror that exactly, or the decomposition is
                # computed in a different precision than production.
                _h = st.h_norm(h[:, :-1, :].to(st.event_proj.weight.dtype))
                _ev = st.event_proj(_h)
                _ev = _ev / _ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                _ev_term = st.event_drive_scale * _ev              # (B,T-1,8)
                _pe_in = (pe_raw.unsqueeze(-1) / st.pe_reference_scale).to(
                    st.pe_proj.weight.dtype)
                _pe_arg = st.pe_proj(_pe_in)                       # (B,T-1,8)
                _arg = _ev_term + _pe_arg
                _drv_t = torch.tanh(_arg[:, :, _T])
                row["threat_event_term_absmax"] = float(_ev_term[:, :, _T].abs().max())
                row["threat_event_term_mean"] = float(_ev_term[:, :, _T].mean())
                row["threat_pe_term_absmax"] = float(_pe_arg[:, :, _T].abs().max())
                row["threat_pe_term_mean"] = float(_pe_arg[:, :, _T].mean())
                row["threat_tanh_arg_absmax"] = float(_arg[:, :, _T].abs().max())
                row["threat_tanh_arg_mean"] = float(_arg[:, :, _T].mean())
                # pre-clip: the tanh output the recurrence mixes into the state
                row["threat_drive_preclip_absmax"] = float(_drv_t.abs().max())
                row["threat_drive_preclip_mean"] = float(_drv_t.mean())
                row["threat_E_absmax"] = float(E_seq[:, :, _T].abs().max())
                row["threat_E_mean"] = float(E_seq[:, :, _T].mean())
                # learned PE gain for this drive -- its "own" input weight
                row["threat_pe_proj_b"] = float(st.pe_proj.bias[_T])
                row["threat_pe_proj_w"] = float(st.pe_proj.weight[_T].abs().sum())
                row["threat_applied_gain"] = float((1.0 + g_scale).abs().max())

                # ----------------------------------------------------------------
                # PER-POSITION DRIVE DISTRIBUTION (diagnostics only).
                #
                # The scalar mean/absmax above CANNOT distinguish a bimodal
                # drive from a unimodal one near -1 carrying a thin positive
                # tail: a mean of -0.97 with an absmax of +0.99 is consistent
                # with both. That question is unresolved in E13-fixed-3000
                # and can only be settled by looking at the distribution
                # itself, so the quantiles and tail fractions are logged
                # every step, and a full histogram at selected steps.
                #
                # Written to a SEPARATE file so the existing per-step records
                # are byte-for-byte unchanged.
                # ----------------------------------------------------------------
                _flat = _drv_t.detach().float().flatten()
                _q = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
                                  device=_flat.device, dtype=torch.float32)
                _qs = torch.quantile(_flat, _q)
                _drow = {"step": step,
                         "threat_applied_gain": row["threat_applied_gain"]}
                for _nm, _v in zip(("p1", "p5", "p25", "p50",
                                    "p75", "p95", "p99"), _qs):
                    _drow["threat_drv_q_" + _nm] = float(_v)
                _drow["threat_drv_frac_gt_0p9"] = float((_flat > 0.9).float().mean())
                _drow["threat_drv_frac_lt_0p9"] = float((_flat < -0.9).float().mean())
                _drow["threat_drv_n"] = int(_flat.numel())
                DIST_ROWS.append(_drow)
                if step in DIST_STEPS:
                    # 100 bins over [-1.2, 1.2] plus explicit under/overflow.
                    # The range deliberately EXCEEDS the tanh range (-1, 1)
                    # so mass piling up at a bound is visible inside the
                    # histogram rather than silently clipped.
                    _hist = torch.histc(
                        _flat, bins=DIST_BINS, min=DIST_LO, max=DIST_HI)
                    DIST.setdefault(step, {
                        "step": step,
                        "applied_gain": float((1.0 + g_scale).abs().max()),
                        "threat_E_absmax": float(E_seq[:, :, _T].abs().max()),
                        "threat_E_mean": float(E_seq[:, :, _T].mean()),
                        "threat_drive_mean": float(_drv_t.mean()),
                        "threat_drive_absmax": float(_drv_t.abs().max()),
                        "n_positions": int(_flat.numel()),
                        "bin_lo": DIST_LO,
                        "bin_hi": DIST_HI,
                        "n_bins": int(DIST_BINS),
                        "underflow_count": int((_flat < DIST_LO).sum()),
                        "overflow_count": int((_flat > DIST_HI).sum()),
                        "hist": [int(v) for v in _hist.tolist()],
                    })

                # ----------------------------------------------------------------
                # (1) PRE-TANH ARGUMENT, DECOMPOSED  +  (2) GRADIENTS
                #     +  (3) INFORMATION / CORRELATION.  threat only.
                #
                # The pre-tanh argument is exactly
                #     arg = event_drive_scale * ev_RMS[threat]
                #         + pe_proj.weight[threat,0] * (pe/10)
                #         + pe_proj.bias[threat]
                # so the three additive terms are separable and each is
                # logged as mean and p5/p95 across positions. Signs are kept
                # everywhere: the sign of each term is the whole point, since
                # "which way is this dimension being pushed" is the question.
                #
                # Gradients are read HERE, which is valid because loss.backward()
                # ran at line 293 and opt.step() does not run until 499, so
                # .grad holds this step's gradient unmodified.
                _argT = _arg[:, :, _T]                            # (B,T-1)
                _evT = _ev_term[:, :, _T]
                _peT = _pe_arg[:, :, _T]
                _wf = st.pe_proj.weight[_T, 0]
                _bf = st.pe_proj.bias[_T]
                _wterm = _wf * _pe_in[:, :, 0]                   # weight*PE
                _bterm = torch.full_like(_argT, float(_bf))      # bias
                _prow = {"step": step,
                         "pe_proj_bias_threat": float(_bf),
                         "pe_proj_weight_threat": float(_wf)}
                for _nm, _src in (("ev", _evT), ("pe", _peT), ("wpe", _wterm),
                                  ("bias", _bterm), ("arg", _argT),
                                  ("drv", _drv_t)):
                    _fq = _src.detach().float().flatten()
                    _qq = torch.quantile(
                        _fq, torch.tensor(QLEVELS, device=_fq.device,
                                          dtype=torch.float32))
                    _prow[_nm + "_mean"] = float(_fq.mean())
                    for _n, _v in zip(QNAME, _qq):
                        _prow["%s_q_%s" % (_nm, _n)] = float(_v)
                    _prow[_nm + "_min"] = float(_fq.min())
                    _prow[_nm + "_max"] = float(_fq.max())
                    _prow[_nm + "_std"] = float(_fq.std())
                # tanh'(arg) = 1 - drive^2. If this goes to ~0 the channel is
                # DEAD: no gradient can flow back through the tanh and the
                # dimension is frozen regardless of what the weights do.
                _d1 = (1.0 - _drv_t.detach().float() ** 2)
                _prow["tanh_deriv_mean"] = float(_d1.mean())
                _prow["tanh_deriv_p05"] = float(torch.quantile(
                    _d1.flatten(), 0.05))
                _prow["tanh_deriv_frac_lt_0p01"] = float((_d1 < 0.01).float().mean())
                if step in PREACT_STEPS:
                    _ph = torch.histc(_argT.detach().float().flatten(),
                                      bins=PRE_BINS, min=PRE_LO, max=PRE_HI)
                    _prow["hist"] = [int(v) for v in _ph.tolist()]
                    _prow["hist_lo"], _prow["hist_hi"] = PRE_LO, PRE_HI
                    _prow["hist_under"], _prow["hist_over"] = [
                        int(v) for v in (
                            (_argT < PRE_LO).sum().item(),
                            (_argT > PRE_HI).sum().item())]
                PREACT_ROWS.append(_prow)

                # (2) GRADIENTS on the threat row/entry of pe_proj.
                # gn() returns the L2 norm, so these are non-negative by
                # construction; the SIGNED value is logged alongside for the
                # bias, because "is the bias still being pushed down" is the
                # question and a norm cannot answer it.
                _gb = st.pe_proj.bias.grad
                _gw = st.pe_proj.weight.grad
                _grow = {"step": step,
                         "pe_proj_bias_threat": float(_bf)}
                if _gb is not None:
                    _grow["g_bias_threat_signed"] = float(_gb[_T])
                    _grow["g_bias_threat_abs"] = float(_gb[_T].abs())
                    _grow["g_bias_threat_norm_all"] = gn(_gb)
                if _gw is not None:
                    _grow["g_weight_threat_signed"] = float(_gw[_T].sum())
                    _grow["g_weight_threat_abs"] = float(_gw[_T].abs().sum())
                    _grow["g_weight_threat_norm"] = float(_gw[_T].norm())
                    _grow["g_weight_threat_absmax"] = float(_gw[_T].abs().max())
                _grow["g_event_proj_threat_norm"] = (
                    float(st.event_proj.weight.grad[_T].norm())
                    if st.event_proj.weight.grad is not None else None)
                _grow["tanh_deriv_mean"] = _prow["tanh_deriv_mean"]
                GRAD_ROWS.append(_grow)

                # (3) INFORMATION CHECK: does the drive still track its inputs?
                # Correlation is computed across POSITIONS (flattened), which
                # is the only axis along which these vary together. Reported
                # for every raw input of the threat drive. Also the std of the
                # drive across positions: if the drive is collapsing to a
                # constant, this goes to 0 and correlation becomes undefined
                # (0/0), which is itself the finding.
                if step in PREACT_STEPS:
                    _d = _drv_t.detach().float().flatten()
                    _crow = {"step": step,
                             "drive_std": float(_d.std()),
                             "drive_mean": float(_d.mean()),
                             "drive_absmax": float(_d.abs().max()),
                             "drive_min": float(_d.min()),
                             "drive_max": float(_d.max())}
                    for _nm, _src in (("ev_term", _evT), ("pe_term", _peT),
                                      ("wpe_term", _wterm)):
                        _s = _src.detach().float().flatten()
                        _s = _s - _s.mean()
                        _dd = _d - _d.mean()
                        _den = float(_s.norm() * _dd.norm())
                        _crow["corr_" + _nm] = (
                            float((_s * _dd).sum() / _den) if _den > 1e-12
                            else None)
                        _crow[_nm + "_std"] = float(_s.std())
                    # correlation of the drive with its own PRE-tanh argument:
                    # for a monotone tanh this is always high, so it
                    # distinguishes almost nothing. Logged anyway to make the
                    # near-1 value explicit rather than assumed.
                    _a = _argT.detach().float().flatten()
                    _a = _a - _a.mean()
                    _den = float(_a.norm() * _dd.norm())
                    _crow["corr_arg"] = (float((_a * _dd).sum() / _den)
                                         if _den > 1e-12 else None)
                    INFO_CORR[step] = _crow

        # ----------------------------------------------------------------
        # ALL 8 DIMENSIONS -- the generalization of the block above.
        #
        # _ev_term, _pe_arg, _arg, _drv are (B, T-1, 8) tensors already
        # computed above under no_grad, so the decomposition is read from
        # the same tensors the threat block used. The threat numbers here
        # are therefore IDENTICAL to the threat-only block by construction,
        # not by coincidence.
        # ----------------------------------------------------------------
        with torch.no_grad():
            _driveg = LAST_DRIVE.get("t")
            # The drive for ALL dimensions. The threat-only block above only
            # ever indexed _arg[:, :, _T]; this is the same tensor before
            # that indexing, so the threat numbers agree by construction.
            _drv = torch.tanh(_arg)
            _dgrad = None
            if _driveg is not None and _driveg.grad is not None:
                _dgrad = _driveg.grad.detach().float()
            _arow = {"step": step}
            # pe_proj is Linear(1, n_drives), so the weight term for every
            # dimension is the SAME scalar pe/10 times that dimension's own
            # single weight. Read the two parameter vectors once.
            _pew = st.pe_proj.weight.detach().flatten()      # (8,)
            _peb = st.pe_proj.bias.detach()                  # (8,)
            for _i, _nm in enumerate(NAMES):
                _a = _arg[:, :, _i]
                _e = _ev_term[:, :, _i]
                # pe weight term and bias for this dimension
                _wt = (_pew[_i] * _pe_in[:, :, 0])
                _bt = _peb[_i]
                _fa = _a.detach().float().flatten()
                _fe = _e.detach().float().flatten()
                _fw = _wt.detach().float().flatten()
                _qa = torch.quantile(_fa, torch.tensor(
                    (0.05, 0.50, 0.95), device=_fa.device, dtype=torch.float32))
                _qf = torch.quantile(_fw, torch.tensor(
                    (0.05, 0.50, 0.95), device=_fw.device, dtype=torch.float32))
                _arow["%s_arg_p5" % _nm] = float(_qa[0])
                _arow["%s_arg_p50" % _nm] = float(_qa[1])
                _arow["%s_arg_p95" % _nm] = float(_qa[2])
                _arow["%s_arg_mean" % _nm] = float(_fa.mean())
                _arow["%s_arg_min" % _nm] = float(_fa.min())
                _arow["%s_arg_max" % _nm] = float(_fa.max())
                _arow["%s_ev_mean" % _nm] = float(_fe.mean())
                _arow["%s_wpe_mean" % _nm] = float(_fw.mean())
                _arow["%s_bias" % _nm] = float(_bt)
                _arow["%s_pe_weight" % _nm] = float(_pew[_i])
                # IDENTITY, all 8 dims, every step.
                _res = abs(float(_fe.mean() + _fw.mean() + _bt) - _fa.mean())
                _arow["%s_identity_residual" % _nm] = _res
                if _res > IDENTITY_WORST["residual"]:
                    IDENTITY_WORST.update(residual=_res, step=step, dim=_nm)
                _fd = _drv[:, :, _i].detach().float()
                _qdf = torch.quantile(_fd.flatten(), torch.tensor(
                    (0.05, 0.50, 0.95), device=_fd.device, dtype=torch.float32))
                _arow["%s_drv_p5" % _nm] = float(_qdf[0])
                _arow["%s_drv_p50" % _nm] = float(_qdf[1])
                _arow["%s_drv_p95" % _nm] = float(_qdf[2])
                _arow["%s_drv_absmax" % _nm] = float(_fd.abs().max())
                _arow["%s_drv_min" % _nm] = float(_fd.min())
                _arow["%s_drv_max" % _nm] = float(_fd.max())
                _arow["%s_tanh_deriv_mean" % _nm] = float((1.0 - _fd ** 2).mean())
                _arow["%s_ev_std" % _nm] = float(_fe.std())
                _gb = st.pe_proj.bias.grad
                _gw = st.pe_proj.weight.grad
                if _gb is not None:
                    _arow["%s_g_bias_signed" % _nm] = float(_gb[_i])
                    _arow["%s_g_bias_norm" % _nm] = float(_gb[_i].abs())
                if _gw is not None:
                    _arow["%s_g_weight_signed" % _nm] = float(_gw[_i].sum())
                    _arow["%s_g_weight_norm" % _nm] = float(_gw[_i].norm())
                # (8) dL/d(drive_d) per position -- the loss-side gradient,
                # distinct from the tanh derivative. Signed, with std.
                if _dgrad is not None:
                    _gg = _dgrad[:, :, _i]
                    _arow["%s_dLddrive_mean" % _nm] = float(_gg.mean())
                    _arow["%s_dLddrive_std" % _nm] = float(_gg.std())
                    _arow["%s_dLddrive_absmax" % _nm] = float(_gg.abs().max())
            ALL_ROWS.append(_arow)

            # (7) variance shares at the sampled steps
            if step in PREACT_ALL_STEPS:
                _irow = {"step": step}
                for _i, _nm in enumerate(NAMES):
                    _e = _ev_term[:, :, _i].detach().float().flatten()
                    _w = (_pew[_i] * _pe_in[:, :, 0]).detach().float().flatten()
                    _ec, _wc = _e - _e.mean(), _w - _w.mean()
                    _ce = float((_ec ** 2).mean())
                    _cw = float((_wc ** 2).mean())
                    _cov = float((_ec * _wc).mean())
                    _irow["%s_ev_var" % _nm] = _ce
                    _irow["%s_wpe_var" % _nm] = _cw
                    _irow["%s_cov" % _nm] = _cov
                    _tot = _ce + _cw
                    _irow["%s_var_total" % _nm] = _tot
                    _irow["%s_ev_var_share" % _nm] = (_ce / _tot) if _tot > 0 else None
                    _irow["%s_wpe_var_share" % _nm] = (_cw / _tot) if _tot > 0 else None
                    _irow["%s_ev_std" % _nm] = float(_e.std())
                    _irow["%s_wpe_std" % _nm] = float(_w.std())
                ALL_INFO[step] = _irow

        # ----------------------------------------------------------------
        # FILM COLUMN WEIGHTS AND PER-CHANNEL E CONTRIBUTION.
        #
        # Only at FILM_STEPS: this is 8 dims x 768 channels of weight and a
        # full pass over positions, and the quantity of interest (how much
        # of a channel's contribution is DC) changes slowly.
        # ----------------------------------------------------------------
        if step in FILM_STEPS and "film_E_in" in cap:
            with torch.no_grad():
                _FE = cap["film_E_in"]                      # (B,T,8)
                _Wsc = st.film_scale.weight.detach()        # (d_model, 8)
                _Wsh = st.film_shift.weight.detach()        # (d_model, 8)
                _frow = {"step": step,
                         "film_rms": float(st.film_rms),
                         "note": "E is the FiLM INPUT (film_state, "
                                 "cat([E_start, E_next_seq], dim=1)), not "
                                 "E_next_seq and not the drive"}
                for _i, _nm in enumerate(NAMES):
                    # --- column of the weight that reads E_d ---
                    _csc = _Wsc[:, _i]
                    _csh = _Wsh[:, _i]
                    _frow["%s_scale_col_norm" % _nm] = float(_csc.norm())
                    _frow["%s_scale_col_absmax" % _nm] = float(_csc.abs().max())
                    _frow["%s_shift_col_norm" % _nm] = float(_csh.norm())
                    _frow["%s_shift_col_absmax" % _nm] = float(_csh.abs().max())
                    # signed mean of the column (not just its norm)
                    _frow["%s_scale_col_mean_signed" % _nm] = float(_csc.mean())
                    _frow["%s_shift_col_mean_signed" % _nm] = float(_csh.mean())
                    # --- E_d across positions, signed ---
                    _e = _FE[:, :, _i].detach().float()
                    _frow["%s_E_mean" % _nm] = float(_e.mean())
                    _frow["%s_E_std" % _nm] = float(_e.std())
                    _frow["%s_E_min" % _nm] = float(_e.min())
                    _frow["%s_E_max" % _nm] = float(_e.max())
                    _frow["%s_E_absmax" % _nm] = float(_e.abs().max())
                    # --- contribution to the FiLM output ---
                    # --- contribution to the FiLM output ---
                    #
                    # AXES MATTER, and the first version of this block got it
                    # wrong. For each output unit j the contribution is
                    #     c_j(p) = W[j,d] * E_d(p)
                    # so reducing over POSITIONS ONLY:
                    #     offset_dj  = W[j,d] * E_mean_d
                    #     varying_dj = W[j,d] * E_std_d
                    # and aggregating over units by RMS:
                    #     offset_rms  = |E_mean| * ||W[:,d]||_rms
                    #     varying_rms =  E_std  * ||W[:,d]||_rms
                    # The column factor CANCELS in the ratio, so the correct
                    # ratio is exactly E_mean/E_std, independent of column norm.
                    #
                    # The earlier version called .mean()/.std() on the
                    # (d_model, B, T) tensor with NO dim argument, reducing
                    # over UNITS and POSITIONS together. That contaminates
                    # "varying" with the spread of the column DIRECTION
                    # across units, which is not position-variation at all.
                    # Those values are retained below, clearly labelled
                    # FLATTENED, because they produced an earlier wrong answer.
                    _Wsc_d = _Wsc[:, _i].detach().float()      # (d_model,)
                    _Wsh_d = _Wsh[:, _i].detach().float()
                    _rms_sc = float(_Wsc_d.pow(2).mean().sqrt())
                    _rms_sh = float(_Wsh_d.pow(2).mean().sqrt())
                    for _tag, _Wd, _rms in (("scale", _Wsc_d, _rms_sc),
                                           ("shift", _Wsh_d, _rms_sh)):
                        _off_u = _Wd * float(_e.mean())        # (d_model,)
                        _var_u = _Wd * float(_e.std())
                        _frow["%s_%s_offset_rms" % (_nm, _tag)] = float(
                            _off_u.pow(2).mean().sqrt())
                        _frow["%s_%s_varying_rms" % (_nm, _tag)] = float(
                            _var_u.pow(2).mean().sqrt())
                        _frow["%s_%s_col_rms" % (_nm, _tag)] = _rms
                        # signed: the offset inherits E's sign
                        _frow["%s_%s_offset_over_varying" % (_nm, _tag)] = (
                            float(_e.mean() / _e.std())
                            if float(_e.std()) > 1e-12 else None)
                        _frow["%s_%s_offset_absmax_unit" % (_nm, _tag)] = float(
                            _off_u.abs().max())
                        _frow["%s_%s_varying_absmax_unit" % (_nm, _tag)] = float(
                            _var_u.abs().max())

                    # Combined (scale + shift) -- what modulate() adds.
                    _oc = _Wsc_d * float(_e.mean()) + _Wsh_d * float(_e.mean())
                    _vc = _Wsc_d * float(_e.std()) + _Wsh_d * float(_e.std())
                    _frow["%s_film_offset_rms" % _nm] = float(
                        _oc.pow(2).mean().sqrt())
                    _frow["%s_film_varying_rms" % _nm] = float(
                        _vc.pow(2).mean().sqrt())
                    _frow["%s_film_offset_over_varying" % _nm] = (
                        float(_e.mean() / _e.std())
                        if float(_e.std()) > 1e-12 else None)

                    # --- RETAINED-BUT-WRONG, kept for provenance ---
                    # These reduce over (d_model, B, T) together. The outer
                    # product is explicit: (768,1,1) * (1,B,T).
                    _contrib = (_Wsc_d.reshape(-1, 1, 1)
                                * _e.reshape(1, *_e.shape))
                    _ctot = _contrib + (_Wsh_d.reshape(-1, 1, 1)
                                        * _e.reshape(1, *_e.shape))
                    _frow["%s_FLATTENED_contrib_mean" % _nm] = float(_ctot.mean())
                    _frow["%s_FLATTENED_contrib_std" % _nm] = float(_ctot.std())
                    _frow["%s_FLATTENED_off_over_var" % _nm] = (
                        float(_ctot.mean() / _ctot.std())
                        if float(_ctot.std()) > 1e-12 else None)
                FILM_ROWS.append(_frow)

        # ----------------------------------------------------------------
        # RECONSTRUCTION CHECK. Rebuild film_scale and film_shift from their
        # own definitions and compare against the CAPTURED module outputs.
        #
        #     film_scale(E) = W_scale @ E + b_scale
        #                   = sum_d W_scale[:, d] * E[:, d] + b_scale
        #
        # If this disagrees with the captured film_scale_out by more than
        # 1e-4, then the per-dimension decomposition above is being applied to
        # a tensor that is NOT the one FiLM used, and every offset/varying
        # number is void. So this is a gate, not a diagnostic.
        # ----------------------------------------------------------------
        if step in FILM_STEPS and "film_E_in" in cap and "film_scale_out" in cap:
            with torch.no_grad():
                _FEf = cap["film_E_in"]
                # Forward-time weight clones (hypothesis A). Using these
                # rather than the live parameters is what makes the check a
                # test rather than a coin flip.
                _Ws = cap["film_W_scale"]
                _bs = cap["film_b_scale"]
                _Wh = cap["film_W_shift"]
                _bh = cap["film_b_shift"]
                _cap_s = cap["film_scale_out"].float()
                _cap_h = cap["film_shift_out"].float()

                # --- (2) Test C: is the E used here the SAME tensor FiLM saw?
                # Compared by value (torch.equal) and by identity (data_ptr),
                # because a value-equal copy would hide a substitution.
                _recon = {
                    "step": step,
                    "E_equal": bool(torch.equal(_FEf, cap["film_E_in"])),
                    "E_data_ptr_match": (int(_FEf.data_ptr())
                                         == int(cap["film_E_in"].data_ptr())),
                    "E_shape": list(_FEf.shape),
                    "W_scale_ptr": int(_Ws.data_ptr()),
                    "W_scale_live_ptr": int(st.film_scale.weight.data_ptr()),
                    # If the live parameter has been updated in place, the
                    # storage pointer is unchanged while the VALUES differ.
                    "W_scale_values_drifted_vs_live": float(
                        (_Ws - st.film_scale.weight.detach()).abs().max()),
                    "W_shift_values_drifted_vs_live": float(
                        (_Wh - st.film_shift.weight.detach()).abs().max()),
                }

                # --- (1) Test A: float64 reconstruction from the clones.
                _sum64_s = _bs.double().reshape(-1, 1, 1).clone()
                _sum64_h = _bh.double().reshape(-1, 1, 1).clone()
                _E64 = _FEf.double()
                for _i in range(len(NAMES)):
                    _e_i = _E64[:, :, _i]
                    _sum64_s = _sum64_s + (_Ws[:, _i].double().reshape(-1, 1, 1)
                                           * _e_i.reshape(1, *_e_i.shape))
                    _sum64_h = _sum64_h + (_Wh[:, _i].double().reshape(-1, 1, 1)
                                           * _e_i.reshape(1, *_e_i.shape))
                _r64_s = _sum64_s.permute(1, 2, 0)
                _r64_h = _sum64_h.permute(1, 2, 0)
                _d_s = float((_r64_s - _cap_s.double()).abs().max())
                _d_h = float((_r64_h - _cap_h.double()).abs().max())
                _recon.update({
                    "fp64_resid_scale": _d_s,
                    "fp64_resid_shift": _d_h,
                    "fp64_rel_scale": _d_s / max(float(_cap_s.abs().max()), 1e-30),
                    "fp64_rel_shift": _d_h / max(float(_cap_h.abs().max()), 1e-30),
                    "cap_abs_scale": float(_cap_s.abs().max()),
                    "cap_abs_shift": float(_cap_h.abs().max()),
                })

                # --- (4) Test B: force highest precision / no TF32 and
                # recompute with F.linear from the same clones. Training
                # precision is NOT changed: the flags are set and restored
                # inside a try/finally, and the restoration is verified.
                _p0 = torch.get_float32_matmul_precision()
                _t0 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.set_float32_matmul_precision("highest")
                    torch.backends.cuda.matmul.allow_tf32 = False
                    # F.linear on a (B,T,n_drives) input already returns
                    # (B,T,d_model); no permute is needed or correct here.
                    _lin_s = torch.nn.functional.linear(_FEf, _Ws, _bs)
                    _lin_h = torch.nn.functional.linear(_FEf, _Wh, _bh)
                finally:
                    torch.set_float32_matmul_precision(_p0)
                    torch.backends.cuda.matmul.allow_tf32 = _t0
                _recon["flags_restored"] = bool(
                    torch.get_float32_matmul_precision() == _p0
                    and torch.backends.cuda.matmul.allow_tf32 == _t0)
                # --- (B') Reproduce the FORWARD's arithmetic exactly: the
                # forward pass runs with whatever precision was live at the
                # time (measured inside the wrapper as
                # float32_matmul_precision="high", allow_tf32=True), which
                # on this GPU means TF32 for fp32 matmuls. Reconstructing
                # under THOSE settings must agree with the captured output to
                # near round-off; reconstructing under 'highest' cannot,
                # because it is a different arithmetic, not a more accurate
                # view of the same one.
                try:
                    torch.set_float32_matmul_precision(
                        cap["film_env"]["float32_matmul_precision"])
                    torch.backends.cuda.matmul.allow_tf32 = bool(
                        cap["film_env"]["cuda_matmul_allow_tf32"])
                    _tf_s = torch.nn.functional.linear(_FEf, _Ws, _bs)
                    _tf_h = torch.nn.functional.linear(_FEf, _Wh, _bh)
                finally:
                    torch.set_float32_matmul_precision(_p0)
                    torch.backends.cuda.matmul.allow_tf32 = _t0
                _t_s = float((_tf_s - _cap_s).abs().max())
                _t_h = float((_tf_h - _cap_h).abs().max())
                _b_s = float((_lin_s - _cap_s).abs().max())
                _b_h = float((_lin_h - _cap_h).abs().max())
                _recon.update({
                    "highest_resid_scale": _b_s,
                    "highest_resid_shift": _b_h,
                    "highest_rel_scale": _b_s / max(float(_cap_s.abs().max()),
                                                    1e-30),
                    "highest_rel_shift": _b_h / max(float(_cap_h.abs().max()),
                                                    1e-30),
                    "forward_settings_resid_scale": _t_s,
                    "forward_settings_resid_shift": _t_h,
                    "forward_settings_rel_scale": _t_s / max(
                        float(_cap_s.abs().max()), 1e-30),
                    "forward_settings_rel_shift": _t_h / max(
                        float(_cap_h.abs().max()), 1e-30),
                    "flags_restored_after_tf32": bool(
                        torch.get_float32_matmul_precision() == _p0
                        and torch.backends.cuda.matmul.allow_tf32 == _t0),
                    "verdict": ("TF32 forward: reproduces to round-off"
                                if max(_t_s, _t_h) <= RECON_TOL
                                else "UNEXPLAINED"),
                    "env": cap.get("film_env"),
                })
                RECON.append(_recon)
        rec.append(row)

        # ABORT RULE, identical to base_train.py: a drive pinned above the
        # bound for N consecutive steps means the tanh has stopped responding
        # and no later number is attributable to the drive. Stop and record.
        if a.abort_steps > 0:
            if row["E_absmax"] >= a.abort_bound:
                _over += 1
            else:
                _over = 0
            if _over >= a.abort_steps:
                abort_fired = True
                say("")
                say("*** ABORT: max|E| >= %.3f for %d consecutive steps (ending step %d)"
                    % (a.abort_bound, a.abort_steps, step))
                say("*** max|E| = %.4f. Stopping; recorded as FAIL."
                    % row["E_absmax"])
                break

        # A frozen parameter has requires_grad=False, so autograd never
        # populates its .grad. The fused Muon/AdamW kernels dereference every
        # grad unconditionally, so hand them an explicit zero: the update is
        # then exactly a no-op for those tensors.
        for grp in opt.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
        opt.step()
        # base_train.py's get_lr_multiplier: 40 warmup steps, so the drive
        # group LR ramps from ~0 to full over the first 40 steps. Matching it
        # matters -- the onset timing is the thing under investigation.
        warmup = 40
        lr_mult = min(1.0, (step + 1) / warmup)
        for g in opt.param_groups:
            if "initial_lr" in g:
                g["lr"] = g["initial_lr"] * lr_mult
        carry = out["E_next"]

        if step % 25 == 0 or step == a.steps - 1:
            say("step %3d | loss=%.4f pe_max=%.3f peWb=%.3f peOut=%.3f | "
                "film s=%.3f APPLIED=%.3f shift=%.3f filmW=%.3f | maxE=%.4f | gbase=%.1f"
                % (step, row["loss"], row["pe_max"],
                   row["pe_proj_b_norm"], row["pe_out_absmax"],
                   row["mod_scale_absmax"], row["applied_scale_absmax"],
                   row["film_h_max"],
                   row["film_w_norm"], row["E_absmax"], row["g_base"]))

    # ----------------------------------------------------------------
    # END-OF-RUN PER-DIMENSION ABLATION (diagnostics only).
    #
    # Runs AFTER the final training step. Weights are frozen (requires_grad
    # off) and no optimizer step occurs here, so nothing about training can
    # change. The held-out batches are drawn ONCE, up front, and the SAME
    # list is used for every condition -- redrawing per condition would
    # compare different data and swamp the effect being measured.
    #
    # The carry state (E_t_external) is reset to the SAME value before every
    # condition, taken from the final training step. It is a recurrent
    # quantity, so letting it drift between conditions would make the deltas
    # incomparable.
    # ----------------------------------------------------------------
    ABLATION = None
    if a.ablate:
        say("\n=== END-OF-RUN PER-DIMENSION ABLATION ===")
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        # Fixed held-out batches, drawn once.
        val_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok, 4, 1024, split="val", device=device)
        batches = []
        _vg = torch.Generator().manual_seed(1234)   # fixed, independent of a.seed
        for _ in range(max(1, a.ablate_batches)):
            xb, yb, _ = next(val_loader)
            batches.append((xb, yb))
        say("held-out batches: %d (split=val, drawn once, same for all "
            "conditions)" % len(batches))

        # The carry state every condition starts from.
        carry_fixed = carry.detach().clone() if carry is not None else None
        if carry_fixed is not None:
            say("carry E_t_external reset per condition: shape %s, "
                "mean %s" % (list(carry_fixed.shape),
                             ["%+.5f" % v for v in carry_fixed.mean(0).tolist()]))

        def _eval_condition(mode, dims, label):
            """Mean loss over the fixed batches under one ablation."""
            # Restore carry identically before EVERY condition.
            m.zero_grad(set_to_none=True)
            if carry_fixed is not None:
                c = carry_fixed.detach().clone()
            else:
                c = None
            if mode is None:
                ABL["enabled"] = False
                ABL["recorded"] = False
            else:
                ABL["recorded"] = False
                set_ablation(mode, dims)
            try:
                tot, nb = 0.0, 0
                per_batch = []
                with torch.no_grad():
                    for xb, yb in batches:
                        o = m(xb, yb, E_t_external=c)
                        l = float(o["loss"])
                        per_batch.append(l)
                        tot += l
                        nb += 1
                state = ABL.get("state_used")
                return {"label": label, "mode": mode or "baseline",
                        "dims": (list(dims) if dims is not None else "all"),
                        "loss_mean": tot / nb, "per_batch": per_batch,
                        "state_used": state}
            finally:
                # Restored unconditionally, including on an exception, so the
                # ablation cannot leak into anything after it.
                ABL["enabled"] = False
                ABL["mode"] = None
                ABL["dims"] = None

        conds = [("baseline", None, None), ("baseline_repeat", None, None)]
        for i, nm in enumerate(NAMES):
            conds.append(("mean_ablate_" + nm, "mean", (i,)))
        for i, nm in enumerate(NAMES):
            conds.append(("zero_ablate_" + nm, "zero", (i,)))
        conds.append(("mean_ablate_ALL", "mean", None))
        conds.append(("zero_ablate_ALL", "zero", None))

        rows_abl = []
        for lbl, md, dm in conds:
            r = _eval_condition(md, dm, lbl)
            rows_abl.append(r)
            say("  %-28s loss %.6f" % (lbl, r["loss_mean"]))
        base = rows_abl[0]["loss_mean"]
        rep = rows_abl[1]["loss_mean"]
        noise = abs(rep - base)
        say("  baseline %.6f | repeat %.6f | NOISE FLOOR %.3e" %
            (base, rep, noise))
        for r in rows_abl:
            r["delta_vs_baseline"] = r["loss_mean"] - base
            r["abs_delta"] = abs(r["delta_vs_baseline"])
            r["within_noise"] = bool(abs(r["delta_vs_baseline"]) <= noise)
        ABLATION = {
            "batches": len(batches),
            "split": "val",
            "batch_size": 4, "seq_len": 1024,
            "n_conditions": len(rows_abl),
            "baseline_loss": base,
            "baseline_repeat_loss": rep,
            "noise_floor_abs": noise,
            "noise_floor_signed": rep - base,
            "noise_note": "the run is not bit-deterministic (GPU "
                          "nondeterminism in the backward pass), so the "
                          "baseline repeat bounds what 'within noise' means",
            "dims": NAMES,
            "results": rows_abl,
        }
        # Disarm and put the model back exactly as found.
        ABL["enabled"] = False
        ABL["mode"] = None
        ABL["dims"] = None
        m.train()
        say("ablation complete; modulate wrapper disarmed in finally")

    for h in handles:
        h.remove()
    left = (len(st.pe_proj._forward_hooks) + len(st.pe_proj._backward_hooks) +
            len(st.film_scale._forward_hooks) + len(st.film_shift._forward_hooks))
    # The method wrappers (retain_grad on drive_seq, the film_E_in capture on
    # modulate) are NOT module hooks and so are invisible to the counter
    # above. Restoring them is what makes the check meaningful: a wrapper
    # left in place would silently outlive the run.
    _left_methods = 0
    if getattr(st, "_probe_orig_modulate", None) is not None:
        st.modulate = st._probe_orig_modulate
        st._probe_orig_modulate = None
        _left_methods += 1
    if getattr(st, "_probe_orig_drive_seq", None) is not None:
        st.drive_seq = st._probe_orig_drive_seq
        st._probe_orig_drive_seq = None
        _left_methods += 1
    say("\nhooks after run: %d (expect 0)" % left)
    say("method wrappers restored: %d (expect 2 when instrumented)" % _left_methods)
    first = next((r["step"] for r in rec if r["E_absmax"] > 0.98), None)
    say("\n=== %s ===" % a.label)
    say("first step |E|>0.98 : %s" % ("never" if first is None else first))
    say("max |E|             : %.4f" % max(r["E_absmax"] for r in rec))
    say("final |E|           : %.4f" % rec[-1]["E_absmax"])
    say("PE   first/max/final: %.3f / %.3f / %.3f"
        % (rec[0]["pe_max"], max(r["pe_max"] for r in rec), rec[-1]["pe_max"]))
    say("peW  first/final    : %.4f / %.4f" % (rec[0]["pe_proj_w_norm"], rec[-1]["pe_proj_w_norm"]))
    say("peB  first/final    : %.4f / %.4f" % (rec[0]["pe_proj_b_norm"], rec[-1]["pe_proj_b_norm"]))
    say("RAW   film scale max: %.4f   (max|1+fs|, pre-_unit_rms)"
        % max(r["mod_scale_absmax"] for r in rec))
    say("APPLIED gain max    : %.4f   (max|1+_unit_rms(fs,%.2f)|)"
        % (max(r["applied_scale_absmax"] for r in rec), st.film_rms))
    say("applied gain rms chk: %.4f   (must equal film_rms=%.2f)"
        % (rec[-1]["applied_gain_rms_check"], st.film_rms))
    say("RAW   film shift max: %.4f" % max(r["film_h_max"] for r in rec))
    say("APPLIED shift max   : %.4f" % max(r["applied_shift_absmax"] for r in rec))
    say("hidden rms first/last: %.4f / %.4f" % (rec[0]["hidden_rms"], rec[-1]["hidden_rms"]))
    say("g_base max          : %.2f" % max(r["g_base"] for r in rec))
    say("loss first/last     : %.4f / %.4f" % (rec[0]["loss"], rec[-1]["loss"]))
    if a.abort_steps > 0:
        say("ABORT RULE          : |E|>0.98 for %d consecutive steps -> %s"
            % (a.abort_steps, "FIRED" if abort_fired else "not fired"))

    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        say("wrote %s" % a.out)
    if a.dist_out:
        # A short run may never reach DIST_STEPS[-1]; record the last step
        # actually visited so the file is not silently missing its endpoint.
        if (a.steps - 1) in DIST:
            pass
        elif DIST:
            last = max(DIST)
            if a.steps - 1 != last:
                say("  note: step %d not in DIST_STEPS; last histogram is at %d"
                    % (a.steps - 1, last))
        json.dump({
            "label": a.label,
            "seed": a.seed,
            "steps": a.steps,
            "dimension": "threat (DRIVE_DIMS[2])",
            "quantity": "pre-clip drive, tanh(event_term + pe_term), "
                        "flattened over batch and position",
            "quantiles_every_step": True,
            "quantile_levels": [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            "histogram_steps": sorted(DIST),
            "per_step": DIST_ROWS,
            "histograms": [DIST[s] for s in sorted(DIST)],
        }, open(a.dist_out, "w"), indent=1)
        say("wrote %s (%d step rows, %d histograms)"
            % (a.dist_out, len(DIST_ROWS), len(DIST)))
    if a.preact_out:
        json.dump({
            "label": a.label,
            "seed": a.seed,
            "steps": a.steps,
            "dimension": "threat (DRIVE_DIMS[2])",
            "identity": "arg = event_drive_scale*ev_RMS[T] "
                        "+ pe_proj.weight[T,0]*(pe/10) + pe_proj.bias[T]",
            "wpe_term": "pe_proj.weight[T,0]*(pe/pe_reference_scale); the "
                        "weight's own contribution",
            "pe_term": "pe_proj(pe/10)[T]; weight term AND bias together",
            "gradient_note": "read after loss.backward() and before "
                             "opt.step(), so these are this step's gradients "
                             "unmodified. bias/weight entries are SIGNED.",
            "quantile_levels": list(QLEVELS),
            "preact_every_step": PREACT_ROWS,
            "grad_every_step": GRAD_ROWS,
            "info_steps": [INFO_CORR[s] for s in sorted(INFO_CORR)],
        }, open(a.preact_out, "w"), indent=1)
        say("wrote %s (%d preact rows, %d grad rows, %d corr steps)"
            % (a.preact_out, len(PREACT_ROWS), len(GRAD_ROWS),
               len(INFO_CORR)))
    if a.preact_all_out:
        # A stray Tensor would surface as a bare "Object of type Tensor is
        # not JSON serializable" with no indication of WHICH field, which is
        # useless when there are ~40 fields per row x 8 dims x 3000 steps.
        # This walks the structure and names any non-JSON leaf, then coerces
        # 0-dim tensors (the only legitimate kind that could appear here).
        def _jsonable(o, path="root", bad=None):
            import torch
            if bad is None:
                bad = []
            if isinstance(o, torch.Tensor):
                if o.numel() == 1:
                    return float(o.reshape(()).item())
                bad.append((path, "Tensor%s" % (tuple(o.shape),)))
                return None
            if isinstance(o, dict):
                return {k: _jsonable(v, "%s.%s" % (path, k), bad) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_jsonable(v, "%s[%d]" % (path, i), bad) for i, v in enumerate(o)]
            if isinstance(o, (int, float, str, bool)) or o is None:
                return o
            bad.append((path, type(o).__name__))
            return None
        _payload = {
            "label": a.label,
            "seed": a.seed,
            "steps": a.steps,
            "dims": NAMES,
            "identity": "arg[d] = event_drive_scale*ev_RMS[d] "
                        "+ pe_proj.weight[d,0]*(pe/pe_reference_scale) "
                        "+ pe_proj.bias[d]",
            "sign_convention": "ALL values are SIGNED. *_absmax is reported "
                               "alongside a signed min/max for every "
                               "dimension precisely because an absmax alone "
                               "cannot say which bound a value sits against.",
            "identity_tolerance": IDENTITY_TOL,
            "identity_worst": IDENTITY_WORST,
            "identity_ok": IDENTITY_WORST["residual"] <= IDENTITY_TOL,
            "dLddrive_note": "captured via retain_grad on the drive tensor "
                              "inside drive_seq; read after loss.backward() "
                              "and before opt.step()",
            "consumers": RDA_CONSUMERS,
            "per_step": ALL_ROWS,
            "variance_steps": [ALL_INFO[s] for s in sorted(ALL_INFO)],
        }
        _bad = []
        _payload = _jsonable(_payload, "root", _bad)
        if _bad:
            say("  *** %d non-JSON field(s), first few: %s"
                % (len(_bad), _bad[:5]))
        json.dump(_payload, open(a.preact_all_out, "w"), indent=1)
        say("wrote %s (%d all-dim rows, %d variance steps, identity worst %.3e %s)"
            % (a.preact_all_out, len(ALL_ROWS), len(ALL_INFO),
               IDENTITY_WORST["residual"],
               "OK" if IDENTITY_WORST["residual"] <= IDENTITY_TOL else "*** FAIL ***"))
    if a.film_out:
        _frows = []
        for _r in FILM_ROWS:
            _frows.append({("step" if k == "step" else k): v
                           for k, v in _r.items() if v is not None or k != "step"})
        json.dump({
            "label": a.label,
            "seed": a.seed,
            "steps": a.steps,
            "dims": NAMES,
            "e_tensor": "film_state = cat([E_start.unsqueeze(1), E_next_seq], "
                        "dim=1) at rda.py:1022, i.e. the (B,T,8) input FiLM "
                        "actually receives. NOT E_next_seq and NOT the drive.",
            "column_convention": "film_scale/film_shift are Linear(8, d_model) "
                                 "so weight is (d_model, 8); the column "
                                 "weight[:, d] reads E_d. weight[d, :] would "
                                 "be the wrong axis.",
            "contribution": "W[:, d] * E[:, d]; mean is the DC/offset part, "
                            "std is the varying part, and their ratio is the "
                            "offset-to-varying measure. All SIGNED.",
            "off_over_var": "None means the varying part is ~0, i.e. that "
                            "channel is a pure constant for this dimension.",
            "axes": "offset_dj = W[j,d]*E_mean_d and varying_dj = "
                    "W[j,d]*E_std_d are reduced over POSITIONS ONLY, then "
                    "aggregated over units j by RMS. Positions and units are "
                    "NEVER flattened together in offset_rms/varying_rms. The "
                    "FLATTENED_* fields are the earlier, WRONG reduction over "
                    "(d_model, B, T) together, retained for provenance.",
            "ratio_identity": "offset_rms/varying_rms = E_mean/E_std exactly; "
                              "the column factor ||W[:,d]||_rms cancels.",
            "reconstruction_tolerance": RECON_TOL,
            "reconstruction": RECON,
            "reconstruction_ok": all(
                max(r["fp64_resid_scale"], r["fp64_resid_shift"]) <= RECON_TOL
                for r in RECON) if RECON else None,
            "reconstruction_ok_highest": all(
                max(r["highest_resid_scale"],
                    r["highest_resid_shift"]) <= RECON_TOL
                for r in RECON) if RECON else None,
            "reconstruction_ok_forward_settings": all(
                max(r["forward_settings_resid_scale"],
                    r["forward_settings_resid_shift"]) <= RECON_TOL
                for r in RECON) if RECON else None,
            "rows": _frows,
        }, open(a.film_out, "w"), indent=1)
        say("wrote %s (%d steps, %d dims each)"
            % (a.film_out, len(FILM_ROWS), len(NAMES)))
        if RECON:
            _w64 = max(max(r["fp64_resid_scale"], r["fp64_resid_shift"])
                       for r in RECON)
            _whi = max(max(r["highest_resid_scale"], r["highest_resid_shift"])
                       for r in RECON)
            say("  A: fp64-from-forward-clones worst %.3e  (tol %.0e) %s"
                % (_w64, RECON_TOL, "OK" if _w64 <= RECON_TOL else "FAIL"))
            say("  B: highest/no-tf32 F.linear  worst %.3e  (tol %.0e) %s"
                % (_whi, RECON_TOL, "OK" if _whi <= RECON_TOL else "FAIL"))
            _wtf = max(max(r["forward_settings_resid_scale"],
                           r["forward_settings_resid_shift"])
                       for r in RECON)
            say("  B': reproduce the FORWARD's own settings  worst %.3e  "
                "(tol %.0e) %s"
                % (_wtf, RECON_TOL,
                   "OK  <-- residual was TF32, not a decomposition error"
                   if _wtf <= RECON_TOL else "FAIL"))
            say("  flags restored: fp64=%s  highest=%s  forward=%s"
                % (all(r["flags_restored"] for r in RECON),
                   all(r["flags_restored"] for r in RECON),
                   all(r["flags_restored_after_tf32"] for r in RECON)))
            say("  C: E identical at check: %s"
                % ("yes" if all(r["E_equal"] and r["E_data_ptr_match"]
                                for r in RECON) else "NO -- see file"))
    if a.ablate and ABLATION is not None:
        _ao = a.ablate_out or ((a.out[:-5] + "_ablation.json")
                               if a.out and a.out.endswith(".json")
                               else "ablation.json")
        json.dump(ABLATION, open(_ao, "w"), indent=1)
        say("wrote %s (%d conditions, noise floor %.3e)"
            % (_ao, len(ABLATION["results"]), ABLATION["noise_floor_abs"]))

    if _log is not None:
        _log.close()


if __name__ == "__main__":
    main()
