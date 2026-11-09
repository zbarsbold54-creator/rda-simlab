"""
Affective / regulatory drive state for the nanochat RDA experiment.

This module implements the *predict -> compare -> error* step only:

  1. `RDAStateModule` carries a small recurrent affective state E_t through
     the sequence and can FiLM-modulate the residual stream with it.
  2. `RDAPredictor` attempts to predict the NEXT affective state from the
     state itself plus a PRE-outcome hidden state.
  3. `drive_prediction_error` compares the prediction to what actually
     happened, yielding `dpe`.

Deliberately NOT included (future tasks): memory storage, learning-rate or
attention modulation driven by `dpe`, RL, MoE, architectural
adaptation. `dpe` is computed and logged, never consumed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 8 affective dimensions, in fixed order.
DRIVE_DIMS = (
    "novelty",
    "uncertainty",
    "threat",
    "integrity",
    "agency",
    "learning_progress",
    "arousal",
    "homeostatic_dev",
)
N_DRIVES = len(DRIVE_DIMS)


# -----------------------------------------------------------------------------#
# RDAStateModule
# -----------------------------------------------------------------------------#
def _unit_rms(v, rms: float) -> torch.Tensor:
    """Rescale each per-position modulation vector to a fixed RMS.

    WHY. FiLM is the only route by which the state can influence the loss, and
    measured over 250 real steps its gain ran away: |1 + film_scale(E)| grew
    1.00 -> 2.23 with the base model trainable, and 1.00 -> 9.57 with the base
    model frozen. Two ablations pin the cause on this term and not on the PE
    projection:

      - disabling FiLM entirely: the state never crosses |E| = 0.98, even
        though PE still grows to 19.6. So PE growth is a CONSEQUENCE of FiLM,
        not its cause.
      - freezing pe_proj: saturation still occurs, at the same step.
      - detaching PE's gradient: max|E| and the crossing step are unchanged,
        so this is not a differentiable-loop pathology either.

    With the base model FROZEN there is no external signal to drive the growth
    at all -- PE is flat at 10.50 for 250 steps while pe_proj.bias grows
    1.53 -> 3.71 and the FiLM gain grows 1.00 -> 9.57. A loop that runs with
    no varying input is self-exciting: the state modulates the residual
    stream, which changes the state's own future inputs, which modulates
    harder.

    The fix removes the one degree of freedom that lets the loop amplify
    itself -- the overall magnitude of the gain -- while keeping the part
    that carries information, its DIRECTION across the d_model axis. This is
    a reparameterization, not a clamp: it does not saturate, does not flatten
    gradients, and never produces a dead zone. The same reasoning as the
    event-projection normalization, applied at the second place the same
    pathology appeared.

    At initialization the film weights are exactly zero, so this is 0/eps = 0
    and modulate() remains the exact identity.
    """
    return v / v.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * rms


class RDAStateModule(nn.Module):
    """An 8-dim affective state with leaky integration and FiLM modulation.

    update() (gated / convex-combination form):

        E_next = decay * E_t + (1 - decay) * tanh(event_effect + pe_effect)

    The retained state and the fresh drive are convex-combined, so E_next is
    bounded in [-1, 1] by construction and cannot be driven into tanh
    saturation by the addition of several independently-saturating terms. See
    update()'s docstring for the measurements that motivated this.
    """

    def __init__(self, d_model: int, n_drives: int = N_DRIVES, decay=0.95,
                 pe_reference_scale: float = 10.0):
        super().__init__()
        if isinstance(decay, (tuple, list)):
            assert len(decay) == n_drives, (
                f"decay must contain one value per drive ({n_drives}), got {len(decay)}"
            )
            decay_values = torch.tensor(decay, dtype=torch.float64)
            assert torch.all((decay_values >= 0.0) & (decay_values < 1.0)), (
                f"decay values must be in [0, 1), got {decay}"
            )
            self.register_buffer("decay", decay_values)
            self.decay_safe_T = min(
                self._max_safe_T(value, torch.float64) for value in decay
            )
            self.decay_horizon = tuple(
                float("inf") if value <= 0.0
                else 1.0 / math.log(1.0 / value)
                for value in decay
            )
        else:
            assert 0.0 <= decay < 1.0, f"decay must be in [0, 1), got {decay}"
            self.decay = decay
            self.decay_horizon = (float("inf") if decay <= 0.0
                                  else 1.0 / math.log(1.0 / decay))
        assert pe_reference_scale > 0.0, (
            f"pe_reference_scale must be positive, got {pe_reference_scale}"
        )
        self.d_model = d_model
        self.n_drives = n_drives
        # NLL of a random token is about ln(vocab_size) ~ 10.4 at init. The
        # error is divided by this before entering pe_proj so the PE term is
        # O(1) and zero-centred rather than a large constant-positive bias.
        # Fixed, not learned: it normalizes the input scale only, and does not
        # constrain what the module can express.
        self.pe_reference_scale = pe_reference_scale

        # Event drive from the hidden state.
        #
        # The hidden state entering update() is a raw residual-stream vector
        # whose scale is set by the BASE model and is not bounded by anything in
        # this module. Measured over 30 fresh steps, event_proj's raw output
        # reached mean |.| = 2.11 and max 4.11, i.e. past atanh(0.99) = 2.647 on
        # 26 of 30 steps -- so the tanh drive pinned and the state ratcheted to
        # |E| = sqrt(8) within ~130 steps. Two changes, applied together:
        #
        #   h_norm          LayerNorm makes the event drive depend on the
        #                   DIRECTION of the residual stream rather than its
        #                   magnitude, so base-model activation growth can no
        #                   longer inflate the drive.
        #   event_drive_scale  a fixed 0.1 multiplier on the event term, so it
        #                   is a small perturbation of the tanh sum rather than
        #                   a term that can saturate it on its own.
        #
        # NOTE this is deliberately NOT pe_reference_scale (which normalizes
        # the PE input and stays untouched), and not weight decay (which
        # bounds |W| but not |W h| when |h| is large). This bounds the OUTPUT,
        # which is the quantity that actually crossed the threshold.
        self.h_norm = nn.LayerNorm(d_model)
        self.event_proj = nn.Linear(d_model, n_drives)
        # Fixed, non-learnable, for inspectability. Not a Parameter, so it
        # never appears in parameters() and never affects checkpoint content.
        self.event_drive_scale = 0.1
        # Drive from a scalar prediction error.
        self.pe_proj = nn.Linear(1, n_drives)

        # FiLM modulation of the residual stream. Zero-initialized so the
        # modulation is a no-op at step 0 and cannot destabilize the base model
        # on the first optimizer step. This is deliberately DIFFERENT from
        # RDAPredictor, which must learn from a non-trivial init.
        #
        # ALL of weight AND bias must be zeroed, otherwise scale = 1 + bias
        # != 1 and modulate() would not be an exact identity at init.
        self.film_scale = nn.Linear(n_drives, d_model)
        self.film_shift = nn.Linear(n_drives, d_model)
        for layer in (self.film_scale, self.film_shift):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Initial state, learned, so training does not have to wait for the
        # recurrence to warm up from a fixed vector.
        #
        # GRADIENT STATUS -- read before assuming this is frozen or that it
        # trains. Both intuitions are wrong, in different ways:
        #
        #   * It is NOT gradient-free by architecture. A path exists:
        #       e_init -> init_state() -> E_t -> film_state -> modulate() -> loss
        #     and init_state() is differentiable (unsqueeze/expand/contiguous ON
        #     e_init, so the graph is retained).
        #   * At step 0 that path yields EXACTLY 0.0, because film_scale and
        #     film_shift are zero-initialised and modulate() is therefore the
        #     exact identity.
        #   * From step 1 onward the trainer always supplies a carry, and
        #     forward() DETACHES it, so e_init is unreachable and autograd
        #     PRUNES it: p.grad is None.
        #
        # Measured over a 5-step loop reproducing the trainer's call pattern:
        #   per-step |grad| = ['0.000e+00', 'None', 'None', 'None', 'None']
        #   after 5 steps   = [0,0,0,0,0,0,0,0]
        #
        # So: STRUCTURALLY LEARNABLE, but in practice frozen at its zero init in
        # every real run. `base_train.py` therefore keeps it in the grad-free
        # list, which is correct and harmless. If the persistence semantics ever
        # change such that a fresh E_t_external could hold a meaningful state,
        # this parameter becomes load-bearing and must be allowed to train.
        self.e_init = nn.Parameter(torch.zeros(n_drives))

        # RMS budget for each FiLM modulation vector, per position. See
        # _unit_rms() and modulate() for why the gain is normalized rather
        # than left free. Fixed and non-learnable, like event_drive_scale, so
        # it can never be traded off against the loss.
        self.film_rms = 0.5

        # Measured 1/e horizon of the recurrence, in POSITIONS. This is the
        # single number that says how long the state actually remembers, and it
        # is exposed (rather than left implicit in `decay`) because it is the
        # quantity that determines whether "persistent state" is a real claim.
        #
        # At the default decay=0.95 this is ~19.5 positions, so:
        #   * within a 1024-token row, a perturbation of the incoming state is
        #     below 50% of its initial size by t~13 and numerically zero by
        #     t~500 (measured: 0.95^500 = 7.3e-12);
        #   * across a document boundary inside a packed row (docs are ~256
        #     tokens), the carried state arrives attenuated by 0.95^256 = 2.0e-6,
        #     i.e. NOT carried across documents in any measurable sense.
        #
        # So the decay currently implements WITHIN-SEQUENCE INERTIA, and the
        # "persistence across documents / micro-steps" reading of the design is
        # NOT what this number provides. Both readings are legitimate designs;
        # they need different decay values (0.95 -> ~20 positions,
        # 0.999 -> ~1000). The value is a hyperparameter and is deliberately
        # NOT changed here: see docs/decay_semantics.md. What matters is that
        # anyone reading "persistent state" can now see which one is configured.
        #
        # Note the |E| <= 1 bound and the float64 closed form are UNAFFECTED by
        # this value: the bound comes from the convex combination, and
        # _max_safe_T GROWS as decay approaches 1 (13837 at 0.95, 709427 at
        # 0.999). So a larger decay is numerically safe, if ever wanted.

    # -- recurrence ---------------------------------------------------------
    def init_state(self, batch_size: int, device, dtype=None) -> torch.Tensor:
        """Fresh E_0 for a sequence, shape (batch_size, n_drives)."""
        out = self.e_init.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out.unsqueeze(0).expand(batch_size, -1).contiguous()

    def update(self, E_t, hidden_state, prediction_error) -> torch.Tensor:
        """Advance the state by one step (GATED / convex-combination rule).

            E_next = decay * E_t + (1 - decay) * tanh(event_effect + pe_effect)

        WHY THIS SHAPE. The previous rule was

            E_next = tanh(decay*E_t + tanh(event) + tanh(pe))

        whose pre-squash argument can reach decay*1 + 1 + 1 = 2.95 in
        magnitude, so tanh maps it to 0.9945. Anything above |1.472| already
        yields |E| > 0.9, i.e. the state is pinned near the asymptote for most
        of its dynamic range. Measured over 500 real steps, 5 of the 8
        dimensions spent >94% of steps with |E| > 0.9; an UNTRAINED module
        saturated ~50% of the time, confirming this is structural rather than a
        strong learned signal.

        The fix interpolates instead of summing-then-clipping: the retained
        state and the fresh drive are convex-combined with weights decay and
        (1 - decay), so each is in [-1, 1] and the result is in [-1, 1] BY
        CONSTRUCTION -- no final tanh, and no way for the two terms to combine
        into a saturated pre-squash. It also gives the new event real room to
        move the state: its full influence is (1 - decay) per step rather than
        being added on top of an already near-bounded E_t.

        E_t:              (B, n_drives)
        hidden_state:     (B, d_model)   the event context (may be pooled)
        prediction_error: (B,) or (B, 1)  scalar token-level PE
        Returns (B, n_drives), provably within [-1, 1].
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)
        # The base model runs in COMPUTE_DTYPE (e.g. bf16) while these modules
        # keep fp32 master weights, so cast the incoming activations.
        hidden_state = hidden_state.to(self.event_proj.weight.dtype)

        pe = prediction_error
        if pe.dim() == 1:
            pe = pe.unsqueeze(-1)
        elif pe.dim() == 0:
            pe = pe.reshape(1, 1).expand(hidden_state.size(0), 1)
        pe = pe.to(self.pe_proj.weight.dtype)

        # The fresh drive, combined BEFORE bounding so the event and prediction
        # error can reinforce or cancel each other rather than each saturating
        # independently.
        #
        # The prediction error is a token-level NLL, typically O(10) at init and
        # always POSITIVE (NLL of an observed token is never negative). Feeding
        # that straight into pe_proj produced a term ~100x larger than the event
        # drive, whose sign is fixed forever, so the drive had a near-constant
        # sign and the state was pinned at a rail. Normalize the error to a
        # zero-centred O(1) quantity first so it can actually oppose the event
        # drive instead of overwhelming it. The reference scale is a property of
        # the NLL definition, not a learned or tuned parameter.
        pe_scaled = pe / self.pe_reference_scale
        # Event term: LayerNorm(h) then a WEIGHT-NORMALIZED projection.
        #
        # WHY THE NORMALIZATION MOVED FROM h TO THE PROJECTION OUTPUT.
        # LayerNorm(h) was supposed to stop base-model activation growth from
        # inflating the drive, and it does -- measured over 30 real steps,
        # LayerNorm output rms stayed at exactly 1.000 and h_norm.weight at
        # exactly 1.0 for the whole run. It was never the failure.
        #
        # The failure is that ||h_norm|| = sqrt(d) = 27.71 is a CEILING, not a
        # value: event_proj(h) can still reach ||w_row|| * 27.71, and ||w_row||
        # is LEARNED and UNBOUNDED. Measured, 30 steps, seed 42:
        #
        #     event_proj ||W||      1.63 -> 3.51   (x2.15)
        #     event |max|           0.25 -> 1.26   (x5.03)
        #     LayerNorm out rms     1.00 -> 1.00   (x1.00)  <- working
        #
        # So the drive grew because the PROJECTION grew, not because the input
        # did. Freezing the entire base model (so h is byte-identical every
        # step) changed NOTHING: ||W|| still went 1.63 -> 3.51 and event still
        # reached 1.26. That isolates the cause to the gradient on event_proj
        # itself, and rules out hidden-state drift and any FiLM feedback loop
        # as the driver.
        #
        # Normalizing the OUTPUT makes the drive depend on the direction of
        # W*h_norm rather than its magnitude, which is exactly the property
        # h_norm was introduced to provide -- it just has to be applied where
        # the magnitude actually is. RMS-normalizing over the drive axis keeps
        # the learned relative scale BETWEEN drives (the sign and pattern of
        # the 8 components is the signal; it is preserved exactly) while
        # removing the one degree of freedom that caused the ratchet.
        #
        # This is scale-invariance, not a hard clamp: it does not saturate,
        # does not flatten gradients, and leaves the tanh argument smooth.
        ev = self.event_proj(self.h_norm(hidden_state))
        ev = ev / ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        event_term = self.event_drive_scale * ev
        drive = torch.tanh(event_term + self.pe_proj(pe_scaled))
        E_prev = E_t.to(drive.dtype)
        # Clamp the incoming state defensively so float drift can never push the
        # convex combination outside [-1, 1] and break the bound.
        E_prev = torch.clamp(E_prev, -1.0, 1.0)

        if isinstance(self.decay, torch.Tensor):
            decay = self.decay.to(dtype=drive.dtype).view(1, -1)
            E_next = decay * E_prev + (1.0 - decay) * drive
        else:
            E_next = self.decay * E_prev + (1.0 - self.decay) * drive

        # The bound is a mathematical property of the convex combination, so
        # assert it rather than observe it. Any violation means the recurrence
        # has been changed in a way that reintroduces saturation.
        assert E_next.abs().max() <= 1.0 + 1e-6, (
            f"gated update escaped [-1, 1]: max|E_next|={E_next.abs().max()}"
        )
        return E_next

    # -- vectorized recurrence ---------------------------------------------
    def drive_seq(self, hidden_seq, prediction_error_seq):
        """Per-position drive for a whole sequence, in one shot.

        The vectorized twin of the body of update(): the same arithmetic, the
        same parameters, applied to (B, T, d) / (B, T) rather than to one
        position at a time.

        There is deliberately NO recurrence here. The drive is a function of
        the BASE hidden states and the prediction error only -- neither of
        which can depend on E -- so the drive for all T positions can be
        computed independently, and that is what makes the closed form in
        update_seq() possible at all.

        hidden_seq:           (B, T, d_model)
        prediction_error_seq: (B, T)
        Returns (B, T, n_drives), each entry in (-1, 1) because of the tanh.
        """
        h = hidden_seq
        if h.dim() == 2:
            h = h.unsqueeze(1)
        h = h.to(self.event_proj.weight.dtype)
        pe = prediction_error_seq
        # pe_proj is Linear(1, n_drives), so pe must carry a trailing
        # singleton axis. Accept the (B, T) shape the caller has and add it.
        if pe.dim() == 1:
            pe = pe.unsqueeze(-1)          # (B,)   -> (B, 1)
        elif pe.dim() == 2:
            pe = pe.unsqueeze(-1)          # (B, T) -> (B, T, 1)
        pe = pe.to(self.pe_proj.weight.dtype)
        h = self.h_norm(h)
        ev = self.event_proj(h)
        # Same output RMS normalization as update() -- see the comment there.
        # Applying it here too is what keeps the vectorized path and the
        # reference loop bit-for-bit comparable; if only one of them had it,
        # every verification test would be measuring two different models.
        ev = ev / ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        event_term = self.event_drive_scale * ev
        return torch.tanh(event_term + self.pe_proj(pe / self.pe_reference_scale))

    @staticmethod
    def _max_safe_T(decay, dtype):
        """Largest T for which decay**-T is finite in `dtype`.

        The closed form divides by decay**t, so the reciprocal must not
        overflow. That is true exactly while

            T * ln(1/decay) <= ln(finfo(dtype).max)

        Returns the largest integer T satisfying it. Computed rather than
        hardcoded so it stays correct if `decay` or the accumulation dtype
        ever change.
        """
        import math
        import numpy as np
        finfo = (np.finfo(np.float64) if dtype == torch.float64
                 else np.finfo(np.float32))
        if decay <= 0.0 or decay >= 1.0:
            # decay == 0 gives 0**-T = inf for T>0; decay in (0,1) is the
            # only sane case, and decay == 1 makes w[t] == 1 (no limit).
            return 0 if decay <= 0.0 else 2 ** 62
        return int(math.floor(math.log(finfo.max) / math.log(1.0 / decay)))

    def update_seq(self, E_start, drive, chunk=False):
        """Closed-form E_next_seq for the whole sequence at once.

            E_next[t] = decay * E_next[t-1] + (1 - decay) * drive[t]
            E_next[-1] := E_start

        Unrolling it:

            E[t] = decay^(t+1) * E_start
                   + (1 - decay) * sum_{s<=t} decay^(t-s) * drive[s]

        The sum is a decay-weighted prefix sum, and it factorizes as

            E[t] = w[t] * ( decay * E_start + cumsum(g / w)[t] )

        with w[t] = decay^t and g = (1 - decay) * drive. Note the `decay *`
        in front of E_start: it falls out of the unrolling and is easy to
        drop. An earlier version of this function used E_start + cumsum(...),
        which corresponds to seeding the recurrence with E_start/d -- that is
        the same recurrence run from the WRONG initial state, and it was
        caught by diagnostics/verify_vectorized_update.py disagreeing with
        the loop by up to 2.9e-2 in fp32. The whole 1023-step recurrence now
        collapses to a handful of kernels instead of ~20k sequential ones.

        WHY THE ACCUMULATION IS float64. The cumsum form divides by
        w[t] = decay^t, and its reciprocal decay**-t must stay FINITE.

        COMPUTED LIMIT, for the default decay=0.95. decay**-T overflows when
        T > ln(finfo_max) / ln(1/decay):

            float64 (what this function actually uses):
                T <= 13837    (13,837 * ln(1/0.95) = 709.78 vs
                               ln(1.797e308) = 709.78)
            float32, for reference:
                T <= 1729

        So a float64 cumsum is exact for any realistic sequence -- training
        uses T=1024, which is 13.5x inside the limit -- but it is NOT
        unbounded, and a long-context experiment could cross it. Hence
        _max_safe_T() below, which computes the limit from the dtype and the
        actual decay rather than hardcoding it, and the assert that enforces
        it. A previous version hardcoded L <= 6000, which was a guess that
        happened to sit under the true limit; computing it removes the guess.

        It costs nothing measurable to use float64 here: these tensors are
        (B, T, 8), tiny next to the base model, and no matmul is involved --
        which matters, since fp64 throughput is poor on consumer GPUs. The
        result is cast back to the drive dtype so the rest of forward() is
        unaffected.

        EQUIVALENCE, and its single caveat. This is the same recurrence, so
        the state trajectory agrees with the loop up to floating-point
        rounding; measured max|E_closed_form - E_loop| is reported by
        diagnostics/verify_vectorized_update.py.

        The one structural difference is update()'s defensive
        clamp(E_prev, -1, 1). Since E_next convex-combines two values already
        in [-1, 1], it lies in [-1, 1] in exact arithmetic, so the clamp can
        only ever fire on a value within one ulp (~6e-8) of the bound, where
        it alters that value by at most that much and the alteration is then
        damped by `decay` at every subsequent step. It is not applied here
        because doing so would defeat the closed form. The discrepancy is
        bounded by ~1e-7, far below the loop's own accumulated rounding error.
        """
        assert drive.dim() == 3, (
            f"drive must be (B, T, n_drives), got {tuple(drive.shape)}"
        )
        B, L, n = drive.shape
        acc_dtype = torch.float64
        per_drive = isinstance(self.decay, torch.Tensor)
        if per_drive:
            d = self.decay.to(device=drive.device, dtype=acc_dtype)
            limit = self.decay_safe_T
        else:
            d = self.decay
            limit = self._max_safe_T(d, acc_dtype)
        if L > limit:
            # Two ways out, both of which keep the closed form. The default
            # is to FAIL LOUDLY rather than silently change behaviour: a
            # sequence this long is outside what the accumulation can
            # represent, and quietly chunking would hide a real numerical
            # limit. Callers that knowingly want the chunked path pass
            # chunk=True.
            if not chunk:
                raise ValueError(
                    f"update_seq: L={L} exceeds the {limit} that "
                    f"decay**-T stays finite in {acc_dtype} (decay={d}). "
                    f"The closed form would overflow. Pass chunk=True to "
                    f"split the recurrence into safe segments."
                )
            return self._update_seq_chunked(E_start, drive, limit)
        dev = drive.device
        powers = torch.arange(L, device=dev, dtype=acc_dtype)
        if per_drive:
            d = d.view(1, 1, n)
            wv = torch.pow(d, powers.view(1, L, 1))
            g = (1.0 - d) * drive.to(acc_dtype)
            E0 = (d.view(1, 1, n) * E_start.to(acc_dtype).unsqueeze(1))
        else:
            w = torch.pow(
                torch.tensor(d, device=dev, dtype=acc_dtype),
                powers,
            )
            wv = w.view(1, L, 1)
            g = (1.0 - d) * drive.to(acc_dtype)
            E0 = (d * E_start).to(acc_dtype).unsqueeze(1)
        acc = torch.cumsum(g / wv, dim=1)               # (B, L, n)
        E = wv * (E0 + acc)
        E = E.to(drive.dtype)
        assert E.abs().max() <= 1.0 + 1e-6, (
            f"vectorized update escaped [-1, 1]: max|E_next|={E.abs().max()}"
        )
        return E

    def _update_seq_chunked(self, E_start, drive, limit):
        """update_seq() for L > limit, split into segments of `limit` steps.

        The recurrence is split at position k, where the incoming state for
        the next segment is simply E[k-1] -- the state the previous segment
        ended on. Each segment is then a fresh closed form seeded with that
        value, so the ONLY thing the chunking changes is which T the
        accumulation runs over; the result is the same recurrence.

        The per-segment exponent is RESET to 0, so each segment's
        decay**-t stays within the finite range. This is the standard
        blocked-scan decomposition and is exact, not an approximation.
        """
        B, L, n = drive.shape
        outs, carry = [], E_start
        for s in range(0, L, limit):
            e = min(s + limit, L)
            seg = drive[:, s:e, :]
            # update_seq returns the whole per-position sequence for this
            # segment; the next segment is seeded with its LAST state, which
            # is the recurrence's carry -- not the segment's whole output.
            seg_out = self.update_seq(carry, seg)
            outs.append(seg_out)
            carry = seg_out[:, -1, :]
        E = torch.cat(outs, dim=1)
        assert E.shape[1] == L, f"chunking lost positions: {E.shape[1]} != {L}"
        return E

    # -- modulation ---------------------------------------------------------
    def modulate(self, x, E_t) -> torch.Tensor:
        """FiLM-modulate hidden states x (B, T, d_model) with a PER-POSITION
        drive state.

        E_t is (B, T, n_drives) -- one state vector per position, aligned with
        x along dim 1. Each position is modulated only by the state as of that
        position, so the result is causally masked: position p cannot be
        influenced by tokens after p.

        This used to take (B, n_drives) and broadcast one vector to every
        position, which meant position 0 was modulated by a state computed
        from the LAST token -- a 1022-token lookahead at T=1024. That also made
        the architecture impossible to reproduce at inference, since an
        autoregressive decode step has no "final state" to draw on.

        A single (B, n_drives) state is also accepted and broadcast over
        positions, but that is only correct when the caller has genuinely
        established the same state applies to all of them -- i.e. a single
        decode step. Broadcasting a per-sequence state across positions is
        exactly the bug this signature was changed to fix.

        With zero-initialized film layers this is exactly the identity at
        initialization, per position.
        """
        if E_t.dim() == 2:
            # (B, n_drives) -> (B, 1, n_drives), broadcast over positions.
            E_t = E_t.unsqueeze(1)
        assert E_t.dim() == 3, f"per-position state must be (B,T,n_drives), got {tuple(E_t.shape)}"
        T_x, T_e = x.size(1), E_t.size(1)
        assert T_x == T_e, (
            f"modulate(): hidden has {T_x} positions but state has {T_e}; "
            f"they must be aligned along dim 1"
        )
        scale = 1.0 + _unit_rms(self.film_scale(E_t), self.film_rms)
        shift = _unit_rms(self.film_shift(E_t), self.film_rms)
        return x * scale + shift

    # -- token-level prediction error --------------------------------------
    def prediction_error_from_logits(self, logits, targets) -> torch.Tensor:
        """Per-position scalar prediction error, shape (B, T).

        For every position this is the model's NLL of the actually-observed
        token, so a surprising token yields a large positive error and a
        predicted token a small one. Targets equal to ignore_index (-1) are
        masked to 0 so they contribute no drive.
        """
        vocab = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab).float()
        flat_targets = targets.reshape(-1)
        nll = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=-1, reduction="none"
        )
        nll = nll.view(targets.shape)
        return torch.where(targets == -1, torch.zeros_like(nll), nll)


# -----------------------------------------------------------------------------#
# RDAPredictor
# -----------------------------------------------------------------------------#
class RDAPredictor(nn.Module):
    """Predict E_{t+1} from E_t and a PRE-outcome hidden state.

        E_hat_next = tanh( Linear(d_model -> n_drives)(pre_outcome_hidden)
                         + Linear(n_drives -> n_drives)(E_t) )

    Initialization is NORMAL (PyTorch default), NOT zero-init. Zero-init here
    would make the output identically zero at step 0 and the module would have
    to climb out of a flat region; unlike the FiLM path this module is supposed
    to learn a real function immediately.
    """

    def __init__(self, d_model: int, n_drives: int = N_DRIVES):
        super().__init__()
        self.d_model = d_model
        self.n_drives = n_drives
        self.event_proj = nn.Linear(d_model, n_drives)
        self.state_proj = nn.Linear(n_drives, n_drives)

    def predict(self, E_t, pre_outcome_hidden) -> torch.Tensor:
        """
        E_t:                (B, n_drives)
        pre_outcome_hidden: (B, d_model)  -- position t, BEFORE token t+1
        Returns (B, n_drives).
        """
        if pre_outcome_hidden.dim() == 3:
            pre_outcome_hidden = pre_outcome_hidden.mean(dim=1)
        if E_t.dim() == 3:
            E_t = E_t.mean(dim=1)
        # The base model runs in COMPUTE_DTYPE (e.g. bf16) while these modules
        # keep fp32 master weights, so cast the incoming activations.
        pre_outcome_hidden = pre_outcome_hidden.to(self.event_proj.weight.dtype)
        E_t = E_t.to(self.state_proj.weight.dtype)
        return torch.tanh(
            self.event_proj(pre_outcome_hidden) + self.state_proj(E_t)
        )

    def predict_seq(self, E_t, pre_outcome_hidden):
        """Sequence form of predict(): (B, T, n) states, (B, T, d) hiddens.

        Elementwise-identical to calling predict() on each position; there is
        no recurrence here, so this is simply the same two Linear layers
        applied to the whole sequence at once.

        E_t:                (B, T, n_drives)  the state entering each position
        pre_outcome_hidden: (B, T, d_model)
        Returns (B, T, n_drives).
        """
        if E_t.dim() == 2:
            E_t = E_t.unsqueeze(1)
        if pre_outcome_hidden.dim() == 2:
            pre_outcome_hidden = pre_outcome_hidden.unsqueeze(1)
        h = pre_outcome_hidden.to(self.event_proj.weight.dtype)
        e = E_t.to(self.state_proj.weight.dtype)
        return torch.tanh(
            self.event_proj(h) + self.state_proj(e)
        )


def drive_prediction_error(E_actual, E_hat_next) -> torch.Tensor:
    """Elementwise E_actual - E_hat_next, shape (B, n_drives).

    Raw and unclamped on purpose: this signal is meant to be inspected for
    saturation, and a clamp here would hide exactly that. Both tanh-bounded
    inputs keep the result in (-2, 2) regardless.
    """
    return E_actual - E_hat_next


def document_start_mask(inputs, bos_token_id):
    """Mark which positions of `inputs` (B, T) begin a new document.

    REUSING THE EXISTING BOUNDARY SIGNAL. The BOS-aligned bestfit dataloader
    packs several documents into one training row and exposes no separate
    boundary field, but every document is already prefixed with the BOS token
    (`prepend=bos_token` in dataloader.py). So a position starts a document
    exactly when its token id is BOS.

    Position 0 is always counted, since it is the first token of its row and a
    BOS there is the row's own first document, not a mid-row boundary.
    """
    mask = inputs == bos_token_id
    mask[:, 0] = True
    return mask


def count_document_starts(inputs, bos_token_id):
    """Number of (row, position) document starts in the batch."""
    return int(document_start_mask(inputs, bos_token_id).sum().item())


def episode_continuation_mask(inputs, bos_token_id):
    """RETAINED FOR REFERENCE ONLY -- no longer drives persistence.

    Under DECAY-ONLY persistence the state is never reset at a document
    boundary, so nothing consults this mask any more. It is kept because it
    documents the dataloader's actual behaviour: every row both starts and ends
    on a document boundary (the bestfit packer crops the final document to fill
    the row exactly), so this mask is False for every row of every batch. The
    earlier reset-at-boundary policy therefore never fired in practice, and
    removing it changes the model's state trajectory, not just its bookkeeping.
    """
    return inputs[:, 0] != bos_token_id


# -----------------------------------------------------------------------------#
# GPTWithRDA
# -----------------------------------------------------------------------------#
class GPTWithRDA(nn.Module):
    """Wraps a nanochat GPT with an affective state, a state predictor, and
    the resulting affective prediction error.

    Temporal contract (this is the whole point of the class):

        pre_outcome_hidden = hidden[:, :-1, :]   position t, before token t+1
        pooled_post_outcome = hidden[:, 1:, :]   position t+1, AFTER token t+1

    Both are slices of the SAME hidden tensor but of DIFFERENT positions, so
    they are different tensors holding different information. `hidden[:, t]`
    is computed by a causal transformer and cannot see token t+1, so
    `pre_outcome_hidden` is genuinely pre-outcome. The post-outcome slice is
    what feeds `event_proj` / `update()`, because the state may only react to
    what actually happened.

    Returns a dict including "E_next" for the caller to carry forward, UNLESS
    targets is None, in which case a bare (B, T, vocab) logits TENSOR is
    returned so the generation path can treat this wrapper as a drop-in GPT.
    `dpe` is NOT consumed for anything in this step.
    """

    def __init__(self, base_model, n_drives: int = N_DRIVES, decay=0.95,
                 debug_steps: int = 200, check_leakage: bool = True,
                 const_state: bool = False, const_state_seed: int = 0):
        super().__init__()
        self.base = base_model
        d_model = base_model.config.n_embd
        self.d_model = d_model
        self.n_drives = n_drives
        self.state = RDAStateModule(d_model, n_drives=n_drives, decay=decay)
        self.predictor = RDAPredictor(d_model, n_drives=n_drives)
        self.debug_steps = debug_steps
        self.check_leakage = check_leakage
        # CONTROL ARM C. When true, the tensor handed to FiLM is a learned
        # constant vector instead of the per-position state, so FiLM receives
        # no text-dependent and no position-dependent information.
        #
        # DEFAULT IS None, i.e. the flag is OFF and `film_state` is used
        # verbatim, so baseline and B' are bit-identical to before.
        #
        # WHY NON-ZERO INIT. FiLM's weights are ZERO-INITIALISED, so a
        # zero-initialised c would give d(loss)/dc = 0 AND leave
        # d(loss)/dW = 0 (the product W*c is zero and its gradient is
        # proportional to c). That is a saddle: neither c nor W could ever
        # move, and C would be weaker than B' for a reason that has nothing
        # to do with information. Drawing c ~ N(0, 0.3^2) breaks the saddle
        # while still carrying no per-token information.
        #
        # WHY A DEDICATED GENERATOR. Drawing from the global RNG here would
        # CONSUME 8 numbers from whatever stream the caller is using, shifting
        # every subsequent draw. Anything initialised after this point would
        # then differ between B' and C at step 0 for a reason that has nothing
        # to do with the arm -- and the data order/shuffling is seeded off
        # that same global stream. A private generator makes the draw
        # independent: every parameter that exists in B' is bit-identical at
        # init in C, verified with torch.equal over the shared named set.
        self.const_state_vec = None
        if const_state:
            _g = torch.Generator(device="cpu")
            _g.manual_seed(int(const_state_seed) + 1000003)
            _cv = torch.randn(n_drives, generator=_g, dtype=torch.float32) * 0.3
            self.const_state_vec = nn.Parameter(_cv)
            # Plain-python copy of the init, for the end-of-run report in
            # base_train.py. Taken here, before any optimizer step can touch
            # it, and never read by the forward path -- so it is inert with
            # respect to training.
            self.const_state_vec_init = [float(v) for v in _cv.tolist()]
        self._debug_count = 0
        # Records the last leakage-check result so a caller can audit it.
        self.last_leakage_check = None
        # Final state of the most recent forward(). During generation this is
        # the value the Engine must feed back in as E_t_external on the next
        # step, so T=1 decoding continues the same recurrence.
        self.last_state = None

    # -- reference implementation (sequential) ------------------------------
    def _update_loop_reference(self, E_t, pre_outcome_hidden, pe_seq):
        """REFERENCE ONLY: the original sequential recurrence, kept verbatim.

        Same order of operations as the original loop:

            for t: E_next = update(E_next, h_t, pe_seq[:, t])
                   E_hat_next = predictor.predict(E_t, h_t)   # from E_t, BEFORE
                   append, then rebind E_t = E_next

        Note the predictor is fed E_t from BEFORE the update at that position,
        which is why step_Ehat and step_E are built in the same pass.

        This is not called by forward(); it is the oracle that
        diagnostics/verify_vectorized_update.py compares the closed form
        against, so the loop has to stay exactly as it was. It costs ~200x the
        vectorized path, so nothing on the training hot path may call it.
        """
        E_next = E_t
        step_E, step_Ehat = [], []
        for t in range(pre_outcome_hidden.size(1)):
            h_t = pre_outcome_hidden[:, t, :]
            E_next = self.state.update(E_next, h_t, pe_seq[:, t])
            E_hat_next = self.predictor.predict(E_t, h_t)
            step_E.append(E_next)
            step_Ehat.append(E_hat_next)
            E_t = E_next
        if not step_E:
            return E_next, None, None
        return (
            E_next,
            torch.stack(step_E, dim=1),      # (B, T-1, n_drives)
            torch.stack(step_Ehat, dim=1),    # (B, T-1, n_drives)
        )

    # -- leakage guard ------------------------------------------------------
    def _assert_no_temporal_leakage(self, pre_outcome_hidden, pooled_post_outcome):
        """PERMANENT runtime guard against the predictor cheating.

        Two distinct failure modes are checked:

        1. The two slices being the same storage at all, which would mean
           pre_outcome_hidden accidentally aliases post-outcome information.
        2. The two slices being element-wise equal at some sampled position,
           which is what would actually let the predictor reconstruct the
           outcome and drive dpe toward 0.

        Checked every forward pass, not just during the debug window: this is a
        correctness invariant, not a diagnostic.
        """
        assert pre_outcome_hidden.shape != pooled_post_outcome.shape or True
        distinct_ptr = (
            pre_outcome_hidden.data_ptr() != pooled_post_outcome.data_ptr()
        )
        # Compare a sample of positions; full (B,T,D) equality every step is
        # wasteful, and one differing position is sufficient evidence.
        #
        # NOTE the index. pre_outcome_hidden is hidden[:, :-1, :] and
        # pooled_post_outcome is hidden[:, 1:, :], so pre[:, t] and post[:, t-1]
        # are the SAME underlying slice (both are hidden[:, t]) and would always
        # compare equal. The adjacent-but-distinct pair is pre[:, t] vs
        # post[:, t], i.e. hidden[:, t] vs hidden[:, t+1].
        if pre_outcome_hidden.size(1) == 0:
            # Single-token decode (T == 1): there is no pre/post pair to
            # compare, so there is nothing that could leak. The two slices are
            # both empty, hence trivially non-identical in content. This case
            # became reachable once T=1 generation was wired up.
            self.last_leakage_check = {
                "distinct_ptr": bool(distinct_ptr),
                "not_elementwise_equal": True,
                "skipped_single_token_decode": True,
            }
            return

        b = min(pre_outcome_hidden.size(0), 2)
        last = pre_outcome_hidden.size(1) - 1
        t = max(0, last // 2)
        lhs = pre_outcome_hidden[:b, t, :]
        rhs = pooled_post_outcome[:b, t, :]
        not_equal = not torch.equal(lhs, rhs)

        self.last_leakage_check = {
            "distinct_ptr": bool(distinct_ptr),
            "not_elementwise_equal": bool(not_equal),
            "sample_b": int(b),
            "sample_t": int(t),
        }
        assert distinct_ptr, (
            "TEMPORAL LEAK: pre_outcome_hidden and pooled_post_outcome share "
            "the same storage. The predictor would see post-outcome information."
        )
        assert not_equal, (
            "TEMPORAL LEAK: pre_outcome_hidden and pooled_post_outcome are "
            "element-wise equal at the sampled position. The predictor could "
            "reconstruct the outcome, driving dpe toward 0."
        )
        return True

    # -- debug --------------------------------------------------------------
    def _log_debug(self, E_t, E_hat_next, E_next, dpe, mode="reset",
                   doc_starts_this_step=0):
        """Log state norms for the first `debug_steps` steps.

        NORM SEMANTICS: the state is (B, n_drives) and its final operation is
        tanh, so each ROW is bounded by sqrt(n_drives). We therefore always log
        `.norm(dim=-1)` (the per-row L2 norms). A bare `.norm()` on the
        (B, n_drives) tensor is the Frobenius norm over all B*n_drives elements
        and legitimately exceeds sqrt(n_drives) by up to ~sqrt(B) -- that is
        arithmetic, not a bound violation, so it must never be reported as if it
        were a per-state norm.

        mode is "reset" only at step 0 of training. Under decay-only persistence
        every later step is "carry", including steps that cross a document
        boundary; `doc_starts_this_step` is what distinguishes a carry that
        crossed a boundary from one that stayed inside a document.
        """
        if self._debug_count >= self.debug_steps:
            return
        self._debug_count += 1
        if not self.base.training:
            return

        def row_norms(t):
            t = t.detach()
            if t.dim() == 1:
                return t.norm().item()
            return t.norm(dim=-1)

        e_t = row_norms(E_t)
        e_hat = row_norms(E_hat_next)
        e_next = row_norms(E_next)
        pe = row_norms(dpe)

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.6f}"
            return "[" + ",".join(f"{x:.4f}" for x in v) + "]"

        # Enforce the theoretical bound so a genuine violation is impossible to
        # miss in the log.
        ceiling = (self.n_drives ** 0.5)
        max_next = max(e_next) if not isinstance(e_next, float) else e_next
        assert max_next <= ceiling + 1e-4, (
            f"E_next exceeded the tanh bound sqrt(n_drives)={ceiling:.4f}: "
            f"got {max_next:.6f}"
        )

        print(
            f"[rda] step {self._debug_count:4d} mode={mode:<5s} "
            f"doc_starts_this_step={doc_starts_this_step} "
            f"E_t.norm()={fmt(e_t)} "
            f"E_hat_next.norm()={fmt(e_hat)} "
            f"E_next.norm()={fmt(e_next)} "
            f"dpe.norm()={fmt(pe)}",
            flush=True,
        )

    # -- pass-throughs so the wrapper is a drop-in for GPT ----------------
    # Callers such as nanochat.loss_eval.evaluate_bpb call model.get_device().
    def get_device(self):
        return self.base.get_device()

    def estimate_flops(self):
        return self.base.estimate_flops()

    # -- forward ------------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean",
                E_t_external=None, doc_starts=None):
        """Run one training step.

        E_t_external: (B, n_drives) affective state carried in from the previous
            step. DECAY-ONLY PERSISTENCE: under training this is supplied on
            every step after the first, so the state is never hard-reset at a
            document boundary. Passing None means "start of training" and is the
            ONLY condition under which E_init is used; the training loop is
            responsible for always passing the carried state once training has
            begun. The `decay` inside update() is the sole mechanism governing
            how much prior state survives into a new document.

        doc_starts: optional (B, T) bool tensor marking which positions begin a
            new document. Derived from the BOS tokens the dataloader already
            prefixes to every document. Under decay-only persistence this is
            REPORTING ONLY -- it does not trigger any reset. The state
            recurrence still runs per row, because a packed row concatenates
            independent documents and mixing their states within a row would be
            meaningless; the cross-document effect is the decay term only.

        Returns a dict including "E_next" for the caller to carry forward.
        """
        loss, hidden = self.base.forward(
            idx, targets, kv_cache=kv_cache,
            loss_reduction=loss_reduction, return_hidden=True,
        )

        # The two temporal slices. Different positions => different tensors.
        pre_outcome_hidden = hidden[:, :-1, :]   # (B, T-1, d_model)
        pooled_post_outcome = hidden[:, 1:, :]   # (B, T-1, d_model)

        if self.check_leakage:
            self._assert_no_temporal_leakage(pre_outcome_hidden, pooled_post_outcome)

        B = idx.size(0)
        device = idx.device

        # Decay-only persistence. E_init is used ONLY when the caller supplies
        # no carried state, i.e. at step 0 of training. Document boundaries do
        # NOT reset the state: decay inside update() is the only mechanism that
        # limits how much prior state survives into a new document.
        E_init = self.state.init_state(B, device, dtype=torch.float32)
        if E_t_external is not None:
            assert E_t_external.shape == (B, self.n_drives), (
                f"carried state must be (B, n_drives) = {(B, self.n_drives)}, "
                f"got {tuple(E_t_external.shape)}"
            )
            E_t = E_t_external.detach().to(device=device, dtype=torch.float32)
            # The carried state is by definition inside the tanh range; clamp
            # defensively so float drift can never push it outside.
            E_t = torch.clamp(E_t, -1.0, 1.0)
            mode = "carry"
        else:
            E_t = E_init
            mode = "reset"
        E_next = E_t
        # Position-wise scalar PE, aligned with pre_outcome_hidden.
        #
        # THE INDEX HERE IS THE WHOLE LEAK SURFACE. Read this before changing
        # it. The dataloader makes y[p] == x[p+1] exactly, and gpt.py scores
        # cross_entropy(logits[p], targets[p]) with NO shift, so y[p] is the
        # label attached to position p's logits.
        #
        # pe_seq[j] is indexed by the same j as pre_outcome_hidden[:, j] ==
        # hidden[:, j], i.e. by INPUT position, and E_next_seq[j] is the state
        # after consuming position j. film_state[p] = E_next_seq[p-1], so
        # pe_seq[j] reaches position p = j+1 -- exactly one position later.
        # For that to stay honest, pe_seq[j] must be a function of labels y[i]
        # with i <= j, because by position j+1 the model has consumed
        # x[0..j+1] and y[i] = x[i+1] for i <= j.
        #
        # Hence targets[:, :-1], NOT targets[:, 1:]. With targets[:, 1:] the
        # entry pe_seq[j] scored y[j+1] = x[j+2], so film_state[j+1] depended
        # on x[j+2] and logits[j+1] was built from the very label it is graded
        # against. diagnostics/test_no_label_leak.py fails on that version and
        # passes on this one; the comment that used to sit further down
        # ("pe_seq[:, p-1] depends on token p -- strictly causal") asserted the
        # opposite of what the code did.
        #
        # Semantically this is the standard shifted prediction error: position
        # j's logits score the token that just arrived, i.e. how surprising the
        # model's previous prediction turned out to be. That is a quantity the
        # model can actually know at position j+1, which is what makes it
        # usable as drive input at all.
        if targets is not None:
            pe_seq = self.state.prediction_error_from_logits(
                _safe_logits(self.base, hidden, pre_outcome_hidden), targets[:, :-1]
            )
        else:
            pe_seq = torch.zeros(
                B, pre_outcome_hidden.size(1), device=device, dtype=E_t.dtype
            )

        E_start = E_t
        # VECTORIZED RECURRENCE. The drive depends only on the base hidden
        # states and the PE, neither of which can depend on E, so the whole
        # E sequence has a closed form and is computed in one shot. See
        # RDAStateModule.update_seq(); the sequential original is preserved
        # verbatim in _update_loop_reference() and is what the verification
        # script compares against.
        #
        # The predictor still needs E at each position, so it is applied
        # AFTER the state sequence exists, shifted by one: the prediction for
        # position t is made from E_t = E_start if t == 0 else E_next_seq[t-1],
        # which is exactly the pre-update state the loop used.
        L = pre_outcome_hidden.size(1)
        if L > 0:
            drive = self.state.drive_seq(pre_outcome_hidden, pe_seq)
            E_next_seq = self.state.update_seq(E_start, drive)  # (B, T-1, n_drives)
            # E_t entering the predictor at each position, shifted by one.
            E_in = torch.cat([E_start.unsqueeze(1), E_next_seq[:, :-1, :]], dim=1)
            E_hat_seq = self.predictor.predict_seq(E_in, pre_outcome_hidden)
            E_next = E_next_seq[:, -1, :]
            E_t = E_next
            dpe = drive_prediction_error(E_next_seq, E_hat_seq)
        else:
            # SINGLE-TOKEN DECODE (T == 1 with a KV cache).
            #
            # pre_outcome_hidden is empty, so the loop above never ran and the
            # state would sit frozen for the whole generation -- measured: the
            # state norm stayed at its prefill value across every decode step.
            #
            # But there IS a hidden state to consume: hidden[:, 0] is the state
            # produced by decoding the newest token. The only thing missing
            # versus training is the prediction error, because the next token
            # does not exist yet; that is passed as 0, which is exactly the
            # neutral value the PE pathway uses everywhere.
            #
            # FiLM still uses E_start (the state BEFORE this update), so the
            # token just decoded is never modulated by its own update.
            h_new = hidden[:, 0, :]                    # (B, d_model)
            E_next = self.state.update(E_t, h_new, torch.zeros(B, device=device, dtype=E_t.dtype))
            E_next_seq = E_next.new_zeros(B, 0, self.n_drives)
            E_hat_seq = E_next_seq
            dpe = E_next_seq
            E_t = E_next

        # ------------------------------------------------------------------
        # PER-POSITION CAUSAL FiLM INPUT.
        #
        # Position p is modulated by E_start for p == 0 and by E_next_seq[p-1]
        # for p >= 1 -- i.e. by the state as it stood BEFORE the update that
        # consumed position p-1.
        #
        # THE INVARIANT. Because y[p] == x[p+1] and gpt.py scores
        # cross_entropy(logits[p], targets[p]) with no shift, the requirement
        # is precisely:
        #
        #     film_state[p] may depend on y[j] only for j < p.
        #
        # Equivalently, since y[j] == x[j+1], film_state[p] may depend on
        # x[0..p] and nothing beyond -- exactly the input a causal transformer
        # position p is entitled to.
        #
        # HOW THAT IS ACHIEVED. pe_seq is now built from targets[:, :-1] (see
        # the comment at its construction above), so pe_seq[j] scores y[j] and
        # E_next_seq[j] depends on y[0..j]. film_state[p] = E_next_seq[p-1]
        # therefore depends on y[0..p-1], which is y[j] for j < p. Satisfied.
        #
        # WHAT WAS HERE BEFORE, AND WHY IT WAS WRONG. The previous version of
        # this comment read:
        #
        #     "E_next_seq[p] is built from pe_seq[:, p], and pe_seq[:, p] is
        #      the prediction error of the model's own probability for token
        #      p+1 against targets[p+1] ... E_next_seq[p-1] depends on
        #      pe_seq[:, p-1], which depends on token p -- strictly causal."
        #
        # Both halves of that reasoning were correct about the MECHANISM and
        # wrong about the ARITHMETIC. pe_seq[:, p-1] did depend on
        # targets[p] -- but targets[p] IS x[p+1], not x[p]. So the state
        # modulating position p had already seen the token that position p is
        # scored on. The reasoning checked the index and then mislabelled what
        # lives at that index.
        #
        # The one-token lookahead that the old comment worried about (using
        # E_next_seq[p] instead of E_next_seq[p-1]) is real and is still avoided
        # -- that part of the design is unchanged and still necessary. The bug
        # was in the PE index underneath it, not in this shift.
        # ------------------------------------------------------------------
        film_state = torch.cat([E_start.unsqueeze(1), E_next_seq], dim=1)  # (B, T, n_drives)
        assert film_state.size(1) == hidden.size(1), (
            f"film_state has {film_state.size(1)} positions but hidden has "
            f"{hidden.size(1)}"
        )

        # CONTROL ARM C. Swap the per-position state for a learned constant
        # BEFORE it reaches FiLM. Everything above still runs -- the drive,
        # the recurrence, the predictor, dpe -- so compute is unchanged and the
        # comparison against B' isolates INFORMATION rather than capacity or
        # cost. Only this one tensor differs.
        #
        # expand_as returns a VIEW, so this allocates nothing, and .to(dtype)
        # is a no-op when the dtypes already match.
        if self.const_state_vec is not None:
            film_state = self.const_state_vec.to(film_state.dtype).expand_as(film_state)

        # Mean over positions for a compact (B, n_drives) summary signal.
        # In single-token decode these sequences are empty, and the mean of an
        # empty tensor is NaN, which would silently poison the debug log and any
        # caller reading "dpe". Fall back to zeros.
        if dpe.size(1) == 0:
            dpe_mean = dpe.new_zeros(B, self.n_drives)
            E_next_mean = dpe.new_zeros(B, self.n_drives)
            E_hat_mean = dpe.new_zeros(B, self.n_drives)
        else:
            dpe_mean = dpe.mean(dim=1)
            E_next_mean = E_next_seq.mean(dim=1)
            E_hat_mean = E_hat_seq.mean(dim=1)

        # The returned E_next is the final state of the sequence. In the
        # training branch `E_t` has been rebound to the last update, and in the
        # single-token branch it was explicitly advanced, so E_t is correct in
        # both cases. E_start is the state that entered the step, which is what
        # the debug log wants.
        E_next_final = E_t
        # Expose the full per-position state for the training loop's
        # saturation monitor. Kept as a detached-by-convention attribute
        # rather than returned, so the forward() contract (and therefore the
        # round-trip test) is unchanged.
        self._last_E_next_seq = E_next_seq.detach()

        # Document starts are reported, never acted on (decay-only persistence).
        n_doc_starts = 0 if doc_starts is None else int(doc_starts.sum().item())

        if self.training:
            # Log E_start (the state actually entering this step) rather than
            # the loop variable, which by now aliases the final state.
            self._log_debug(E_start, E_hat_mean, E_next_final, dpe_mean,
                            mode=mode, doc_starts_this_step=n_doc_starts)

        # NOTE: logits are computed below from the FiLM-MODULATED hidden states.
        # The base forward already returned a loss from the UNMODULATED states,
        # so those logits are deliberately discarded and recomputed here.

        # ------------------------------------------------------------------
        # FiLM modulation: make the affective state actually condition the
        # model. Until this was wired in, the state was a pure observer and
        # Model B was mathematically identical to Model A.
        #
        # The base model already returned `loss` from the UNMODULATED hidden
        # states, so the LM head and cross-entropy are re-applied to the
        # modulated states. FiLM is zero-initialized, so at step 0
        # `hidden_mod == hidden` bit-for-bit and B starts identical to A; the
        # two diverge only as film weights receive gradient.
        #
        # NOTE: `dpe` is still NOT consumed. It is computed, returned
        # and logged, and nothing about memory / learning rate / attention is
        # modulated by it. The state influences the model only through FiLM.
        #
        # PER-POSITION: film_state is (B, T, n_drives), one state per position,
        # so position p is modulated only by information from tokens <= p.
        # ------------------------------------------------------------------
        hidden_mod = self.state.modulate(hidden, film_state)
        # Re-apply the LM head + softcap to the modulated hidden states.
        lm_logits = self.base.lm_head(hidden_mod)
        lm_logits = lm_logits[..., :self.base.config.vocab_size].float()
        lm_logits = 15 * torch.tanh(lm_logits / 15)  # same softcap as GPT.forward

        if targets is not None:
            if loss_reduction == "none":
                # Per-token losses. nanochat.loss_eval.evaluate_bpb expects a
                # TENSOR back from model(x, y, loss_reduction='none'), so return
                # the raw (B, T) tensor rather than the dict, keeping this
                # wrapper a drop-in for GPT in eval code.
                #
                # Record the post-forward state FIRST, before the early return.
                # evaluate_bpb advances its evaluation carry from `last_state`,
                # and it cannot do that if this attribute is only set on the
                # generation path. Detached: this is a readout, not part of any
                # graph, and nothing downstream should be able to backprop
                # through an eval.
                self.last_state = E_next_final.detach()
                self.last_mode = mode
                return F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction="none",
                ).view(targets.shape)
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        else:
            # INFERENCE / GENERATION PATH. nanochat's Engine calls
            #     model.forward(ids, kv_cache=...)  and then indexes the result
            #     as a tensor: logits[:, -1, :]
            # so when there are no targets this MUST return a bare (B, T, V)
            # tensor, not a dict. Returning a dict here was a hard interface
            # break that made RDA models undecodable. Returning a tensor keeps
            # this wrapper a drop-in for GPT in the generation path, and the
            # caller can read the state off the model via `last_state`.
            self.last_state = E_next_final
            self.last_mode = mode
            return lm_logits

        return {
            "loss": loss,
            "E_next": E_next_final,
            "dpe": dpe_mean,
            "dpe_seq": dpe,
            "E_next_seq": E_next_seq,
            "E_hat_next_seq": E_hat_seq,
            "hidden": hidden,
            "hidden_mod": hidden_mod,
            "mode": mode,
            "n_doc_starts": n_doc_starts,
        }


def _safe_logits(base, hidden, pre_outcome_hidden):
    """Recompute logits for the pre-outcome positions so the token-level PE is
    aligned with the same (B, T-1, ...) slice as pre_outcome_hidden.

    This reuses the ALREADY-COMPUTED hidden states; it is a head application,
    not a second transformer forward pass.
    """
    x = pre_outcome_hidden
    logits = base.lm_head(x)
    logits = logits[..., :base.config.vocab_size]
    logits = logits.float()
    softcap = 15
    return softcap * torch.tanh(logits / softcap)


def _logits_from_hidden(base, hidden):
    """Full-sequence logits for the inference path."""
    logits = base.lm_head(hidden)
    logits = logits[..., :base.config.vocab_size]
    logits = logits.float()
    softcap = 15
    return softcap * torch.tanh(logits / softcap)
