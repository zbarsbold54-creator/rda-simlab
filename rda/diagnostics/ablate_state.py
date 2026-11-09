"""
State-content ablation on trained checkpoints (eval-time only).

Loads a B'/C checkpoint the same way base_train.py does, then computes
validation bpb with the same eval function (nanochat.loss_eval.evaluate_bpb)
and the same eval tokens (524288) at device batch size 2.

Variants are applied at EVAL TIME ONLY by monkeypatching
GPTWithRDA.forward (the function that produces film_state, rda.py:1111)
in this script. nanochat/rda.py is never edited.

Variants:
  V0 normal      -- no change
  V1 const-state -- the same constant arm C uses (const_state_vec)
  V2 shuffled    -- each batch row uses the state trajectory computed from a
                    DIFFERENT row of the same batch (roll batch dim by 1)
  V3 time-shuffled -- the state trajectory of the same row, positions reversed
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.rda import GPTWithRDA, drive_prediction_error


def build_model_from_checkpoint(checkpoint_dir, step, device, seed,
                                decay_override=None):
    """Mirror base_train.py's load path: the checkpoint holds the full
    GPTWithRDA state dict (keys prefixed base./state./predictor.)."""
    model_data, _optim_data, meta_data = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    cfg_kwargs = meta_data["model_config"]
    config = GPTConfig(**cfg_kwargs)
    saved_decays = (meta_data.get("user_config") or {}).get("rda_decays")
    decay = decay_override if decay_override is not None else saved_decays
    if decay is not None:
        if len(decay) != 8 or any(not 0.0 <= value < 1.0 for value in decay):
            raise ValueError("--rda-decays must contain exactly 8 values in [0, 1)")
        decay = [float(value) for value in decay]
    # A C checkpoint carries the learned const_state_vec; a B' one does not.
    has_const = "const_state_vec" in model_data
    with torch.device("meta"):
        base = GPT(config)
    # Wrap exactly as base_train.py does (const_state per the arm).
    model = GPTWithRDA(base, debug_steps=0, const_state=has_const,
                       const_state_seed=seed,
                       decay=decay if decay is not None else 0.95)
    model.to_empty(device=device)
    model.base.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    if decay_override is not None:
        model.state.decay.copy_(
            torch.tensor(decay_override, device=device, dtype=model.state.decay.dtype)
        )
        model.state.decay_safe_T = min(
            model.state._max_safe_T(value, torch.float64)
            for value in decay_override
        )
    del model_data
    return model, meta_data, has_const


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--step", type=int, default=600)
    ap.add_argument("--device-batch-size", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--eval-tokens", type=int, default=524288)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="V0,V1,V2,V3")
    ap.add_argument("--rda-decays", default=None,
                    help="optional comma-separated decay values, one per drive")
    args = ap.parse_args()
    decay_override = None
    if args.rda_decays is not None:
        try:
            decay_override = [float(value) for value in args.rda_decays.split(",")]
        except ValueError as exc:
            ap.error(f"invalid --rda-decays: {exc}")
        if len(decay_override) != 8 or any(
                not 0.0 <= value < 1.0 for value in decay_override):
            ap.error("--rda-decays must contain exactly 8 values in [0, 1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()

    model, meta_data, has_const = build_model_from_checkpoint(
        args.checkpoint_dir, args.step, device, args.seed,
        decay_override=decay_override)
    model.eval()

    # The constant arm C hands to FiLM. For a C checkpoint this is the
    # LEARNED const_state_vec loaded from the checkpoint; for a B'
    # checkpoint it is constructed exactly as arm C does at init
    # (rda.py:746-756): private generator seeded (seed + 1000003),
    # randn * 0.3.
    if has_const:
        const_vec = model.const_state_vec.detach().to(device)
    else:
        _g = torch.Generator(device="cpu")
        _g.manual_seed(int(args.seed) + 1000003)
        const_vec = (torch.randn(model.n_drives, generator=_g,
                                 dtype=torch.float32) * 0.3).to(device)

    eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
    val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)

    orig_forward = GPTWithRDA.forward

    def make_variant_forward(variant):
        if variant == "V0":
            return orig_forward

        def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean",
                    E_t_external=None, doc_starts=None):
            # Replicate GPTWithRDA.forward up to film_state, then swap it.
            loss, hidden = self.base.forward(
                idx, targets, kv_cache=kv_cache,
                loss_reduction=loss_reduction, return_hidden=True)
            pre_outcome_hidden = hidden[:, :-1, :]
            pooled_post_outcome = hidden[:, 1:, :]
            B = idx.size(0)
            device_ = idx.device
            E_init = self.state.init_state(B, device_, dtype=torch.float32)
            if E_t_external is not None:
                E_t = E_t_external.detach().to(device=device_, dtype=torch.float32)
                E_t = torch.clamp(E_t, -1.0, 1.0)
                mode = "carry"
            else:
                E_t = E_init
                mode = "reset"
            E_start = E_t
            # pe_seq: per-position scalar NLE of the observed token,
            # computed from the PRE-outcome hidden states (rda.py:1017).
            if targets is not None:
                _x = pre_outcome_hidden
                _logits = self.base.lm_head(_x)
                _logits = _logits[..., :self.base.config.vocab_size].float()
                _logits = 15 * torch.tanh(_logits / 15)
                pe_seq = self.state.prediction_error_from_logits(
                    _logits, targets[:, :-1])
            else:
                pe_seq = torch.zeros(
                    B, pre_outcome_hidden.size(1), device=device_, dtype=E_t.dtype)
            L = pre_outcome_hidden.size(1)
            drive = self.state.drive_seq(pre_outcome_hidden, pe_seq)
            E_next_seq = self.state.update_seq(E_start, drive)
            E_in = torch.cat([E_start.unsqueeze(1), E_next_seq[:, :-1, :]], dim=1)
            E_hat_seq = self.predictor.predict_seq(E_in, pre_outcome_hidden)
            E_next = E_next_seq[:, -1, :]
            E_t = E_next
            dpe = drive_prediction_error(E_next_seq, E_hat_seq)
            film_state = torch.cat([E_start.unsqueeze(1), E_next_seq], dim=1)

            # Mirror the real forward's const-state replacement
            # (rda.py:1125-1126) so V0 is bit-identical to the
            # unmodified C forward.
            if self.const_state_vec is not None:
                film_state = self.const_state_vec.to(
                    film_state.dtype).expand_as(film_state)

            # Apply the variant AFTER the const-state replacement, so
            # V2/V3 are observable on a C checkpoint too.
            if variant == "V1":
                film_state = const_vec.to(film_state.dtype).expand_as(film_state)
            elif variant == "V2":
                # each row uses the state trajectory from a DIFFERENT row
                film_state = film_state.roll(1, dims=0)
            elif variant == "V3":
                # same row, positions reversed (position 0 keeps E_start)
                rev = film_state[:, 1:, :].flip(dims=[1])
                film_state = torch.cat([film_state[:, :1, :], rev], dim=1)
            else:
                raise ValueError(variant)

            hidden_mod = self.state.modulate(hidden, film_state)
            lm_logits = self.base.lm_head(hidden_mod)
            lm_logits = lm_logits[..., :self.base.config.vocab_size].float()
            lm_logits = 15 * torch.tanh(lm_logits / 15)
            if targets is not None:
                if loss_reduction == "none":
                    self.last_state = E_next.detach()
                    self.last_mode = mode
                    return torch.nn.functional.cross_entropy(
                        lm_logits.view(-1, lm_logits.size(-1)),
                        targets.view(-1), ignore_index=-1, reduction="none",
                    ).view(targets.shape)
                loss = torch.nn.functional.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            else:
                self.last_state = E_next
                self.last_mode = mode
                return lm_logits
            return {"loss": loss, "E_next": E_next, "dpe": dpe,
                    "E_next_seq": E_next_seq, "hidden": hidden,
                    "hidden_mod": hidden_mod, "mode": mode}
        return forward

    results = {}
    for variant in [v.strip() for v in args.variants.split(",")]:
        GPTWithRDA.forward = make_variant_forward(variant)
        # Fresh eval carry per variant, exactly as base_train.py does.
        carry = model.state.init_state(args.device_batch_size, device, dtype=torch.float32)
        bpb = evaluate_bpb(model, val_loader(), eval_steps, token_bytes, rda_carry=carry)
        results[variant] = bpb
        print(f"VARIANT={variant} val_bpb={bpb:.6f}", flush=True)
        if variant == "V0" and decay_override is None:
            recorded_bpb = meta_data.get("val_bpb")
            if recorded_bpb is not None:
                difference = abs(bpb - float(recorded_bpb))
                print(f"V0_CHECK recorded={float(recorded_bpb):.9f} "
                      f"reevaluated={bpb:.9f} abs_diff={difference:.3e}",
                      flush=True)
                if difference > 1e-6:
                    raise SystemExit(
                        "V0 does not reproduce checkpoint val_bpb within 1e-6"
                    )

    GPTWithRDA.forward = orig_forward
    print("RESULT_JSON=" + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
