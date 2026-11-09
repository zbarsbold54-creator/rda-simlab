"""Checkpoint loader shared by the synthetic-recall diagnostic scripts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA


def parse_decays(value):
    decays = [float(item) for item in value.split(",")]
    if len(decays) != 8 or any(not 0.0 <= decay < 1.0 for decay in decays):
        raise ValueError("decays must contain exactly 8 values in [0, 1)")
    return decays


def load_model(checkpoint_dir, step, device, seed, segment_len,
               decay_override=None):
    model_data, _, meta = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False
    )
    model_data = {
        key.removeprefix("_orig_mod."): value
        for key, value in model_data.items()
    }
    config_kwargs = dict(meta["model_config"])
    config_kwargs["segment_len"] = segment_len
    config = GPTConfig(**config_kwargs)
    user_config = meta.get("user_config") or {}
    saved_decays = user_config.get("rda_decays")
    decays = decay_override if decay_override is not None else saved_decays
    if decays is not None:
        decays = [float(value) for value in decays]
        if len(decays) != 8 or any(not 0.0 <= value < 1.0 for value in decays):
            raise ValueError("checkpoint decay schedule must have 8 values in [0, 1)")

    with torch.device("meta"):
        base = GPT(config)
    is_rda = bool(meta.get("rda")) or any(
        key.startswith(("base.", "state.", "predictor."))
        for key in model_data
    )
    has_const_state = "const_state_vec" in model_data
    if is_rda:
        model = GPTWithRDA(
            base,
            debug_steps=0,
            const_state=has_const_state,
            const_state_seed=int(user_config.get("seed", seed)),
            decay=decays if decays is not None else 0.95,
        )
    else:
        model = base
    model.to_empty(device=device)
    model.base.init_weights() if is_rda else model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    if is_rda and decay_override is not None:
        model.state.decay.copy_(
            torch.tensor(
                decay_override,
                device=device,
                dtype=model.state.decay.dtype,
            )
        )
        model.state.decay_safe_T = min(
            model.state._max_safe_T(value, torch.float64)
            for value in decay_override
        )
    del model_data
    model.eval()
    return model, meta, decays
