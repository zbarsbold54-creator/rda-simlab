"""Check causal segment isolation and the intentional RDA cross-segment path."""

import argparse
import os

os.environ["NANOCHAT_DTYPE"] = "float32"

import torch

from diagnostics.recall_models import load_model, parse_decays
from nanochat.recall_data import (
    FILLER_START,
    N_FILLER,
    VALUE_START,
    build_batch,
    row,
)

DECAY_SCHEDULE = "0.9,0.95,0.975,0.9875,0.99375,0.996875,0.9984375,0.99921875"
SEGMENT_LEN = 128
N_ROWS = 64


def prepare_rows():
    rows = [
        row(1_000_000_000, index, delay_bucket=2 + index % 6)
        for index in range(N_ROWS)
    ]
    changed_values = []
    changed_controls = []
    selected_events = []
    for inputs, _, _, metadata in rows:
        event = next(
            event for event in metadata["events"] if event["delay"] >= 192
        )
        value_position = event["event_key_position"] + 1
        old_value = int(inputs[value_position])
        replacement = VALUE_START + ((old_value - VALUE_START + 1) % 16)
        changed = inputs.clone()
        changed[value_position] = replacement
        changed_values.append(changed)

        query_position = event["query_position"]
        segment_start = query_position // SEGMENT_LEN * SEGMENT_LEN
        protected = set()
        for item in metadata["events"]:
            protected.update((
                item["event_key_position"],
                item["event_key_position"] + 1,
                item["query_position"] + 1,
            ))
        control_candidates = [
            position
            for position in range(query_position - 1, segment_start - 1, -1)
            if position not in protected
        ]
        control_position = (
            control_candidates[0] if control_candidates else query_position
        )
        control = inputs.clone()
        old_control = int(control[control_position])
        control[control_position] = (
            FILLER_START
            + ((old_control - FILLER_START + 1) % N_FILLER)
        )
        changed_controls.append(control)
        selected_events.append((event, control_position))
    return rows, changed_values, changed_controls, selected_events


@torch.inference_mode()
def logits_for(model, tokens, device):
    tokens = tokens.unsqueeze(0).to(device)
    if hasattr(model, "state"):
        return model(tokens)
    return model(tokens)


@torch.inference_mode()
def measure_model(label, checkpoint_dir, step, seed, rows, changed_values,
                  changed_controls, selected_events, device, decay_override=None):
    model, _, _ = load_model(
        checkpoint_dir, step, device, seed, SEGMENT_LEN,
        decay_override=decay_override,
    )
    maxima = []
    answer_max = 0.0
    answer_abs_sum = 0.0
    answer_value_count = 0
    control_diffs = []
    for index, ((inputs, _, _, _), changed, control, (event, control_pos)) in enumerate(
        zip(rows, changed_values, changed_controls, selected_events)
    ):
        original_logits = logits_for(model, inputs, device)
        changed_logits = logits_for(model, changed, device)
        later = (
            torch.arange(inputs.numel(), device=device) // SEGMENT_LEN
            > event["event_segment"]
        )
        if label == "A":
            maxima.append(float(
                (original_logits[0, later] - changed_logits[0, later])
                .abs().max().item()
            ))
        else:
            answer_diff = (
                original_logits[0, event["query_position"]]
                - changed_logits[0, event["query_position"]]
            ).abs()
            answer_max = max(answer_max, float(answer_diff.max().item()))
            answer_abs_sum += float(answer_diff.sum().item())
            answer_value_count += answer_diff.numel()
        control_logits = logits_for(model, control, device)
        control_diffs.append(float(
            (original_logits[0, event["query_position"]]
             - control_logits[0, event["query_position"]])
            .abs().max().item()
        ))
        del original_logits, changed_logits, control_logits
    del model
    torch.cuda.empty_cache()

    step1 = max(maxima) if label == "A" else answer_max
    mean_later = (
        0.0 if label == "A" else answer_abs_sum / answer_value_count
    )
    control_change = sum(control_diffs) / len(control_diffs)
    print(
        f"MODEL={label} step1_max_abs_diff={step1:.9g} "
        f"step1_mean_abs_answer_diff={mean_later:.9g} "
        f"step2_mean_max_abs_answer_change={control_change:.9g}",
        flush=True,
    )
    if label == "A" and step1 != 0.0:
        raise SystemExit(
            "A segment invariance failed: a prior segment changed later logits"
        )
    if label != "A" and step1 == 0.0:
        raise SystemExit(f"{label} expected a nonzero cross-segment RDA effect")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-checkpoint-dir", required=True)
    parser.add_argument("--bprime-checkpoint-dir", required=True)
    parser.add_argument("--d-checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, default=600)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rda-decays", default=DECAY_SCHEDULE)
    args = parser.parse_args()
    decays = parse_decays(args.rda_decays)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the checkpoint invariance test")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rows, changed_values, changed_controls, selected_events = prepare_rows()

    measure_model(
        "A", args.a_checkpoint_dir, args.step, args.seed,
        rows, changed_values, changed_controls, selected_events, device,
    )
    measure_model(
        "Bprime", args.bprime_checkpoint_dir, args.step, args.seed,
        rows, changed_values, changed_controls, selected_events, device,
    )
    measure_model(
        "D", args.d_checkpoint_dir, args.step, args.seed,
        rows, changed_values, changed_controls, selected_events, device,
        decay_override=decays,
    )


if __name__ == "__main__":
    main()
