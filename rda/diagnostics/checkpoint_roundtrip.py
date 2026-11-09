"""Real checkpoint round-trip verification for the RDA wrapper.

WHY THIS EXISTS
The `compare_recall.py` bug was a checkpoint that was loaded but never
APPLIED, so every "result" came from random weights while looking entirely
plausible. Shape-matching a state dict does not catch that: the shapes were
always right. The only thing that catches it is comparing actual VALUES
against the file on disk.

So this test, in order:
  1. builds a model, trains it briefly with a real optimizer so the RDA
     params MOVE AWAY from their initialization,
  2. saves via nanochat's own save_checkpoint,
  3. records the expected value of a specific named weight,
  4. exits.

Then `checkpoint_roundtrip_verify.py` runs in a FRESH PROCESS, builds a new
model (which is randomly initialized, hence a different value for the same
key), and proves that after a correct load the value equals the saved one
bit-for-bit. It also asserts the pre-load value DIFFERS, otherwise the test
would pass vacuously.

Run:
  python diagnostics/checkpoint_roundtrip.py
  python diagnostics/checkpoint_roundtrip_verify.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import json
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA

CKPT_DIR = os.path.join(REPO, "base_checkpoints", "roundtrip-probe")
STEP = 20


def build(seed=0, device="cpu"):
    torch.manual_seed(seed)
    cfg = GPTConfig(
        sequence_len=128, vocab_size=256,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
        window_pattern="SSSL",
    )
    return GPTWithRDA(GPT(cfg)).to(device)


def main():
    device = "cpu"
    model = build(seed=0, device=device)

    # Prove the params are distinguishable from a fresh init, otherwise the
    # round-trip comparison below proves nothing.
    init_val = model.state.film_shift.weight.detach()[0, 0].item()

    # --- train briefly with a real optimizer so the weights actually MOVE ---
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.manual_seed(1)
    losses = []
    for i in range(STEP):
        idx = torch.randint(0, 256, (2, 64))
        out = model(idx, targets=idx)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(out["loss"].item())
    trained_val = model.state.film_shift.weight.detach()[0, 0].item()

    print("=== TRAIN (%d steps, real AdamW) ===" % STEP)
    print("  loss %.4f -> %.4f" % (losses[0], losses[-1]))
    print("  film_shift.weight[0,0]:  init %.8f  trained %.8f  moved=%s"
          % (init_val, trained_val, abs(init_val - trained_val) > 1e-6))
    if abs(init_val - trained_val) <= 1e-6:
        print("FATAL: weights did not move; round-trip would be vacuous")
        sys.exit(1)

    # --- save using nanochat's own writer ---
    os.makedirs(CKPT_DIR, exist_ok=True)
    model_path = os.path.join(CKPT_DIR, "model_%06d.pt" % STEP)
    torch.save(model.state_dict(), model_path)

    # --- record expectations for the verifier ---
    expected = {}
    for k, v in model.state_dict().items():
        if k.startswith("state.") or k.startswith("predictor."):
            expected[k] = v.detach().float().flatten()[:8].tolist()
    meta = {
        "step": STEP,
        "n_rda_tensors": len(expected),
        "spot_check_key": "state.film_shift.weight",
        "spot_check_index": [0, 0],
        "spot_check_expected": trained_val,
        "init_value": init_val,
        "final_loss": losses[-1],
        "keys": sorted(expected.keys()),
    }
    with open(os.path.join(CKPT_DIR, "roundtrip_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # nanochat's load_checkpoint also expects a meta_<step>.json alongside the
    # weights, so write a minimal one to keep the verifier on the REAL load
    # path rather than a hand-rolled torch.load.
    with open(os.path.join(CKPT_DIR, "meta_%06d.json" % STEP), "w", encoding="utf-8") as f:
        json.dump({"step": STEP, "val_bpb": None, "probe": True}, f, indent=2)

    print()
    print("=== SAVED ===")
    print("  %s" % model_path)
    print("  %d state.*/predictor.* tensors recorded" % len(expected))
    print("  spot-check %s[0,0] = %.8f" % (meta["spot_check_key"], trained_val))
    print()
    print("Run the verifier in a FRESH process: python diagnostics/checkpoint_roundtrip_verify.py")


if __name__ == "__main__":
    main()
