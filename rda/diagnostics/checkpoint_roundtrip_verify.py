"""Fresh-process verification that a saved RDA checkpoint actually LOADS.

This is the half that catches the `compare_recall.py` class of bug: a
checkpoint that is read off disk but never applied to the model leaves the
model at its random initialization, and shape checks look perfectly healthy.

Run AFTER diagnostics/checkpoint_roundtrip.py, in a separate process:

  python diagnostics/checkpoint_roundtrip.py
  python diagnostics/checkpoint_roundtrip_verify.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import json
import torch

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA

CKPT_DIR = os.path.join(REPO, "base_checkpoints", "roundtrip-probe")
STEP = 20


def build(seed=123, device="cpu"):
    """Deliberately a DIFFERENT seed from the training process, so a fresh
    model is guaranteed to disagree with the saved values."""
    torch.manual_seed(seed)
    cfg = GPTConfig(
        sequence_len=128, vocab_size=256,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
        window_pattern="SSSL",
    )
    return GPTWithRDA(GPT(cfg)).to(device)


def main():
    device = "cpu"
    with open(os.path.join(CKPT_DIR, "roundtrip_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    # 1) A fresh model: values are random and must NOT match the checkpoint.
    model = build(device=device)
    key = meta["spot_check_key"]
    r, c = meta["spot_check_index"]
    before = float(model.state_dict()[key][r, c].item())
    after = float(meta["spot_check_expected"])
    print("=== PRE-LOAD (fresh, random init) ===")
    print("  %s[%d,%d] = %.8f   (checkpoint has %.8f)" % (key, r, c, before, after))
    differs_before = abs(before - after) > 1e-6
    print("  differs from checkpoint: %s" % differs_before)
    if not differs_before:
        print("FATAL: fresh init already matches; the test cannot detect a no-op load")
        sys.exit(1)

    # 2) Load the way nanochat does, and APPLY it -- the step that was missed.
    model_data, _, _ = load_checkpoint(CKPT_DIR, STEP, device)
    missing, unexpected = model.load_state_dict(model_data, strict=True), None
    print()
    print("=== APPLIED load_state_dict ===")
    print("  load_state_dict -> %s" % (missing,))

    # 3) Spot-check the specific weight against the FILE, not against memory.
    disk = torch.load(os.path.join(CKPT_DIR, "model_%06d.pt" % STEP),
                      map_location="cpu")
    key = meta["spot_check_key"]
    r, c = meta["spot_check_index"]
    file_val = float(disk[key][r, c].item())
    model_val = float(model.state_dict()[key][r, c].item())
    print()
    print("=== SPOT CHECK vs FILE ON DISK ===")
    print("  %s[%d,%d]" % (key, r, c))
    print("    file   %.8f" % file_val)
    print("    model  %.8f" % model_val)
    print("    exact match: %s" % (file_val == model_val))
    if file_val != model_val:
        print("FAIL: loaded model does not match the file")
        sys.exit(1)

    # 4) EVERY rda tensor, not just one, must match exactly.
    print()
    print("=== ALL %d state.*/predictor.* TENSORS ===" % meta["n_rda_tensors"])
    worst, worst_key, n_exact = 0.0, None, 0
    for k in meta["keys"]:
        a = disk[k].float()
        b = model.state_dict()[k].float()
        if a.shape != b.shape:
            print("  SHAPE MISMATCH %s %s vs %s" % (k, tuple(a.shape), tuple(b.shape)))
            sys.exit(1)
        d = float((a - b).abs().max())
        if d == 0.0:
            n_exact += 1
        if d > worst:
            worst, worst_key = d, k
    print("  exact (max|diff| == 0): %d / %d" % (n_exact, len(meta["keys"])))
    print("  worst tensor: %s  max|diff| = %.3e" % (worst_key, worst))
    if n_exact != len(meta["keys"]):
        print("FAIL: not all RDA tensors round-tripped exactly")
        sys.exit(1)

    # 5) And confirm the BASE model weights came along too, so we know the
    #    load was real and not a partial application.
    base_keys = [k for k in disk if k.startswith("base.")]
    base_exact = sum(
        1 for k in base_keys
        if float((disk[k].float() - model.state_dict()[k].float()).abs().max()) == 0.0
    )
    print()
    print("=== BASE (sanity: the load was a real full load) ===")
    print("  base.* tensors exact: %d / %d" % (base_exact, len(base_keys)))

    print()
    print("CHECKPOINT ROUND-TRIP: PASS")


if __name__ == "__main__":
    main()
