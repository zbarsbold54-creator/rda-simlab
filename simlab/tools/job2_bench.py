"""Job 2 in-process update-only benchmark (no workers)."""
import faulthandler
import sys
import time
from pathlib import Path

def collect_rollout():
    sys.path.insert(0, ".")
    from simlab.training.ppo import PPOConfig, PPOTrainer
    from simlab.training.arch_spec import default_preset_selection
    cfg = PPOConfig(obs_mode="obs_drives", total_steps=20480, workers=4, seed=100000, eval_every=0, profile=False, update_device="cpu", minibatch_chunks=32)
    cfg.arch = default_preset_selection("rda_gru_surprise", seed=100000)
    cfg.obs_mode = cfg.arch.get("obs_mode", cfg.obs_mode)
    trainer = PPOTrainer(cfg, "logs/overnight/job2_collect")
    trainer.start()
    try:
        trainer._broadcast()
        trajs = trainer._collect()
    finally:
        trainer.stop()
    # save trajs to .pt
    import torch
    Path("logs/overnight/job2_rollout.pt").parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajs, "logs/overnight/job2_rollout.pt")
    print(f"collected {len(trajs)} segments, saved to logs/overnight/job2_rollout.pt", flush=True)
    return trajs

def bench_update(device, minibatch, venv_python):
    faulthandler.dump_traceback_later(90, exit=True)
    import torch
    sys.path.insert(0, ".")
    from simlab.training.ppo import PPOConfig, PPOTrainer
    from simlab.training.arch_spec import default_preset_selection
    cfg = PPOConfig(obs_mode="obs_drives", total_steps=20480, workers=4, seed=100000, eval_every=0, profile=True, update_device=device, minibatch_chunks=minibatch)
    cfg.arch = default_preset_selection("rda_gru_surprise", seed=100000)
    cfg.obs_mode = cfg.arch.get("obs_mode", cfg.obs_mode)
    if device == "cpu":
        torch.set_num_threads(4)
    trainer = PPOTrainer(cfg, f"logs/overnight/job2_bench_{device}_{minibatch}")
    trajs = torch.load("logs/overnight/job2_rollout.pt", weights_only=False)
    # warmup
    trainer._ppo_update(trajs)
    times = []
    for _ in range(3):
        t0 = time.time()
        if device == "cuda":
            torch.cuda.synchronize()
        trainer._ppo_update(trajs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    times.sort()
    median = times[1]
    peak_vram = None
    if device == "cuda" and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"RESULT device={device} minibatch={minibatch} update_median={median:.3f}s peak_vram={peak_vram}", flush=True)
    return median, peak_vram

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--minibatch", type=int, default=None)
    args = ap.parse_args()
    if args.collect:
        collect_rollout()
    else:
        bench_update(args.device, args.minibatch, sys.executable)
