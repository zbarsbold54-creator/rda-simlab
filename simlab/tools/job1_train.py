"""Job 1: long learning quality runs (scratch, CPU only)."""
import sys, time, subprocess, os, json
sys.path.insert(0, ".")
from pathlib import Path

def train(arm, minibatch, rollout, lr_mult, seed, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    code = f"""
import sys
sys.path.insert(0, ".")
from simlab.training.ppo import PPOConfig, PPOTrainer
from simlab.training.arch_spec import default_preset_selection
cfg = PPOConfig(obs_mode="obs_drives", total_steps=300000, workers=4, seed={seed}, eval_every=0, profile=True, update_device="cpu", minibatch_chunks={minibatch}, env_steps_per_rollout={rollout}, lr0=3e-4*{lr_mult})
cfg.arch = default_preset_selection("rda_gru_surprise", seed={seed})
cfg.obs_mode = cfg.arch.get("obs_mode", cfg.obs_mode)
trainer = PPOTrainer(cfg, r"{out}")
trainer.train()
"""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "-c", code], cwd=".", env=env, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(f"FAILED {arm} seed {seed}: {r.stderr[-1000:]}")
        return None
    # parse learning.csv for final kl/vloss, updates, optimizer steps (minibatches)
    csv = out / "learning.csv"
    kl = vloss = None
    iters = 0
    if csv.exists():
        lines = csv.read_text().splitlines()
        for ln in lines:
            if ln and not ln.startswith("#") and not ln.startswith("iteration"):
                iters += 1
                parts = ln.split(",")
                kl = parts[5]
                vloss = parts[6]
    # profile gives updates/minibatches; parse stdout for updates/minibatches
    updates = minibatches = None
    for ln in r.stdout.splitlines():
        if "updates=" in ln and "minibatches=" in ln:
            # e.g. updates=20 minibatches=404 ...
            try:
                updates = int(ln.split("updates=")[1].split()[0])
                minibatches = int(ln.split("minibatches=")[1].split()[0])
            except Exception:
                pass
    print(f"RESULT arm={arm} seed={seed} wall={wall:.1f}s iters={iters} updates={updates} minibatches={minibatches} kl={kl} vloss={vloss} csv={csv}")
    return {"wall": wall, "iters": iters, "updates": updates, "minibatches": minibatches, "kl": kl, "vloss": vloss, "csv": str(csv)}

def eval_ckpt(ckpt, scenario, out_json):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [sys.executable, "-m", "simlab.cli", "eval", "--policy", "learned_torch", "--scenario", scenario, "--agents", "5", "--seeds", "20", "--seed-offset", "0", "--cap", "10000", "--workers", "4", "--checkpoint", str(ckpt), "--out", str(out_json)]
    r = subprocess.run(cmd, cwd=".", env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"eval failed {scenario}: {r.stderr[-500:]}")
        return None
    data = json.loads(Path(out_json).read_text())
    # aggregate returns cap_pct and median? Check eval_lib aggregate keys
    return data

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--minibatch", type=int, required=True)
    ap.add_argument("--rollout", type=int, required=True)
    ap.add_argument("--lr-mult", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    out_dir = f"logs/overnight/job1_{args.arm}_{args.seed}"
    res = train(args.arm, args.minibatch, args.rollout, args.lr_mult, args.seed, out_dir)
    if res:
        ckpt = Path(out_dir) / "final.pt"
        for scen in ["balanced", "hard"]:
            out_json = Path(out_dir) / f"eval_{scen}.json"
            data = eval_ckpt(ckpt, scen, out_json)
            print(f"EVAL {args.arm} seed={args.seed} {scen}: {data}")
