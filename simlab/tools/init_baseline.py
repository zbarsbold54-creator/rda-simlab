"""Init baseline: fresh rda_gru_surprise per training seed, eval Balanced/Hard."""
import sys, subprocess, os, json
sys.path.insert(0, ".")
from pathlib import Path
from simlab.training.arch_spec import default_preset_selection, build_policy_cfg
from simlab.training.torch_policy import make_policy

for seed in range(100000, 100012):
    out = Path(f"logs/overnight/init_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "final.pt"
    if not ckpt.exists():
        sel = default_preset_selection("rda_gru_surprise", seed=seed)
        cfg = build_policy_cfg(sel)
        pol = make_policy(cfg)
        pol.save_checkpoint(ckpt)
        print(f"saved init {seed} -> {ckpt}", flush=True)
    for scen in ["balanced", "hard"]:
        out_json = out / f"eval_{scen}.json"
        if out_json.exists():
            continue
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        cmd = [sys.executable, "-m", "simlab.cli", "eval", "--policy", "learned_torch", "--scenario", scen, "--agents", "5", "--seeds", "20", "--seed-offset", "0", "--cap", "10000", "--workers", "4", "--checkpoint", str(ckpt), "--out", str(out_json)]
        r = subprocess.run(cmd, cwd=".", env=env, capture_output=True, text=True, timeout=600)
        print(f"eval {seed} {scen} rc={r.returncode}", flush=True)
        if r.returncode == 0:
            data = json.loads(out_json.read_text())
            print(f" {seed} {scen} cap={data['cap_pct']}% median={data['median_survival']}", flush=True)
