"""Command line entry point for SimLab."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from simlab import __version__
from simlab.core.config import load_config
from simlab.core.runner import Runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simlab-cli")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command")
    run = subs.add_parser("run")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--seed", type=int, default=None)
    bench = subs.add_parser("bench")
    bench.add_argument("--steps", type=int, default=1000)
    bench.add_argument("--agents", type=int, default=1)
    subs.add_parser("gui")
    evalp = subs.add_parser("eval")
    evalp.add_argument("--policy", required=True)
    evalp.add_argument("--drive", default="simple")
    evalp.add_argument(
        "--scenario", choices=["balanced", "solo", "hard"], required=True
    )
    evalp.add_argument("--agents", type=int, required=True)
    evalp.add_argument("--seeds", type=int, default=20)
    evalp.add_argument("--seed-offset", type=int, default=0)
    evalp.add_argument("--cap", type=int, default=10000)
    evalp.add_argument("--workers", type=int, default=4)
    evalp.add_argument("--out", default=None)
    evalp.add_argument("--compare", action="store_true")
    evalp.add_argument("--allow-stale", action="store_true")
    evalp.add_argument("--checkpoint", default=None)
    trainp = subs.add_parser("train")
    trainp.add_argument("--obs", choices=["obs_only", "obs_drives"], default="obs_only")
    trainp.add_argument("--total-steps", type=int, default=300000)
    trainp.add_argument("--workers", type=int, default=4)
    trainp.add_argument("--seed", type=int, default=0)
    trainp.add_argument("--arch", default=None)
    trainp.add_argument("--out", required=True)
    trainp.add_argument("--eval-every", type=int, default=20)
    trainp.add_argument("--profile", action="store_true")
    trainp.add_argument("--update-threads", type=int, default=None)
    trainp.add_argument("--update-device", choices=["cpu", "cuda"], default="cpu")
    trainp.add_argument("--gamma", type=float, default=None)
    trainp.add_argument("--rollout", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_simulation(args.steps, args.seed)
    if args.command == "bench":
        return run_benchmark(args.steps, args.agents)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "gui":
        from simlab.gui.app import main as gui_main

        return gui_main()
    parser.print_help()
    return 1


def _eval_worker(
    task: tuple[str, str, int, int, int, str, str | None],
) -> dict[str, Any]:
    # One thread per worker avoids BLAS oversubscription in the spawned pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    policy, drive, seed, agents, cap, config_path, checkpoint = task
    if policy == "learned_torch":
        import simlab.training.torch_policy  # noqa: F401  (registers policy)
    from simlab.eval_lib import run_episode_metrics

    return run_episode_metrics(
        policy, seed, agents, cap, config_path, drive=drive, checkpoint=checkpoint
    )


def _print_param_count(policy_name: str, config_path: str, checkpoint: str | None) -> None:
    try:
        from dataclasses import replace

        from simlab.core.config import load_config
        from simlab.core.registry import get

        if policy_name == "learned_torch":
            import simlab.training.torch_policy  # noqa: F401  (registers policy)
        cls = get("policy", policy_name)
        if cls is None or not hasattr(cls, "param_count"):
            return
        cfg = replace(
            load_config(config_path),
            policy=policy_name,
            policy_checkpoint=checkpoint,
        )
        inst = cls(seed=0, config=cfg)
        print(f"param_count={inst.param_count()}")
    except Exception:  # noqa: BLE001
        return


def _validate_policy_name(policy: str) -> None:
    """Reject unknown policy names up front instead of silently falling back.

    The eval path registers its policies lazily (torch is imported only when
    ``learned_torch`` is requested), so we load the registrar and compare
    against the registered set. Backbone names (e.g. ``gru_legacy``) and any
    typo are not valid policy names and must fail loudly.
    """
    from simlab.core.registry import list_, load_plugins

    load_plugins()
    import simlab.training.torch_policy  # noqa: F401  (registers learned_torch)

    valid = sorted(list_("policy"))
    if policy not in valid:
        raise ValueError(
            f"unknown policy {policy!r}; valid policies: {valid}"
        )


def run_eval(args: argparse.Namespace) -> int:
    _validate_policy_name(args.policy)
    from simlab.eval_lib import (
        REPO,
        aggregate,
        compare_to_baselines,
        scenario_baseline_key,
    )

    config_path = str(
        REPO
        / {
            "balanced": "configs/balanced.toml",
            "solo": "configs/solo.toml",
            "hard": "configs/hard.toml",
        }[args.scenario]
    )
    seed_start = getattr(args, "seed_offset", 0)
    checkpoint = getattr(args, "checkpoint", None)
    tasks = [
        (
            args.policy,
            args.drive,
            seed,
            args.agents,
            args.cap,
            config_path,
            checkpoint,
        )
        for seed in range(seed_start, seed_start + args.seeds)
    ]
    start = time.monotonic()
    with Pool(processes=max(1, args.workers)) as pool:
        results = pool.map(_eval_worker, tasks)
    wall = time.monotonic() - start
    agg = aggregate(results, wall, args.cap)

    s = agg
    print(
        f"policy={args.policy} drive={args.drive} scenario={args.scenario} agents={args.agents}"
    )
    print(f"seeds={args.seeds} cap={args.cap} workers={args.workers}")
    print(
        f"cap_pct={s['cap_pct']:.1f}% 95%CI=[{s['ci'][0]:.1f},{s['ci'][1]:.1f}] "
        f"median_survival={s['median_survival']}"
    )
    print(
        "death_cause starve={} hazard={} trunc={}".format(
            s["death_cause"]["starvation"],
            s["death_cause"]["hazard"],
            s["death_cause"]["truncation"],
        )
    )
    print(
        f"food_eaten_per_agent={s['food_eaten_per_agent']} "
        f"hazard_contacts_per_1000={s['hazard_contacts_per_1000']} "
        f"mean_energy={s['mean_energy']} food_left_mean={s['food_left_mean']}"
    )
    print(f"wall_time_s={s['wall_time_s']} hash={s['hash']}")
    _print_param_count(args.policy, config_path, checkpoint)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")

    if args.compare:
        baseline = REPO / "docs" / "baselines.json"
        key = scenario_baseline_key(args.scenario, args.agents)
        seed_set = "test" if seed_start >= 1000 else "dev"
        res = compare_to_baselines(
            agg,
            key,
            baseline,
            config_path,
            args.agents,
            args.policy,
            args.allow_stale,
            seed_set=seed_set,
            checkpoint=checkpoint,
            cap=args.cap,
            seeds=list(range(seed_start, seed_start + args.seeds)),
        )
        if res.get("not_comparable"):
            print(f"NOT COMPARABLE: {res['not_comparable']}")
            return 0
        for reason in res["stale"]:
            print(f"STALE: {reason}")
        if res["stale"] and not args.allow_stale:
            print(
                "refusing verdicts (baselines out of date); use --allow-stale to override"
            )
        for entry in res["verdicts"]:
            print(f"vs {entry['against']:<18}: {entry['verdict']}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    from simlab.training.ppo import PPOConfig, PPOTrainer

    cfg_kwargs = {
        "obs_mode": args.obs,
        "total_steps": args.total_steps,
        "workers": args.workers,
        "seed": args.seed,
        "eval_every": args.eval_every,
        "profile": args.profile,
        "update_threads": args.update_threads,
        "update_device": args.update_device,
    }
    if args.gamma is not None:
        cfg_kwargs["gamma"] = args.gamma
    if args.rollout is not None:
        cfg_kwargs["env_steps_per_rollout"] = args.rollout
    cfg = PPOConfig(**cfg_kwargs)
    if args.arch:
        from pathlib import Path

        from simlab.training.arch_spec import (
            default_preset_selection,
            get_spec,
            load_arch_toml,
        )

        arch_path = Path(args.arch)
        if arch_path.is_file():
            cfg.arch = load_arch_toml(arch_path)
        elif get_spec("backbone", args.arch) is not None:
            cfg.arch = default_preset_selection(args.arch, seed=args.seed)
        else:
            cfg.arch = load_arch_toml(arch_path)
        cfg.obs_mode = cfg.arch.get("obs_mode", args.obs)
    trainer = PPOTrainer(cfg, args.out)
    trainer.train()
    print(f"training done -> {args.out}")
    return 0


def run_simulation(steps: int, seed: int | None = None) -> int:
    runner = Runner(load_config(), seed)
    runner.setup()
    initial = {aid: tuple(s["position"]) for aid, s in runner._world._agents.items()}
    for _ in range(steps):
        runner.run_step()
    print(f"steps={runner._step_count} seed={runner._seed}")
    for aid, state in runner._world._agents.items():
        distance = (
            (state["position"][0] - initial[aid][0]) ** 2
            + (state["position"][1] - initial[aid][1]) ** 2
        ) ** 0.5
        print(
            f"{aid} position=({state['position'][0]:.3f},{state['position'][1]:.3f}) distance_traveled={distance:.3f} food_eaten={state['food_eaten']} energy={state['energy']:.3f}"
        )
    print(f"state_hash={runner._world.state_hash()}")
    return 0


def run_benchmark(steps: int, agents: int = 1) -> int:
    runner = Runner(load_config(), 42, agents=agents)
    runner.setup()
    runner.run_bench(steps)
    print(f"agents={agents} steps={steps}")
    print(f"world_only_steps_per_second={runner.last_bench_world_steps_per_second:.1f}")
    print(f"full_pipeline_steps_per_second={runner.last_bench_steps_per_second:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
