"""KL between init and final policies over dev seed 0 states."""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from pathlib import Path
from tools.decay_probe import _make_runner, TRAIN_ROOT
from simlab.training.arch_spec import default_preset_selection, build_policy_cfg
from simlab.training.torch_policy import make_policy

def collect_states(final_ckpt, variant, seed=0, n_states=2000):
    runner = _make_runner(final_ckpt, seed, variant)
    policy = runner._policy
    world = runner._world
    drive_core = runner._drive_core
    policy.reset(5, np.random.default_rng(seed))
    states = []
    for tick in range(512):
        obs_batch = world._observation_batch()
        live = [aid for aid, st in world._agents.items() if st["is_alive"]]
        drives = {}
        if drive_core is not None:
            live_obs = [obs_batch[aid] for aid in live]
            vals = drive_core.step(live_obs, {"rng": world._rng})
            drives = {aid: tuple(float(v) for v in vals[i]) for i, aid in enumerate(live)}
        actions = {}
        for aid in live:
            obs = obs_batch[aid]
            drv = drives.get(aid, ())
            x = policy._encode(obs, drv)
            idx = policy._call_index % policy.n_agents
            h_prev = policy._hidden_t[idx:idx+1].clone()
            if len(states) < n_states:
                states.append((x.clone(), h_prev.clone()))
            act = policy.act(obs, drv)
            actions[aid] = act
        runner._world.step(actions, include_observations=False, include_snapshot=False, include_agent_states=False)
        runner._step_count = runner._world.get_step_count()
        if runner._world._agents and all(not st["is_alive"] for st in runner._world._agents.values()):
            break
        if len(states) >= n_states:
            break
    return states

def kl_for_seed(arm, seed):
    final_ckpt = TRAIN_ROOT / f"{arm}_{seed}" / "final.pt" if arm in ["D", "C", "A"] else None
    # arm dirs are job1_{arm}_{seed} under logs/overnight
    final_ckpt = Path(f"logs/overnight/job1_{arm}_{seed}/final.pt")
    if not final_ckpt.exists():
        print(f"{arm} {seed} no final")
        return None
    variant = "rda_gru_surprise"
    states = collect_states(final_ckpt, variant, seed=0, n_states=2000)
    # init policy
    sel = default_preset_selection(variant, seed=seed)
    cfg = build_policy_cfg(sel)
    init_pol = make_policy(cfg)
    init_pol._backbone._retention.eval()
    init_pol._backbone._net.eval()
    final_pol = make_policy({**cfg, "checkpoint": str(final_ckpt)})
    final_pol._backbone._retention.eval()
    final_pol._backbone._net.eval()
    kls = []
    for x, h in states:
        with torch.no_grad():
            a_init, v_init, _ = init_pol._backbone.frame_step(x, h, None)
            a_final, v_final, _ = final_pol._backbone.frame_step(x, h, None)
        # action distributions: turn Normal(tanh(a0), std_t), thrust Normal(sigmoid(a1), std_th), eat Bernoulli(sigmoid(a2))
        # Use log_std from backbone _net
        std_t = torch.exp(init_pol._backbone._net.log_std_turn)
        std_th = torch.exp(init_pol._backbone._net.log_std_thrust)
        # KL for Normal: 0.5*(log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 -1)
        mu_t_init = torch.tanh(a_init[:,0])
        mu_t_final = torch.tanh(a_final[:,0])
        var_t = std_t**2
        kl_t = 0.5*(torch.log(var_t/var_t) + (var_t + (mu_t_init-mu_t_final)**2)/var_t - 1)  # same var (same std? init vs final std may differ, use respective)
        # Actually std differs between init and final (log_std trained). Use respective vars.
        var_t_init = torch.exp(init_pol._backbone._net.log_std_turn)**2
        var_t_final = torch.exp(final_pol._backbone._net.log_std_turn)**2
        kl_t = 0.5*(torch.log(var_t_final/var_t_init) + (var_t_init + (mu_t_init-mu_t_final)**2)/var_t_final - 1)
        mu_th_init = torch.sigmoid(a_init[:,1])
        mu_th_final = torch.sigmoid(a_final[:,1])
        var_th_init = torch.exp(init_pol._backbone._net.log_std_thrust)**2
        var_th_final = torch.exp(final_pol._backbone._net.log_std_thrust)**2
        kl_th = 0.5*(torch.log(var_th_final/var_th_init) + (var_th_init + (mu_th_init-mu_th_final)**2)/var_th_final - 1)
        p_init = torch.sigmoid(a_init[:,2])
        p_final = torch.sigmoid(a_final[:,2])
        kl_eat = p_init*torch.log((p_init+1e-8)/(p_final+1e-8)) + (1-p_init)*torch.log((1-p_init+1e-8)/(1-p_final+1e-8))
        kls.append(float((kl_t + kl_th + kl_eat).mean().item()))
    return float(np.mean(kls))

for arm, seeds in [("D", [100000,100001,100002,100003,100004,100005]), ("C", [100000,100001,100002,100003,100004,100005]), ("A", [100000,100001,100002])]:
    for seed in seeds:
        kl = kl_for_seed(arm, seed)
        print(f"{arm} {seed} mean KL init->final={kl:.4f}" if kl is not None else f"{arm} {seed} missing")
