#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Profile Arm C vs Arm B' (nanochat RDA experiment) under the base_train.py setup.

Usage:
    python diagnostics\\profile_arm_c.py                  # CUDA: depth 8, seq 1024, device batch 4, micro 8
    python diagnostics\\profile_arm_c.py --dry            # CPU: tiny config, micro 1, no profiler, no JSON
    python diagnostics\\profile_arm_c.py --micro 4 --leakage off --compile

Arm C:  --rda --rda-const-state  (const_state=True)
Arm B': --rda                    (const_state=False)
"""

import argparse
import gc
import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import COMPILE_STEPS
from nanochat.rda import GPTWithRDA, document_start_mask
from nanochat.tokenizer import get_tokenizer

# Mirrors base_train.py argparse defaults and build_model_meta().
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = 32768
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.008
MATRIX_LR = 0.02
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.28
RDA_DRIVE_WD = 0.01
B_REF = 2 ** 19
SEED = 42
PROFILE_STEPS = 5
TOP_K = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Time Arm C vs Arm B' (base_train.py-matched)")
    parser.add_argument("--dry", action="store_true",
                        help="CPU dry-run: tiny config, no profiler, no JSON")
    parser.add_argument("--micro", type=int, default=None,
                        help="forward+backward passes per optimizer step (default 8 GPU, 1 dry)")
    parser.add_argument("--leakage", choices=["on", "off"], default="on",
                        help="check_leakage for GPTWithRDA (default on, matches base_train.py; forced off in --dry)")
    parser.add_argument("--compile", action="store_true",
                        help="apply base_train.py's torch.compile call (only if COMPILE_STEPS)")
    parser.add_argument("--debug-steps", type=int, default=200,
                        help="debug_steps passed to GPTWithRDA for both arms (default 200)")
    parser.add_argument("--only", choices=["bp", "c", "both"], default="both",
                        help="which arm(s) to build and run: bp=Arm B', c=Arm C, both (default both)")
    parser.add_argument("--order", choices=["bp_first", "c_first"], default="bp_first",
                        help="--only both ordering: bp_first or c_first (default bp_first)")
    parser.add_argument("--device-batch", type=int, default=None,
                        help="device batch size (default 4; keeps total_batch=32768 so grad-accum adjusts)")
    parser.add_argument("--no-profile", action="store_true",
                        help="skip the torch.profiler section entirely")
    return parser.parse_args()


def make_config(depth, seq_len, vocab_size):
    # Same arithmetic as base_train.py build_model_meta()
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                     n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
                     window_pattern=WINDOW_PATTERN)


def scaling_params(model):
    # Same as base_train.py get_scaling_params()
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def optimizer_scales(depth, seq_len, vocab_size):
    # Same as base_train.py: batch_lr_scale and scaled weight decay
    batch_ratio = TOTAL_BATCH_SIZE / B_REF
    batch_lr_scale = batch_ratio ** 0.5
    with torch.device("meta"):
        d12 = GPT(make_config(12, seq_len, vocab_size))
        dn = GPT(make_config(depth, seq_len, vocab_size))
    # D_REF / target_tokens reduces to S(d12) / S(depth) because target_param_data_ratio cancels
    wd_scaled = WEIGHT_DECAY * math.sqrt(batch_ratio) * (scaling_params(d12) / scaling_params(dn))
    return batch_lr_scale, wd_scaled


def make_batches(n_micro, batch, seq_len, vocab_size, device):
    gen = torch.Generator().manual_seed(SEED)
    batches = []
    for _ in range(n_micro):
        toks = torch.randint(0, vocab_size, (batch, seq_len + 1), generator=gen)
        x = toks[:, :-1].contiguous().to(device)
        y = toks[:, 1:].contiguous().to(device)
        assert x.shape == (batch, seq_len), f"bad batch shape {tuple(x.shape)}"
        batches.append((x, y))
    return batches


def add_rda_groups(optimizer, rda, const_state, batch_lr_scale):
    # Mirrors the RDA optimizer coverage block in base_train.py
    s = batch_lr_scale
    state_matrix = list(rda.state.film_scale.parameters()) + list(rda.state.film_shift.parameters())
    state_vector = [rda.state.e_init]
    if rda.const_state_vec is not None:
        state_vector = state_vector + [rda.const_state_vec]
    predictor_matrix = list(rda.predictor.parameters())
    drive_matrix = (list(rda.state.event_proj.parameters())
                    + list(rda.state.pe_proj.parameters())
                    + list(rda.state.h_norm.parameters()))
    groups = [
        dict(kind="adamw", params=state_matrix, lr=MATRIX_LR * s,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=drive_matrix, lr=MATRIX_LR * s,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=RDA_DRIVE_WD),
        dict(kind="adamw", params=state_vector, lr=SCALAR_LR * s,
             betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=predictor_matrix, lr=MATRIX_LR * s,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for g in groups:
        g["initial_lr"] = g["lr"]
        optimizer.add_param_group(g)

    grad_free = {id(rda.state.e_init)} | {id(p) for p in rda.predictor.parameters()}
    if const_state:
        grad_free |= ({id(p) for p in rda.state.event_proj.parameters()}
                      | {id(p) for p in rda.state.pe_proj.parameters()}
                      | {id(p) for p in rda.state.h_norm.parameters()})
    for g in optimizer.param_groups:
        if any(id(p) in grad_free for p in g["params"]):
            g["grad_free"] = True


def build_arm(const_state, cfg, device, leakage, use_compile, batch_lr_scale, wd_scaled,
              debug_steps):
    # Same construction order as base_train.py: base optimizer -> RDA wrap -> RDA groups
    torch.manual_seed(SEED)
    with torch.device("meta"):
        orig = GPT(cfg)
    orig.to_empty(device=device)
    orig.init_weights()
    base = torch.compile(orig, dynamic=False) if use_compile else orig
    optimizer = base.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR * batch_lr_scale,
        embedding_lr=EMBEDDING_LR * batch_lr_scale,
        scalar_lr=SCALAR_LR * batch_lr_scale,
        matrix_lr=MATRIX_LR * batch_lr_scale,
        weight_decay=wd_scaled,
    )
    # The RDA wrapper is built on the uncompiled model, as in base_train.py
    rda = GPTWithRDA(orig, const_state=const_state, const_state_seed=SEED,
                     debug_steps=debug_steps,
                     check_leakage=(leakage == "on")).to(device)
    add_rda_groups(optimizer, rda, const_state, batch_lr_scale)
    return rda, optimizer


def prime_grad_free(optimizer):
    # MuonAdamW needs a grad on every enrolled param; re-done on every backward as in base_train.py
    for g in optimizer.param_groups:
        if g.get("grad_free"):
            for p in g["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_step(rda, optimizer, batches, bos, device, carry):
    """One optimizer step = N x (forward+backward) then one optimizer.step().

    Returns (carry, micro_forward_ms list, micro_backward_ms list, optimizer_ms, total_ms).
    """
    n_micro = len(batches)
    fwd_ms, bwd_ms = [], []
    sync(device)
    t_start = time.perf_counter()
    for x, y in batches:
        sync(device)
        t0 = time.perf_counter()
        out = rda(x, y, E_t_external=carry, doc_starts=document_start_mask(x, bos))
        loss = out["loss"]
        carry = out["E_next"].detach()
        sync(device)
        t1 = time.perf_counter()
        (loss / n_micro).backward()
        prime_grad_free(optimizer)
        sync(device)
        t2 = time.perf_counter()
        fwd_ms.append((t1 - t0) * 1000.0)
        bwd_ms.append((t2 - t1) * 1000.0)
    sync(device)
    t3 = time.perf_counter()
    optimizer.step()
    sync(device)
    t4 = time.perf_counter()
    rda.zero_grad(set_to_none=True)
    sync(device)
    t5 = time.perf_counter()
    return carry, fwd_ms, bwd_ms, (t4 - t3) * 1000.0, (t5 - t_start) * 1000.0


def stats(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max()),
            "values": [float(v) for v in arr]}


def self_cuda_us(evt):
    return float(getattr(evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0.0)))


def profile_arm(rda, optimizer, batches, bos, device, carry):
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        for _ in range(PROFILE_STEPS):
            carry, *_ = timed_step(rda, optimizer, batches, bos, device, carry)
    sync(device)
    avgs = prof.key_averages()

    def fmt(e):
        return {"name": e.key, "self_cuda_ms": self_cuda_us(e) / 1000.0,
                "self_cpu_ms": float(e.self_cpu_time_total) / 1000.0, "count": int(e.count)}

    top_cuda = [fmt(e) for e in sorted(avgs, key=self_cuda_us, reverse=True)[:TOP_K]]
    top_cpu = [fmt(e) for e in sorted(avgs, key=lambda e: float(e.self_cpu_time_total), reverse=True)[:TOP_K]]
    return top_cuda, top_cpu


def print_top(title, rows, key):
    print(f"  top {TOP_K} by self {title}:")
    for r in rows:
        print(f"    {r['name'][:55]:<55} {r[key]:10.3f} ms  x{r['count']}")


def run_arm(name, const_state, cfg, device, leakage, use_compile, scales, batches, bos,
            n_warmup, n_timed, do_profile, debug_steps, no_profile=False):
    batch_lr_scale, wd_scaled = scales
    rda, optimizer = build_arm(const_state, cfg, device, leakage, use_compile,
                               batch_lr_scale, wd_scaled, debug_steps)
    rda.train()
    carry = None
    for _ in range(n_warmup):
        carry, *_ = timed_step(rda, optimizer, batches, bos, device, carry)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fwd, bwd, opt, tot = [], [], [], []
    for _ in range(n_timed):
        carry, f, b, o, t = timed_step(rda, optimizer, batches, bos, device, carry)
        fwd += f
        bwd += b
        opt.append(o)
        tot.append(t)

    res = {"forward_per_micro": stats(fwd), "backward_per_micro": stats(bwd),
           "optimizer_per_step": stats(opt), "total_per_step": stats(tot)}
    fwd_m, bwd_m = res["forward_per_micro"]["mean"], res["backward_per_micro"]["mean"]
    print(f"[{name}] forward  per micro-step mean: {fwd_m:.2f} ms  median: {res['forward_per_micro']['median']:.2f} ms  (n={len(fwd)})")
    print(f"[{name}] backward per micro-step mean: {bwd_m:.2f} ms  median: {res['backward_per_micro']['median']:.2f} ms  (n={len(bwd)})")
    print(f"[{name}] optimizer per step mean:      {res['optimizer_per_step']['mean']:.2f} ms  median: {res['optimizer_per_step']['median']:.2f} ms")
    print(f"[{name}] total per optimizer step mean: {res['total_per_step']['mean']:.2f} ms  median: {res['total_per_step']['median']:.2f} ms")
    if bwd_m < 0.5 * fwd_m:
        print("!" * 72)
        print(f"WARNING [{name}]: mean backward {bwd_m:.2f} ms < 0.5 x mean forward "
              f"{fwd_m:.2f} ms. Timing is probably wrong.")
        print("!" * 72)

    if do_profile and not no_profile:
        top_cuda, top_cpu = profile_arm(rda, optimizer, batches, bos, device, carry)
        res["top_self_cuda"], res["top_self_cpu"] = top_cuda, top_cpu
        print_top("CUDA time", top_cuda, "self_cuda_ms")
        print_top("CPU time (with call counts)", top_cpu, "self_cpu_ms")

    if device.type == "cuda":
        res["max_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        res["memory_reserved_mib"] = torch.cuda.memory_reserved() / 2 ** 20
        print(f"[{name}] torch.cuda.max_memory_allocated: {res['max_memory_allocated_mib']:.1f} MiB")
        print(f"[{name}] torch.cuda.memory_reserved     : {res['memory_reserved_mib']:.1f} MiB")
    else:
        res["max_memory_allocated_mib"] = None
        res["memory_reserved_mib"] = None
    print()
    return res


def main():
    args = parse_args()

    # Resolve --device-batch
    if args.device_batch is not None:
        batch = args.device_batch
    elif args.dry:
        batch = 1
    else:
        batch = 4

    if args.dry:
        depth, seq_len = 2, 128
        n_warmup, n_timed = 1, 2
        micro = args.micro if args.micro is not None else 1
        leakage = "off"
        device = torch.device("cpu")
        print("WARNING: --dry forces --leakage off (temporal leakage check on tiny config)")
    else:
        depth, seq_len = 8, 1024
        n_warmup, n_timed = 5, 20
        micro = args.micro if args.micro is not None else 8
        leakage = args.leakage
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available; run with --dry for CPU.")
            sys.exit(1)
        device = torch.device("cuda")
    if micro < 1:
        print("ERROR: --micro must be >= 1")
        sys.exit(1)

    if args.compile and not COMPILE_STEPS:
        print("compile not available, eager")
    use_compile = args.compile and COMPILE_STEPS
    if use_compile:
        print("compile: torch.compile(dynamic=False) on base GPT; RDA wraps uncompiled base (as base_train.py)")

    torch.manual_seed(SEED)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    bos = tokenizer.get_bos_token_id()
    cfg = make_config(depth, seq_len, vocab_size)
    scales = optimizer_scales(depth, seq_len, vocab_size)
    batches = make_batches(micro, batch, seq_len, vocab_size, device)
    grad_accum = TOTAL_BATCH_SIZE // (batch * seq_len)

    print(f"Mode={'dry' if args.dry else 'cuda'} device={device} depth={depth} "
          f"seq_len={seq_len} device_batch={batch} micro={micro} leakage={leakage} "
          f"warmup={n_warmup} timed={n_timed} only={args.only} order={args.order}")
    print(f"total_batch={TOTAL_BATCH_SIZE} (base_train grad_accum={grad_accum}, "
          f"--micro {micro} used) batch_lr_scale={scales[0]:.4f} "
          f"weight_decay_scaled={scales[1]:.6f}")
    print()

    # Build arm list respecting --order
    arm_order = [("B'", False), ("C", True)] if args.order == "bp_first" else [("C", True), ("B'", False)]
    arms_to_run = [(n, cs) for n, cs in arm_order if args.only == "both" or
                   (args.only == "bp" and n == "B'") or
                   (args.only == "c" and n == "C")]

    results = {}
    try:
        for name, const_state in arms_to_run:
            results[name] = run_arm(name, const_state, cfg, device, leakage, use_compile,
                                    scales, batches, bos, n_warmup, n_timed,
                                    do_profile=not args.dry,
                                    debug_steps=args.debug_steps,
                                    no_profile=args.no_profile)
            # Arm model and optimizer are local to run_arm and are freed on return
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except AssertionError as e:
        print(f"STOP: AssertionError during forward: {e}")
        sys.exit(2)

    if not args.dry:
        out_path = os.path.join(PROJECT_ROOT, "logs", "profile_arm_c.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {"config": {"depth": depth, "seq_len": seq_len, "device_batch": batch,
                              "micro": micro, "total_batch": TOTAL_BATCH_SIZE,
                              "batch_lr_scale": scales[0], "weight_decay_scaled": scales[1]},
                   "warmup": n_warmup, "timed": n_timed, "profile_steps": PROFILE_STEPS,
                   "leakage": leakage, "compile": bool(use_compile),
                   "no_profile": bool(args.no_profile), "arms": results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")
    else:
        print("Dry run complete; JSON not written.")


if __name__ == "__main__":
    main()
