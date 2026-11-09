"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import subprocess
import shutil
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core

# Repo root, used to locate the pre-flight regression test.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
parser.add_argument("--segment-len", type=int, default=0,
                    help="restrict attention to causal segments of this length (0 = off)")
parser.add_argument("--synthetic-recall-frac", type=float, default=0.0,
                    help="fraction of batch rows sourced from the synthetic recall generator (0 = off)")
parser.add_argument("--recall-answer-weight", type=float, default=1.0,
                    help="loss multiplier for synthetic recall answer positions (1 = off)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Reproducibility
parser.add_argument("--seed", type=int, default=42,
                    help="seed for model weight initialization. Default 42 is the "
                         "value that was previously hardcoded in compute_init(), so "
                         "omitting this reproduces the old behavior exactly. NOTE: "
                         "this does not reseed the dataloader, which uses its own "
                         "explicit rng objects.")
# Regulatory Drive Architecture / RDA (opt-in; off = original training path, unchanged)
parser.add_argument("--rda", action="store_true",
                    help="wrap the model in GPTWithRDA and carry regulatory drive state across micro-steps within an episode")
def parse_rda_decays(value):
    try:
        decays = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected 8 comma-separated decay values") from exc
    if len(decays) != 8 or any(not 0.0 <= decay < 1.0 for decay in decays):
        raise argparse.ArgumentTypeError(
            "expected exactly 8 decay values, each in [0, 1)"
        )
    return decays

parser.add_argument("--rda-decays", type=parse_rda_decays, default=None,
                    help="optional comma-separated decay values, one per RDA drive")
parser.add_argument("--rda-const-state", action="store_true",
                    help="CONTROL ARM C: keep every RDA module, parameter and the "
                         "recurrence exactly as in --rda, but hand FiLM a LEARNED "
                         "CONSTANT vector (nn.Parameter of shape [8], init N(0,0.3^2)) "
                         "broadcast over batch and positions instead of the "
                         "per-position state. FiLM then carries no text-dependent "
                         "and no position-dependent information. Requires --rda; "
                         "has NO effect without it. Default OFF.")
parser.add_argument("--rda-debug-steps", type=int, default=200,
                    help="number of steps to log E_t / E_hat_next / E_next / dpe norms")
parser.add_argument("--rda-drive-wd", type=float, default=0.01,
                    help="weight decay on event_proj/pe_proj only, bounding drive magnitude. "
                         "Independent of pe_reference_scale (which is input-side).")
# DEPRECATED ALIASES. The project was renamed from "primal state" to
# "Regulatory Drive Architecture"; these keep older run scripts and notes
# working instead of failing with an opaque "unrecognized arguments" error.
parser.add_argument("--primal-state", dest="rda", action="store_true",
                    help=argparse.SUPPRESS)
parser.add_argument("--primal-debug-steps", dest="rda_debug_steps", type=int,
                    default=argparse.SUPPRESS, help=argparse.SUPPRESS)
parser.add_argument("--primal-drive-wd", dest="rda_drive_wd", type=float,
                    default=argparse.SUPPRESS, help=argparse.SUPPRESS)
args = parser.parse_args()
if args.segment_len < 0:
    parser.error("--segment-len must be 0 or a positive integer")
if not 0.0 <= args.synthetic_recall_frac <= 1.0:
    parser.error("--synthetic-recall-frac must be in [0, 1]")
if not math.isfinite(args.recall_answer_weight) or args.recall_answer_weight < 0.0:
    parser.error("--recall-answer-weight must be finite and nonnegative")
if args.synthetic_recall_frac > 0.0 and args.max_seq_len != 1024:
    parser.error("--synthetic-recall-frac requires --max-seq-len 1024")
_deprecated = [f for f in ("--primal-state", "--primal-debug-steps",
                            "--primal-drive-wd") if f in sys.argv]
if _deprecated:
    # print0 is not imported this early in the file, so warn on stderr.
    print("WARNING: " + ", ".join(_deprecated) + " is deprecated; use the --rda* "
          "names instead. Mapping to the new flags.", file=sys.stderr)
user_config = vars(args).copy()  # for logging

# TREE STATE, printed at the TOP of the log before anything else. A run that
# started on a dirty tree cannot be tied to its commit hash alone, so this is
# the first thing in the log rather than something you have to go looking for.
def _tree_state():
    import subprocess
    def g(*a):
        try:
            return subprocess.check_output(["git"] + list(a), stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except Exception as e:
            return "ERROR: %s" % e
    import hashlib
    st = g("status", "--porcelain")
    # Ignore manifests/: the run driver writes this run's OWN manifest a
    # moment before training starts, so listing it would make every run
    # report DIRTY and train the reader to ignore the line. Only tracked
    # SOURCE should decide whether this run executed the committed code.
    st = "\n".join(l for l in st.splitlines() if "manifests/" not in l)
    h = hashlib.sha256()
    h.update(g("diff", "HEAD").encode())
    h.update(b"\n--MANIFESTS-EXCLUDED--\n")
    return st, h.hexdigest()

_status, _diff_sha = _tree_state()
import subprocess as _sp
_commit = _sp.check_output(["git", "rev-parse", "HEAD"],
                           stderr=_sp.DEVNULL, text=True).strip()
if not _status:
    print("TREE: CLEAN  commit=%s  diff_sha256=%s (empty diff)"
          % (_commit, _diff_sha))
else:
    print("TREE: DIRTY -- this run did NOT execute the committed code alone.")
    print("  commit=%s" % _commit)
    print("  Uncommitted changes present at start:")
    for _line in _status.splitlines():
        print("    %s" % _line)
    print("  diff_sha256=%s" % _diff_sha)
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, args.seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern, segment_len=args.segment_len,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    rng_path = os.path.join(checkpoint_dir, f"rng_{args.resume_from_step:06d}_rank{ddp_rank:d}.pt")
    rng_data = torch.load(rng_path, map_location="cpu")
    # RDA checkpoints contain the wrapper's full state dict (base.*, state.*,
    # predictor.*), so defer model loading until after GPTWithRDA is built.
    if not args.rda:
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data
else:
    rng_data = None

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

# Windows/minimal-CPU environments: torch.compile's inductor backend needs Triton,
# which has no official Windows wheel. Reuse the same detection as the optimizer so
# we fall back to eager mode rather than crashing. Performance only, not numerics.
# Can also be forced off with NANOCHAT_COMPILE=0.
from nanochat.optim import COMPILE_STEPS as do_compile
if not do_compile:
    print0("WARNING: torch.compile disabled (Triton unavailable on this platform); running eager. Slower, same numerics.")

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) if do_compile else model # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# --------------------------------------------------------------------------------
# Affective / regulatory drive state wiring (opt-in via --RDA)
#
# Off by default: with the flag absent the training loop below is byte-for-byte
# the original `loss = model(x, y)` path and no RDA module is built.
#
# When enabled, GPTWithRDA wraps the base model and uses DECAY-ONLY
# PERSISTENCE: the affective state is carried across every training step and is
# never hard-reset at a document boundary. `rda_carry_state` is None only
# until the first step produces a state (i.e. step 0 uses E_init exactly once).
# RDAStateModule's `decay` is the sole mechanism controlling how much prior
# state survives into a new document.
rda_active = args.rda
rda_carry_state = None
rda_carry_counts = {"carry": 0, "reset": 0}
rda_doc_start_counts = []
# FiLM movement tracking. If these stay near zero the regulatory drive state never
# actually influences the model and any A/B difference is not a test of the
# hypothesis. Sampled at a few milestones rather than every step.
rda_film_checkpoints = {0, 150, 300, 450, 600}
rda_film_log = []
# Per-dimension E_t saturation trace. Sampled at the same milestones as the
# FiLM trace, so a table can be read off the run without extra machinery.
rda_sat_checkpoints = {0, 150, 300, 450, 600}
rda_sat_log = []
# ABORT RULE. A drive dimension pinned at |E| > 0.98 for 10 consecutive steps
# is saturated: the tanh has stopped responding, that dimension has stopped
# carrying information, and any loss improvement from then on is not
# attributable to the drive. Abort rather than report a number that looks
# fine and is actually measuring a stuck state.
rda_sat_abort_dim = rda_sat_abort_steps = 10
rda_sat_streak = 0
rda_sat_worst_dim = -1
rda_sat_dims = []
# TEST-ONLY HOOK. Unset in every real run; set only by
# diagnostics/test_saturation_abort.py to force the abort and prove the rule
# actually fires. It can only INFLATE |E|, so leaving it on would abort every
# run; it cannot mask saturation. Verified inert when unset.
rda_force_sat = os.environ.get("RDA_FORCE_SAT", "")

# The state.* group LR multiplier, applied per-step at the rescale below as
# `group["lr"] = initial_lr * lrm * _rda_diag_lr_scale`.
#
# IT MUST BE INITIALISED HERE, OUTSIDE `if rda_active:`. The rescale that
# consumes it runs on EVERY optimizer step of EVERY run, including the
# baseline arm that never passes --rda, so a definition that only exists
# inside the RDA block made the BASELINE ARM die with
#   NameError: name '_rda_diag_lr_scale' is not defined
# at the first optimizer step. Both halves arrived together in ab9b560: the
# assignment was indented under `if rda_active:` while the use stayed
# unconditional in the shared training loop.
#
# 1.0 is the documented default and is exactly the pre-ab9b560 behaviour
# (`initial_lr * lrm`), so a baseline run's learning rates are UNCHANGED by
# this fix. RDA_STATE_LR still overrides it inside the RDA block below.
_rda_diag_lr_scale = 1.0

if rda_active:
    from nanochat.rda import GPTWithRDA, document_start_mask, DRIVE_DIMS
    bos_token_id = tokenizer.get_bos_token_id()

    # ---------------------------------------------------------------------
    # MANDATORY PRE-FLIGHT GATE: no label leak.
    #
    # The dataloader makes y[p] == x[p+1], and this script's loss is
    # cross_entropy(logits[p], targets[p]) with no shift, so y[p] is the
    # label attached to position p's logits. rda.py used to build pe_seq
    # from targets[:, 1:], which made film_state[p] a function of y[p]: the
    # drive state could see the very token it was being graded on. The model
    # still trained, the loss still fell, and val BPB came out BETTER than
    # baseline -- so the failure was completely silent, and the B'-s42
    # canary's 1.226305 (vs baseline 1.301217) was an artifact of it.
    #
    # Checking in the entry point rather than only in run_sweep.ps1 because
    # that file is gitignored: a gate that does not travel with the repo is
    # not a gate. RDA_NO_LEAK_CHECK=1 exists only for debugging the test
    # itself and must not be used for a reported run.
    # ---------------------------------------------------------------------
    if os.environ.get("RDA_NO_LEAK_CHECK", "") != "1":
        _leak_rc = subprocess.call(
            [sys.executable, os.path.join(REPO, "diagnostics", "test_no_label_leak.py")]
            + (["--rda-decays", ",".join(map(str, args.rda_decays))]
               if args.rda_decays is not None else []),
            cwd=REPO,
        )
        if _leak_rc != 0:
            print0("")
            print0(f"ABORT: test_no_label_leak.py FAILED (exit {_leak_rc}).")
            print0("The RDA drive state can see the label it is scored on.")
            print0("Every metric from this run would be invalid. Refusing to start.")
            sys.exit(1)
        print0("PRE-FLIGHT: label-leak regression test PASSED")
    else:
        print0("WARNING: RDA_NO_LEAK_CHECK=1 -- label-leak gate SKIPPED")
        print0("WARNING: this run is NOT valid for reporting.")

    # Wrap the base model. The state module and predictor need real gradients,
    # so they stay in eager mode.
    #
    # --rda-const-state is OPT-IN and DEFAULT OFF. When it is off,
    # `const_state=False` is passed and GPTWithRDA sets
    # `self.const_state_vec = None`, so the forward pass uses `film_state`
    # verbatim and is BIT-IDENTICAL to plain --rda. Guarded so that passing the
    # flag without --rda is a loud error rather than a silent no-op.
    if args.rda_const_state and not args.rda:
        print0("ABORT: --rda-const-state requires --rda (it is a control arm for it).")
        sys.exit(1)
    rda_model = GPTWithRDA(orig_model, debug_steps=args.rda_debug_steps,
                           const_state=args.rda_const_state,
                           const_state_seed=args.seed,
                           decay=args.rda_decays if args.rda_decays is not None else 0.95)
    rda_model = rda_model.to(device)
    model = rda_model
    if resuming:
        # The optimizer was created from orig_model before the wrapper was
        # built. Copy in place so assign=True cannot replace its Parameter
        # objects with stale optimizer references.
        model.load_state_dict(model_data, strict=True, assign=False)
        del model_data
    _decay_label = (", ".join("%.8g" % v for v in args.rda_decays)
                    if args.rda_decays is not None else "0.95")
    print0(f"RDA ENABLED | n_drives=8 | decay={_decay_label} | persistence=decay-only | debug_steps={args.rda_debug_steps}")
    # STATE THE PERSISTENCE SEMANTICS EXPLICITLY. The 1/e horizon is the number
    # that says how long the state actually remembers, and at decay=0.95 it is
    # ~19.5 POSITIONS -- not 1024 tokens and not one document. So the decay
    # implements within-sequence inertia; the "carries across documents" reading
    # of decay-only persistence is NOT what this value provides (a doc boundary
    # inside a packed row attenuates the carried state by 0.95^256 = 2e-6).
    # Printing it here means no run record can be read as claiming a persistence
    # the configuration does not provide. See research/CODE_REVIEW.md F3.
    if args.rda_decays is None:
        print0("  persistence: 1/e horizon = %.1f positions "
               "(= %.1f%% of a %d-token row); carried state reaches t=100 of a row at %.1e"
               % (rda_model.state.decay_horizon,
                  100.0 * rda_model.state.decay_horizon / args.max_seq_len,
                  args.max_seq_len,
                  0.95 ** 100))
    else:
        print0("  per-drive 1/e horizons: %s positions"
               % ", ".join("%.1f" % horizon
                           for horizon in rda_model.state.decay_horizon))
    if args.rda_const_state:
        _cv = rda_model.const_state_vec
        print0("RDA CONTROL ARM C | FiLM input = LEARNED CONSTANT (no state information)")
        print0("  const_state_vec shape=%s init=%s" % (
            list(_cv.shape), ["%+.6f" % v for v in _cv.detach().tolist()]))
        print0("  NOTE: +8 parameters vs B'. The drive/recurrence still runs but "
               "is now UNTRAINED (no gradient path from the loss).")

    # ---------------------------------------------------------------------
    # Optimizer coverage for the RDA parameters.
    #
    # GPT.setup_optimizer only enumerates the BASE model's parameters, and
    # asserts its own groups sum to len(self.parameters()) -- so the wrapper's
    # state.* / predictor.* parameters were silently left unoptimized and FiLM
    # stayed exactly 0.0 for the whole run. Enroll them here.
    #
    # HYPERPARAMETERS (deliberately reusing the base scheme, not inventing one):
    #   film_* are Linear MATRIX weights, so they follow the matrix path's LR
    #   (args.matrix_lr * batch_lr_scale) with NO weight decay -- they start at
    #   exactly 0 and are meant to grow freely. event_proj / pe_proj follow the
    #   same LR but get their own small weight decay (see below). All are
    #   routed through AdamW (kind='adamw') rather than Muon because Muon is
    #   shape-bucketed across the base trunk and mixing these 8-wide state
    #   layers in would perturb the base model's own groups. e_init is a
    #   vector, so it follows the scalar LR (args.scalar_lr * batch_lr_scale)
    #   with the x0-style zero decay.
    # ---------------------------------------------------------------------
    # FiLM only -- the conditioning path. The drive projections are split out
    # below into their own weight-decayed group.
    _state_matrix = (list(rda_model.state.film_scale.parameters())
                     + list(rda_model.state.film_shift.parameters()))
    _state_vector = [rda_model.state.e_init]
    # CONTROL ARM C: the learned constant that replaces the per-position state
    # at FiLM. It MUST be enrolled in a group or the coverage assertion below
    # raises, and it MUST be enrolled for real: it is the one arm-C parameter
    # that actually receives gradient, so leaving it out would freeze FiLM's
    # only input. Same kind/scalar-LR treatment as e_init, weight_decay 0.
    if rda_model.const_state_vec is not None:
        _state_vector = _state_vector + [rda_model.const_state_vec]
    _predictor_matrix = list(rda_model.predictor.parameters())

    # event_proj / pe_proj get their OWN group with an explicit, small weight
    # decay (--rda-drive-wd, default 0.01), separate from every other group.
    #
    # WHY A SEPARATE GROUP RATHER THAN MORE INPUT SCALING. The drive is
    #     drive = tanh( event_proj(h) + pe_proj(pe / pe_reference_scale) )
    # and the recurrence pins a dimension at a rail once |drive| exceeds
    # atanh(0.99) = 2.6467. Measured over 30 fresh steps, event_proj outputs
    # already sit at mean |.| = 2.11 and reach 4.11, while pe_proj sits at
    # 0.85 -- so event_proj alone is past the threshold 26 of those 30 steps,
    # and the state ratchets to |E| = sqrt(8) within ~130 steps.
    #
    # Shrinking pe_reference_scale would tune an INPUT magnitude and leave
    # event_proj(h) -- which depends on the learned hidden state h, not on the
    # PE normalization at all -- free to grow without bound. Decaying the
    # weights themselves constrains the magnitude directly and independently of
    # how large h happens to be. These are two different levers; this is the
    # weight-side one, and it is deliberately separate from pe_reference_scale
    # so a future input-side change cannot silently undo it.
    # ---------------------------------------------------------------------
    # h_norm (LayerNorm in front of event_proj) must be enrolled too, or the
    # coverage assertion below raises: it is a state.* parameter and would
    # otherwise silently never update. It goes in the drive group so it
    # inherits the same LR and weight decay as the projection it feeds.
    _drive_matrix = (list(rda_model.state.event_proj.parameters())
                     + list(rda_model.state.pe_proj.parameters())
                     + list(rda_model.state.h_norm.parameters()))

    _dmodel_lr_scale = (model_config_kwargs["n_embd"] / 768) ** -0.5
    # NOTE: the training loop rescales every group as
    #   group["lr"] = group["initial_lr"] * lrm
    # so each group added here MUST carry its own initial_lr or the very first
    # step raises KeyError("initial_lr"). GPT.setup_optimizer() sets this field
    # on its own groups; groups added afterwards do not inherit it.
    _rda_groups = [
        # FiLM: conditioning path. No weight decay -- these start at exactly 0
        # and are meant to grow freely into whatever scale is useful.
        dict(kind='adamw', params=_state_matrix,
             lr=args.matrix_lr * batch_lr_scale,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        # Drive: magnitude-bounded by explicit weight decay (see above).
        dict(kind='adamw', params=_drive_matrix,
             lr=args.matrix_lr * batch_lr_scale,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=args.rda_drive_wd),
        dict(kind='adamw', params=_state_vector,
             lr=args.scalar_lr * batch_lr_scale,
             betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=_predictor_matrix,
             lr=args.matrix_lr * batch_lr_scale,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for _pg in _rda_groups:
        _pg["initial_lr"] = _pg["lr"]
        optimizer.add_param_group(_pg)

    # ---- DIAGNOSTIC-ONLY HOOKS (env-gated, inert by default) -------------
    # RDA_DIAG=<step> logs the per-drive decomposition at each of those
    # optimizer steps, so the drive trajectory can be attributed to either
    # the INITIALIZATION or the GRADIENT.
    # RDA_DISABLE_SAT=1 suppresses the saturation abort so a short
    # diagnostic run can continue past the rail.
    _rda_diag_steps = set()
    _rda_diag_prev = {}
    if os.environ.get("RDA_DIAG"):
        _rda_diag_steps = {int(s) for s in os.environ["RDA_DIAG"].split(",")}
        print0("DIAG: logging per-drive decomposition at steps %s"
               % sorted(_rda_diag_steps))
    if os.environ.get("RDA_DISABLE_SAT"):
        rda_sat_abort_steps = 10 ** 9   # effectively never
        print0("DIAG: saturation abort DISABLED (RDA_DISABLE_SAT)")
    # RDA_STATE_LR=<f> scales the learning rate of every state.* group.
    # Applied as a MULTIPLIER inside the per-step rescale
    # (`group["lr"] = initial_lr * lrm * _rda_diag_lr_scale`) rather than by
    # editing initial_lr, so it composes correctly with lrm and survives
    # every step. 1.0 (the default) is exactly the old behaviour.
    # NOTE: the `= 1.0` default is initialised UNCONDITIONALLY above the
    # `if rda_active:` block, because the rescale that consumes it also runs
    # for the baseline arm. Only the override is RDA-specific.
    if os.environ.get("RDA_STATE_LR"):
        _rda_diag_lr_scale = float(os.environ["RDA_STATE_LR"])
        print0("DIAG: state.* group lr multiplier = %g" % _rda_diag_lr_scale)

    # ---- coverage assertion BY NAME (in addition to the count assert) ----
    _covered = {id(p) for g in optimizer.param_groups for p in g["params"]}
    _missing = [n for n, p in rda_model.named_parameters()
                if n.startswith("state.") or n.startswith("predictor.")
                if id(p) not in _covered]
    if _missing:
        raise RuntimeError(
            "OPTIMIZER COVERAGE GAP: these RDA parameters are in no "
            f"optimizer param group and would never update: {_missing}"
        )

    # ---- skip-list for params that legitimately receive NO gradient -------
    #
    # MuonAdamW._compute_adamw calls grad.float() unconditionally, so a param
    # with grad=None crashes the step. Two RDA params are genuinely
    # grad-free:
    #   predictor.*         - dpe is computed and logged but NOT
    #                         consumed (out of scope by design), so nothing in
    #                         the loss depends on the predictor. Autograd PRUNES
    #                         these: p.grad is None, not a zero tensor.
    #   state.e_init        - see the long note below. Structurally learnable,
    #                         but in practice frozen at its zero init.
    # They are still ENROLLED (so coverage holds and they update if that ever
    # changes) but carry weight_decay=0 and get a zero grad each backward, so
    # the step is a no-op on them until something in the loss actually
    # consumes them.
    #
    # -------------------------------------------------------------------------
    # WHY state.e_init IS IN THIS LIST. The earlier comment here read "the loss
    # has no gradient path to it", which is WRONG. The true mechanism is a
    # COMBINATION of two facts, and only together do they hold at every step:
    #
    #   1. There IS a gradient path, but only when E_t_external is None:
    #          e_init -> init_state() -> E_t -> film_state -> modulate() -> loss
    #      init_state() is differentiable (unsqueeze/expand/contiguous ON
    #      e_init, so the graph is retained). At step 0 that path is live but
    #      yields EXACTLY 0.0, because film_scale/film_shift are zero-init and
    #      modulate() is therefore the exact identity.
    #   2. From step 1 onward the trainer ALWAYS supplies a carry
    #      (E_t_external=rda_carry_state) and forward() DETACHES it. A detached
    #      tensor carries no graph, so e_init is unreachable from the loss and
    #      autograd PRUNES it: p.grad is None from step 1 onward.
    #
    # Measured over a 5-step loop reproducing the trainer's exact call pattern:
    #     e_init per-step |grad| = ['0.000e+00', 'None', 'None', 'None', 'None']
    #     e_init after 5 steps   = [0,0,0,0,0,0,0,0]   max|delta| = 0.0
    #
    # So the practical status is: STRUCTURALLY LEARNABLE, but frozen at its zero
    # init in every real run. Zero-priming it is therefore harmless and this
    # list entry is CORRECT. It is not "no gradient path" -- it is "the only
    # step with a path has a zero gradient, and every later step has no path".
    #
    # CONSEQUENCE IF THIS EVER CHANGES. If `decay` were raised so a carried state
    #actually mattered (see docs/decay_semantics.md), a run could be launched
    # in which E_t_external=None holds a MEANINGFUL state, and e_init would then
    # be the thing that seeds it and MUST be allowed to train. Any such change
    # has to revisit this entry.
    _grad_free = {id(rda_model.state.e_init)} | {
        id(p) for p in rda_model.predictor.parameters()}
    # CONTROL ARM C ONLY. When the FiLM input is a learned constant, nothing in
    # the loss depends on E any more, so the drive projections get NO gradient
    # at all (verified: autograd prunes them, p.grad is None, not a zero
    # tensor). MuonAdamW._compute_adamw calls grad.float() unconditionally and
    # crashes on grad=None (optim.py:73), so they must be listed here or arm C
    # dies on the first optimizer step.
    #
    # IMPORTANT: listing them here makes the step SAFE, not FROZEN. `grad_free`
    # is read ONLY by the zero-grad priming at base_train.py:935-940 --
    # optim.py never sees the key. These params stay in the state.* group with
    # weight_decay=args.rda_drive_wd, and adamw_step_fused applies decoupled
    # decay unconditionally (optim.py:75, p32.mul_(1 - lr*wd)) before it ever
    # looks at the gradient. Measured over 3 steps at lr=3e-3, wd=0.01: every
    # one of the six tensors shrank by exactly (1-3e-5)^3, matching the
    # predicted scale 0.999910003 to 1.8e-7. See
    # diagnostics/probe_arm_c_decay.py. So "grad-free" here means "gets no
    # update", NOT "stays bit-identical to init".
    #
    # SCOPE: guarded on the flag, so with --rda-const-state OFF this list is
    # BYTE-FOR-BYTE the same two entries as before and baseline and B' are
    # unaffected.
    if args.rda_const_state:
        _grad_free = _grad_free | {id(p) for p in rda_model.state.event_proj.parameters()} \
                                   | {id(p) for p in rda_model.state.pe_proj.parameters()} \
                                   | {id(p) for p in rda_model.state.h_norm.parameters()}
    for _g in optimizer.param_groups:
        if any(id(p) in _grad_free for p in _g["params"]):
            _g["grad_free"] = True

    # NOTE: nn.Module.parameters() is a generator, so it needs len(list(...)).
    print0(f"RDA optimizer coverage: {len(list(rda_model.state.parameters()))} state + "
           f"{len(list(rda_model.predictor.parameters()))} predictor params enrolled "
           f"({len(_state_matrix) + len(_drive_matrix) + len(_state_vector) + len(_predictor_matrix)} tensors); "
           f"weight decay {args.rda_drive_wd} on event_proj/pe_proj; "
           f"{len(_grad_free)} grad-free by design (e_init, predictor.*)")
else:
    rda_model = None
    if resuming:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data


# Go!
if resuming and args.rda:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data
if resuming and rng_data is not None:
    torch.set_rng_state(rng_data["torch_cpu"])
    if torch.cuda.is_available() and rng_data.get("torch_cuda"):
        torch.cuda.set_rng_state_all(rng_data["torch_cuda"])
    del rng_data
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        # ------------------------------------------------------------------
        # VALIDATION STATE MODE (RDA). Decided explicitly, not by accident.
        #
        # THE BUG THIS FIXES. `evaluate_bpb` calls
        #     model(x, y, loss_reduction='none')
        # with no `E_t_external`, so GPTWithRDA took the `E_t_external is None`
        # branch, seeded from `E_init`, and reported `mode='reset'`. Training,
        # meanwhile, always passes `E_t_external=rda_carry_state` and is always
        # `mode='carry'`. So the headline "Validation bpb" was measuring a
        # DIFFERENT state regime from the one being trained, and only for RDA
        # arms -- the baseline arm is unaffected because it has no state.
        #
        # WHY THIS IS A CORRECTNESS PROBLEM, NOT A TUNING CHOICE. This project
        # already retracted one measurement for exactly this defect: evaluating
        # FiLM "in an unseen state distribution" was ruled an invalid
        # counterfactual (see research/EXPERIMENT_LOG.md, "PE off via
        # targets=None"). Evaluating the carried-state model from E_init on every
        # batch is the same error: it reports the quality of a state regime the
        # optimizer never saw. We must NOT pick a mode because it makes the
        # number look better.
        #
        # THE POLICY, and why. "Match training" is the defensible default: the
        # quantity reported is then the BPB of the object actually being
        # trained. Concretely, a SEPARATE evaluation carry is used:
        #
        #   * It starts at E_init once per eval call, so every eval is
        #     deterministic and independent of training-step timing.
        #   * It is carried across the eval batches within one eval, matching
        #     the decay-only persistence the trainer uses.
        #   * It is NEVER the training carry. Sharing that object would make the
        #     reported metric depend on how many training steps preceded the
        #     eval, i.e. on the eval cadence -- a worse confound than the one
        #     being removed.
        #
        # The trade-off is stated rather than hidden: because decay=0.95 gives a
        # 1/e horizon of ~19.5 positions, the carried state is numerically
        # irrelevant past ~130 positions of a 1024-token row. So this change
        # moves the metric only slightly at T=1024 -- but it removes a silent
        # train/eval mismatch, and it will matter at any longer context. See F3
        # in research/CODE_REVIEW.md: the persistence itself is weak, which is a
        # separate problem from measuring it consistently.
        #
        # `rda_eval_state` is local to this eval call on purpose (see above).
        _rda_eval_state = None
        if rda_active:
            _rda_eval_state = rda_model.state.init_state(
                args.device_batch_size, device, dtype=torch.float32)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes,
                                   rda_carry=_rda_eval_state)
        # Report the mode the metric was actually produced under, so a reader
        # never has to infer it.
        _eval_mode = "carry(rda)" if rda_active else "n/a(no-rda)"
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f} | state_mode={_eval_mode}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
            "val/rda_state_mode": _eval_mode,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    save_recall_step_zero = (
        step == 0 and args.synthetic_recall_frac > 0.0 and not resuming
    )
    if (last_step or save_recall_step_zero
            or (step > 0 and step != args.resume_from_step
                and args.save_every > 0 and step % args.save_every == 0)):
        # When the regulatory drive state is active, the module and predictor live on the
        # GPTWithRDA wrapper, NOT on orig_model, so orig_model.state_dict()
        # alone would silently drop them and reloading would yield a randomly
        # re-initialised state. Save the wrapper's full state dict instead, which
        # covers the base trunk (keys prefixed "base.") plus state.* / predictor.*.
        model_state = (rda_model.state_dict() if rda_active
                       else orig_model.state_dict())
        os.makedirs(checkpoint_dir, exist_ok=True)
        staging_dir = os.path.join(checkpoint_dir, f".checkpoint_{step:06d}.tmp")
        os.makedirs(staging_dir, exist_ok=True)
        save_checkpoint(
            staging_dir,
            step,
            model_state, # model parameters (includes RDA module when enabled)
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "rda": rda_active, # so loaders know the key layout
                "rda_film_log": rda_film_log, # FiLM weight movement trace
                "rda_sat_log": rda_sat_log, # per-dimension E saturation trace
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        rng_staging_path = os.path.join(staging_dir, f"rng_{step:06d}_rank{ddp_rank:d}.pt")
        torch.save({
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, rng_staging_path)
        if is_ddp_initialized():
            dist.barrier()
        if ddp_rank == 0:
            # Publish data files first and metadata last; metadata is the commit marker.
            for rank in range(ddp_world_size):
                for prefix in ("optim", "rng"):
                    filename = f"{prefix}_{step:06d}_rank{rank:d}.pt"
                    os.replace(os.path.join(staging_dir, filename), os.path.join(checkpoint_dir, filename))
            os.replace(os.path.join(staging_dir, f"model_{step:06d}.pt"), os.path.join(checkpoint_dir, f"model_{step:06d}.pt"))
            os.replace(os.path.join(staging_dir, f"meta_{step:06d}.json"), os.path.join(checkpoint_dir, f"meta_{step:06d}.json"))
            shutil.rmtree(staging_dir, ignore_errors=True)
        if is_ddp_initialized():
            dist.barrier()

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    answer_loss_sums = None
    answer_loss_counts = None
    if args.synthetic_recall_frac > 0.0 and step % 50 == 0:
        answer_loss_sums = torch.zeros(2, dtype=torch.float64, device=x.device)
        answer_loss_counts = torch.zeros(2, dtype=torch.float64, device=x.device)
    for micro_step in range(grad_accum_steps):
        recall_answer_mask = None
        recall_class_masks = None
        if args.synthetic_recall_frac > 0.0:
            from nanochat.recall_data import build_batch, row
            batch_size = x.size(0)
            recall_rows = min(
                batch_size,
                max(1, int(round(batch_size * args.synthetic_recall_frac))),
            )
            batch_id = step * grad_accum_steps + micro_step
            synthetic_rows = []
            for row_idx in range(recall_rows):
                row_index = (
                    (batch_id * recall_rows + row_idx) * ddp_world_size
                    + ddp_rank
                )
                synthetic_rows.append(row(0, row_index))
            recall_x, recall_y, recall_answer_mask, recall_metadata = build_batch(
                synthetic_rows, device=x.device
            )
            recall_class_masks = torch.zeros(
                (recall_rows, recall_answer_mask.size(1), 2),
                dtype=torch.bool,
                device=x.device,
            )
            for row_idx, row_meta in enumerate(recall_metadata):
                for event in row_meta["events"]:
                    class_idx = 0 if event["class"] == "SALIENT" else 1
                    recall_class_masks[
                        row_idx, event["query_position"], class_idx
                    ] = True
            recall_class_masks = torch.cat([
                recall_class_masks,
                torch.zeros(
                    (batch_size - recall_rows, y.size(1), 2),
                    dtype=torch.bool,
                    device=y.device,
                ),
            ], dim=0)
            x = torch.cat([recall_x, x[recall_rows:]], dim=0)
            y = torch.cat([recall_y, y[recall_rows:]], dim=0)
            recall_answer_mask = torch.cat([
                recall_answer_mask,
                torch.zeros(
                    (batch_size - recall_rows, y.size(1)),
                    dtype=torch.bool,
                    device=y.device,
                ),
            ], dim=0)
        if rda_active:
            # DECAY-ONLY PERSISTENCE. The state is never hard-reset at a
            # document boundary: `rda_carry_state` stays None only until the
            # first step produces a state, and from then on every step carries
            # the previous E_next forward. The `decay` inside
            # RDAStateModule.update() is the only mechanism governing how
            # much prior state survives into a new document. Document starts
            # are counted and logged so a carry that crossed a boundary is
            # distinguishable from one that stayed within a document.
            doc_starts = document_start_mask(x, bos_token_id)
            if recall_answer_mask is not None and args.recall_answer_weight != 1.0:
                per_token_loss = model(
                    x, y, loss_reduction="none",
                    E_t_external=rda_carry_state, doc_starts=doc_starts,
                )
                loss_weights = torch.where(
                    recall_answer_mask.reshape(per_token_loss.shape),
                    args.recall_answer_weight,
                    1.0,
                )
                loss = (per_token_loss * loss_weights).sum() / (
                    loss_weights.sum() + torch.finfo(loss_weights.dtype).eps
                )
                if answer_loss_sums is not None:
                    detached_loss = per_token_loss.detach()
                    for class_idx in range(2):
                        class_mask = recall_class_masks[
                            :, :, class_idx
                        ].reshape(per_token_loss.shape)
                        answer_loss_sums[class_idx] += detached_loss[
                            class_mask
                        ].double().sum()
                        answer_loss_counts[class_idx] += class_mask.sum()
                rda_carry_state = model.last_state.detach()
                _rda_mode = model.last_mode
                rda_carry_counts[_rda_mode] = rda_carry_counts.get(_rda_mode, 0) + 1
                rda_doc_start_counts.append(int(doc_starts.sum().item()))
            else:
                out = model(x, y, E_t_external=rda_carry_state, doc_starts=doc_starts)
                loss = out["loss"]
                rda_carry_state = out["E_next"].detach()
                rda_carry_counts[out["mode"]] = rda_carry_counts.get(out["mode"], 0) + 1
                rda_doc_start_counts.append(out["n_doc_starts"])
        else:
            if recall_answer_mask is not None and args.recall_answer_weight != 1.0:
                per_token_loss = model(x, y, loss_reduction="none")
                loss_weights = torch.where(
                    recall_answer_mask.reshape(per_token_loss.shape),
                    args.recall_answer_weight,
                    1.0,
                )
                loss = (per_token_loss * loss_weights).sum() / (
                    loss_weights.sum() + torch.finfo(loss_weights.dtype).eps
                )
                if answer_loss_sums is not None:
                    detached_loss = per_token_loss.detach()
                    for class_idx in range(2):
                        class_mask = recall_class_masks[
                            :, :, class_idx
                        ].reshape(per_token_loss.shape)
                        answer_loss_sums[class_idx] += detached_loss[
                            class_mask
                        ].double().sum()
                        answer_loss_counts[class_idx] += class_mask.sum()
            else:
                loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            # MuonAdamW requires a grad on every enrolled param. The RDA
            # e_init and predictor.* are grad-free by design (nothing in the
            # loss depends on them), so give them an explicit zero grad rather
            # than letting the step crash. This must be re-done on EVERY
            # backward: model.zero_grad(set_to_none=True) runs after each
            # optimizer.step(), so the grads are None again by the next
            # micro-step. Priming only once would survive step 0 and then
            # crash on step 1.
            for _g in optimizer.param_groups:
                if _g.get("grad_free"):
                    for _p in _g["params"]:
                        if _p.grad is None:
                            _p.grad = torch.zeros_like(_p)
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    if answer_loss_sums is not None:
        if is_ddp_initialized():
            dist.all_reduce(answer_loss_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(answer_loss_counts, op=dist.ReduceOp.SUM)
        answer_means = answer_loss_sums / answer_loss_counts.clamp_min(1)
        print0(
            f"ANSWER_LOSS step={step} "
            f"SALIENT={answer_means[0].item():.6f} "
            f"ORDINARY={answer_means[1].item():.6f} "
            f"ln16={math.log(16):.6f}",
            flush=True,
        )
    # FiLM trace. This MUST sit outside the micro-step loop: `step` is constant
    # across all grad_accum_steps micro-steps, so logging inside recorded the
    # same pre-optimizer-step weights once per micro-step (8 duplicate entries
    # per checkpoint, all identical). Reading it here gives the weights after
    # opt.step(), which is what "is FiLM training?" actually means.
    if rda_active and step in rda_film_checkpoints:
        fs = float(rda_model.state.film_scale.weight.detach().abs().mean())
        fh = float(rda_model.state.film_shift.weight.detach().abs().mean())
        rda_film_log.append({"step": step,
                                "to_scale_w_absmean": fs,
                                "to_shift_w_absmean": fh})
        print0(f"[rda] FILM step={step} "
               f"to_scale.weight.abs().mean()={fs:.6e} "
               f"to_shift.weight.abs().mean()={fh:.6e}")
    # Per-dimension saturation of the drive state E, plus the abort rule.
    # Uses the FINAL E_next_seq, i.e. the state as it stood at the last
    # position of the last micro-step of this step. `.norm(dim=-1)` gives the
    # per-ROW norm; for the per-DIMENSION view wanted here, take the max |.|
    # over batch and positions, which is the quantity the |E| > 0.98 rule is
    # about -- a single dimension, not an aggregate.
    if rda_active:
        with torch.no_grad():
            E_seq = getattr(model, "_last_E_next_seq", None)
            if E_seq is not None:
                dim_max = E_seq.detach().abs().amax(dim=(0, 1)).float()  # (n,)
            else:
                dim_max = rda_carry_state.detach().abs().amax(dim=-1).float()
        if rda_force_sat:
            # TEST-ONLY: drive the named dimensions to 0.999 so the rule
            # below is exercised without waiting for a real saturation.
            idx = [i for i, n in enumerate(DRIVE_DIMS)
                   if n in rda_force_sat.split(",")]
            if idx:
                dim_max[idx] = 0.999
        n_sat = int((dim_max > 0.98).sum().item())
        if n_sat > 0:
            rda_sat_streak += 1
            # Record EVERY offending dimension, not just the argmax: naming
            # one when several are pinned hides most of the problem.
            _sat = (dim_max > 0.98).nonzero().flatten().tolist()
            rda_sat_dims = _sat
            rda_sat_worst_dim = int(dim_max.argmax().item())
        else:
            rda_sat_streak = 0
            rda_sat_dims = []
        if step in rda_sat_checkpoints:
            rda_sat_log.append({"step": step,
                                "dim_absmax": [round(float(v), 6) for v in dim_max],
                                "n_saturated": n_sat})
            print0("[rda] SAT step=%d n_dim(|E|>0.98)=%d  per-dim |E|max: %s"
                   % (step, n_sat,
                      " ".join("%.3f" % v for v in dim_max.tolist())))
        if rda_sat_streak >= rda_sat_abort_steps:
            print0("ABORT: %d drive dimension(s) exceeded |E| > 0.98 for %d "
                   "consecutive steps."
                   % (len(rda_sat_dims), rda_sat_streak))
            for _d in rda_sat_dims:
                print0("ABORT:   saturated dimension '%s' (index %d) "
                       "|E|max=%.4f"
                       % (DRIVE_DIMS[_d], _d, float(dim_max[_d])))
            print0("ABORT: The tanh is pinned; any further loss improvement is "
                   "not attributable to the drive.")
            print0("ABORT: run label %r, see manifests/%s.json"
                   % (args.model_tag, args.model_tag))
            sys.exit(3)
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm * _rda_diag_lr_scale
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)

    # ---- DIAGNOSTIC: per-drive decomposition (env-gated, inert unless
    # RDA_DIAG lists this step). Answers whether the climb is init or grad.
    if rda_active and step in _rda_diag_steps:
        with torch.no_grad():
            E_seq = getattr(model, "_last_E_next_seq", None)
            pe_proj = rda_model.state.pe_proj
            W = pe_proj.weight.detach().flatten().cpu()
            B = pe_proj.bias.detach().cpu()
            # lr actually applied to each state.* group, AFTER lrm
            lrs = []
            for _pg in _rda_groups:
                _nms = {n for n, p in rda_model.named_parameters()
                        if any(p is q for q in _pg["params"])
                        and n.startswith("state.")}
                if _nms:
                    lrs.append((min(_nms), _pg["lr"]))
            if E_seq is not None:
                h = out["hidden"][:, :-1, :].to(rda_model.state.event_proj.weight.dtype)
                ev = (rda_model.state.event_drive_scale
                      * rda_model.state.event_proj(rda_model.state.h_norm(h)))[0]
                lg = rda_model.base.lm_head(out["hidden"][:, :-1, :])[
                    ..., :rda_model.base.config.vocab_size].float()
                lg = 15 * torch.tanh(lg / 15)
                pe_s = rda_model.state.prediction_error_from_logits(lg, y[:, 1:])
                pv = rda_model.state.pe_proj(
                    (pe_s / rda_model.state.pe_reference_scale).unsqueeze(-1)
                    .to(pe_proj.weight.dtype))[0]
                Efin = E_seq[0].abs().amax(dim=0)
                # Combine on-device FIRST, then move to cpu once. Calling
                # .cpu() on pv while ev is still on cuda makes the add fail.
                tanh_max = torch.tanh(pv + ev).abs().amax(dim=0)
                pe_m = pv.abs().amax(dim=0).cpu()
                ev_m = ev.abs().amax(dim=0).cpu()
                tanh_m = tanh_max.cpu()
                pv_c, Efin = pv.cpu(), Efin.cpu()
                for d in range(len(DRIVE_DIMS)):
                    prev = _rda_diag_prev.get(d)
                    db = (B[d].item() - prev if prev is not None else 0.0)
                    print0("DIAG step=%d drive=%d name=%s pe_term=%+.4f "
                           "event=%+.4f tanh_sum=%+.4f maxE=%.4f "
                           "w=%+.4f b=%+.4f db=%+.5f"
                           % (step, d, DRIVE_DIMS[d], pe_m[d].item(),
                              ev_m[d].item(), tanh_m[d].item(),
                              Efin[d].item(), W[d].item(), B[d].item(), db))
                _rda_diag_prev = {d: B[d].item() for d in range(len(DRIVE_DIMS))}
            for _n, _lr in lrs:
                print0("DIAG step=%d lr_group[%s]=%.6g" % (step, _n, _lr))
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# ---- CONTROL ARM C: end-of-run const_state_vec report --------------------
# LOGGING ONLY. Reads two tensors that already exist, runs no forward pass,
# consumes no RNG, and writes nothing the training loop reads back. So it
# cannot change training behaviour: the values printed here are the values
# that were already in memory when the loop exited.
#
# Why it matters: const_state_vec is the ONLY thing that distinguishes arm C
# from arm B'. If it never moved from its init, arm C was not a control at all
# but a re-parameterisation of the zero-init, and the whole comparison would
# be void. So the movement is a first-class result, not a diagnostic.
if rda_active and rda_model.const_state_vec is not None:
    _cv_final = rda_model.const_state_vec.detach().float().cpu().tolist()
    _cv_init = rda_model.const_state_vec_init  # captured before any step
    _cv_delta = [f - i0 for f, i0 in zip(_cv_final, _cv_init)]
    print0("const_state_vec final : " + " ".join("%+.6f" % v for v in _cv_final))
    print0("const_state_vec init  : " + " ".join("%+.6f" % v for v in _cv_init))
    print0("const_state_vec delta : " + " ".join("%+.6e" % v for v in _cv_delta))
    _cv_absmax = max(abs(v) for v in _cv_delta)
    print0("const_state_vec max|delta| = %.6e  (moved from init: %s)"
           % (_cv_absmax, _cv_absmax > 0.0))
    # dims are DRIVE_DIMS order; printed by name so a reader cannot misattribute
    for _nm, _f, _d in zip(DRIVE_DIMS, _cv_final, _cv_delta):
        print0("    %-18s final %+.6f  delta %+.6e" % (_nm, _f, _d))
    _cv_path = os.path.join("logs", "ab_v3_const_state_s%d.json" % args.seed)
    try:
        with open(_cv_path, "w") as _f:
            json.dump({"seed": args.seed,
                       "dims": list(DRIVE_DIMS),
                       "init": _cv_init,
                       "final": _cv_final,
                       "delta": _cv_delta,
                       "max_abs_delta": _cv_absmax,
                       "final_val_bpb": None if val_bpb is None else float(val_bpb),
                       "num_iterations": int(num_iterations)},
                      _f, indent=2)
        print0("const_state_vec written to %s" % _cv_path)
    except Exception as _e:  # logging must never kill a finished run
        print0("const_state_vec JSON write FAILED: %r" % (_e,))

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
