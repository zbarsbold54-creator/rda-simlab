"""Write a run manifest BEFORE a training run starts.

WHY THIS EXISTS. The previous round of A/B runs could not be compared,
because the logs had TRUNCATED command lines (`--model-...`) and recorded
neither the dataset nor the shard list. Two runs that look identical in the
log may have used different data, and there is no way to tell afterwards.
This writes the facts that make a run reproducible AFTER THE FACT.

Written at the START of each run, so a run that crashes still leaves a
manifest describing what it was trying to do.

Contents:
  - the FULL untruncated command line (sys.argv, not a reconstructed string)
  - dataset name and the exact shard files on disk, with sizes
  - seed, git commit, GPU

Run (from the repo root, before base_train.py):
  python diagnostics/write_manifest.py --label a-s42 --seed 42 [--rda]
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MANIFEST_DIR = os.path.join(REPO, "manifests")


def git(*a):
    try:
        return subprocess.check_output(["git"] + list(a), cwd=REPO,
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception as e:
        return "ERROR: %s" % e


def data_state():
    """Exact shard files present on disk, with size and a content hash of
    the first and last 1 MB (enough to tell two datasets apart cheaply)."""
    os.environ["NANOCHAT_DATASET"] = os.environ.get("NANOCHAT_DATASET",
                                                     "fineweb-edu")
    import importlib
    import nanochat.dataset as dsmod
    importlib.reload(dsmod)
    out = {"dataset": dsmod.DATASET_NAME, "data_dir": dsmod.DATA_DIR,
           "max_shard": dsmod.MAX_SHARD, "shards": []}
    if not os.path.isdir(dsmod.DATA_DIR):
        out["error"] = "data dir missing"
        return out
    names = sorted(f for f in os.listdir(dsmod.DATA_DIR)
                   if f.endswith(".parquet"))
    for n in names:
        p = os.path.join(dsmod.DATA_DIR, n)
        sz = os.path.getsize(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))                  # first MB
            if sz > (2 << 20):
                f.seek(-(1 << 20), os.SEEK_END)        # last MB
                h.update(f.read(1 << 20))
        out["shards"].append({"file": n, "bytes": sz,
                              "edge_sha256": h.hexdigest()[:16]})
    return out


def diff_hash():
    """SHA-256 of the working-tree diff, so a run started on a DIRTY tree can
    be tied to the exact content that was on disk.

    Two runs whose commits match but whose diff hashes differ ran different
    code, and nothing else in the log would show that. Returns (clean, hash).
    """
    status = git("status", "--porcelain")
    # Hash both tracked changes and untracked file CONTENTS, since an
    # untracked diagnostics script is very much part of what ran.
    tracked = git("diff", "HEAD")
    untracked = ""
    for line in status.splitlines():
        if line.startswith("??"):
            p = line[3:].strip()
            fp = os.path.join(REPO, p)
            if os.path.isdir(fp):
                for root, _, files in os.walk(fp):
                    for f in sorted(files):
                        untracked += p + "/" + f + "\n"
            elif os.path.isfile(fp):
                untracked += p + "\n"
    h = hashlib.sha256()
    h.update(tracked.encode())
    h.update(b"\n--UNTRACKED--\n")
    h.update(untracked.encode())
    return (len(status.strip()) == 0), h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run label, e.g. a-s42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rda", action="store_true")
    ap.add_argument("--num-iterations", type=int, default=600)
    ap.add_argument("--device-batch-size", type=int, default=4)
    ap.add_argument("--total-batch-size", type=int, default=32768)
    ap.add_argument("--note", type=str, default="")
    a = ap.parse_args()

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    clean, dh = diff_hash()

    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass

    m = {
        "label": a.label,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # sys.argv, so this is exactly what was executed -- never a
        # reconstructed or truncated version of it.
        "argv": sys.argv,
        "argv_full_command_line": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": a.seed,
        "rda": bool(a.rda),
        "train_config": {
            "num_iterations": a.num_iterations,
            "device_batch_size": a.device_batch_size,
            "total_batch_size": a.total_batch_size,
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # Tree state at the moment the run STARTED. A dirty tree means the
        # commit alone does not identify the code, so the diff hash does.
        "git_clean": clean,
        "git_diff_sha256": dh,
        "git_status_porcelain": git("status", "--porcelain"),
        "git_diff_stat": git("diff", "--stat"),
        "gpu": gpu,
        "note": a.note,
        "data": data_state(),
    }

    path = os.path.join(MANIFEST_DIR, "%s.json" % a.label)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    d = m["data"]
    print("manifest: %s" % path)
    print("  commit : %s (%s)" % (m["git_commit"][:12], m["git_branch"]))
    print("  tree   : %s  diff_sha256=%s"
          % ("CLEAN" if clean else "DIRTY", dh[:16]))
    print("  seed   : %d   rda: %s" % (a.seed, a.rda))
    print("  dataset: %s  dir=%s" % (d["dataset"], d["data_dir"]))
    print("  shards : %d" % len(d.get("shards", [])))
    for s in d.get("shards", []):
        print("           %s  %d bytes  %s"
              % (s["file"], s["bytes"], s["edge_sha256"]))
    if d.get("error"):
        print("  WARNING: %s" % d["error"])


if __name__ == "__main__":
    main()
