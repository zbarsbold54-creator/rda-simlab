"""Test the RDA saturation abort rule in base_train.py. CPU, tiny config.

The abort rule is a safety mechanism: it fires when a drive dimension sits at
|E| > 0.98 for 10 consecutive steps. An abort rule that has never been fired
is indistinguishable from an abort rule that is broken -- both produce a
finished log with no "ABORT" in it.

So this forces the condition with the RDA_FORCE_SAT environment hook and
checks three things:
  1. the run ABORTS,
  2. it names the offending drive dimensions,
  3. it exits NON-ZERO (a silent abort would look like a clean finish).

Then it checks the negative: with the hook unset, the identical config runs to
completion and does NOT abort. Without that, a rule that aborts unconditionally
would pass test 1.

The hook can only RAISE |E|, so leaving it on would abort every run and
leaving it off cannot hide a real saturation.

Run:
  python diagnostics/test_saturation_abort.py
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")

# Tiny: 2 layers, 128-dim, 256 ctx, 8 steps. The abort needs 10 consecutive
# saturated steps, so this asks for 20 to leave room for the streak to build.
COMMON = [
    "-m", "scripts.base_train",
    "--device-type", "cpu",
    "--depth", "2", "--max-seq-len", "128", "--aspect-ratio", "8",
    "--device-batch-size", "1", "--total-batch-size", "256",
    "--num-iterations", "20", "--eval-every", "-1",
    "--core-metric-every", "-1", "--sample-every", "-1",
    "--rda", "--rda-debug-steps", "0",
    "--model-tag", "satabort-test",
]


def run(env_extra, timeout=1800):
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": REPO,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "NANOCHAT_DATASET": "fineweb-edu",
    })
    env.update(env_extra)
    try:
        p = subprocess.run([PY] + COMMON, cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


FAIL = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    print("=== 1. abort rule FIRES when forced ===")
    rc, out = run({"RDA_FORCE_SAT": "threat,arousal"})
    tail = out[-2500:]
    print("  exit code: %r" % rc)
    for ln in out.splitlines():
        if "ABORT" in ln or "TREE:" in ln:
            print("  | %s" % ln.strip())
    check("exits NON-ZERO", rc is not None and rc != 0, "rc=%r" % rc)
    check("prints ABORT", "ABORT" in out)
    check("names offending dimensions",
          ("threat" in out and "arousal" in out),
          "threat=%s arousal=%s" % ("threat" in out, "arousal" in out))
    check("did NOT finish 20 iterations", "Minimum validation bpb" not in out)

    print("\n=== 2. hook is INACTIVE by default ===")
    rc2, out2 = run({})
    check("no ABORT without the hook", "ABORT" not in out2)
    check("runs to completion", rc2 == 0, "rc=%r" % rc2)
    _tree = [l for l in out2.splitlines()[:40] if "TREE:" in l]
    check("reports TREE state at top of log", bool(_tree),
          _tree[0].strip() if _tree else "not found")

    print("\n" + "=" * 60)
    if FAIL:
        print("SATURATION ABORT TEST FAILED (%d): %s" % (len(FAIL), ", ".join(FAIL)))
        sys.exit(1)
    print("SATURATION ABORT TEST PASS")


if __name__ == "__main__":
    main()
