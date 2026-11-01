"""Offscreen tests for the Research cockpit logic.

Covers experiment-builder protocol guards, queue resume, follow-latest
checkpoint, results aggregation/differences, gate diagnostics helpers, the
torch-free GUI import guarantee, and (guarded) parameter counts.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simlab.research.experiment import (
    DEFAULT_WORKERS,
    TRAIN_SEED_MIN,
    Arm,
    Experiment,
    arm_param_count,
    arm_param_counts,
    dump_experiment_toml,
    load_experiment_toml,
)
from simlab.research.follower import FollowState, poll_latest
from simlab.research.gates import correlate_events, gate_stats, rail_warning
from simlab.research.logs import latest_checkpoint, parse_learning_csv, parse_quick_eval
from simlab.research.queue import QueueLockError, QueueManager
from simlab.research.results import (
    SeedRow,
    default_control,
    difference_rows,
    load_eval_file,
    rows_from_seed_results,
    summarize_rows,
    to_csv,
)

REPO = Path(__file__).resolve().parents[1]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _make_experiment(**overrides: Any) -> Experiment:
    defaults: dict[str, Any] = {
        "name": "test_grid",
        "arms": [Arm(name="rda_g1", backbone="rda_g1")],
        "workers": DEFAULT_WORKERS,
        "train_seeds": [TRAIN_SEED_MIN],
        "dev_seeds": [0, 1, 2, 3, 4],
    }
    defaults.update(overrides)
    return Experiment(**defaults)


# ------------------------------------------------------ protocol guards
def test_experiment_valid_no_errors() -> None:
    exp = _make_experiment()
    errors, warnings = exp.validate()
    assert errors == []
    assert warnings == []


def test_train_seed_below_floor_is_error() -> None:
    exp = _make_experiment(train_seeds=[99999])
    errors, _ = exp.validate()
    assert any("100000" in e for e in errors)


def test_workers_not_pinned_warns() -> None:
    exp = _make_experiment(workers=8)
    errors, warnings = exp.validate()
    assert errors == []
    assert any("workers" in w and "4" in w for w in warnings)


def test_dev_seed_out_of_range_is_error() -> None:
    exp = _make_experiment(dev_seeds=[42])
    errors, _ = exp.validate()
    assert any("dev seed" in e for e in errors)


def test_test_seeds_require_confirm() -> None:
    exp = _make_experiment(test_seeds=[1000, 1001, 1002])
    assert exp.requires_test_confirm() is True
    errors, _ = exp.validate()
    assert errors == []


def test_fewer_than_3_seeds_warns() -> None:
    exp = _make_experiment(dev_seeds=[0, 1])
    errors, warnings = exp.validate()
    assert errors == []
    assert any("3" in w for w in warnings)


def test_unknown_backbone_is_error() -> None:
    exp = _make_experiment(arms=[Arm(name="x", backbone="does_not_exist")])
    errors, _ = exp.validate()
    assert any("unknown backbone" in e for e in errors)


def test_toml_round_trip() -> None:
    exp = _make_experiment(
        test_seeds=[1000, 1001, 1002], control_arm="rda_g1", eval_every=2
    )
    text = dump_experiment_toml(exp)
    tmp = REPO / "runs" / "test_roundtrip.toml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    loaded = load_experiment_toml(tmp)
    tmp.unlink(missing_ok=True)
    assert loaded.name == exp.name
    assert loaded.arms[0].backbone == "rda_g1"
    assert loaded.test_seeds == [1000, 1001, 1002]
    assert loaded.control_arm == "rda_g1"
    assert loaded.eval_every == 2


def test_eval_every_cannot_be_negative() -> None:
    errors, _ = _make_experiment(eval_every=-1).validate()
    assert "eval_every must be >= 0" in errors


# ------------------------------------------------------- queue resume
def test_queue_resume_reconciles_crashed_job(tmp_path: Path) -> None:
    exp = _make_experiment(train_seeds=[TRAIN_SEED_MIN, TRAIN_SEED_MIN + 1])
    qdir = tmp_path / "q"
    QueueManager.create(qdir, exp)
    # Simulate a crash: mark the first job running with a dead pid.
    jobs = QueueManager(qdir).load_jobs()
    jobs[0].status = "running"
    jobs[0].pid = 999_999_999  # certainly not alive
    QueueManager(qdir).save_jobs(jobs)

    ran: list[str] = []

    def fake(runner_job, runner_exp) -> int:
        ran.append(runner_job.id)
        return 0

    QueueManager(qdir).run_forever(runner=fake)
    assert len(ran) == 2
    final = QueueManager(qdir).load_jobs()
    assert all(j.status == "done" for j in final)


def test_queue_lock_blocks_second_server(tmp_path: Path) -> None:
    exp = _make_experiment()
    qdir = tmp_path / "qlock"
    QueueManager.create(qdir, exp)
    mgr = QueueManager(qdir)
    mgr.acquire_lock()
    try:
        with pytest.raises(QueueLockError):
            QueueManager(qdir).acquire_lock()
    finally:
        mgr.release_lock()


def test_pid_alive_does_not_send_ctrl_c() -> None:
    """_pid_alive must probe liveness, not emit a Ctrl+C (SIGINT) event.

    On Windows ``os.kill(pid, 0)`` maps to CTRL_C_EVENT and would interrupt the
    calling process at its next blocking call (regression: an ordering-dependent
    KeyboardInterrupt when the lock test ran before the kill-tree test).
    """
    import signal
    import time

    from simlab.research.queue import _pid_alive

    fired: list[int] = []

    def handler(sig, frame):
        fired.append(sig)

    previous = signal.signal(signal.SIGINT, handler)
    try:
        assert _pid_alive(os.getpid()) is True
        # Leave time for an (incorrect) pending ctrl+C event to be delivered.
        time.sleep(0.2)
        assert fired == [], f"SIGINT was delivered: {fired}"
    finally:
        signal.signal(signal.SIGINT, previous)


def test_queue_status_counts(tmp_path: Path) -> None:
    exp = _make_experiment(train_seeds=[TRAIN_SEED_MIN, TRAIN_SEED_MIN + 1])
    qdir = tmp_path / "qstatus"
    QueueManager.create(qdir, exp)
    status = QueueManager(qdir).status()
    assert status["counts"]["queued"] == 2
    assert len(status["jobs"]) == 2


# ---------------------------------------------- stop kills child tree
def test_kill_tree_terminates_child_not_parent(tmp_path: Path) -> None:
    """Stopping a run kills the child tree without interrupting the parent."""
    import subprocess as sp
    import sys
    import time

    from simlab.research.queue import _kill_tree

    # Spawn the child in its own process group (as the queue server does) so
    # killing it never delivers a console control event to the parent. Without
    # this, taskkill /T would interrupt the shared console and raise
    # KeyboardInterrupt in the pytest (parent) process.
    creationflags = getattr(sp, "CREATE_NEW_PROCESS_GROUP", 0)
    code = "import time; time.sleep(30)"
    child = sp.Popen(
        [sys.executable, "-c", code], creationflags=creationflags
    )
    time.sleep(0.5)
    assert child.poll() is None  # child is running
    _kill_tree(child.pid)
    try:
        child.wait(timeout=5)
    except sp.TimeoutExpired:
        child.kill()
    assert child.returncode is not None  # child was terminated
    # The test process (the "parent") is still alive and executing.
    assert _kill_tree is not None


# ------------------------------------------- follow latest checkpoint
def _write_zip(path: Path, payload: bytes = b"ckpt") -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data", payload)


def test_follow_latest_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "latest.pt"
    _write_zip(ckpt)
    state = FollowState()
    first = poll_latest(run_dir, state)
    assert first == ckpt
    second = poll_latest(run_dir, state)
    assert second is None
    # touching the file makes it "new" again
    _write_zip(ckpt, b"two")
    third = poll_latest(run_dir, state)
    assert third == ckpt


def test_follow_ignores_truncated_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "latest.pt"
    # Truncated / in-flight write: non-zip bytes.
    ckpt.write_bytes(b"PK\x03\x04 truncated-not-a-valid-zip")
    state = FollowState()
    assert poll_latest(run_dir, state) is None
    assert state.last_sig is None  # signature not advanced, will retry
    # Once the write completes (valid zip) the follower picks it up.
    _write_zip(ckpt)
    assert poll_latest(run_dir, state) == ckpt
    # An empty file is also not readable.
    (run_dir / "final.pt").write_bytes(b"")
    assert poll_latest(run_dir, state) is None


def test_latest_checkpoint_name_preference(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latest.pt").write_bytes(b"a")
    (run_dir / "final.pt").write_bytes(b"b")
    ckpt = latest_checkpoint(run_dir)
    assert ckpt.name == "latest.pt"


# ------------------------------------------------------- logs parsing
def test_parse_learning_csv_and_quick_eval(tmp_path: Path) -> None:
    learn = tmp_path / "learning.csv"
    learn.write_text(
        "iteration,env_steps,mean_episode_len,mean_return,entropy,approx_kl,value_loss\n"
        "0,1024,105.0,12.5,1.1,0.01,0.2\n"
        "# quick_eval it=0 cap_pct=62.5\n"
        "1,2048,107.0,13.1,1.05,0.012,0.19\n"
        "# quick_eval it=1 cap_pct=70.0\n",
        encoding="utf-8",
    )
    rows = parse_learning_csv(learn)
    assert len(rows) == 2
    assert rows[1]["mean_return"] == 13.1
    evals = parse_quick_eval(learn)
    assert evals == [(0, 62.5), (1, 70.0)]


# ------------------------------------------------------- results
def test_rows_from_seed_results_aggregation() -> None:
    results = [
        {
            "seed": 0,
            "per_agent": [
                {
                    "class": "truncation",
                    "death_step": 200,
                    "food_eaten": 3,
                    "hazard_contacts": 1,
                },
                {
                    "class": "starvation",
                    "death_step": 90,
                    "food_eaten": 1,
                    "hazard_contacts": 0,
                },
            ],
        }
    ]
    rows = rows_from_seed_results(results)
    assert len(rows) == 1
    assert rows[0].seed == 0
    assert rows[0].trunc == 1
    assert rows[0].starve == 1
    assert rows[0].cap_pct == 50.0
    assert rows[0].median_survival == 145.0


def test_summarize_rows_mean_spread() -> None:
    rows = [
        SeedRow(
            seed=0, cap_pct=50.0, median_survival=100.0, starve=1, hazard=0, trunc=1
        ),
        SeedRow(
            seed=1, cap_pct=70.0, median_survival=120.0, starve=0, hazard=0, trunc=2
        ),
        SeedRow(
            seed=2, cap_pct=90.0, median_survival=160.0, starve=0, hazard=1, trunc=3
        ),
    ]
    summary = summarize_rows(rows)
    mean, spread = summary["cap_pct"]
    assert mean == 70.0
    assert spread == pytest.approx(16.33, abs=0.02)


def test_difference_rows_vs_control() -> None:
    arm = [
        SeedRow(
            seed=0, cap_pct=70.0, median_survival=120.0, starve=0, hazard=0, trunc=5
        )
    ]
    control = [
        SeedRow(
            seed=0, cap_pct=50.0, median_survival=100.0, starve=1, hazard=0, trunc=2
        )
    ]
    diffs = difference_rows(arm, control)
    assert diffs[0]["cap_pct"] == 20.0
    assert diffs[0]["median_survival"] == 20.0


def test_missing_control_marked() -> None:
    arm = [
        SeedRow(
            seed=0, cap_pct=70.0, median_survival=120.0, starve=0, hazard=0, trunc=5
        )
    ]
    diff = difference_rows(arm, [])[0]
    assert diff.get("missing_control") is True


def test_default_control_for_rda() -> None:
    assert default_control("rda_g1") == "rda_const"
    assert default_control("rda_const") is None
    assert default_control("gru") is None


def test_to_csv_includes_diff_block() -> None:
    arm_rows = {
        "rda_g1": [
            SeedRow(
                seed=0, cap_pct=70.0, median_survival=120.0, starve=0, hazard=0, trunc=5
            )
        ],
        "rda_const": [
            SeedRow(
                seed=0, cap_pct=50.0, median_survival=100.0, starve=1, hazard=0, trunc=2
            )
        ],
    }
    text = to_csv(arm_rows, control="rda_const")
    assert "rda_g1" in text
    assert "diff_vs_rda_const" in text


def test_load_eval_file(tmp_path: Path) -> None:
    p = tmp_path / "eval.json"
    p.write_text(
        json.dumps(
            {
                "seed": 3,
                "per_agent": [{"class": "truncation", "death_step": 120}],
            }
        ),
        encoding="utf-8",
    )
    rows = load_eval_file(p)
    assert rows[0].seed == 3
    assert rows[0].trunc == 1


# ------------------------------------------------------- gate helpers
def test_gate_stats_fraction_at_max() -> None:
    lam = np.array([0.9, 0.9984, 0.9985, 0.5])
    stats = gate_stats(lam, lam_max=0.9984375)
    assert 0.25 <= stats["fraction_at_max"] <= 0.5
    assert stats["hist"].sum() == 4


def test_rail_warning_detects_long_saturation() -> None:
    mag = np.full(250, 0.99)
    runs = rail_warning(mag, rail=1.0, tol=0.02, min_steps=100)
    assert len(runs) == 1
    assert runs[0]["length"] == 250


def test_rail_warning_no_warning_short() -> None:
    mag = np.full(50, 0.99)
    runs = rail_warning(mag, rail=1.0, tol=0.02, min_steps=100)
    assert runs == []


def test_correlate_events() -> None:
    lam = np.array([0.9, 0.99, 0.99, 0.91])
    events = np.array([0.0, 1.0, 1.0, 0.0])
    r = correlate_events(lam, events)
    assert r > 0.9


# ------------------------------------------------------- param counts
@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_arm_param_count_rda_g1() -> None:
    arm = Arm(name="rda_g1", backbone="rda_g1")
    params, h, _ = arm_param_count(arm)
    assert params > 0
    assert h


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_arm_param_counts_table_within_5pct() -> None:
    arms = [
        Arm(name="rda_g1", backbone="rda_g1"),
        Arm(name="rda_const", backbone="rda_const"),
    ]
    rows = arm_param_counts(arms)
    assert len(rows) == 2
    a, b = rows[0]["params"], rows[1]["params"]
    assert abs(a - b) / max(a, b) <= 0.05


# ------------------------------------------------------- torch-free GUI import
def test_cockpit_import_does_not_load_torch() -> None:
    code = (
        "import sys; "
        "import simlab.gui.cockpit as c; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
        env=env,
        creationflags=creationflags,
    )
    assert proc.returncode == 0, proc.stderr

def test_live_follow_uses_checkpoint_arch(tmp_path: Path) -> None:
    """Live follow must prefer the architecture stored next to the checkpoint."""
    from simlab.research.replay import _infer_arm_from_checkpoint

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": {
                    "drive_core": "rda_g1",
                    "drive_config": {"base": 0.9},
                    "obs_mode": "obs_drives",
                }
            }
        ),
        encoding="utf-8",
    )
    ckpt = tmp_path / "final.pt"
    ckpt.write_bytes(b"dummy")
    inf = _infer_arm_from_checkpoint(ckpt)
    assert inf["drive_core"] == "rda_g1"
    assert inf["drive_config"] == {"base": 0.9}
    assert inf["obs_mode"] == "obs_drives"


def test_infer_arm_fallback_when_config_missing(tmp_path: Path) -> None:
    from simlab.research.replay import _infer_arm_from_checkpoint

    ckpt = tmp_path / "final.pt"
    ckpt.write_bytes(b"dummy")
    assert _infer_arm_from_checkpoint(ckpt) == {}


# ------------------------------------------------------- gui-v1: stalls
_STALL_PRESETS = [
    "gru_legacy",
    "gru",
    "transformer",
    "rda_gru_surprise",
    "rda_tf_surprise",
    "rda_gru_per_drive",
]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
@pytest.mark.parametrize("preset", _STALL_PRESETS)
def test_gui_thread_stall_per_preset(preset: str) -> None:
    """GUI thread must stay responsive (max stall < 200ms) during 300 ticks."""
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dataclasses import replace

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from simlab.core.config import load_config
    from simlab.gui.main_window import MainWindow
    from simlab.training.arch_spec import build_policy_cfg

    app = QApplication.instance() or QApplication([])
    window = MainWindow(seed=1, agent_count=3)
    window.worker.set_playing(False)
    window.show()
    selection = {
        "backbone": preset,
        "drive_core": "rda8",
        "fusion": "film",
        "obs_mode": "obs_drives",
        "seed": 0,
        "backbone_config": {},
        "drive_config": {},
    }
    cfg = replace(
        load_config(),
        policy="learned_torch",
        drive="rda8",
        drive_config=None,
        policy_checkpoint=None,
        policy_cfg=build_policy_cfg(selection),
    )
    heartbeat: list[float] = []

    def beat() -> None:
        heartbeat.append(time.monotonic())

    timer = QTimer()
    timer.setInterval(16)
    timer.timeout.connect(beat)
    timer.start()
    window._apply_arch(cfg)
    window.worker.request_steps(300)
    deadline = time.monotonic() + 60.0
    while (
        window.worker.latest_snapshot.step < 300
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.005)
    timer.stop()
    max_stall = max(
        (b - a for a, b in itertools.pairwise(heartbeat)), default=0.0
    )
    window.close()
    app.processEvents()
    assert window.worker.latest_snapshot.step >= 300
    assert max_stall < 0.200, (
        f"{preset}: max GUI stall {max_stall * 1000:.1f}ms >= 200ms"
    )


# ------------------------------------------------------- gui-v1: quick demo
def test_quick_demo_wires_tabs() -> None:
    """Quick demo launches a valid run and wires Live/Curves/Gates to it."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication

    from simlab.gui.cockpit import CockpitWidget

    app = QApplication.instance() or QApplication([])
    widget = CockpitWidget(None)
    spawned: list[list[str]] = []
    original_start = QProcess.start

    def stub_start(process, program, arguments=None):
        spawned.append([program, *(arguments or [])])

    QProcess.start = stub_start
    try:
        widget._quick_demo()
    finally:
        QProcess.start = original_start
    assert spawned, "expected QProcess queue server launch"
    assert any("serve" in a for a in spawned)
    assert widget.follow_cb.isChecked()
    assert widget.live_run.currentText().endswith("00_rda_gru_surprise_100001")
    assert "quick_demo" in widget.curves_run.currentText()
    app.processEvents()
    widget.deleteLater()
    app.processEvents()


def test_queue_launch_wires_live_and_curves(tmp_path: Path, monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtWidgets import QApplication

    from simlab.gui import cockpit as cockpit_module
    from simlab.gui.cockpit import CockpitWidget

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(cockpit_module, "COCKPIT_DIR", tmp_path / "cockpit")
    starts: list[tuple[str, list[str]]] = []
    original_start = QProcess.start

    def stub_start(process, program, arguments=None):
        starts.append((program, list(arguments or [])))

    QProcess.start = stub_start
    widget = CockpitWidget(None)
    try:
        for index in range(widget.arm_list.count()):
            item = widget.arm_list.item(index)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.text() == "rda_gru_surprise"
                else Qt.CheckState.Unchecked
            )
        widget.exp_name.setText("queue_follow_test")
        widget.total_steps.setValue(4096)
        widget.train_seeds.setText("100000")
        widget.dev_seeds.setText("")
        widget.test_seeds.setText("")
        widget.scen_hard.setChecked(False)
        widget._launch_grid()
        expected = tmp_path / "cockpit" / "queue_follow_test" / "runs" / "00_rda_gru_surprise_100000"
        assert starts and "serve" in starts[0][1]
        assert widget.live_run.currentText() == str(expected)
        assert widget.curves_run.currentText() == str(expected)
        assert widget.follow_cb.isChecked()
        assert widget._curves_timer.interval() == 5000
    finally:
        widget.follow_cb.setChecked(False)
        widget.deleteLater()
        app.processEvents()
        QProcess.start = original_start
