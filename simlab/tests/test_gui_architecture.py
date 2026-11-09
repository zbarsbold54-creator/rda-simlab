"""Offscreen tests for the Architecture panel (no torch at import)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from simlab.gui.architecture import ArchitectureWidget
from simlab.training.arch_spec import (
    dump_arch_toml,
    get_spec,
    list_specs,
    load_arch_toml,
    scan_plugin_dir,
    spec_schema,
)


@pytest.fixture(autouse=True)
def _ensure_qapp() -> None:
    QApplication.instance() or QApplication([])
    yield
    QApplication.processEvents()


def _qapp():
    return QApplication.instance() or QApplication([])


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _wait_until(cond, timeout=10.0) -> None:
    app = _qapp()
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met within timeout")


def test_gui_architecture_import_does_not_load_torch() -> None:
    code = (
        "import os, sys; os.environ.setdefault('QT_QPA_PLATFORM','offscreen'); "
        "import simlab.gui.main_window; import simlab.gui.architecture; "
        "from simlab.gui.architecture import ArchitectureWidget; "
        "from PySide6.QtWidgets import QApplication; "
        "app=QApplication.instance() or QApplication([]); "
        "w=ArchitectureWidget(None); "
        "print('TORCH_PRESENT=' + str('torch' in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TORCH_PRESENT=False" in out.stdout, out.stderr
    assert "TORCH_PRESENT=True" not in out.stdout, out.stdout


def test_panel_builds_form_for_every_spec() -> None:
    widget = ArchitectureWidget(None)
    for name in list_specs("backbone"):
        widget.backbone.setCurrentText(name)
        widget._rebuild_forms()
        spec = get_spec("backbone", name)
        # A spec with config fields produces that many widgets; specs without
        # config produce none but must not crash.
        if spec.config_type is not None:
            assert len(widget._backbone_widgets) == len(spec_schema(spec))
    for name in list_specs("drive_core"):
        widget.drive.setCurrentText(name)
        widget._rebuild_forms()
        spec = get_spec("drive_core", name)
        if spec.config_type is not None:
            assert len(widget._drive_widgets) == len(spec_schema(spec))


def test_dropdown_change_rebuilds_form() -> None:
    widget = ArchitectureWidget(None)
    widget.backbone.setCurrentText("gru")
    widget._rebuild_forms()
    assert "hidden" in widget._backbone_widgets
    widget.backbone.setCurrentText("transformer")
    widget._rebuild_forms()
    assert "context" in widget._backbone_widgets
    assert "d_model" in widget._backbone_widgets


def test_save_load_round_trip() -> None:
    selection = {
        "backbone": "transformer",
        "drive_core": "rda8",
        "fusion": "film",
        "obs_mode": "obs_drives",
        "seed": 7,
        "backbone_config": {"context": 8, "d_model": 64, "n_heads": 2, "n_layers": 1},
        "drive_config": {"decay": 0.7},
    }
    text = dump_arch_toml(selection)
    loaded = load_arch_toml(_write_toml(text))
    assert loaded["backbone"] == "transformer"
    assert loaded["drive_core"] == "rda8"
    assert loaded["fusion"] == "film"
    assert loaded["obs_mode"] == "obs_drives"
    assert loaded["backbone_config"]["context"] == 8
    assert loaded["drive_config"]["decay"] == 0.7


def _write_toml(text: str) -> str:
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "exp.toml"
    tmp.write_text(text, encoding="utf-8")
    return str(tmp)


def test_good_user_plugin_appears_after_rescan(monkeypatch, tmp_path) -> None:
    good = tmp_path / "good_arch.py"
    good.write_text(
        "from dataclasses import dataclass, field\n"
        "from simlab.training.arch_spec import register_arch\n"
        "@dataclass\n"
        "class GoodConfig:\n"
        "    width: int = field(default=32, metadata={'min': 8, 'max': 256})\n"
        "@register_arch('backbone', 'goodbb', description='test', config=GoodConfig)\n"
        "class GoodBackbone:\n"
        "    def __init__(self, **kw):\n"
        "        self.width = kw.get('width', 32)\n"
    )
    errors = scan_plugin_dir(tmp_path)
    assert errors == []
    spec = get_spec("backbone", "goodbb")
    assert spec is not None
    assert spec_schema(spec)[0]["name"] == "width"
    # GUI Rescan picks it up when the folder is configured.
    monkeypatch.setenv("SIMLAB_ARCH_DIR", str(tmp_path))
    widget = ArchitectureWidget(None)
    assert "goodbb" in [widget.backbone.itemText(i) for i in range(widget.backbone.count())]


def test_broken_user_plugin_shows_error_no_crash(tmp_path) -> None:
    broken = tmp_path / "broken_arch.py"
    broken.write_text("raise RuntimeError('boom')\n")
    errors = scan_plugin_dir(tmp_path)
    assert errors
    assert "boom" in errors[0]


def test_build_through_button() -> None:
    widget = ArchitectureWidget(None)
    widget.backbone.setCurrentText("gru")
    widget._rebuild_forms()
    widget._build()
    _wait_until(lambda: "param_count=" in widget.param_label.text() or "failed" in widget.param_label.text())
    assert widget.param_label.text().startswith("param_count=")


def test_build_rda_variant_succeeds() -> None:
    widget = ArchitectureWidget(None)
    widget.backbone.setCurrentText("rda_g1")
    widget._rebuild_forms()
    widget._build()
    _wait_until(lambda: "param_count=" in widget.param_label.text() or "failed" in widget.param_label.text())
    assert widget.param_label.text().startswith("param_count=")


def test_backbone_selection_matches_canonical_preset() -> None:
    from simlab.training.arch_spec import default_preset_selection

    widget = ArchitectureWidget(None)
    widget.backbone.setCurrentText("rda_gru_surprise")
    assert widget._collect() == default_preset_selection("rda_gru_surprise")


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_checkpoint_playback_rebuilds_arch_runs_world(tmp_path) -> None:
    from simlab.core.config import load_config
    from simlab.core.runner import Runner
    from simlab.training.arch_spec import build_policy_cfg
    from simlab.training.torch_policy import TorchPolicy

    selection = {
        "backbone": "gru",
        "drive_core": "rda8",
        "fusion": "concat",
        "obs_mode": "obs_only",
        "seed": 0,
        "backbone_config": {"hidden": 32},
        "drive_config": {"decay": 0.5},
    }
    ckpt = tmp_path / "p.pt"
    TorchPolicy(seed=0, cfg=build_policy_cfg(selection)).save_checkpoint(ckpt)

    widget = ArchitectureWidget(load_config())
    widget.drive.setCurrentText("rda8")
    widget._checkpoint_path = str(ckpt)
    captured: dict[str, Any] = {}
    widget.playbackRequested.connect(lambda cfg: captured.update(cfg=cfg))
    widget._apply()
    assert "cfg" in captured
    cfg = captured["cfg"]
    assert cfg.policy == "learned_torch"
    assert str(cfg.policy_checkpoint) == str(ckpt)
    assert cfg.policy_cfg is None
    assert cfg.drive == "rda8"

    runner = Runner(cfg, seed=0)
    runner.setup()
    result = runner.run_step(
        include_observations=False,
        include_snapshot=False,
        include_agent_states=False,
    )
    assert result is not None
    assert runner._step_count == 1
    assert runner._policy is not None
    assert runner._policy._backbone.config_dict()["backbone"] == "gru"
