"""Architecture specs: torch-free metadata for backbones and drive cores.

An ``ArchSpec`` describes one registered architecture component (a backbone
or a drive core): its human-readable name/description, a config dataclass
(typed fields with defaults, min/max, choices and help text), and a lazy
loader that returns the actual class on demand. Specs are importable without
torch; the classes they point at are only imported when the user builds or
loads a policy.

The built-in backbone specs live here (importing ``simlab.training.backbones``
would import torch). Drive-core specs are registered beside their classes in
``simlab.plugins.drive_cores``, which is torch-free. User plugins register
extra specs with the ``register_arch`` decorator from this module.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchSpec:
    """Metadata for one registered architecture component."""

    name: str
    kind: str  # "backbone" | "drive_core"
    description: str
    config_type: type | None
    loader: Callable[[], type] | None
    source: str = ""


ARCH_SPECS: dict[tuple[str, str], ArchSpec] = {}


def add_spec(
    kind: str,
    name: str,
    *,
    description: str = "",
    config: type | None = None,
    loader: Callable[[], type] | None = None,
    source: str = "",
) -> ArchSpec:
    """Register an ArchSpec directly (used for built-ins)."""
    spec = ArchSpec(
        name=name,
        kind=kind,
        description=description,
        config_type=config,
        loader=loader,
        source=source,
    )
    ARCH_SPECS[(kind, name)] = spec
    return spec


def register_arch(
    kind: str,
    name: str,
    *,
    description: str = "",
    config: type | None = None,
):
    """Decorator to register a spec for the decorated class (user plugins).

    The decorated class is the component; its spec is importable without
    torch because the class is only referenced lazily through the returned
    spec. ``config`` may be a dataclass defining the architecture's settings.
    """

    def decorator(cls: type) -> type:
        add_spec(
            kind,
            name,
            description=description,
            config=config,
            loader=lambda cls=cls: cls,
            source=getattr(cls, "__module__", ""),
        )
        return cls

    return decorator


def get_spec(kind: str, name: str) -> ArchSpec | None:
    return ARCH_SPECS.get((kind, name))


def list_specs(kind: str) -> list[str]:
    return sorted(name for (k, name) in ARCH_SPECS if k == kind)


def load_class(kind: str, name: str) -> type:
    """Lazily load the concrete class for a spec (may import torch)."""
    spec = get_spec(kind, name)
    if spec is None or spec.loader is None:
        raise ValueError(f"no loader for {kind} '{name}'")
    return spec.loader()


# --------------------------------------------------------------------------- #
# Field schema and validation (from the spec alone, no torch).
# --------------------------------------------------------------------------- #
def _field_schema(f) -> dict[str, Any]:
    import dataclasses

    meta = f.metadata or {}
    schema = {
        "name": f.name,
        "default": f.default if f.default is not dataclasses.MISSING else None,
        "help": meta.get("help", ""),
    }
    if f.type in (int, "int") or f.type is int:
        schema["type"] = "int"
        schema["min"] = meta.get("min")
        schema["max"] = meta.get("max")
        schema["choices"] = meta.get("choices")
    elif f.type in (float, "float") or f.type is float:
        schema["type"] = "float"
        schema["min"] = meta.get("min")
        schema["max"] = meta.get("max")
        schema["choices"] = None
    elif f.type in (bool, "bool") or f.type is bool:
        schema["type"] = "bool"
    else:
        schema["type"] = "str"
        schema["choices"] = meta.get("choices")
    return schema


def spec_schema(spec: ArchSpec) -> list[dict[str, Any]]:
    """Return a list of field schemas for form generation."""
    if spec.config_type is None or not is_dataclass(spec.config_type):
        return []
    return [
        _field_schema(f)
        for f in _fields(spec.config_type)
    ]


def _fields(cls: type) -> list:
    import dataclasses

    return list(dataclasses.fields(cls))


def spec_defaults(spec: ArchSpec) -> dict[str, Any]:
    return {s["name"]: s["default"] for s in spec_schema(spec)}


def validate_config(spec: ArchSpec, values: dict[str, Any]) -> list[str]:
    """Validate a config dict against the spec alone.

    Returns a list of human-readable error messages; empty means valid.
    """
    errors: list[str] = []
    for s in spec_schema(spec):
        name = s["name"]
        if name not in values or values[name] is None:
            values[name] = s["default"]
        raw = values[name]
        try:
            if s["type"] == "int":
                val = int(raw)
                if s.get("min") is not None and val < s["min"]:
                    errors.append(f"{name} must be >= {s['min']}")
                if s.get("max") is not None and val > s["max"]:
                    errors.append(f"{name} must be <= {s['max']}")
                if s.get("choices") and val not in s["choices"]:
                    errors.append(f"{name} must be one of {s['choices']}")
            elif s["type"] == "float":
                val = float(raw)
                if s.get("min") is not None and val < s["min"]:
                    errors.append(f"{name} must be >= {s['min']}")
                if s.get("max") is not None and val > s["max"]:
                    errors.append(f"{name} must be <= {s['max']}")
            elif s["type"] == "bool":
                val = bool(raw)
            else:
                if s.get("choices") and str(raw) not in [str(c) for c in s["choices"]]:
                    errors.append(f"{name} must be one of {s['choices']}")
        except (TypeError, ValueError):
            errors.append(f"{name} must be a {s['type']}")
    return errors


def normalize_config(spec: ArchSpec, values: dict[str, Any]) -> dict[str, Any]:
    """Coerce a config dict to typed values using the spec's schema."""
    out: dict[str, Any] = {}
    for s in spec_schema(spec):
        name = s["name"]
        raw = values.get(name, s["default"])
        if raw is None:
            raw = s["default"]
        if s["type"] == "int":
            out[name] = int(raw)
        elif s["type"] == "float":
            out[name] = float(raw)
        elif s["type"] == "bool":
            out[name] = bool(raw)
        else:
            out[name] = str(raw)
    return out


# --------------------------------------------------------------------------- #
# Architecture selection -> hash (no torch involved).
# --------------------------------------------------------------------------- #
def arch_hash(selection: dict[str, Any]) -> str:
    """Stable short hash of a full architecture selection."""
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Discovery of user-supplied architecture plugins.
# --------------------------------------------------------------------------- #
DEFAULT_ARCH_DIR = Path(__file__).resolve().parents[2] / "plugins" / "architectures"


def arch_dir() -> Path:
    """The user plugin folder for architectures (configurable via env)."""
    override = os.environ.get("SIMLAB_ARCH_DIR", "").strip()
    return Path(override) if override else DEFAULT_ARCH_DIR


def scan_plugin_dir(path: Path | None = None) -> list[str]:
    """Import every ``*.py`` in the architecture plugin folder.

    Errors are collected and returned as messages; they are never raised.
    Returns the list of error messages (empty if all plugins imported clean).
    """
    folder = Path(path) if path is not None else arch_dir()
    errors: list[str] = []
    if not folder.exists():
        return ["architecture plugin folder not found: " + str(folder)]
    for py in sorted(folder.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = f"simlab_arch_user_{py.stem}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, py)
            if spec is None or spec.loader is None:
                errors.append(f"{py.name}: could not create import spec")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{py.name}: {type(exc).__name__}: {exc}")
        finally:
            sys.modules.pop(mod_name, None)
    return errors


TEMPLATE_PLUGIN = '''"""Example architecture plugin for SimLab.

Dropped into the architecture plugin folder (plugin/architectures/), this file
registers a new backbone spec that appears automatically in the Architecture
panel after Rescan. It imports only the torch-free spec machinery; the class
is loaded lazily when the user builds the policy.

Field metadata supports: ``min``/``max`` for int/float, ``choices`` for int and
string fields, and ``help`` text shown in the panel form.
"""

from dataclasses import dataclass, field

from simlab.training.arch_spec import register_arch


@dataclass
class MyArchConfig:
    hidden: int = field(default=64, metadata={"min": 32, "max": 1024, "help": "Hidden size"})
    activation: str = field(default="relu", metadata={"choices": ["relu", "tanh"]})


class MyBackbone:
    """Placeholder architecture. Implement the Backbone protocol here."""

    state_dim = 64

    def __init__(self, obs_dim: int, hidden: int = 64, fusion: str = "concat", n_drives: int = 8, **kwargs):
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.fusion = fusion
        self.n_drives = n_drives

    def initial_state(self, batch: int):
        raise NotImplementedError("implement me")

    def frame_step(self, x, h, drive=None):
        raise NotImplementedError("implement me")

    def param_count(self) -> int:
        return 0

    def config_dict(self) -> dict:
        return {"backbone": "mybb", "hidden": self.hidden, "fusion": self.fusion}


@register_arch("backbone", "mybb", description="Example custom backbone", config=MyArchConfig)
class RegisteredBackbone(MyBackbone):
    pass
'''


# --------------------------------------------------------------------------- #
# Built-in backbone specs (lazy loaders; torch only on Build).
# --------------------------------------------------------------------------- #
@dataclass
class GRUConfig:
    hidden: int = field(
        default=64,
        metadata={"min": 16, "max": 1024, "help": "GRU hidden size."},
    )


@dataclass
class TransformerConfig:
    context: int = field(
        default=16,
        metadata={"min": 2, "max": 128, "help": "Causal context length K."},
    )
    d_model: int = field(
        default=64,
        metadata={"min": 16, "max": 1024, "help": "Embedding / transformer width."},
    )
    n_heads: int = field(
        default=4,
        metadata={"min": 1, "max": 16, "help": "Attention heads."},
    )
    n_layers: int = field(
        default=2,
        metadata={"min": 1, "max": 16, "help": "Transformer blocks."},
    )


@dataclass
class RDAConfig:
    hidden: int = field(
        default=64,
        metadata={"min": 16, "max": 1024, "help": "Encoder hidden size."},
    )
    rho: float = field(
        default=0.99,
        metadata={"min": 0.0, "max": 1.0, "help": "Trace decay rho for m_t."},
    )
    baseline: float = field(
        default=0.9,
        metadata={"min": 0.0, "max": 1.0, "help": "Baseline retention lam floor."},
    )
    gate_lambda_max: float = field(
        default=0.9984375,
        metadata={"min": 0.0, "max": 1.0, "help": "Retention ceiling."},
    )
    m_scale: float = field(
        default=4.0,
        metadata={"min": 0.01, "max": 100.0, "help": "Trace normalizer."},
    )
    surprise_loss_weight: float = field(
        default=1.0,
        metadata={"min": 0.0, "max": 100.0, "help": "Aux-head loss weight."},
    )
    trainable_gates: bool = field(
        default=False,
        metadata={"help": "Expose trainable gate parameters."},
    )


def _load_backbone(name: str) -> type:
    from simlab.training.backbones import BACKBONES

    return BACKBONES[name]


add_spec(
    "backbone",
    "gru",
    description="The existing GRU cell, reproduced exactly.",
    config=GRUConfig,
    loader=lambda: _load_backbone("gru"),
    source=__name__,
)
add_spec(
    "backbone",
    "transformer",
    description="Small causal Transformer over the last K obs steps.",
    config=TransformerConfig,
    loader=lambda: _load_backbone("transformer"),
    source=__name__,
)
_RDA_VARIANTS = {
    "rda": "fixed decay",
    "rda_const": "learned constant E (control)",
    "rda_g1": "shared surprise gate (G1)",
    "rda_g1_rand": "surprise_rand control gate",
    "rda_g3": "per-drive surprise gate (G3)",
    "rda_g3_tied": "per-drive gate tied to G1",
}
for _name, _desc in _RDA_VARIANTS.items():
    add_spec(
        "backbone",
        _name,
        description=f"RDA variant: {_desc}.",
        config=RDAConfig,
        loader=lambda n=_name: _load_backbone(n),
        source=__name__,
    )


def build_policy_cfg(selection: dict[str, Any]) -> dict[str, Any]:
    """Translate a GUI selection dict into a ``make_policy`` config dict.

    Torch-free. The selection has keys: backbone, drive_core, fusion,
    obs_mode, seed, backbone_config, drive_config. Drive configuration is
    returned separately (the world side consumes it as ``config.drive``).

    RDA backbones need ``d_t`` at every step, so they force
    ``obs_mode="obs_drives"`` (drives appended as the trailing 8 dims).
    """
    backbone = selection.get("backbone", "gru")
    obs_mode = selection.get("obs_mode", "obs_only")
    if backbone in _RDA_VARIANTS:
        obs_mode = "obs_drives"
    return {
        "backbone": backbone,
        "fusion": selection.get("fusion", "concat"),
        "obs_mode": obs_mode,
        "seed": int(selection.get("seed", 0)),
        "kwargs": dict(selection.get("backbone_config", {})),
        "drive_core": selection.get("drive_core", "none"),
    }


def load_arch_toml(path: str | Path) -> dict[str, Any]:
    """Load an architecture selection from a TOML file (torch-free)."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    a = data.get("architecture", {})
    return {
        "backbone": a.get("backbone", "gru"),
        "drive_core": a.get("drive_core", "none"),
        "fusion": a.get("fusion", "concat"),
        "obs_mode": a.get("obs_mode", "obs_only"),
        "seed": int(a.get("seed", 0)),
        "backbone_config": dict(a.get("backbone_config", {})),
        "drive_config": dict(a.get("drive_config", {})),
    }


def dump_arch_toml(selection: dict[str, Any]) -> str:
    """Serialize a selection to TOML text (torch-free)."""
    def _dump_value(key: str, value: Any, indent: int) -> list[str]:
        pad = " " * indent
        if isinstance(value, bool):
            return [f'{pad}{key} = {str(value).lower()}']
        if isinstance(value, (int, float)):
            return [f"{pad}{key} = {value}"]
        if isinstance(value, str):
            return [f'{pad}{key} = "{value}"']
        if isinstance(value, dict):
            lines = [f"{pad}{key}"]
            for k, v in value.items():
                lines.extend(_dump_value(k, v, indent + 2))
            return lines
        return [f"{pad}{key} = {json.dumps(value)}"]

    lines = ["[architecture]"]
    lines.extend(_dump_value("backbone", selection.get("backbone", "gru"), 0))
    lines.extend(_dump_value("drive_core", selection.get("drive_core", "none"), 0))
    lines.extend(_dump_value("fusion", selection.get("fusion", "concat"), 0))
    lines.extend(_dump_value("obs_mode", selection.get("obs_mode", "obs_only"), 0))
    lines.extend(_dump_value("seed", int(selection.get("seed", 0)), 0))
    lines.append("[architecture.backbone_config]")
    for k, v in (selection.get("backbone_config") or {}).items():
        lines.extend(_dump_value(k, v, 2))
    lines.append("[architecture.drive_config]")
    for k, v in (selection.get("drive_config") or {}).items():
        lines.extend(_dump_value(k, v, 2))
    return "\n".join(lines) + "\n"


def write_template(folder: Path | None = None) -> Path:
    """Write the 'New architecture template' plugin file into the folder."""
    target = (Path(folder) if folder is not None else arch_dir()) / "exam_arch.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(TEMPLATE_PLUGIN, encoding="utf-8")
    return target
