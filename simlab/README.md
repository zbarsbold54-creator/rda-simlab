# SimLab

SimLab is a modular 2D survival simulation with a headless core and a Qt simulation studio.

## Requirements

- Python 3.11 or newer
- Windows PowerShell examples below; use the equivalent virtual-environment activation command in another shell.

## Fresh checkout setup

```powershell
git clone <repository-url> SimLab
cd SimLab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, run commands with `.venv\Scripts\python.exe -m ...` directly. Test and lint dependencies are in the `dev` extra.

## Run SimLab

```powershell
python -m simlab.cli run --steps 100 --seed 1
python -m simlab.cli gui
python -m simlab.cli bench --agents 20 --steps 1000
```

The GUI opens the Simulation Studio. Use the toolbar to play, pause, step, reset, and change speed; the docks expose scenario settings, agent inspection, drives, timeline, snapshots, and replay.

## Verify a checkout

```powershell
python -m pytest -v
ruff check .
ruff format --check .
```

See [docs/EXTENDING.md](docs/EXTENDING.md) for adding registry plugins. See [PROGRESS.md](PROGRESS.md) for implementation history and gate results.
