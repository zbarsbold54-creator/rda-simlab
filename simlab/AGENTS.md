# Standing project rules

- Work only inside `C:\AI\SimulationCode`. Never touch `C:\AI\nanochat`. Do not run commands that affect the system outside this folder (shutdown, restart, killing processes you did not start, system settings) without asking me in plain text first.
- Run Python and tools from `.venv`. Allowed dependencies: numpy, PySide6, pyqtgraph, pytest, ruff. Ask before installing anything else. Torch must never be imported by `simlab.core`.
- The core (`simlab.core`) never imports the GUI or Qt. The World owns all simulation state; the GUI only reads immutable snapshots. All randomness comes from the one seeded numpy Generator. All GUI colors, sizes and fonts live in `gui/theme.py` and `style.qss`.
- Keep files to about 300 lines, use type hints and dataclasses, use logging rather than `print` in library code, and keep Ruff clean. Do not use emojis.
- Never claim a phase or item is done without showing raw command output. Stop at each gate and wait for me. Update `PROGRESS.md` and commit after every phase.
- Explain in one plain sentence what any shell or git command does and why.
