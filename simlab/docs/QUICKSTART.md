# Quickstart

This page gives the exact commands and click paths for the SimLab GUI and the
Research cockpit. Everything runs inside `C:\AI\SimulationCode`; use the
project virtualenv (`.venv`).

## Open the GUI

```powershell
cd C:\AI\SimulationCode
.\.venv\Scripts\python.exe -c "from simlab.gui.app import main; main()"
```

The Research cockpit is a dock inside the main window. If it is not visible,
open it from the View menu (View -> Research cockpit).

## Build and apply an architecture

1. In the **Architecture** toolbar, open the architecture dropdown.
2. Pick a preset (for example `gru`, `transformer`, or an RDA preset such as
   `rda_gru_surprise`).
3. Click **Build**. A status line shows "Loading torch…" then
   "Building <preset>…"; keep the window responsive while it runs.
4. Click **Apply to world**. The world view starts stepping with that policy.
   Drives (`d_t`) are shown in the world inspector for every policy; retention
   inner state (`E_t`, `lambda_t`, `m_t`, `s_t`) appears for RDA presets.

## Run the regression experiment (gru_legacy, 300k steps)

1. Go to the **Experiment** tab.
2. Set **Name** to e.g. `regression_gru_legacy`.
3. Set **Total steps** to `300000`.
4. Set **Rollout length** to `1024`.
5. Leave **Workers** at `4` (the default; changing it shows a warning).
6. Set **Train seeds** to `100000`.
7. Leave **Dev seeds** at `0, 1, 2, 3, 4` and scenarios `balanced` + `hard`
   checked.
8. In the arm list, select `gru_legacy`. The parameter table fills with the
   per-arm parameter counts.
9. Click **Validate** (must pass), then **Save TOML**. This writes a run file
   under `runs/`.
10. Switch to the **Queue** tab and click **Launch grid**. Each job runs in a
    separate subprocess; the table shows per-run status.

## Watch it live

1. In the **Live** tab, select the run directory (it is pre-filled after a
   launch) and check **Follow latest checkpoint**.
2. The world view plays the newest checkpoint every N ms (set the interval in
   the box next to the checkbox; default 3000 ms). Training stays in its
   subprocess.
3. The header above the plot shows run name, steps done, steps/s, and the last
   quick eval cap%.

## Read the Results tab

1. Wait for runs to finish (status `done` in the Queue tab), or follow live.
2. In the **Results** tab click **Load eval files…** and pick the per-seed
   eval JSON files from `runs/.../runs/<job>/`.
3. Choose a **Control arm** (`rda_const` for RDA arms) to see per-seed
   differences.
4. Click **Export CSV** to save the aggregated table.

## Quick demo

Click **Quick demo** in the Queue tab. It launches a 20k-step
`rda_gru_surprise` run (4 workers, seed 100001) and wires Live, Curves,
Results, and Gates to it. The whole pipeline is usable in about a minute.

## Notes

- All six cockpit tabs have empty states that tell you the next step when
  there is no data (for example "No data: launch a run in the Queue tab").
- Long operations (torch load, build, gate traces) run off the GUI thread;
  the status lines keep you informed.
- Help -> Copy diagnostics copies the last 200 `logs/gui.log` lines plus
  Python/torch versions and the active preset to the clipboard.
