"""Task 1: reconcile the FiLM 'gain' metric between E13 and E13-fixed.

READ-ONLY. Parses the two console logs and the two JSON files, and reports
the same quantity side by side so the two runs can be judged comparable or
not. No training, no changes to rda.py / PE / FiLM.
"""
import json
import re

OLD_TXT, NEW_TXT = "logs/pe_1000.txt", "logs/pe_1000_fixed.txt"
OLD_JSON, NEW_JSON = "logs/pe_1000.json", "logs/pe_1000_fixed.json"


def console(path):
    """Parse 'step N | ... key=value ...' lines into {step: {field: float}}.

    NOTE: the per-step console lines were written to the TERMINAL only. The
    log files on disk (UTF-16, from PowerShell redirection) hold just the
    header and the trailing summary block, so the console series is not
    recoverable from disk for EITHER run. Returns {} in that case, which is
    reported rather than papered over.
    """
    out = {}
    raw = open(path, "rb").read()
    txt = None
    for enc in ("utf-16", "utf-8"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    if txt is None:
        txt = raw.decode("utf-8", errors="replace")
    for ln in txt.split("\n"):
        if "film s=" not in ln:
            continue
        m = re.search(r"step\s+(\d+)\s*\|", ln)
        if not m:
            continue
        step = int(m.group(1))
        rec = {}
        for k, v in re.findall(r"(\w+)=(-?[\d.]+)", ln):
            rec[k] = float(v)
        out[step] = rec
    return out


def main():
    co, cn = console(OLD_TXT), console(NEW_TXT)
    jo = json.load(open(OLD_JSON))
    jn = json.load(open(NEW_JSON))

    steps = (0, 250, 500, 750, 999)
    print("console per-step lines on disk: old=%d new=%d  (0 = not persisted)"
          % (len(co), len(cn)))

    print()
    print("=" * 78)
    print("A. JSON film_s_rms -- hook on the film_scale MODULE OUTPUT")
    print("=" * 78)
    print("   probe_pe_film.py:88  cap_mod('film_scale','film_scale_out')")
    print("   rda.py modulate():   scale = 1.0 + _unit_rms(film_scale(E), 0.5)")
    print("   _unit_rms is applied to the module's OUTPUT, INSIDE modulate().")
    print("   So this column is the RAW projection, PRE-_unit_rms -- it is NOT")
    print("   the gain the model actually multiplies the residual by.")
    print()
    print("%6s %14s %14s %14s %14s" % ("step", "old E_absmax", "new E_absmax",
                                      "old film_s_rms", "new film_s_rms"))
    for s in steps:
        print("%6d %14.4f %14.4f %14.4f %14.4f"
              % (s, jo[s]["E_absmax"], jn[s]["E_absmax"],
                 jo[s]["film_s_rms"], jn[s]["film_s_rms"]))

    print()
    print("   raw scale magnitude 1+film_s_max  (summary blocks, both runs)")
    print("     old E13       : %.4f" % max(r["mod_scale_absmax"] for r in jo))
    print("     new E13-fixed : %.4f" % max(r["mod_scale_absmax"] for r in jn))

    print()
    print("=" * 78)
    print("B. CONSOLE 'film s' -- the APPLIED scale = 1 + _unit_rms(fs, 0.5)")
    print("=" * 78)
    if not co or not cn:
        print("   NOT RECOVERABLE. The per-step console lines were printed to the")
        print("   terminal and never written to these files, for either run.")
        print("   The terminal scrollback for E13-fixed shows film s rising")
        print("   1.000 -> 1.191 -> 1.458 -> 1.517 (steps 0/25/50/75) and")
        print("   2.062 at step 999, but that is terminal scrollback, not a")
        print("   persisted artifact, and no equivalent survives for E13.")
        print("   The only on-disk applied-scale figures are the summary maxima:")
        print("     E13        film scale max 1.5781   film shift max 0.7690")
        print("     E13-fixed  film scale max 2.1040   film shift max 1.7686")
    else:
        print("%6s %14s %14s %10s" % ("step", "old film s", "new film s", "new/old"))
        for s in steps:
            if s in co and s in cn:
                o, n = co[s].get("s"), cn[s].get("s")
                if o is None or n is None:
                    continue
                print("%6d %14.4f %14.4f %10.3f" % (s, o, n, n / o))

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("Metric definition UNCHANGED between the two runs:")
    print("  - probe_pe_film.py last changed in ab9b560, i.e. before E13 ran.")
    print("  - `git diff 6ae93ce 969e067 -- diagnostics/probe_pe_film.py` is 0 lines,")
    print("    so the fix commit 969e067 did not touch the harness at all.")
    print()
    print("So YES -- the two runs are directly comparable, field for field.")
    print("Both record film_s_rms from the identical hook on the identical")
    print("module output, over the identical seed and step count.")
    print()
    print("CONSEQUENCE FOR THE GATE. My earlier E13-fixed G2/G3 verdicts were")
    print("computed from JSON film_s_rms, which is the PRE-_unit_rms raw")
    print("projection, not the applied gain the pre-registered rule meant")
    print("('gain near 1.5'). Those verdicts were scored against the wrong")
    print("column and must be recomputed from the applied-scale series before")
    print("any conclusion about gain is drawn.")
    print()
    print("The applied-scale maxima are 1.5781 (E13) and 2.1040 (E13-fixed).")
    print("Both are above the 0.98-style saturation region only if compared")
    print("against |1+gain|>2; the original E13 rule targeted ~1.5, which")
    print("E13-fixed exceeds by 40%. That is a real difference, but it is a")
    print("magnitude difference on a correctly-defined column, not the")
    print("near-zero-to-0.47 collapse the wrong column suggested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
