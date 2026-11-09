"""
Task 1: was the optimizer REWARDED for saturating the state?

Extracts the loss series from the killed run and fits the slope before vs after
the saturation onset, to see whether the loss curve inflects at that point.

INDEXING WARNING (this matters):
The killed run logged the state counter once per MICRO-step (`[rda]
step N`) but the loss only once per OPTIMIZER step (`step NNNNN/00600 | loss:`).
grad_accum_steps was 8, so the loss line following micro-step M belongs to
optimizer step floor(M/8). Loss at an exact micro-step is therefore not
available except where the two coincide; we use the nearest optimizer step at
or before each requested micro-step, and say so explicitly.

Saturation onset: micro-step 129, where |E_next| first reads 2.8284 = sqrt(8).
That falls between optimizer steps 15 (micro 128) and 16 (micro 136).
"""
import math

# (optimizer_step, micro_step_at_which_this_loss_was_logged, loss)
LOSS = [
    (0,   8,  10.397895), (1,  16, 10.395393), (2,  24, 10.390843),
    (3,  32, 10.384501), (4,  40, 10.376534), (5,  48, 10.365287),
    (6,  56, 10.350220), (7,  64, 10.331044), (8,  72, 10.310085),
    (9,  80, 10.279119), (10, 88, 10.228955), (11, 96, 10.178369),
    (12, 104, 10.101475), (13, 112, 10.009907), (14, 120, 9.869079),
    (15, 128, 9.678352), (16, 136, 9.470502), (17, 144, 9.279262),
    (18, 152, 9.116363), (19, 160, 8.973216), (20, 168, 8.786981),
    (21, 176, 8.637729), (22, 184, 8.507816),
]
# LR multiplier logged alongside each of those lines (warmup is active here).
LRM = {0:0.03,1:0.05,2:0.07,3:0.10,4:0.12,5:0.15,6:0.17,7:0.20,8:0.23,
       9:0.25,10:0.28,11:0.30,12:0.33,13:0.35,14:0.38,15:0.40,16:0.42,
       17:0.45,18:0.47,19:0.50,20:0.53,21:0.55,22:0.57}

# |E_next| observed at the micro-step where each optimizer step completed.
ENORM = {0:1.82, 1:1.82, 2:1.83, 3:1.82, 4:1.87, 5:1.85, 6:2.01, 7:2.04,
         8:2.12, 9:2.08, 10:2.39, 11:2.53, 12:2.60, 13:2.81, 14:2.83,
         15:2.8284, 16:2.8284, 17:2.8284, 18:2.8284, 19:2.8284,
         20:2.8284, 21:2.8284, 22:2.8284}
SQRT8 = math.sqrt(8)

REQUESTED = [(100, 129, "saturation onset"), (115, 129, "before onset"),
             (129, 129, "ONSET (sqrt(8) first reached)"),
             (150, 129, "after onset"), (186, 129, "last available")]

def nearest(micro):
    """Loss logged at or before `micro`."""
    cands = [r for r in LOSS if r[1] <= micro]
    if cands:
        return cands[-1]
    return LOSS[0]

def slope(rows):
    """Least-squares slope of loss vs optimizer step."""
    n = len(rows)
    if n < 2:
        return float('nan')
    xs = [r[0] for r in rows]; ys = [r[2] for r in rows]
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    return num/den if den else float('nan')

print("="*74)
print("TASK 1  Loss vs the saturation point")
print("="*74)
print()
print("Loss is logged once per OPTIMIZER step (8 micro-steps). Requested points")
print("are MICRO-step numbers, so each is mapped to the nearest loss at-or-before.")
print()
print(f"{'req micro':>9} {'-> opt':>7} {'loss':>10} {'|E_next|':>9} {'lrm':>6}  note")
print("-"*74)
for micro, _, note in REQUESTED:
    s, m, l = nearest(micro)
    e = ENORM.get(s, float('nan'))
    print(f"{micro:>9} {s:>7} {l:>10.6f} {e:>9.4f} {LRM.get(s,0):>6.2f}  {note}"
          + ("   <- no loss logged at this micro" if micro > LOSS[-1][1] else ""))
print()

ONSET_OPT = 16  # first optimizer step whose |E_next| is pinned at sqrt(8)
before = [r for r in LOSS if 11 <= r[0] <= ONSET_OPT-1]
after  = [r for r in LOSS if r[0] >= ONSET_OPT]

print("-"*74)
print(f"Slope BEFORE onset (opt steps {before[0][0]}-{before[-1][0]}, "
      f"|E| {ENORM[before[0][0]]:.2f} -> {ENORM[before[-1][0]]:.2f})")
print(f"Slope AFTER  onset (opt steps {after[0][0]}-{after[-1][0]}, "
      f"|E| pinned at {SQRT8:.4f})")
print("-"*74)
sb, sa = slope(before), slope(after)
print(f"  before: {sb:+.6f} loss per optimizer step   ({len(before)} points)")
print(f"  after : {sa:+.6f} loss per optimizer step   ({len(after)} points)")
print(f"  change: {sa-sb:+.6f}  ({100.0*(sa-sb)/abs(sb):+.1f}% relative)")
print()

deltas = [(LOSS[i+1][0], LOSS[i+1][2]-LOSS[i][2]) for i in range(len(LOSS)-1)]
print("per-step deltas (looking for a discontinuity at the onset):")
for s, d in deltas:
    mark = "  <-- onset boundary" if s == ONSET_OPT else ""
    print(f"  step {s:>2}: {d:+.6f}{mark}")
print()

allrows = [r for r in LOSS if r[0] >= 11]
sall = slope(allrows)
print(f"Smoothness check: slope over the whole window (opt 11-22) = {sall:+.6f}")
print(f"  before {sb:+.6f} / after {sa:+.6f} straddle it: "
      f"{'YES' if sb <= sa or abs(sa-sb) < 0.01 else 'NO - real break'}")
print()
print("CONFOUND: the LR warmup multiplier is still ramping across this whole")
print(f"window (lrm {LRM[11]:.2f} at opt 11 -> {LRM[22]:.2f} at opt 22), so the")
print("loss slope reflects a growing effective step size, not pure learning.")
print()
print(f"Saturation onset in optimizer-step terms: step {ONSET_OPT}")
print(f"  |E_next| at that point = {ENORM[ONSET_OPT]:.4f} vs sqrt(8) = {SQRT8:.4f}")
print(f"  fraction of sqrt(8) already reached: {100.0*ENORM[ONSET_OPT]/SQRT8:.2f}%")
