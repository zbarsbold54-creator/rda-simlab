"""
Compute and cache the neutral-paragraph reference final state for the demo.

The per-token jump signal does not fire: update() runs T-1 times inside one
forward pass, so E_t relaxes geometrically along POSITION and the trajectory
is a near-monotone ramp whose total excursion is small (~0.05/token) and
largely independent of content. Instead of looking for spikes, the demo
compares each input's SETTLED final state against a fixed reference captured
from a long plain neutral paragraph.

Run:  python demo/make_reference.py
Writes demo/reference_state.json  {text_sha, n_tokens, state[8]}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import torch

from nanochat.tokenizer import get_tokenizer
from nanochat.rda import DRIVE_DIMS

OUT = os.path.join(HERE, "reference_state.json")

# Long, plain, neutral expository prose. No narrative tension, no topic shift,
# no rare tokens -- the "nothing is happening here" control.
NEUTRAL = (
    "Water is a polar molecule, which means that the slight positive charge "
    "concentrated on the hydrogen atoms is attracted to the slight negative "
    "charge on the oxygen. This single property explains a great deal about "
    "the behaviour of the substance in its liquid and solid forms. Ice floats "
    "because the crystal lattice that forms when the molecules line up in a "
    "fixed hexagonal arrangement is less dense than the random arrangement "
    "found in liquid water, so a body of ice displaces less mass than the "
    "water that would otherwise occupy the same volume. The same polarity "
    "allows the molecules to climb thin tubes by adhering to the walls of the "
    "container, which is the mechanism behind the rise of water in narrow "
    "glass tubes. Surface tension follows from cohesion between neighbouring "
    "molecules, and it is strong enough to support small insects that would "
    "otherwise be too light to remain on the surface. "
    "Most of the chemistry of life takes place in water for these reasons. "
    "Proteins fold into specific shapes because the polar and nonpolar regions "
    "interact differently with the surrounding water, and the resulting "
    "three dimensional shape determines what a protein is able to bind to. "
    "The temperature range over which liquid water exists is unusually wide "
    "for a substance of its size, and this is largely because of the energy "
    "required to disrupt the hydrogen bonded network that holds the molecules "
    "together. Breaking that network costs a substantial amount of thermal "
    "energy, which raises the boiling point considerably above what would "
    "otherwise be expected. "
    "When the liquid is heated, the molecules move more rapidly and a greater "
    "fraction of them are able to escape the surface into the vapour phase. "
    "The rate of this escape increases smoothly with temperature, which is why "
    "evaporation from a surface is faster on a warm day than on a cool one. "
    "The reverse process, condensation, occurs when molecules in a gas phase "
    "lose energy and return to the liquid, and it happens whenever the "
    "concentration of water vapour exceeds the value that the air can hold at "
    "the current temperature. That limiting concentration is what is measured "
    "as relative humidity, expressed as a percentage of the maximum. "
    "Clouds form when air is forced to rise and cool, and the expanding air "
    "reaches a temperature at which the water it holds condenses onto small "
    "particles of dust or smoke. These droplets are typically only a few "
    "microns across, and it is their enormous combined surface area that makes "
    "visible clouds such an effective reflector of sunlight. "
    "Estimating the total amount of fresh water stored in ice caps and "
    "groundwater is a matter of considerable uncertainty, because those "
    "reserves are distributed unevenly across the surface of the planet and "
    "much of the easily accessible ice is located in regions that are "
    "difficult to survey directly. "
    "In the laboratory these same properties are exploited deliberately. "
    "Chromatography separates a mixture by passing it through a column packed "
    "with an absorbent solid, exploiting the fact that different compounds "
    "adhere to the stationary phase with different strengths. Density "
    "gradient centrifugation achieves a similar separation by suspending "
    "particles in a medium whose density varies from top to bottom, so that "
    "each component settles at the depth where it is neutrally buoyant. Both "
    "techniques rely on the same underlying idea of exploiting small "
    "differences in physical behaviour to separate things that are chemically "
    "similar. "
    "The measurement of temperature itself is a good example of careful "
    "calibration. A thermometer is read against a fixed reference, and the "
    "scale is constructed so that the freezing and boiling points of water "
    "occupy particular marked positions under stated pressure conditions. "
    "Without that agreement between instruments, comparisons of recorded "
    "values would be meaningless. The same principle applies to scales, which "
    "are balanced against a standard mass before use, and to electrical "
    "meters, which are checked against a reference instrument maintained by a "
    "standards body. In each case the instrument is only as trustworthy as its "
    "last comparison with something held fixed. "
    "Keeping records of such comparisons over long periods allows drift to be "
    "detected, because an instrument that has slowly changed will "
    "progressively disagree with the standard even though it continues to "
    "give plausible looking numbers. Regular recalibration is therefore part "
    "of ordinary maintenance rather than a response to visible failure. "
    "The same general approach appears throughout the physical sciences, "
    "where a measurement is only meaningful relative to a stated uncertainty "
    "and a stated reference. Results are expected to carry an error estimate, "
    "and independent replication is used to confirm that the estimate is "
    "reasonable. Where two methods disagree by more than their stated "
    "uncertainties allow, the disagreement itself becomes the subject of "
    "further investigation rather than something to be quietly averaged away. "
    "This habit of treating disagreement as information rather than as an "
    "inconvenience is one of the more durable features of how empirical work "
    "is conducted. "
    "In practical terms the procedure is unremarkable. A measurement is taken, "
    "the instrument is checked against a reference, the result is recorded "
    "with its uncertainty, and the record is retained so that a later "
    "measurement can be compared against it. Most of the intellectual content "
    "lies in deciding what to measure and how precisely, and relatively little "
    "in the mechanics of the measurement itself. "
    "A useful habit when reading a reported value is to ask what would have "
    "been observed had the underlying quantity been somewhat different, since "
    "a quantity that cannot affect the result is not being measured. This "
    "question is easy to ask and frequently reveals that some part of an "
    "apparatus has been included for historical reasons without any effect on "
    "the number being reported. "
    "Similar considerations apply to the design of any experiment intended to "
    "be repeated. The procedure should be written down in enough detail that "
    "an unfamiliar person could follow it, materials should be described in a "
    "way that allows them to be obtained, and the expected range of results "
    "should be stated so that a later run producing something outside that "
    "range is recognised as unusual. The purpose of all this is not "
    "ceremony. It is to make the result available to people who were not "
    "present when it was obtained, which is the condition under which any "
    "accumulation of knowledge is possible at all."
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.rda import GPTWithRDA
    from nanochat.checkpoint_manager import load_checkpoint
    from nanochat.common import get_base_dir

    ck = os.path.join(get_base_dir(), "base_checkpoints", "model-b-primal")
    metas = sorted(f for f in os.listdir(ck) if f.startswith("meta_"))
    meta = json.load(open(os.path.join(ck, metas[-1])))
    md, _, _ = load_checkpoint(ck, int(meta["step"]), device)
    if device.type in {"cpu", "mps"}:
        md = {k: (v.float() if v.dtype == torch.bfloat16 else v) for k, v in md.items()}
    md = {k.removeprefix("_orig_mod."): v for k, v in md.items()}
    cfg = GPTConfig(**meta["model_config"])
    base = GPT(cfg); base.init_weights()
    model = GPTWithRDA(base, debug_steps=0)
    miss, unexp = model.load_state_dict(md, strict=False)
    assert not miss and not unexp, (miss, unexp)
    model = model.to(device).eval()

    tok = get_tokenizer()
    ids = tok.encode(NEUTRAL, prepend=tok.get_bos_token_id())
    if len(ids) > 1024:
        ids = ids[:1024]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, x)
    E = out["E_next_seq"][0]                    # (T-1, 8)
    final = E[-1].tolist()

    import hashlib
    payload = {
        "tag": "model-b-primal",
        "step": int(meta["step"]),
        "n_tokens": len(ids),
        "dims": list(DRIVE_DIMS),
        "state": final,
        "text_sha256": hashlib.sha256(NEUTRAL.encode("utf-8")).hexdigest(),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  n_tokens = {len(ids)}   (transient window is ~20-100 tokens)")
    print("  reference final state:")
    for d, v in zip(DRIVE_DIMS, final):
        print(f"    {d:<20} {v:+.4f}")


if __name__ == "__main__":
    main()
