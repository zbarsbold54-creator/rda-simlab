"""Validate the README Mermaid block against GitHub's stricter parser.

GitHub uses a recent mermaid that rejects some shorthand the local
renderer tolerates (notably [(("label"))] cylinder nodes and |"label"|
edge text). This extracts the block, syncs dev/rda_architecture.mmd,
and flags constructs known to break on GitHub.

Run from anywhere:  python diagnostics/check_mermaid.py
Paths are anchored to the repo root via __file__, not the CWD.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TICK = chr(96) * 3
s = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
m = re.search(re.escape(TICK) + r"mermaid\n(.*?)\n" + re.escape(TICK), s, re.S)
if not m:
    print("NO MERMAID BLOCK FOUND")
    sys.exit(1)
block = m.group(1)
with open(os.path.join(REPO, "dev", "rda_architecture.mmd"), "w", encoding="utf-8") as f:
    f.write(block + "\n")
print("extracted %d lines from README" % len(block.split("\n")))

bad = []
for i, line in enumerate(block.split("\n"), 1):
    stripped = line.strip()
    if not stripped:
        continue
    # Only compound/exotic SHAPES are rejected by GitHub: [(("cylinder"))],
    # [[[subroutine]]], [/parallelogram/], >asymmetric]. Plain ["label"] and
    # {{hexagon}} are fine. So look for those specific openings immediately
    # after the node id -- NOT a bare "[" which is the ordinary rect shape.
    if re.search(r'[A-Za-z0-9_]\s*[\(\[/]\s*[\(\[\{/>]', stripped):
        bad.append((i, "exotic shape syntax", stripped[:70]))
    # dashed edge WITH a label  -.->|"text"|
    if re.search(r'-\.->\s*\|', stripped):
        bad.append((i, "dashed edge w/ label", stripped[:70]))
    # any edge label at all |"text"| -- allowed, but only for non-dashed edges
    if re.search(r'-\.->\s*\|', stripped):
        bad.append((i, "dashed edge w/ label", stripped[:70]))
    if re.search(r'\|[^|]*"', stripped) and '-.->' not in stripped:
        pass  # plain edge label, fine on GitHub
    # ~./ html entities
    if "~/" in stripped:
        bad.append((i, "html entity", stripped[:70]))
    # style props GitHub's renderer ignores
    if re.search(r'classDef.*(stroke-dasharray|opacity)', stripped):
        bad.append((i, "ignored style prop", stripped[:70]))

for i, kind, txt in bad:
    print("  line %d [%s]: %s" % (i, kind, txt))
print("suspicious constructs: %d" % len(bad))

# Sanity: every referenced node should be defined before use where it matters.
defined = set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)\s*[\[\(\{]', block, re.M))
defined |= set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)\[\[', block, re.M))
defined |= set(re.findall(r'subgraph\s+([A-Z][A-Z0-9_]*)', block))
used = set(re.findall(r'([A-Z][A-Z0-9_]*)\s*[-=.]', block))
print("undefined refs:", sorted(used - defined) or "none")
sys.exit(1 if bad else 0)
