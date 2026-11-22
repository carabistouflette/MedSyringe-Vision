"""Draw the frozen box quality results as a standalone local comparison."""

from html import escape

from medsyringe.data import read_json
from scripts.syringe_box_quality import RESULT, WORK

TITLES = {
    "reserve": "New public acquisitions",
    "exposed": "Earlier object-verifier acquisitions",
    "earlier-fresh": "Earlier ranking acquisitions",
    "earlier-replay": "Earlier crowded and lookalike scenes",
}
LABELS = {
    "baseline": "Released detector",
    "previous-calibrated": "Previous recovery",
    "previous-active": "Previous active probe",
    "coupled": "Coupled score",
    "geometry-iou": "Geometry IoU",
    "visual-iou": "Visual IoU",
    "pair-iou": "Pair-supervised IoU",
    "identity-only": "Fitted identity",
    "identity-pair-iou": "Fitted identity and IoU",
    "shared-identity-pair-iou": "Shared identity and IoU",
}
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Syringe detection and box quality</title>
<style>
body { margin:0; font:14px/1.4 var(--font-sans); color:var(--foreground) }
h2 { margin:16px 0 2px; font-size:16px }
p { margin:0 0 8px; color:var(--muted-foreground) }
.row { display:grid; grid-template-columns:minmax(126px,1fr) minmax(74px,1fr) 98px;
       gap:8px; min-height:30px; align-items:center }
.legend { display:flex; gap:12px; margin-bottom:6px; flex-wrap:wrap }
.legend span::before { content:""; display:inline-block; width:.72em; height:.72em;
       background:var(--bar); margin-right:.36em; border-radius:2px }
.track { display:flex; height:10px }
.bar { background:var(--bar); border-radius:1px }
.true { --bar:var(--chart-1,#2563eb) }
.false { --bar:var(--chart-2,#ea580c) }
.missed { --bar:var(--chart-3,#7c3aed) }
.value { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums }
</style></head><body>
<div class="legend"><span class="true">True positive</span><span class="false">False positive</span>
<span class="missed">Missed</span></div>"""


def chart(title, baseline, rows, show):
    cap = max(sum(rows[key][kind] for kind in ("tp", "fp", "fn")) for key in show)
    yield f'<h2>{escape(title)}</h2><p>{baseline["images"]} photos, {baseline["objects"]} reference syringes</p>'
    for key in show:
        m = rows[key]
        counts = [m[k] for k in ("tp", "fp", "fn")]
        bars = "".join(f'<div class="bar {kind}" style="width:{100 * n / max(cap, 1):.5f}%"></div>'
                       for kind, n in zip(("true", "false", "missed"), counts, strict=True))
        label = f'{counts[0]} true positives, {counts[1]} false positives, {counts[2]} missed'
        yield (f'<div class="row"><span>{escape(LABELS[key])}</span>'
               f'<div class="track" role="img" aria-label="{label}">{bars}</div>'
               f'<span class="value" aria-hidden="true">{" / ".join(map(str, counts))}</span></div>')


def render():
    result = read_json(RESULT)
    chosen = (result["winner"] or {"arm": "pair-iou"})["arm"]
    display = list(dict.fromkeys(("baseline", "previous-calibrated", "visual-iou", "pair-iou", chosen)))
    parts = [PAGE]
    for role, entry in result["cohorts"].items():
        baseline = entry["baseline"]
        values = dict(baseline=baseline, **{k: v["metrics"] for k, v in entry["variants"].items()})
        parts.extend(chart(TITLES[role], baseline, values, display))
    parts.append("</body></html>")
    (WORK / "comparison.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    render()
