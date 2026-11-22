"""Generate a local chart of gains and losses at the two localization cutoffs."""
from html import escape

from medsyringe.data import read_json
from scripts.experiments.relocalization.pair_relocalization import RESULT, WORK
from scripts.experiments.relocalization.pair_relocalization_probe import RESULT as PROBE

COHORTS = {
    "reserve": "New public acquisitions",
    "box-quality-reserve": "Box quality study",
    "box-quality-exposed": "Object-verifier study",
    "earlier-fresh": "Earlier ranking acquisitions",
    "earlier-replay": "Earlier crowded and lookalike scenes",
}
ARMS = {
    "previous": "Previous relocalizer",
    "geometry": "Geometry probe",
    "joint": "Visual joint probe",
    "factorized": "Identity and improvement probe",
}
HEADER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Count-preserving syringe relocalization</title>
<style>
body { margin:0; font:14px/1.4 var(--font-sans); color:var(--foreground) }
h2 { margin:16px 0 3px; font-size:16px }
p { margin:0 0 6px; color:var(--muted-foreground) }
.row { display:grid; grid-template-columns:minmax(122px,1.25fr) repeat(2,minmax(76px,1fr));
       gap:8px; min-height:31px; align-items:center }
.header { color:var(--muted-foreground); font-size:12px }
.cell { display:grid; grid-template-columns:1fr 45px; gap:4px; align-items:center }
.track { display:flex; height:9px }
.track span { height:100%; border-radius:1px }
.gain { background:var(--chart-1,#2563eb) }
.loss { background:var(--chart-2,#ea580c) }
.count { font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap }
</style></head><body>
<p>Changes from the released detector, shown as gains / losses.</p>"""


def cell(values, total):
    gain, lost = values
    label = f"{gain} gained and {lost} lost"
    return (f'<div class="cell" title="{label}" aria-label="{label}">'
            f'<div class="track"><span class="gain" style="width:{gain / total * 100:.4f}%"></span>'
            f'<span class="loss" style="width:{lost / total * 100:.4f}%"></span></div>'
            f'<span class="count">{gain} / {lost}</span></div>')


def render():
    main = read_json(RESULT)["cohorts"]
    active = read_json(PROBE)["cohorts"]
    out = [HEADER]
    for key, record in main.items():
        guard = {"previous": record["variants"]["previous"]["guard"]}
        guard.update({arm:v["guard"] for arm,v in active[key].items()})
        totals = {arm: [(len(v["gained"][t]), len(v["lost"][t])) for t in ("0.5","0.75")]
                  for arm,v in guard.items()}
        scale = max(1, *(sum(x) for values in totals.values() for x in values))
        out.append(f'<h2>{escape(COHORTS[key])}</h2>')
        base = record["baseline"]
        out.append(f'<p>{base["images"]} photos, {base["objects"]} reference syringes</p>')
        out.append('<div class="row header"><span>Policy</span><span>IoU 0.50</span><span>IoU 0.75</span></div>')
        for arm, values in totals.items():
            out.append(f'<div class="row"><span>{escape(ARMS[arm])}</span>'
                       + "".join(cell(x,scale) for x in values) + "</div>")
    out.append("</body></html>")
    (WORK / "comparison.html").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    render()
