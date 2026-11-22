"""Create a local comparison of reference gains and harmful replacement edits."""
from html import escape

from medsyringe.data import read_json
from scripts.replacement_appearance import POLICY, RESULT, WORK

LABELS = {
    "ungated-joint": "Visual pair rule",
    "released-veto": "Issued SigLIP 2 veto",
    "targeted-zero": "Targeted SigLIP 2 at zero",
    "targeted-proposal": "Targeted proposed crop",
    "targeted-pair": "Targeted both crops",
    "typed-proposal": "Typed hybrid on proposal",
    "selected": "Selected appearance rule",
}
HEADER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Appearance verification for syringe recropping</title>
<style>
body { margin:0; font:14px/1.4 var(--font-sans); color:var(--foreground) }
h2 { font-size:16px; margin:16px 0 2px }
p { color:var(--muted-foreground); margin:0 0 8px }
.row { display:grid; grid-template-columns:minmax(132px,1.35fr) minmax(92px,1fr) 92px;
       gap:8px; align-items:center; min-height:31px }
.legend { display:flex; gap:12px; flex-wrap:wrap }
.legend span::before { content:""; display:inline-block; width:.75em; height:.75em;
       margin-right:.33em; background:var(--bar); border-radius:2px }
.gain { --bar:var(--chart-1,#2563eb) }
.negative { --bar:var(--chart-2,#ea580c) }
.harm { --bar:var(--chart-3,#7c3aed) }
.track { display:flex; height:10px }
.track span { background:var(--bar); border-radius:1px }
.value { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap }
.crop { display:grid; grid-template-columns:minmax(132px,1.35fr) repeat(2,minmax(72px,1fr));
        gap:8px; align-items:center; min-height:31px }
.crop span:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums }
.column-label { color:var(--muted-foreground); font-size:12px }
</style></head><body>
<div class="legend"><span class="gain">New matches at IoU 0.50</span>
<span class="negative">Edits on negatives</span><span class="harm">Other harmful edits</span></div>"""


def values(v):
    g = v["guard"]
    negative = sum(h["reason"] == "edited-negative-image" for h in g["harmful_changes"])
    other = len(g["harmful_changes"]) - negative
    return len(g["gained"]["0.5"]), negative, other


def crop_panel(record):
    examples = record["cohorts"]["reserve"]["examples"]
    policy = read_json(POLICY)
    controls = policy["controls"]
    active = policy["selection"]["active"]
    true = [p["scores"] for p in examples if p["kind"] == "syringe"]
    lookalikes = [p["scores"] for p in examples if p["kind"] != "syringe"]
    yield '<h2>Independent candidate crops</h2><p>Best localized RF-DETR crop per scene and category.</p>'
    yield ('<div class="crop column-label"><span>Verification rule</span><span>Syringes kept</span>'
           '<span>Lookalikes blocked</span></div>')
    for label, field, floor in (
        (LABELS["released-veto"], "release-veto", controls["released-veto"]["cutoff"]),
        (LABELS["targeted-zero"], "targeted-text", controls["targeted-zero"]["cutoff"]),
        (LABELS["targeted-proposal"], "targeted-text", active["targeted-proposal"]["rule"]["cutoff"]),
        (LABELS["typed-proposal"], "typed-hybrid", active["typed-proposal"]["rule"]["cutoff"]),
    ):
        kept = sum(p[field] >= floor for p in true)
        blocked = sum(p[field] < floor for p in lookalikes)
        yield (f'<div class="crop"><span>{escape(label)}</span><span>{kept}/{len(true)}</span>'
               f'<span>{blocked}/{len(lookalikes)}</span></div>')


def render():
    record = read_json(RESULT)
    sections = [HEADER]
    for name, label in (("reserve","New public scenes"), ("exposed","Previously evaluated scenes")):
        cohort = record["cohorts"][name]
        measurements = {arm:values(item) for arm,item in cohort["variants"].items()}
        maximum = max(1, *(sum(v) for v in measurements.values()))
        base = cohort["baseline"]
        sections.append(f'<h2>{label}</h2><p>{base["images"]} photos, {base["objects"]} reference syringes</p>')
        for arm, counts in measurements.items():
            bars = "".join(f'<span class="{kind}" style="width:{100*n/maximum:.4f}%"></span>'
                           for kind,n in zip(("gain","negative","harm"),counts,strict=True))
            accessible = f"{counts[0]} new matches, {counts[1]} edits on negatives, {counts[2]} other harms"
            sections.append(f'<div class="row"><span>{escape(LABELS[arm])}</span>'
                            f'<div class="track" role="img" aria-label="{accessible}">{bars}</div>'
                            f'<span class="value" aria-hidden="true">{" / ".join(map(str, counts))}</span></div>')
    sections.extend(crop_panel(record))
    sections.append("</body></html>")
    (WORK / "comparison.html").write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    render()
