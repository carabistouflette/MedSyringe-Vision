"""Render the frozen verifier comparison as a self-contained HTML figure."""
from html import escape

from medsyringe.data import read_json
from scripts.syringe_object_references import WORK
from scripts.syringe_object_verifier import RESULT

LABELS = {
    "off": "Recovery only",
    "release-veto": "Issued SigLIP 2 veto",
    "targeted-text": "Targeted SigLIP 2 prompts",
    "binary": "Binary DINOv2",
    "typed": "Typed DINOv2",
    "typed-hybrid": "Typed DINOv2 and SigLIP 2",
}
HEADER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Syringe recovery verifier comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { color:var(--foreground); font: 14px/1.4 var(--font-sans); margin:0 }
h2 { font-size:16px; margin:12px 0 2px }
h3 { font-size:14px; margin:10px 0 1px }
p { margin:0 0 8px; color:var(--muted-foreground) }
.key { display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 }
.key span::before { content:""; display:inline-block; width:.75em; height:.75em;
margin-right:.35em; border-radius:2px; background:var(--bar-color) }
.tp { --bar-color:var(--chart-1,#2563eb) }
.fp { --bar-color:var(--chart-2,#ea580c) }
.fn { --bar-color:var(--chart-3,#7c3aed) }
.row { display:grid; grid-template-columns:minmax(132px,1.3fr) minmax(90px,1fr) 74px;
align-items:center; gap:8px; padding:2px 0; min-height:28px }
.track { display:flex; align-items:center; height:10px }
.bar { height:100%; background:var(--bar-color); border-radius:1px }
.numbers { font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap }
</style></head><body>
<div class="key"><span class="tp">True positive</span><span class="fp">False positive</span>
<span class="fn">Missed</span></div>"""


def chart(entry, reference):
    metrics = {"Released detector": entry["baseline"]}
    metrics.update({LABELS[k]: entry["routes"][reference][k]["metrics"] for k in LABELS})
    total = max(sum(m[k] for k in ("tp", "fp", "fn")) for m in metrics.values())
    for label, m in metrics.items():
        values = [m[k] for k in ("tp", "fp", "fn")]
        bars = "".join(f'<div class="bar {kind}" style="width:{100 * n / max(total, 1):.4f}%"></div>'
                       for kind, n in zip(("tp", "fp", "fn"), values, strict=True))
        accessible = f"{values[0]} true positives, {values[1]} false positives, {values[2]} missed"
        yield (
            f'<div class="row"><span>{escape(label)}</span>'
            f'<div class="track" role="img" aria-label="{accessible}">{bars}</div>'
            f'<span class="numbers" aria-hidden="true">{" / ".join(map(str, values))}</span></div>'
        )


def render():
    result = read_json(RESULT)
    parts = [HEADER]
    for role, label in (("fresh", "New public scenes"), ("exposed", "Previously evaluated scenes")):
        entry = result["cohorts"][role]
        base = entry["baseline"]
        parts += [f'<h2>{label}</h2>', f'<p>{base["images"]} photos, {base["objects"]} reference syringes</p>']
        for key, title in (("calibrated", "Calibrated recovery"), ("active", "Active recovery probe")):
            parts += [f"<h3>{title}</h3>", *chart(entry, key)]
    parts.append("</body></html>")
    (WORK / "comparison.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    render()
