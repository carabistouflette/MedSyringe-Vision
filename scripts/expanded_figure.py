"""Render the recorded detector and OCR comparisons as a shareable HTML figure."""
from html import escape
from pathlib import Path

from medsyringe.data import read_json
from scripts.expanded_dataset import ART


def main():
    trials = read_json(ART / "detector_recall_freeze.json")
    calibration = read_json(ART / "detector_trials.json")["baseline"]
    mixed = trials["selected"]["outcome"]["calibration"]["result"]
    test = read_json(ART / "detector_guarded_test.json")
    public = read_json(ART / "detector_guarded_reserve.json")
    marks = read_json(ART / "markings_calibration.json")["totals"]

    def pair(label, first, second, unit):
        first_label = f"{first*100:.1f}%" if unit == "f1" else f"{round(first*68)}/68"
        second_label = f"{second*100:.1f}%" if unit == "f1" else f"{round(second*68)}/68"
        return f"""<section class="metric"><h3>{escape(label)}</h3>
<div class="line"><div class="track"><span style="width:{100*first:.2f}%;background:var(--chart-2)"></span></div><b>{first_label}</b></div>
<div class="line"><div class="track"><span style="width:{100*second:.2f}%;background:var(--chart-1)"></span></div><b>{second_label}</b></div></section>"""

    content = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Expanded Fil rouge results</title>
<style>
body{margin:0;color:var(--foreground);font-family:var(--font-sans,system-ui,sans-serif)}
h2{margin:9px 0 6px;font-size:16px}h3{margin:0 0 5px;font-size:13px;font-weight:600}
p{margin:4px 0 10px;font-size:12px;color:var(--muted-foreground);line-height:1.35}
.metric{margin-bottom:10px}.line{display:grid;grid-template-columns:1fr 56px;gap:9px;align-items:center;min-height:17px}
.track{height:7px;background:var(--secondary);border-radius:7px;overflow:hidden}.track span{display:block;height:100%;border-radius:7px}
b{text-align:right;font-size:11px;font-variant-numeric:tabular-nums}.legend{display:flex;flex-wrap:wrap;gap:8px 14px}
.dot{display:inline-block;width:8px;height:8px;border-radius:8px;margin-right:5px}
</style></head><body><h2>Detection F1</h2>
<p class="legend"><span><i class="dot" style="background:var(--chart-2)"></i>Installed detector</span>
<span><i class="dot" style="background:var(--chart-1)"></i>Hybrid candidate</span></p>
"""
    content += pair("Mixed calibration · 87 photos", calibration["f1"], mixed["f1"], "f1")
    content += pair("Held-out clear block · 14 photos", test["baseline"]["f1"], test["candidate"]["f1"], "f1")
    content += pair("Public regression · 21 photos", public["baseline"]["f1"], public["candidate"]["f1"], "f1")
    content += """<p>The hybrid lost one previously matched public syringe and was held back.</p>
<h2>Nominal-capacity readings</h2><p class="legend"><span><i class="dot" style="background:var(--chart-2)"></i>Previous reader</span>
<span><i class="dot" style="background:var(--chart-1)"></i>Two corroborating views</span></p>"""
    content += pair("Correct calibration readings · reference crops", marks["baseline"]["capacity"]["correct"]/68,
                    marks["both"]["capacity"]["correct"]/68, "count")
    content += "<p>Both configurations emitted zero wrong capacities on calibration.</p></body></html>"
    output = Path("artifacts/figures/expanded_fil_rouge.html")
    output.write_text(content)


if __name__ == "__main__":
    main()
