"""Produce a compact visual summary of the frozen fill validation."""
from html import escape
from pathlib import Path

from medsyringe.data import read_json


def main():
    root = Path("artifacts/fill")
    audit = read_json(root / "audit.json")["cohorts"]
    groups = [("Amber development", read_json(root / "development.json")["metrics"]),
              ("Orange 20 mL holdout", audit["holdout"]["metrics"]),
              ("Orange 60 mL holdout", audit["holdout-60"]["metrics"]),
              ("Uncolored control", audit["clear"]["metrics"])]
    rows = []
    for label, record in groups:
        answer, total = record["answered"], record["images"]
        rows.append(f"""<section><header>{escape(label)}<strong>{answer} / {total}</strong></header>
<div class="track"><span style="width:{answer/total*100:.4f}%"></span></div></section>""")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Fill measurement coverage</title>
<style>body{margin:0;color:var(--foreground);font-family:var(--font-sans,system-ui,sans-serif)}
h2{font-size:16px;margin:10px 0 4px}p{font-size:12px;line-height:1.35;color:var(--muted-foreground);margin:3px 0 11px}
section{margin-bottom:11px}header{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin-bottom:5px}
strong{font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.track{height:8px;background:var(--secondary);border-radius:8px;overflow:hidden}
span{display:block;height:100%;border-radius:8px;background:var(--chart-1)}
</style></head><body><h2>Apparent fill validation</h2><p>Photos with a reported estimate.</p>"""
    page += "".join(rows)
    page += "<p>All reported holdout estimates fell within visually read brackets. Reference volumes were read from photos.</p></body></html>"
    Path("artifacts/figures/fill_coverage.html").write_text(page)


if __name__ == "__main__":
    main()
