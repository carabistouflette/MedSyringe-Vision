"""Build a local shareable chart from the frozen Fil rouge follow-up."""
from html import escape
from pathlib import Path

from medsyringe.data import read_json

ART=Path("artifacts")
OUTPUT=ART/"figures/fil_rouge_followup.html"


def main():
    detect=read_json(ART/"sam3_audit_fil_rouge.json")
    text=read_json(ART/"capacity_ocr_detected_results.json")["all"]
    liquid=read_json(ART/"liquid_appearance_validation.json")["all"]
    total=text["n"]
    measured=(
        ("Previous localization",detect["baseline"]["tp"],"var(--chart-2)"),
        ("SAM 3 hybrid localization",detect["candidate"]["tp"],"var(--chart-1)"),
        ("Nominal capacity read",text["capacity"]["correct"],"var(--chart-3)"),
        ("Scale type read",text["type"]["correct"],"var(--chart-3)"),
        ("Liquid lightness measured",liquid["centerline_supported"],"var(--chart-4)"),
    )
    rows=[]
    for title,count,color in measured:
        rows.append(f"""<div class="row"><span>{escape(title)}</span><span class="count">{count} / {total}</span>
<div class="track"><div class="mark" style="width:{100*count/total:.3f}%;background:{color}"></div></div></div>""")
    html="""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fil rouge validation</title>
<style>
body{margin:0;color:var(--foreground);font-family:var(--font-sans,system-ui,sans-serif)}
h2{font-size:16px;margin:10px 0 3px;font-weight:650}p{font-size:12px;margin:0 0 11px;color:var(--muted-foreground)}
.row{display:grid;grid-template-columns:1fr auto;gap:4px 10px;margin-bottom:10px;align-items:baseline;font-size:13px}
.count{font-size:12px;font-variant-numeric:tabular-nums}.track{grid-column:1/-1;height:7px;border-radius:8px;background:var(--secondary);overflow:hidden}
.mark{height:100%;border-radius:8px}small{font-size:11.5px;color:var(--muted-foreground)}
</style></head><body>
<h2>Fil rouge validation</h2>
<p>74 unique photographs from one acquisition. Measurements can abstain.</p>
"""+"".join(rows)+"""<small>Detection uses IoU ≥ 0.50. Lightness uses a visually supported liquid region.</small>
</body></html>
"""
    OUTPUT.write_text(html)


if __name__=="__main__":
    main()
