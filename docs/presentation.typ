#set page(width: 16in, height: 9in, margin: (x: 0.82in, y: 0.43in), fill: rgb("#f3f6f1"))
#set text(font: "DejaVu Sans", fill: rgb("#182e2a"), size: 20pt)
#set par(leading: 0.57em)

#let dark = rgb("#164e44")
#let mint = rgb("#dfefe5")
#let quiet = rgb("#48635e")
#let slide(title, body) = {
  text(size: 15pt, weight: "bold", fill: dark)[MEDSYRINGE VISION]
  v(8pt)
  text(size: 42pt, weight: "bold")[#title]
  v(9pt)
  line(length: 100%, stroke: 1pt + rgb("#adc4b7"))
  v(10pt)
  body
}
#let statistic(value, label) = block(width: 100%, fill: mint, radius: 12pt, inset: 13pt)[
  #text(size: 33pt, weight: "bold", fill: dark)[#value] #linebreak()
  #text(size: 18pt)[#label]
]

#v(0.62in)
#text(size: 17pt, weight: "bold", fill: dark)[MASTER'S PROJECT · COMPUTER VISION]
#v(15pt)
#text(size: 66pt, weight: "bold")[MedSyringe Vision]
#v(12pt)
#text(size: 28pt, fill: quiet)[Local detection, visual evidence, and auditable review]
#v(0.27in)
#grid(columns: (1fr, 1fr, 1fr), gutter: 13pt,
  statistic("74", "Unique source views in one session"),
  statistic("10 / 15", "External reference boxes recovered"),
  statistic("8 / 10", "Exact external scene counts"),
)
#v(0.28in)
#text(size: 16pt, fill: quiet)[November 2026 · Local CPU execution]
#pagebreak()

#slide("The source is one acquisition group", [
  #grid(columns: (1fr, 1fr), gutter: 30pt,
    [
      - 75 original HEIC files, 74 unique.
      - One camera over 18 minutes and repeated physical objects.
      - One visible syringe per image, with no verified negative or temporal sequence.
      - Draft visual scales: 58 × 20-scale and 16 × 60-scale.
    ],
    [
      #statistic("Development", "All 74 source photos stay together.")
      #v(8pt)
      #statistic("Validation", "19 Wikimedia Commons scenes.")
      #v(8pt)
      #statistic("Test", "10 official Open Images scenes, opened after freeze.")
    ],
  )
])
#pagebreak()

#slide("Local adaptation offered no measured gain", [
  #grid(columns: (2fr, 1fr), gutter: 23pt,
    image("../artifacts/figures/validation_models.png", width: 100%),
    [
      - Shared localization metric and images.
      - Grounding DINO Tiny and OWLv2 Base ran on CPU.
      - Trained re-ranking and augmentation used the local group only.
      - Validation selected Grounding DINO and froze a 0.539851 score cutoff.
    ],
  )
])
#pagebreak()

#slide("Crowded scenes drive the observed misses", [
  #grid(columns: (2.1fr, 1fr), gutter: 23pt,
    image("../artifacts/figures/external_test.png", width: 100%),
    [
      - Precision 10/10, recall 10/15, F1 0.800.
      - All five missed boxes came from two scenes with multiple syringes.
      - Geometrically matching proposals scored below the frozen cutoff.
      - Open Images sample size limits generalization.
    ],
  )
])
#pagebreak()

#slide("A field can remain unknown", [
  #grid(columns: (1.8fr, 1fr), gutter: 25pt,
    image("../artifacts/figures/attribute_diagnostics.png", width: 100%),
    [
      - Cropping improved observed scale coverage.
      - Correct color suggestions were equal in both modes.
      - OCR recovered the selected visible marking in five of nine development crops.
      - A visible number supplies review evidence. Apparent volume remains unknown.
    ],
  )
])
#pagebreak()

#slide("Demonstration and remaining experiments", [
  #grid(columns: (1fr, 1fr), gutter: 34pt,
    [
      #text(weight: "bold", fill: dark)[Demonstration]
      #v(7pt)
      + Open a HEIC photo and run local inference.
      + Inspect a box and its uncertain attributes.
      + Correct the observation and record the reviewer note.
      + Compare a reference count, then export the image and report.
    ],
    [
      #text(weight: "bold", fill: dark)[Next experiments]
      #v(7pt)
      - Independent capture sessions with new physical syringes.
      - Blinded annotation adjudication and timed reviewer comparisons.
      - Crowded scenes, validated unknown examples and tracked sequences.
      - Compare an efficient supervised detector after expanding the data.
    ],
  )
  #v(0.12in)
  #text(size: 16pt, fill: quiet)[Complete methods, run artifacts, and requirement-by-requirement status are included in the repository.]
])
