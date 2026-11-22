"""Recover explicitly printed small capacities from separate graduation crops."""
from itertools import combinations

import cv2
import numpy as np
from PIL import Image

from .capacity_ocr import MODEL_ROOT, RECOGNIZER, UNIT, barrel_angle, bounded_image, records, rotated_image, view_angles
from .small_fill import inspect_barrel


def major_strokes(view):
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    vertical = np.ones((max(7, round(height * .015)), 1), np.uint8)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vertical)
    long = cv2.morphologyEx((dark > 21).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((1, max(6, round(.085 * width))), np.uint8))
    components = cv2.connectedComponentsWithStats(long)[2][1:]
    return [(x, y + h / 2, x + w) for x, y, w, h, _ in components
            if .14 * width <= w <= .55 * width and w >= .12 * width + 1.5 * h and h <= .019 * height]


def tick_tracks(strokes, height, width):
    tracks = {}
    for first, second in combinations(strokes, 2):
        pitch = second[1] - first[1]
        if not .045 * height <= pitch <= .25 * height:
            continue
        tolerance = max(3., .10 * pitch)
        aligned = [s for s in strokes
                   if abs(first[0] - s[0]) <= .04 * width and abs(first[2] - s[2]) <= .04 * width]
        slots = {}
        for s in aligned:
            index = round((s[1] - first[1]) / pitch)
            residual = abs(s[1] - first[1] - index * pitch)
            if residual <= tolerance and residual < slots.get(index, (None, float("inf")))[1]:
                slots[index] = s, residual
        sequence = [slots[0][0]]
        for direction in (-1, 1):
            step = direction
            while step in slots:
                sequence.append(slots[step][0])
                step += direction
        if 3 <= len(sequence) <= 11:
            line = tuple(sorted(sequence, key=lambda s: s[1]))
            tracks[line] = float(np.median(np.diff([s[1] for s in line])))
    return sorted(tracks.items(), key=lambda item: (-len(item[0]), item[0][0][1]))[:3]


def regions(track, pitch, sign, dimensions):
    h, w = dimensions
    terminal = track[-1]
    stroke_width = float(np.median([right - left for left, _, right in track]))
    boxes = []
    for stroke, delta, y_span in [*((s, 0., .28) for s in track),
                                   (terminal, .36, .22), (terminal, .58, .24)]:
        left, y, right = stroke
        edge = right if sign == 1 else left
        offsets = sorted((edge - sign * .035 * stroke_width, edge + sign * .38 * stroke_width))
        x0, x1 = max(0, round(offsets[0])), min(w, round(offsets[1]))
        y0, y1 = max(0, round(y + (delta - y_span) * pitch)), min(h, round(y + (delta + y_span) * pitch))
        if x1 - x0 < 5 or y1 - y0 < 7:
            return []
        boxes.append((x0, y0, x1, y1))
    left, y, right = terminal
    edge = right if sign == 1 else left
    for inside, outside, start, stop in ((.095, .22, .21, .59), (.14, .29, .17, .64),
                                          (.065, .255, .24, .58)):
        offsets = sorted((edge - sign * inside * stroke_width, edge + sign * outside * stroke_width))
        x0, y0 = max(0, round(offsets[0])), max(0, round(y + start * pitch))
        x1, y1 = min(w, round(offsets[1])), min(h, round(y + stop * pitch))
        if x1 - x0 < 5 or y1 - y0 < 7:
            return []
        boxes.append((x0, y0, x1, y1))
    return boxes


def accept_markings(digits, units):
    known = [(i, int(text), score) for i, (text, score) in enumerate(digits)
             if text in {str(n) for n in range(1, 11)} and score >= .72]
    if len(known) < 3 or known[-1][0] != len(digits) - 1:
        return None
    if len({number - index for index, number, _ in known}) != 1:
        return None
    explicit = [(index, score) for index, (text, score) in enumerate(units)
                if UNIT.fullmatch(text) and score >= .71]
    if not explicit:
        return None
    unit, score = max(explicit, key=lambda item: item[1])
    return known[-1][1], min(score, *(p[2] for p in known)), unit


def unit_contrast(view, box):
    l, t, r, b = box
    patch = cv2.resize(view[t:b, l:r], None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = cv2.createCLAHE(clipLimit=3., tileGridSize=(3, 3)).apply(patch)
    patch = cv2.copyMakeBorder(patch, 10, 10, 18, 18, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)


def scale_views(bgr):
    views = view_angles(bgr)
    if len(views) == 4:
        roll = barrel_angle(bgr)
        if .35 <= abs(roll) < 3:
            views.extend((((angle + roll) % 360, "fine aligned") for angle in (0., 180.)))
    return views


class SmallCapacity:
    def __init__(self, model_root=MODEL_ROOT):
        from paddleocr import TextRecognition

        self.reader = TextRecognition(model_name=RECOGNIZER, model_dir=str(model_root / RECOGNIZER),
                                      device="cpu", cpu_threads=2, enable_mkldnn=False)

    def read(self, image):
        bgr = bounded_image(image, edge=1500)
        findings, diagnostic = self.scan(bgr, "native")
        if not findings and max(bgr.shape[:2]) < 750:
            enlarged = bounded_image(image.resize((image.width * 2, image.height * 2),
                                                   Image.Resampling.BICUBIC), edge=1500)
            findings, extra = self.scan(enlarged, "2x")
            diagnostic.extend(extra)
        values = {r["value"] for r in findings}
        return dict(capacity=max(findings, key=lambda p: p["score"]) if len(values) == 1 else None,
                    views=diagnostic)

    def scan(self, bgr, sampling):
        findings, diagnostic = [], []
        for angle, family in scale_views(bgr):
            view, inverse = rotated_image(bgr, angle)
            h, w = view.shape[:2]
            if h < 2 * w:
                continue
            for track, pitch in tick_tracks(major_strokes(view), h, w):
                for sign in (-1, 1):
                    boxes = regions(track, pitch, sign, (h, w))
                    if not boxes:
                        continue
                    patches = [cv2.resize(view[t:b, l:r], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                               for l, t, r, b in boxes[:-3]]
                    patches.extend(unit_contrast(view, box) for box in boxes[-3:])
                    reads = [(p["rec_text"].strip(), float(p["rec_score"]))
                             for p in self.reader.predict_iter(patches, batch_size=8)]
                    digits, units = reads[:len(track)], reads[len(track):]
                    diagnostic.append(dict(angle=angle, view=family, side=sign, sampling=sampling,
                                           digits=digits, units=units, stroke_count=len(track)))
                    match = accept_markings(digits, units)
                    if match:
                        capacity, certainty, unit = match
                        polys = [[(l, t), (r, t), (r, b), (l, b)] for l, t, r, b in boxes]
                        evidence = records(dict(rec_texts=[a for a, _ in reads], rec_scores=[z for _, z in reads],
                                                rec_polys=polys), inverse, bgr.shape, angle, "small separated scale")
                        findings.append(dict(value=capacity, unit="mL", score=certainty, sampling=sampling,
                                             evidence=evidence, unit_patch=unit,
                                             barrel=inspect_barrel(view, track, pitch, sign, capacity),
                                             method="Repeated graduation strokes, separate OCR digits and printed unit"))
        return findings, diagnostic
