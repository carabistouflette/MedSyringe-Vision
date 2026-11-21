"""Read actual printed scale marks from stroke-guided original-resolution patches."""
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np

from .capacity_ocr import ONLY_NUMBER, PAIR, UNIT, bounded_image, records, resolve, rotated_image

SCALES = {20: 5, 60: 10}


def anchors_in_view(tokens, inverse, raw_shape):
    """Map crop-normalized OCR evidence back into one high-resolution OCR view."""
    h, w = raw_shape[:2]
    matrix = cv2.invertAffineTransform(inverse)
    points = []
    for token in tokens:
        if not ONLY_NUMBER.fullmatch(token["text"]) or token["score"] < .55:
            continue
        l, t, r, b = np.asarray(token["crop_bbox"])*[w, h, w, h]
        patch = np.array([[l, t], [l, b], [r, t], [r, b]]) @ matrix[:, :2].T + matrix[:, 2]
        box = [*patch.min(axis=0), *patch.max(axis=0)]
        points.append((float(token["text"]), token["score"], np.mean(patch, axis=0), box))
    return points


def layout(points, maximum, step):
    valid = [p for p in points if p[0] in range(step, maximum+1, step)]
    options = []
    for low, high in combinations(sorted(valid, key=lambda p: p[0]), 2):
        if high[0] == low[0]:
            continue
        delta = high[2]-low[2]
        extent = (high[0]-low[0])/step
        pitch = delta[1]/extent
        height = np.median([p[3][3]-p[3][1] for p in (low, high)])
        if not max(10, 1.05*height) <= pitch <= 10*height or abs(delta[0]) > .64*delta[1]:
            continue
        slope = delta/extent
        zero = low[2]-slope*low[0]/step
        matches = {}
        for p in sorted(valid, key=lambda p: -p[1]):
            err = p[2]-zero-slope*p[0]/step
            if abs(err[0]) < .60*pitch and abs(err[1]) < .25*pitch:
                matches.setdefault(p[0], p)
        if len(matches) >= 2 and maximum-max(matches) <= 2*step:
            options.append(((len(matches), extent, low[1]+high[1]), (zero, slope, matches)))
    return max(options, key=lambda value: value[0], default=None)


def stroke_at(gray, x, y, width, height, pitch):
    """Find a dark nearly horizontal stroke in the lanes adjacent to a number."""
    h, w = gray.shape
    dy = int(max(3, min(.24*pitch, .66*height)))
    top, bottom = max(0, round(y)-dy), min(h, round(y)+dy+1)
    if bottom-top < 5:
        return 0., y
    matches = [(0., y)]
    for sign in (-1, 1):
        left, right = sorted([round(x+sign*.43*width), round(x+sign*1.30*width)])
        left, right = max(0, left), min(w, right)
        if right-left < 5:
            continue
        padding = max(2, round(height*.18))
        first, last = max(0, top-padding), min(h, bottom+padding)
        region = gray[first:last, left:right]
        kernel = np.ones((2*padding+1, 1), np.uint8)
        dark = cv2.morphologyEx(region, cv2.MORPH_BLACKHAT, kernel)
        row = np.mean(np.minimum(dark, 45.), axis=1)[top-first:bottom-first]
        peak = int(row.argmax())
        matches.append((float(row[peak]), top+peak))
    return max(matches)


def hypotheses(image, inscriptions):
    bgr = bounded_image(image, edge=1900)
    views = defaultdict(list)
    for token in inscriptions["candidates"]:
        views[token["rotation"], token["family"]].append(token)
    options = []
    cache = {}
    for (angle, family), tokens in views.items():
        if angle not in cache:
            cache[angle] = rotated_image(bgr, angle)
        pixels, inverse = cache[angle]
        points = anchors_in_view(tokens, inverse, bgr.shape)
        for maximum, step in SCALES.items():
            match = layout(points, maximum, step)
            if match:
                options.append((match[0], maximum, step, angle, pixels, inverse, match[1]))
    seen = set()
    for _, maximum, step, angle, pixels, inverse, match in sorted(options, key=lambda o: o[0], reverse=True):
        if (maximum, angle) in seen:
            continue
        seen.add((maximum, angle))
        yield maximum, step, angle, pixels, inverse, bgr.shape, match
        if len(seen) == 3:
            break


def recognition_patches(image, inscriptions):
    for maximum, step, angle, pixels, inverse, raw_shape, (zero, slope, anchors) in hypotheses(image, inscriptions):
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        typical = np.median([p[3] for p in anchors.values()], axis=0)
        width, height = typical[2:]-typical[:2]
        if not width or not height:
            continue
        patch_sets = []
        family = f"stroke-{maximum}-{angle}"
        for value in range(max(step, int(min(anchors))-step), maximum+1, step):
            x, y = zero+slope*value/step
            strength, row = stroke_at(gray, x, y, width, height, slope[1])
            if value not in anchors:
                patch_sets.extend([(value, x, y, strength, "anchor"), (value, x, row, strength, "stroke")])
            if value == maximum:
                # Units are physically printed above the terminal number on these
                # layouts. This does not supply any label to the recognizer.
                patch_sets += [("unit", x, y-height*.92, strength, "unit"),
                               ("pair", x, y-height*.47, strength, "pair")]
        patches, metadata = [], []
        for expected, x, y, strength, mode in patch_sets:
            if mode == "stroke" and strength < 2.0:
                continue
            half_width = max(width*.76, height*.64)
            half_height = height*(1.15 if mode == "pair" else .55)
            l, t = max(0, round(x-half_width)), max(0, round(y-half_height))
            r, b = min(w, round(x+half_width)), min(h, round(y+half_height))
            if r-l < 8 or b-t < 8:
                continue
            patch = pixels[t:b, l:r]
            zoom = min(3., max(1., 55/patch.shape[0]))
            patches.append(cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC))
            metadata.append((expected, strength, mode, (l,t,r,b)))
        if not patches:
            continue
        keep = []
        for value, score, _, box in anchors.values():
            l,t,r,b = box
            result = dict(rec_texts=[str(int(value))], rec_scores=[score], rec_polys=[[(l,t),(r,t),(r,b),(l,b)]])
            keep.extend(records(result, inverse, raw_shape, angle, family))
        yield maximum, angle, family, inverse, raw_shape, keep, patches, metadata


def collect(image, inscriptions, recognizers):
    output = []
    for maximum, angle, family, inverse, raw_shape, anchors, patches, metadata in recognition_patches(image, inscriptions):
        reads = []
        for name, recognizer in recognizers.items():
            answers = recognizer.predict_iter(patches, batch_size=6)
            for answer, (expected, strength, mode, (l,t,r,b)) in zip(answers, metadata, strict=True):
                text = answer["rec_text"].strip()
                score = float(answer["rec_score"])
                if expected == "unit":
                    valid = bool(UNIT.fullmatch(text))
                elif expected == "pair":
                    match = PAIR.fullmatch(text)
                    valid = bool(match and (match[1] or match[2]) == str(maximum))
                else:
                    valid = text == str(expected)
                if valid:
                    result = dict(rec_texts=[text], rec_scores=[score], rec_polys=[[(l,t),(r,t),(r,b),(l,b)]])
                    tokens = records(result, inverse, raw_shape, angle, family)
                    reads.extend(dict(**token, reader=name, stroke_strength=strength, window=mode) for token in tokens)
        output.append(dict(expected_template=maximum, anchors=anchors, reads=reads))
    return output


def supplement_graduations(inscriptions, hypotheses, policy):
    """Keep the published reading and add only directly seen, compatible markings."""
    current = inscriptions.get("capacity")
    original_scale = inscriptions.get("scale")
    proposals = []
    for guess in hypotheses:
        maximum = guess["expected_template"]
        if current and current["value"] != maximum:
            continue
        if original_scale and original_scale["value"] != f"{maximum}-scale":
            continue
        reads = [t for t in guess["reads"] if t["reader"] in policy["readers"] and t["score"] >= policy["confidence"]
                 and (t["window"] in ("unit", "pair") or t["stroke_strength"] >= policy["stroke"])]
        if not reads:
            continue
        tokens = [*guess["anchors"], *reads]
        new = resolve(tokens, cutoff=policy["confidence"], scale_cutoff=.77)
        candidate = new.get("capacity")
        if not current and (not candidate or candidate["value"] != maximum or not new["scale"]):
            continue
        proposals.append((candidate, new["scale"], tokens, reads))
    if not proposals or len({p[0]["value"] for p in proposals if p[0]}) > 1:
        return inscriptions
    capacity, scale, tokens, reads = max(proposals, key=lambda p: (bool(p[0]), len(p[3])))
    return {**inscriptions, "capacity": current or capacity, "scale": original_scale or scale,
            "status": "capacity read" if current or capacity else inscriptions["status"],
            "candidates": [*inscriptions["candidates"], *tokens],
            "graduation_crop_evidence": reads}
