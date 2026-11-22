"""View-consistent, class-agnostic syringe proposal fusion."""
from PIL import Image, ImageOps

from medsyringe.detection.detectors import box_iou, suppress

VIEWS = ("original", "quarter_turn", "mirror", "opposite_turn")


def view_image(image, view):
    if view == "quarter_turn":
        return image.transpose(Image.Transpose.ROTATE_90)
    if view == "opposite_turn":
        return image.transpose(Image.Transpose.ROTATE_270)
    if view == "mirror":
        return ImageOps.mirror(image)
    if view == "original":
        return image
    raise ValueError(f"Unknown detection view: {view}")


def restore_box(box, view):
    l, t, r, b = box
    if view == "quarter_turn":
        return [1-b, l, 1-t, r]
    if view == "opposite_turn":
        return [t, 1-r, b, 1-l]
    if view == "mirror":
        return [1-r, t, 1-l, b]
    if view == "original":
        return box
    raise ValueError(f"Unknown detection view: {view}")


def view_proposals(detector, image, view):
    return [{**p, "bbox": restore_box(p["bbox"], view)}
            for p in detector.propose(view_image(image, view))]


def fuse(proposals, *, strategy, overlap):
    if strategy == "max":
        return suppress([p for values in proposals.values() for p in values], overlap)
    # A view contributes at most one vote per box. Match against the highest
    # scoring member so a sequence of overlapping boxes cannot bridge objects.
    pending = sorted(
        ((view, p) for view, values in proposals.items() for p in suppress(values, overlap)),
        key=lambda item: -item[1]["score"],
    )
    fused = []
    while pending:
        first_view, anchor = pending[0]
        votes = {first_view: anchor}
        remaining = []
        for view, p in pending[1:]:
            if view not in votes and box_iou(anchor["bbox"], p["bbox"]) >= 0.55:
                votes[view] = p
            else:
                remaining.append((view, p))
        pending = remaining
        weight = sum(p["score"] for p in votes.values())
        if weight <= 0:
            continue
        bbox = [sum(p["bbox"][i] * p["score"] for p in votes.values()) / weight
                for i in range(4)]
        strongest = anchor["score"]
        # Reward independent geometric support, keeping the strongest isolated
        # proposal available for partially visible or unusual objects.
        other = sum(p["score"] for v, p in votes.items() if v != first_view)
        score = strongest + (1-strongest) * other / max(1, len(proposals)-1)
        fused.append(dict(bbox=bbox, score=score, support=len(votes), view_count=len(proposals)))
    return suppress(fused, overlap)
