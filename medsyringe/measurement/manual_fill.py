"""Calculate reviewed apparent fill from three visible image landmarks."""
from math import hypot, isfinite


def estimate(landmarks, box, size, scale):
    """Project the piston onto the line defined by two numbered graduations."""
    capacity = landmarks["capacity_ml"]
    low, high, piston = landmarks["low"], landmarks["high"], landmarks["piston"]
    left, top, right, bottom = box
    points = (low["point"], high["point"], piston)
    if not all(left <= x <= right and top <= y <= bottom for x, y in points):
        raise ValueError("Place all three fill landmarks inside the current syringe box.")
    values = (low["ml"], high["ml"], capacity)
    if not all(isfinite(v) for v in values) or not 0 <= low["ml"] < high["ml"] <= capacity:
        raise ValueError("Enter two increasing graduations between zero and the nominal capacity.")
    if scale is not None and float(scale.split("-")[0]) != capacity:
        raise ValueError("The nominal capacity must agree with the reviewed visual scale.")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("The source image dimensions are unavailable.")
    dx = (high["point"][0]-low["point"][0])*width
    dy = (high["point"][1]-low["point"][1])*height
    px = (piston[0]-low["point"][0])*width
    py = (piston[1]-low["point"][1])*height
    length = hypot(dx, dy)
    box_width, box_height = (right-left)*width, (bottom-top)*height
    if length < max(6., .075*max(box_width, box_height)):
        raise ValueError("Choose two visibly separated graduations on the same side of the barrel.")
    if abs(dx*py-dy*px)/length > .75*min(box_width, box_height):
        raise ValueError("Place the piston point close to the graduation line.")
    fraction = (dx*px+dy*py)/(length*length)
    if not 0. <= fraction <= 1.:
        raise ValueError("Choose graduations on either side of the piston edge.")
    value = low["ml"] + fraction*(high["ml"]-low["ml"])
    return dict(status="manual estimate", method="Reviewed graduation and piston landmarks",
                source="human", capacity_ml=capacity, ml=round(value, 1),
                percent=round(100*value/capacity), landmarks=landmarks,
                coordinates="normalized source image")
