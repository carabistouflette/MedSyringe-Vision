"""Require evidence across contrast views for an uncorroborated capacity."""


def confirm_untyped_capacity(result, corroboration):
    capacity = result.get("capacity")
    if capacity is None or result.get("scale"):
        return result
    other = corroboration.get("capacity")
    if other and (other["value"], other["unit"]) == (capacity["value"], capacity["unit"]):
        return result
    return {**result, "capacity": None, "status": "uncorroborated capacity",
            "capacity_reason": "A capacity without a supported scale needs the same reading in two contrast views."}
