"""Skip crop embeddings for pending boxes already covered by retained objects."""
import numpy as np

from medsyringe.detection.additive import overlap_fraction
from medsyringe.detection.detectors import box_iou
from scripts.experiments.detector_audits.detection_crop_semantics import PADDINGS, proposals, scores
from scripts.experiments.detector_audits.detection_semantics_audit import crop_frame


def covered(box, retained):
    return any(box_iou(box,p["bbox"])>=.42 or overlap_fraction(box,p["bbox"])>=.80 for p in retained)


def indexed(frame, indexes):
    count = len(frame["candidates"])
    return {"candidates":[frame["candidates"][i] for i in indexes],
            "values":{name:[values[pad*count+i] for pad in range(len(PADDINGS)) for i in indexes]
                      for name,values in frame["values"].items()}}


def select_needed(original, accepted_features, pending, policy):
    initial, *rest = policy
    if any(part["bounds"][1]>-1e4 for part in rest):
        return range(len(pending))
    floor = original["floor"]
    threshold = min(part["bounds"][2] for part in policy)
    margins = scores(accepted_features,initial["name"],initial["head"])
    retained = [p for p,margin in zip(accepted_features["candidates"],margins,strict=True)
                if p["score"]>=floor and margin>=initial["bounds"][1]]
    return [i for i,p in enumerate(pending) if p["score"]>=threshold and not covered(p["bbox"],retained)]


def features(encoder,image,original,policy):
    all_candidates = proposals(original)
    n_accepted = sum(p["score"]>=original["floor"] for p in original["ranked"])
    if not n_accepted or n_accepted==len(all_candidates) or not policy:
        return crop_frame(encoder,image,original)
    accepted,pending = all_candidates[:n_accepted],all_candidates[n_accepted:]
    head = crop_frame(encoder,image,{"floor":0.,"ranked":accepted})
    chosen = select_needed(original,head,pending,policy)
    if not chosen:
        return head
    tail = crop_frame(encoder,image,{"floor":0.,"ranked":[pending[i] for i in chosen]})
    result = {}
    for name,left in head["values"].items():
        right = tail["values"][name]
        a = np.asarray(left).reshape(len(PADDINGS),len(accepted),-1)
        b = np.asarray(right).reshape(len(PADDINGS),len(chosen),-1)
        result[name] = np.concatenate((a,b),axis=1).reshape(-1,a.shape[2]).tolist()
    return dict(candidates=head["candidates"]+tail["candidates"],values=result)
