"""Run the frozen crop detector with its audited CPU model and tokenizer versions."""
import json
import sys
import traceback

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_json
from medsyringe.local_models import decode_picture
from scripts.appearance_text_encoder import TextMarginEncoder, detector_encoder, unpack_patches
from scripts.detection_crop_semantics import VisualSemantics, proposals
from scripts.detection_semantics_audit import apply_rules, crop_frame, encode_frame
from scripts.sam3_study import configure_cpu


def explain(original, updated):
    floor = original["floor"]
    boxes = {tuple(p["bbox"]) for p in updated if p["score"]>=floor}
    demoted = [{**p,"score":floor*.01,"original_score":p["score"],
                "review_reason":"Independent crop evidence was too weak for automatic acceptance"}
               for p in original["ranked"] if p["score"]>=floor and tuple(p["bbox"]) not in boxes]
    accepted = []
    pending = []
    for p in updated:
        if "semantic_margin" in p:
            accepted.append({**p,"source":"crop appearance","verification_score":p["semantic_margin"]})
        elif p["score"]>=floor:
            accepted.append(p)
        else:
            pending.append(p)
    return [*accepted,*demoted,*pending]


def main():
    pipe = sys.stdout
    sys.stdout = sys.stderr
    configure_cpu()
    release = read_json(RELEASE)
    seal = digest(RELEASE)
    encoder = detector_encoder(release) if release.get("encoder") == "text-only" else VisualSemantics()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            record = request["options"]
            if record["release_sha256"]!=seal:
                raise ValueError("The application and local worker use different policies.")
            image = decode_picture(request["image"])
            fitting = record
            if release.get("mode")=="veto-only":
                fitting = {**record,"ranked":[p for p in record["ranked"] if p["score"]>=record["floor"]]}
            if request.get("operation") == "appearance crops":
                if not isinstance(encoder, TextMarginEncoder) or release.get("mode") != "veto-only":
                    raise ValueError("Prepared crops require the text-only appearance verifier.")
                candidates = proposals(fitting)
                regions = encode_frame(encoder, candidates, unpack_patches(image, candidates))
            else:
                regions = crop_frame(encoder, image, fitting)
            candidates,_ = apply_rules(record,regions,release["components"])
            response = {"result":explain(record,candidates)}
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = {"error":"The local crop detector did not produce a verified result."}
        pipe.write(json.dumps(response)+"\n")
        pipe.flush()


if __name__ == "__main__":
    main()
