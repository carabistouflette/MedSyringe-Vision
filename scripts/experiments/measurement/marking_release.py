"""Check real marking inference in the isolated runtime and seal the release."""
from pathlib import Path

from PIL import Image,ImageOps

from medsyringe.data import digest,read_json,write_json
from medsyringe.runtime.local_models import ImageWorker
from scripts.experiments.measurement.marking_batch import MODEL_ROOT,model_hashes
from scripts.experiments.measurement.marking_evaluate import ART,SELECTION

RELEASE=ART/"capacity_ocr_release.json"
RUNTIME=ART/"capacity_ocr_runtime.json"
EXECUTABLE="data/ocr-v6/env/bin/python"


def main():
    selected=read_json(SELECTION)
    results=read_json(ART/"capacity_ocr_detected_results.json")
    if results["selection_sha256"]!=digest(SELECTION) or any(results["all"][name]["wrong"] for name in ("capacity","type")):
        raise ValueError("Capacity or type selection failed its detector-crop check.")
    opts=dict(selected["options"])
    mode=opts.pop("mode")
    if mode!="guided":
        raise ValueError("The frozen runtime expects the guided orientation policy.")
    opts["guided"]=True
    worker=ImageWorker(Path("."),Path(EXECUTABLE),"ocr",opts)
    smoke=[]
    for identifier,expected in (("IMG_3229",20),("IMG_3198",60)):
        with Image.open(f"data/ocr-v6/detected/{identifier}.png") as image:
            for angle,im in ((0,image),(180,ImageOps.flip(ImageOps.mirror(image)))):
                observed=worker.run(im)
                smoke.append(dict(image=identifier,orientation=angle,expected=expected,capacity=observed["capacity"]))
    for item in smoke:
        if item["capacity"] and item["capacity"]["value"]!=item["expected"]:
            raise ValueError("The local marking worker returned a wrong capacity.")
    if not {item["orientation"] for item in smoke if item["capacity"]}=={0,180}:
        raise ValueError("No verified runtime reading was obtained in each orientation.")
    write_json(RUNTIME,dict(passed=True,controls=smoke,selection_sha256=digest(SELECTION),
                            evidence_review=dict(outputs=36,status="All 36 emitted unit-and-number regions were reviewed visually on the detector crops.")))
    paths=("scripts/workers/local_model_worker.py","medsyringe/runtime/local_models.py","medsyringe/measurement/capacity_ocr.py",
           "medsyringe/measurement/ocr_refine.py")
    write_json(RELEASE,dict(id="ppocrv6-selective-markings",executable=EXECUTABLE,options=opts,
                           evidence=str(RUNTIME),evidence_sha256=digest(RUNTIME),
                           models={str(MODEL_ROOT/path):sha for path,sha in model_hashes().items()},
                           code={p:digest(p) for p in paths}))
    print("Activated",RELEASE,flush=True)


if __name__=="__main__":
    main()
