"""Read local image requests and emit one JSON result per line."""
import argparse
import json
import sys
import traceback

from medsyringe.local_models import decode_picture


def load_runner(mode,options):
    if mode=="sam3":
        from itertools import chain

        from medsyringe.additive import overlapping_tiles
        from medsyringe.detectors import suppress
        from scripts.sam3_study import Segmenter,configure_cpu,crop_predictions

        configure_cpu()
        model=Segmenter(options["edge"],options["device"])
        def predict(image,operation):
            if operation=="liquid":
                from scripts.liquid_measure import measure_crop

                return measure_crop(model,image)
            windows=chain([(0,0,*image.size)],overlapping_tiles(image.size,.66))
            return suppress(list(crop_predictions(model.predict,image,windows)),.45)
        return predict
    from medsyringe.capacity_ocr import SceneOCR,resolve
    from medsyringe.ocr_refine import guided_read

    reader=SceneOCR()
    def predict(image,operation):
        tokens=reader.read(image)
        if options["guided"]:
            tokens.extend(guided_read(reader,image,tokens))
        return resolve(tokens,cutoff=options["cutoff"],repeats=options["repeats"],
                       scale_cutoff=options["scale_cutoff"])
    return predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("sam3","ocr"))
    args=parser.parse_args()
    channel=sys.stdout
    sys.stdout=sys.stderr
    runner=None
    for line in sys.stdin:
        try:
            payload=json.loads(line)
            if runner is None:
                runner=load_runner(args.mode,payload["options"])
            response=dict(result=runner(decode_picture(payload["image"]),payload.get("operation")))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response=dict(error=f"{type(error).__name__}: the local {args.mode} runner failed.")
        channel.write(json.dumps(response)+"\n")
        channel.flush()


if __name__=="__main__":
    main()
