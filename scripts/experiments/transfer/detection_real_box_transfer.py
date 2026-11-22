"""Train matched RF-DETR box detectors on reviewed real sources and simulator controls."""
import argparse
from collections import Counter
from hashlib import md5, file_digest
from pathlib import Path
from random import Random
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

from medsyringe.data import digest, leakage, read_json, valid_box, write_json
from scripts.experiments.transfer.detection_dental_transfer_data import MANIFEST as DENTAL
from scripts.experiments.detector_audits.detection_empty_scene_data import MANIFEST as NEW_SCENES
from scripts.experiments.detector_training.detection_hillclimb_corpus import LOCK as SIMULATOR
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, HELD_OUT, inputs
from scripts.experiments.detector_audits.detection_medbin_data import MANIFEST as MEDBIN
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_transfer import ANNOTATIONS

DATASET=DATA/"reviewed-box-transfer"
DESIGN=ART/"reviewed_box_transfer_data.json"
INITIAL=Path("data/rfdetr/rf-detr-nano.pth")
MD5="fb6504cce7fbdc783f7a46991f07639f"
SOURCE="https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
PUBLIC=ART/"public_additional_fit.json"
OPENIMAGES=ART/"openimages_additional_manifest.json"
SEED=88271
ARMS=("repeat-real","mix-simulator")
EDGE=320


def benchmark_sources():
    real=[{**row,"domain":name} for path,role,name in (
        (PUBLIC,"fit","commons"),(OPENIMAGES,"openimages-fit","open-images"),(DENTAL,"fit","dental-photos"))
        for row in read_json(path)["rows"] if row["split"]==role]
    simulated=[{**r,"domain":"simulator"} for r in read_json(SIMULATOR)["rows"] if r["split"]=="fit"]
    dev=[{**r,"domain":name} for name in DEVELOPMENT for r in inputs(name)]
    dev.extend({**r,"domain":"dental-development"} for r in read_json(DENTAL)["rows"] if r["split"]=="development")
    frozen=[r for name in HELD_OUT for r in inputs(name)]
    frozen.extend(r for path in (NEW_SCENES,MEDBIN,ART/"consensus_audit_manifest.json",
                                  ART/"external_lookalike_manifest.json",ART/"precision_negative_manifest.json")
                  for r in read_json(path)["rows"])
    frozen.extend(r for path,role in ((OPENIMAGES,"openimages-audit"),(DENTAL,"audit"))
                  for r in read_json(path)["rows"] if r["split"]==role)
    pool={role:rows for role,rows in (("fit",real+simulated),("development",dev),("reserve",frozen))}
    protected=[{**r,"split":role} for role,rows in pool.items() for r in rows]
    checks=leakage(protected)
    if checks:
        raise ValueError(f"Proposed box transfer would overlap protected groups: {checks[:4]}")
    return real,simulated,dev


def coco(records, folder, assets):
    images,objects=[],[]
    folder.mkdir(parents=True)
    for index,row in enumerate(records,1):
        key=row["id"]
        if key not in assets:
            raster=source(row)
            assets[key]=(raster.size,folder/f"{index:04d}.png")
            raster.save(assets[key][1])
        (width,height),previous=assets[key]
        filename=f"{index:04d}.png"
        if previous!=folder/filename:
            (folder/filename).hardlink_to(previous)
        images.append(dict(id=index,file_name=filename,width=width,height=height))
        for obj in row["objects"]:
            box=obj["bbox"]
            if not valid_box(box):
                raise ValueError(f"Invalid reviewed box for {row['id']}")
            x,y,r,b=box
            w,h=(r-x)*width,(b-y)*height
            objects.append(dict(id=len(objects)+1,image_id=index,category_id=1,
                                bbox=[x*width,y*height,w,h],area=w*h,iscrowd=0,segmentation=[]))
    annotation=folder/ANNOTATIONS
    write_json(annotation,dict(images=images,annotations=objects,
                               categories=[dict(id=1,name="syringe",supercategory="medical instrument")]))
    return dict(images=len(images),targets=len(objects),sha256=digest(annotation),
                negatives=sum(not row["objects"] for row in records))


def build():
    if DATASET.exists() or DESIGN.exists():
        raise ValueError("The controlled box transfer has already been prepared.")
    real,sim,dev=benchmark_sources()
    rng=Random(SEED)
    arms={arm:real.copy() for arm in ARMS}
    for images,extra in ((0,22),(1,42)):
        arms["repeat-real"].extend(rng.sample([r for r in real if len(r["objects"])==images],extra))
        arms["mix-simulator"].extend(rng.sample([r for r in sim if len(r["objects"])==images],extra))
    with TemporaryDirectory(prefix="reviewed-box-prep-",dir=DATA) as temporary:
        root=Path(temporary)/"prepared"
        root.mkdir()
        shared={}
        validation=coco(dev,root/"development",shared)
        reports={}
        for arm,rows in arms.items():
            rng.shuffle(rows)
            here=root/arm
            reports[arm]=coco(rows,here/"train",shared)
            (here/"valid").symlink_to(Path("../development"),target_is_directory=True)
        if len({(r["images"],r["targets"],r["negatives"]) for r in reports.values()})!=1:
            raise ValueError("Box transfer would not have a matched update and label budget.")
        root.rename(DATASET)
    summary=dict(seed=SEED,training=reports,development=validation,code_sha256=digest(Path(__file__)),
                 arms={k:[dict(id=r["id"],domain=r["domain"],sha256=r["sha256"],group=r["group"]) for r in v]
                       for k,v in arms.items()},
                 development_groups=dict(Counter(r["domain"] for r in dev)),
                 protocol="Both arms include the same independently reviewed real fit photographs. One arm repeats "
                          "42 single-object and 22 empty real scenes, the other replaces those repetitions with "
                          "the same counts of simulator fit scenes. Previously reserved data play no fitting role.")
    write_json(DESIGN,summary)
    print("Prepared matched real-photo arms",reports,validation,flush=True)


def checkpoint():
    with INITIAL.open("rb") as payload:
        observed=file_digest(payload,lambda:md5(usedforsecurity=False)).hexdigest()
    if observed!=MD5:
        raise ValueError("The RF-DETR Nano asset failed the publisher checksum.")
    return digest(INITIAL)


def train(arm):
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    record=ART/f"reviewed_box_transfer_{arm}.json"
    if record.exists() and read_json(record)["status"]=="completed":
        raise ValueError("This arm completed already.")
    known=read_json(DESIGN)
    data=DATASET/arm
    if (known["training"][arm]["sha256"]!=digest(data/"train"/ANNOTATIONS) or
        known["development"]["sha256"]!=digest(data/"valid"/ANNOTATIONS)):
        raise ValueError("Box-supervised folds changed after selection.")
    out=DATA/"runs"/f"box-transfer-{arm}"
    configuration=dict(dataset_dir=str(data),output_dir=str(out),device="cpu",epochs=2,batch_size=4,
                       resolution=EDGE,seed=SEED,lr=.00015,lr_encoder=0,num_workers=0,num_sanity_val_steps=0,
                       compute_val_loss=True,use_ema=False,multi_scale=False,save_dataset_grids=False,
                       run_test=False,progress_bar=None,train_log_on_step=False)
    proof=dict(configuration=configuration,dataset_sha256=digest(DESIGN),pretrained_sha256=checkpoint(),
               pretrained_source=SOURCE,license="Apache-2.0")
    write_json(record,dict(status="started",**proof))
    model=RFDETRNano(device="cpu",pretrain_weights=str(INITIAL),freeze_encoder=True,num_classes=1)
    begin=perf_counter()
    model.train(**configuration)
    output=out/"checkpoint_best_total.pth"
    copyfile(out/"metrics.csv",ART/f"reviewed_box_transfer_{arm}_metrics.csv")
    write_json(record,dict(status="completed",seconds=perf_counter()-begin,checkpoint=str(output),
                           checkpoint_sha256=digest(output),**proof))
    print("Completed supervised box transfer",arm,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","train"))
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.action=="build":
        build()
    elif args.arm:
        train(args.arm)
    else:
        parser.error("Choose a fixed-budget training arm.")
