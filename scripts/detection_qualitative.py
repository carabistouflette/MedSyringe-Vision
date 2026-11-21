"""Render image-and-box evidence for the frozen public comparison."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont,ImageOps

from medsyringe.data import digest, read_json
from medsyringe.evaluate import match
from scripts.detection_hillclimb_experiment import FLOOR,inputs,loaded
from scripts.detection_semantics_audit import LOCK,output
from scripts.precision_study import source


PALETTE = {"reference":"#50d3b4","match":"#66a9ff","error":"#ffad49","canvas":"#142228","text":"#eef5f3"}


def boxes(im, candidates, labels, fixed_size=(354,225)):
    photo = ImageOps.contain(im,fixed_size)
    pane = Image.new("RGB",fixed_size,PALETTE["canvas"])
    offset = ((pane.width-photo.width)//2,(pane.height-photo.height)//2)
    pane.paste(photo,offset)
    draw = ImageDraw.Draw(pane)
    for coords,label in zip(candidates,labels,strict=True):
        left,top,right,bottom = coords
        x,y = offset
        draw.rectangle((x+int(left*photo.width),y+int(top*photo.height),
                        x+int(right*photo.width)-1,y+int(bottom*photo.height)-1),
                        outline=PALETTE[label],width=2)
    return pane


def gallery(folder):
    study = "public-audit"
    cutoff = read_json(FLOOR)["cutoff"]
    rows = inputs(study)
    before = loaded(study,"baseline")
    audit = read_json(output(study))
    after = audit["frames"]
    if audit["key"]["freeze_sha256"]!=digest(LOCK) or set(before)!=set(after):
        raise ValueError("The public gallery requires both complete prediction caches.")
    accepted = [{key:[p for p in f["ranked"] if p["score"]>=cutoff] for key,f in result.items()}
                for result in (before,after)]
    assays = [match(rows,ranks,.50) for ranks in accepted]
    by_key = []
    for assignments, used in assays:
        output_rows = {row["id"]:[] for row in rows}
        for id_,prediction,truth,_ in assignments:
            label = "error" if truth is None else "match"
            output_rows[id_].append((prediction["bbox"],label))
        by_key.append((output_rows,used))
    selected = []
    for row in rows:
        key = row["id"]
        delta = len(by_key[1][1][key])-len(by_key[0][1][key])
        errors = Counter(tag for _,tag in by_key[1][0][key])
        if delta or errors["error"] or len(by_key[1][1][key])<len(row["objects"]):
            selected.append((delta,key,row))
    selected.sort(key=lambda obj:(-obj[0],obj[1]))
    font = ImageFont.load_default(size=14)
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(selected),4):
        batch = selected[start:start+4]
        width,height = 1100,285
        page = Image.new("RGB",(width,60+height*len(batch)),PALETTE["canvas"])
        draw = ImageDraw.Draw(page)
        for i,title in enumerate(("Visual reference", "Released baseline", "Frozen crop candidate")):
            draw.text((12+i*362,11),title,fill=PALETTE["text"],font=font)
        legend = (("reference","Reference"),("match","One-to-one match"),("error","Unmatched prediction"))
        for i,(name,label) in enumerate(legend):
            x = 12+360*i
            draw.rectangle((x,36,x+12,48),fill=PALETTE[name])
            draw.text((x+20,35),label,fill=PALETTE["text"],font=font)
        for i,(delta,key,row) in enumerate(batch):
            top = 57+i*height
            im = source(row)
            sources = [(list(p["bbox"] for p in row["objects"]),["reference"]*len(row["objects"]))]
            sources += [tuple(zip(*hits[key],strict=True)) if hits[key] else ([],[]) for hits,_ in by_key]
            draw.text((12,top),f"Commons {key} | {len(row['objects'])} reference syringes",fill=PALETTE["text"],font=font)
            for c,(coords,labels) in enumerate(sources):
                page.paste(boxes(im,coords,labels),(12+c*362,top+21))
            attribution = f"{row['author']} | {row['license_label']} | {row['title'][5:85]}"
            draw.text((12,top+251),attribution,fill=PALETTE["text"],font=font)
        page.save(folder/f"commons-frozen-errors-{start//4+1:02}.jpg",quality=94)
    print(f"Exported {len(selected)} real public cases with predicted and reference boxes.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("data/detection_hillclimb/review"))
    gallery(parser.parse_args().output)
