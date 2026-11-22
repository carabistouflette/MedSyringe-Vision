"""Stage a source-disjoint dental syringe challenge from the publisher archive."""
import argparse
from collections import Counter
from hashlib import sha256
import io
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps
import yaml

from medsyringe.data import dhash, digest, leakage, read_json, valid_box, write_json
from scripts.experiments.detector_training.detection_hillclimb_corpus import ART, DATA, SEED
from scripts.experiments.public_data.detection_public_benchmark import prior_exposure

CARD = "https://doi.org/10.17632/mh6zpc7y9g.1"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
API = ("https://data.mendeley.com/public-files/datasets/mh6zpc7y9g/files/"
       "97267a03-b736-4094-9502-af47462a34bf/file_downloaded")
SIZE = 343389889
MEMBER = "Dental tools Datasets/Dental Surgical Tools.v1i.yolov11.zip"
TARGET = DATA / "dental" / "annotated.zip"
LOCK = ART / "dental_manifest.json"


class RangeSource(io.RawIOBase):
    """Seek within the large remote archive and fetch validated byte ranges."""
    block_size = 4*1024**2

    def __init__(self):
        self.pos = 0
        self.block_start = -1
        self.block = b""

    def tell(self):
        return self.pos

    def seek(self,offset,whence=0):
        target = (0,self.pos,SIZE)[whence]+offset
        if target < 0:
            raise OSError("Invalid remote archive offset.")
        self.pos = target
        return self.pos

    def read(self, amount=-1):
        stop = min(SIZE,self.pos+amount) if amount>=0 else SIZE
        content=[]
        while self.pos<stop:
            block = self.pos//self.block_size*self.block_size
            if block != self.block_start:
                end = min(SIZE,block+self.block_size)-1
                request=Request(API,headers={"Range":f"bytes={block}-{end}",
                                             "User-Agent":"MedSyringeVision/1.0 public dataset research"})
                with urlopen(request,timeout=25) as response:
                    if (response.status!=206 or
                        response.headers.get("Content-Range")!=f"bytes {block}-{end}/{SIZE}"):
                        raise OSError("Mendeley did not return the requested archive range.")
                    self.block=response.read()
                if len(self.block)!=end-block+1:
                    raise OSError("The remote archive range was truncated.")
                self.block_start=block
            inside=self.pos-block
            take=min(len(self.block)-inside,stop-self.pos)
            content.append(self.block[inside:inside+take])
            self.pos+=take
        return b"".join(content)


def acquire():
    if TARGET.exists():
        return
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    draft=TARGET.with_suffix(".part")
    with ZipFile(RangeSource()) as archive:
        info=archive.getinfo(MEMBER)
        if not 30_000_000<=info.file_size<=60_000_000:
            raise ValueError("The publisher's nested archive has changed.")
        with archive.open(info) as src, draft.open("wb") as matching:
            copyfileobj(src,matching,1024**2)
    with ZipFile(draft) as zipfile:
        if zipfile.testzip() is not None:
            raise ValueError("A compressed dental dataset member failed its CRC check.")
    draft.replace(TARGET)


def labels(text):
    result=[]
    for line in text.splitlines():
        fields=line.split()
        if not fields:
            continue
        key,*values=fields
        coords=list(map(float,values))
        if len(coords)==4:
            cx,cy,w,h=coords
            box=[max(0.,cx-w/2),max(0.,cy-h/2),min(1.,cx+w/2),min(1.,cy+h/2)]
        elif len(coords)>=6 and len(coords)%2==0 and all(0<=v<=1 for v in coords):
            x,y=coords[::2],coords[1::2]
            box=[min(x),min(y),max(x),max(y)]
        else:
            raise ValueError("An unexpected dental annotation type was found.")
        if not valid_box(box):
            raise ValueError("A dental object box is degenerate.")
        result.append((int(key),box))
    return result


def design():
    if LOCK.exists():
        raise ValueError("The dental references have already been frozen.")
    acquire()
    previous=[]
    for path in (ART/"public_manifest.json",ART/"phlebotomy.json",Path("artifacts/expanded/inventory.json")):
        if path.exists():
            previous.extend(read_json(path)["rows"])
    blocked={int(r["dhash"],16) for r in previous if r.get("dhash")} | prior_exposure()["difference"]
    folder=TARGET.parent/"images"
    folder.mkdir(exist_ok=True)
    found=[]
    with ZipFile(TARGET) as archive:
        groups=set(archive.namelist())
        yaml_path=next(p for p in groups if PurePosixPath(p).name=="data.yaml")
        cat=yaml.safe_load(archive.read(yaml_path))["names"]
        syringe=next(i for i,name in enumerate(cat) if "aspirating syringe" in name.casefold())
        suitable=[]
        for path in sorted(groups):
            name=PurePosixPath(path)
            if (name.parts[0] not in ("test","valid") or name.parent.name!="images"
                or name.suffix.casefold() not in (".jpg",".png",".jpeg")):
                continue
            label_name=str(name.parent.parent/"labels"/(name.stem+".txt"))
            if label_name not in groups:
                continue
            annotation=labels(archive.read(label_name).decode())
            objects=[dict(bbox=box,subtype="dental aspirating syringe") for key,box in annotation if key==syringe]
            distractors=[cat[key] for key,_ in annotation if key!=syringe]
            if not objects and not any("suction" in name.casefold() or "elevator" in name.casefold()
                                       for name in distractors):
                continue
            parent=name.name.split(".rf.")[0]
            suitable.append((path,parent,objects,distractors))
        count=Counter()
        parents=set()
        # Ignore the export's published split. The entire source is withheld.
        for path,parent,objects,distractors in sorted(suitable,key=lambda r:sha256(f"{SEED}:{r[0]}".encode()).digest()):
            positive=bool(objects)
            if count[positive]>=(12 if positive else 8) or parent in parents:
                continue
            blob=archive.read(path)
            with Image.open(io.BytesIO(blob)) as image:
                value=dhash(image)
                width,height=image.size
            key=int(value,16)
            if any((key^other).bit_count()<=4 for other in blocked):
                continue
            checksum=sha256(blob).hexdigest()
            identifier="dental-"+checksum[:16]
            target=folder/f"{identifier}.jpg"
            target.write_bytes(blob)
            found.append(dict(id=identifier,image=target.relative_to("data").as_posix(),sha256=checksum,
                              dhash=value,width=width,height=height,member=path,source_url=CARD+"#"+path,
                              license=LICENSE,split="dental-audit",group="mendeley-dental-instruments-v1",
                              source_parent=parent,objects=objects,distractors=distractors))
            count[positive]+=1
            parents.add(parent)
            blocked.add(key)
    if not count[True] or not count[False] or leakage([*found,*previous]):
        # Historical manifests may overlap each other. Evaluate only pairs
        # involving the new source.
        current=[{**r,"split":"history"} for r in previous]
        if not count[True] or not count[False] or leakage([*found,*current]):
            raise ValueError("Insufficient independent dental reference evidence.")
    write_json(LOCK,dict(rows=found,source=CARD,license=LICENSE,archive_sha256=digest(TARGET),
                         source_member=MEMBER,images=len(found),positives=count[True],
                         negatives=count[False],scope="Single external dental source. Repeated specimens possible.",
                         reference_status="Publisher annotations pending the separate visual check."))
    print(len(found), count[True], count[False],flush=True)


def review():
    rows=read_json(LOCK)["rows"]
    sorted_rows=sorted(rows,key=lambda r:(not r["objects"],r["id"]))
    canvas=Image.new("RGB",(1160,248*((len(rows)+3)//4)),"#29292c")
    draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(sorted_rows):
        with Image.open(Path("data")/row["image"]) as source:
            image=ImageOps.contain(source,(266,213))
        left,top=(i%4)*290+8,(i//4)*248+24
        canvas.paste(image,(left,top))
        color="#ffe728" if row["objects"] else "#e4e4e4"
        draw.text((left,top-19),row["id"],fill=color)
        for obj in row["objects"]:
            box=obj["bbox"]
            draw.rectangle([left+box[0]*image.width,top+box[1]*image.height,
                            left+box[2]*image.width,top+box[3]*image.height],outline=color,width=2)
    canvas.save(TARGET.parent/"reference-review.jpg",quality=90)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("design","review"),default="design")
    (design if parser.parse_args().action=="design" else review)()
