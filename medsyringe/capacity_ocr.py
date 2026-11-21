"""Read original-resolution syringe crops and identify supported capacity markings."""
from collections import defaultdict
from math import atan2, degrees
from pathlib import Path
import re

import cv2
import numpy as np

NUMBER = r"(?:[1-9]\d{0,2}(?:[.,]\d{1,2})?|0[.,][1-9]\d?)"
ONLY_NUMBER = re.compile(NUMBER)
PAIR = re.compile(rf"(?:({NUMBER})\s*m\s*l|m\s*l\s*({NUMBER}))", re.IGNORECASE)
UNIT = re.compile(r"m\s*l",re.IGNORECASE)
MODEL_ROOT = Path("data/ocr-v6/models")
DETECTOR = "PP-OCRv6_tiny_det"
RECOGNIZER = "PP-OCRv6_tiny_rec"


def numeric(text):
    return float(text.replace(",", "."))


def bounded_image(image, edge=1248):
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")),cv2.COLOR_RGB2BGR)
    height,width = bgr.shape[:2]
    scale = min(1.,edge/max(width,height))
    if scale == 1:
        return bgr
    return cv2.resize(bgr,(max(1,round(width*scale)),max(1,round(height*scale))),interpolation=cv2.INTER_AREA)


def barrel_angle(bgr):
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,45,135)
    span=max(gray.shape)
    lines=cv2.HoughLinesP(edge,1,np.pi/180,threshold=max(25,span//22),
                          minLineLength=max(25,span//6),maxLineGap=max(8,span//35))
    if lines is None or len(lines)<2:
        return 0.
    endpoints=lines[:,0,:]
    delta=endpoints[:,2:]-endpoints[:,:2]
    weights=np.sum(delta.astype(float)**2,axis=1)
    orientations=np.arctan2(delta[:,1],delta[:,0])
    resultant=np.sum(weights*np.exp(2j*orientations))/weights.sum()
    if abs(resultant)<.26:
        return 0.
    direction=degrees(atan2(resultant.imag,resultant.real))/2
    return round(direction%180-90,1)


def rotated_image(bgr, degrees_ccw):
    height,width=bgr.shape[:2]
    rotate=cv2.getRotationMatrix2D((width/2,height/2),degrees_ccw,1)
    width_out=round(abs(rotate[0,0])*width+abs(rotate[0,1])*height)
    height_out=round(abs(rotate[1,0])*width+abs(rotate[1,1])*height)
    rotate[0,2]+=(width_out-width)/2
    rotate[1,2]+=(height_out-height)/2
    pixels=cv2.warpAffine(bgr,rotate,(width_out,height_out),flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=(160,160,160))
    return pixels,cv2.invertAffineTransform(rotate)


def view_angles(bgr):
    """Return raw quarter turns and extra views aligned to the barrel when estimated."""
    raw=[(float(angle),"raw") for angle in (0,180,90,270)]
    roll=barrel_angle(bgr)
    if abs(roll)<3:
        return raw
    aligned=[((angle+roll)%360,"aligned") for angle in (0,180)]
    return [*raw,*aligned]


def records(result, inverse, raw_shape, rotation, family):
    height,width=raw_shape[:2]
    out=[]
    for word, score, polygon in zip(result["rec_texts"],result["rec_scores"],result["rec_polys"],strict=True):
        text=word.strip()
        score=float(score)
        if not text or not np.isfinite(score) or score<.35:
            continue
        points=np.asarray(polygon,dtype=float).reshape(-1,2)
        if not len(points) or not np.isfinite(points).all():
            continue
        mapped=points@inverse[:,:2].T+inverse[:,2]
        def bounds(points):
            return [*points.min(axis=0).tolist(),*points.max(axis=0).tolist()]
        crop_box=np.asarray(bounds(mapped))/[width,height,width,height]
        out.append(dict(text=text,score=score,crop_bbox=crop_box.clip(0,1).tolist(),
                        view_bbox=bounds(points),rotation=rotation,family=family))
    return out


class SceneOCR:
    def __init__(self, model_root=MODEL_ROOT):
        from paddleocr import PaddleOCR

        self.model_root=model_root
        self.recognizer=None
        self.ocr=PaddleOCR(
            text_detection_model_name=DETECTOR,text_detection_model_dir=str(model_root/DETECTOR),
            text_recognition_model_name=RECOGNIZER,text_recognition_model_dir=str(model_root/RECOGNIZER),
            use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
            device="cpu",cpu_threads=2,enable_mkldnn=False,
        )

    def read(self,image):
        bgr=bounded_image(image)
        data, spaces=[], {}
        for angle,family in view_angles(bgr):
            space,inverse=rotated_image(bgr,angle)
            prediction=next(self.ocr.predict_iter(space,text_rec_score_thresh=.35))
            view=records(prediction,inverse,bgr.shape,angle,family)
            data.extend(view)
            spaces[angle,family]=(space,inverse)
        data.extend(self.target_units(data,spaces,bgr.shape))
        return data

    def target_units(self, tokens, spaces, raw_shape):
        anchors=[t for t in tokens if t["text"] in ("20","60") and t["score"]>=.69]
        anchors.sort(key=lambda t:(t["score"],t["view_bbox"][1]),reverse=True)
        if not anchors:
            return []
        if self.recognizer is None:
            from paddleocr import TextRecognition

            self.recognizer=TextRecognition(model_name=RECOGNIZER,
                model_dir=str(self.model_root/RECOGNIZER),device="cpu",cpu_threads=2,enable_mkldnn=False)
        patches, provenance=[], []
        for token in anchors[:3]:
            image, inverse=spaces[token["rotation"],token["family"]]
            height,width=image.shape[:2]
            left,top,right,bottom=token["view_bbox"]
            w,h=right-left,bottom-top
            for start,end in ((-.95,.2),(-.40,.54),(.02,.46)):
                x0=max(0,round(left-.20*w)); x1=min(width,round(right+.30*w))
                y0=max(0,round(top+start*h)); y1=min(height,round(top+end*h))
                if x1-x0<5 or y1-y0<5:
                    continue
                patch=image[y0:y1,x0:x1]
                patch=cv2.resize(patch,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
                patches.append(patch)
                provenance.append((token,inverse,(x0,y0,x1,y1)))
        result=[]
        if not patches:
            return result
        for recognized,(anchor,inverse,(x0,y0,x1,y1)) in zip(
                self.recognizer.predict_iter(patches,batch_size=6),provenance,strict=True):
            observed=recognized["rec_text"].strip()
            if not UNIT.fullmatch(observed):
                continue
            score=min(float(recognized["rec_score"]),anchor["score"])
            if score<.6:
                continue
            transformed=records(dict(rec_texts=[observed],rec_scores=[score],
                                     rec_polys=[[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]),
                                inverse,raw_shape,anchor["rotation"],anchor["family"])[0]
            # The direct recognizer read the unit in the patch. Keep the crop
            # and the previously recognized anchor as distinct evidence.
            transformed["source"]="unit crop"
            transformed["anchor_bbox"]=anchor["crop_bbox"]
            result.append(transformed)
        return result


def center(token):
    x,y,r,b=token["view_bbox"]
    return (x+r)/2,(y+b)/2,max(1.,r-x),max(1.,b-y)


def link_distance(unit,number):
    ux,uy,uw,uh=center(unit)
    nx,ny,nw,nh=center(number)
    width,height=max(uw,nw),max(uh,nh)
    # The supplied barrels print the unit immediately above the terminal number.
    if 0 < ny-uy <= 2.25*height and abs(nx-ux) <= .8*width:
        return (ny-uy)/height+abs(nx-ux)/width
    # In-line layouts also print the number and unit next to each other.
    if abs(ny-uy)<=.5*height and abs(nx-ux)<=1.9*width:
        return abs(nx-ux)/width+abs(ny-uy)/height+1.
    return None


def endpoint(number,value,numbers):
    x,y,w,h=center(number)
    return not any(other is not number and numeric(other["text"])>value and
                   abs(center(other)[0]-x)<1.6*max(w,center(other)[2]) and
                   abs(center(other)[1]-y)>h*.5 for other in numbers)


def candidates_for_view(tokens):
    numbers=[t for t in tokens if ONLY_NUMBER.fullmatch(t["text"])]
    for token in tokens:
        pair=PAIR.fullmatch(token["text"])
        if pair:
            value=numeric(next(s for s in pair.groups() if s))
            if endpoint(token,value,numbers):
                yield value,token["score"],"joined number and unit",[token]
        if not UNIT.fullmatch(token["text"]):
            continue
        if token.get("anchor_bbox") is not None:
            nearby=[(0.,n) for n in numbers if n["crop_bbox"]==token["anchor_bbox"]]
        else:
            nearby=[(distance,number) for number in numbers
                    if (distance:=link_distance(token,number)) is not None]
        if not nearby:
            continue
        _,number=min(nearby,key=lambda item:item[0])
        value=numeric(number["text"])
        if endpoint(number,value,numbers):
            yield value,min(token["score"],number["score"]),"adjacent number and unit",[token,number]


def visible_scale(tokens, cutoff):
    marks=[t for t in tokens if ONLY_NUMBER.fullmatch(t["text"]) and t["score"]>=cutoff]
    for anchor in marks:
        value=numeric(anchor["text"])
        step={20:5,60:10}.get(value)
        if step is None or not endpoint(anchor,value,marks):
            continue
        ax,ay,aw,ah=center(anchor)
        compatible={}
        for mark in marks:
            value_here=numeric(mark["text"])
            mx,my,mw,mh=center(mark)
            if not 0<value_here<value or value_here%step or my>=ay or abs(mx-ax)>max(aw,mw):
                continue
            if value_here not in compatible or mark["score"]>compatible[value_here]["score"]:
                compatible[value_here]=mark
        if len(compatible)<2:
            continue
        ordered=[(*center(compatible[k])[:2],k) for k in sorted(compatible)]
        ordered.append((ax,ay,value))
        slopes=[(b[1]-a[1])/(b[2]-a[2]) for a,b in zip(ordered,ordered[1:])]
        if min(slopes)<=0 or max(slopes)>2.1*min(slopes) or ay-ordered[0][1] < 1.7*ah:
            continue
        results=[*compatible.values(),anchor]
        yield value,min(t["score"] for t in results),results


def resolve(tokens, cutoff=.78, repeats=1, scale_cutoff=.80):
    views=defaultdict(list)
    for token in tokens:
        views[token["rotation"],token["family"]].append(token)
    by_value=defaultdict(list)
    for key,readings in views.items():
        for value,score,method,evidence in candidates_for_view(readings):
            if score>=cutoff:
                by_value[value].append((key,score,method,evidence))
    capacity=None
    if len(by_value)==1:
        value, evidence=next(iter(by_value.items()))
        if len({r[0] for r in evidence})>=repeats:
            _,score,method,tokens_used=max(evidence,key=lambda r:r[1])
            capacity=dict(value=value,unit="mL",score=score,method=method,
                          evidence=[dict(text=t["text"],crop_bbox=t["crop_bbox"],rotation=t["rotation"])
                                    for t in tokens_used])
    scales=defaultdict(list)
    for readings in views.values():
        for value,score,evidence in visible_scale(readings,scale_cutoff):
            scales[value].append((score,evidence))
    scale=None
    if len(scales)==1 and len(by_value)<=1:
        value,evidence=next(iter(scales.items()))
        if not capacity or capacity["value"]==value:
            score,items=max(evidence,key=lambda r:r[0])
            scale=dict(value=f"{int(value)}-scale",score=score,method="ordered terminal graduations",
                       evidence=[dict(text=t["text"],crop_bbox=t["crop_bbox"],rotation=t["rotation"])
                                 for t in items])
    if capacity and capacity["value"] in (20,60) and not scales.keys()-{capacity["value"]}:
        scale=dict(value=f'{int(capacity["value"])}-scale',score=capacity["score"],
                   method="read number and unit",evidence=capacity["evidence"])
    if capacity and scales.keys()-{capacity["value"]}:
        capacity=None
        scale=None
    status=("capacity read" if capacity else "ambiguous" if len(by_value)>1 or len(scales)>1
            else "scale read" if scale else "markings only" if tokens else "unreadable")
    return dict(status=status,candidates=sorted(tokens,key=lambda t:-t["score"])[:24],
                scale=scale,capacity=capacity,apparent_volume=None)
