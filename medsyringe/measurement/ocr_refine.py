"""Re-read small terminal markings at a scale suggested by visible graduations."""
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np

from medsyringe.measurement.capacity_ocr import (ONLY_NUMBER, bounded_image, center, numeric, records, rotated_image)


def marking_regions(tokens):
    views=defaultdict(list)
    for token in tokens:
        if token["score"]>=.80 and ONLY_NUMBER.fullmatch(token["text"]):
            views[token["rotation"],token["family"]].append(token)
    proposals=[]
    for view,marks in views.items():
        for capacity,step in ((20,5),(60,10)):
            valid=[t for t in marks if 0<numeric(t["text"])<=capacity and numeric(t["text"])%step==0]
            for a,b in combinations(sorted(valid,key=lambda o:center(o)[1]),2):
                av,bv=numeric(a["text"]),numeric(b["text"])
                if not av<bv or capacity-bv>2*step:
                    continue
                ax,ay,aw,ah=center(a)
                bx,by,bw,bh=center(b)
                w,h=max(aw,bw),max(ah,bh)
                if abs(ax-bx)>.9*w or by-ay<h*1.4:
                    continue
                pitch=(by-ay)/(bv-av)
                if not .85*h <= pitch*step <= 9*h:
                    continue
                y=by+(capacity-bv)*pitch
                if any(numeric(t["text"])>capacity and abs(center(t)[0]-bx)<1.2*w for t in marks):
                    continue
                supporting=sum(abs(center(t)[0]-bx)<1.1*w and
                               abs(center(t)[1]-(by+(numeric(t["text"])-bv)*pitch))<=1.15*h
                               for t in valid)
                bounds=(bx-1.15*w,y-2.6*h,bx+1.15*w,y+1.40*h)
                priority=(supporting,bv==capacity,min(a["score"],b["score"]),-abs(ax-bx)/w)
                proposals.append((priority,view,bounds))
    used=set()
    for _,view,bounds in sorted(proposals,reverse=True):
        if view not in used:
            yield view,bounds
            used.add(view)
        if len(used)==3:
            break


def guided_read(reader,image,tokens):
    regions=list(marking_regions(tokens))
    if not regions:
        return []
    low,high=bounded_image(image),bounded_image(image,edge=1900)
    refined,spaces=[],{}
    for (rotation,family),(l,t,r,b) in regions:
        space,inverse=rotated_image(low,rotation)
        larger,_=rotated_image(high,rotation)
        h,w=space.shape[:2]
        H,W=larger.shape[:2]
        sx,sy=W/w,H/h
        left,top=max(0,round(l*sx)),max(0,round(t*sy))
        right,bottom=min(W,round(r*sx)),min(H,round(b*sy))
        if right-left<10 or bottom-top<10:
            continue
        patch=larger[top:bottom,left:right]
        zoom=max(1.,min(3.,380/max(patch.shape[:2])))
        patch=cv2.resize(patch,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC)
        result=next(reader.ocr.predict_iter(patch,text_rec_score_thresh=.45))
        offsets=np.array([left,top]); transform=np.array([sx,sy])
        out=dict(rec_texts=result["rec_texts"],rec_scores=result["rec_scores"],
                 rec_polys=[(np.asarray(poly,dtype=float)/zoom+offsets)/transform
                            for poly in result["rec_polys"]])
        extracted=records(out,inverse,low.shape,rotation,family)
        for token in extracted:
            token["source"]="graduation-guided crop"
        refined.extend(extracted)
        spaces[rotation,family]=(space,inverse)
    return [*refined,*reader.target_units(refined,spaces,low.shape)]
