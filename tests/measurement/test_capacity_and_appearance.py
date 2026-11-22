import numpy as np
import pytest

from medsyringe.capacity_ocr import resolve
from medsyringe.liquid_appearance import appearance, relative_opacity


def token(text, y, rotation=0, score=.98):
    return dict(text=text,score=score,view_bbox=[10,y,30,y+10],crop_bbox=[.1,y/200,.3,(y+10)/200],
                rotation=rotation,family="raw")


def test_capacity_requires_unit_evidence_and_abstains_on_conflict():
    assert resolve([token("20",70)])["capacity"] is None
    response=resolve([token("ml",56,180),token("20",70,180)])
    assert response["capacity"]["value"]==20
    assert response["capacity"]["evidence"][0]["rotation"]==180
    assert response["scale"]["value"]=="20-scale"
    assert resolve([token("20 mL",70),token("60 mL",120,180)])["capacity"] is None


def test_ordered_scale_does_not_invent_unit_or_liquid_volume():
    results=resolve([token("5",36),token("10",66),token("15",96),token("20",126)])
    assert results["scale"]["value"]=="20-scale"
    assert results["capacity"] is None
    assert results["apparent_volume"] is None


def test_unclipped_matched_reference_supports_only_relative_opacity():
    reference=np.broadcast_to(np.linspace(55,240,100,dtype=np.uint8)[None,:,None],(50,100,3)).copy()
    srgb=reference.astype(float)/255
    linear=((srgb+.055)/1.055)**2.4
    observed=np.round(np.clip(1.055*(.4*linear+.03)**(1/2.4)-.055,0,1)*255).astype(np.uint8)
    assert relative_opacity(observed,reference)["value"] is None
    assert relative_opacity(observed,reference,matched_setup=True)["value"]==pytest.approx(.6,abs=.015)
    assert relative_opacity(observed,reference[:,::-1],matched_setup=True)["value"] is None


def test_lightness_uses_supported_interior_and_preserves_unknown_opacity():
    image=np.full((180,100,3),220,dtype=np.uint8)
    mask=np.zeros((180,100),dtype=np.uint8)
    mask[10:166,31:69]=1
    image[24:103,36:64]=[185,87,35]
    # Saturated pixels outside the parent silhouette are irrelevant.
    image[:,0:20]=[255,70,25]
    report=appearance(image,mask)
    assert report["lightness"]["category"]=="medium"
    assert report["opacity"] is None
    assert report["region"][0]>=.35 and report["region"][2]<=.65
    assert appearance(np.full_like(image,220),mask)["lightness"] is None
