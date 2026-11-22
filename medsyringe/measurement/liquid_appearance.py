"""Describe visible liquid brightness and measure opacity given a matched reference."""
import cv2
import numpy as np

OPACITY_REASON="A registered clear-barrel reference at the same exposure is required."


def unavailable(reason):
    return dict(status="insufficient evidence",reason=reason,lightness=None,opacity=None,
                opacity_reason=OPACITY_REASON)


def appearance(rgb, syringe_mask, liquid_prior=None):
    """Measure lightness only in the colored interior supported by the syringe mask."""
    if rgb.shape[:2]!=syringe_mask.shape:
        raise ValueError("The syringe mask and image must use the same coordinates.")
    h,w=rgb.shape[:2]
    mask=np.asarray(syringe_mask,dtype=np.uint8)
    y,x=mask.nonzero()
    if len(x)<h*w*.10:
        return unavailable("The syringe silhouette is too small or incomplete.")
    points=np.column_stack((x,y)).astype(float)
    origin=points.mean(axis=0)
    _,vectors=np.linalg.eigh(np.cov(points.T))
    axis=vectors[:,-1]
    lateral=vectors[:,0]
    axial=(points-origin)@axis
    radial=(points-origin)@lateral
    axial_span=np.ptp(np.quantile(axial,[.02,.98]))
    radial_span=np.ptp(np.quantile(radial,[.05,.95]))
    if axial_span<1.55*radial_span:
        return unavailable("The visible barrel axis is ambiguous.")
    yy,xx=np.indices((h,w))
    radius=(xx-origin[0])*lateral[0]+(yy-origin[1])*lateral[1]-np.median(radial)
    core=cv2.erode(mask,np.ones((3,3),dtype=np.uint8)).astype(bool)

    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue,saturation,value=np.moveaxis(hsv,-1,0)
    warm=(hue>=5)&(hue<=39)&(saturation>=45)&(value>=25)
    support=warm & core & (np.abs(radius)<=.45*radial_span)
    if liquid_prior is not None:
        if liquid_prior.shape!=support.shape:
            raise ValueError("The liquid prior and syringe mask must use the same pixel grid.")
        support &= liquid_prior
    support=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    support=support.astype(bool)&warm&core
    if liquid_prior is not None:
        support &= liquid_prior
    # Close small gaps for component discovery only. Measure warm interior
    # pixels rather than the black printed marks or shadows between them.
    links=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,components,_=cv2.connectedComponentsWithStats(links)
    if count<2:
        return unavailable("A colored liquid region could not be isolated.")
    index=1+int(np.argmax(components[1:,cv2.CC_STAT_AREA]))
    interior=(labels==index)&support
    fy,fx=interior.nonzero()
    if len(fx)<max(80,.055*len(x)):
        return unavailable("The visible color covers too little of the barrel.")
    span=np.ptp((np.column_stack((fx,fy))-origin)@axis)
    if span<axial_span*.13:
        return unavailable("The colored region does not extend along the barrel.")
    L=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)[...,0][interior]
    lo,mid,hi=(float(k) for k in np.quantile(L,[.25,.5,.75]))
    category="dark" if mid<35 else "light" if mid>=65 else "medium"
    return dict(status="appearance measured",reason="Colored interior of the segmented syringe",
                lightness=dict(lstar=round(mid,1),iqr=[round(lo,1),round(hi,1)],category=category),
                region=[float(fx.min()/w),float(fy.min()/h),float((fx.max()+1)/w),float((fy.max()+1)/h)],
                support_fraction=round(len(fx)/len(x),3),opacity=None,opacity_reason=OPACITY_REASON)


def luminance(rgb):
    srgb=rgb.astype(float)/255
    linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    return linear@np.array([.2126,.7152,.0722])


def relative_opacity(observed,clear_reference,*,matched_setup=False):
    """Measure contrast attenuation in registered crops with a known textured background."""
    if not matched_setup:
        return dict(value=None,reason=OPACITY_REASON)
    if observed.shape!=clear_reference.shape or observed.ndim!=3 or observed.shape[-1]!=3:
        return dict(value=None,reason="The registered color crops must have identical dimensions.")
    actual,reference=luminance(observed).ravel(),luminance(clear_reference).ravel()
    keep=(reference>.025)&(reference<.97)&(actual>.01)&(actual<.98)
    if keep.sum()<100 or np.ptp(reference[keep])<.18:
        return dict(value=None,reason="Too little unclipped contrast in the reference crop.")
    X,Y=reference[keep],actual[keep]
    A=np.column_stack((X,np.ones(len(X))))
    slope,offset=np.linalg.lstsq(A,Y,rcond=None)[0]
    residual=Y-(slope*X+offset)
    quality=1-np.sum(residual**2)/max(np.sum((Y-Y.mean())**2),1e-12)
    if not 0.05<=slope<=1.05 or quality<.85:
        return dict(value=None,reason="The matched texture does not support a reliable attenuation estimate.")
    return dict(value=round(float(np.clip(1-slope,0,1)),3),
                method="Relative luminance contrast attenuation",fit_r2=round(float(quality),3),
                pixels=int(keep.sum()))
