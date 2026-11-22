# Separating overlapping syringe bodies with SAM 3

The released hybrid returned two detections on the supplied screenshot and four on its public source. A separate manual reading identified ten visible bodies. We tested whether more image detail and mask-based duplicate suppression preserve those thin, overlapping instances. The experimental branch uses the pinned SAM 3 weights on Intel Arc, with no new model fitting. [Released output](../../../artifacts/detection_hillclimb/public_overlap_count_case.json)

## Development and fixed transfer

The four development photos contained 18 references. The factorial comparison tested 504 and 756 pixel input, three syringe prompts and their union, the model box and silhouette bounds, box or pixel-mask duplicate suppression, and four score floors. It selected the whole-device prompt `syringe`, 756 pixels, model boxes and mask IoU 0.45. The fixed confidence floor is 0.50. The 504-pixel box-suppression control matched 15 of 18 and produced 11 false positives. The selected arm matched 17 of 18 with one false positive and two exact counts. The selected branch matched all ten approximate visible-body boxes on the focal public photo. [Development results](../../../artifacts/detection_hillclimb/crowded_sam_development.json)

Screenshot alignment used 297 SIFT correspondences with a median residual of 0.21 pixels. Against the transformed manual boxes, the released detector matched two of ten and the experimental branch matched ten of ten at IoU 0.50. These are two renditions of the same photograph, from a development group. The comparison does not estimate independent accuracy. [Screenshot diagnostics](../../../artifacts/detection_hillclimb/crowded_sam_screenshot_development.json) · [Photograph and per-instance overlays](../../../data/public-syringe-dataset/crowded-sam-before-after.jpg)

The fixed arm was then run on three earlier public evaluation photos from three other creators. It matched nine of nine visible objects with zero false positives and three exact counts, including a negative scene. The historical hybrid had the same scene counts with eight matches and one false-positive box. The related `nathanf` Commons scene was excluded from this comparison. One crowded positive scene limits the transfer claim. [Fixed transfer](../../../artifacts/detection_hillclimb/crowded_sam_transfer.json) · [SAM 3 vision-feature reuse](https://huggingface.co/docs/transformers/model_doc/sam3#efficient-multi-prompt-inference)

## Integrating with the existing detector

Two fixed fusion rules invoke the 756-pixel branch when the hybrid already accepts multiple objects. They require the supplemental branch to propose more objects and pair each existing box with a separate supplemental box at IoU 0.40. The more conservative rule retains the original boxes and appends the unmatched supplemental boxes. Both rules corrected all three undercounts in the native 23-photo collection. On its original frozen labels, matched instances increased from 39 to 43 of 48, with four to five false positives. All original matched instances were preserved. The fixed qualification rejected the rules because of that extra unmatched box. The median supplemental time on five triggered native photos was 2.55 s, with the SAM process loaded. [Frozen qualification](../../../artifacts/detection_hillclimb/crowded_sam_hybrid_qualification.json)

Two separate readers then marked the five crowded photos using only the raw images. They agreed on all five scene counts. Their paired box IoU median was 0.977. Both placed the disputed syringe inside the red review overlay. The frozen reference box extends above the visible barrel. On either new reading, the same fixed fusion increased matched instances from 40 to 45 of 48, retained the previous matches and kept false positives at three. The frozen result remains available for comparison. This sensitivity analysis uses one historical acquisition. [Both readings and sensitivity results](../../../artifacts/detection_hillclimb/crowded_sam_reference_sensitivity.json) · [Disagreement image](../../../data/public-syringe-dataset/crowded-native-error.jpg)

On six additional prospectively marked public photos with nine visible syringe bodies and two hard negatives, the released detector achieved seven matches and four exact counts. Both fixed fusion policies increased this to eight matches and five exact counts with no added false positives or lost matches. The standalone SAM detector matched all nine bodies and added two false positives. One negative source supplies both hard-negative images. [Prospective results](../../../artifacts/detection_hillclimb/crowded_sam_prospective.json) · [Frozen reference manifest](../../../artifacts/detection_hillclimb/crowded_sam_prospective_reference.json)

Click-assisted segmentation was also tried as an annotation shortcut. One click on each known body gave three correctly localized proposals. The nearest adjacent negative click raised that to four. These geometry results are on the single development image and the clicks already specify the count. [Click-assisted diagnostic](../../../artifacts/detection_hillclimb/crowded_click_review_development.json)

## Run the separate review tool

```bash
HF_HUB_OFFLINE=1 data/sam3/xpu-env/bin/python -m scripts.experiments.crowds.crowded_sam_review \
    /path/to/photo.jpg --output /path/to/new-review.json
```

The command exports the fixed branch's normalized boxes, proposed count and checkpoint binding. The published Intel result on the focal source contained ten proposals. The API still uses the verified release. The proposal count requires review and is not promoted into the observed inventory automatically.
