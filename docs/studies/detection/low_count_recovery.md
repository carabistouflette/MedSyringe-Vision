# Low-count syringe recovery and OCR-independent chamber appearance

## Additive count recovery

The released detector missed 45 of the 50 visible syringes in eleven already observed crowded scenes where it emitted at most one detection. The original crowd fusion starts at two detections. This study adds a branch for the lower-count case while retaining the released boxes and their ordering.

SAM 3 ran locally on Intel Arc at 756 pixels with the syringe, clear-plastic syringe, and syringe-barrel prompts. The verifier compared prompt scores and mask alignment with an extended feature set containing principal-axis shape, mask fill, connectivity, and size. It fitted four regularized logistic variants using four acquisition-grouped folds. The fitting dataset supplied 32 photos with verified individual provenance and 36 proposals, of which 15 matched visible references. It contained seven negative scenes and no multi-syringe fitting scenes. The selected model used prompt evidence and kept eight of 15 matched proposals at its out-of-fold zero-false-positive operating point. Its features, coefficients, threshold, input checksums and checkpoint are frozen in the experiment record. [Fit and freeze](../../../artifacts/detection_hillclimb/low_count_recovery_freeze.json)

| Method | Matched / 50 | False positives | Exact scenes / 11 |
|---|---:|---:|---:|
| Released detector | 5 | 0 | 0 |
| Direct SAM recovery | 34 | 3 | 5 |
| Rule-based prompt agreement | 30 | 2 | 4 |
| Selected verifier | 25 | 1 | 4 |
| Selected verifier with peer association | 25 | 0 | 3 |

The selected verifier's extra unmatched prediction overlapped a reference already matched by the released box. Its IoU with that same reference was 0.88. The post-audit association reserves the strongest SAM peer for the existing box using the earlier crowd fusion's 0.40 association rule. On the same observed cohort, it removes that duplicate and retains all 20 recovered matches. The duplicate had concealed a remaining omission in the aggregate scene count. Across all 24 observed scenes, matched instances go from 33 to 53 of 81 and false positives remain at one. Exact counts rise from 10 to 13. These post-audit values measure the correction on already observed data. [Fixed comparison](../../../artifacts/detection_hillclimb/low_count_recovery_results.json) · [Association diagnostic](../../../artifacts/detection_hillclimb/low_count_alignment_diagnostic.json)

The prospective pilot freezes a new photograph of ten prefilled syringes and two pipette lookalikes before inference. Its author, source URL, exact bytes and transformed visual fingerprints were screened against the protected sources. The original held-back split for the verifier has no crowded scenes. This new pilot provides one crowded acquisition, with manual box annotations from one visual pass. The two negative photos were selected from the local Openverse collection and verified visually. The search did not reach their Flickr pages in the current session. Cached attribution is retained. [Frozen pilot](../../../artifacts/detection_hillclimb/low_count_prospective_reference.json) · [Epolk source and license](https://commons.wikimedia.org/wiki/File:Influenza_vaccine_syringes.jpg)

The released detector matched one of the ten syringes and counted zero objects on each pipette photo. The recovered branch still matched one syringe and added one false positive on each pipette photo. It fails the release criteria. The observational review runner keeps these suggestions separate from the verified count and rejects an observation if its source SHA-256 differs. No new automatic detection rule was installed.

The failure analysis found a SAM proposal overlapping nine of ten reference boxes using the general syringe prompt and ten using the prefilled-syringe prompt. Overlapping horizontal bands covered six. This is oracle proposal coverage at IoU 0.50, before pairing or acceptance. The laboratory-pipette prompt also overlapped all ten syringe references, while both pipette photographs receive general syringe detections. Prompt overlap is therefore insufficient evidence of device identity on this pilot. The observations diagnose verification errors and do not change the frozen rule. [Candidate-generation diagnostic](../../../artifacts/detection_hillclimb/low_count_candidate_diagnostic.json)

## Priority annotation and provenance

The visual audit checked seven multi-syringe fitting-role candidates and seven already admitted negative photos. It confirmed that the useful negatives include detached needles and equipment containers. The crowded photos contain overlapping clear barrels and orderly sets of three to twelve syringes. The two public project receipts confirm the project listings and sizes, but contain no item-level original photo credits. All seven crowded files remain pending source verification. No extra fitting photos were admitted. [Per-image review and receipts](../../../artifacts/detection_hillclimb/low_count_priority_review.json) · [Creative Commons attribution reference](https://creativecommons.org/reusing-cc-licensed-content/)

## Chamber appearance independent of OCR

An experimental color reader detects a coherent chromatic column inside a narrow barrel crop and requires an adjacent transverse dark band. The first version mislabeled the short colored stopper on the known empty 5 mL example. The revised geometry and saturation requirements reject the stopper and recover the yellow liquid on the public development photo in both directions. Its apparent lightness is 62.1 L*. Two of five known filled rendered controls emit a supported color. All five empty controls across the first public and rendered suites stay unknown for this reader, as does the rendered transparent liquid. The known empty cases have no automatic empty verdict from this branch. The small-field results retain original visual estimates and abstention by field.

A separate transfer uses seven manually oriented crops from one staged photo. It reports one red column with apparent lightness 54.4 L* and abstains on the other six. Manual axes remove part of the scene geometry problem, and the seven crops share a single setup. The measured coverage does not support deployment. Nominal capacity, numeric liquid amount and opacity remain independent fields. [Initial experiment](../../../artifacts/detection_hillclimb/chamber_appearance_initial.json) · [Revised pixel experiment](../../../artifacts/detection_hillclimb/chamber_appearance_results.json) · [Staged transfer](../../../artifacts/detection_hillclimb/chamber_public_transfer_results.json) · [Ivan Radic, CC BY 2.0](https://www.flickr.com/photos/26344495@N05/52609355361)

## Reproduce locally

~~~sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.low_count_recovery measure
.venv/bin/python -m scripts.low_count_review diagnose
data/sam3/xpu-env/bin/python -m scripts.low_count_prospective released
data/sam3/xpu-env/bin/python -m scripts.low_count_prospective measure
data/sam3/xpu-env/bin/python -m scripts.low_count_candidate_diagnostic
.venv/bin/python -m scripts.low_count_priority_review
.venv/bin/python -m scripts.chamber_appearance_study
.venv/bin/python -m scripts.chamber_public_transfer
.venv/bin/python -m pytest -q
~~~

For a local reviewed observation use the fixed policy's separate review export. The supplied observation must contain the original image checksum and the accepted object boxes. The result reports a verified count and offers review geometry without changing the source observation.

~~~sh
data/sam3/xpu-env/bin/python -m scripts.low_count_review review photograph.jpg \
    --observation observation.json --output recovery-review.json
~~~

Fil Rouge was not a fitting or augmentation parent in these experiments. The release has no newly deployed counting or appearance rule.
