# Timed operator review protocol

Measure whether local model assistance makes syringe counting faster at the same or better final accuracy. This protocol is ready for an operator study. The results currently available describe machine inference and the reach of review proposals.

## Inputs and reference standard

Freeze the application release and its model hashes before testing. Use independently sourced native photos with empty, single-syringe and crowded scenes. Record specimen and acquisition groups when available. Two readers independently locate the visible syringe bodies without seeing model output. An adjudicator resolves disagreements before operators begin. Save both reads, the resolution and the original raster SHA-256. Keep the references out of the operator interface.

## Paired comparison

Use a common local machine and display for both conditions. In the manual condition, the operator views the original photo at native resolution and submits the count. In the assisted condition, the operator uses the ordinary upload, inspect and review workflow and submits the reviewed count. Record the revision and local report SHA-256 as evidence.

Assign matched acquisition groups to the two conditions and counterbalance their order across operators. Each operator sees a specific image once. Across operators, each image is evaluated in both conditions. Keep photographic bursts and near duplicates together within an operator's assignment. Report the allocation seed, group assignments, and any excluded session.

## Outcomes

The primary outcomes are final exact-count rate and elapsed time from showing the task to submitting a final count. Record model waiting time separately in the assisted condition. Record absolute count error, missed instances, unnecessary additions, revisions, abandoned tasks and the number of candidate inspections. Use the same completion instructions and time allowance for both conditions. Record unavailable inference stages as part of the assisted workflow.

Report paired differences and uncertainty by operator and acquisition group. Use group-level resampling only when there are enough independent groups. State the full sample size, including how many groups and operators contributed to each condition. Publish the raw anonymized timing records, frozen instructions and aggregate calculation. Store the source rasters and review exports locally.
