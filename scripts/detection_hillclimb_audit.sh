#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export TORCH_CPP_LOG_LEVEL=ERROR

if [[ $# -ne 1 || ( "$1" != public-audit && "$1" != previous-reserve ) ]]; then
  printf 'Usage: %s public-audit|previous-reserve\n' "$0" >&2
  exit 2
fi
scope="$1"
models=$(.venv/bin/python - <<'PY'
from scripts.detection_hillclimb_rank import MODEL_COMBINATIONS
from scripts.detection_hillclimb_experiment import SELECTED
from medsyringe.data import read_json
freeze=read_json(SELECTED)
policies=[freeze["shadow"]]
if freeze.get("review_selected"):
    policies.append(freeze["review_selected"])
models=set()
for item in policies:
    model=item["policy"]["model"]
    models.update(MODEL_COMBINATIONS.get(model,(model,)))
print("\n".join(sorted(models)))
PY
)

data/sam3/xpu-env/bin/python -m scripts.detection_hillclimb_experiment collect \
  --scope "$scope" --backend baseline --device xpu
while IFS= read -r model; do
  case "$model" in
    dfine) .venv/bin/python -m scripts.detection_hillclimb_dfine collect --scope "$scope" ;;
    dfine-multidomain) .venv/bin/python -m scripts.detection_hillclimb_mix collect --scope "$scope" ;;
    dino-base|owlv2)
      data/sam3/xpu-env/bin/python -m scripts.detection_open_models --scope "$scope" --backend "$model" --device xpu ;;
    original-rf|adapted-rf)
      data/rfdetr/env/bin/python -m scripts.detection_hillclimb_experiment collect --scope "$scope" --backend "$model" ;;
    *) printf 'Unknown frozen model: %s\n' "$model" >&2; exit 2 ;;
  esac
done <<< "$models"
