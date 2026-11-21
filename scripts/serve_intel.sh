#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MEDSYRINGE_PRIMARY_DEVICE=xpu
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec data/sam3/xpu-env/bin/python -m uvicorn medsyringe.app:app --host 127.0.0.1 --port "${1:-8765}"
