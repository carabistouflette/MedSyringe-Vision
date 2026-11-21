#!/usr/bin/env bash
set -euo pipefail
exec data/sam3/xpu-env/bin/python -u -m scripts.shared_sam_worker
