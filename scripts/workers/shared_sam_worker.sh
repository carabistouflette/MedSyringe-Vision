#!/usr/bin/env bash
set -euo pipefail
exec data/sam3/xpu-env/bin/python -u -m scripts.workers.shared_sam_worker
