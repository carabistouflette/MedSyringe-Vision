#!/usr/bin/env bash
set -euo pipefail
exec data/ocr-v6/env/bin/python -u -m scripts.workers.scale_worker
