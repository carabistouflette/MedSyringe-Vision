#!/usr/bin/env bash
set -euo pipefail
exec .venv/bin/python -u -m scripts.workers.appearance_worker
