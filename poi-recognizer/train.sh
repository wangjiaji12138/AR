#!/usr/bin/env bash
# Activate the project Conda environment before invoking this script.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python -m poi_recognizer "$@"
