#!/usr/bin/env bash
# Activate myenv; run manually after training for a clean speed measurement.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python -m poi_recognizer.evaluate "$@"
