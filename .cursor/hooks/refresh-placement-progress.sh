#!/usr/bin/env bash
# After a logged placement experiment, refresh the markdown progress rollup.
set -euo pipefail
input=$(cat)
command=$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("command",""))')
if [[ "$command" != *"placement_experiment.py"* ]]; then
  exit 0
fi
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
uv run python scripts/placement_progress_report.py --hours 168 \
  --out results/experiments/latest_report.md >/dev/null 2>&1 || true
exit 0
