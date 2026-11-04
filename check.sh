#!/usr/bin/env bash
# Run the acceptance tests for one drill (or all of them).
#   ./check.sh t2               # test your mine/t2_*.py
#   ./check.sh l2 -x            # extra args go to pytest
#   ./check.sh all              # every drill
#   ./check.sh t2 --solutions   # run the reference solution instead
set -euo pipefail
cd "$(dirname "$0")"

id="${1:-all}"
shift || true
args=()
for a in "$@"; do
  case "$a" in
    --solutions) export DRILL_IMPL=solutions ;;
    *) args+=("$a") ;;
  esac
done

PY="${PYTHON:-python3}"
if [[ -x .venv/bin/python && -z "${PYTHON:-}" ]]; then PY=.venv/bin/python; fi

if [[ "$id" == "all" ]]; then
  targets=(tests)
else
  shopt -s nullglob
  targets=(tests/test_"${id}"_*.py)
  if (( ${#targets[@]} == 0 )); then
    echo "no test file matches tests/test_${id}_*.py" >&2
    exit 1
  fi
fi

exec "$PY" -m pytest "${targets[@]}" ${args[@]+"${args[@]}"}
