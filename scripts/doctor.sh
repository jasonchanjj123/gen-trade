#!/usr/bin/env bash
# Report the state of this checkout's environment; changes nothing.
#   bash scripts/doctor.sh [--json]
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer the project interpreter; fall back to any python3 so a broken .venv can be diagnosed.
for candidate in "$project_root/.venv/bin/python" python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
        exec "$candidate" "$project_root/environment/doctor.py" "$@"
    fi
done
echo "No working Python 3.9+ found (.venv or python3). Install uv, then: bash scripts/setup-environment.sh" >&2
exit 1
