#!/usr/bin/env bash
# docker compose wrapper for the local PostgreSQL/Redis services.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v docker >/dev/null; then
    cat >&2 <<'EOF'
Docker is not installed, so the optional PostgreSQL/Redis services cannot start.
They are only needed for --persist and the Redis cache; backtests run without them.
Install a Docker engine with Compose v2 (Docker Desktop, OrbStack or colima on macOS), then rerun.
EOF
    exit 127
fi
if [ -f /etc/codex/network-policy.json ]; then
    # Use the managed local daemon, preserving Docker configuration and credentials.
    exec env -u DOCKER_HOST -u DOCKER_CONTEXT -u DOCKER_TLS -u DOCKER_TLS_VERIFY -u DOCKER_CERT_PATH \
        docker --host=unix:///var/run/docker.sock compose \
        --project-directory "$project_root" --file "$project_root/compose.yaml" "$@"
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is missing ('docker compose'); install the Compose plugin and rerun." >&2
    exit 127
fi
if ! docker info >/dev/null 2>&1; then
    echo "The Docker daemon is not running (or this user cannot reach it). Start Docker and rerun." >&2
    exit 1
fi
exec docker compose --project-directory "$project_root" --file "$project_root/compose.yaml" "$@"
