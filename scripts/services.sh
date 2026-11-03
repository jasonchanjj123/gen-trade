#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f /etc/codex/network-policy.json ]; then
    # Use the managed local daemon, preserving Docker configuration and credentials.
    exec env -u DOCKER_HOST -u DOCKER_CONTEXT -u DOCKER_TLS -u DOCKER_TLS_VERIFY -u DOCKER_CERT_PATH \
        docker --host=unix:///var/run/docker.sock compose \
        --project-directory "$project_root" --file "$project_root/compose.yaml" "$@"
else
    exec docker compose --project-directory "$project_root" --file "$project_root/compose.yaml" "$@"
fi
