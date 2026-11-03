#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GEN_TRADE_TOOLS_DIR="${GEN_TRADE_TOOLS_DIR:-$project_root/.tools}"
export CARGO_HOME="${CARGO_HOME:-$GEN_TRADE_TOOLS_DIR/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$GEN_TRADE_TOOLS_DIR/rustup}"
export RUSTUP_TOOLCHAIN=1.90.0
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GEN_TRADE_TOOLS_DIR/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$GEN_TRADE_TOOLS_DIR/python}"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$GEN_TRADE_TOOLS_DIR" "$UV_CACHE_DIR"
if ! command -v uv >/dev/null; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
cd "$project_root"
# uv finds an existing interpreter or downloads the pinned Python with verification.
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$project_root/.venv" \
    uv sync --locked --all-extras --python "$(cat .python-version)"
if ! command -v rustup >/dev/null; then
    curl --fail --show-error --silent --location https://sh.rustup.rs \
        --output "$GEN_TRADE_TOOLS_DIR/rustup-init.sh"
    sh "$GEN_TRADE_TOOLS_DIR/rustup-init.sh" -y --no-modify-path \
        --profile minimal --default-toolchain 1.90.0 --component rustfmt,clippy
fi
rustup toolchain install 1.90.0 --profile minimal --component rustfmt,clippy
for crate in "$project_root/native" "$project_root/environment/native-probe"; do
    cargo fetch --locked --manifest-path "$crate/Cargo.toml"
    (
        cd "$crate"
        cargo fmt --check
        "$project_root/.venv/bin/maturin" build --release --locked \
            --interpreter "$project_root/.venv/bin/python" --out "$GEN_TRADE_TOOLS_DIR/wheels"
    )
done
uv pip install --python "$project_root/.venv/bin/python" --no-deps --reinstall \
    "$GEN_TRADE_TOOLS_DIR"/wheels/gen_trade_native-*.whl \
    "$GEN_TRADE_TOOLS_DIR"/wheels/gen_trade_env_native-*.whl
# Remember reused nonsecret tool paths for new shells; never save process env/secrets.
printf 'GEN_TRADE_SAVED_CARGO_HOME=%q\nGEN_TRADE_SAVED_RUSTUP_HOME=%q\n' \
    "$CARGO_HOME" "$RUSTUP_HOME" > "$GEN_TRADE_TOOLS_DIR/toolchain-paths.sh"
. "$project_root/scripts/activate.sh"
uv pip check --python "$project_root/.venv/bin/python"
python "$project_root/environment/verify.py"
echo 'Environment installed. Source scripts/activate.sh; start services with scripts/services.sh up -d --wait.'
