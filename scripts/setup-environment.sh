#!/usr/bin/env bash
# Install or repair the locked environment. Safe to rerun: it reuses what is
# already installed, keeps a log, and verifies the result before succeeding.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup-environment.sh [PROFILE] [--native|--no-native]

Profiles (default: the profile of the previous setup, else --full):
  --core         numerical stack, tests and storage clients
  --market-data  core + exchange calendars and plotting (stock/FX benchmark)
  --full         every research extra, both Rust modules; services verified separately

Options:
  --native       also build the Rust directional-change backend for --core/--market-data
                 (same results as the Python fallback, modestly faster; needs a C compiler)
  --no-native    skip it (Python fallback). Light profiles remember the last choice.
  -h, --help     show this help

Logs go to .tools/logs/. Check the result any time with: bash scripts/doctor.sh
EOF
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export GEN_TRADE_TOOLS_DIR="${GEN_TRADE_TOOLS_DIR:-$project_root/.tools}"
state_file="$GEN_TRADE_TOOLS_DIR/environment.json"

previous() {  # previous KEY -> value recorded by the last successful setup
    [ -f "$state_file" ] || return 0
    sed -n "s/^ *\"$1\": *\"\{0,1\}\([^\",]*\)\"\{0,1\},\{0,1\}$/\1/p" "$state_file" | head -n 1
}

profile="" native=""
for argument in "$@"; do
    case "$argument" in
        --core|core) profile=core ;;
        --market-data|market-data) profile=market-data ;;
        --full|full) profile=full ;;
        --native) native=yes ;;
        --no-native) native=no ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $argument" >&2; usage >&2; exit 2 ;;
    esac
done
profile_note=""
if [ -z "$profile" ]; then
    profile="$(previous profile)"
    if [ -n "$profile" ]; then
        profile_note=" (from the previous setup; pass --core, --market-data or --full to change)"
    else
        profile=full
    fi
fi
if [ "$profile" = full ]; then
    [ "$native" = no ] && { echo "--no-native is not available with --full (its verifier needs both Rust modules)" >&2; exit 2; }
    native=yes
elif [ -z "$native" ]; then
    native="$(previous native)"
    [ "$native" = yes ] && profile_note="$profile_note; Rust backend kept from the previous setup (--no-native drops it)"
    native="${native:-no}"
fi

if [ -z "${GEN_TRADE_SETUP_LOG:-}" ]; then
    # Rerun this script with all output copied to a log; waiting on the pipe
    # keeps the log output ahead of the shell prompt.
    mkdir -p "$GEN_TRADE_TOOLS_DIR/logs"
    export GEN_TRADE_SETUP_LOG="$GEN_TRADE_TOOLS_DIR/logs/setup-$(date +%Y%m%d-%H%M%S).log"
    set +e
    bash "${BASH_SOURCE[0]}" "$@" 2>&1 | tee -a "$GEN_TRADE_SETUP_LOG"
    exit "${PIPESTATUS[0]}"
fi
log_file="$GEN_TRADE_SETUP_LOG"

current_step="starting"
step() {
    current_step="$1"
    printf '\n==> %s\n' "$1"
}
hint=""
on_error() {
    local code=$?
    printf '\nSetup FAILED (exit %s) while: %s\n' "$code" "$current_step" >&2
    [ -n "$hint" ] && printf 'Hint: %s\n' "$hint" >&2
    printf 'Full log: %s\n' "$log_file" >&2
    printf 'After fixing the cause, rerun the same command; finished steps are reused.\n' >&2
}
trap on_error ERR

echo "gen-trade setup · profile: $profile$profile_note · Rust backend: $native"
echo "Log: $log_file"

step "Checking prerequisites"
if ! command -v uv >/dev/null; then
    echo 'uv is required: https://docs.astral.sh/uv/getting-started/installation/' >&2
    echo '  curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
    current_step="checking for uv"; false
fi
echo "uv: $(uv --version)"
pinned_python="$(cat .python-version)"
echo "Python pin: $pinned_python (uv downloads it if absent)"
if [ "$native" = yes ]; then
    if ! command -v cc >/dev/null; then
        hint="Install a C compiler/linker. macOS: xcode-select --install; Debian/Ubuntu: apt install build-essential."
        echo "No C compiler (cc) found; the Rust build needs one to link." >&2
        false
    fi
    echo "C compiler: $(command -v cc)"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GEN_TRADE_TOOLS_DIR/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$GEN_TRADE_TOOLS_DIR/python}"
mkdir -p "$UV_CACHE_DIR"

case "$profile" in
    core) python_extras=(--extra dev --extra storage) ;;
    market-data) python_extras=(--extra dev --extra storage --extra market-data) ;;
    full) python_extras=(--all-extras) ;;
esac
step "Installing locked Python $pinned_python packages (${python_extras[*]})"
hint="If uv reports the lockfile format is unsupported, update uv (uv self update) and rerun. Network or proxy errors: rerun once the connection works; downloads are cached."
# uv finds an existing interpreter or downloads the pinned Python with verification.
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$project_root/.venv" \
    uv sync --locked "${python_extras[@]}" --python "$pinned_python"
python="$project_root/.venv/bin/python"
hint=""

rust_source=""
if [ "$native" = yes ]; then
    step "Preparing Rust 1.90.0"
    if [ -z "${CARGO_HOME:-}${RUSTUP_HOME:-}" ] && command -v rustup >/dev/null; then
        # Reuse this machine's rustup rather than downloading a second toolchain into .tools.
        RUSTUP_HOME="$(rustup show home)"
        CARGO_HOME="$HOME/.cargo"
        rust_source="existing rustup ($(command -v rustup))"
    else
        CARGO_HOME="${CARGO_HOME:-$GEN_TRADE_TOOLS_DIR/cargo}"
        RUSTUP_HOME="${RUSTUP_HOME:-$GEN_TRADE_TOOLS_DIR/rustup}"
        rust_source="$RUSTUP_HOME"
    fi
    export CARGO_HOME RUSTUP_HOME
    export RUSTUP_TOOLCHAIN=1.90.0
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
    export PATH="$CARGO_HOME/bin:$PATH"
    if ! command -v rustup >/dev/null; then
        hint="rustup download failed; check the network/proxy, or install rustup yourself (https://rustup.rs) and rerun."
        curl --fail --show-error --silent --location --retry 3 https://sh.rustup.rs \
            --output "$GEN_TRADE_TOOLS_DIR/rustup-init.sh"
        sh "$GEN_TRADE_TOOLS_DIR/rustup-init.sh" -y --no-modify-path \
            --profile minimal --default-toolchain 1.90.0 --component rustfmt,clippy
        rust_source="$RUSTUP_HOME"
    fi
    rustup toolchain install 1.90.0 --profile minimal --component rustfmt,clippy
    echo "Rust: $(rustc --version) from $rust_source"
    hint=""

    crates=("$project_root/native")
    [ "$profile" = full ] && crates+=("$project_root/environment/native-probe")
    rm -f "$GEN_TRADE_TOOLS_DIR"/wheels/gen_trade_native-*.whl "$GEN_TRADE_TOOLS_DIR"/wheels/gen_trade_env_native-*.whl
    for crate in "${crates[@]}"; do
        step "Building $(basename "$crate") (release)"
        cargo fetch --locked --manifest-path "$crate/Cargo.toml"
        (
            cd "$crate"
            cargo fmt --check
            "$project_root/.venv/bin/maturin" build --release --locked \
                --interpreter "$python" --out "$GEN_TRADE_TOOLS_DIR/wheels"
        )
    done
    step "Installing the native wheel(s)"
    # uv sync removes undeclared packages, so the wheels are reinstalled after every sync.
    uv pip install --python "$python" --no-deps --reinstall "$GEN_TRADE_TOOLS_DIR"/wheels/*.whl
    # Remember reused nonsecret tool paths for new shells; never save process env/secrets.
    printf 'GEN_TRADE_SAVED_CARGO_HOME=%q\nGEN_TRADE_SAVED_RUSTUP_HOME=%q\n' \
        "$CARGO_HOME" "$RUSTUP_HOME" > "$GEN_TRADE_TOOLS_DIR/toolchain-paths.sh"
fi

step "Verifying the installation"
uv pip check --python "$python"
if [ "$profile" = full ]; then
    . "$project_root/scripts/activate.sh"
    python "$project_root/environment/verify.py"
else
    "$python" "$project_root/environment/verify.py" --core
fi

rust_version="none"
[ "$native" = yes ] && rust_version="$(rustc --version | awk '{print $2}')"
cat > "$state_file.tmp" <<EOF
{
  "profile": "$profile",
  "native": "$native",
  "python": "$("$python" -c 'import platform; print(platform.python_version())')",
  "uv": "$(uv --version | awk '{print $2}')",
  "rust": "$rust_version",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "log": "$log_file"
}
EOF
mv "$state_file.tmp" "$state_file"

current_step="finished"
printf '\nEnvironment ready (profile %s, Rust backend %s).\n' "$profile" "$native"
echo "Activate it in each new shell (bash or zsh):  source scripts/activate.sh"
echo "Check it any time:                           bash scripts/doctor.sh"
[ "$profile" = full ] && echo "Local PostgreSQL/Redis (needs Docker):       bash scripts/services.sh up -d --wait --wait-timeout 60"
exit 0
