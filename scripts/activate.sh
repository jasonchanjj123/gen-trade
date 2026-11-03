# Source this Bash file from any directory after setup-environment.sh.
export GEN_TRADE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GEN_TRADE_TOOLS_DIR="${GEN_TRADE_TOOLS_DIR:-$GEN_TRADE_ROOT/.tools}"
if [ -f "$GEN_TRADE_TOOLS_DIR/toolchain-paths.sh" ]; then
    . "$GEN_TRADE_TOOLS_DIR/toolchain-paths.sh"
fi
export CARGO_HOME="${CARGO_HOME:-${GEN_TRADE_SAVED_CARGO_HOME:-$GEN_TRADE_TOOLS_DIR/cargo}}"
export RUSTUP_HOME="${RUSTUP_HOME:-${GEN_TRADE_SAVED_RUSTUP_HOME:-$GEN_TRADE_TOOLS_DIR/rustup}}"
export RUSTUP_TOOLCHAIN=1.90.0
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-3}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-3}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$GEN_TRADE_TOOLS_DIR/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GEN_TRADE_TOOLS_DIR/cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$GEN_TRADE_TOOLS_DIR/cache/matplotlib}"
export DATABASE_URL="${DATABASE_URL:-postgresql://gen_trade@127.0.0.1:${GEN_TRADE_PG_PORT:-55432}/gen_trade}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:${GEN_TRADE_REDIS_PORT:-56379}/0}"
export PATH="$CARGO_HOME/bin:$PATH"
. "$GEN_TRADE_ROOT/.venv/bin/activate"
