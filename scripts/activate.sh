# Source this file (bash or zsh) from any directory after setup-environment.sh:
#   source scripts/activate.sh
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _gen_trade_script="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    # zsh has no BASH_SOURCE; %x expands to the file being sourced.
    eval '_gen_trade_script="${(%):-%x}"'
else
    _gen_trade_script="$0"
fi
case "${_gen_trade_script##*/}" in
    activate.sh) ;;
    *) echo "scripts/activate.sh: cannot locate itself; run 'source scripts/activate.sh' from bash or zsh" >&2
       unset _gen_trade_script; return 1 2>/dev/null || exit 1 ;;
esac
if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "scripts/activate.sh must be sourced, not executed: source scripts/activate.sh" >&2
    exit 1
fi
export GEN_TRADE_ROOT="$(cd -- "$(dirname -- "$_gen_trade_script")/.." && pwd)"
unset _gen_trade_script
if [ ! -x "$GEN_TRADE_ROOT/.venv/bin/python" ]; then
    echo "No virtual environment at $GEN_TRADE_ROOT/.venv. Create it first:" >&2
    echo "  bash scripts/setup-environment.sh --market-data   # or --core / --full" >&2
    return 1 2>/dev/null || exit 1
fi
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
case ":$PATH:" in
    *":$CARGO_HOME/bin:"*) ;;
    *) export PATH="$CARGO_HOME/bin:$PATH" ;;
esac
. "$GEN_TRADE_ROOT/.venv/bin/activate"
