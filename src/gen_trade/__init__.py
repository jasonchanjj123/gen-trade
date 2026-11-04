"""Research implementations; see docs/PAPER_METHODS.md for fidelity limitations."""

import os
from pathlib import Path
import tempfile


def _configure_runtime_cache():
    root=Path(__file__).resolve().parents[2]
    default=root/'.tools/cache' if (root/'pyproject.toml').exists() else Path(tempfile.gettempdir())/'gen-trade-cache'
    cache=Path(os.environ.get('GEN_TRADE_RUNTIME_CACHE_DIR',default))
    os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
    for path in (os.environ['XDG_CACHE_HOME'],os.environ['MPLCONFIGDIR']):
        Path(path).mkdir(parents=True,exist_ok=True)


_configure_runtime_cache()

__version__ = "0.4.0"
