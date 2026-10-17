"""Run the ETA service.

    python -m eta_forge.server --config configs/eta_forge.yaml
    python -m eta_forge.server --synthetic --port 8080          # no database

``uvicorn`` is imported here, not at package import time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import load_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETA Forge HTTP service")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="route on the in-memory synthetic grid instead of PostGIS")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config(args.config, overrides=args.overrides)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - environment dependent
        print("error: uvicorn is not installed; pip install 'uvicorn[standard]'", file=sys.stderr)
        return 2

    from .api.app import create_app
    from .api.deps import build_state

    state = build_state(cfg, synthetic=args.synthetic)
    app = create_app(state)
    logging.getLogger("eta_forge.server").info(
        "serving models %s on %s router", state.service.bundle.versions, state.backend
    )
    uvicorn.run(
        app,
        host=args.host or cfg.service.host,
        port=args.port or cfg.service.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
