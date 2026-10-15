"""Launch unmodified PaddleOCR with a single output stream, without duplicate logs."""

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Initialize the official logger without its redundant per-batch file handler."""
    source = Path(sys.argv[1])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "tools"))
    from ppocr.utils.logging import get_logger

    get_logger()
    sys.argv = [str(source / "tools/train.py"), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
