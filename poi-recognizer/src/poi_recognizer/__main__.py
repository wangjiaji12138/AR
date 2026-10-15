"""Allow python -m poi_recognizer after package installation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
