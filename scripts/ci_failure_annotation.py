"""Expose the tail of a failed CI log through a readable Checks annotation."""
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/ci_failure_annotation.py <log-file>")
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")[-12000:]
    escaped = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    payload = f"::error::{escaped}\n".encode("ascii", "backslashreplace")
    sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    main()
