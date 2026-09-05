#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Consistent NupicAI SQLite backup")
    parser.add_argument("--database", type=Path, default=Path("runtime/nupicai.sqlite3"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=14)
    args = parser.parse_args()
    source = args.database.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Database not found: {source}")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"nupicai-{time.strftime('%Y%m%d-%H%M%S')}.sqlite3"
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
    destination.chmod(0o600)
    backups = sorted(out_dir.glob("nupicai-*.sqlite3"), key=lambda path: path.stat().st_mtime)
    for old in backups[:-max(1, args.keep)]:
        old.unlink()
    print(destination)


if __name__ == "__main__":
    main()
