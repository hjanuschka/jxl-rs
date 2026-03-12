#!/usr/bin/env python3
"""Audit selected high-level encoder files for panic-prone calls outside tests."""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN = [".unwrap(", ".expect(", "panic!(", "assert!("]
FILES = [
    "jxl/src/encode/encoder.rs",
    "jxl/src/encode/input.rs",
    "jxl/src/encode/container.rs",
    "jxl/src/encode/options.rs",
]


def strip_test_module(text: str) -> str:
    marker = "#[cfg(test)]"
    i = text.find(marker)
    if i == -1:
        return text
    return text[:i]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for rel in FILES:
        p = root / rel
        s = strip_test_module(p.read_text(encoding="utf-8"))
        lines = s.splitlines()
        for idx, line in enumerate(lines, start=1):
            if line.strip().startswith("//"):
                continue
            for needle in FORBIDDEN:
                if needle in line:
                    failures.append(f"{rel}:{idx}: contains {needle}")

    if failures:
        print("encoder panic audit failed:")
        for f in failures:
            print(f)
        return 1

    print("encoder panic audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
