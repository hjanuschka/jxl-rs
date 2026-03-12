#!/usr/bin/env python3
"""Render a tiny HTML dashboard from rate_distortion_compare CSV output."""

from __future__ import annotations

import csv
import html
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: rd_dashboard.py <input.csv> <output.html>")
        return 1

    in_csv = Path(sys.argv[1])
    out_html = Path(sys.argv[2])

    rows = []
    with in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    headers = [
        "image",
        "jxlrs_bytes",
        "libjxl_bytes",
        "size_pct",
        "jxlrs_psnr",
        "libjxl_psnr",
        "psnr_gap",
    ]

    def td(v: str) -> str:
        return f"<td>{html.escape(v)}</td>"

    body = []
    for r in rows:
        body.append("<tr>" + "".join(td(r.get(h, "")) for h in headers) + "</tr>")

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Encoder R/D dashboard</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
    th {{ background: #f2f2f2; text-align: left; }}
  </style>
</head>
<body>
  <h1>Encoder R/D dashboard</h1>
  <p>Generated from {html.escape(in_csv.name)}</p>
  <table>
    <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
    <tbody>
      {''.join(body)}
    </tbody>
  </table>
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
