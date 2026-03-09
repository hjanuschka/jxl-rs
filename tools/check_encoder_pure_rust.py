#!/usr/bin/env python3
"""Checks that the selected encoder dependency graph stays pure Rust.

The script fails if any reachable crate in the selected graph uses Cargo `links`
metadata (typical marker for native libraries / -sys crates).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path


def run_metadata(manifest_path: Path, features: str) -> dict:
    cmd = [
        "cargo",
        "metadata",
        "--format-version=1",
        "--manifest-path",
        str(manifest_path),
        "--no-default-features",
        "--features",
        features,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def find_root_package_id(metadata: dict, package_name: str) -> str:
    matches = [p for p in metadata["packages"] if p["name"] == package_name]
    if not matches:
        raise RuntimeError(f"Package not found in metadata: {package_name}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Package name is ambiguous in metadata: {package_name}. "
            f"Matches: {[m['id'] for m in matches]}"
        )
    return matches[0]["id"]


def reachable_packages(metadata: dict, root_id: str) -> set[str]:
    resolve = metadata.get("resolve")
    if resolve is None:
        raise RuntimeError("Cargo metadata did not include resolve graph")

    edges: dict[str, list[str]] = {}
    for node in resolve["nodes"]:
        edges[node["id"]] = [dep["pkg"] for dep in node.get("deps", [])]

    seen: set[str] = set()
    queue: deque[str] = deque([root_id])

    while queue:
        pkg_id = queue.popleft()
        if pkg_id in seen:
            continue
        seen.add(pkg_id)
        for nxt in edges.get(pkg_id, []):
            if nxt not in seen:
                queue.append(nxt)

    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path",
        default="jxl/Cargo.toml",
        help="Manifest to evaluate (default: jxl/Cargo.toml)",
    )
    parser.add_argument(
        "--package",
        default="jxl",
        help="Root package name in metadata graph (default: jxl)",
    )
    parser.add_argument(
        "--features",
        default="encoder",
        help="Feature set to check (default: encoder)",
    )
    args = parser.parse_args()

    metadata = run_metadata(Path(args.manifest_path), args.features)
    root_id = find_root_package_id(metadata, args.package)
    reachable = reachable_packages(metadata, root_id)

    packages_by_id = {p["id"]: p for p in metadata["packages"]}

    # Some pure-Rust crates use `links` for global symbol uniqueness / metadata,
    # not native library binding. Keep this list very small and explicit.
    allowed_links = {"rayon-core", "wasm_bindgen"}

    offenders = []
    for pkg_id in sorted(reachable):
        pkg = packages_by_id[pkg_id]
        links = pkg.get("links")
        if links and links not in allowed_links:
            offenders.append((pkg["name"], pkg["version"], links))

    if offenders:
        print("FAIL: native-linked crates found in encoder dependency graph:")
        for name, version, links in offenders:
            print(f"  - {name} {version} (links={links})")
        return 1

    print(
        f"OK: pure Rust check passed for package={args.package} "
        f"features={args.features}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
