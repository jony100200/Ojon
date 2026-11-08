#!/usr/bin/env python3
"""Enforce Ojon's compact public Markdown file set."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ALLOWED = {
    ".github/pull_request_template.md",
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "QUALITY_STRATEGY.md",
    "README.md",
    "USER_GUIDE.md",
    "docs/MINIMAX_H3.md",
    "docs/RECIPE_SEARCH.md",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    names = [entry.decode("utf-8") for entry in tracked.split(b"\0") if entry]
    unexpected = sorted(name for name in names if name.lower().endswith((".md", ".markdown", ".mdx")) and name not in ALLOWED)
    if unexpected:
        print("Unexpected public documentation files:")
        for name in unexpected:
            print("  " + name)
        print("A new public document requires explicit owner approval and allowlist update.")
        return 1

    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    opening = readme[:1200]
    if "quantiz" not in opening or "uncensor" not in opening:
        print("README must lead with Ojon's two core product jobs: quantize and uncensor.")
        return 1

    print("Public documentation check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
