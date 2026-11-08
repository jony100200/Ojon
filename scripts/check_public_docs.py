#!/usr/bin/env python3
"""Keep the public Ojon source tree free of private development material."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PUBLIC_DOCS = {"README.md", "USER_GUIDE.md", "docs/MINIMAX_H3.md"}
PRIVATE_ROOTS = {
    "_local", "_private", "development", "research", "experiments",
    "session-logs", "logs", "_logs", "scratch", "_scratch",
    "outputs", "output", "models", "weights", "checkpoints", "downloads",
    ".agents", ".claude", ".codex",
}
PRIVATE_EXTENSIONS = {
    ".log", ".jsonl", ".sqlite", ".sqlite3", ".db", ".safetensors",
    ".gguf", ".ggml", ".ckpt", ".pth", ".onnx",
}
PRIVATE_NAMES = {
    "agents.md", "claude.md", "codex.md", "gemini.md",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    names = [s.decode("utf-8") for s in tracked.split(b"\0") if s]
    rejected = []
    for name in names:
        path = Path(name)
        components = [part.lower() for part in path.parts]
        if name.lower().endswith((".md", ".markdown", ".mdx")) and name not in PUBLIC_DOCS:
            rejected.append(name)
        elif components[0] in PRIVATE_ROOTS or any(part in {".agents", ".claude", ".codex"} for part in components):
            rejected.append(name)
        elif path.name.lower() in PRIVATE_NAMES or path.suffix.lower() in PRIVATE_EXTENSIONS:
            rejected.append(name)

    if rejected:
        print("Private or unapproved files tracked in the public repository:")
        for name in sorted(set(rejected)):
            print("  " + name)
        return 1

    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    if "quantiz" not in readme[:700] or "uncensor" not in readme[:700]:
        print("README must introduce the two product functions: quantize and uncensor.")
        return 1

    print("Ojon public repository boundary: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
