#!/usr/bin/env python3
"""Patch installed OpenNMT model_builder to pass weights_only=False in torch.load.

This script only edits site-packages file in the active Python environment.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys


def main() -> int:
    try:
        mod = importlib.import_module("onmt.model_builder")
    except Exception as exc:
        print(f"[error] failed to import onmt.model_builder: {exc}")
        return 2

    path = pathlib.Path(mod.__file__).resolve()
    text = path.read_text(encoding="utf-8")

    if "weights_only=False" in text:
        print(f"[ok] already patched: {path}")
        return 0

    pattern = re.compile(
        r"checkpoint\s*=\s*torch\.load\(model_path,\s*\n\s*map_location=lambda storage, loc: storage\)",
        re.MULTILINE,
    )
    replacement = (
        "checkpoint = torch.load(model_path,\n"
        "                            map_location=lambda storage, loc: storage,\n"
        "                            weights_only=False)"
    )

    new_text, n = pattern.subn(replacement, text, count=1)
    if n == 0:
        print(f"[error] patch pattern not found in: {path}")
        return 3

    path.write_text(new_text, encoding="utf-8")
    print(f"[done] patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
