"""MkDocs hook: copy frameworks/ and case-studies/ into docs/ before build.

This lets us keep the framework + case study markdown at the repo root (where
they're naturally read from GitHub) while still having MkDocs render them as
part of the documentation site.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def on_files(files, config):
    docs_dir = Path(config["docs_dir"])

    for src in ["frameworks", "case-studies"]:
        src_path = REPO_ROOT / src
        dst_path = docs_dir / src
        if dst_path.exists():
            shutil.rmtree(dst_path)
        if src_path.exists():
            shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".DS_Store", "*.csv", "*.env"
            ))

    # Re-scan files after copy so MkDocs picks them up
    from mkdocs.structure.files import get_files
    return get_files(config)
