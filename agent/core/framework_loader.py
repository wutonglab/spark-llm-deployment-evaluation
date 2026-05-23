"""Load framework + prompt markdown into agent context."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRAMEWORKS_DIR = REPO_ROOT / "frameworks"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_framework(name: str) -> str:
    """Load a framework markdown by stem (e.g. '01-fp8-vs-nvfp4-decision')."""
    f = FRAMEWORKS_DIR / f"{name}.md"
    if not f.is_file():
        raise FileNotFoundError(f"framework not found: {f}")
    return f.read_text(encoding="utf-8")


def load_prompt(name: str) -> str:
    """Load a prompt markdown by stem (e.g. 'system_prompt', 'stage-1-applying-selection')."""
    f = PROMPTS_DIR / f"{name}.md"
    if not f.is_file():
        raise FileNotFoundError(f"prompt not found: {f}")
    return f.read_text(encoding="utf-8")


def load_tool_schemas() -> list[dict]:
    """Load the tool schema JSON and return the list of tool definitions."""
    import json

    f = Path(__file__).resolve().parent.parent / "schemas" / "tool_schemas.json"
    return json.loads(f.read_text(encoding="utf-8"))["tools"]
