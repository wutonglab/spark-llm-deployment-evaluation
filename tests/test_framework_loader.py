"""Verify framework markdown files load + are non-empty."""
from pathlib import Path


def test_all_frameworks_exist_and_nonempty():
    repo_root = Path(__file__).resolve().parent.parent
    frameworks_dir = repo_root / "frameworks"
    required = [
        "01-fp8-vs-nvfp4-decision",
        "02-new-model-selection",
        "03-inference-research-channels",
        "04-new-model-evaluation-channels",
        "05-mtp-speculative-decoding",
        "how-to-use-these-frameworks",
    ]
    for name in required:
        f = frameworks_dir / f"{name}.md"
        assert f.is_file(), f"missing {f}"
        text = f.read_text(encoding="utf-8")
        assert len(text) > 500, f"{f} unexpectedly short: {len(text)} bytes"


def test_framework_loader_loads():
    from core.framework_loader import load_framework, load_prompt, load_tool_schemas

    fw01 = load_framework("01-fp8-vs-nvfp4-decision")
    assert "FP8" in fw01 and "NVFP4" in fw01

    sys_prompt = load_prompt("system_prompt")
    assert "LLM Deployment Evaluation Agent" in sys_prompt

    tools = load_tool_schemas()
    assert isinstance(tools, list) and len(tools) >= 4
    names = {t["name"] for t in tools}
    for required_tool in ("run_shell", "http_get", "read_file", "write_artifact"):
        assert required_tool in names, f"missing tool {required_tool}"
