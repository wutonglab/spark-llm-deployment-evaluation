"""Verify the agent's tool layer enforces sandboxing + redaction."""
from pathlib import Path


def test_write_artifact_blocks_path_escape(tmp_path: Path):
    from core.tools import ToolRegistry

    reg = ToolRegistry(run_dir=tmp_path)

    # Absolute path: rejected
    r = reg.invoke("write_artifact", {"name": "/tmp/leak.txt", "content": "x"})
    assert "error" in r

    # Parent traversal: rejected
    r = reg.invoke("write_artifact", {"name": "../leak.txt", "content": "x"})
    assert "error" in r

    # Relative path inside run_dir: allowed
    r = reg.invoke("write_artifact", {"name": "report.md", "content": "hello"})
    assert "path" in r
    assert (tmp_path / "report.md").read_text() == "hello"


def test_redaction_strips_secrets(tmp_path: Path):
    from core.tools import ToolRegistry, redact

    # Construct fake tokens at runtime so the source file doesn't contain a string
    # that triggers the repo's sanitization lint.
    fake_hf = "hf_" + ("a" * 20) + "BCDEFGHIJK1234567890"
    fake_sk = "sk-ant-" + ("a" * 30)
    secret_blob = (
        f"see {fake_hf} also "
        "Authorization: Bearer abc.def.ghi-jkl.mno=pq=rs "
        f"{fake_sk}"
    )
    red = redact(secret_blob)
    assert "***REDACTED***" in red
    assert fake_hf not in red
    assert fake_sk not in red

    # Redaction is applied to tool outputs too
    reg = ToolRegistry(run_dir=tmp_path, dry_run=True)
    r = reg.invoke(
        "run_shell",
        {"command": f"echo {fake_hf}"},
    )
    assert fake_hf not in str(r)


def test_unknown_tool_returns_error(tmp_path: Path):
    from core.tools import ToolRegistry

    reg = ToolRegistry(run_dir=tmp_path)
    r = reg.invoke("nonexistent_tool", {})
    assert "error" in r and "unknown tool" in r["error"]
