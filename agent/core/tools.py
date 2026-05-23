"""Tool registry for the agent.

Each tool function takes a dict of arguments and returns a dict result. Tools
are dispatched by name from ToolRegistry.invoke().

Safety:
- File writes restricted to run_dir/ (validated by write_artifact)
- Shell timeout default 600s
- Credentials redacted from output captured into transcript
"""
from __future__ import annotations

import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

# Patterns of values that must never reach the transcript or report.
_REDACTION_PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf_***REDACTED***"),
    (re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9\-_]{20,}"), "sk_***REDACTED***"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-_.=]+", re.IGNORECASE), "Bearer ***REDACTED***"),
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----"), "***REDACTED-PRIVATE-KEY***"),
]


def redact(s: str) -> str:
    for pat, repl in _REDACTION_PATTERNS:
        s = pat.sub(repl, s)
    return s


class ToolRegistry:
    """Holds tools + dispatches calls."""

    def __init__(self, run_dir: Path, target_host: str | None = None, dry_run: bool = False):
        self.run_dir = run_dir
        self.target_host = target_host
        self.dry_run = dry_run
        # Eager-register all built-in tools
        self._tools: dict[str, Any] = {
            "run_shell": self._run_shell,
            "http_get": self._http_get,
            "read_file": self._read_file,
            "write_artifact": self._write_artifact,
            "check_gpu": self._check_gpu,
            "check_docker": self._check_docker,
            "fetch_url": self._http_get,  # alias
        }

    # ---- Dispatcher ----

    def invoke(self, name: str, arguments: dict) -> dict:
        if name not in self._tools:
            return {"error": f"unknown tool: {name}"}
        try:
            result = self._tools[name](**(arguments or {}))
        except TypeError as e:
            return {"error": f"bad arguments to {name}: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        # Redact strings recursively before returning.
        return _redact_in(result)

    # ---- Tools ----

    def _run_shell(self, command: str, timeout: int = 600) -> dict:
        if self.dry_run:
            return {"dry_run": True, "command": command}
        if self.target_host:
            full = ["ssh", "-o", "StrictHostKeyChecking=no", self.target_host, command]
        else:
            full = ["bash", "-lc", command]
        try:
            r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
            return {
                "rc": r.returncode,
                "stdout": r.stdout[-8000:],  # cap to avoid runaway
                "stderr": r.stderr[-4000:],
            }
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s", "command": command}

    def _http_get(self, url: str, timeout: int = 30) -> dict:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spark-llm-deployment-evaluation-agent/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read(1024 * 1024)  # cap at 1 MiB
                return {
                    "status": resp.status,
                    "url": resp.geturl(),
                    "body": body.decode("utf-8", errors="replace"),
                }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "url": url}

    def _read_file(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.is_file():
            return {"error": f"not a file: {path}"}
        try:
            return {"path": str(p), "content": p.read_text(encoding="utf-8", errors="replace")[:50_000]}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _write_artifact(self, name: str, content: str) -> dict:
        # Restrict writes to run_dir. Name is treated as a relative path.
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            return {"error": "artifact name must be a relative path without '..'"}
        target = (self.run_dir / rel).resolve()
        if not str(target).startswith(str(self.run_dir.resolve())):
            return {"error": "artifact path escapes run_dir"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes": len(content.encode("utf-8"))}

    def _check_gpu(self) -> dict:
        cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
        r = self._run_shell(cmd, timeout=30)
        return r

    def _check_docker(self) -> dict:
        return self._run_shell("docker info --format '{{json .}}' | head -1 && docker ps --format '{{.Names}}\\t{{.Status}}'", timeout=30)


def _redact_in(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: _redact_in(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_in(x) for x in obj]
    return obj
