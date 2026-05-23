"""Stage-by-stage orchestrator for the evaluation agent."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .framework_loader import load_prompt, load_tool_schemas
from .llm_client import ChatResponse, LLMClient
from .tools import ToolRegistry


@dataclass
class Stage:
    name: str
    prompt_file: str
    description: str
    expected_artifacts: list[str] = field(default_factory=list)


STAGES: list[Stage] = [
    Stage(
        name="stage-1-selection",
        prompt_file="stage-1-applying-selection",
        description="Apply Framework 02 (Model Selection)",
        expected_artifacts=["stage-1-selection-score.json"],
    ),
    Stage(
        name="stage-2-quantization",
        prompt_file="stage-2-channel-research",
        description="Apply Framework 01 + 05 (Quantization + Speculative)",
        expected_artifacts=["stage-2-quantization-score.json"],
    ),
    Stage(
        name="stage-3-config",
        prompt_file="stage-3-applying-quantization",
        description="Generate deployment configuration",
        expected_artifacts=["proposed-config.env", "proposed-launch-command.sh"],
    ),
    Stage(
        name="stage-4-benchmark",
        prompt_file="stage-4-benchmark-validate",
        description="Benchmark validation",
        expected_artifacts=["prediction-vs-actual.md"],
    ),
]


class Orchestrator:
    """Walks the agent through the 4 stages, calling LLM + tools in a loop."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        run_dir: Path,
        inputs: dict[str, Any],
        max_turns_per_stage: int = 20,
    ):
        self.llm = llm
        self.tools = tools
        self.run_dir = run_dir
        self.inputs = inputs
        self.max_turns = max_turns_per_stage
        self.transcript_path = run_dir / "transcript.jsonl"
        self.tool_schemas = load_tool_schemas()
        self._system_prompt = load_prompt("system_prompt")

    # ---- Public API ----

    def run(self) -> int:
        """Run all stages. Returns exit code per agent/README contract."""
        self._init_run()

        # Stage 1
        rc = self._run_stage(STAGES[0])
        if rc != 0:
            return rc

        # Check decision: if Stage 1 said drop/wait, don't proceed
        s1 = self._read_json("stage-1-selection-score.json")
        if s1 and s1.get("decision") in ("drop", "wait") and not s1.get("next_stage", False):
            self._final_report()
            return 2

        # Stages 2 + 3
        for stage in STAGES[1:3]:
            rc = self._run_stage(stage)
            if rc != 0:
                return rc

        # Stage 4 (optional)
        if self.inputs.get("benchmark", True):
            rc = self._run_stage(STAGES[3])
            if rc != 0:
                return rc

        # Final report
        self._final_report()

        # Check Stage 4 verdict for exit code 3
        if self.inputs.get("benchmark", True):
            verdict_path = self.run_dir / "prediction-vs-actual.md"
            if verdict_path.is_file() and "❌" in verdict_path.read_text(encoding="utf-8"):
                return 3

        return 0

    # ---- Internals ----

    def _init_run(self):
        env_info = {
            "model": self.inputs["model"],
            "target_hardware": self.inputs["target_hardware"],
            "business_scenario": self.inputs["business_scenario"],
            "provider": self.inputs.get("provider"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (self.run_dir / "env.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")
        # Seed the report + citations files
        (self.run_dir / "evaluation-report.md").write_text(
            f"# Evaluation Report\n\n_Run started {env_info['started_at']}_\n\n"
            f"- model: `{env_info['model']}`\n"
            f"- target hardware: `{env_info['target_hardware']}`\n"
            f"- business scenario: `{env_info['business_scenario']}`\n",
            encoding="utf-8",
        )
        (self.run_dir / "research-citations.md").write_text("# Research Citations\n\n", encoding="utf-8")

    def _read_json(self, name: str) -> dict | None:
        p = self.run_dir / name
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _run_stage(self, stage: Stage) -> int:
        stage_prompt_tpl = load_prompt(stage.prompt_file)
        # Substitute placeholders manually so JSON braces in the prompt body don't confuse str.format().
        replacements = {
            "{model}": self.inputs["model"],
            "{target_hardware}": self.inputs["target_hardware"],
            "{business_scenario}": self.inputs["business_scenario"],
            "{quant}": self._lookup_quant(),
            "{spec}": self._lookup_spec(),
            "{target_host}": self.inputs.get("target_host") or "local",
            "{no_benchmark}": str(not self.inputs.get("benchmark", True)).lower(),
        }
        stage_prompt = stage_prompt_tpl
        for k, v in replacements.items():
            stage_prompt = stage_prompt.replace(k, v)

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": stage_prompt},
        ]

        for turn in range(self.max_turns):
            resp = self.llm.chat(messages, self.tool_schemas)
            self._log_turn(stage.name, turn, resp, messages)

            if not resp.tool_calls:
                # Final text — stage complete
                if resp.text:
                    messages.append({"role": "assistant", "content": resp.text})
                # Verify expected artifacts
                missing = [a for a in stage.expected_artifacts if not (self.run_dir / a).is_file()]
                if missing:
                    print(f"  [stage {stage.name}] ⚠️ missing artifacts: {missing}")
                else:
                    print(f"  [stage {stage.name}] ✅ complete")
                return 0

            # Tool calls — invoke each and append results
            messages.append(self._assistant_tool_call_message(resp))
            for tc in resp.tool_calls:
                result = self.tools.invoke(tc.name, tc.arguments)
                messages.append(self._tool_result_message(tc, result))

        print(f"  [stage {stage.name}] ❌ exhausted {self.max_turns} turns")
        return 4

    def _assistant_tool_call_message(self, resp: ChatResponse) -> dict:
        # Provider-agnostic shape; per-provider client adapter remaps on next call.
        # We use OpenAI's shape as the canonical intermediate.
        return {
            "role": "assistant",
            "content": resp.text or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ],
        }

    def _tool_result_message(self, tc, result: dict) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(result),
        }

    def _log_turn(self, stage: str, turn: int, resp: ChatResponse, messages: list[dict]):
        entry = {
            "stage": stage,
            "turn": turn,
            "ts": time.time(),
            "finish_reason": resp.finish_reason,
            "text_len": len(resp.text or ""),
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in resp.tool_calls
            ],
        }
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _lookup_quant(self) -> str:
        d = self._read_json("stage-2-quantization-score.json") or {}
        return d.get("decision", "unknown")

    def _lookup_spec(self) -> str:
        d = self._read_json("stage-2-quantization-score.json") or {}
        return d.get("speculative_method", "unknown")

    def _final_report(self):
        # Append a footer with the run-end timestamp; the agent has been
        # writing the body throughout.
        path = self.run_dir / "evaluation-report.md"
        if path.is_file():
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\n\n---\n\n_Run completed {time.strftime('%Y-%m-%dT%H:%M:%S%z')}_\n")
