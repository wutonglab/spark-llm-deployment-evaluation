"""End-to-end orchestrator dry-run with a mocked LLM.

Verifies that the stage machine wires up correctly and writes the expected
artifacts when stage prompts trigger tool calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


def test_orchestrator_runs_stages_with_mock_llm(tmp_path: Path):
    from core.llm_client import ChatResponse, ToolCall
    from core.orchestrator import Orchestrator
    from core.tools import ToolRegistry

    # Stage 1: the LLM writes the score JSON, then returns final text.
    s1_calls = [
        ChatResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_artifact",
                    arguments={
                        "name": "stage-1-selection-score.json",
                        "content": json.dumps(
                            {
                                "stage": 1,
                                "framework": "02-new-model-selection",
                                "scores": {"capability": 0.2},
                                "total": 0.6,
                                "decision": "prefer-adopt",
                                "next_stage": True,
                            }
                        ),
                    },
                )
            ],
            raw=None,
        ),
        ChatResponse(text="Stage 1 done.", tool_calls=[], raw=None),
    ]
    s2_calls = [
        ChatResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    id="t2",
                    name="write_artifact",
                    arguments={
                        "name": "stage-2-quantization-score.json",
                        "content": json.dumps(
                            {
                                "stage": 2,
                                "framework": "01-fp8-vs-nvfp4-decision",
                                "scores": {"hardware": 0.15},
                                "total": 0.2,
                                "decision": "fp8",
                                "speculative_method": "mtp-1",
                                "speculative_rationale": "ok",
                            }
                        ),
                    },
                )
            ],
            raw=None,
        ),
        ChatResponse(text="Stage 2 done.", tool_calls=[], raw=None),
    ]
    s3_calls = [
        ChatResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    id="t3a",
                    name="write_artifact",
                    arguments={"name": "proposed-config.env", "content": "VARIANT=with-mtp1\n"},
                ),
                ToolCall(
                    id="t3b",
                    name="write_artifact",
                    arguments={"name": "proposed-launch-command.sh", "content": "#!/bin/bash\necho mock\n"},
                ),
            ],
            raw=None,
        ),
        ChatResponse(text="Stage 3 done.", tool_calls=[], raw=None),
    ]

    all_calls = iter(s1_calls + s2_calls + s3_calls)

    mock_llm = MagicMock()
    mock_llm.chat.side_effect = lambda messages, tools: next(all_calls)
    mock_llm.model = "mock-model"

    tools = ToolRegistry(run_dir=tmp_path, dry_run=True)
    orch = Orchestrator(
        llm=mock_llm,
        tools=tools,
        run_dir=tmp_path,
        inputs={
            "model": "fake/model",
            "target_hardware": "fake-gpu",
            "business_scenario": "test",
            "provider": "mock",
            "benchmark": False,  # skip Stage 4
        },
        max_turns_per_stage=5,
    )

    rc = orch.run()

    # Stage 4 skipped via benchmark=False; orchestrator should return 0
    assert rc == 0
    assert (tmp_path / "stage-1-selection-score.json").is_file()
    assert (tmp_path / "stage-2-quantization-score.json").is_file()
    assert (tmp_path / "proposed-config.env").is_file()
    assert (tmp_path / "proposed-launch-command.sh").is_file()
    assert (tmp_path / "evaluation-report.md").is_file()
    # Transcript was written
    assert (tmp_path / "transcript.jsonl").is_file()
