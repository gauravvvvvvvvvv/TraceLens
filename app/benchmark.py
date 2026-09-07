from __future__ import annotations

import json
import time
from pathlib import Path

from .agent import Investigator
from .metrics import aggregate_by_memory_mode, benchmark_episode_metrics
from .schemas import BenchmarkCase, InvestigationRequest, MemoryMode


def load_cases(path: str = "benchmarks/cases.json") -> list[BenchmarkCase]:
    return [BenchmarkCase.model_validate(item) for item in json.loads(Path(path).read_text())]


def run_benchmarks(investigator: Investigator, modes: list[MemoryMode] | None = None) -> dict:
    selected_modes = modes or [MemoryMode.none, MemoryMode.short, MemoryMode.long]
    episodes = []
    for case in load_cases():
        for mode in selected_modes:
            report = investigator.investigate(
                InvestigationRequest(
                    subject=case.subject,
                    question=case.claim,
                    subject_type=case.subject_type,
                    memory_mode=mode,
                )
            )
            episodes.append(
                {
                    "case_id": case.id,
                    "memory_mode": mode.value,
                    "expected_verdict": case.expected_verdict.value,
                    "actual_verdict": report.verdict.value,
                    "correct": report.verdict == case.expected_verdict,
                    "confidence": report.confidence,
                    "metrics": report.metrics,
                    "evaluation": benchmark_episode_metrics(report, case.expected_verdict),
                    "status": report.status,
                }
            )

            # Pace live benchmark episodes for free-tier inference providers.
            time.sleep(15)
    completed = [item for item in episodes if item["status"] == "completed"]
    accuracy = sum(item["correct"] for item in completed) / max(1, len(completed))
    result = {
        "episodes": episodes,
        "summary": {
            "completed": len(completed),
            "accuracy": round(accuracy, 3),
            "by_memory_mode": aggregate_by_memory_mode(episodes),
        },
    }
    Path("outputs/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("outputs/benchmarks/latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
