from __future__ import annotations

from collections import defaultdict

from .schemas import InvestigationReport, Verdict


def benchmark_episode_metrics(report: InvestigationReport, expected: Verdict) -> dict:
    correctness = float(report.verdict == expected)
    coverage = float(report.metrics.get("evidence_coverage", 0))
    citation_validity = float(report.metrics.get("citation_validity", 0))
    diversity = float(report.metrics.get("source_diversity", 0))
    tool_calls = int(report.metrics.get("tool_calls", 0))
    tokens = int(report.metrics.get("input_tokens", 0)) + int(report.metrics.get("output_tokens", 0))
    efficiency = max(0.0, 1 - min(1.0, tool_calls / 30) * 0.6 - min(1.0, tokens / 100_000) * 0.4)
    brier = _brier(report, expected)
    calibration = 0.5 if brier is None else 1 - brier
    reward = (
        0.30 * correctness
        + 0.20 * coverage
        + 0.15 * citation_validity
        + 0.15 * calibration
        + 0.10 * diversity
        + 0.10 * efficiency
    )
    return {
        "verdict_correctness": round(correctness, 3),
        "evidence_coverage": round(coverage, 3),
        "citation_validity": round(citation_validity, 3),
        "brier_score": None if brier is None else round(brier, 3),
        "source_diversity": round(diversity, 3),
        "efficiency_score": round(efficiency, 3),
        "episode_reward": round(reward, 3),
    }


def aggregate_by_memory_mode(episodes: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        if episode["status"] == "completed":
            grouped[episode["memory_mode"]].append(episode)
    result = {}
    for mode, items in grouped.items():
        briers = [i["evaluation"]["brier_score"] for i in items if i["evaluation"]["brier_score"] is not None]
        result[mode] = {
            "episodes": len(items),
            "accuracy": round(sum(i["correct"] for i in items) / max(1, len(items)), 3),
            "mean_reward": round(sum(i["evaluation"]["episode_reward"] for i in items) / max(1, len(items)), 3),
            "mean_brier": round(sum(briers) / len(briers), 3) if briers else None,
            "mean_tool_calls": round(sum(i["metrics"].get("tool_calls", 0) for i in items) / max(1, len(items)), 2),
        }
    return result


def _brier(report: InvestigationReport, expected: Verdict) -> float | None:
    if expected not in {Verdict.supported, Verdict.refuted}:
        return None
    if report.verdict == Verdict.supported:
        probability_supported = report.confidence
    elif report.verdict == Verdict.refuted:
        probability_supported = 1 - report.confidence
    else:
        probability_supported = 0.5
    target = 1.0 if expected == Verdict.supported else 0.0
    return (probability_supported - target) ** 2

