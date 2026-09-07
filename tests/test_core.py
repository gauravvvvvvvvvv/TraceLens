from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import _contains_excerpt
from app.database import Database
from app.fetcher import SafeDocumentFetcher
from app.metrics import benchmark_episode_metrics
from app.schemas import AtomicClaim, EvidenceItem, InvestigationReport, Verdict
from app.scoring import evaluate_evidence, source_score


def evidence(claim_id: str, stance: str, url: str, score: float = 0.8) -> EvidenceItem:
    return EvidenceItem(
        atomic_claim_id=claim_id,
        stance=stance,
        excerpt="This is a sufficiently long exact source excerpt.",
        interpretation="Relevant evidence",
        source_url=url,
        source_title="Source",
        source_type="test",
        source_reliability=score,
        relevance=1,
        independence_group=url,
        excerpt_verified=True,
    )


def test_excerpt_verification_normalizes_whitespace_and_case():
    assert _contains_excerpt(
        "The SOURCE says this is a sufficiently long exact source excerpt today.",
        "source says this is a sufficiently   long exact source excerpt",
    )
    assert not _contains_excerpt("Completely different text", "Invented quotation that is long enough")


def test_conflicting_evidence_produces_mixed_verdict():
    claim = AtomicClaim(text="A disputed claim")
    items = [
        evidence(claim.id, "supports", "https://source-a.example"),
        evidence(claim.id, "refutes", "https://source-b.example"),
    ]
    verdict, confidence, coverage, conflicts = evaluate_evidence([claim], items)
    assert verdict == Verdict.mixed
    assert confidence < 0.8
    assert coverage == 1
    assert conflicts


def test_unverified_evidence_causes_abstention():
    claim = AtomicClaim(text="A claim")
    item = evidence(claim.id, "supports", "https://example.com")
    item.excerpt_verified = False
    verdict, _, coverage, _ = evaluate_evidence([claim], [item])
    assert verdict == Verdict.insufficient_evidence
    assert coverage == 0


def test_duplicate_domain_does_not_increase_independence():
    claim = AtomicClaim(text="A claim")
    first = evidence(claim.id, "supports", "https://example.com/a")
    second = evidence(claim.id, "supports", "https://example.com/b")
    first.independence_group = second.independence_group = "example.com"
    _, confidence_same, _, _ = evaluate_evidence([claim], [first, second])
    second.independence_group = "independent.example"
    _, confidence_independent, _, _ = evaluate_evidence([claim], [first, second])
    assert confidence_independent > confidence_same


def test_database_round_trip(tmp_path: Path):
    db = Database(str(tmp_path / "test.db"))
    report = InvestigationReport(
        investigation_id="abc", subject="Subject", question="Was this true?", status="completed"
    )
    db.save_report(report)
    loaded = db.get_report("abc")
    assert loaded is not None
    assert loaded.investigation_id == "abc"


def test_memory_search_filters_unrelated_records(tmp_path: Path):
    db = Database(str(tmp_path / "test.db"))
    db.save_memory("Microsoft GitHub", "Microsoft acquired GitHub", "https://example.com", "hash", {"fact": "match"})
    db.save_memory("Ada Lovelace", "Analytical Engine notes", "https://example.org", "hash2", {"fact": "noise"})
    results = db.search_memories("Microsoft", "GitHub acquisition")
    assert results
    assert results[0]["fact"] == "match"


def test_fetcher_blocks_private_targets():
    with pytest.raises(ValueError):
        SafeDocumentFetcher._validate_url("http://127.0.0.1/admin")
    with pytest.raises(ValueError):
        SafeDocumentFetcher._validate_url("file:///etc/passwd")


def test_source_scores_are_bounded():
    for url in ["https://example.gov/report", "https://reuters.com/story", "https://unknown.example/post"]:
        score, _, _ = source_score(url)
        assert 0 <= score <= 1


def test_episode_metrics_are_bounded():
    report = InvestigationReport(
        investigation_id="metrics",
        subject="Subject",
        question="Is this claim true?",
        status="completed",
        verdict=Verdict.supported,
        confidence=0.8,
        metrics={
            "evidence_coverage": 1,
            "citation_validity": 1,
            "source_diversity": 0.5,
            "tool_calls": 10,
            "input_tokens": 5000,
            "output_tokens": 1000,
        },
    )
    values = benchmark_episode_metrics(report, Verdict.supported)
    assert 0 <= values["episode_reward"] <= 1
    assert values["brier_score"] == pytest.approx(0.04)


def test_tool_budget_message_is_specific():
    message = "The tool-call budget was exhausted before additional evidence could be retrieved."
    assert "budget" in message and "evidence" in message
