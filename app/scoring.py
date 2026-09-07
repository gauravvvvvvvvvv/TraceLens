from __future__ import annotations

from collections import defaultdict
from urllib.parse import urlparse

from .schemas import AtomicClaim, Conflict, EvidenceItem, Verdict


HIGH_TRUST_SUFFIXES = (".gov", ".gov.uk", ".europa.eu", ".edu")


def source_score(url: str) -> tuple[float, str, str]:
    host = (urlparse(url).hostname or "unknown").lower().removeprefix("www.")
    if host.endswith(HIGH_TRUST_SUFFIXES):
        return 0.95, "government_or_academic", host
    if any(token in host for token in ("reuters.com", "apnews.com", "bbc.", "nature.com")):
        return 0.82, "established_news_or_academic", host
    if any(token in host for token in ("wikipedia.org", "britannica.com")):
        return 0.60, "general_secondary", host
    return 0.50, "unknown_web", host


def evaluate_evidence(
    claims: list[AtomicClaim], evidence: list[EvidenceItem]
) -> tuple[Verdict, float, float, list[Conflict]]:
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)

    for item in evidence:
        if item.excerpt_verified:
            grouped[item.atomic_claim_id].append(item)

    required_claims = [claim for claim in claims if claim.required] or claims
    claim_results: dict[str, str] = {}
    claim_strengths: list[float] = []
    conflicts: list[Conflict] = []

    for claim in required_claims:
        items = grouped.get(claim.id, [])

        supporting_items = [
            item for item in items if item.stance == "supports"
        ]
        refuting_items = [
            item for item in items if item.stance == "refutes"
        ]

        support_score = sum(
            item.source_reliability * item.relevance
            for item in supporting_items
        )
        refute_score = sum(
            item.source_reliability * item.relevance
            for item in refuting_items
        )

        claim_strengths.append(
            min(1.0, max(support_score, refute_score))
        )

        if supporting_items and refuting_items:
            claim_results[claim.id] = "mixed"

            conflicts.append(
                Conflict(
                    claim_id=claim.id,
                    description=(
                        "Retrieved evidence both supports and refutes "
                        "this required claim."
                    ),
                    supporting_evidence_ids=[
                        item.id for item in supporting_items
                    ],
                    refuting_evidence_ids=[
                        item.id for item in refuting_items
                    ],
                )
            )
        elif support_score >= 0.40:
            claim_results[claim.id] = "supported"
        elif refute_score >= 0.40:
            claim_results[claim.id] = "refuted"
        else:
            # Neutral evidence may provide context, but it does not
            # establish or refute the required atomic claim.
            claim_results[claim.id] = "unresolved"

    resolved_claims = sum(
        result != "unresolved"
        for result in claim_results.values()
    )

    coverage = resolved_claims / max(1, len(required_claims))

    independence = len(
        {
            item.independence_group
            for item in evidence
            if item.excerpt_verified
            and item.stance in {"supports", "refutes"}
        }
    )

    mean_strength = (
        sum(claim_strengths) / len(claim_strengths)
        if claim_strengths
        else 0.0
    )

    confidence = min(
        0.97,
        coverage * 0.55
        + mean_strength * 0.25
        + min(independence / 3, 1.0) * 0.20,
    )

    results = set(claim_results.values())

    if "mixed" in results or (
        "supported" in results and "refuted" in results
    ):
        verdict = Verdict.mixed
        confidence *= 0.75

    elif "refuted" in results:
        # Required atomic claims form a conjunction. Reliable
        # refutation of one required proposition refutes the
        # combined claim.
        verdict = Verdict.refuted

    elif "unresolved" in results or coverage < 0.75:
        verdict = Verdict.insufficient_evidence
        confidence *= 0.60

    else:
        verdict = Verdict.supported

    return (
        verdict,
        round(confidence, 3),
        round(coverage, 3),
        conflicts,
    )

