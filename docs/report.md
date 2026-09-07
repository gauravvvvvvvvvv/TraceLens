# TraceLens: Design and Evaluation Report

## Problem interpretation

The assignment is not merely a search-and-summary task. A usable OSINT agent must show where claims came from, distinguish corroboration from repetition, express uncertainty, and expose whether additional steps improve the investigation. TraceLens therefore treats evaluation and provenance as first-class system behavior.

## Design

The system uses a configurable Claude model through an OpenAI-compatible inference endpoint for semantic planning, evidence interpretation, gap analysis, and final synthesis. Search, retrieval, quote validation, source scoring, persistence, confidence, budgets, and metrics remain deterministic. This boundary uses the model where language understanding is valuable without trusting it to invent citations or grade itself.

Each question is decomposed into atomic claims. Search results are fetched through a bounded public-URL client. HTML and PDF text are extracted, hashed, and deduplicated. Claude returns structured evidence containing an exact excerpt and supplied URL. The application accepts the evidence only when the excerpt occurs in the retrieved document.

## Memory

None mode disables cross-episode retrieval. Short mode retains typed working state during the episode. Long mode retrieves persistent provenance-backed leads using subject and claim similarity. Retrieved memories suggest sources or queries but are not evidence until re-fetched and revalidated. This prevents stale memory from silently determining a verdict.

## Reliability and confidence

Source scores are transparent domain-level priors, not claims of objective truth. Confidence combines evidence coverage, weighted support/refute margin, and independent source count. Sparse evidence leads to abstention; balanced opposing evidence leads to a mixed verdict. The system records confidence and verdict changes in the trace.

## Evaluation

The included benchmark covers company, person, true, false, ambiguous-date, insufficient-evidence, and memory-interference behavior. Each case can run in all three memory modes. The harness records verdict correctness, coverage, citation validity, source diversity, model/tool calls, tokens, latency, and optional model cost.

### Observed benchmark results

The benchmark executed 18 planned episodes across six cases and three memory modes. Fifteen episodes completed successfully, giving an episode completion rate of 83.3%. Eight of the fifteen completed episodes produced the expected verdict, giving 53.3% accuracy on completed episodes.

| Memory mode | Completed episodes | Accuracy | Mean reward | Mean Brier score | Mean tool calls |
|---|---:|---:|---:|---:|---:|
| None | 5 | 60% | 0.709 | 0.128 | 12.0 |
| Short | 5 | 40% | 0.592 | 0.188 | 12.0 |
| Long | 5 | 60% | 0.731 | 0.141 | 10.4 |

Long-term memory did not improve verdict accuracy over no cross-episode memory in this small benchmark. It did, however, produce a slightly higher mean reward and require fewer retrieval tool calls. Short memory performed worse on both accuracy and reward. This indicates that additional context does not automatically improve reliability and can amplify incomplete or weak evidence.

The system performed consistently on the synthetic private claim, correctly returning `insufficient_evidence` in all three memory modes. It performed well on the GitHub acquisition and Ada Lovelace cases, but struggled with date-sensitive claims involving Python and the YouTube acquisition. These failures were primarily associated with source availability, incomplete claim coverage, and conflicting evidence.

Three JavaScript episodes failed because the free Groq inference tier reached its token-per-minute limit. These episodes are excluded from completed-episode accuracy but remain visible in the raw benchmark output. This demonstrates an operational failure mode of agentic systems: external inference quotas can prevent otherwise valid investigations from completing. A production implementation should use provider-aware backoff, resumable episodes, and fallback model routing.

## Failure analysis

Expected failure modes include search instability, blocked or dynamic pages, duplicated reporting, heuristic misclassification of source quality, extraction truncation, model schema errors, and premature convergence. TraceLens makes these failures visible through stage-level traces and returns insufficient evidence when tool budgets expire without adequate support.

## Limitations and future work

This implementation favors auditability and completion within the take-home window. Its benchmark is small, long-term retrieval is lexical, live search is not deterministic, and source trust is modeled coarsely. A production version should add cached replay evaluation, entity-aware embedding memory, source-dependency graphs, a durable task queue, a larger calibration dataset, and human escalation for high-impact conclusions.
