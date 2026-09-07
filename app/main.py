from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .agent import Investigator
from .benchmark import run_benchmarks
from .database import Database
from .schemas import InvestigationReport, InvestigationRequest

app = FastAPI(
    title="TraceLens",
    version="1.0.0",
    description="Evaluation-first agentic OSINT investigation harness",
)
db = Database()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/investigations", response_model=InvestigationReport)
def create_investigation(request: InvestigationRequest) -> InvestigationReport:
    return Investigator().investigate(request)


@app.get("/investigations/{investigation_id}", response_model=InvestigationReport)
def get_investigation(investigation_id: str) -> InvestigationReport:
    report = db.get_report(investigation_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return report


@app.get("/investigations/{investigation_id}/report")
def get_report(investigation_id: str) -> dict:
    report = get_investigation(investigation_id)
    return report.model_dump(exclude={"trace", "usage"})


@app.get("/investigations/{investigation_id}/trace")
def get_trace(investigation_id: str) -> dict:
    report = get_investigation(investigation_id)
    return {"investigation_id": investigation_id, "trace": report.trace, "usage": report.usage}


@app.post("/benchmarks/run")
def benchmarks() -> dict:
    return run_benchmarks(Investigator())

