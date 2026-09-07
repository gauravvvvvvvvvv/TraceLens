from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings
from .schemas import InvestigationReport


class Database:
    def __init__(self, path: str | None = None):
        self.path = path or settings.database_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
                """
            )

    def save_report(self, report: InvestigationReport) -> None:
        payload = report.model_dump_json()
        now = report.created_at.isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO investigations(id, subject, question, status, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (report.investigation_id, report.subject, report.question, report.status, payload, now, now),
            )

    def get_report(self, investigation_id: str) -> InvestigationReport | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM investigations WHERE id = ?", (investigation_id,)
            ).fetchone()
        return InvestigationReport.model_validate_json(row["payload"]) if row else None

    def save_memory(self, subject: str, claim: str, source_url: str, content_hash: str, payload: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO memories(subject, claim, source_url, content_hash, payload, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (subject.lower(), claim, source_url, content_hash, json.dumps(payload)),
            )

    def search_memories(self, subject: str, query: str, limit: int = 5) -> list[dict]:
        wanted = set(_tokens(subject + " " + query))
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM memories ORDER BY id DESC LIMIT 100").fetchall()
        scored = []
        for row in rows:
            candidate = set(_tokens(row["subject"] + " " + row["claim"]))
            union = wanted | candidate
            score = len(wanted & candidate) / len(union) if union else 0
            if score >= 0.15:
                item = json.loads(row["payload"])
                item["retrieval_similarity"] = round(score, 4)
                scored.append((score, item))
        return [
            item
            for _, item in sorted(
                scored,
                key=lambda pair: pair[0],
                reverse=True,
            )[:limit]
        ]


def _tokens(value: str) -> list[str]:
    return [token.strip(".,:;!?()[]{}\"'").lower() for token in value.split() if len(token) > 2]

