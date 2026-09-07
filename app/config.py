from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_temperature: float | None = _optional_float("LLM_TEMPERATURE")
    database_path: str = os.getenv("DATABASE_PATH", "tracelens.db")
    max_steps: int = int(os.getenv("MAX_STEPS", "8"))
    max_tool_calls: int = int(os.getenv("MAX_TOOL_CALLS", "12"))
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "4"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    max_document_bytes: int = int(os.getenv("MAX_DOCUMENT_BYTES", "3000000"))
    input_cost_per_mtok: float | None = _optional_float("INPUT_COST_PER_MTOK")
    output_cost_per_mtok: float | None = _optional_float("OUTPUT_COST_PER_MTOK")

    def validate_runtime(self) -> None:
        if not self.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        if not self.llm_base_url:
            raise RuntimeError("LLM_BASE_URL is not configured")
        if not self.llm_model:
            raise RuntimeError("LLM_MODEL is not configured")


settings = Settings()
