from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from json_repair import repair_json

from .config import settings
from .schemas import ModelUsage

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    def generate(self, stage: str, system: str, prompt: str, schema: type[T]) -> tuple[T, ModelUsage]: ...


class OpenAICompatibleProvider:
    def __init__(self, cache_dir: str = ".cache/llm"):
        settings.validate_runtime()
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, stage: str, system: str, prompt: str, schema: type[T]) -> tuple[T, ModelUsage]:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        full_prompt = (
            f"{prompt}\n\nReturn only a JSON object conforming exactly to this schema:\n{schema_json}"
        )
        key = hashlib.sha256(
            (settings.llm_model + settings.llm_base_url + system + full_prompt).encode("utf-8")
        ).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            return schema.model_validate(cached["data"]), ModelUsage.model_validate(cached["usage"])

        last_error: Exception | None = None
        for attempt in range(2):
            started = time.perf_counter()
            request = {
                "model": settings.llm_model,
                "max_tokens": 3000,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
            }
            # Some OpenAI-compatible model routes constrain or reject temperature.
            # Omit it unless explicitly configured and let the provider choose its default.
            if settings.llm_temperature is not None:
                request["temperature"] = settings.llm_temperature
            response = self.client.chat.completions.create(**request)
            latency = int((time.perf_counter() - started) * 1000)
            text = response.choices[0].message.content or ""
            try:
                data = schema.model_validate(_extract_json(text))
                input_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
                cost = _cost(input_tokens, output_tokens)
                usage = ModelUsage(
                    stage=stage,
                    model=settings.llm_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency,
                    estimated_cost_usd=cost,
                )
                cache_file.write_text(
                    json.dumps({"data": data.model_dump(mode="json"), "usage": usage.model_dump(mode="json")}),
                    encoding="utf-8",
                )
                return data, usage
            except Exception as exc:  # one bounded repair attempt
                last_error = exc
                full_prompt += f"\n\nYour previous response failed validation: {exc}. Return corrected JSON only."
        raise RuntimeError(f"Model output failed validation: {last_error}")


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        candidate = cleaned[start : end + 1] if start >= 0 and end > start else cleaned
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # OpenAI-compatible proxy routes do not always support constrained JSON.
            # Repair common truncation, quoting, comma, and fence defects before the
            # result is subjected to strict Pydantic validation.
            repaired = repair_json(candidate, return_objects=True)
            if not isinstance(repaired, dict):
                raise ValueError("Model response could not be repaired into a JSON object")
            return repaired


def _cost(input_tokens: int, output_tokens: int) -> float | None:
    if settings.input_cost_per_mtok is None or settings.output_cost_per_mtok is None:
        return None
    return round(
        input_tokens / 1_000_000 * settings.input_cost_per_mtok
        + output_tokens / 1_000_000 * settings.output_cost_per_mtok,
        6,
    )
