from app.config import settings
from app.providers import _extract_json


def test_temperature_is_optional_for_provider_compatibility():
    assert settings.llm_temperature is None or isinstance(settings.llm_temperature, float)


def test_malformed_model_json_is_repaired_before_validation():
    malformed = '```json\n{"items": [{"stance": "supports" "excerpt": "source text"}]}\n```'
    repaired = _extract_json(malformed)
    assert repaired["items"][0]["stance"] == "supports"
    assert repaired["items"][0]["excerpt"] == "source text"
