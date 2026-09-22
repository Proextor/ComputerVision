"""Validate the display's deterministic fixtures against the authoritative M5 schema."""
import json
from pathlib import Path

from reactive_vision.schemas import SceneSnapshot


def test_display_demo_fixtures_use_real_schema():
    fixtures = json.loads((Path(__file__).parents[1] / "display/src/demo/fixtures.json").read_text(encoding="utf-8"))
    assert len(fixtures) == 14
    for fixture in fixtures:
        scene = SceneSnapshot.model_validate(fixture["snapshot"])
        assert scene.schema_version == "4.0"
