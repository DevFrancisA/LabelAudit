import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def toy() -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "toy.json").read_text())
