import os
import sys
import types

from fastapi.testclient import TestClient

os.environ.setdefault("ALLOW_FAKE_GEN", "1")
os.environ.setdefault("SKIP_AUDIO", "1")


class _StubCore:
    def __init__(self) -> None:
        self.available_devices: list[str] = []

    def compile_model(self, *args, **kwargs):  # pragma: no cover - unused in fake mode
        raise RuntimeError("OpenVINO not available in tests")


if "openvino" not in sys.modules:
    sys.modules["openvino"] = types.SimpleNamespace(Core=_StubCore)


def test_health() -> None:
    from src.api.server import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body


def test_compose_fake_ok() -> None:
    from src.api.server import app

    client = TestClient(app)
    payload = {
        "base_style": "rock",
        "bpm": 120,
        "key": "C",
        "sections": [
            {"name": "intro", "duration": 2},
            {"name": "verse", "duration": 2},
        ],
        "seed": 1,
        "with_vocal": False,
        "max_tokens": 64,
    }
    response = client.post("/v1/midi/compose_full", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body.get("format") == "wav"
    assert "b64" in body
