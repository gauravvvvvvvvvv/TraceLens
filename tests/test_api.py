from fastapi.testclient import TestClient

from app.main import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_investigation():
    response = TestClient(app).get("/investigations/does-not-exist")
    assert response.status_code == 404

