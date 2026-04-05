import pytest
from fastapi.testclient import TestClient
from app.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "All systems operational"

def test_predict_endpoint_fail_on_missing_data(client):
    # Sending empty body should fail validation
    # Actually, it will try to resolve dependencies FIRST.
    # But since we are in a 'with' block, lifespan will have run.
    response = client.post("/predict", json={})
    assert response.status_code == 422 # Pydantic validation error
