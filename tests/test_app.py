from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "All systems operational"

def test_predict_endpoint_fail_on_missing_data():
    # Sending empty body should fail validation
    response = client.post("/predict", json={})
    assert response.status_code == 422 # Pydantic validation error
