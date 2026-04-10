import pytest
import io
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

def test_batch_predict_endpoint(client):
    # Create a valid CSV file
    csv_content = "magnitude,depth,cdi,mmi,sig\n5.5,25.0,6.0,5.5,500\n4.2,10.0,3.5,3.0,100"
    csv_file = io.BytesIO(csv_content.encode())
    csv_file.name = "test_data.csv"
    
    response = client.post(
        "/predict/batch",
        files={"payload": ("test_data.csv", csv_file, "text/csv")}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert len(data["prediction"]) == 2

def test_batch_predict_endpoint_invalid_file(client):
    # Upload non-CSV file should fail
    text_file = io.BytesIO(b"not a csv")
    text_file.name = "test.txt"
    
    response = client.post(
        "/predict/batch",
        files={"payload": ("test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 422

def test_batch_predict_endpoint_wrong_columns(client):
    # CSV with wrong columns should fail validation
    csv_content = "wrong_column1,wrong_column2\n1,2\n3,4"
    csv_file = io.BytesIO(csv_content.encode())
    csv_file.name = "test_data.csv"
    
    response = client.post(
        "/predict/batch",
        files={"payload": ("test_data.csv", csv_file, "text/csv")}
    )
    
    assert response.status_code == 422
