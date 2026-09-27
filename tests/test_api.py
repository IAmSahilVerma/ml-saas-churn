from fastapi.testclient import TestClient
 
from api.main import app
 
 
def test_health_endpoint_returns_ok():
    # Using the client as a context manager runs the app's startup events,
    # so this also checks that the saved models load correctly.
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200