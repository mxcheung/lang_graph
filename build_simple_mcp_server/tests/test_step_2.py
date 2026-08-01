import pytest
from fastapi.testclient import TestClient

# We need to be careful with imports to avoid loading solution code too early.

def test_TestStep2Task1():
    """Tests if the FastAPI app instance is created."""
    try:
        from app.main import app
        assert app is not None, "The 'app' instance is not created in app/main.py."
        from fastapi import FastAPI
        assert isinstance(app, FastAPI), "The 'app' variable is not an instance of FastAPI."
    except ImportError as e:
        pytest.fail(f"AssertionFailedError: Failed to import 'app' from app.main. {e}")
    except Exception as e:
        pytest.fail(f"AssertionFailedError: An error occurred: {e}")

def test_TestStep2Task2():
    """Tests if the /mcp POST endpoint exists."""
    try:
        from app.main import app
        client = TestClient(app)
        # A 422 error means the endpoint exists but the body is invalid, which is expected for now
        # A 405 error means the endpoint exists but the method is wrong
        # A 404 error means the endpoint does not exist
        response = client.post("/mcp", json={})
        assert response.status_code != 404, "The /mcp endpoint was not found. Did you create it with @app.post('/mcp')?"
        assert response.status_code != 405, "The /mcp endpoint does not support POST requests."
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import 'app' from app.main.")
