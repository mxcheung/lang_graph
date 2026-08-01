import pytest
from fastapi.testclient import TestClient

def test_TestStep6Task1():
    """Tests error handling for unknown tools."""
    try:
        from app.main import app
        client = TestClient(app)
        request_body = {
            "verb": "execute",
            "tool_name": "non_existent_tool",
            "arguments": {}
        }
        response = client.post("/mcp", json=request_body)
        assert response.status_code == 404, f"Expected status 404 for unknown tool, but got {response.status_code}"
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import app from app.main.")
    except Exception as e:
        pytest.fail(f"AssertionFailedError: Error handling test failed. {e}")
