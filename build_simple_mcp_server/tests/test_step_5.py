import pytest
from fastapi.testclient import TestClient

def test_TestStep5Task1():
    """Tests if the tool_registry is created correctly."""
    try:
        from app.main import tool_registry
        from app.tools import get_weather
        assert isinstance(tool_registry, dict)
        assert 'get_weather' in tool_registry
        assert tool_registry['get_weather'] == get_weather
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import tool_registry from app.main.")
    except (KeyError, AssertionError) as e:
        pytest.fail(f"AssertionFailedError: tool_registry is not configured correctly. {e}")

def test_TestStep5Task2():
    """Tests the execute verb implementation."""
    try:
        from app.main import app
        client = TestClient(app)
        request_body = {
            "verb": "execute",
            "tool_name": "get_weather",
            "arguments": {"location": "Cloud City"}
        }
        response = client.post("/mcp", json=request_body)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}. Response: {response.text}"
        data = response.json()
        assert data['tools'] is None
        assert 'result' in data
        assert "Cloud City" in data['result']
        assert "sunny" in data['result']
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import app from app.main.")
    except Exception as e:
        pytest.fail(f"AssertionFailedError: Execute request failed. {e}")
