import pytest
from fastapi.testclient import TestClient

def test_TestStep4Task1():
    """Tests the get_weather tool function."""
    try:
        from app.tools import get_weather
        location = "Testville"
        result = get_weather(location=location)
        assert isinstance(result, str)
        assert location in result
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import get_weather from app.tools.")
    except Exception as e:
        pytest.fail(f"AssertionFailedError: get_weather function failed: {e}")

def test_TestStep4Task2():
    """Tests the GET_WEATHER_TOOL schema definition."""
    try:
        from app.tools import GET_WEATHER_TOOL
        from app.models import Tool
        assert isinstance(GET_WEATHER_TOOL, Tool)
        assert GET_WEATHER_TOOL.name == 'get_weather'
        assert len(GET_WEATHER_TOOL.parameters) == 1
        assert GET_WEATHER_TOOL.parameters[0].name == 'location'
        assert GET_WEATHER_TOOL.parameters[0].type == 'string'
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import GET_WEATHER_TOOL from app.tools.")
    except (AttributeError, AssertionError) as e:
        pytest.fail(f"AssertionFailedError: GET_WEATHER_TOOL schema is incorrect. {e}")

def test_TestStep4Task3():
    """Tests the discovery verb implementation."""
    try:
        from app.main import app
        client = TestClient(app)
        response = client.post("/mcp", json={"verb": "discovery"})
        assert response.status_code == 200
        data = response.json()
        assert 'tools' in data
        assert data['result'] is None
        assert isinstance(data['tools'], list)
        assert len(data['tools']) == 1
        assert data['tools'][0]['name'] == 'get_weather'
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import app from app.main.")
    except Exception as e:
        pytest.fail(f"AssertionFailedError: Discovery request failed. {e}")
