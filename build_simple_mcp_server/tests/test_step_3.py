import pytest
from pathlib import Path
from pydantic import BaseModel
from typing import get_type_hints

def test_TestStep3Task1():
    """Tests the Tool and ToolParameter models."""
    try:
        from app.models import Tool, ToolParameter
        assert issubclass(ToolParameter, BaseModel)
        assert issubclass(Tool, BaseModel)
        # Test ToolParameter fields
        tp_fields = ToolParameter.model_fields
        assert 'name' in tp_fields
        # In Pydantic v2, check annotation using get_type_hints or directly
        tp_annotations = get_type_hints(ToolParameter)
        assert tp_annotations['name'] == str
        assert 'type' in tp_fields
        assert tp_annotations['type'] == str

        # Test Tool fields
        t_fields = Tool.model_fields
        assert 'name' in t_fields
        t_annotations = get_type_hints(Tool)
        assert t_annotations['name'] == str
        assert 'description' in t_fields
        assert t_annotations['description'] == str
        assert 'parameters' in t_fields
        # This is how you check for List[ToolParameter]
        params_str = str(t_annotations['parameters'])
        assert "List[ToolParameter]" in params_str or "list[ToolParameter]" in params_str.lower() or "typing.List" in params_str
    except ImportError:
        pytest.fail("AssertionFailedError: Could not import Tool or ToolParameter from app.models.")
    except AssertionError as e:
        pytest.fail(f"AssertionFailedError: {e}")

def test_TestStep3Task2():
    """Tests the ModelContextRequest and ModelContextResponse models."""
    try:
        # Check if app.main imports ModelContextRequest and ModelContextResponse
        import app.main
        # Get the source file path
        main_file_path = Path(app.main.__file__)
        with open(main_file_path, 'r') as f:
            main_source = f.read()
        
        # Check for the import statement - handle both single line and multi-line imports
        has_import = (
            'from app.models import ModelContextRequest, ModelContextResponse' in main_source or
            ('from app.models import' in main_source and 
             'ModelContextRequest' in main_source and 
             'ModelContextResponse' in main_source)
        )
        assert has_import, \
               "AssertionFailedError: app.main.py must import ModelContextRequest and ModelContextResponse from app.models."
        
        from app.models import ModelContextRequest, ModelContextResponse, Tool
        from typing import Optional, List, Any, Dict, Union, get_origin, get_args

        # Test ModelContextRequest
        req_fields = ModelContextRequest.model_fields
        req_annotations = get_type_hints(ModelContextRequest)
        assert 'verb' in req_fields and req_annotations['verb'] == str
        
        # Check tool_name: Optional[str] = None
        assert 'tool_name' in req_fields
        tool_name_type = req_annotations['tool_name']
        tool_name_str = str(tool_name_type)
        tool_name_origin = get_origin(tool_name_type)
        # Optional[str] is Union[str, None] in Python's typing system
        assert (tool_name_origin is Union and 
                len(get_args(tool_name_type)) == 2 and
                str in get_args(tool_name_type))
        
        # Check arguments: Optional[Dict[str, Any]] = None
        assert 'arguments' in req_fields
        arguments_type = req_annotations['arguments']
        arguments_str = str(arguments_type).lower()
        arguments_origin = get_origin(arguments_type)
        # Optional[Dict[str, Any]] is Union[Dict[str, Any], None]
        assert (arguments_origin is Union and 
                len(get_args(arguments_type)) == 2)
        arguments_args = get_args(arguments_type)
        # One arg should be None, the other should be Dict[str, Any]
        assert type(None) in arguments_args or None in arguments_args
        dict_arg = next((arg for arg in arguments_args if arg is not type(None) and arg is not None), None)
        assert dict_arg is not None
        dict_origin = get_origin(dict_arg)
        assert dict_origin is dict or dict_origin == dict

        # Test ModelContextResponse
        res_fields = ModelContextResponse.model_fields
        res_annotations = get_type_hints(ModelContextResponse)
        
        # Check tools: Optional[List[Tool]] = None
        assert 'tools' in res_fields
        tools_type = res_annotations['tools']
        tools_str = str(tools_type).lower()
        tools_origin = get_origin(tools_type)
        # Optional[List[Tool]] is Union[List[Tool], None]
        assert (tools_origin is Union and 
                len(get_args(tools_type)) == 2)
        tools_args = get_args(tools_type)
        # One arg should be None, the other should be List[Tool]
        assert type(None) in tools_args or None in tools_args
        list_arg = next((arg for arg in tools_args if arg is not type(None) and arg is not None), None)
        assert list_arg is not None
        list_origin = get_origin(list_arg)
        assert list_origin is list or list_origin == list
        
        # Check result: Optional[Any] = None
        assert 'result' in res_fields
        result_type = res_annotations['result']
        result_str = str(result_type).lower()
        result_origin = get_origin(result_type)
        # Optional[Any] is Union[Any, None]
        assert (result_origin is Union and 
                len(get_args(result_type)) == 2)
        result_args = get_args(result_type)
        assert Any in result_args or type(None) in result_args or None in result_args

    except ImportError:
        pytest.fail("AssertionFailedError: Could not import models from app.models.")
    except AssertionError as e:
        pytest.fail(f"AssertionFailedError: {e}")
