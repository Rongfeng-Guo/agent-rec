import json

import pytest

from user_simulator.persona.model.model import JsonResponseError, extract_first_json_object, parse_json_response


def test_direct_valid_json():
    text = '{"reason":"ok","response":"hello"}'
    parsed, status = extract_first_json_object(text)
    assert parsed["response"] == "hello"
    assert status == "VALID_JSON"


def test_json_inside_markdown_fence():
    text = "```json\n{\"reason\": \"ok\", \"response\": \"hello\"}\n```"
    parsed, status = extract_first_json_object(text)
    assert parsed["reason"] == "ok"
    assert status == "RECOVERED_FROM_CODE_FENCE"


def test_json_with_prefix_suffix_text():
    text = 'prefix {"reason":"ok","response":"hello"} suffix'
    parsed, status = extract_first_json_object(text)
    assert parsed["response"] == "hello"
    assert status == "RECOVERED_FROM_PREFIX_SUFFIX_TEXT"


def test_nested_json_object():
    text = '{"reason":"ok","response":"hello","meta":{"a":1,"b":{"c":2}}}'
    parsed, status = extract_first_json_object(text)
    assert parsed["meta"]["b"]["c"] == 2
    assert status == "VALID_JSON"


def test_incomplete_json_raises():
    text = '{"reason":"ok","response":"hello"'
    with pytest.raises(JsonResponseError) as excinfo:
        extract_first_json_object(text)
    assert excinfo.value.status == "INCOMPLETE_JSON"


def test_schema_mismatch_raises():
    text = '{"reason":"ok"}'
    with pytest.raises(JsonResponseError) as excinfo:
        parse_json_response(text, required_keys=["reason", "response"], debug_label="responser")
    assert excinfo.value.status == "SCHEMA_MISMATCH"


def test_empty_response_raises():
    with pytest.raises(JsonResponseError) as excinfo:
        extract_first_json_object("   ")
    assert excinfo.value.status == "EMPTY_RESPONSE"


def test_parser_does_not_invent_missing_fields():
    text = '{"reason":"only"}'
    with pytest.raises(JsonResponseError):
        parse_json_response(text, required_keys=["reason", "response"], debug_label="responser")
