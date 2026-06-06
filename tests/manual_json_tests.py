from user_simulator.persona.model.model import JsonResponseError, extract_first_json_object, parse_json_response


def expect(condition, label):
    if not condition:
        raise AssertionError(label)


def main():
    cases = []

    parsed, status = extract_first_json_object('{"reason":"ok","response":"hello"}')
    expect(parsed["response"] == "hello" and status == "VALID_JSON", "test_direct_valid_json")
    cases.append("test_direct_valid_json")

    parsed, status = extract_first_json_object("```json\n{\"reason\":\"ok\",\"response\":\"hello\"}\n```")
    expect(parsed["reason"] == "ok" and status == "RECOVERED_FROM_CODE_FENCE", "test_json_inside_markdown_fence")
    cases.append("test_json_inside_markdown_fence")

    parsed, status = extract_first_json_object('prefix {"reason":"ok","response":"hello"} suffix')
    expect(parsed["response"] == "hello" and status == "RECOVERED_FROM_PREFIX_SUFFIX_TEXT", "test_json_with_prefix_suffix_text")
    cases.append("test_json_with_prefix_suffix_text")

    parsed, status = extract_first_json_object('{"reason":"ok","response":"hello","meta":{"a":1,"b":{"c":2}}}')
    expect(parsed["meta"]["b"]["c"] == 2 and status == "VALID_JSON", "test_nested_json_object")
    cases.append("test_nested_json_object")

    try:
        extract_first_json_object('{"reason":"ok","response":"hello"')
        raise AssertionError("test_incomplete_json_raises")
    except JsonResponseError as exc:
        expect(exc.status == "INCOMPLETE_JSON", "test_incomplete_json_raises")
    cases.append("test_incomplete_json_raises")

    try:
        parse_json_response('{"reason":"ok"}', required_keys=["reason", "response"], debug_label="responser")
        raise AssertionError("test_schema_mismatch_raises")
    except JsonResponseError as exc:
        expect(exc.status == "SCHEMA_MISMATCH", "test_schema_mismatch_raises")
    cases.append("test_schema_mismatch_raises")

    try:
        extract_first_json_object("   ")
        raise AssertionError("test_empty_response_raises")
    except JsonResponseError as exc:
        expect(exc.status == "EMPTY_RESPONSE", "test_empty_response_raises")
    cases.append("test_empty_response_raises")

    try:
        parse_json_response('{"reason":"only"}', required_keys=["reason", "response"], debug_label="responser")
        raise AssertionError("test_parser_does_not_invent_missing_fields")
    except JsonResponseError:
        pass
    cases.append("test_parser_does_not_invent_missing_fields")

    for case in cases:
        print(case)


if __name__ == "__main__":
    main()
