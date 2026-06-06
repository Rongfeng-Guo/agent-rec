from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import textwrap


DEBUG_DIR = Path("outputs/server184_gimo/prompt_ira_debug")


class JsonResponseError(ValueError):
    def __init__(
        self,
        status: str,
        message: str,
        raw_text: str,
        extracted_text: str | None = None,
        validation: dict | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.raw_text = raw_text
        self.extracted_text = extracted_text
        self.validation = validation or {}


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def find_first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise JsonResponseError(
            status="SCHEMA_MISMATCH",
            message="No JSON object start token found in model response.",
            raw_text=text,
            validation={"actual_schema": "no_json_object_start"},
        )

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    raise JsonResponseError(
        status="INCOMPLETE_JSON",
        message="Model response contains an opening JSON object but no balanced closing brace.",
        raw_text=text,
        extracted_text=text[start:],
    )


def write_debug_artifacts(*, raw_text: str, parsed_object: dict | None, validation: dict) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    if parsed_object is not None:
        (DEBUG_DIR / "extracted_response.json").write_text(
            json.dumps(parsed_object, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    validation_json = json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
    (DEBUG_DIR / "validation.json").write_text(validation_json, encoding="utf-8")
    validation_md = textwrap.dedent(
        f"""\
        # Prompt IRA Validation

        - status: {validation.get("status")}
        - debug_label: {validation.get("debug_label")}
        - expected_schema: {validation.get("expected_schema")}
        - actual_schema: {validation.get("actual_schema")}
        - parse_exception: {validation.get("parse_exception")}
        - response_format: `{json.dumps(validation.get("response_format"), ensure_ascii=False)}`
        - model_alias: {validation.get("model_alias")}
        - endpoint: {validation.get("endpoint")}
        - temperature: {validation.get("temperature")}
        """
    )
    (DEBUG_DIR / "validation.md").write_text(validation_md, encoding="utf-8")


def extract_first_json_object(text: str) -> tuple[dict, str]:
    raw_text = text if text is not None else ""
    stripped = raw_text.strip()
    if not stripped:
        raise JsonResponseError(
            status="EMPTY_RESPONSE",
            message="Model returned an empty response.",
            raw_text=raw_text,
        )

    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise JsonResponseError(
                status="SCHEMA_MISMATCH",
                message="Parsed JSON is not an object.",
                raw_text=raw_text,
                extracted_text=stripped,
                validation={"actual_schema": type(parsed).__name__},
            )
        return parsed, "VALID_JSON"
    except json.JSONDecodeError:
        pass

    fenced = strip_markdown_code_fence(stripped)
    if fenced != stripped:
        try:
            parsed = json.loads(fenced)
            if not isinstance(parsed, dict):
                raise JsonResponseError(
                    status="SCHEMA_MISMATCH",
                    message="Parsed fenced JSON is not an object.",
                    raw_text=raw_text,
                    extracted_text=fenced,
                    validation={"actual_schema": type(parsed).__name__},
                )
            return parsed, "RECOVERED_FROM_CODE_FENCE"
        except json.JSONDecodeError:
            pass

    candidate = find_first_balanced_json_object(fenced)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JsonResponseError(
            status="SCHEMA_MISMATCH",
            message=f"Failed to decode extracted JSON object: {exc}",
            raw_text=raw_text,
            extracted_text=candidate,
            validation={"parse_exception": repr(exc)},
        ) from exc
    if not isinstance(parsed, dict):
        raise JsonResponseError(
            status="SCHEMA_MISMATCH",
            message="Extracted JSON content is not an object.",
            raw_text=raw_text,
            extracted_text=candidate,
            validation={"actual_schema": type(parsed).__name__},
        )
    return parsed, "RECOVERED_FROM_PREFIX_SUFFIX_TEXT"


def parse_json_response(
    text: str,
    *,
    required_keys: list[str] | None = None,
    response_format: dict | None = None,
    debug_label: str | None = None,
    model_alias: str | None = None,
    endpoint: str | None = None,
    temperature=None,
    max_tokens=None,
) -> str:
    try:
        parsed, status = extract_first_json_object(text)
        missing_keys = [key for key in (required_keys or []) if key not in parsed]
        if missing_keys:
            raise JsonResponseError(
                status="SCHEMA_MISMATCH",
                message=f"Missing required keys: {missing_keys}",
                raw_text=text,
                extracted_text=json.dumps(parsed, ensure_ascii=False),
                validation={"missing_keys": missing_keys, "actual_schema": sorted(parsed.keys())},
            )
        validation = {
            "status": status,
            "debug_label": debug_label,
            "expected_schema": required_keys or [],
            "actual_schema": sorted(parsed.keys()),
            "parse_exception": None,
            "response_format": response_format,
            "model_alias": model_alias,
            "endpoint": endpoint,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        write_debug_artifacts(raw_text=text, parsed_object=parsed, validation=validation)
        return json.dumps(parsed, ensure_ascii=False)
    except JsonResponseError as exc:
        validation = {
            "status": exc.status,
            "debug_label": debug_label,
            "expected_schema": required_keys or [],
            "actual_schema": exc.validation.get("actual_schema"),
            "parse_exception": exc.validation.get("parse_exception") or str(exc),
            "response_format": response_format,
            "model_alias": model_alias,
            "endpoint": endpoint,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "missing_keys": exc.validation.get("missing_keys"),
        }
        parsed_object = None
        extracted = exc.extracted_text
        if extracted:
            try:
                maybe = json.loads(extracted)
                if isinstance(maybe, dict):
                    parsed_object = maybe
            except Exception:
                parsed_object = None
        write_debug_artifacts(raw_text=exc.raw_text, parsed_object=parsed_object, validation=validation)
        raise


def make_openai_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The openai package is required to instantiate OpenAIClient or OpenAIChatClient. "
            "JSON parsing helpers can be used without it."
        ) from exc
    return OpenAI(base_url=base_url, api_key=api_key)


class OpenAIClient:
    def __init__(self, base_url: str, api_key: str, model_path: str, response_format=None):
        self.client = make_openai_client(base_url=base_url, api_key=api_key)
        self.model_path = model_path
        self.response_format = response_format
        self.base_url = base_url

    def get_single_chat_completion(
        self,
        user_message: str,
        sys_prompt: str = "You are a helpful assistant",
        response_format=None,
        required_keys: list[str] | None = None,
        debug_label: str | None = None,
        temperature=None,
        max_tokens=None,
    ):
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_message}]
        effective_response_format = response_format or self.response_format
        effective_temperature = 0 if temperature is None and effective_response_format is not None else temperature
        effective_max_tokens = 256 if max_tokens is None and effective_response_format is not None else max_tokens
        request_kwargs = dict(
            model=self.model_path,
            messages=messages,
        )
        if effective_response_format is not None:
            request_kwargs["response_format"] = effective_response_format
        if effective_temperature is not None:
            request_kwargs["temperature"] = effective_temperature
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens
        completion = self.client.chat.completions.create(**request_kwargs)
        content = completion.choices[0].message.content
        if effective_response_format is not None:
            return parse_json_response(
                content,
                required_keys=required_keys,
                response_format=effective_response_format,
                debug_label=debug_label,
                model_alias=self.model_path,
                endpoint=self.base_url,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
        return content

    def get_multi_chat_completions(
        self,
        user_messages: list,
        sys_prompt: str = "You are a helpful assistant",
        response_format=None,
        required_keys: list[str] | None = None,
        debug_label: str | None = None,
        temperature=None,
        max_tokens=None,
    ):
        def fetch_response(user_message):
            return self.get_single_chat_completion(
                user_message,
                sys_prompt,
                response_format=response_format,
                required_keys=required_keys,
                debug_label=debug_label,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_response, user_messages))

        return results


class OpenAIChatClient:
    def __init__(self, base_url: str, api_key: str, model_path: str, response_format=None):
        self.client = make_openai_client(base_url=base_url, api_key=api_key)
        self.model_path = model_path
        self.response_format = response_format
        self.base_url = base_url

    def get_single_chat_completion(
        self,
        chat_messages: list,
        sys_prompt: str = "You are a helpful assistant",
        response_format=None,
        required_keys: list[str] | None = None,
        debug_label: str | None = None,
        temperature=None,
        max_tokens=None,
    ):
        messages = [{"role": "system", "content": sys_prompt}] + chat_messages
        effective_response_format = response_format or self.response_format
        effective_temperature = 0 if temperature is None and effective_response_format is not None else temperature
        effective_max_tokens = 256 if max_tokens is None and effective_response_format is not None else max_tokens
        request_kwargs = dict(
            model=self.model_path,
            messages=messages,
        )
        if effective_response_format is not None:
            request_kwargs["response_format"] = effective_response_format
        if effective_temperature is not None:
            request_kwargs["temperature"] = effective_temperature
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens
        completion = self.client.chat.completions.create(**request_kwargs)
        content = completion.choices[0].message.content
        if effective_response_format is not None:
            return parse_json_response(
                content,
                required_keys=required_keys,
                response_format=effective_response_format,
                debug_label=debug_label,
                model_alias=self.model_path,
                endpoint=self.base_url,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
        return content

    def get_multi_chat_completions(
        self,
        chat_messages: list[list],
        sys_prompt: str = "You are a helpful assistant",
        response_format=None,
        required_keys: list[str] | None = None,
        debug_label: str | None = None,
        temperature=None,
        max_tokens=None,
    ):
        def fetch_response(user_message):
            return self.get_single_chat_completion(
                user_message,
                sys_prompt,
                response_format=response_format,
                required_keys=required_keys,
                debug_label=debug_label,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_response, chat_messages))

        return results
