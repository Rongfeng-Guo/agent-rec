"""CritiqueScope parser with deterministic and optional LLM backends."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List

from user_simulator.state.critique_scope import infer_critiques


SYSTEM_PROMPT = """You parse interactive recommendation feedback into CritiqueScope JSON.
Return only a JSON array. Each object must include:
target, operation, reason, object_scope, temporal_scope, horizon, hardness,
confidence, promotion_condition.
Use temporal_scope values: next_slate, session, contextual, persistent.
Temporary fatigue and diversity requests must not become persistent by default.
"""


def parse_deterministic(utterance: str) -> List[dict]:
    return infer_critiques(utterance)


def parse_openai_compatible(
    utterance: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 60,
) -> List[dict]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM parser request failed: {exc}") from exc

    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("LLM parser must return a JSON array")
    return parsed


def parse_utterance(utterance: str, backend: str, **kwargs) -> List[dict]:
    if backend == "deterministic":
        return parse_deterministic(utterance)
    if backend == "openai":
        return parse_openai_compatible(utterance, **kwargs)
    raise ValueError(f"Unsupported parser backend: {backend}")


def load_inputs(path: str | None) -> Iterable[str]:
    if not path:
        return [
            "I have seen too much UFC lately. Switch it up for a bit.",
            "Please never recommend political content to me.",
            "Recommend something different but still related.",
            "Tonight I need a family-friendly dinner place.",
            "I do not want Windows anymore. Going forward, prioritize Mac laptops.",
        ]
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as file:
            utterances = []
            for line in file:
                item = json.loads(line)
                utterances.append(item["utterance"] if isinstance(item, dict) else str(item))
            return utterances
    return [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["deterministic", "openai"], default="deterministic")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    args = parser.parse_args()

    if args.backend == "openai" and (not args.base_url or not args.api_key or not args.model):
        raise SystemExit("openai backend requires --base-url, --api-key, and --model")

    rows = []
    for utterance in load_inputs(args.input):
        critiques = parse_utterance(
            utterance,
            args.backend,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
        rows.append({"utterance": utterance, "backend": args.backend, "critiques": critiques})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "rows": len(rows), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
