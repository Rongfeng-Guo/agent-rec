# DriftAware-GIMO: Structured Memory for Interest Drift

This extension studies how a conversational recommendation agent should update
memory when a user supplements, revises, or cancels preferences across turns.
It is designed to plug into GIMO without changing the alignment pipeline.

## Memory strategies

| Method | Context state |
| --- | --- |
| Full History | Put the whole dialogue in the prompt. |
| Summary Memory | Use a running natural-language preference summary. |
| Retrieval Memory | Retrieve only history snippets related to the current command. |
| Structured Memory | Maintain explicit positive, negative, hard, and soft preference slots with confidence. |

## Structured memory operations

The new `StructuredMemory` state supports four operations:

| Operation | Use case |
| --- | --- |
| retain | A new turn confirms an existing preference. |
| merge | A user adds a compatible preference. |
| overwrite | A new preference replaces a previous value. |
| forget | A preference becomes stale after drift. |

The state is rendered as a compact prompt block:

```text
Structured memory for the current user:
Hard preferences:
- [hard/active/c=0.86/t=3] audience: family-friendly
Positive preferences:
- [positive/active/c=0.82/t=2] occasion: group dinner
Use active hard constraints first; treat low-confidence soft preferences as tentative.
```

## Metrics

| Metric | Meaning |
| --- | --- |
| Recovery Turns | Number of turns after drift before the recommendation satisfies the new target. |
| Stale Preference Violation Rate | Fraction of recommendations that still follow forgotten preferences. |
| Constraint Satisfaction Rate | Fraction of post-drift recommendations satisfying active hard constraints. |
| Success@K | Whether a target-matching item appears in the top K. |
| Token Cost | Estimated prompt words consumed by the memory strategy. |

## Quick benchmark

Run a model-free smoke test from the repository root:

```bash
python -m user_simulator.evaluation.drift_memory_eval
```

The default scenario mirrors a restaurant drift case:

1. The user asks for a group dinner.
2. The user adds a quietness preference.
3. The user changes the situation to bringing kids, so family-friendly becomes a hard constraint and quietness is forgotten.

Use `--scenarios` to pass a JSON or JSONL file with the same schema as
`DEFAULT_SCENARIOS` in `user_simulator/evaluation/drift_memory_eval.py`.

## Integration points

`UserAgentEnv` now accepts `memory_mode="structured"`. The default remains
`"none"` for backward compatibility.

```python
env = UserAgentEnv(
    persona_path="user_simulator/task/Yelp_test.jsonl",
    user_id=0,
    item_id=0,
    config_path="config/api_config.json",
    format_path="config",
    domain="restaurant",
    model_type="openai",
    memory_mode="structured",
)
```

In full GIMO experiments, replace the cue-based fallback with an LLM or
annotation-based memory extractor that emits explicit `updates` dictionaries.
The rest of the benchmark can remain unchanged.
