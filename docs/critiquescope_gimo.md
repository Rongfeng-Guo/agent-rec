# CritiqueScope-GIMO: Scope-Aware Critique Alignment

CritiqueScope models natural-language feedback as an exposure-conditioned
intervention request rather than a durable user preference label.

The core question is:

```text
Is this utterance describing the user, or correcting the current recommendation policy?
```

## Critique schema

Instead of only parsing feedback into positive and negative preferences, the
parser should emit:

```json
{
  "target": "UFC content",
  "operation": "attenuate",
  "reason": "exposure fatigue",
  "object_scope": "category",
  "temporal_scope": "session",
  "horizon": 5,
  "hardness": "soft",
  "confidence": 0.82,
  "promotion_condition": "never"
}
```

| Field | Values |
| --- | --- |
| `operation` | `promote`, `attenuate`, `filter`, `diversify`, `explore`, `rollback` |
| `object_scope` | `item`, `attribute`, `category`, `creator`, `slate`, `global` |
| `temporal_scope` | `next-slate`, `session`, `contextual`, `persistent` |
| `horizon` | Number of future slates/turns where the critique remains active. |
| `promotion_condition` | Evidence needed before the critique can enter slow memory. |

## Fast and slow memory

`CritiqueScopeMemory` keeps two channels:

| Channel | Contents |
| --- | --- |
| Fast Memory | Temporary critiques, fatigue, session context, and slate-level diversity requests. |
| Slow Memory | Durable constraints and repeatedly validated long-term preferences. |

Temporary critiques decay by their semantic horizon. Persistent language such as
“never recommend political content” enters slow memory immediately. Soft
complaints such as “too much UFC lately” remain in fast memory and should not
contaminate the durable profile unless the promotion condition is satisfied.

## Counterfactual uplift

The reward target should measure the value of following the critique, not only
the absolute satisfaction after a recommendation:

```text
instruction_uplift =
  V(s_t, follow critique, c_t) - V(s_t, ignore critique, c_t)
```

For diagnostics, compare three branches from the same state:

| Branch | Meaning |
| --- | --- |
| Follow | Execute the scoped critique as intended. |
| Ignore | Keep recommending as if the critique was not issued. |
| Over-apply | Treat a temporary critique as a permanent preference update. |

This makes over-correction visible: a system can look good immediately after
temporary attenuation while hurting long-term utility by permanently suppressing
content the user still likes.

## Metrics

| Metric | Meaning |
| --- | --- |
| Instruction Uplift | Follow-branch value minus ignore-branch value. |
| Over-Application Regret | Follow-branch value minus over-applied branch value. |
| Over-Correction Regret | Post-expiry value lost by suppressing still-relevant content. |
| Memory Contamination Rate | Temporary critiques incorrectly stored in slow memory. |
| Token Cost | Prompt words consumed by the memory representation. |

## Quick benchmark

Run without API keys:

```bash
python -B -m user_simulator.evaluation.critique_scope_eval
```

The default benchmark includes:

1. Temporary UFC fatigue.
2. Persistent political-content filtering.
3. A diversity request that should not become a negative preference.

## GIMO integration

The simulator now supports:

```python
env = UserAgentEnv(
    persona_path="user_simulator/task/Yelp_test.jsonl",
    user_id=0,
    item_id=0,
    config_path="config/api_config.json",
    format_path="config",
    domain="restaurant",
    model_type="openai",
    memory_mode="critiquescope",
)
```

For full experiments, use a strong LLM parser to emit the critique schema, then
feed follow/ignore/over-apply rollouts into GIMO/CDPO preference construction.
