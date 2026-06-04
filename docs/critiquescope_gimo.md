# CritiqueScope-GIMO: Scope-Aware Critique Alignment

## Motivation

CritiqueScope models natural-language feedback as an exposure-conditioned
intervention request rather than a durable user preference label.

The core question is:

```text
Is this utterance describing the user, or correcting the current recommendation policy?
```

## Feedback Is Not Always Long-Term Preference

The same negative surface form can mean different things:

| Feedback | Scope interpretation | Correct update |
| --- | --- | --- |
| "Too much UFC lately." | Temporary fatigue after recent exposure. | Fast-memory attenuation with a short horizon. |
| "Never recommend political content." | Durable dislike or safety constraint. | Slow-memory persistent filter. |
| "Show something different." | Slate-level diversity request. | Diversify/explore next slate; do not blacklist the old category. |
| "Tonight I need a family place." | Session context. | Session-scoped promotion that expires. |
| "I am switching from Windows to Mac." | Genuine drift. | Roll back old preference and promote the new one. |

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

## Promotion, Decay, and Rollback

CritiqueScope uses explicit lifecycle rules:

- Persistent instructions and hard constraints enter slow memory immediately.
- Temporary fatigue and diversity requests default to `promotion_condition="never"`.
- Repeated evidence can promote a critique only if the parser marks it as
  promotable and the configurable confidence/evidence thresholds are met.
- Fast-memory critiques decay by semantic horizon.
- Positive behavior on an attenuated target triggers rollback, preventing a
  temporary complaint from becoming an accidental long-term ban.

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

## Baselines

| Method | Description |
| --- | --- |
| None | Ignores memory updates. |
| Flat Memory | Promotes every critique to slow memory. |
| Structured Memory | Stores preference buckets but does not model critique scope. |
| Time Decay | Applies a uniform temporary horizon to all critiques. |
| CritiqueScope | Uses operation, object scope, temporal scope, horizon, promotion, and rollback. |

## Quick benchmark

Run without API keys:

```bash
python -B -m user_simulator.evaluation.critique_scope_eval
python -B -m user_simulator.evaluation.critique_scope_eval --scenario-set noisy
```

Validate scenario schemas:

```bash
python -B -m user_simulator.evaluation.validate_critique_scenarios --scenario-set deterministic
python -B -m user_simulator.evaluation.validate_critique_scenarios --scenario-set noisy
```

Parse feedback with the deterministic fallback:

```bash
python -B -m user_simulator.evaluation.critique_parser \
  --backend deterministic \
  --output outputs/parser_smoke/parsed.jsonl
```

Normalize existing rollout utilities and build uplift pairs:

```bash
python -B -m user_simulator.evaluation.critique_rollout_adapter \
  --output-dir outputs/rollout_adapter_smoke
```

The default benchmark includes:

1. Temporary UFC fatigue.
2. Persistent political-content filtering.
3. A diversity request that should not become a negative preference.

To build CDPO/DPO-style preference pairs from the same counterfactual branches:

```bash
python -B -m user_simulator.evaluation.critique_uplift_pairs --output critique_pairs.jsonl
```

Each pair prefers the follow branch over either the ignore branch or the
over-apply branch when the counterfactual uplift is positive.

Aggregate runner outputs for paper-style tables:

```bash
python -B -m user_simulator.evaluation.summarize_memory_baselines \
  --input outputs/memory_baselines/summary.csv \
  --output-dir outputs/memory_baselines/aggregate
```

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

## Limitations

- The deterministic benchmark is a controlled smoke test, not evidence of
  real-world recommendation quality.
- Counterfactual uplift is implemented as a rollout proxy with fixed utilities;
  it is not a full causal identification strategy.
- The current parser fallback is cue based. Full experiments should replace it
  with an LLM parser or annotated critique labels.
- No SFT/GPE/HAP/CDPO training result is claimed by this branch.

## Next Steps

1. Add an LLM critique parser that emits the schema above.
2. Generate follow/ignore/over-apply rollouts from real GIMO agents.
3. Feed real rollout JSONL into `critique_rollout_adapter.py`.
4. Convert uplift-positive branches into CDPO pairs with
   `user_simulator.evaluation.critique_uplift_pairs`.
5. Evaluate transfer under multiple user simulators and small human checks.
