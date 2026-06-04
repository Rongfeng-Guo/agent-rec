# Baseline Matrix

| Baseline | Runnable now | Needs GPU | Needs API key | Needs download | Current status | Command | Blocker |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Prompt-based IRA | No | No | Yes | Maybe | BLOCKED | `bash main.sh` | `main.sh` is referenced in README but not present in this checkout; config/API files are absent. |
| SFT baseline | Config only | Yes | No | Yes | BLOCKED | `cd LLaMA-Factory && bash gimo/{dataset}/sft/sft.sh` | Requires model weights, configured dataset paths, and training resources. |
| GPE | Script exists | Maybe | Yes | Maybe | BLOCKED | `cd GPE_HAP && python rewrite_v3.py --domain {dataset}` | Requires configured OpenAI-compatible endpoint/model and generated rollout inputs. |
| HAP | Partial | Maybe | Yes | Maybe | BLOCKED | `cd GPE_HAP && python refine_prompts_v2.py` | No end-to-end command documented in the current checkout. |
| CDPO | Config only | Yes | No | Yes | BLOCKED | `cd LLaMA-Factory && bash gimo/{dataset}/gimo/adpo_v1_sample1.sh` | Requires model weights, preference data, GPU, and LLaMA-Factory dependencies. |
| DriftAware structured memory | Yes | No | No | No | SMOKE_TEST_ONLY | `python -B -m user_simulator.evaluation.drift_memory_eval` | Deterministic toy benchmark; not a full GIMO rollout. |
| CritiqueScope memory | Yes | No | No | No | SMOKE_TEST_ONLY | `python -B -m user_simulator.evaluation.critique_scope_eval` | Deterministic controlled counterfactual proxy; not a full user simulator rollout. |
| Unified memory baselines | Yes | No | No | No | SMOKE_TEST_ONLY | `python -B -m user_simulator.evaluation.run_memory_baselines --modes none flat structured time_decay critiquescope --scenario-set deterministic --seeds 0 1 2 3 4 --output-dir outputs/memory_baselines` | Deterministic benchmark only. |

## Notes

- The current CPU/API-free path is complete for deterministic diagnosis.
- The original GIMO training path is preserved but not executed here because it
  requires data/model configuration and likely GPU training time.
- No full Recall/NDCG or CDPO result is claimed in this branch.
