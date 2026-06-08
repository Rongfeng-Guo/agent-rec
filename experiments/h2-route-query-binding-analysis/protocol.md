# H2: Route/Query Binding Error Analysis

- Status: protocol active
- Type: diagnostic tooling and experiment design
- Claim status: not a model-performance claim

## Question

After the H1 diagnostic repair shows high prefix-1 route coverage for many
fusion policies, which failure mode dominates the remaining misses?

- route miss or missing route diagnostic;
- route hit but candidate-pool miss;
- candidate-pool hit but ranking outside top-k.

## Motivation

H1 corrected fusion `RouteHitRate` accounting without changing `Recall@50`.
The fixed ablation shows that several route-source fusion policies hit the true
prefix-1 route on most cold-like validation samples, but still have low final
Recall. The next useful step is to quantify whether the remaining loss is mostly
candidate generation/query binding or final ranking.

## Tooling Change

Add:

```text
scripts/oracle_route_memory/analyze_route_query_binding_errors.py
```

The script reads explicit selector `selector_rows.json` files and writes:

- `error_summary.json`
- `error_summary.csv`
- `error_summary.md`

Each row is classified as:

- `hit_at_k`
- `route_miss_or_unreported`
- `route_hit_pool_miss`
- `pool_hit_rank_miss`

Default grouping is `split policy_name domain` at `top_k=50`.

## First Run

Use the H1 diagnostic-fix selector rows:

```bash
/home/grf/.conda/envs/gdpo/bin/python3 \
  scripts/oracle_route_memory/analyze_route_query_binding_errors.py \
  --selector-rows outputs/oracle_route_memory/validation_fusion_selector_v3_route_source_ablation_diagfix_20260608/selector_rows.json \
  --output-dir outputs/oracle_route_memory/route_query_binding_error_analysis_h2_20260608 \
  --top-k 50 \
  --group-by split policy_name domain
```

## Validation Gates

Run:

```bash
PYTHONPATH=/home/grf/agent-rec /home/grf/.conda/envs/gdpo/bin/python3 -m pytest \
  /home/grf/agent-rec/tests/test_route_query_binding_error_analysis.py \
  /home/grf/agent-rec/tests/test_late_bound_router.py

/home/grf/.conda/envs/gdpo/bin/python3 -m compileall \
  /home/grf/agent-rec/scripts/oracle_route_memory/analyze_route_query_binding_errors.py
```

## Next Target

H3 should target candidate-pool retrieval inside high-hit prefix-1 buckets first.
Late-bound reranking is still relevant, but H2 shows that many high-route-hit
policies lose the target before ranking can help.

## First Run Result

Output:

```text
outputs/oracle_route_memory/route_query_binding_error_analysis_h2_20260608
```

Summary:

- `row_count`: `3971`
- `summary_count`: `76`
- Cold-like `fusion_domain_prior_predicted_history_vote_rrf`:
  `RouteHitRate=0.977941`, `CandidatePoolHitRate=0.044118`,
  `RouteHitPoolMissRate=0.933824`, `PoolHitRankMissRate=0.000000`.
- Cold-like `predicted_route_p1_top4`: `RouteHitRate=0.941176`,
  `CandidatePoolHitRate=0.154412`, `RouteHitPoolMissRate=0.786765`,
  `PoolHitRankMissRate=0.117647`.
- Game-domain top-4 route policies often have `RouteHitRate=1.0000` but `CandidatePoolHitRate=0.1831` or lower.

Interpretation: H2 supports the candidate-pool/query-binding bottleneck. The
next intervention should improve target entry into the per-route candidate pool
before spending another round on route classifier accuracy.
