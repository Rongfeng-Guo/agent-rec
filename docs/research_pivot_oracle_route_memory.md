# Research Pivot: oracle_route + memory

## What Already Exists

The current workspace is a mixed research directory. The main checked-in repo is `agent-rec`, and it already contains:

- CritiqueScope / GIMO / CritiqueWorld code paths and evaluation reports.
- Closed-loop outputs under `outputs/closed_loop_*`.
- Memory baseline summaries under `outputs/memory_baselines*`.
- Existing task splits and metadata-style user/item records under `user_simulator/task` and `user_simulator/raw_data`.

These assets are useful as context, but they do not yet validate the core cold-item routing question we now care about.

## Why Freeze CritiqueScope / GIMO / reranking

Current CritiqueScope / GIMO / inference-time reranking work is better treated as a side branch rather than the main recommendation thesis. It has already produced artifacts, diagnostics, and implementation experience, but it is not the cleanest path for answering whether cold-item identity recovery can be helped by a routing bottleneck plus external memory.

Current CritiqueScope / GIMO / inference-time reranking direction is now frozen as a negative or inconclusive finding. We will not continue tuning, scaling training, or chasing assertion passes in that direction. The main line now shifts to oracle_route + memory: instead of asking a generative recommender to emit the full cold-item SID directly, we ask whether a coarse route can narrow the candidate space to the right region, and then let an external item memory finish the item identity binding.

## What We Are Not Continuing Right Now

- No more new GIMO work for this round.
- No more CritiqueScope iteration for this round.
- No more GenRecEdit model-editing work for this round.
- No new TIGER training for this round.
- No new push to make inference-time reranking the primary result.

## Why Pivot to oracle_route + memory

The original generative formulation implicitly asks the model to solve:

`History -> full SID -> item`

That is brittle for cold items because the full SID pattern of a target cold item may never have appeared in training. The model then has a natural bias toward seen SID patterns.

The new route-conditioned formulation is:

`History -> route -> dynamic item memory -> item`

Here, `route` is only a short SID prefix, and the dynamic item memory is the current catalog embedding bank. This isolates a cleaner question: if the route is correct, can the item memory recover the true cold target?

## MVP Hypothesis

The MVP hypothesis for this round is:

If oracle route is given correctly, metadata memory retrieval should recover cold target items much better than no-route metadata retrieval. If oracle_route + memory clearly beats metadata-only retrieval, route-conditioned memory binding has positive signal. If oracle_route + memory is still weak, then the bottleneck is more likely item/user representation quality rather than full SID generation.
