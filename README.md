# Macro Chip placement

;tldr see placer.py

Beats both the SA proxy baseline and greedy placer across IBM ICCAD04 and NanGate45 benchmarks.

A two-stage macro placement engine combining surrogate multi-objective Bayesian optimization with true Tier-1 proxy (PlacementCost) survivor selection. The core insight: run cheap surrogate SA across multiple scalarizations to explore a sparse Pareto frontier, then let the real oracle pick winners — rather than committing to a single objective from the start.

Pipeline
Seed → Nesterov Analytical (pre-pool) → ePlace/DREAMPlace-style warmstart
     → Multi-mode SA (parallel Pareto scalarizations)
     → Pool oracle scoring → DPP diversity survivors
     → IncreMacro-lite (periphery/hotspot nudges)
     → Hotspot evacuation → Fresco congestion repack → Cool-bin relocation
     → Hot/cool macro swap → Axis-greed refinement (3 passes)
     → Multi-pole congestion evacuation → WL refinement
     → Channel enforcement (NG45) → Orientation search → Polish

Key Innovations
1. Multi-Mode Surrogate Pareto Exploration
Five surrogate scalarizations run in parallel, each trading off wirelength, density, routing balance, and periphery relief differently. A Tier-1 oracle then selects the best survivor — not the surrogate's own ranking. Surrogate weights are optionally refitted from in-run oracle labels via pairwise logistic (RankNet-style) regression.
2. Multi-Pole Congestion Evacuation (primary innovation)
PlacementCost exposes a multi-modal max(H,V) routing stress field. Rather than reacting to a single hotspot centroid (as in MaskPlace-style masks), this placer erects K spatially-separated pole sites on the scoring grid and superimposes inverse-power repulsive drifts on macro centroids. Extensions include:

Pole line search: probes multiple step lengths along the field vector, oracle-gated
Paired pole drift: coordinates two macros for joint congestion escape (correlated move)

3. Halo-Weighted Legalization (AutoDMP-style)
Before any search begins, macros are legalized with inflated half-extents matching the analytical density grid — ensuring the surrogate's density penalty aligns with Tier-1 spacing requirements from the first iteration.
4. ePlace / DREAMPlace-style Analytical Warmstart
A WL + density electrostatic relaxation warm-starts coordinates before SA, with adaptive handoff (stops when density violations fall and WL plateaus). For comp/full profiles, a Nesterov analytical global placement pass precedes ePlace.
5. Quality-Diversity Survivor Selection (DPP-style)
After oracle scoring, survivors are chosen via a greedy quality-scaled Deterministic Point Process — maximizing both layout quality and geometric diversity in placement space. This prevents collapsed surrogate modes from all feeding the same refinement hypothesis.
6. IncreMacro-lite with Incremental Net HPWL
Post-SA oracle refinement updates only the nets incident to moved macros (not full recomputation), enabling more oracle calls within the same time budget. Macro selection is biased toward routing-hot bins and high-fanout nets.



