# Learning architecture

## Principle

Do not teach "Transformer" as one opaque block. Expose one transformation at a time and tie every tensor to a physical Cart-Pole meaning.

## Stage A — transparent attention

Input sequence:
`s_0, s_1, ..., s_t`, where each `s` contains `x, x_dot, theta, theta_dot`.

Expose:
1. normalization (scale-only, per-field, no learned embedding)
2. fixed 4×4 Wq / Wk projections; V is the identity (the normalized token itself)
3. query-key dot products, scaled by sqrt(4)
4. a fixed (not learned) recency bias of `+1.20 × keyIndex` added to every score, before the causal mask
5. causal mask, then softmax
6. weighted values
7. context (per Query row)
8. a fixed feedback-gain dot product (`[1.50, 0.50, 8.00, 3.00] · context`) reduced to a scalar action score, then `10 · tanh(score)` as a single continuous force command — not a two-action categorical head

This stage (the toy/unlearned fallback) uses no trained weights at all; it is an intentionally transparent, hand-authored explainer, not a trained policy, and must stay clearly labelled as such wherever it is shown. It has no relation to Stage B's own head: Stage B projects tokens 4→8, adds learned positional embeddings, and runs LayerNorm + MLP blocks before a learned scalar action head — none of Stage A's fixed recency-bias/feedback-gain prior applies there.

## Stage B — learned causal Transformer policy

Train a small policy on Cart-Pole trajectories. Keep architecture small enough that browser inference and tensor inspection remain practical.

The explainer consumes the real intermediate tensors from the policy.

## Stage C — vision

Render frames, convert frames to patches/features, and form temporal visual tokens. Initially keep state input disabled so the learner can see why one frame is ambiguous about velocity.

## Stage D — multimodal state + vision

Represent state and visual observations as distinct token types, then compare simple early fusion with cross-attention only if the simpler baseline is insufficient.

## Non-goals for the first release

- large language models
- GPT-2 inference
- high-dimensional robotics
- claiming attention is a causal explanation of policy behaviour
- adding multimodal complexity before the state-only path is understandable
