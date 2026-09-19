# Learning architecture

## Principle

Do not teach "Transformer" as one opaque block. Expose one transformation at a time and tie every tensor to a physical Cart-Pole meaning.

## Stage A — transparent attention

Input sequence:
`s_0, s_1, ..., s_t`, where each `s` contains `x, x_dot, theta, theta_dot`.

Expose:
1. normalization
2. state embedding
3. Wq / Wk / Wv projections
4. query-key dot products
5. scale by sqrt(d_k)
6. softmax
7. weighted values
8. context
9. two-action head

This stage may use intentionally tiny hand-authored weights. It must be labelled as an explainer, not a trained policy.

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
