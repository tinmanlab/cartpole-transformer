# Cart-Pole Transformer

A beginner-first interactive lab for learning **Transformer attention by seeing it control Cart-Pole**.

The project starts deliberately small: before using a trained Transformer, it makes every step of attention visible and editable.

## Why Cart-Pole?

Text makes Q/K/V feel abstract. Cart-Pole gives each token a physical meaning:

```
state token s_t = [cart position x, cart velocity x_dot, pole angle theta, pole angular velocity theta_dot]
```

A learner can watch the pole move while following:

```
state -> embedding -> Q / K / V -> QK^T -> scale -> softmax -> weighted V -> context -> action
```

The first goal is not state-of-the-art control. The first goal is to make the calculation understandable.

## Learning modes

### 0.1 — State-only attention explainer
Small, inspectable single-head attention. Every number is visible. Sliders change Cart-Pole state and immediately update Q, K, V, scores, softmax weights, context and the toy left/right action.

### 0.2 — Learned Cart-Pole Transformer
Replace hand-picked weights with a tiny trained causal Transformer policy. Compare learned attention with the transparent toy model and verify actual episode return.

### 0.3 — Vision-only
Treat rendered Cart-Pole frames as visual tokens. Show how an image/frame becomes patches/features and how temporal attention uses several frames to infer motion.

### 0.4 — State + vision
Fuse explicit simulator state with visual observations. Compare state tokens, visual tokens, self-attention/cross-attention, and the resulting action.

### Later
Video history, missing/noisy sensors, partial observability, multimodal fusion strategies, attention-head comparison, and extensions to more complex control tasks.

## Beginner contract

The UI must always answer four questions:

1. **What went in?**
2. **What arithmetic happened?**
3. **What information received more attention, and why?**
4. **How did that change the action?**

No matrix is allowed to appear without a plain-language explanation and a concrete Cart-Pole interpretation.

## Upstream

The interaction and visual-explanation approach is inspired by and may selectively adapt components from:

- **Polo Club of Data Science — Transformer Explainer**
- https://github.com/poloclub/transformer-explainer
- Baseline inspected: `bfe50afba10b9b560b84143ee1107d977defa74f`

Useful upstream components include `QKV.svelte`, `Attention.svelte`, `AttentionMatrix.svelte`, `Embedding.svelte`, and `HeadStack.svelte`.

Transformer Explainer is MIT licensed. Any upstream-derived source must retain the applicable copyright and license notice. See [NOTICE.md](NOTICE.md).

## Architecture rule

Keep three layers separate:

- **Environment** — deterministic Cart-Pole dynamics / replay data.
- **Model** — toy attention or trained Transformer policy.
- **Explainer** — visualization only; it reads model intermediates instead of reimplementing hidden math.

This separation lets later state, image, video, and multimodal models reuse the same teaching UI.

## Status

Repository bootstrap is complete. The first implementation target is **v0.1 state-only attention explainer**.

## License

MIT. See [LICENSE](LICENSE).
