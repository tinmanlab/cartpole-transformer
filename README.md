# Cart-Pole Transformer

A beginner-first interactive lab for learning **Transformer attention by seeing it control Cart-Pole**.

The project starts deliberately small and now includes both a transparent fallback controller and a trained one-block, one-head causal Transformer whose real intermediate tensors are visualized in the browser.

## Live demo

**https://tinmanlab.github.io/cartpole-transformer/**

The live page provides two observation modes: **State** uses the learned state Transformer, while **Vision** hides explicit simulator state from the controller and runs a trained pixels-only temporal Transformer. State mode falls back to the transparent toy controller only if its learned artifact cannot be loaded.

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

### 0.2 — Learned Cart-Pole Transformer — complete
A deterministic 1-block, 1-head causal Transformer (sequence 8, d_model 8, FFN 16) is behavior-cloned from a simple linear state-feedback teacher. The browser directly executes the serialized learned weights and exposes learned embedding, Q/K/V, pre-mask scores, causal mask, softmax weights, weighted V/context and action.

Closed-loop evaluation: 80 randomized episodes with occasional ±4 N disturbance pulses, 500-step horizon. Both the learned Transformer and the simpler linear baseline achieved 500/500 mean, median and minimum steps (100% completion). See [docs/learned-model.md](docs/learned-model.md).

### 0.3 — Vision-only — complete
The controller can switch to a genuinely pixels-only path:

`32×32 frame → 16×16 patch means + Δpatch → 512D visual input → learned 24D frame token → 8-frame causal temporal attention → inferred [x, x_dot, theta, theta_dot] → transparent force law`.

Eight frames are sampled 60 ms apart, spanning 0.42 s. Simulator state is withheld from the controller and is available only behind an explicit ground-truth teaching toggle.

Validation shows why temporal context matters: repeating the latest frame instead of using real history increases x_dot MAE from 0.145 to 0.194 m/s and theta_dot MAE from 0.147 to 0.220 rad/s. Closed-loop mean episode length is 321/500 steps with full visual history versus 121/500 with the latest frame repeated. This is an educational temporal-vision result, not a claim that the visual policy matches the state policy. See [docs/vision-model.md](docs/vision-model.md).

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

The current visualization directly vendors/adapts the upstream `VectorCanvas.svelte`, `MatrixSvg.svelte`, generic gradient/path/redraw machinery from `Sankey.svelte`, and the staged attention expansion/GSAP pattern from `AttentionMatrix.svelte`, pinned to commit `bfe50afba10b9b560b84143ee1107d977defa74f`. Cart-Pole-specific code adapts the learned tensors, time-token selection, and DOM selectors.

Transformer Explainer is MIT licensed. File-level attribution and the exact reuse boundary are recorded in [NOTICE.md](NOTICE.md).

## Architecture rule

Keep three layers separate:

- **Environment** — deterministic Cart-Pole dynamics / replay data.
- **Model** — toy attention or trained Transformer policy.
- **Explainer** — visualization only; it reads model intermediates instead of reimplementing hidden math.

This separation lets later state, image, video, and multimodal models reuse the same teaching UI.

## Status

**v0.3 state + vision learning modes are implemented.** State mode exposes the learned Transformer down to selected-cell arithmetic. Vision mode hides explicit simulator state, renders eight synchronized frames, exposes patch and Δpatch features, learned frame tokens, temporal attention, inferred motion/state, and a latest-frame ablation. Deterministic training, model/runtime checks, desktop/mobile browser QA, screenshot artifacts, and Pages deployment are automated. The next frontier is **v0.4 state + vision fusion**, but only after preserving the current simple two-mode baseline as the comparison reference.

## License

MIT. See [LICENSE](LICENSE).
