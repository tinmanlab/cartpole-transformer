# Cart-Pole Transformer

A beginner-first interactive lab for learning **Transformer attention by seeing it control Cart-Pole**.

The project starts deliberately small and now includes both a transparent fallback controller and a trained one-block, one-head causal Transformer whose real intermediate tensors are visualized in the browser.

## Live demo

**https://tinmanlab.github.io/cartpole-transformer/**

The live page provides **State**, **Vision**, **Fusion**, and **Compare** modes. State uses explicit simulator state, Vision hides explicit state and uses rendered frame history, Fusion places aligned state/vision tokens into one time-causal self-attention block, and Compare runs the three learned controllers in independent environments with an identical deterministic initial state and disturbance schedule. State mode falls back to the transparent toy controller only if its learned artifact cannot be loaded.

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

### 0.4 — State + vision fusion — complete
The simplest typed-token fusion is implemented without a separate cross-attention block. Each of eight aligned timestamps contributes two tokens:

- State token: normalized `[x, x_dot, theta, theta_dot]` + availability bit
- Vision token: 256 patch means + 256 Δpatch values + availability bit

This produces 16 typed tokens in one time-causal self-attention matrix. Same-time State↔Vision attention is allowed; all future timestamps are masked.

The clean fusion and state-only baseline both complete 500/500 steps. Vision-only reaches 329/500 mean steps. Under partial vision, adding state raises mean closed-loop duration from 177 to 500 steps. Under the deterministic biased noisy-state ablation, however, adding vision lowers average state-estimation MAE (0.080→0.058) but **worsens** closed-loop duration (146→117 steps). That negative result is retained instead of adding gating or cross-attention solely to improve the demo. See [docs/fusion-model.md](docs/fusion-model.md).

### 0.5 — Deterministic side-by-side replay — complete
Compare mode runs State, Vision, and Fusion in three independent Cart-Pole environments from the exact same initial condition:

`[x, x_dot, theta, theta_dot] = [0, 0, 0.08, 0]`

All three receive the same ±4 N disturbance pulses at the same simulation ticks. Every tick is recorded, so the run can be paused, scrubbed and replayed synchronously. If a controller falls, its failed pose is frozen while the common replay clock and the other controllers continue.

For the fixed 10 s replay currently used by the lab:
- State survives 10.0 s.
- Vision falls at 4.26 s.
- Fusion survives 10.0 s.

These numbers describe one deterministic replay configuration, not a general controller ranking. The UI therefore exposes raw survival, maximum/mean pole angle and control effort without declaring a winner.

### 0.6 — Atomic live tick synchronization — complete
Single-controller State, Vision, and Fusion views now use one atomic snapshot contract. The displayed Cart-Pole state, newest State raw token, newest sampled Vision/Fusion state, model intermediates, and displayed policy force all refer to the same current simulation tick.

The transition order is now explicit:

`current snapshot + current action → physics step → append new observations → recompute all model intermediates/action → render next snapshot`

When paused, **Step** advances exactly one 20 ms physics tick. Browser QA numerically verifies state/token equality and `force = 10*tanh(active action score)` before and after Step in all three single-controller modes.

### 0.7 — Exact one-step closed-loop decision trace — complete
The live single-controller modes now connect the Transformer calculation to the plant in one exact 20 ms transition:

`observation_t → controller score → policy force u_t → policy + disturbance → Cart-Pole dynamics → x_ddot / theta_ddot → Euler state_{t+1} → next policy force u_{t+1}`.

The plant intermediates are returned directly by the environment's transition function; the explainer does not recompute lookalike dynamics. When paused, pressing **Step** populates the trace with the exact transition that was executed. State, Vision, and Fusion all reuse the same plant trace while keeping their different controller observations.

### 0.8 — Comprehensive visual QA and responsive cleanup — complete
The Playwright QA loop now audits all major State, Vision, Fusion, Compare, and Decision Trace views at **1440, 1024, 768, and 390 px** widths.

The audit checks normal visible HTML text size, clipped text/content, major-container overflow, unintended sibling overlap, undersized controls, viewport escape, page-level horizontal overflow, and stacked-mobile Sankey behavior. State detail stages are individually opened and checked, including Embedding, Q/K/V, Attention, Residual/MLP, and Action.

The cleanup raised the normal typography floor to 10 px, made controls at least 28–30 px high, moved the two-panel lab to a stacked layout before 1024 px becomes constrained, reflowed State detail calculations before they clip, and replaced rotated narrow-screen arrows with real down-arrow layout elements.

The final strict sweep covers **48 mode/detail/viewport states** with zero layout errors or warnings, followed by manual review of the generated desktop/laptop/tablet/mobile screenshot set.

### Later
Additional replay scenarios, video history, partial observability, attention-head comparison, and extensions to more complex control tasks.

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

**v0.8 comprehensive visual QA is implemented.** State / Vision / Fusion / Compare / Decision Trace and every State detail stage are now swept at 1440/1024/768/390 px for typography, clipping, overlap, control size, viewport escape and horizontal overflow. The strict sweep covers 48 rendered states with zero errors/warnings, in addition to the existing state/vision/fusion/comparison/dynamics checks and screenshot-based manual review.

**v0.9 readability repair.** The sweep now runs at 320/390/768/1024/1440px and measures the *effective* on-screen size of SVG `<text>` (via each element's own `getScreenCTM`) instead of trusting declared font-size, since viewBox scaling was silently shrinking Cart-Pole diagram labels on narrow screens; `CartPoleView` now counter-scales those labels to stay readable at every width, and the numeric labels that duplicated the HTML readout were dropped from the SVG (the arrows stay). A curated set of essential explanations and readouts (state readout, sim controls, `.steps`/`.claim`/`.qkv-key`, stage/panel headings, attention/force readouts) carries a stricter 14px text / 44px touch-target floor; this is scoped to that curated set, not every label in the app. The Fusion pipeline's token grid and 16×16 attention matrix reflow/scroll locally instead of clipping or forcing the page wider, and a Fusion mode-switch bug that could cascade into unrelated test failures (reusing a possibly-fallen episode from Vision mode) is fixed. Full browser QA (320–1440px, all modes) is green.

## License

MIT. See [LICENSE](LICENSE).
