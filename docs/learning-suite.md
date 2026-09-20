# CartPole learning-suite presentation contract

This is a small, shared **presentation contract** for the three independent
CartPole teaching apps (PPO, Transformer, DiffusionPolicy). It is
documentation only — each repo implements it in its own UI code. There is no
shared framework, package, or runtime dependency between the three repos,
and this file is not a single source of truth for any algorithm's numbers:
each app computes and displays only its own real values.

PPO, Transformer, and DiffusionPolicy are not three members of one
"algorithm family" competing on the same axis: **PPO is a learning method**
(how a policy is optimized from experience), **Transformer here is a network
architecture** (a sequence model used to produce one control decision), and
**DiffusionPolicy is a conditional generative policy** (an action sampled by
iterative denoising). Naming all three in one sentence does not mean they do
the same kind of work — see the per-stage table below for what each one
actually shows.

## Four stages, one reading direction

Every view that explains "how did this decision get made" presents the same
four stages, left-to-right (top-to-bottom on narrow screens):

1. **Input / 입력**
2. **Calculation / 계산**
3. **Action / 행동**
4. **Result / 결과**

A view may add its own detail behind any stage, and the concrete meaning of
each stage differs by algorithm (see below). Keep this order and these four
names as the shared orientation — a reader who knows one repo's tabs should
recognize the other two — but where a repo's own lesson genuinely needs a
different split (e.g. a stage that has no honest single value for that
algorithm), say so explicitly in that repo's own docs rather than forcing a
mismatched quantity into the shared label. What must not happen is silently
reusing one label for two unrelated things in the same view, or renaming a
label without recording why.

## What each stage actually is, per repo

The three repos fill these four labels with different real quantities.
Showing them side by side must not imply the quantities are equivalent:

|  | Input | Calculation | Action | Result |
|---|---|---|---|---|
| **PPO** | a recorded experience (or batch of them) from a rollout buffer | the learning signals/loss computed from that batch | the recorded action that was actually chosen at rollout time (a physical action, already in the past) — the optimizer update computed from the batch is a *separate* quantity shown alongside it, never itself a physical action | the policy's before/after state from that **whole minibatch update** — never presented as if one experience alone caused it |
| **Transformer** | the real observation for one live tick | real attention weights over the token history for that tick | the real force command produced for that tick | one real physical transition (the plant's next state) |
| **DiffusionPolicy** | the real observation conditioning the sample | the model's native planning cycle: a sampled action-plan produced by iterative denoising | one step executed from that plan (not the whole plan) | the executed prefix of the plan, or a re-observed plant state — not a full-plan claim |

Transformer's Result is the only one of the three that is a single physical
plant transition. PPO's Result is a policy comparison across a batch, not a
plant outcome. DiffusionPolicy's Result is about what was actually executed
or re-observed, not the plan itself. If a view cannot honestly fill a stage
with a real quantity from its own algorithm, it must say so rather than
substitute a placeholder.

## Vocabulary that must stay distinct

These terms describe different things and must never share a label or be
merged into one number:

- **Attention weights** (Transformer, self-attention over tokens) vs.
  **action probabilities** (a policy's action distribution) vs.
  **sampled plans** (DiffusionPolicy's denoised trajectory/action sequence).
- **Sampler steps** (DiffusionPolicy's denoising iterations) vs. **physical
  ticks** (the environment's real 20 ms integration steps). A view must
  label which one a "step" counter refers to.
- **Force command** (what the controller/policy output, before actuation),
  **actuator-delivered force** (what the actuator itself physically
  produces — may differ from the command, e.g. under saturation or a
  quantized actuator model), and **total plant input** (actuator-delivered
  force combined with any external disturbance, i.e. what the plant
  equation actually integrates) are three separate quantities. Delivered
  force is not defined as "command plus disturbance" — the disturbance is
  summed in afterward, at the plant-input stage, not at the actuator.

## Live / frozen / recorded / replay

Every view must explicitly label its data source as one of:

- **live** — currently advancing with the simulation clock.
- **frozen** — a live view paused on a specific real tick, not advancing.
- **recorded** — data captured from a past run (e.g. a training rollout),
  not from the currently visible plant.
- **replay** — played-back recorded transitions; this is playback of a
  recorded trace and does not necessarily recompute the policy that
  produced it.

Alongside the label, show the concrete identifier for the shown event (a
tick number, episode/step index, or similar) — not just the word "frozen"
with no way to tell which instant it refers to.

## Visual fundamentals (shared, on each repo's existing light theme)

No new design system or shared CSS package — each repo keeps its existing
styles and only aligns these floors:

- Body text: 16px baseline (native default is fine).
- Core/essential explanations, readouts, and control labels — including any
  captured-event badge or identifier — are **≥14px**. Compact indices inside
  dense matrices/grids may stay smaller than 14px only if the same real
  value is also shown somewhere readable at ≥14px.
- Primary interactive controls (play/pause/step/reset/apply-style buttons):
  **≥44px** touch target.
- A four-stage navigator (Input/Calculation/Action/Result) collapses to a
  compact 2×2 grid below the app's existing mobile breakpoint, never to a
  single unreadable row or a hidden overflow list.

## Review guidance

- Each stage must expose at least one actual arithmetic witness in place —
  real numbers computed from the frozen/live/recorded/replay data the stage
  is labeled with — not a "see above" pointer to a detached detail panel.
  An advanced full detail view may still exist alongside it as a single,
  non-duplicated instance; it supplements the in-place witness, it does not
  replace it.
- Tests for a stage must verify the frozen/live/recorded/replay provenance
  of what's shown, that real (non-placeholder) values are visible, that the
  displayed source semantics match the data-source label, and keyboard/focus
  behavior — not just that some fixed number of buttons or panels exist.

## Primary guide vs. reference material

A view's primary teaching surface (the plant/simulation and the one guide a
learner is actively following) stays adjacent and reachable without scrolling
past secondary material. Everything else — a default overview of the same
pipeline the guide already explains, a secondary/more-detailed replay of a
physical transition, static reading-order cards — is reference: reachable
through one named, keyboard-operable native disclosure (`<details>`), closed
by default, and never displacing or duplicating the primary guide's own
in-place arithmetic. Collapsing or reopening a reference must never change
which model, state, selection, or captured event is being shown.

## Explicitly out of scope for this contract

This contract governs presentation only. It does not require, and this
slice does not add, any cross-repo controlled-input-perturbation study or
any comparative learning-outcome analysis across PPO/Transformer/
DiffusionPolicy — that is future work, tracked separately, not implied by
any wording above.
