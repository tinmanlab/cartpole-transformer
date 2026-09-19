#!/usr/bin/env python3
"""Train the simplest adequate state+vision typed-token fusion Transformer.

Per sampled time step:
  state token  = normalized [x, x_dot, theta, theta_dot] + availability bit
  vision token = patch means + delta patches + availability bit

Eight aligned time steps produce 16 typed tokens. A time-causal mask allows
state/vision tokens at the same timestamp to attend to each other while blocking
future timestamps. The model predicts normalized state and uses the same fixed,
transparent state-feedback law to produce force.

Training includes modality degradation so the fusion mode can be evaluated
against state-only and vision-only baselines under missing/noisy observations.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20260919
SEQ_LEN = 8
TOKEN_COUNT = SEQ_LEN * 2
FRAME_SIZE = 32
PATCH_SIZE = 2
GRID_SIZE = FRAME_SIZE // PATCH_SIZE
PATCH_DIM = GRID_SIZE * GRID_SIZE
VISION_DIM = PATCH_DIM * 2
STATE_DIM = 4
D_MODEL = 24
D_FF = 48
TAU = 0.02
FRAME_STRIDE = 3
BUFFER_LEN = (SEQ_LEN - 1) * FRAME_STRIDE + 1
STATE_SCALE = np.array([2.4, 3.0, 0.38, 3.5], dtype=np.float32)
EXPERT_GAIN = np.array([1.5, 0.5, 8.0, 3.0], dtype=np.float32)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "public" / "model" / "fusion-transformer.json"
DOC_PATH = ROOT / "docs" / "fusion-model.md"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


def physics_step(state: np.ndarray, control: float, disturbance: float = 0.0) -> np.ndarray:
    gravity, mass_cart, mass_pole, half_len = 9.8, 1.0, 0.1, 0.5
    total_mass = mass_cart + mass_pole
    pole_mass_length = mass_pole * half_len
    force = float(np.clip(control, -10.0, 10.0) + disturbance)
    x, xdot, theta, thetadot = [float(v) for v in state]
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + pole_mass_length * thetadot * thetadot * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        half_len * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass)
    )
    x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass
    return np.array(
        [x + TAU * xdot, xdot + TAU * x_acc, theta + TAU * thetadot, thetadot + TAU * theta_acc],
        dtype=np.float32,
    )


def terminal(state: np.ndarray) -> bool:
    return abs(float(state[0])) > 2.4 or abs(float(state[2])) > 0.38


def expert_force(state: np.ndarray) -> float:
    z = state / STATE_SCALE
    return 10.0 * math.tanh(float(np.dot(EXPERT_GAIN, z)))


def set_pixel(frame: np.ndarray, x: int, y: int, value: float) -> None:
    if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
        frame[y, x] = max(float(frame[y, x]), float(value))


def render_frame(state: np.ndarray) -> np.ndarray:
    x, _, theta, _ = [float(v) for v in state]
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    frame[27, 1 : FRAME_SIZE - 1] = 0.15

    cx = int(round(16 + np.clip(x / 2.4, -1.0, 1.0) * 12))
    for py in range(23, 27):
        for px in range(cx - 4, cx + 5):
            set_pixel(frame, px, py, 0.72)

    pivot_x, pivot_y = cx, 23
    pole_length = 11
    tip_x = int(round(pivot_x + math.sin(theta) * pole_length))
    tip_y = int(round(pivot_y - math.cos(theta) * pole_length))
    for i in range(37):
        t = i / 36
        px = int(round(pivot_x + (tip_x - pivot_x) * t))
        py = int(round(pivot_y + (tip_y - pivot_y) * t))
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                if abs(ox) + abs(oy) <= 1:
                    set_pixel(frame, px + ox, py + oy, 1.0)
    set_pixel(frame, pivot_x, pivot_y, 0.9)
    return frame


def patch_features(frame: np.ndarray) -> np.ndarray:
    out = np.empty(PATCH_DIM, dtype=np.float32)
    idx = 0
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            patch = frame[
                gy * PATCH_SIZE : (gy + 1) * PATCH_SIZE,
                gx * PATCH_SIZE : (gx + 1) * PATCH_SIZE,
            ]
            out[idx] = float(patch.mean())
            idx += 1
    return out


def vision_features_from_state_buffer(sampled_states: np.ndarray) -> np.ndarray:
    patches = np.stack([patch_features(render_frame(s)) for s in sampled_states]).astype(np.float32)
    delta = np.zeros_like(patches)
    delta[1:] = patches[1:] - patches[:-1]
    return np.concatenate([patches, delta], axis=-1)


def initial_state(rng: np.random.Generator, hard: bool = False, zero_velocity: bool = False) -> np.ndarray:
    x_lim = 0.34 if hard else 0.24
    theta_lim = 0.11 if hard else 0.085
    return np.array([
        rng.uniform(-x_lim, x_lim),
        0.0 if zero_velocity else rng.uniform(-0.20, 0.20),
        rng.uniform(-theta_lim, theta_lim),
        0.0 if zero_velocity else rng.uniform(-0.22, 0.22),
    ], dtype=np.float32)


def sample_state_buffer(buffer: list[np.ndarray]) -> np.ndarray:
    tail = buffer[-BUFFER_LEN:]
    return np.stack([tail[i * FRAME_STRIDE] for i in range(SEQ_LEN)]).astype(np.float32)


def make_dataset(episodes: int = 170, horizon: int = 220):
    rng = np.random.default_rng(SEED)
    state_seq, vision_seq, targets = [], [], []
    sample_idx = np.arange(SEQ_LEN) * FRAME_STRIDE

    for episode in range(episodes):
        state = initial_state(rng, hard=(episode % 4 == 0), zero_velocity=(episode % 3 == 0))
        first_patch = patch_features(render_frame(state))
        state_buffer = np.repeat(state[None, :], BUFFER_LEN, axis=0)
        patch_buffer = np.repeat(first_patch[None, :], BUFFER_LEN, axis=0)
        pulse_left, disturbance = 0, 0.0

        for step in range(horizon):
            sampled_state = state_buffer[sample_idx]
            sampled_patch = patch_buffer[sample_idx]
            delta = np.zeros_like(sampled_patch)
            delta[1:] = sampled_patch[1:] - sampled_patch[:-1]

            state_seq.append(sampled_state / STATE_SCALE)
            vision_seq.append(np.concatenate([sampled_patch, delta], axis=-1))
            targets.append(np.clip(state / STATE_SCALE, -1.0, 1.0))

            if pulse_left <= 0 and step > 8 and rng.random() < 0.020:
                pulse_left = int(rng.integers(3, 9))
                disturbance = float(rng.choice([-6.0, 6.0]))
            if pulse_left > 0:
                pulse_left -= 1
            else:
                disturbance = 0.0

            state = physics_step(state, expert_force(state), disturbance)
            next_patch = patch_features(render_frame(state))
            state_buffer = np.concatenate([state_buffer[1:], state[None, :]], axis=0)
            patch_buffer = np.concatenate([patch_buffer[1:], next_patch[None, :]], axis=0)

            if terminal(state):
                state = initial_state(rng, hard=False, zero_velocity=True)
                first_patch = patch_features(render_frame(state))
                state_buffer = np.repeat(state[None, :], BUFFER_LEN, axis=0)
                patch_buffer = np.repeat(first_patch[None, :], BUFFER_LEN, axis=0)
                pulse_left, disturbance = 0, 0.0

    return (
        np.stack(state_seq).astype(np.float16),
        np.stack(vision_seq).astype(np.float16),
        np.stack(targets).astype(np.float32),
    )


class FusionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_embed = nn.Linear(STATE_DIM + 1, D_MODEL)
        self.vision_embed = nn.Linear(VISION_DIM + 1, D_MODEL)
        self.time_embedding = nn.Parameter(torch.zeros(SEQ_LEN, D_MODEL))
        self.type_embedding = nn.Parameter(torch.zeros(2, D_MODEL))
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.state_head = nn.Linear(D_MODEL, 4)
        self.register_buffer("controller_gain", torch.tensor(EXPERT_GAIN, dtype=torch.float32))
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

        token_times = torch.arange(SEQ_LEN).repeat_interleave(2)
        mask = token_times[None, :] > token_times[:, None]
        self.register_buffer("time_causal_mask", mask)

    def forward(
        self,
        state_seq: torch.Tensor,
        vision_seq: torch.Tensor,
        state_available: torch.Tensor,
        vision_available: torch.Tensor,
    ):
        # availability: [B, T, 1]
        state_input = torch.cat([state_seq, state_available], dim=-1)
        vision_input = torch.cat([vision_seq, vision_available], dim=-1)

        state_tokens = self.state_embed(state_input)
        vision_tokens = self.vision_embed(vision_input)

        pairs = []
        for t in range(SEQ_LEN):
            pairs.append(state_tokens[:, t, :] + self.time_embedding[t] + self.type_embedding[0])
            pairs.append(vision_tokens[:, t, :] + self.time_embedding[t] + self.type_embedding[1])
        tokens = torch.stack(pairs, dim=1)

        norm1 = self.ln1(tokens)
        q, k, v = self.q(norm1), self.k(norm1), self.v(norm1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D_MODEL)
        scores = scores.masked_fill(self.time_causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        residual1 = tokens + self.o(context)
        norm2 = self.ln2(residual1)
        hidden = residual1 + self.ff2(F.gelu(self.ff1(norm2), approximate="tanh"))

        # Fuse the final timestamp explicitly, not by token ordering.
        final_state = hidden[:, -2, :]
        final_vision = hidden[:, -1, :]
        final = 0.5 * (final_state + final_vision)
        inferred_state = torch.tanh(self.state_head(final))
        action_score = (inferred_state * self.controller_gain).sum(dim=-1)
        return action_score, inferred_state, weights


def degrade_batch(
    state: torch.Tensor,
    vision: torch.Tensor,
    generator: torch.Generator,
):
    b = state.shape[0]
    state = state.clone()
    vision = vision.clone()
    state_av = torch.ones((b, SEQ_LEN, 1), dtype=state.dtype)
    vision_av = torch.ones((b, SEQ_LEN, 1), dtype=vision.dtype)

    scenario = torch.rand((b,), generator=generator)
    noise = torch.randn(state.shape, generator=generator, dtype=state.dtype)

    noisy = (scenario >= 0.50) & (scenario < 0.70)
    missing_state = (scenario >= 0.70) & (scenario < 0.82)
    partial_vision = (scenario >= 0.82) & (scenario < 0.94)
    missing_vision = scenario >= 0.94

    if noisy.any():
        state[noisy] += 0.12 * noise[noisy]
    if missing_state.any():
        state[missing_state] = 0
        state_av[missing_state] = 0
    if partial_vision.any():
        # Mask the right half of each patch map in both absolute and delta channels.
        mask = torch.ones(VISION_DIM, dtype=vision.dtype)
        for gy in range(GRID_SIZE):
            for gx in range(GRID_SIZE // 2, GRID_SIZE):
                idx = gy * GRID_SIZE + gx
                mask[idx] = 0
                mask[PATCH_DIM + idx] = 0
        vision[partial_vision] *= mask
    if missing_vision.any():
        vision[missing_vision] = 0
        vision_av[missing_vision] = 0

    return state, vision, state_av, vision_av


def train_model(state_x, vision_x, targets):
    n = len(state_x)
    split = int(n * 0.88)
    sx, vsx = state_x[:split], state_x[split:]
    vx, vvx = vision_x[:split], vision_x[split:]
    ty, vty = targets[:split], targets[split:]

    model = FusionTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)

    sx_t = torch.from_numpy(sx)
    vx_t = torch.from_numpy(vx)
    ty_t = torch.from_numpy(ty)
    vsx_t = torch.from_numpy(vsx).float()
    vvx_t = torch.from_numpy(vvx).float()
    vty_t = torch.from_numpy(vty)

    batch = 256
    epochs = 26
    losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(sx_t), generator=generator)
        total, count = 0.0, 0

        for start in range(0, len(sx_t), batch):
            idx = perm[start:start+batch]
            s = sx_t[idx].float()
            v = vx_t[idx].float()
            target = ty_t[idx]
            s, v, sa, va = degrade_batch(s, v, generator)

            action_score, pred_state, _ = model(s, v, sa, va)
            pred_action = torch.tanh(action_score)
            target_action = torch.tanh((target * model.controller_gain).sum(dim=-1))
            state_loss = F.mse_loss(pred_state, target)
            action_loss = F.mse_loss(pred_action, target_action)
            loss = state_loss + 0.45 * action_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
            count += len(idx)

        model.eval()
        with torch.no_grad():
            one = torch.ones((len(vsx_t), SEQ_LEN, 1))
            val_score, val_state, _ = model(vsx_t, vvx_t, one, one)
            val_action = torch.tanh(val_score)
            val_target_action = torch.tanh((vty_t * model.controller_gain).sum(dim=-1))
            val_state_mse = float(F.mse_loss(val_state, vty_t))
            val_action_mse = float(F.mse_loss(val_action, val_target_action))
        mean = total / count
        losses.append((mean, val_state_mse, val_action_mse))
        print(f"epoch {epoch+1:02d}/{epochs}: train={mean:.7f} val_state={val_state_mse:.7f} val_action={val_action_mse:.7f}")

    return model.eval(), losses, (vsx.astype(np.float32), vvx.astype(np.float32), vty)


@torch.no_grad()
def apply_eval_scenario(s: torch.Tensor, v: torch.Tensor, scenario: str):
    s = s.clone()
    v = v.clone()
    b = s.shape[0]
    sa = torch.ones((b, SEQ_LEN, 1), dtype=s.dtype)
    va = torch.ones((b, SEQ_LEN, 1), dtype=v.dtype)

    if scenario == "state_only":
        v.zero_(); va.zero_()
    elif scenario == "vision_only":
        s.zero_(); sa.zero_()
    elif scenario == "noisy_state":
        pattern = torch.linspace(-1, 1, STATE_DIM, dtype=s.dtype)[None, None, :]
        s = s + 0.18 * pattern
        v.zero_(); va.zero_()
    elif scenario == "noisy_state_plus_vision":
        pattern = torch.linspace(-1, 1, STATE_DIM, dtype=s.dtype)[None, None, :]
        s = s + 0.18 * pattern
    elif scenario in ("partial_vision", "partial_vision_plus_state"):
        if scenario == "partial_vision":
            s.zero_(); sa.zero_()
        mask = torch.ones(VISION_DIM, dtype=v.dtype)
        for gy in range(GRID_SIZE):
            for gx in range(GRID_SIZE // 2, GRID_SIZE):
                idx = gy * GRID_SIZE + gx
                mask[idx] = 0
                mask[PATCH_DIM + idx] = 0
        v = v * mask[None, None, :]

    return s, v, sa, va


@torch.no_grad()
def predict_batch(model, s, v, scenario="clean"):
    if not torch.is_tensor(s):
        s = torch.from_numpy(np.asarray(s, dtype=np.float32))
    else:
        s = s.float()
    if not torch.is_tensor(v):
        v = torch.from_numpy(np.asarray(v, dtype=np.float32))
    else:
        v = v.float()

    s, v, sa, va = apply_eval_scenario(s, v, scenario)
    score, state_pred, weights = model(s, v, sa, va)
    force = 10.0 * torch.tanh(score)
    return force.cpu().numpy(), state_pred.cpu().numpy(), weights.cpu().numpy()


@torch.no_grad()
def predict(model, s, v, scenario="clean"):
    force, state_pred, weights = predict_batch(model, s[None], v[None], scenario)
    return float(force[0]), state_pred[0] * STATE_SCALE, weights[0]


@torch.no_grad()
def validation_ablation(model, val, batch_size: int = 512):
    s, v, y = val
    scenarios = [
        "clean", "state_only", "vision_only",
        "noisy_state", "noisy_state_plus_vision",
        "partial_vision", "partial_vision_plus_state",
    ]
    metrics = {}
    for scenario in scenarios:
        preds = []
        for start in range(0, len(s), batch_size):
            _, pred, _ = predict_batch(
                model,
                s[start:start + batch_size],
                v[start:start + batch_size],
                scenario,
            )
            preds.append(pred)
        p = np.concatenate(preds, axis=0)
        metrics[scenario] = {
            "state_mae": float(np.mean(np.abs(p - y))),
            "xdot_mae": float(np.mean(np.abs((p[:, 1] - y[:, 1]) * STATE_SCALE[1]))),
            "thetadot_mae": float(np.mean(np.abs((p[:, 3] - y[:, 3]) * STATE_SCALE[3]))),
        }
    return metrics


def physics_step_batch(states: np.ndarray, controls: np.ndarray, disturbances: np.ndarray) -> np.ndarray:
    gravity, mass_cart, mass_pole, half_len = 9.8, 1.0, 0.1, 0.5
    total_mass = mass_cart + mass_pole
    pole_mass_length = mass_pole * half_len

    force = np.clip(controls, -10.0, 10.0) + disturbances
    x = states[:, 0]
    xdot = states[:, 1]
    theta = states[:, 2]
    thetadot = states[:, 3]
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + pole_mass_length * thetadot * thetadot * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        half_len * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass)
    )
    x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass

    return np.stack([
        x + TAU * xdot,
        xdot + TAU * x_acc,
        theta + TAU * thetadot,
        thetadot + TAU * theta_acc,
    ], axis=-1).astype(np.float32)


def evaluate_closed_loop(model, scenario: str, episodes: int = 40, horizon: int = 500):
    rng = np.random.default_rng(SEED + 133)
    sample_idx = np.arange(SEQ_LEN) * FRAME_STRIDE

    states = np.stack([
        initial_state(rng, hard=True, zero_velocity=True)
        for _ in range(episodes)
    ]).astype(np.float32)
    first_patches = np.stack([patch_features(render_frame(s)) for s in states]).astype(np.float32)
    state_buffer = np.repeat(states[:, None, :], BUFFER_LEN, axis=1)
    patch_buffer = np.repeat(first_patches[:, None, :], BUFFER_LEN, axis=1)

    active = np.ones(episodes, dtype=bool)
    lengths = np.full(episodes, horizon, dtype=np.int32)
    pulse_left = np.zeros(episodes, dtype=np.int32)
    disturbances = np.zeros(episodes, dtype=np.float32)

    for step in range(horizon):
        sampled_state = state_buffer[:, sample_idx, :] / STATE_SCALE
        sampled_patch = patch_buffer[:, sample_idx, :]
        delta = np.zeros_like(sampled_patch)
        delta[:, 1:, :] = sampled_patch[:, 1:, :] - sampled_patch[:, :-1, :]
        vision = np.concatenate([sampled_patch, delta], axis=-1)

        controls, _, _ = predict_batch(model, sampled_state, vision, scenario)
        controls = controls.astype(np.float32)

        if step > 18:
            for i in np.flatnonzero(active):
                if pulse_left[i] <= 0 and rng.random() < 0.012:
                    pulse_left[i] = int(rng.integers(3, 7))
                    disturbances[i] = float(rng.choice([-4.0, 4.0]))

        has_pulse = pulse_left > 0
        pulse_left[has_pulse] -= 1
        disturbances[~has_pulse] = 0.0
        disturbances[~active] = 0.0

        next_states = physics_step_batch(states, controls, disturbances)
        states[active] = next_states[active]

        new_patches = np.stack([patch_features(render_frame(s)) for s in states]).astype(np.float32)
        state_buffer = np.concatenate([state_buffer[:, 1:, :], states[:, None, :]], axis=1)
        patch_buffer = np.concatenate([patch_buffer[:, 1:, :], new_patches[:, None, :]], axis=1)

        just_failed = active & ((np.abs(states[:, 0]) > 2.4) | (np.abs(states[:, 2]) > 0.38))
        lengths[just_failed] = step + 1
        active[just_failed] = False
        if not active.any():
            break

    a = lengths.astype(np.float32)
    return {
        "episodes": episodes,
        "horizon_steps": horizon,
        "mean_steps": float(a.mean()),
        "median_steps": float(np.median(a)),
        "min_steps": int(a.min()),
        "success_rate_500": float(np.mean(a >= horizon)),
    }


def serial_linear(layer: nn.Linear):
    return {"weight": layer.weight.detach().cpu().tolist(), "bias": layer.bias.detach().cpu().tolist()}


def save_artifact(model, losses, ablation, closed):
    artifact = {
        "format": "cartpole-fusion-transformer-v1",
        "seed": SEED,
        "sequence_length": SEQ_LEN,
        "token_count": TOKEN_COUNT,
        "frame_stride_steps": FRAME_STRIDE,
        "frame_stride_seconds": FRAME_STRIDE * TAU,
        "history_span_seconds": (SEQ_LEN - 1) * FRAME_STRIDE * TAU,
        "state_dim": STATE_DIM,
        "vision_dim": VISION_DIM,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "state_scale": STATE_SCALE.tolist(),
        "controller_gain": EXPERT_GAIN.tolist(),
        "training": {
            "objective": "typed state+vision tokens with modality degradation",
            "epochs": len(losses),
            "final_train_loss": losses[-1][0],
            "final_val_state_mse": losses[-1][1],
            "final_val_action_mse": losses[-1][2],
        },
        "ablation": ablation,
        "closed_loop": closed,
        "weights": {
            "state_embed": serial_linear(model.state_embed),
            "vision_embed": serial_linear(model.vision_embed),
            "time_embedding": model.time_embedding.detach().cpu().tolist(),
            "type_embedding": model.type_embedding.detach().cpu().tolist(),
            "ln1": {"weight":model.ln1.weight.detach().cpu().tolist(),"bias":model.ln1.bias.detach().cpu().tolist()},
            "q": serial_linear(model.q),
            "k": serial_linear(model.k),
            "v": serial_linear(model.v),
            "o": serial_linear(model.o),
            "ln2": {"weight":model.ln2.weight.detach().cpu().tolist(),"bias":model.ln2.bias.detach().cpu().tolist()},
            "ff1": serial_linear(model.ff1),
            "ff2": serial_linear(model.ff2),
            "state_head": serial_linear(model.state_head),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact,separators=(",",":")),encoding="utf-8")

    DOC_PATH.write_text(
        f"""# State + vision typed-token fusion

Generated deterministically by `scripts/train_fusion_transformer.py`.

## Token contract

Each of eight aligned timestamps contributes two tokens:

1. state token: normalized `[x, x_dot, theta, theta_dot]` + availability bit
2. vision token: 256 patch means + 256 delta-patch values + availability bit

The model therefore runs one-head self-attention over 16 typed tokens. Its causal mask is based on **timestamp**, not token ordering, so state and vision tokens from the same timestamp can attend to each other while all future timestamps remain blocked.

No cross-attention module is used.

## Validation ablation

```json
{json.dumps(ablation,indent=2)}
```

## Closed-loop evaluation

```json
{json.dumps(closed,indent=2)}
```

## Claim boundary

Fusion is not expected to outperform clean explicit state on Cart-Pole. The useful question is how the modalities interact under degradation. In this simple typed-token model, state strongly repairs partial vision. Vision lowers average state-estimation error under biased state noise, but does **not** improve noisy-state closed-loop control; that negative result is retained rather than hidden or patched with a more complex fusion mechanism.
""",
        encoding="utf-8"
    )


def main():
    s, v, y = make_dataset()
    print("dataset", s.shape, v.shape, y.shape)
    model, losses, val = train_model(s, v, y)
    ablation = validation_ablation(model, val)
    print("ablation", json.dumps(ablation, sort_keys=True))

    closed = {
        scenario:evaluate_closed_loop(model,scenario)
        for scenario in ["clean","state_only","vision_only","noisy_state","noisy_state_plus_vision","partial_vision","partial_vision_plus_state"]
    }
    print("closed_loop", json.dumps(closed, sort_keys=True))

    if closed["clean"]["mean_steps"] < 420:
        raise SystemExit("fusion clean acceptance failed: mean_steps < 420")
    if closed["state_only"]["mean_steps"] < 420:
        raise SystemExit("fusion state-only baseline acceptance failed")
    if closed["vision_only"]["mean_steps"] < 250:
        raise SystemExit("fusion vision-only baseline acceptance failed")
    if closed["partial_vision_plus_state"]["mean_steps"] <= closed["partial_vision"]["mean_steps"] * 1.20:
        raise SystemExit("fusion partial-vision complementarity acceptance failed")
    if ablation["noisy_state_plus_vision"]["state_mae"] >= ablation["noisy_state"]["state_mae"]:
        raise SystemExit("fusion noisy-state state-estimation MAE did not improve")
    if ablation["partial_vision_plus_state"]["state_mae"] >= ablation["partial_vision"]["state_mae"]:
        raise SystemExit("fusion partial-vision state-estimation MAE did not improve")

    save_artifact(model,losses,ablation,closed)
    print("wrote",MODEL_PATH)
    print("wrote",DOC_PATH)


if __name__ == "__main__":
    main()
