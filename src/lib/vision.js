export const VISION_FRAME_SIZE = 32;
export const VISION_PATCH_SIZE = 2;
export const VISION_GRID_SIZE = VISION_FRAME_SIZE / VISION_PATCH_SIZE;
export const VISION_FEATURE_DIM = VISION_GRID_SIZE * VISION_GRID_SIZE;
export const VISION_SEQUENCE_LENGTH = 8;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function setPixel(frame, x, y, value) {
  if (x < 0 || x >= VISION_FRAME_SIZE || y < 0 || y >= VISION_FRAME_SIZE) return;
  const idx = y * VISION_FRAME_SIZE + x;
  frame[idx] = Math.max(frame[idx], value);
}

export function renderVisionFrame(state) {
  const [x, , theta] = state;
  const frame = new Array(VISION_FRAME_SIZE * VISION_FRAME_SIZE).fill(0);

  // Track: intentionally low contrast so the cart/pole dominates patch features.
  const trackY = 27;
  for (let px = 1; px < VISION_FRAME_SIZE - 1; px++) setPixel(frame, px, trackY, 0.15);

  const cx = Math.round(16 + clamp(x / 2.4, -1, 1) * 12);
  const cartHalfW = 4;
  for (let py = 23; py <= 26; py++) {
    for (let px = cx - cartHalfW; px <= cx + cartHalfW; px++) setPixel(frame, px, py, 0.72);
  }

  const pivotX = cx;
  const pivotY = 23;
  const poleLength = 11;
  const tipX = Math.round(pivotX + Math.sin(theta) * poleLength);
  const tipY = Math.round(pivotY - Math.cos(theta) * poleLength);

  const samples = 36;
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const px = Math.round(pivotX + (tipX - pivotX) * t);
    const py = Math.round(pivotY + (tipY - pivotY) * t);
    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        if (Math.abs(ox) + Math.abs(oy) <= 1) setPixel(frame, px + ox, py + oy, 1.0);
      }
    }
  }
  setPixel(frame, pivotX, pivotY, 0.9);
  return frame;
}

export function extractVisionPatchFeatures(frame) {
  const features = [];
  for (let gy = 0; gy < VISION_GRID_SIZE; gy++) {
    for (let gx = 0; gx < VISION_GRID_SIZE; gx++) {
      let sum = 0;
      for (let py = 0; py < VISION_PATCH_SIZE; py++) {
        for (let px = 0; px < VISION_PATCH_SIZE; px++) {
          const x = gx * VISION_PATCH_SIZE + px;
          const y = gy * VISION_PATCH_SIZE + py;
          sum += frame[y * VISION_FRAME_SIZE + x];
        }
      }
      features.push(sum / (VISION_PATCH_SIZE * VISION_PATCH_SIZE));
    }
  }
  return features;
}

export function visionObservationFromState(state) {
  const frame = renderVisionFrame(state);
  const patches = extractVisionPatchFeatures(frame);
  return { frame, patches };
}
