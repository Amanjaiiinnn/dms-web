// Decoding for the custom YOLOv8n phone/cigarette model. Same steps as the
// board's smoking_calling_yolov8n_custom.py: letterbox, decode, per-class
// threshold, drop tiny boxes, non-maximum suppression.

export const INPUT_SIZE = 320;
export const CLASSES = ["phone", "smoking"]; // model class 0 and 1

/** Scale and padding that fit a width×height frame into the square model input. */
export function letterbox(width, height, size = INPUT_SIZE) {
  const scale = Math.min(size / width, size / height);
  const w = Math.trunc(width * scale);
  const h = Math.trunc(height * scale);
  return { scale, left: Math.floor((size - w) / 2), top: Math.floor((size - h) / 2), w, h };
}

/**
 * Turns the raw model output into boxes in frame pixels.
 *
 * `output` is the flattened (6, N) tensor: rows 0–3 are cx, cy, w, h
 * normalised to the model input, rows 4–5 are phone and cigarette
 * probabilities. Returns every box whose best class scores at least `floor`.
 */
export function decode(output, lb, width, height, floor, size = INPUT_SIZE) {
  const n = output.length / 6;
  const boxes = [];
  for (let a = 0; a < n; a++) {
    const phone = output[4 * n + a];
    const smoke = output[5 * n + a];
    const cls = smoke > phone ? 1 : 0;
    const score = cls ? smoke : phone;
    if (score < floor) continue;

    const cx = (output[a] * size - lb.left) / lb.scale;
    const cy = (output[n + a] * size - lb.top) / lb.scale;
    const bw = (output[2 * n + a] * size) / lb.scale;
    const bh = (output[3 * n + a] * size) / lb.scale;
    const x1 = clamp(Math.trunc(cx - bw / 2), 0, width);
    const y1 = clamp(Math.trunc(cy - bh / 2), 0, height);
    const x2 = clamp(Math.trunc(cx + bw / 2), 0, width);
    const y2 = clamp(Math.trunc(cy + bh / 2), 0, height);
    if (x2 - x1 <= 4 || y2 - y1 <= 4) continue;

    boxes.push({ x1, y1, x2, y2, score, cls });
  }
  return boxes;
}

/** Keeps the highest-scoring box of each overlapping group (any class), like cv2.dnn.NMSBoxes. */
export function nms(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of sorted) {
    if (kept.every((k) => iou(k, box) <= iouThreshold)) kept.push(box);
  }
  return kept;
}

export function iou(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
