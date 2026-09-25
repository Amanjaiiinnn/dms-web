// Face-geometry rules. The ratio functions are adapted from NXP's DMS demo
// (mouth.py, BSD-3-Clause, see NOTICE). Indices are MediaPipe face-mesh
// landmark numbers, the same numbering the board's 468-point model uses.

export const MOUTH = { left: 78, top: 13, right: 308, bottom: 14 };
export const CHEEK = { left: 132, right: 361 };
export const FOREHEAD = 10;
export const CHIN = 152;
export const NOSE_TIP = 1;
const EYE_CORNERS_AND_LIDS = [33, 133, 159, 145, 362, 263, 386, 374];

/** Landmarks the rules read, highlighted on the video. */
export const KEY_POINTS = [
  ...Object.values(MOUTH),
  ...Object.values(CHEEK),
  FOREHEAD,
  CHIN,
  NOSE_TIP,
  ...EYE_CORNERS_AND_LIDS,
];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * MediaPipe gives coordinates from 0 to 1 on each axis. The video is not
 * square, so ratios must be computed in pixels or they come out skewed.
 */
export function toPixels(landmarks, width, height) {
  return landmarks.map((p) => ({ x: p.x * width, y: p.y * height }));
}

/** Mouth opening: height ÷ width. Near 0 when closed. */
export function yawnRatio(p) {
  const width = dist(p[MOUTH.left], p[MOUTH.right]);
  return width > 0 ? dist(p[MOUTH.top], p[MOUTH.bottom]) / width : 0;
}

/** Head turn: upper lip to one cheek ÷ upper lip to the other. About 1 when facing forward. */
export function facingRatio(p) {
  const toRight = dist(p[CHEEK.right], p[MOUTH.top]);
  return toRight > 0 ? dist(p[MOUTH.top], p[CHEEK.left]) / toRight : 0;
}

/** Head down: where the nose tip sits between forehead (0) and chin (1). */
export function headDownRatio(p) {
  const faceHeight = p[CHIN].y - p[FOREHEAD].y;
  return faceHeight > 0 ? (p[NOSE_TIP].y - p[FOREHEAD].y) / faceHeight : 0;
}

/** Eye-closure scores from the face blendshapes: 0 = open, 1 = closed. */
export function blinkScores(blendshapes) {
  let left = 0;
  let right = 0;
  for (const c of blendshapes?.categories ?? []) {
    if (c.categoryName === "eyeBlinkLeft") left = c.score;
    else if (c.categoryName === "eyeBlinkRight") right = c.score;
  }
  return { left, right };
}

export function boundingBox(points) {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const { x, y } of points) {
    if (x < x1) x1 = x;
    if (x > x2) x2 = x;
    if (y < y1) y1 = y;
    if (y > y2) y2 = y;
  }
  return { x1, y1, x2, y2 };
}

/** Index of the box closest to the frame centre (the driver), or -1 if none. */
export function pickCentreFace(boxes, width, height) {
  let best = -1;
  let bestDistance = Infinity;
  boxes.forEach((b, i) => {
    const d = Math.hypot((b.x1 + b.x2 - width) / 2, (b.y1 + b.y2 - height) / 2);
    if (d < bestDistance) {
      best = i;
      bestDistance = d;
    }
  });
  return best;
}

/**
 * True when the centre of `box` lies in the area where a phone or cigarette
 * would be: 1.25 face widths to each side, 0.75 face heights above and 1.75
 * below the face. Same rule as the board version.
 */
export function isNearFace(box, face, width, height) {
  const fw = face.x2 - face.x1;
  const fh = face.y2 - face.y1;
  if (fw <= 0 || fh <= 0) return false;

  const cx = (box.x1 + box.x2) / 2;
  const cy = (box.y1 + box.y2) / 2;
  return (
    cx >= Math.max(0, face.x1 - 1.25 * fw) &&
    cx <= Math.min(width, face.x2 + 1.25 * fw) &&
    cy >= Math.max(0, face.y1 - 0.75 * fh) &&
    cy <= Math.min(height, face.y2 + 1.75 * fh)
  );
}

/** Turns a noisy per-frame condition into an alert that must last `seconds`. */
export class Hold {
  constructor(seconds) {
    this.seconds = seconds;
    this.since = null;
  }

  update(active, nowSeconds) {
    if (!active) {
      this.since = null;
      return false;
    }
    this.since ??= nowSeconds;
    return nowSeconds - this.since >= this.seconds;
  }

  reset() {
    this.since = null;
  }
}
