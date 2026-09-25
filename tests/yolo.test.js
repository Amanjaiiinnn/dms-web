import { test } from "node:test";
import assert from "node:assert/strict";

import { CLASSES, decode, iou, letterbox, nms } from "../js/yolo.js";

/** Builds a flattened (6, N) model output from per-anchor [cx, cy, w, h, phone, smoke] rows. */
function output(anchors) {
  const n = anchors.length;
  const out = new Float32Array(6 * n);
  anchors.forEach((row, a) => row.forEach((v, r) => (out[r * n + a] = v)));
  return out;
}

test("class order matches the model: 0 = phone, 1 = cigarette", () => {
  assert.deepEqual(CLASSES, ["phone", "smoking"]);
});

test("letterbox fits a 640×480 frame into 320×320 with bars top and bottom", () => {
  assert.deepEqual(letterbox(640, 480), { scale: 0.5, left: 0, top: 40, w: 320, h: 240 });
});

test("decode maps boxes back to frame pixels and picks the best class", () => {
  const lb = letterbox(640, 480);
  const boxes = decode(
    output([
      [0.5, 0.5, 0.25, 0.25, 0.9, 0.1],  // phone in the centre
      [0.25, 0.5, 0.1, 0.2, 0.1, 0.6],   // cigarette
      [0.5, 0.5, 0.5, 0.5, 0.01, 0.02],  // below the floor
      [0.5, 0.5, 0.001, 0.2, 0.9, 0.0],  // too thin, dropped
    ]),
    lb, 640, 480, 0.05,
  );
  assert.equal(boxes.length, 2);
  assert.deepEqual(boxes[0], { x1: 240, y1: 160, x2: 400, y2: 320, score: boxes[0].score, cls: 0 });
  assert.ok(Math.abs(boxes[0].score - 0.9) < 1e-6);
  assert.equal(boxes[1].cls, 1);
});

test("decode clips boxes to the frame", () => {
  const [box] = decode(output([[0.98, 0.5, 0.2, 0.2, 0.8, 0]]), letterbox(640, 480), 640, 480, 0.05);
  assert.equal(box.x2, 640);
});

test("nms keeps the best of overlapping boxes, across classes", () => {
  const a = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.9, cls: 0 };
  const b = { x1: 10, y1: 10, x2: 110, y2: 110, score: 0.8, cls: 1 };
  const c = { x1: 300, y1: 300, x2: 350, y2: 350, score: 0.5, cls: 1 };
  assert.deepEqual(nms([b, c, a], 0.4), [a, c]);
});

test("iou of identical, disjoint and half-overlapping boxes", () => {
  const box = { x1: 0, y1: 0, x2: 10, y2: 10 };
  assert.equal(iou(box, box), 1);
  assert.equal(iou(box, { x1: 20, y1: 20, x2: 30, y2: 30 }), 0);
  assert.ok(Math.abs(iou(box, { x1: 5, y1: 0, x2: 15, y2: 10 }) - 1 / 3) < 1e-9);
});
