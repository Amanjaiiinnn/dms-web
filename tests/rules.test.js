import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CHEEK,
  CHIN,
  FOREHEAD,
  Hold,
  MOUTH,
  NOSE_TIP,
  blinkScores,
  boundingBox,
  facingRatio,
  headDownRatio,
  isNearFace,
  pickCentreFace,
  toPixels,
  yawnRatio,
} from "../js/rules.js";
import { statusFor, updateScore } from "../js/score.js";
import { DANGER_AT, NO_FACE_PENALTY, REF_FPS, WARNING_AT } from "../js/config.js";

/** A 478-point face with only the given landmarks placed. */
function face(placed) {
  const points = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
  for (const [index, [x, y]] of Object.entries(placed)) points[index] = { x, y };
  return points;
}

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} ≉ ${expected}`);

test("toPixels scales each axis by its own size", () => {
  assert.deepEqual(toPixels([{ x: 0.5, y: 0.25, z: 0 }], 640, 480), [{ x: 320, y: 120 }]);
});

test("yawnRatio is mouth height over width", () => {
  const closed = face({ [MOUTH.left]: [100, 200], [MOUTH.right]: [160, 200], [MOUTH.top]: [130, 199], [MOUTH.bottom]: [130, 201] });
  const open = face({ [MOUTH.left]: [100, 200], [MOUTH.right]: [160, 200], [MOUTH.top]: [130, 180], [MOUTH.bottom]: [130, 225] });
  near(yawnRatio(closed), 2 / 60);
  near(yawnRatio(open), 45 / 60);
  assert.equal(yawnRatio(face({})), 0, "zero width must not divide by zero");
});

test("facingRatio is about 1 facing forward and leaves 0.5–2 when turned", () => {
  const forward = face({ [MOUTH.top]: [100, 100], [CHEEK.left]: [40, 100], [CHEEK.right]: [160, 100] });
  const turned = face({ [MOUTH.top]: [100, 100], [CHEEK.left]: [85, 100], [CHEEK.right]: [160, 100] });
  near(facingRatio(forward), 1);
  assert.ok(facingRatio(turned) < 0.5);
});

test("headDownRatio is the nose position between forehead and chin", () => {
  const level = face({ [FOREHEAD]: [100, 100], [CHIN]: [100, 300], [NOSE_TIP]: [100, 200] });
  const down = face({ [FOREHEAD]: [100, 160], [CHIN]: [100, 300], [NOSE_TIP]: [100, 250] });
  near(headDownRatio(level), 0.5);
  assert.ok(headDownRatio(down) > 0.58);
  assert.equal(headDownRatio(face({})), 0);
});

test("blinkScores reads the eyeBlink blendshapes", () => {
  const blendshapes = {
    categories: [
      { categoryName: "jawOpen", score: 0.1 },
      { categoryName: "eyeBlinkLeft", score: 0.8 },
      { categoryName: "eyeBlinkRight", score: 0.6 },
    ],
  };
  assert.deepEqual(blinkScores(blendshapes), { left: 0.8, right: 0.6 });
  assert.deepEqual(blinkScores(undefined), { left: 0, right: 0 });
});

test("boundingBox and pickCentreFace choose the face nearest the centre", () => {
  const edge = boundingBox([{ x: 0, y: 0 }, { x: 100, y: 100 }]);
  const centre = boundingBox([{ x: 270, y: 190 }, { x: 370, y: 290 }]);
  assert.deepEqual(centre, { x1: 270, y1: 190, x2: 370, y2: 290 });
  assert.equal(pickCentreFace([edge, centre], 640, 480), 1);
  assert.equal(pickCentreFace([], 640, 480), -1);
});

test("isNearFace accepts a phone at the ear and rejects one across the frame", () => {
  const faceBox = { x1: 270, y1: 150, x2: 370, y2: 270 };
  const atEar = { x1: 360, y1: 180, x2: 420, y2: 280 };
  const farAway = { x1: 590, y1: 20, x2: 630, y2: 60 };
  assert.equal(isNearFace(atEar, faceBox, 640, 480), true);
  assert.equal(isNearFace(farAway, faceBox, 640, 480), false);
  assert.equal(isNearFace(atEar, { x1: 10, y1: 10, x2: 10, y2: 50 }, 640, 480), false);
});

test("Hold fires only after the condition has lasted long enough", () => {
  const hold = new Hold(0.8);
  assert.equal(hold.update(true, 10.0), false);
  assert.equal(hold.update(true, 10.5), false);
  assert.equal(hold.update(true, 10.8), true);
  assert.equal(hold.update(false, 10.9), false, "a break resets it");
  assert.equal(hold.update(true, 11.0), false);
});

test("updateScore adds penalties, recovers, and stays within 0–100", () => {
  const none = {};
  near(updateScore(0, none, false, 1), NO_FACE_PENALTY * REF_FPS);
  assert.equal(updateScore(10, none, true, 1), 0, "recovers with a face and no alerts, not below 0");
  assert.equal(updateScore(90, { yawn: true, drowsy: true }, true, 1), 100, "never above 100");
  assert.ok(updateScore(0, { distracted: true, phone: true }, true, 0.1) > updateScore(0, { distracted: true }, true, 0.1));
});

test("updateScore gives the same result at 15 and 60 FPS", () => {
  let slow = 0;
  let fast = 0;
  for (let i = 0; i < 15; i++) slow = updateScore(slow, { distracted: true }, true, 1 / 15);
  for (let i = 0; i < 60; i++) fast = updateScore(fast, { distracted: true }, true, 1 / 60);
  near(slow, fast);
});

test("statusFor maps the score to a status", () => {
  assert.equal(statusFor(0, true), "ok");
  assert.equal(statusFor(WARNING_AT, true), "warning");
  assert.equal(statusFor(DANGER_AT, true), "danger");
  assert.equal(statusFor(90, false), "noface");
});
