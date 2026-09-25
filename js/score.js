import {
  DANGER_AT,
  NO_FACE_PENALTY,
  PENALTY,
  REF_FPS,
  RESTORE_CREDIT,
  WARNING_AT,
} from "./config.js";

/**
 * Advances the 0–100 risk score by `dt` seconds.
 *
 * Several alerts add up. With a face and no alerts the score recovers.
 * The board applies these points once per frame; here they are multiplied by
 * dt × REF_FPS so the result does not depend on the browser's frame rate.
 */
export function updateScore(score, alerts, faceFound, dt) {
  let perFrame = 0;
  if (!faceFound) {
    perFrame = NO_FACE_PENALTY;
  } else {
    for (const [name, points] of Object.entries(PENALTY)) {
      if (alerts[name]) perFrame += points;
    }
    if (perFrame === 0) perFrame = RESTORE_CREDIT;
  }
  return Math.min(100, Math.max(0, score + perFrame * dt * REF_FPS));
}

/** "ok" | "warning" | "danger" | "noface" */
export function statusFor(score, faceFound) {
  if (!faceFound) return "noface";
  if (score >= DANGER_AT) return "danger";
  if (score >= WARNING_AT) return "warning";
  return "ok";
}
