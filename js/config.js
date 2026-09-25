// Models, thresholds and timings. Values marked "board" are the ones the
// original STM32MP2 version uses (pc_dms_runner.py / STM32_Dms.py).

export const MEDIAPIPE_VERSION = "1.0.1";
export const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
export const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

export const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Custom YOLOv8n phone + cigarette detector, the same INT8 model the board runs,
// with a float32 input/output wrapper added (see tools/make_float_io_model.py).
// It runs with LiteRT.js, Google's runtime for .tflite models in the browser.
export const DETECTOR_MODEL_URL = new URL("../models/phone_smoke.tflite", import.meta.url).href;
export const LITERT_VERSION = "2.5.3";
export const LITERT_URL = `https://cdn.jsdelivr.net/npm/@litertjs/core@${LITERT_VERSION}/+esm`;
export const LITERT_WASM_URL = `https://cdn.jsdelivr.net/npm/@litertjs/core@${LITERT_VERSION}/wasm/`;

// Every threshold can be changed live in the page's Tuning panel.
export const DEFAULT_THRESHOLDS = {
  eyeClosed: 0.5,   // eyeBlink blendshape score (0 open – 1 closed); both eyes above = closed
  yawn: 0.28,       // board: mouth height ÷ mouth width
  facingMin: 0.5,   // board: lip-to-cheek distance ratio; outside [min, max] = head turned
  facingMax: 2.0,   // board
  headDown: 0.58,   // board: nose position between forehead (0) and chin (1)
  // Detector probabilities, counted only near the face. The board applies a
  // second sigmoid to scores that are already probabilities and uses 0.55 / 0.60
  // on that scale; 0.20 / 0.40 are the same cut-offs as plain probabilities.
  phone: 0.2,
  smoking: 0.4,
};

// How long a condition must last before it becomes an alert (seconds).
// A normal blink is 0.1–0.4 s, so "drowsy" needs the eyes shut for longer.
export const HOLD_SECONDS = {
  drowsy: 0.8,
  yawn: 0.4,
  distracted: 0.4,
  headDown: 0.5,
};

export const DETECT_EVERY_MS = 150;   // run the phone/cigarette detector at most this often
export const DETECT_MEMORY_MS = 600;  // an alert stays active this long after the last sighting
export const DETECT_FLOOR = 0.05;     // lowest score kept, so the Tuning panel can show live values
export const NMS_IOU = 0.4;           // board: overlapping boxes above this IoU are merged

// Risk score. The board adds these points on every frame. Here they are scaled
// by elapsed time × REF_FPS, so the score moves at the same speed on a 15 FPS
// phone and a 60 FPS laptop. Raise REF_FPS to make the score react faster.
export const REF_FPS = 10;
export const PENALTY = {
  distracted: 2,
  drowsy: 5,
  yawn: 7,
  headDown: 5,
  phone: 2,
  smoking: 2,
};
export const NO_FACE_PENALTY = 0.7;
export const RESTORE_CREDIT = -5;  // per frame with a face and no alerts

export const WARNING_AT = 33;
export const DANGER_AT = 66;
