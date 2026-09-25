import {
  DEFAULT_THRESHOLDS,
  FACE_MODEL_URL,
  HOLD_SECONDS,
  MEDIAPIPE_URL,
  PHONE_CHECK_MS,
  PHONE_MEMORY_MS,
  PHONE_MODEL_URL,
  WASM_URL,
} from "./config.js";
import {
  Hold,
  KEY_POINTS,
  blinkScores,
  boundingBox,
  facingRatio,
  headDownRatio,
  isNearFace,
  pickCentreFace,
  toPixels,
  yawnRatio,
} from "./rules.js";
import { statusFor, updateScore } from "./score.js";
import { Alarm } from "./alarm.js";

const ALERT_LABELS = {
  drowsy: "Drowsy",
  yawn: "Yawning",
  distracted: "Distracted",
  headDown: "Head down",
  phone: "Phone use",
};

const STATUS_LABELS = {
  idle: "Not running",
  ok: "Driver OK",
  warning: "Warning",
  danger: "Danger",
  noface: "Driver not found",
};

// Drawn on top of the video, so the same in light and dark mode.
const DRAW_COLORS = { ok: "#4ade80", warning: "#fbbf24", danger: "#f87171", noface: "#94a3b8" };
const KEY_POINT_COLOR = "#38bdf8";
const PHONE_COLOR = "#fb923c";

// Rows of the Tuning panel. `above`: the alert fires when the live value is above the threshold.
const TUNING = [
  { key: "eyeClosed", measure: "eyes", label: "Eyes closed", hint: "Blink score of the more open eye", min: 0.1, max: 0.9, step: 0.01, above: true },
  { key: "yawn", measure: "yawn", label: "Yawn", hint: "Mouth height ÷ width", min: 0.1, max: 1.2, step: 0.01, above: true },
  { key: "facingMin", measure: "facing", label: "Head turned (low)", hint: "Alert below this cheek ratio", min: 0.2, max: 1, step: 0.01, above: false },
  { key: "facingMax", measure: "facing", label: "Head turned (high)", hint: "Alert above this cheek ratio", min: 1, max: 4, step: 0.05, above: true },
  { key: "headDown", measure: "headDown", label: "Head down", hint: "Nose between forehead (0) and chin (1)", min: 0.4, max: 0.8, step: 0.01, above: true },
  { key: "phone", measure: "phone", label: "Phone", hint: "Detector confidence near the face", min: 0.1, max: 0.9, step: 0.05, above: true },
];

// localStorage can be unavailable (private mode, blocked site data). Settings then just don't persist.
const storage = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* not persisted */
    }
  },
};

const $ = (id) => document.getElementById(id);
const app = $("app");
const stage = $("stage");
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const startScreen = $("start-screen");
const loadText = $("load-text");
const startBtn = $("start-btn");
const stopBtn = $("stop-btn");
const cameraField = $("camera-field");
const cameraSelect = $("camera-select");
const soundBtn = $("sound-btn");
const meshToggle = $("mesh-toggle");
const errorBox = $("error");
const statusText = $("status-text");
const hudStatus = $("hud-status");
const hudScore = $("hud-score");
const scoreValue = $("score-value");
const scoreFill = $("score-fill");
const fpsText = $("fps");
const backendText = $("backend");
const eventsList = $("events");
const tuningPanel = $("tuning");
const tuningRows = $("tuning-rows");
const resetBtn = $("reset-thresholds");
const alertItems = Object.fromEntries(
  [...document.querySelectorAll("[data-alert]")].map((li) => [li.dataset.alert, li]),
);

const alarm = new Alarm();
alarm.enabled = storage.get("dms.sound", true) !== false;
let showMesh = storage.get("dms.mesh", true) !== false;
let thresholds = loadThresholds();

let faceLandmarker = null;
let phoneDetector = null;

// Session state
let running = false;
let rafId = 0;
let beginToken = 0;
let mirrored = true;
let lastVideoTime = -1;
let lastTick = 0;
let lastUiUpdate = 0;
let frames = 0;
let fpsSince = 0;

// Detection state
const holds = Object.fromEntries(
  Object.entries(HOLD_SECONDS).map(([key, seconds]) => [key, new Hold(seconds)]),
);
let score = 0;
let alerts = noAlerts();
let prevAlerts = noAlerts();
let prevStatus = "idle";
let measures = null;
let phoneBoxes = [];
let lastPhoneCheck = -Infinity;
let lastPhoneSeen = -Infinity;
let lastPhoneScore = 0;
let faceLostSince = null;
let faceLostLogged = false;

buildTuning();
renderSound();
meshToggle.checked = showMesh;
clearEvents();
renderIdle();

// Start downloading the models right away, so they are ready by the time the visitor clicks Start.
const modelsReady = loadModels();

// ─── Model loading ──────────────────────────────────────────────────────────

async function loadModels() {
  try {
    loadText.textContent = "Loading MediaPipe…";
    const { FilesetResolver, FaceLandmarker, ObjectDetector } = await import(MEDIAPIPE_URL);
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

    loadText.textContent = "Loading face model…";
    const faceOptions = (delegate) => ({
      baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numFaces: 2,
      outputFaceBlendshapes: true,
    });
    let delegate = "GPU";
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(fileset, faceOptions("GPU"));
    } catch (err) {
      console.warn("GPU not available, using CPU:", err);
      delegate = "CPU";
      faceLandmarker = await FaceLandmarker.createFromOptions(fileset, faceOptions("CPU"));
    }
    backendText.textContent = delegate;

    loadText.textContent = "Loading phone model…";
    try {
      phoneDetector = await ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: PHONE_MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        scoreThreshold: 0.2, // low on purpose: the Tuning threshold is applied afterwards
        maxResults: 3,
        categoryAllowlist: ["cell phone"],
      });
    } catch (err) {
      console.warn("Phone detector unavailable:", err);
      alertItems.phone.dataset.unavailable = "true";
      alertItems.phone.querySelector(".state").textContent = "Unavailable";
    }

    loadText.textContent = "Models ready";
    startScreen.dataset.state = "ready";
    return true;
  } catch (err) {
    console.error(err);
    startScreen.dataset.state = "error";
    loadText.textContent = "Couldn't load the models";
    showError(
      "Couldn't load the AI models. Check your internet connection, turn off content blockers for this page, and reload.",
    );
    return false;
  }
}

// ─── Starting and stopping ──────────────────────────────────────────────────

startBtn.addEventListener("click", () => startCamera(storage.get("dms.camera", null)));

stopBtn.addEventListener("click", stop);

cameraSelect.addEventListener("change", () => {
  storage.set("dms.camera", cameraSelect.value);
  startCamera(cameraSelect.value);
});

// A webcam was plugged in or out: refresh the picker.
navigator.mediaDevices?.addEventListener?.("devicechange", () => {
  if (running) listCameras();
});

/** Opens a camera connected to this device and starts monitoring it. */
async function startCamera(deviceId) {
  alarm.unlock();
  hideError();
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("This browser can't open the camera here. Use an up-to-date browser, over HTTPS or on localhost.");
    return;
  }
  // Release the current camera first: some devices can't open two at once.
  if (running) teardown();

  let stream;
  try {
    stream = await openCamera(deviceId);
  } catch (err) {
    stop();
    showError(cameraError(err));
    return;
  }
  stream.getVideoTracks()[0]?.addEventListener("ended", () => {
    stop();
    showError("The camera was disconnected. Reconnect it and press Start camera.");
  });
  if (await begin(stream)) await listCameras();
}

/** The chosen camera if it is still connected, otherwise the default front camera. */
async function openCamera(deviceId) {
  const size = { width: { ideal: 640 }, height: { ideal: 480 } };
  if (deviceId) {
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: { ...size, deviceId: { exact: deviceId } },
        audio: false,
      });
    } catch (err) {
      if (err?.name !== "OverconstrainedError" && err?.name !== "NotFoundError") throw err;
    }
  }
  return navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: "user" }, audio: false });
}

/** Fills the camera picker. It only appears when the device has more than one camera. */
async function listCameras() {
  let cameras = [];
  try {
    cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  } catch {
    /* keep the picker hidden */
  }
  const activeId = video.srcObject?.getVideoTracks()[0]?.getSettings().deviceId;
  cameraSelect.replaceChildren(
    ...cameras.map(
      (d, n) => new Option(d.label || `Camera ${n + 1}`, d.deviceId, false, d.deviceId === activeId),
    ),
  );
  cameraField.hidden = cameras.length < 2;
}

/** Shows `stream` and starts the detection loop. Returns false if it didn't start. */
async function begin(stream) {
  const token = ++beginToken;
  teardown();
  startBtn.disabled = true;

  const ready = await modelsReady;
  if (!ready || token !== beginToken) {
    stream.getTracks().forEach((t) => t.stop());
    startBtn.disabled = false;
    return false;
  }

  video.srcObject = stream;
  // Mirror the view like a selfie camera, except for a rear-facing phone camera.
  mirrored = stream.getVideoTracks()[0]?.getSettings().facingMode !== "environment";
  video.classList.toggle("mirrored", mirrored);

  try {
    await video.play();
    if (!video.videoWidth) throw new Error("No video track");
  } catch (err) {
    if (token !== beginToken) return false; // a newer start or a stop already took over
    console.error(err);
    startBtn.disabled = false;
    stop();
    showError("Couldn't start the camera video.");
    return false;
  }
  if (token !== beginToken) return false;

  fitStage(video.videoWidth, video.videoHeight);
  resetDetection();
  clearEvents();
  running = true;
  startScreen.hidden = true;
  startBtn.disabled = false;
  stopBtn.disabled = false;
  app.dataset.running = "true";
  rafId = requestAnimationFrame(loop);
  return true;
}

/** Stops the loop and releases the camera, without touching the page layout. */
function teardown() {
  running = false;
  cancelAnimationFrame(rafId);
  video.pause();
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function stop() {
  beginToken++;
  teardown();
  resetDetection();
  startScreen.hidden = false;
  stopBtn.disabled = true;
  app.dataset.running = "false";
  renderIdle();
}

function fitStage(width, height) {
  canvas.width = width;
  canvas.height = height;
  stage.style.setProperty("--ratio", String(width / height));
}

function resetDetection() {
  score = 0;
  alerts = noAlerts();
  prevAlerts = noAlerts();
  prevStatus = "idle";
  measures = null;
  Object.values(holds).forEach((h) => h.reset());
  phoneBoxes = [];
  lastPhoneCheck = -Infinity;
  lastPhoneSeen = -Infinity;
  lastPhoneScore = 0;
  faceLostSince = null;
  faceLostLogged = false;
  lastVideoTime = -1;
  lastTick = 0;
  frames = 0;
  fpsSince = performance.now();
}

// ─── Per-frame processing ───────────────────────────────────────────────────

function loop(now) {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return; // no new frame yet
  lastVideoTime = video.currentTime;

  // Cap the step so a hidden tab or a slow frame doesn't make the score jump.
  const dt = lastTick ? Math.min((now - lastTick) / 1000, 0.25) : 0;
  lastTick = now;

  try {
    processFrame(now, dt);
  } catch (err) {
    console.error(err);
    stop();
    showError("Something went wrong while analysing the video. Reload the page and try again.");
  }
}

function processFrame(now, dt) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) fitStage(w, h); // phones can rotate mid-session

  const result = faceLandmarker.detectForVideo(video, now);
  const faces = result.faceLandmarks.map((lm) => toPixels(lm, w, h));
  const boxes = faces.map(boundingBox);
  const i = pickCentreFace(boxes, w, h);
  const faceFound = i >= 0;

  if (faceFound) {
    const points = faces[i];
    const blink = blinkScores(result.faceBlendshapes[i]);
    const nowSec = now / 1000;
    measures = {
      eyes: Math.min(blink.left, blink.right), // both eyes must be closed
      yawn: yawnRatio(points),
      facing: facingRatio(points),
      headDown: headDownRatio(points),
      phone: lastPhoneScore,
    };

    alerts.drowsy = holds.drowsy.update(measures.eyes > thresholds.eyeClosed, nowSec);
    alerts.yawn = holds.yawn.update(measures.yawn > thresholds.yawn, nowSec);
    alerts.distracted = holds.distracted.update(
      measures.facing < thresholds.facingMin || measures.facing > thresholds.facingMax,
      nowSec,
    );
    alerts.headDown = holds.headDown.update(measures.headDown > thresholds.headDown, nowSec);

    if (phoneDetector && now - lastPhoneCheck >= PHONE_CHECK_MS) {
      lastPhoneCheck = now;
      checkForPhone(now, boxes[i], w, h);
      measures.phone = lastPhoneScore;
    }
    alerts.phone = now - lastPhoneSeen < PHONE_MEMORY_MS;
  } else {
    Object.values(holds).forEach((hold) => hold.reset());
    alerts = noAlerts();
    measures = null;
    phoneBoxes = [];
    lastPhoneSeen = -Infinity;
    lastPhoneScore = 0;
  }

  score = updateScore(score, alerts, faceFound, dt);
  const status = statusFor(score, faceFound);

  draw(faceFound ? faces[i] : null, faceFound ? boxes[i] : null, status);
  alarm.update(status === "danger", now);
  logChanges(status, faceFound, now);
  countFrame(now);

  if (now - lastUiUpdate >= 100) {
    lastUiUpdate = now;
    renderStatus(status);
    renderAlerts();
    renderMeasures();
  }
}

function checkForPhone(now, face, width, height) {
  const { detections } = phoneDetector.detectForVideo(video, now);
  phoneBoxes = [];
  lastPhoneScore = 0;
  for (const d of detections) {
    const b = d.boundingBox;
    if (!b) continue;
    const box = {
      x1: b.originX,
      y1: b.originY,
      x2: b.originX + b.width,
      y2: b.originY + b.height,
      score: d.categories[0]?.score ?? 0,
    };
    if (!isNearFace(box, face, width, height)) continue;
    lastPhoneScore = Math.max(lastPhoneScore, box.score);
    if (box.score >= thresholds.phone) phoneBoxes.push(box);
  }
  if (phoneBoxes.length) lastPhoneSeen = now;
}

// ─── Drawing ────────────────────────────────────────────────────────────────

function draw(points, face, status) {
  const w = canvas.width;
  ctx.clearRect(0, 0, w, canvas.height);
  // The webcam view is mirrored with CSS; mirror the coordinates here instead of
  // the canvas so that text stays readable.
  const X = (x) => (mirrored ? w - x : x);
  const unit = Math.max(1, w / 640);

  if (points && showMesh) {
    const size = 1.6 * unit;
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    for (const p of points) ctx.fillRect(X(p.x) - size / 2, p.y - size / 2, size, size);
    ctx.fillStyle = KEY_POINT_COLOR;
    for (const idx of KEY_POINTS) {
      const p = points[idx];
      ctx.beginPath();
      ctx.arc(X(p.x), p.y, 2.4 * unit, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (face) {
    const pad = (face.x2 - face.x1) * 0.1;
    ctx.strokeStyle = DRAW_COLORS[status];
    ctx.lineWidth = 3 * unit;
    ctx.lineCap = "round";
    drawCorners(mirrorBox(face, pad, w), 18 * unit);
  }

  if (alerts.phone) {
    ctx.font = `600 ${Math.round(13 * unit)}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    for (const b of phoneBoxes) {
      const r = mirrorBox(b, 0, w);
      ctx.strokeStyle = PHONE_COLOR;
      ctx.lineWidth = 2.5 * unit;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      const label = `Phone ${Math.round(b.score * 100)}%`;
      const tw = ctx.measureText(label).width + 10 * unit;
      const th = 20 * unit;
      ctx.fillStyle = PHONE_COLOR;
      ctx.fillRect(r.x, r.y - th, tw, th);
      ctx.fillStyle = "#1c1917";
      ctx.fillText(label, r.x + 5 * unit, r.y - 4 * unit);
    }
  }
}

function mirrorBox(b, pad, width) {
  const x1 = b.x1 - pad;
  const x2 = b.x2 + pad;
  return {
    x: mirrored ? width - x2 : x1,
    y: b.y1 - pad,
    w: x2 - x1,
    h: b.y2 - b.y1 + 2 * pad,
  };
}

function drawCorners({ x, y, w, h }, len) {
  const l = Math.min(len, w / 3, h / 3);
  ctx.beginPath();
  ctx.moveTo(x, y + l); ctx.lineTo(x, y); ctx.lineTo(x + l, y);
  ctx.moveTo(x + w - l, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + l);
  ctx.moveTo(x + w, y + h - l); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - l, y + h);
  ctx.moveTo(x + l, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - l);
  ctx.stroke();
}

// ─── Page updates ───────────────────────────────────────────────────────────

function renderStatus(status) {
  app.dataset.status = status;
  const label = STATUS_LABELS[status];
  // Only touch the live region when the text changes, so screen readers announce changes only.
  if (statusText.textContent !== label) statusText.textContent = label;
  hudStatus.textContent = label;
  const rounded = Math.round(score);
  scoreValue.textContent = String(rounded);
  hudScore.textContent = `${rounded}`;
  scoreFill.style.width = `${score}%`;
}

function renderAlerts() {
  for (const [key, li] of Object.entries(alertItems)) {
    if (li.dataset.unavailable) continue;
    const on = String(Boolean(alerts[key]));
    if (li.dataset.active === on) continue;
    li.dataset.active = on;
    li.querySelector(".state").textContent = alerts[key] ? "Detected" : "Clear";
  }
}

function renderIdle() {
  renderStatus("idle");
  for (const li of Object.values(alertItems)) {
    if (li.dataset.unavailable) continue;
    li.dataset.active = "idle";
    li.querySelector(".state").textContent = "–";
  }
  fpsText.textContent = "– FPS";
  renderMeasures();
}

function countFrame(now) {
  frames += 1;
  if (now - fpsSince >= 1000) {
    fpsText.textContent = `${Math.round((frames * 1000) / (now - fpsSince))} FPS`;
    frames = 0;
    fpsSince = now;
  }
}

function logChanges(status, faceFound, now) {
  for (const key of Object.keys(ALERT_LABELS)) {
    if (alerts[key] && !prevAlerts[key]) addEvent(ALERT_LABELS[key], "alert");
  }
  prevAlerts = { ...alerts };

  if (status === "danger" && prevStatus !== "danger") addEvent("Danger level reached", "danger");
  prevStatus = status;

  if (faceFound) {
    faceLostSince = null;
    faceLostLogged = false;
  } else {
    faceLostSince ??= now;
    if (!faceLostLogged && now - faceLostSince > 1000) {
      addEvent("Driver not found", "noface");
      faceLostLogged = true;
    }
  }
}

function addEvent(label, kind) {
  eventsList.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.dataset.kind = kind;
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const text = document.createElement("span");
  text.textContent = label;
  li.append(time, text);
  eventsList.prepend(li);
  while (eventsList.children.length > 8) eventsList.lastElementChild.remove();
}

function clearEvents() {
  eventsList.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "empty";
  empty.textContent = "No alerts yet";
  eventsList.append(empty);
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

function cameraError(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera access was blocked. Allow the camera for this site (icon in the address bar), then press Start again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found. Connect a webcam and press Start camera again.";
    case "NotReadableError":
      return "The camera is being used by another app. Close that app and try again.";
    default:
      return `Couldn't open the camera (${err?.name ?? "unknown error"}).`;
  }
}

function noAlerts() {
  return Object.fromEntries(Object.keys(ALERT_LABELS).map((key) => [key, false]));
}

// ─── Settings ───────────────────────────────────────────────────────────────

function loadThresholds() {
  const saved = storage.get("dms.thresholds", {});
  const result = { ...DEFAULT_THRESHOLDS };
  for (const key of Object.keys(result)) {
    if (typeof saved?.[key] === "number" && Number.isFinite(saved[key])) result[key] = saved[key];
  }
  return result;
}

function buildTuning() {
  for (const spec of TUNING) {
    const id = `tune-${spec.key}`;
    const row = document.createElement("div");
    row.className = "tune-row";
    row.innerHTML = `
      <label for="${id}">${spec.label}<small>${spec.hint}</small></label>
      <output class="live" title="Live value">–</output>
      <input id="${id}" type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}">
      <output class="value" for="${id}" title="Threshold"></output>`;
    const input = row.querySelector("input");
    const value = row.querySelector(".value");
    input.value = String(thresholds[spec.key]);
    value.textContent = thresholds[spec.key].toFixed(2);
    input.addEventListener("input", () => {
      thresholds[spec.key] = Number(input.value);
      value.textContent = thresholds[spec.key].toFixed(2);
      storage.set("dms.thresholds", thresholds);
    });
    spec.row = { live: row.querySelector(".live"), input, value };
    tuningRows.append(row);
  }
}

function renderMeasures() {
  if (!tuningPanel.open) return;
  for (const spec of TUNING) {
    const { live } = spec.row;
    const v = measures?.[spec.measure];
    if (v == null) {
      live.textContent = "–";
      live.dataset.over = "false";
      continue;
    }
    live.textContent = v.toFixed(2);
    const t = thresholds[spec.key];
    live.dataset.over = String(spec.above ? v > t : v < t);
  }
}

resetBtn.addEventListener("click", () => {
  thresholds = { ...DEFAULT_THRESHOLDS };
  storage.set("dms.thresholds", thresholds);
  for (const spec of TUNING) {
    spec.row.input.value = String(thresholds[spec.key]);
    spec.row.value.textContent = thresholds[spec.key].toFixed(2);
  }
});

tuningPanel.addEventListener("toggle", renderMeasures);

function renderSound() {
  soundBtn.setAttribute("aria-pressed", String(alarm.enabled));
  soundBtn.textContent = alarm.enabled ? "Sound on" : "Sound off";
}

soundBtn.addEventListener("click", () => {
  alarm.unlock();
  alarm.enabled = !alarm.enabled;
  storage.set("dms.sound", alarm.enabled);
  renderSound();
  if (alarm.enabled) alarm.beep();
});

meshToggle.addEventListener("change", () => {
  showMesh = meshToggle.checked;
  storage.set("dms.mesh", showMesh);
});
