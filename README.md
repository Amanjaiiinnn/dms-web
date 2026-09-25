# Driver Monitoring System (Web)

Real-time driver monitoring that runs entirely in the browser. It watches the driver through the live camera on your device and flags drowsiness, yawning, distraction, looking down, phone use and smoking, then combines them into a 0–100 risk score with an audible alarm.

**Live demo:** https://amanjaiiinnn.github.io/dms-web/

All processing happens on your device with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) and [LiteRT.js](https://ai.google.dev/edge/litert/web). The video is never uploaded or recorded.

This is a browser port of a driver monitoring system I first built to run on an STM32MP2 edge board, where the models run on the board's NPU. Phone use and smoking are detected with the same custom-trained YOLOv8n model the board uses.

## What it detects

| Alert | Rule | Default |
|---|---|---|
| **Drowsy** | Both eyes closed (MediaPipe `eyeBlink` blendshapes) | score > 0.5 for 0.8 s |
| **Yawning** | Mouth height ÷ mouth width | > 0.28 for 0.4 s |
| **Distracted** | Upper lip to left cheek ÷ upper lip to right cheek (head turned) | outside 0.5–2.0 for 0.4 s |
| **Head down** | Nose position between forehead (0) and chin (1) | > 0.58 for 0.5 s |
| **Phone use** | Custom YOLOv8n model: "phone" box whose centre is near the face | probability ≥ 0.20 |
| **Smoking** | Same model: "cigarette" box whose centre is near the face | probability ≥ 0.40 |
| **Driver not found** | No face in the frame | – |

If several people are in view, the face closest to the centre of the frame is treated as the driver.

## Risk score

The score starts at 0 and always stays between 0 and 100. Several alerts add up.

| Condition | Change per reference frame |
|---|---|
| Yawning | +7 |
| Drowsy | +5 |
| Head down | +5 |
| Distracted | +2 |
| Phone use | +2 |
| Smoking | +2 |
| No face | +0.7 |
| Face found, no alerts | −5 |

| Score | Status |
|---|---|
| under 33 | Driver OK |
| 33 to under 66 | Warning |
| 66 and above | Danger (alarm beeps every 2 s) |

The board version applies these points once per frame, at a fixed frame rate. Browsers run anywhere from about 10 to 60 FPS, so here each change is multiplied by the elapsed time × `REF_FPS` (10). The score then moves at the same speed on a slow phone and a fast laptop.

## How it works

```mermaid
flowchart LR
  cam[Live camera] --> face[MediaPipe Face Landmarker<br/>478 points + blendshapes]
  face --> rules[Rules<br/>eyes · yawn · head turn · head down]
  cam --> phone[Custom YOLOv8n INT8<br/>phone + cigarette, every 150 ms]
  face --> near[Near the face?]
  phone --> near
  rules --> score[Risk score 0–100]
  near --> score
  score --> ui[Status · alerts · alarm]
```

1. The Face Landmarker runs on every new camera frame (GPU, with a CPU fallback) and returns 478 face points plus blendshape scores.
2. The rules turn those points into ratios. Each condition must last a short time (the hold time) before it becomes an alert, so a normal blink or a quick glance doesn't trigger one.
3. The phone/cigarette model runs in the background at most every 150 ms, with LiteRT.js on the CPU (about 150 ms per frame on a laptop). A box only counts if its centre falls in the area around the face where a phone held to the ear or a cigarette at the mouth would be.
4. The risk score is updated from the active alerts, and the page shows the status, the alerts, a log of recent events and an alarm when the score reaches Danger.

## Run it locally

The camera only works on `https://` or `localhost`, so serve the folder instead of opening `index.html` directly:

```bash
python -m http.server 8000
```

Then open http://localhost:8000 and press **Start camera**. The first visit downloads about 28 MB (the MediaPipe and LiteRT runtimes plus the two models); after that the browser caches them.

If the device has more than one camera (for example a laptop with a USB webcam), a **Camera** picker appears under the video. The choice is remembered.

## Deploy on GitHub Pages

1. Create a public repository named `dms-web` and push this folder to it.
2. On GitHub, open **Settings → Pages**. Under **Build and deployment**, set **Source** to **Deploy from a branch**, choose `main` and `/ (root)`, and save.
3. After about a minute the site is live at `https://<your-username>.github.io/dms-web/`. Pages serves it over HTTPS, which the camera needs.

There is no build step. Any static host (Netlify, Vercel, Cloudflare Pages) works the same way.

## Phone and cigarette model

`models/phone_smoke.tflite` is the custom YOLOv8n detector trained for the board (320×320 input, classes `phone` and `cigarette`, INT8 quantised, 3 MB). It is the board's `yolov8n_smk_call_custom` export with one change: LiteRT.js cannot take INT8 inputs or outputs, so [`tools/make_float_io_model.py`](tools/make_float_io_model.py) adds a float → int8 step at the input and an int8 → float step at the output. The network inside is untouched and gives identical results.

The browser pipeline matches the board's `smoking_calling_yolov8n_custom.py`: letterbox to 320×320 with grey padding, RGB scaled to 0–1, decode the (6, 2100) output, per-class threshold, drop boxes 4 px or smaller, non-maximum suppression at IoU 0.4, then keep boxes near the face.

Two differences from the board code:

- **Thresholds.** The model's class scores are already probabilities (it ends in a sigmoid). The board code applies a second sigmoid and uses 0.55 for phone and 0.60 for cigarette on that scale. Those are the same cut-offs as 0.20 and 0.40 on plain probabilities, which is what this version uses and shows in the Tuning panel.
- **Colour order.** The model was trained on RGB images. The browser feeds RGB. The board code passes OpenCV's BGR frames, which in testing hid real phones and produced false cigarette detections.

## Tuning

Open the **Tuning** panel on the page. Each row shows the live value next to its threshold, and the live value turns red when it would trigger the alert. Move the slider until the alert fires only when it should. Changes are saved in your browser.

To change the defaults for everyone, edit `DEFAULT_THRESHOLDS`, `HOLD_SECONDS` and the score values in [`js/config.js`](js/config.js).

In testing, a broad smile measured about 0.2 on the yawn ratio. If laughing or talking sets off the yawn alert, raise the yawn threshold to about 0.35.

## Tests

The rules, the score and the YOLO decoding are plain functions with unit tests. They run with Node's built-in test runner (Node 18 or newer, no install needed):

```bash
node --test
```

GitHub Actions runs them on every push.

## Project structure

```
dms-web/
├── index.html          Page layout
├── style.css           Styles, light and dark mode
├── js/
│   ├── main.js         Camera, models, detection loop, drawing, page updates
│   ├── rules.js        Face ratios, face selection, near-face check, hold timer
│   ├── detector.js     Runs the phone/cigarette model with LiteRT.js
│   ├── yolo.js         Letterbox, YOLOv8 output decoding, NMS
│   ├── score.js        Risk score and status
│   ├── alarm.js        Alarm beep (Web Audio API)
│   └── config.js       Model URLs, thresholds, hold times, score values
├── models/
│   └── phone_smoke.tflite   Custom YOLOv8n phone + cigarette model
├── tools/
│   └── make_float_io_model.py   Adds float input/output to an INT8 .tflite
├── tests/
│   ├── rules.test.js   Unit tests for rules.js and score.js
│   └── yolo.test.js    Unit tests for yolo.js
├── LICENSE
└── NOTICE              Third-party credits
```

## Differences from the edge version

| | STM32MP2 board | This web version |
|---|---|---|
| Camera | CSI or USB camera on the board | Any camera connected to the visitor's device |
| Face | Face detection, 468-point landmark and iris models on the NPU (`.nb`) | MediaPipe Face Landmarker in the browser (GPU or CPU) |
| Eyes | Eye height ÷ width from the iris model | `eyeBlink` blendshapes |
| Phone / smoking | Custom YOLOv8n INT8 model on the NPU (`.nb`) | Same model (`.tflite`) with LiteRT.js on the CPU |
| Timing | Points per frame | Points per second (frame-rate independent), plus hold times |

## Limitations

- This is a demo, not a certified safety system.
- With sunglasses the eyes can't be seen, so drowsiness can't be judged.
- Low light reduces landmark accuracy.
- The phone score of the INT8 model is sensitive near its threshold and can flicker between frames; the 0.6 s memory smooths this.
- The thresholds were tuned on a small number of faces and may need adjusting.

## Credits and license

Released under the [MIT License](LICENSE).

- The yawning, head-turn and head-down ratios are adapted from NXP's DMS demo (BSD-3-Clause). See [NOTICE](NOTICE).
- [MediaPipe Tasks Vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) and the Face Landmarker model, and [LiteRT.js](https://www.npmjs.com/package/@litertjs/core), are by Google (Apache 2.0). They are loaded from public CDNs at runtime.
- The phone and cigarette model was trained with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
