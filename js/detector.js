// Runs the custom phone/cigarette model in the browser with LiteRT.js.

import { DETECTOR_MODEL_URL, LITERT_URL, LITERT_WASM_URL } from "./config.js";
import { INPUT_SIZE, decode, letterbox } from "./yolo.js";

export class PhoneSmokeDetector {
  static async create() {
    const litert = await import(LITERT_URL);
    // The runtime can only be loaded once per page; reuse it if it already is.
    await (litert.getGlobalLiteRtPromise() ?? litert.loadLiteRt(LITERT_WASM_URL));
    const model = await litert.loadAndCompile(DETECTOR_MODEL_URL, { accelerator: "wasm" });
    return new PhoneSmokeDetector(litert, model);
  }

  constructor(litert, model) {
    this.litert = litert;
    this.model = model;
    this.canvas = document.createElement("canvas");
    this.canvas.width = INPUT_SIZE;
    this.canvas.height = INPUT_SIZE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.input = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  }

  /**
   * Detects phones and cigarettes in `source` (a video, image or canvas of
   * width×height pixels). Returns boxes in source pixels scoring at least `floor`.
   */
  async detect(source, width, height, floor) {
    const lb = this.prepare(source, width, height);
    const tensor = new this.litert.Tensor(this.input, [1, INPUT_SIZE, INPUT_SIZE, 3]);
    let outputs;
    try {
      outputs = await this.model.run(tensor);
    } finally {
      tensor.delete();
    }
    try {
      const output = (await outputs[0].data()).slice();
      return decode(output, lb, width, height, floor);
    } finally {
      outputs.forEach((t) => t.delete());
    }
  }

  /** Letterboxes the frame into the 320×320 RGB input (grey padding, like the board). */
  prepare(source, width, height) {
    const lb = letterbox(width, height);
    const { ctx, input } = this;
    ctx.fillStyle = "rgb(114, 114, 114)";
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(source, lb.left, lb.top, lb.w, lb.h);
    const px = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
      input[j] = px[i] / 255;
      input[j + 1] = px[i + 1] / 255;
      input[j + 2] = px[i + 2] / 255;
    }
    return lb;
  }
}
