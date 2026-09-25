// Short double beep through the Web Audio API. Browsers only allow sound
// after a click, so unlock() must be called from a click handler.

export class Alarm {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.lastBeep = 0;
  }

  unlock() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!this.ctx && AudioCtx) this.ctx = new AudioCtx();
    this.ctx?.resume();
  }

  beep() {
    if (!this.enabled || !this.ctx) return;
    const start = this.ctx.currentTime;
    for (const offset of [0, 0.22]) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.0001, start + offset);
      gain.gain.exponentialRampToValueAtTime(0.3, start + offset + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + offset + 0.16);
      osc.connect(gain).connect(this.ctx.destination);
      osc.start(start + offset);
      osc.stop(start + offset + 0.18);
    }
  }

  /** Beeps on entering danger, then every `everyMs` while it lasts. */
  update(inDanger, nowMs, everyMs = 2000) {
    if (!inDanger) {
      this.lastBeep = 0;
      return;
    }
    if (nowMs - this.lastBeep >= everyMs) {
      this.beep();
      this.lastBeep = nowMs;
    }
  }
}
