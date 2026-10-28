// Runs on the audio thread. Resamples the microphone to 16 kHz mono PCM16 and posts
// 20 ms frames (320 samples) to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // `sampleRate` is a global in the worklet scope
    this.pos = 0; // fractional read position inside this.buf
    this.buf = new Float32Array(0);
    this.out = new Int16Array(320);
    this.outLen = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    const merged = new Float32Array(this.buf.length + input.length);
    merged.set(this.buf);
    merged.set(input, this.buf.length);
    this.buf = merged;

    // Linear-interpolation resampler.
    while (this.pos + 1 < this.buf.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const s = this.buf[i] * (1 - frac) + this.buf[i + 1] * frac;
      this.out[this.outLen++] = Math.max(-1, Math.min(1, s)) * 32767;
      if (this.outLen === 320) {
        const copy = this.out.buffer.slice(0);
        this.port.postMessage(copy, [copy]);
        this.outLen = 0;
      }
      this.pos += this.ratio;
    }

    const consumed = Math.floor(this.pos);
    this.buf = this.buf.slice(consumed);
    this.pos -= consumed;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);