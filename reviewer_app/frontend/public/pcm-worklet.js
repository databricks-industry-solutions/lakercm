// AudioWorklet processor for live speech capture.
//
// Forwards Float32 PCM frames to the main thread in ~75 ms batches. A 128-sample
// render quantum fires ~375x/s at 48 kHz, so batching here avoids a postMessage
// flood. Resampling to 16 kHz is done on the main thread (OfflineAudioContext),
// not hand-rolled in the worklet.
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super()
    this._buf = []
    this._n = 0
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (!channel) return true
    const target = sampleRate * 0.075 // ~75 ms of samples at the native rate
    // Copy — the underlying buffer is reused by the engine after process().
    this._buf.push(channel.slice(0))
    this._n += channel.length
    if (this._n >= target) {
      this.port.postMessage(this._buf)
      this._buf = []
      this._n = 0
    }
    return true
  }
}

registerProcessor('pcm-worklet', PCMWorklet)
