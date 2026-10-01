/**
 * useLiveTranscription — live microphone → text for the chat composer.
 *
 * Pipeline: getUserMedia → AudioWorklet (16 kHz-bound PCM batches) → energy VAD
 * segments speech into phrases → each phrase is encoded to a 16 kHz mono 16-bit
 * WAV and POSTed to /api/transcribe. While a phrase is voiced we re-transcribe
 * the phrase-so-far every ~2.5 s (interim); on ~1.5 s of silence we do a final
 * transcribe and commit the phrase.
 *
 * "Live" is chunky/lagged (~1-3 s) — Databricks serving is request/response,
 * there is no bidirectional streaming ASR.
 *
 * Exposes: { isRecording, start, stop, committedPhrases, interimText, error,
 *            supported }. `stop()` resolves to the final full transcript string.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { postTranscribe } from '../api/transcribeApi'

const TARGET_SR = 16000
const INTERIM_MS = 2500 // re-transcribe the growing phrase this often while voiced
const SILENCE_MS = 1500 // this much silence finalizes a phrase
const MAX_PHRASE_MS = 25000 // hard cap so one phrase can't grow unbounded
const VAD_RMS = 0.008 // energy threshold for "voiced"
const TICK_MS = 250 // boundary-check cadence

const isSupported = () =>
  typeof window !== 'undefined' &&
  window.isSecureContext &&
  !!navigator.mediaDevices?.getUserMedia &&
  !!(window.AudioContext || window.webkitAudioContext)

function concatFloat32(chunks, total) {
  const out = new Float32Array(total)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.length
  }
  return out
}

async function resampleTo(float32, fromSR, toSR) {
  if (fromSR === toSR || float32.length === 0) return float32
  const length = Math.max(1, Math.round((float32.length * toSR) / fromSR))
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext
  const offline = new OfflineCtx(1, length, toSR)
  const buf = offline.createBuffer(1, float32.length, fromSR)
  buf.copyToChannel(float32, 0)
  const src = offline.createBufferSource()
  src.buffer = buf
  src.connect(offline.destination)
  src.start()
  const rendered = await offline.startRendering()
  return rendered.getChannelData(0)
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM fmt chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate (sr * blockAlign)
  view.setUint16(32, 2, true) // block align (1ch * 16-bit / 8)
  view.setUint16(34, 16, true) // bits/sample
  writeStr(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  let off = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    off += 2
  }
  return new Blob([view], { type: 'audio/wav' })
}

export function useLiveTranscription() {
  const [isRecording, setIsRecording] = useState(false)
  const [committedPhrases, setCommittedPhrases] = useState([])
  const [interimText, setInterimText] = useState('')
  const [error, setError] = useState(null)
  const supported = isSupported()

  // Audio graph
  const ctxRef = useRef(null)
  const streamRef = useRef(null)
  const nodeRef = useRef(null)
  const sourceRef = useRef(null)
  const sinkRef = useRef(null)
  const tickRef = useRef(null)

  // Current-phrase buffer (native sample rate) + segmentation clocks
  const chunksRef = useRef([])
  const sampleCountRef = useRef(0)
  const nativeSRRef = useRef(TARGET_SR)
  const phraseStartRef = useRef(0)
  const lastVoiceRef = useRef(0)
  const lastInterimRef = useRef(0)

  // Request sequencing (drop stale interims) + per-phrase epoch
  const seqRef = useRef(0)
  const lastAppliedRef = useRef(-1)
  const epochRef = useRef(0)
  const abortRef = useRef(null)
  const hasVoiceRef = useRef(false) // did the current phrase contain any speech?
  const startingRef = useRef(false) // re-entrancy guard for start()
  const finalizeInFlightRef = useRef(null) // last tick-triggered finalize promise

  // Mirror committed phrases in a ref so stop() can resolve synchronously
  const committedRef = useRef([])

  const resetPhrase = useCallback(() => {
    chunksRef.current = []
    sampleCountRef.current = 0
    phraseStartRef.current = 0
    hasVoiceRef.current = false
    epochRef.current += 1 // invalidate any in-flight interim for the old phrase
  }, [])

  const snapshotWav = useCallback(async () => {
    const total = sampleCountRef.current
    if (total === 0) return null
    const merged = concatFloat32(chunksRef.current, total)
    const resampled = await resampleTo(merged, nativeSRRef.current, TARGET_SR)
    return encodeWAV(resampled, TARGET_SR)
  }, [])

  const sendInterim = useCallback(async () => {
    const epoch = epochRef.current
    const wav = await snapshotWav()
    if (!wav) return
    const id = seqRef.current++
    try {
      const resp = await postTranscribe(wav, id, abortRef.current?.signal)
      // Drop if the phrase was committed/reset, or a newer interim already won.
      if (epoch !== epochRef.current) return
      if (resp.seq_id < lastAppliedRef.current) return
      lastAppliedRef.current = resp.seq_id
      setInterimText(resp.text || '')
    } catch {
      // Aborted or a transient error — interim is best-effort; ignore.
    }
  }, [snapshotWav])

  const finalizePhrase = useCallback(async () => {
    // Detach the buffer SYNCHRONOUSLY (before any await) so a concurrent tick
    // can't re-finalize the same phrase.
    const total = sampleCountRef.current
    const chunks = chunksRef.current
    resetPhrase()
    setInterimText('')
    if (total === 0) return

    let wav
    try {
      const merged = concatFloat32(chunks, total)
      const resampled = await resampleTo(merged, nativeSRRef.current, TARGET_SR)
      wav = encodeWAV(resampled, TARGET_SR)
    } catch {
      return
    }
    const id = seqRef.current++
    try {
      const resp = await postTranscribe(wav, id) // no abort — must complete
      const text = (resp.text || '').trim()
      if (text) {
        committedRef.current = [...committedRef.current, text]
        setCommittedPhrases(committedRef.current)
      }
    } catch {
      // A failed final drops the phrase rather than corrupting the transcript.
    }
  }, [resetPhrase])

  const onTick = useCallback(() => {
    // No speech captured yet — drop buffered silence so we never encode/POST
    // silence (which wastes FM calls + rate budget and returns empty→502), and
    // memory stays bounded while the mic is open but idle.
    if (!hasVoiceRef.current) {
      if (sampleCountRef.current > 0) resetPhrase()
      return
    }
    const now = performance.now()
    const silent = now - lastVoiceRef.current >= SILENCE_MS
    const tooLong = now - phraseStartRef.current >= MAX_PHRASE_MS
    if (silent || tooLong) {
      finalizeInFlightRef.current = finalizePhrase()
      return
    }
    if (now - lastInterimRef.current >= INTERIM_MS) {
      lastInterimRef.current = now
      sendInterim()
    }
  }, [finalizePhrase, sendInterim, resetPhrase])

  const onBatch = useCallback((batch) => {
    let voiced = false
    for (const frame of batch) {
      chunksRef.current.push(frame)
      sampleCountRef.current += frame.length
      // RMS of the frame for VAD.
      let sum = 0
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i]
      if (Math.sqrt(sum / frame.length) >= VAD_RMS) voiced = true
    }
    const now = performance.now()
    if (phraseStartRef.current === 0) {
      phraseStartRef.current = now
      lastInterimRef.current = now
      lastVoiceRef.current = now
    }
    if (voiced) {
      hasVoiceRef.current = true
      lastVoiceRef.current = now
    }
  }, [])

  const teardownAudio = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current)
      tickRef.current = null
    }
    try { nodeRef.current?.disconnect() } catch { /* ignore */ }
    try { sourceRef.current?.disconnect() } catch { /* ignore */ }
    try { sinkRef.current?.disconnect() } catch { /* ignore */ }
    try { streamRef.current?.getTracks().forEach((t) => t.stop()) } catch { /* ignore */ }
    try { ctxRef.current?.close() } catch { /* ignore */ }
    nodeRef.current = null
    sourceRef.current = null
    sinkRef.current = null
    streamRef.current = null
    ctxRef.current = null
  }, [])

  const start = useCallback(async () => {
    // isRecording flips true only after getUserMedia resolves, so guard with a
    // synchronous ref too — otherwise a fast double-click starts two graphs and
    // leaks the first stream/AudioContext/timer.
    if (!supported || isRecording || startingRef.current) return
    startingRef.current = true
    setError(null)
    setCommittedPhrases([])
    setInterimText('')
    committedRef.current = []
    seqRef.current = 0
    lastAppliedRef.current = -1
    resetPhrase()
    abortRef.current = new AbortController()

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      })
      streamRef.current = stream
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      const ctx = new AudioCtx()
      ctxRef.current = ctx
      nativeSRRef.current = ctx.sampleRate
      const source = ctx.createMediaStreamSource(stream)
      sourceRef.current = source

      // Muted sink so the processing node runs without echoing the mic.
      const sink = ctx.createGain()
      sink.gain.value = 0
      sink.connect(ctx.destination)
      sinkRef.current = sink

      if (window.AudioWorklet && ctx.audioWorklet) {
        const url = new URL('/pcm-worklet.js', window.location.origin).href
        await ctx.audioWorklet.addModule(url)
        const node = new AudioWorkletNode(ctx, 'pcm-worklet')
        node.port.onmessage = (e) => onBatch(e.data)
        source.connect(node)
        node.connect(sink)
        nodeRef.current = node
      } else {
        // Fallback: ScriptProcessorNode (deprecated but broadly available).
        const node = ctx.createScriptProcessor(4096, 1, 1)
        node.onaudioprocess = (e) => onBatch([e.inputBuffer.getChannelData(0).slice(0)])
        source.connect(node)
        node.connect(sink)
        nodeRef.current = node
      }

      tickRef.current = setInterval(onTick, TICK_MS)
      setIsRecording(true)
    } catch (err) {
      teardownAudio()
      setError(err?.name === 'NotAllowedError' ? 'Microphone permission denied' : 'Microphone unavailable')
      setIsRecording(false)
    } finally {
      startingRef.current = false
    }
  }, [supported, isRecording, resetPhrase, onBatch, onTick, teardownAudio])

  const stop = useCallback(async () => {
    if (!isRecording) return committedRef.current.join(' ')
    setIsRecording(false)
    teardownAudio()
    // Abort in-flight interims so a late interim can't clobber the final text.
    try { abortRef.current?.abort() } catch { /* ignore */ }
    // Await any tick-triggered finalize already in flight so its committed
    // phrase isn't dropped, then finalize whatever remains in the buffer.
    try { await finalizeInFlightRef.current } catch { /* ignore */ }
    try { await finalizePhrase() } catch { /* ignore */ }
    setInterimText('')
    return committedRef.current.join(' ')
  }, [isRecording, teardownAudio, finalizePhrase])

  // Best-effort cleanup on unmount.
  useEffect(() => () => teardownAudio(), [teardownAudio])

  return { isRecording, start, stop, committedPhrases, interimText, error, supported }
}
