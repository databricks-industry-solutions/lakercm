/**
 * Transcription API client.
 *
 * POSTs a WAV audio phrase to reviewer_app's /api/transcribe route. Relative
 * path only — the Vite dev proxy forwards /api, and prod is same-origin.
 */

export async function postTranscribe(wavBlob, seqId, signal) {
  const form = new FormData()
  form.append('audio', wavBlob, 'phrase.wav')
  form.append('seq_id', String(seqId))

  const resp = await fetch('/api/transcribe', {
    method: 'POST',
    body: form,
    signal,
    credentials: 'same-origin',
  })

  if (!resp.ok) {
    const detail = await resp.text().catch(() => '')
    throw new Error(`transcribe ${resp.status}: ${detail.slice(0, 200)}`)
  }
  return resp.json() // { text, seq_id }
}
