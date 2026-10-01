import './MicButton.css'

/**
 * Microphone toggle for the chat composer. 36px circle matching the send
 * button. Idle = warm grey; recording = pulsing red; disabled = greyed.
 * No Databricks brand — uses the existing warm-gray design system.
 */
function MicButton({ isRecording, onToggle, disabled, title }) {
  const label = isRecording ? 'Stop dictation' : 'Start dictation'
  return (
    <button
      type="button"
      className={`chat-mic ${isRecording ? 'chat-mic--recording' : ''}`}
      onClick={onToggle}
      disabled={disabled}
      aria-label={label}
      aria-pressed={isRecording}
      title={title || label}
    >
      {isRecording ? (
        // Stop square while recording.
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="9" y="2" width="6" height="12" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0" />
          <line x1="12" y1="17" x2="12" y2="21" />
          <line x1="8" y1="21" x2="16" y2="21" />
        </svg>
      )}
    </button>
  )
}

export default MicButton
