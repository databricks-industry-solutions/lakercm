import './SuggestedPrompts.css'

const PROMPTS = [
  { text: 'Show me recently uploaded documents' },
  { text: "What's the overall extraction accuracy?" },
  { text: 'Find documents labeled as prior authorization' },
  { text: 'Show the most recent reviews' },
]

function getFirstName(currentUser) {
  const display = currentUser?.display_name?.trim()
  return display ? display.split(/\s+/)[0] : null
}

function SuggestedPrompts({ onSelect, currentUser }) {
  const firstName = getFirstName(currentUser)
  return (
    <div className="suggested-prompts">
      <div className="suggested-prompts__header">
        {firstName && (
          <p className="suggested-prompts__greeting">Hi {firstName}!</p>
        )}
        <h1 className="suggested-prompts__title">What can I help you with?</h1>
      </div>
      <div className="suggested-prompts__grid">
        {PROMPTS.map((prompt) => (
          <button
            key={prompt.text}
            className="suggested-prompts__card"
            onClick={() => onSelect(prompt.text)}
          >
            <span className="suggested-prompts__text">{prompt.text}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default SuggestedPrompts
