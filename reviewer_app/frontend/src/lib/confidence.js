// Formatting a score so it cannot overstate itself, and reading a boolean that
// may not be a boolean.
//
// Both of these existed inline and both were wrong in the same direction: they
// turned an uncertain value into a confident one.

/**
 * Format a 0-1 score as a percentage, FLOORED to two decimals.
 *
 * `Math.round(score * 100)` turned 0.9996 into "100%" — and that number was
 * printed on the same line as "below the 92% auto-verify threshold". Rounding a
 * score that answers "did this clear the bar" is only safe in one direction, and
 * up is not it. 100% now means 100%.
 *
 * @param {number|null|undefined} score
 * @returns {string|null} e.g. "99.96%", "92%", or null when there is no score
 */
export function formatConfidencePct(score) {
  if (typeof score !== 'number' || Number.isNaN(score)) return null
  const floored = Math.floor(score * 10000) / 100
  // Trim "99.90" -> "99.9" and "92.00" -> "92" without touching "99.96".
  const text = floored.toFixed(2).replace(/\.?0+$/, '')
  return `${text}%`
}

/**
 * Read a boolean that may have arrived as a string, PRESERVING unknown as null.
 *
 * The Databricks Statement Execution API serialises booleans as the strings
 * "true"/"false". The previous check was `record?.is_automated === true`, which
 * read the string "true", a NULL and an absent field all as "not auto-verified"
 * — so a document whose routing was merely unknown was rendered as held, with a
 * reason invented to match. "We do not know" has to stay distinguishable from
 * "no" or the UI will keep asserting things it has not established.
 *
 * @param {*} value
 * @returns {boolean|null}
 */
export function coerceBool(value) {
  if (value === true || value === false) return value
  if (typeof value === 'string') {
    const text = value.trim().toLowerCase()
    if (text === 'true' || text === 't' || text === '1') return true
    if (text === 'false' || text === 'f' || text === '0') return false
    return null
  }
  if (typeof value === 'number' && !Number.isNaN(value)) return Boolean(value)
  return null
}
