import { describe, expect, it } from 'vitest'
import { coerceBool, formatConfidencePct } from './confidence'

describe('formatConfidencePct', () => {
  it('never rounds a near-ceiling score up to 100', () => {
    // 0.9996 is the document that prompted this work. Math.round(0.9996 * 100)
    // is 100, and "100%" was printed beside "below the 92% threshold".
    expect(formatConfidencePct(0.9996)).toBe('99.96%')
    expect(formatConfidencePct(0.99999)).toBe('99.99%')
    expect(formatConfidencePct(0.9199)).toBe('91.99%')
  })

  it('reserves 100% for an actual 1.0', () => {
    expect(formatConfidencePct(1)).toBe('100%')
  })

  it('keeps exact values clean', () => {
    expect(formatConfidencePct(0.92)).toBe('92%')
    expect(formatConfidencePct(0.84)).toBe('84%')
    expect(formatConfidencePct(0.841)).toBe('84.1%')
    expect(formatConfidencePct(0)).toBe('0%')
  })

  it('returns null rather than a misleading zero when there is no score', () => {
    expect(formatConfidencePct(null)).toBeNull()
    expect(formatConfidencePct(undefined)).toBeNull()
    expect(formatConfidencePct('0.99')).toBeNull()
    expect(formatConfidencePct(NaN)).toBeNull()
  })
})

describe('coerceBool', () => {
  it('reads the strings the Statement Execution API returns', () => {
    expect(coerceBool('true')).toBe(true)
    expect(coerceBool('True')).toBe(true)
    expect(coerceBool(' TRUE ')).toBe(true)
    expect(coerceBool('false')).toBe(false)
    expect(coerceBool('False')).toBe(false)
  })

  it('passes real booleans through', () => {
    expect(coerceBool(true)).toBe(true)
    expect(coerceBool(false)).toBe(false)
  })

  it('keeps unknown distinguishable from false', () => {
    // `is_automated === true` collapsed all of these to "not auto-verified",
    // which the UI then rendered as "held" with a fabricated reason.
    expect(coerceBool(null)).toBeNull()
    expect(coerceBool(undefined)).toBeNull()
    expect(coerceBool('')).toBeNull()
    expect(coerceBool('maybe')).toBeNull()
  })
})
