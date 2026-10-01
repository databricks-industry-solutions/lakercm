import { describe, expect, it } from 'vitest'
import { findFieldForNode } from './kgFieldMatch'

// Shaped like parseIdentifiers() output for a family denial document, with one
// trap: the address starts with "197", the same digits as the denial reason.
const PARSED = {
  topLevel: [
    { key: 'id:0', name: 'page header', value: 'Oakhollow Health Plan' },
    { key: 'id:1', name: 'address', value: '197 Figueroa Forest Apt. 511' },
    { key: 'id:2', name: 'patient name', value: 'Steven Mccoy' },
    { key: 'id:3', name: 'member id', value: 'OAK256780352' },
    { key: 'id:4', name: 'payer', value: 'Oakhollow' },
    { key: 'id:5', name: 'diagnosis code 1', value: 'M54.50' },
    { key: 'id:6', name: 'diagnosis code 2', value: 'M25.561' },
    { key: 'id:7', name: 'procedure code', value: '97110' },
    { key: 'id:8', name: 'code', value: 'CARC 197' },
    { key: 'id:9', name: 'appeal deadline days', value: '' },
  ],
  rows: [{ rowNum: '1', fields: [{ key: 'id:20', name: 'cpt', value: '99213' }] }],
}

const find = (node) => findFieldForNode(PARSED, node)

describe('findFieldForNode', () => {
  it('matches a code node to the field holding that code', () => {
    expect(find({ edge: 'hasDiagnosis', neighbour: 'M54.50' })).toBe('id:5')
    expect(find({ edge: 'hasDiagnosis', neighbour: 'M25.561' })).toBe('id:6')
    expect(find({ edge: 'hasProcedure', neighbour: '97110' })).toBe('id:7')
  })

  it('ignores case, so the payer node `oakhollow` finds "Oakhollow"', () => {
    // ...and the payer field, not the page header that merely contains it.
    expect(find({ edge: 'billedTo', neighbour: 'oakhollow', label: 'Oakhollow' })).toBe('id:4')
  })

  it('matches a denial reason by its stored label', () => {
    expect(find({ edge: 'deniedFor', neighbour: '197', label: 'CARC 197' })).toBe('id:8')
  })

  it('finds a bare code inside a hinted field, never inside an address', () => {
    // Without the name hint, "197" would light up the street number.
    expect(find({ edge: 'deniedFor', neighbour: '197', label: null })).toBe('id:8')
  })

  it('points the patient at the member id it was derived from', () => {
    // The node is a pseudonymous hash, so no printed value can match it.
    expect(find({ edge: 'documentsPatient', neighbour: '837274cc5d1a635f', label: 'Patient 837274' })).toBe('id:3')
  })

  it('searches table rows as well as top-level fields', () => {
    expect(find({ edge: 'hasProcedure', neighbour: '99213' })).toBe('id:20')
  })

  it('keeps the dot in a code, so M54 does not match M54.50', () => {
    expect(find({ edge: 'hasDiagnosis', neighbour: 'M54' })).toBeNull()
  })

  it('returns null when nothing on the page says it', () => {
    expect(find({ edge: 'hasDiagnosis', neighbour: 'Z99.89' })).toBeNull()
    expect(find(null)).toBeNull()
    expect(findFieldForNode(null, { edge: 'billedTo', neighbour: 'oakhollow' })).toBeNull()
  })
})
