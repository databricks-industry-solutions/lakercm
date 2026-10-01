// Which extracted field on THIS page a knowledge-graph node came from, so
// selecting the node can light up its source on the document the same way
// hovering the field in the extraction panel does.
//
// The graph and the extraction share VALUES, not ids: the diagnosis node is
// `M54.50` and so is the `diagnosis code 1` field. So this matches on the value
// first, and only falls back to the field NAME for the one node whose value is
// never printed: the patient, a pseudonymous key derived from the member id.

// Lower-cased, punctuation folded to spaces -- but `.` kept, because it is part
// of an ICD-10 code ("M54.50" must not become "m54 50" and match "M54").
const norm = (v) =>
  String(v ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9.]+/g, ' ')
    .trim()

// Field-name fragments per edge. They break ties between equal values, and they
// gate the token match below, so a short code such as "197" cannot match a
// street number in an address.
const NAME_HINTS = {
  documentsPatient: ['member id', 'patient name'],
  billedTo: ['payer'],
  deniedFor: ['denial', 'carc', 'code'],
  hasDiagnosis: ['diagnosis'],
  hasProcedure: ['procedure', 'cpt', 'hcpcs'],
}

// Every field that can carry a value, in the extraction panel's order: the
// top-level identifiers, then each table row's cells.
export function allFields(parsed) {
  return [
    ...(parsed?.topLevel || []),
    ...(parsed?.rows || []).flatMap((r) => r.fields || []),
  ].filter((f) => f?.key && f.value != null && f.value !== '')
}

/**
 * The key of the field a node came from, or null when nothing on the page says it.
 *
 * `node` is `{ edge, neighbour, label }`: the spoke's edge, the entity's local
 * name, and its stored display label ("CARC 197" for denial reason `197`).
 */
export function findFieldForNode(parsed, node) {
  if (!node) return null
  const fields = allFields(parsed)
  const hints = NAME_HINTS[node.edge] || []
  const hinted = (f) => hints.some((h) => norm(f.name).includes(h))
  const targets = [node.neighbour, node.label].map(norm).filter(Boolean)

  // 1. The value itself: "M54.50", "97110", "Oakhollow" for `oakhollow`.
  const exact = fields.filter((f) => targets.includes(norm(f.value)))
  if (exact.length) return (exact.find(hinted) || exact[0]).key

  // 2. The value as a whole token of a hinted field: "CARC 197" for `197`.
  const token = fields.find(
    (f) => hinted(f) && targets.some((t) => norm(f.value).split(' ').includes(t))
  )
  if (token) return token.key

  // 3. The patient. Its node is a hash, so no value can match; point at the
  //    field it was derived from instead.
  if (node.edge === 'documentsPatient') {
    for (const h of hints) {
      const f = fields.find((x) => norm(x.name) === h)
      if (f) return f.key
    }
  }
  return null
}
