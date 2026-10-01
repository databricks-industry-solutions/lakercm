import '@testing-library/jest-dom/vitest'
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// Unmount between tests. Without this, a component that autosaves on unmount
// (ReviewNotes flushes a pending debounce when the document changes) would fire
// its save during the NEXT test and pollute that test's recorded API calls.
afterEach(() => {
  cleanup()
})

// jsdom implements no layout, so scrollIntoView does not exist on Element. The
// chat surfaces call it to keep the newest message in view; without this stub it
// throws from a setTimeout AFTER the test that triggered it has finished, which
// surfaces as an unhandled error attributed to whichever test ran last. Stubbing
// it here rather than per-file keeps that confusion from recurring.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
