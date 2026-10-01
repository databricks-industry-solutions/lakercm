import { useEffect, useRef, useState } from 'react';
import './ConnectionLine.css';

/**
 * ConnectionLine — soft cubic Bézier from a locked field row in the right
 * panel to its first citation bbox in the document image.
 *
 * Driven by a requestAnimationFrame loop while the line is active, so the
 * endpoints stay anchored through ANY layout change source: panel resize,
 * zoom transition, scroll on either pane, the locked-row inline-detail
 * expansion, window resize, etc. The previous implementation relied on
 * discrete scroll + ResizeObserver triggers and went stale during the gaps;
 * react-flow, react-archer, and react-xarrows all use rAF loops for the
 * same reason.
 *
 * Hides itself per-frame when either endpoint has zero size, sits offscreen,
 * or is unresolvable (e.g. row collapsed in a closed section). Pointer-events
 * are disabled on the SVG so the line never intercepts clicks on the bboxes
 * or panel rows underneath.
 */
export default function ConnectionLine({
  fieldRefs,
  fromKey,
  imageEl,
  imageWidth,
  imageHeight,
  box,
  visible,
}) {
  const [coords, setCoords] = useState(null);
  const lastCoordsRef = useRef(null);

  useEffect(() => {
    if (!visible) {
      setCoords(null);
      lastCoordsRef.current = null;
      return undefined;
    }

    let rafId = 0;

    const tick = () => {
      // Re-resolve the row ref every frame: the FieldRow may have just
      // mounted (e.g. after the panel auto-restored on bbox click) and
      // populated the ref Map AFTER the parent's last render. Reading
      // imperatively here avoids the one-frame lag.
      const fromEl =
        fieldRefs?.current && fromKey != null
          ? fieldRefs.current.get(fromKey)
          : null;
      let next = null;
      if (fromEl && imageEl && box && imageWidth && imageHeight) {
        const fromRect = fromEl.getBoundingClientRect();
        const imageRect = imageEl.getBoundingClientRect();

        if (
          fromRect.width > 0 &&
          fromRect.height > 0 &&
          imageRect.width > 0 &&
          imageRect.height > 0
        ) {
          const sx = imageRect.width / imageWidth;
          const sy = imageRect.height / imageHeight;
          // Anchor: bbox's RIGHT edge (faces the panel) → row's LEFT edge.
          const targetX = imageRect.left + (box.x + box.width) * sx;
          const targetY = imageRect.top + (box.y + box.height / 2) * sy;
          const sourceX = fromRect.left;
          const sourceY = fromRect.top + fromRect.height / 2;

          // Clip when either endpoint is outside the viewport — keeps the
          // line from drawing across whitespace when the user scrolls one
          // half of the page off-screen.
          const vh = window.innerHeight;
          const inViewport =
            targetY >= 0 &&
            targetY <= vh &&
            sourceY >= 0 &&
            sourceY <= vh;
          // Sanity: source should sit to the right of target. If the panel
          // is on the wrong side (e.g. mobile reflow), don't draw.
          const correctOrientation = sourceX > targetX + 8;

          if (inViewport && correctOrientation) {
            next = {
              x1: targetX,
              y1: targetY,
              x2: sourceX,
              y2: sourceY,
            };
          }
        }
      }

      // Only update React state when the coords meaningfully changed —
      // avoids a render every frame when the layout is static.
      const last = lastCoordsRef.current;
      const same =
        last &&
        next &&
        Math.abs(last.x1 - next.x1) < 0.5 &&
        Math.abs(last.y1 - next.y1) < 0.5 &&
        Math.abs(last.x2 - next.x2) < 0.5 &&
        Math.abs(last.y2 - next.y2) < 0.5;
      const bothNull = !last && !next;
      if (!same && !bothNull) {
        lastCoordsRef.current = next;
        setCoords(next);
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [visible, fieldRefs, fromKey, imageEl, imageWidth, imageHeight, box]);

  if (!visible || !coords) return null;

  const { x1, y1, x2, y2 } = coords;
  const dx = x2 - x1;
  // Soft cubic — control points offset 40% of horizontal distance, on the
  // SAME y as their respective endpoint so the curve has a flat horizontal
  // approach at each end (d3-sankey style; reads as a flowing connection).
  const cp1x = x1 + dx * 0.4;
  const cp1y = y1;
  const cp2x = x2 - dx * 0.4;
  const cp2y = y2;

  return (
    <svg
      className="connection-line-svg"
      width="100%"
      height="100%"
      style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 25,
      }}
      aria-hidden="true"
    >
      <path
        className="connection-line-path"
        d={`M ${x1} ${y1} C ${cp1x} ${cp1y} ${cp2x} ${cp2y} ${x2} ${y2}`}
      />
      <circle className="connection-line-end" cx={x1} cy={y1} r={3.5} />
      <circle className="connection-line-end" cx={x2} cy={y2} r={3.5} />
    </svg>
  );
}
