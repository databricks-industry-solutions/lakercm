import { useState } from 'react';
import './BoundingBoxLayer.css';

// Box coordinates are raw pixels in the original image's coordinate space.
// We render the SVG with viewBox=`0 0 ${imageWidth} ${imageHeight}` so the
// browser handles every scaling case (window resize, parent transform: scale,
// high-DPI) at rasterization time. Strokes stay visually constant via
// vectorEffect="non-scaling-stroke"; corner circles + labels are sized in
// viewBox units proportional to the image so they remain readable at any zoom.
//
// Hover and click are separate concerns. Hover is ephemeral (clears on
// mouse-leave). Click locks a selection that persists until the user clicks
// elsewhere or hits Esc — `selectedBboxIdx` drives a halo'd "selected"
// visual distinct from the dashed→solid hover treatment. The previous
// cursor-following tooltip is gone; tooltip-style information now lives in
// the docked InspectorCard rendered by the parent.

const BoundingBoxLayer = ({
  boxes,
  imageWidth,
  imageHeight,
  highlightIndices = null,
  hoveredBboxIdx = null,
  selectedBboxIdx = null,
  onBboxHover,
  onBboxClick,
}) => {
  const [internalHovered, setInternalHovered] = useState(null);

  const getBoxColor = (type) => {
    const colors = {
      DATE: '#3b82f6',
      TIME: '#3b82f6',
      MEDICAL_CONDITION: '#10b981',
      MEDICATION: '#10b981',
      PROCEDURE: '#10b981',
      TEST: '#10b981',
      PERSON: '#f59e0b',
      AGE: '#f59e0b',
      CONTACT: '#f59e0b',
      ID: '#f59e0b',
    };
    return colors[type] || '#6b7280';
  };

  const getBoxLabel = (type) => {
    const labels = {
      DATE: 'Date',
      TIME: 'Time',
      MEDICAL_CONDITION: 'Condition',
      MEDICATION: 'Medication',
      PROCEDURE: 'Procedure',
      TEST: 'Test',
      PERSON: 'Person',
      AGE: 'Age',
      CONTACT: 'Contact',
      ID: 'ID',
    };
    return labels[type] || (type || 'Region');
  };

  if (!boxes || boxes.length === 0) {
    return null;
  }

  // viewBox needs known dimensions; image hasn't loaded yet otherwise
  if (!imageWidth || !imageHeight) {
    return null;
  }

  const hasHighlightSet =
    highlightIndices instanceof Set && highlightIndices.size > 0;

  // viewBox-unit sizes for non-scaling-stroke geometry. Tied to image
  // dimensions so corner indicators + label tags stay legible across zoom
  // levels without manual scale propagation.
  const cornerRadius = Math.max(imageWidth, imageHeight) * 0.004;
  const labelFontSize = Math.max(10, imageHeight * 0.014);
  const labelPadX = labelFontSize * 0.5;
  const labelPadY = labelFontSize * 0.3;
  const labelHeight = labelFontSize + labelPadY * 2;

  return (
    <div
      className="bbox-overlay-wrapper"
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
    >
      <svg
        className="bounding-box-layer"
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        preserveAspectRatio="none"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          overflow: 'visible',
        }}
      >
        {boxes.map((box, index) => {
          const { x, y, width, height, type, originalIdx } = box;
          const color = getBoxColor(type);
          const isInternallyHovered = internalHovered === index;
          const isExternallyHovered =
            hoveredBboxIdx != null && hoveredBboxIdx === originalIdx;
          const isHovered = isInternallyHovered || isExternallyHovered;
          const isSelected =
            selectedBboxIdx != null && selectedBboxIdx === originalIdx;

          const isHighlighted =
            hasHighlightSet && highlightIndices.has(originalIdx);
          const isDimmed =
            (hasHighlightSet || selectedBboxIdx != null) &&
            !isHighlighted &&
            !isSelected;

          const classes = [
            'bbox-rect',
            isSelected ? 'is-selected' : '',
            isHighlighted ? 'is-highlighted' : '',
            isDimmed ? 'is-dimmed' : '',
            isHovered ? 'is-hovered' : '',
          ]
            .filter(Boolean)
            .join(' ');

          const strokeWidth = isSelected
            ? 3.5
            : isHovered
              ? 3
              : isHighlighted
                ? 3
                : 2;
          const fillOpacity = isSelected
            ? 0.22
            : isHighlighted
              ? 0.22
              : isHovered
                ? 0.15
                : isDimmed
                  ? 0.02
                  : 0.08;
          const rectOpacity = isDimmed ? 0.25 : 1;

          const labelText = getBoxLabel(type);
          const labelWidth = labelText.length * labelFontSize * 0.6 + labelPadX * 2;

          return (
            <g key={index} opacity={rectOpacity}>
              <rect
                className={classes}
                x={x}
                y={y}
                width={width}
                height={height}
                fill={color}
                fillOpacity={fillOpacity}
                stroke={color}
                strokeWidth={strokeWidth}
                strokeDasharray={
                  isSelected || isHovered || isHighlighted ? '0' : '4 2'
                }
                vectorEffect="non-scaling-stroke"
                style={{
                  pointerEvents: 'all',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={() => {
                  setInternalHovered(index);
                  onBboxHover?.(index);
                }}
                onMouseLeave={() => {
                  setInternalHovered(null);
                  onBboxHover?.(null);
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onBboxClick?.(index);
                }}
              />

              {(isHovered || isSelected) && (
                <>
                  <circle cx={x} cy={y} r={cornerRadius} fill={color} />
                  <circle cx={x + width} cy={y} r={cornerRadius} fill={color} />
                  <circle cx={x} cy={y + height} r={cornerRadius} fill={color} />
                  <circle
                    cx={x + width}
                    cy={y + height}
                    r={cornerRadius}
                    fill={color}
                  />
                </>
              )}

              {(isHovered || isHighlighted || isSelected) && (
                <g
                  transform={`translate(${x}, ${y - labelHeight - labelPadY})`}
                  style={{ pointerEvents: 'none' }}
                >
                  <rect
                    x="0"
                    y="0"
                    width={labelWidth}
                    height={labelHeight}
                    fill={color}
                    rx={labelFontSize * 0.3}
                    opacity={isSelected || isHovered ? 1 : 0.9}
                  />
                  <text
                    x={labelPadX}
                    y={labelHeight - labelPadY}
                    fill="white"
                    fontSize={labelFontSize}
                    fontWeight="600"
                    fontFamily="var(--font-display), sans-serif"
                  >
                    {labelText}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

export default BoundingBoxLayer;
