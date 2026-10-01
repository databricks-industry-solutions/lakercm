import './DonutChart.css';

const DonutChart = ({ correct = 0, partiallyCorrect = 0, incorrect = 0 }) => {
  const total = correct + partiallyCorrect + incorrect;
  const accuracy = total > 0 ? Math.round((correct / total) * 100) : 0;

  // SVG donut parameters
  const size = 180;
  const strokeWidth = 24;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const cx = size / 2;
  const cy = size / 2;

  // Calculate segment lengths
  const correctLen = total > 0 ? (correct / total) * circumference : 0;
  const partialLen = total > 0 ? (partiallyCorrect / total) * circumference : 0;
  const incorrectLen = total > 0 ? (incorrect / total) * circumference : 0;

  // Offsets (each segment starts where the previous ended)
  const correctOffset = 0;
  const partialOffset = correctLen;
  const incorrectOffset = correctLen + partialLen;

  return (
    <div className="donut-chart">
      <div className="donut-svg-wrapper">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* Background ring */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="oklch(0.95 0.005 80)"
            strokeWidth={strokeWidth}
          />

          {total > 0 && (
            <>
              {/* Correct segment */}
              {correct > 0 && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  fill="none"
                  stroke="oklch(0.48 0.20 155)"
                  strokeWidth={strokeWidth}
                  strokeDasharray={`${correctLen} ${circumference - correctLen}`}
                  strokeDashoffset={-correctOffset}
                  strokeLinecap="butt"
                  transform={`rotate(-90 ${cx} ${cy})`}
                  className="donut-segment"
                />
              )}

              {/* Partially correct segment */}
              {partiallyCorrect > 0 && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  fill="none"
                  stroke="oklch(0.60 0.20 60)"
                  strokeWidth={strokeWidth}
                  strokeDasharray={`${partialLen} ${circumference - partialLen}`}
                  strokeDashoffset={-partialOffset}
                  strokeLinecap="butt"
                  transform={`rotate(-90 ${cx} ${cy})`}
                  className="donut-segment"
                />
              )}

              {/* Incorrect segment */}
              {incorrect > 0 && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  fill="none"
                  stroke="oklch(0.48 0.22 25)"
                  strokeWidth={strokeWidth}
                  strokeDasharray={`${incorrectLen} ${circumference - incorrectLen}`}
                  strokeDashoffset={-incorrectOffset}
                  strokeLinecap="butt"
                  transform={`rotate(-90 ${cx} ${cy})`}
                  className="donut-segment"
                />
              )}
            </>
          )}
        </svg>

        {/* Center text */}
        <div className="donut-center">
          <span className="donut-pct">{accuracy}%</span>
          <span className="donut-label">Accuracy</span>
        </div>
      </div>

      {/* Legend */}
      <div className="donut-legend">
        <div className="donut-legend-item">
          <span className="donut-legend-dot donut-dot-correct" />
          <span className="donut-legend-text">Correct</span>
          <span className="donut-legend-value">{correct}</span>
        </div>
        <div className="donut-legend-item">
          <span className="donut-legend-dot donut-dot-partial" />
          <span className="donut-legend-text">Partial</span>
          <span className="donut-legend-value">{partiallyCorrect}</span>
        </div>
        <div className="donut-legend-item">
          <span className="donut-legend-dot donut-dot-incorrect" />
          <span className="donut-legend-text">Incorrect</span>
          <span className="donut-legend-value">{incorrect}</span>
        </div>
      </div>
    </div>
  );
};

export default DonutChart;
