import './ProgressRing.css';

const ProgressRing = ({
  percent = 0,
  size = 128,
  strokeWidth = 10,
  color = 'var(--accent)',
  trackColor = 'oklch(0.93 0.004 80)',
  label,
}) => {
  const clamped = Math.max(0, Math.min(100, percent));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (clamped / 100) * circumference;
  const cx = size / 2;
  const cy = size / 2;

  return (
    <div className="progress-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={trackColor}
          strokeWidth={strokeWidth}
        />
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          transform={`rotate(-90 ${cx} ${cy})`}
          className="progress-ring__arc"
        />
      </svg>
      <div className="progress-ring__center">
        <span className="progress-ring__pct" style={{ color }}>
          {Math.round(clamped)}
          <span className="progress-ring__pct-sym">%</span>
        </span>
        {label && <span className="progress-ring__label">{label}</span>}
      </div>
    </div>
  );
};

export default ProgressRing;
