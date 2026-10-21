import React from 'react';

interface PciGaugeProps {
  score: number;
  conditionCategory: string;
}

export const PciGauge: React.FC<PciGaugeProps> = ({ score, conditionCategory }) => {
  // Clamp score between 0 and 100
  const clampedScore = Math.max(0, Math.min(100, score));

  // Semi-circle angle calculation: 0 score = -90 deg, 100 score = +90 deg
  const needleRotation = -90 + (clampedScore / 100) * 180;

  // Color selection based on PCI tiers
  let statusColor = '#22C55E'; // Green
  let statusBadgeBg = 'bg-emerald-50 text-emerald-700 border-emerald-200';
  if (clampedScore < 40) {
    statusColor = '#EF4444'; // Red (Repair Immediate)
    statusBadgeBg = 'bg-rose-50 text-rose-700 border-rose-200';
  } else if (clampedScore < 60) {
    statusColor = '#F97316'; // Orange (Repair It)
    statusBadgeBg = 'bg-orange-50 text-orange-700 border-orange-200';
  } else if (clampedScore < 80) {
    statusColor = '#EAB308'; // Yellow (Maintain Properly)
    statusBadgeBg = 'bg-amber-50 text-amber-700 border-amber-200';
  }

  return (
    <div className="glass-surface p-6 flex flex-col items-center justify-between text-center relative overflow-hidden">
      <div className="w-full flex items-center justify-between mb-2">
        <span className="eyebrow-label">Structural Metric</span>
        <span className={`text-xs px-2.5 py-1 rounded-full border font-semibold ${statusBadgeBg}`}>
          {conditionCategory}
        </span>
      </div>

      <div className="relative w-64 h-36 flex items-end justify-center my-2">
        {/* SVG Semi-Circle Arc */}
        <svg viewBox="0 0 200 115" className="w-full h-full overflow-visible">
          {/* Background Track Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#F1EAE8"
            strokeWidth="16"
            strokeLinecap="round"
          />

          {/* Repair Immediate Zone (0 - 40) */}
          <path
            d="M 20 100 A 80 80 0 0 1 75 24"
            fill="none"
            stroke="rgba(239, 68, 68, 0.55)"
            strokeWidth="16"
            strokeLinecap="round"
          />

          {/* Repair It Zone (40 - 60) */}
          <path
            d="M 75 24 A 80 80 0 0 1 125 24"
            fill="none"
            stroke="rgba(249, 115, 22, 0.55)"
            strokeWidth="16"
          />

          {/* Maintain Properly Zone (60 - 80) */}
          <path
            d="M 125 24 A 80 80 0 0 1 165 53"
            fill="none"
            stroke="rgba(234, 179, 8, 0.55)"
            strokeWidth="16"
          />

          {/* Good Zone (80 - 100) */}
          <path
            d="M 165 53 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="rgba(34, 197, 94, 0.60)"
            strokeWidth="16"
            strokeLinecap="round"
          />

          {/* Needle Center Pivot */}
          <circle cx="100" cy="100" r="7" fill={statusColor} />
          <circle cx="100" cy="100" r="3" fill="#FFFFFF" />

          {/* Rotating Needle */}
          <g transform={`rotate(${needleRotation} 100 100)`}>
            <polygon points="98,100 102,100 100,24" fill={statusColor} />
          </g>

          {/* Tick Marks */}
          <text x="14" y="112" fill="#7C6A6D" fontSize="9" fontFamily="Poppins">0</text>
          <text x="64" y="18" fill="#7C6A6D" fontSize="9" fontFamily="Poppins">40</text>
          <text x="122" y="18" fill="#7C6A6D" fontSize="9" fontFamily="Poppins">60</text>
          <text x="168" y="48" fill="#7C6A6D" fontSize="9" fontFamily="Poppins">80</text>
          <text x="184" y="112" fill="#7C6A6D" fontSize="9" fontFamily="Poppins">100</text>
        </svg>
      </div>

      {/* PCI Numeric Score Display */}
      <div className="mt-1">
        <div className="flex items-baseline justify-center gap-1">
          <span className="font-heading font-bold text-4xl text-[#2A1418] tracking-tight">
            {clampedScore.toFixed(1)}
          </span>
          <span className="text-sm font-semibold text-[#7C6A6D]">/ 100</span>
        </div>
        <p className="text-xs font-heading font-semibold text-[#7C6A6D] uppercase tracking-wider mt-0.5">
          Pavement Condition Index (PCI)
        </p>
      </div>

      <div className="w-full mt-4 pt-3 border-t border-[#9B4A57]/10 grid grid-cols-4 gap-1 text-[11px] text-[#7C6A6D] text-center">
        <span className="flex items-center justify-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" /> &lt;40 Resurface
        </span>
        <span className="flex items-center justify-center gap-1">
          <span className="w-2 h-2 rounded-full bg-orange-500 shrink-0" /> 40-60 Repair
        </span>
        <span className="flex items-center justify-center gap-1">
          <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0" /> 60-80 Maintain
        </span>
        <span className="flex items-center justify-center gap-1">
          <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" /> &gt;80 Good
        </span>
      </div>
    </div>
  );
};
