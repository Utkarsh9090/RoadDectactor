import React from 'react';
import type { DefectCounts } from '../../types';
import { DEDUCT_WEIGHTS } from '../../lib/pci';

interface DefectBarChartProps {
  defects: DefectCounts;
}

export const DefectBarChart: React.FC<DefectBarChartProps> = ({ defects }) => {
  const items = [
    {
      name: 'Utility Digs',
      count: defects.digs || 0,
      penalty: DEDUCT_WEIGHTS.dig,
      color: '#9B4A57', // Light Maroon
      barBg: 'bg-[#9B4A57]',
    },
    {
      name: 'Asphalt Patches',
      count: defects.patches || 0,
      penalty: DEDUCT_WEIGHTS.patch,
      color: '#7A3441', // Deep Maroon
      barBg: 'bg-[#7A3441]',
    },
    {
      name: 'Surface Cracks',
      count: defects.cracks || 0,
      penalty: DEDUCT_WEIGHTS.crack,
      color: '#D2410D', // Blood Orange
      barBg: 'bg-[#D2410D]',
    },
    {
      name: 'Structural Potholes',
      count: defects.potholes || 0,
      penalty: DEDUCT_WEIGHTS.pothole,
      color: '#DC2626', // Red
      barBg: 'bg-red-600',
    },
  ];

  const maxCount = Math.max(1, ...items.map((i) => i.count));
  const totalDeducts = items.reduce((acc, i) => acc + i.count * i.penalty, 0);

  return (
    <div className="glass-surface p-6 flex flex-col justify-between">
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="eyebrow-label">Surface Diagnostics</span>
          <h3 className="font-heading font-bold text-lg text-[#2A1418]">
            Defect Distribution & Deducts
          </h3>
        </div>
        <div className="text-right">
          <span className="text-xs text-[#7C6A6D]">Total Deduct Points</span>
          <p className="font-heading font-bold text-base text-[#D2410D]">
            -{totalDeducts.toFixed(0)} PCI
          </p>
        </div>
      </div>

      <div className="space-y-4 my-2">
        {items.map((item) => {
          const percentage = Math.round((item.count / maxCount) * 100);
          return (
            <div key={item.name} className="space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-medium text-[#2A1418] flex items-center gap-1.5">
                  <span className={`w-2.5 h-2.5 rounded-sm ${item.barBg}`} />
                  {item.name}
                  <span className="text-[10px] text-[#7C6A6D]">(-{item.penalty} ea)</span>
                </span>
                <span className="font-semibold text-[#2A1418]">
                  {item.count} <span className="text-[#7C6A6D] font-normal">detected</span>
                </span>
              </div>
              <div className="w-full h-2.5 bg-[#F1EAE8] rounded-full overflow-hidden p-0.5 shadow-inner">
                <div
                  className={`h-full rounded-full transition-all duration-500 ease-out ${item.barBg}`}
                  style={{ width: `${Math.max(item.count > 0 ? 8 : 0, percentage)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-[#9B4A57]/10 flex items-center justify-between text-xs text-[#7C6A6D]">
        <span>Total Detected: {items.reduce((acc, i) => acc + i.count, 0)} instances</span>
        <span className="font-medium text-[#9B4A57]">Standard ASTM D6433 Weights</span>
      </div>
    </div>
  );
};
