import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';

interface ActionAlertBannerProps {
  pciScore: number;
  recommendedAction: string;
  totalDefectsCount: number;
}

export const ActionAlertBanner: React.FC<ActionAlertBannerProps> = ({
  pciScore,
  recommendedAction,
  totalDefectsCount,
}) => {
  const isRepairImmediate = recommendedAction.includes('Immediate') || pciScore < 40;
  const isRepairIt = !isRepairImmediate && (recommendedAction.includes('Repair It') || pciScore < 60);
  const isMaintainProperly = !isRepairImmediate && !isRepairIt && (recommendedAction.includes('Maintain') || pciScore < 80);

  if (isRepairImmediate) {
    return (
      <div className="rounded-2xl p-6 bg-gradient-to-r from-rose-50/90 to-red-50/90 border border-red-200 text-red-950 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-red-100 text-red-700 shrink-0 mt-0.5">
            <AlertCircle className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="bg-red-600 text-white font-heading font-bold text-[11px] px-2.5 py-0.5 rounded-full tracking-wider uppercase">
                Emergency Directive
              </span>
              <span className="text-xs font-semibold text-red-700">PCI &lt; 40 / Severe Degradation</span>
            </div>
            <h3 className="text-lg font-heading font-bold text-red-950 mt-1">
              Repair Immediate (Emergency Resurfacing Required)
            </h3>
            <p className="text-sm text-red-800/90 mt-0.5 font-body">
              Critical pavement breakdown detected with {totalDefectsCount} defects. Deep milling, asphalt base restoration, and emergency structural resurfacing are mandatory.
            </p>
          </div>
        </div>
        <div className="shrink-0 bg-white/80 backdrop-blur-sm px-4 py-2.5 rounded-xl border border-red-200 text-right">
          <span className="text-[11px] uppercase tracking-wider text-red-700 font-semibold block">Intervention Window</span>
          <span className="font-heading font-bold text-base text-red-950">Immediate (&lt; 7 Days)</span>
        </div>
      </div>
    );
  }

  if (isRepairIt) {
    return (
      <div className="rounded-2xl p-6 bg-gradient-to-r from-orange-50/90 to-amber-50/90 border border-orange-200 text-orange-950 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-orange-100 text-orange-700 shrink-0 mt-0.5">
            <AlertTriangle className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="bg-orange-600 text-white font-heading font-bold text-[11px] px-2.5 py-0.5 rounded-full tracking-wider uppercase">
                Corrective Action
              </span>
              <span className="text-xs font-semibold text-orange-800">PCI 40–60 / Significant Distress</span>
            </div>
            <h3 className="text-lg font-heading font-bold text-orange-950 mt-1">
              Repair It (Corrective Surface Repair Required)
            </h3>
            <p className="text-sm text-orange-900/90 mt-0.5 font-body">
              Substantial distress identified with {totalDefectsCount} defects. Timely deep cavity patching and localized skin overlays are required to prevent base failure.
            </p>
          </div>
        </div>
        <div className="shrink-0 bg-white/80 backdrop-blur-sm px-4 py-2.5 rounded-xl border border-orange-200 text-right">
          <span className="text-[11px] uppercase tracking-wider text-orange-700 font-semibold block">Target Timeline</span>
          <span className="font-heading font-bold text-base text-orange-950">Within 14–30 Days</span>
        </div>
      </div>
    );
  }

  if (isMaintainProperly) {
    return (
      <div className="rounded-2xl p-6 bg-gradient-to-r from-amber-50/90 to-yellow-50/90 border border-amber-200 text-amber-950 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-amber-100 text-amber-700 shrink-0 mt-0.5">
            <AlertTriangle className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="bg-amber-600 text-white font-heading font-bold text-[11px] px-2.5 py-0.5 rounded-full tracking-wider uppercase">
                Maintenance Priority
              </span>
              <span className="text-xs font-semibold text-amber-800">PCI 60–80 / Moderate Wear</span>
            </div>
            <h3 className="text-lg font-heading font-bold text-amber-950 mt-1">
              Maintain Properly (Preventive Maintenance)
            </h3>
            <p className="text-sm text-amber-900/90 mt-0.5 font-body">
              Moderate surface distress detected with {totalDefectsCount} defects. Sealing cracks and repairing minor holes will cost-effectively protect the road strips and surface health.
            </p>
          </div>
        </div>
        <div className="shrink-0 bg-white/80 backdrop-blur-sm px-4 py-2.5 rounded-xl border border-amber-200 text-right">
          <span className="text-[11px] uppercase tracking-wider text-amber-700 font-semibold block">Scheduled Target</span>
          <span className="font-heading font-bold text-base text-amber-950">Within 30–60 Days</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl p-6 bg-gradient-to-r from-emerald-50/90 to-teal-50/90 border border-emerald-200 text-emerald-950 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div className="flex items-start gap-4">
        <div className="p-3 rounded-xl bg-emerald-100 text-emerald-700 shrink-0 mt-0.5">
          <CheckCircle className="w-6 h-6 stroke-[2.2]" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="bg-emerald-600 text-white font-heading font-bold text-[11px] px-2.5 py-0.5 rounded-full tracking-wider uppercase">
              Good Condition
            </span>
            <span className="text-xs font-semibold text-emerald-800">PCI 80–100 / Optimal Health</span>
          </div>
          <h3 className="text-lg font-heading font-bold text-emerald-950 mt-1">
            Good Condition (Routine Monitoring Only)
          </h3>
          <p className="text-sm text-emerald-900/90 mt-0.5 font-body">
            Pavement condition is optimal (PCI {pciScore.toFixed(1)}/100). Road has minimal defects ({totalDefectsCount} defects detected). No emergency capital expenditure required.
          </p>
        </div>
      </div>
      <div className="shrink-0 bg-white/80 backdrop-blur-sm px-4 py-2.5 rounded-xl border border-emerald-200 text-right">
        <span className="text-[11px] uppercase tracking-wider text-emerald-700 font-semibold block">Budget Action</span>
        <span className="font-heading font-bold text-base text-emerald-950">$0 Capital Required</span>
      </div>
    </div>
  );
};
