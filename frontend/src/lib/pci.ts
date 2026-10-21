import type { DefectCounts } from '../types';

export const DEDUCT_WEIGHTS = {
  dig: 2.0,
  patch: 1.5,
  crack: 0.8,
  pothole: 2.4,
};

export const RL_ACTION_NAMES: Record<number, string> = {
  0: 'Good Condition (Routine Monitoring Only)',
  1: 'Maintain Properly (Preventive Maintenance)',
  2: 'Repair It (Corrective Surface Repair)',
  3: 'Repair Immediate (Emergency Resurfacing Required)',
};

export function calculatePci(defects: DefectCounts): number {
  const deduct =
    (defects.digs || 0) * DEDUCT_WEIGHTS.dig +
    (defects.patches || 0) * DEDUCT_WEIGHTS.patch +
    (defects.cracks || 0) * DEDUCT_WEIGHTS.crack +
    (defects.potholes || 0) * DEDUCT_WEIGHTS.pothole;

  return Math.max(0.0, Math.min(100.0, parseFloat((100.0 - deduct).toFixed(1))));
}

export function getConditionCategory(pci: number): string {
  if (pci >= 80) return 'Good';
  if (pci >= 60) return 'Maintain Properly';
  if (pci >= 40) return 'Repair It';
  return 'Repair Immediate';
}

export function getRecommendedAction(pci: number, roadAge: number): { code: number; name: string } {
  if (pci < 40 || roadAge >= 12.0) {
    return { code: 3, name: RL_ACTION_NAMES[3] };
  }
  if (pci < 60 || roadAge >= 8.0) {
    return { code: 2, name: RL_ACTION_NAMES[2] };
  }
  if (pci < 80 || roadAge >= 5.0) {
    return { code: 1, name: RL_ACTION_NAMES[1] };
  }
  return { code: 0, name: RL_ACTION_NAMES[0] };
}
