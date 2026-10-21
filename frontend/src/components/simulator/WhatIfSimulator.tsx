import React, { useState, useEffect } from 'react';
import { Sliders, Sparkles, Terminal } from 'lucide-react';
import { calculatePci, getConditionCategory } from '../../lib/pci';
import { predictAction } from '../../lib/api';
import { PciGauge } from '../dashboard/PciGauge';
import { ActionAlertBanner } from '../dashboard/ActionAlertBanner';

export const WhatIfSimulator: React.FC = () => {
  const [digs, setDigs] = useState<number>(1);
  const [patches, setPatches] = useState<number>(1);
  const [cracks, setCracks] = useState<number>(5);
  const [potholes, setPotholes] = useState<number>(1);
  const [roadAge, setRoadAge] = useState<number>(4.0);

  const [pci, setPci] = useState<number>(90.1);
  const [condition, setCondition] = useState<string>('Good');
  const [actionName, setActionName] = useState<string>('Good Condition (Routine Monitoring Only)');
  const [actionCode, setActionCode] = useState<number>(0);
  const [stateVector, setStateVector] = useState<number[]>([90.1, 1, 1, 5, 4.0]);

  useEffect(() => {
    const computedPci = calculatePci({ digs, patches, cracks, potholes });
    setPci(computedPci);
    setCondition(getConditionCategory(computedPci));

    const vector = [computedPci, digs, patches, cracks, roadAge];
    setStateVector(vector);

    // Call backend API or local model
    predictAction({
      pci: computedPci,
      digs,
      patches,
      cracks,
      road_age_years: roadAge,
    }).then((res) => {
      setActionCode(res.recommended_action_code);
      setActionName(res.recommended_action);
    });
  }, [digs, patches, cracks, potholes, roadAge]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-surface-static p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <span className="eyebrow-label">Simulation Lab</span>
          <h2 className="text-2xl font-heading font-bold text-[#2A1418] mt-1">
            Interactive "What-If" Defect & Policy Simulator
          </h2>
          <p className="text-sm text-[#7C6A6D] mt-0.5">
            Dynamically adjust defect occurrences and pavement service age to inspect real-time PCI deductions and Deep Q-Network actions.
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#9B4A57]/10 text-[#9B4A57] text-xs font-semibold">
          <Sparkles className="w-4 h-4" />
          <span>Real-time Inference</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Sliders Column */}
        <div className="lg:col-span-6 glass-surface-static p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-[#9B4A57]/10 pb-3">
            <h3 className="font-heading font-bold text-base text-[#2A1418] flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[#9B4A57]" />
              Pavement Distress Inputs
            </h3>
            <button
              type="button"
              onClick={() => {
                setDigs(0);
                setPatches(0);
                setCracks(0);
                setPotholes(0);
                setRoadAge(1.0);
              }}
              className="text-xs text-[#9B4A57] hover:underline font-medium"
            >
              Reset to Pristine
            </button>
          </div>

          {/* Utility Digs */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-[#2A1418]">Unsealed Utility Digs (-2.0 PCI each)</span>
              <span className="font-heading font-bold text-[#9B4A57] bg-[#9B4A57]/10 px-2 py-0.5 rounded">
                {digs} Digs
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="10"
              value={digs}
              onChange={(e) => setDigs(parseInt(e.target.value))}
              className="w-full accent-[#9B4A57] cursor-pointer"
            />
          </div>

          {/* Patches */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-[#2A1418]">Deteriorated Asphalt Patches (-1.5 PCI each)</span>
              <span className="font-heading font-bold text-[#7A3441] bg-[#7A3441]/10 px-2 py-0.5 rounded">
                {patches} Patches
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="15"
              value={patches}
              onChange={(e) => setPatches(parseInt(e.target.value))}
              className="w-full accent-[#7A3441] cursor-pointer"
            />
          </div>

          {/* Cracks */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-[#2A1418]">Surface Cracks (-0.8 PCI each)</span>
              <span className="font-heading font-bold text-[#D2410D] bg-[#D2410D]/10 px-2 py-0.5 rounded">
                {cracks} Cracks
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="30"
              value={cracks}
              onChange={(e) => setCracks(parseInt(e.target.value))}
              className="w-full accent-[#D2410D] cursor-pointer"
            />
          </div>

          {/* Potholes */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-[#2A1418]">Structural Potholes (-2.4 PCI each)</span>
              <span className="font-heading font-bold text-red-600 bg-red-50 px-2 py-0.5 rounded border border-red-200">
                {potholes} Potholes
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="15"
              value={potholes}
              onChange={(e) => setPotholes(parseInt(e.target.value))}
              className="w-full accent-red-600 cursor-pointer"
            />
          </div>

          {/* Road Age */}
          <div className="space-y-2 pt-2 border-t border-[#9B4A57]/10">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-[#2A1418]">Surface In-Service Age (Years)</span>
              <span className="font-heading font-bold text-[#2A1418] bg-white border border-[#9B4A57]/20 px-2 py-0.5 rounded">
                {roadAge.toFixed(1)} Years
              </span>
            </div>
            <input
              type="range"
              min="0.5"
              max="20.0"
              step="0.5"
              value={roadAge}
              onChange={(e) => setRoadAge(parseFloat(e.target.value))}
              className="w-full accent-[#2A1418] cursor-pointer"
            />
          </div>
        </div>

        {/* Real-Time Result Column */}
        <div className="lg:col-span-6 space-y-6">
          <PciGauge score={pci} conditionCategory={condition} />

          {/* DQN State Vector Console */}
          <div className="glass-surface-static p-5 bg-[#2A1418] text-[#FBF7F6] border border-[#9B4A57]/30 shadow-md">
            <div className="flex items-center justify-between text-xs text-[#FBF7F6]/70 mb-2 border-b border-white/10 pb-2">
              <span className="flex items-center gap-1.5 font-mono">
                <Terminal className="w-3.5 h-3.5 text-[#D2410D]" />
                Gymnasium 5-Dim Observation Vector
              </span>
              <span className="text-[10px] font-mono bg-white/10 px-2 py-0.5 rounded">
                Action Code: {actionCode}
              </span>
            </div>
            <pre className="text-xs font-mono text-emerald-400 overflow-x-auto py-1">
              {JSON.stringify(stateVector, null, 2)}
            </pre>
            <p className="text-[11px] text-white/60 mt-1 font-mono">
              [PCI, Digs, Patches, Cracks, Road Age]
            </p>
          </div>
        </div>
      </div>

      {/* Alert Directive */}
      <ActionAlertBanner
        pciScore={pci}
        recommendedAction={actionName}
        totalDefectsCount={digs + patches + cracks + potholes}
      />
    </div>
  );
};
