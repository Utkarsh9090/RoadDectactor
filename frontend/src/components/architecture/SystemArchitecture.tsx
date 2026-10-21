import React from 'react';
import { ArrowRight, Binary, Cpu, Eye, Gauge, ShieldCheck, Zap } from 'lucide-react';

export const SystemArchitecture: React.FC = () => {
  const pipelineSteps = [
    {
      step: '01',
      title: 'Video Ingestion & Frame Sampling',
      engine: 'OpenCV cv2.VideoCapture',
      desc: 'Streams input dashcam or live camera at 30 FPS. Samples every 5th frame for real-time throughput efficiency and splits video into 100-frame (100-meter) operational road chunks.',
      icon: Eye,
      accent: 'border-[#9B4A57]/30 text-[#9B4A57]',
    },
    {
      step: '02',
      title: 'Deep Defect Detection',
      engine: 'Ultralytics YOLOv8n / Custom Weights',
      desc: 'Identifies 4 critical distress classes: Utility Digs, Asphalt Patches, Structural Cracks, and Potholes with spatial centroid non-max suppression to eliminate double-counts.',
      icon: Cpu,
      accent: 'border-[#7A3441]/30 text-[#7A3441]',
    },
    {
      step: '03',
      title: 'Standard PCI Score Computation',
      engine: 'ASTM D6433 Weighted Deduct Engine',
      desc: 'Deducts severity penalties from a pristine base of 100.0: Digs (-12), Patches (-8), Cracks (-4), Potholes (-15). Classifies road into 7 ASTM urgency bands.',
      icon: Gauge,
      accent: 'border-[#D2410D]/30 text-[#D2410D]',
    },
    {
      step: '04',
      title: 'DQN Predictive Maintenance Policy',
      engine: 'Stable-Baselines3 DQN in Gymnasium',
      desc: 'Takes the 5-element state vector [PCI, Digs, Patches, Cracks, Age] and infers the cost-optimal decision: Action 0 (Do Nothing), Action 1 (Preventive Repair), or Action 2 (Major Resurface).',
      icon: Zap,
      accent: 'border-emerald-600/30 text-emerald-700',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="glass-surface-static p-6">
        <span className="eyebrow-label">System Blueprints</span>
        <h2 className="text-2xl font-heading font-bold text-[#2A1418] mt-1">
          Platform Architecture & Engineering Specifications
        </h2>
        <p className="text-sm text-[#7C6A6D] mt-0.5">
          End-to-end integration between computer vision defect extraction, civil engineering PCI metrics, and Reinforcement Learning maintenance optimization.
        </p>
      </div>

      {/* 4-Stage Architectural Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {pipelineSteps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={step.step} className="glass-surface p-5 flex flex-col justify-between relative group">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="font-heading font-bold text-xs tracking-wider text-[#7C6A6D]">
                    PHASE {step.step}
                  </span>
                  <div className={`p-2 rounded-xl bg-white border shadow-sm ${step.accent}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                </div>
                <h3 className="font-heading font-bold text-base text-[#2A1418] group-hover:text-[#9B4A57] transition-colors">
                  {step.title}
                </h3>
                <p className="text-xs font-semibold text-[#D2410D] mt-0.5 font-mono">
                  {step.engine}
                </p>
                <p className="text-xs text-[#7C6A6D] mt-2 leading-relaxed">
                  {step.desc}
                </p>
              </div>

              {idx < 3 && (
                <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10 bg-white border border-[#9B4A57]/20 rounded-full p-1 shadow-sm text-[#9B4A57]">
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Mathematical Formulations & MDP Space */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-surface-static p-6 space-y-4">
          <div className="flex items-center gap-2 text-[#9B4A57]">
            <Binary className="w-5 h-5" />
            <h3 className="font-heading font-bold text-base text-[#2A1418]">
              Mathematical Formulations (ASTM D6433)
            </h3>
          </div>
          <div className="p-4 rounded-xl bg-[#FBF7F6] border border-[#9B4A57]/15 font-mono text-xs space-y-2 text-[#2A1418]">
            <p className="text-[#9B4A57] font-semibold">// Total Deduct Value (DV)</p>
            <p className="bg-white p-2.5 rounded-lg border border-[#9B4A57]/10">
              DV = (Digs × 12.0) + (Patches × 8.0) + (Cracks × 4.0) + (Potholes × 15.0)
            </p>
            <p className="text-[#9B4A57] font-semibold">// Pavement Condition Index (PCI)</p>
            <p className="bg-white p-2.5 rounded-lg border border-[#9B4A57]/10">
              PCI = max(0.0, min(100.0, 100.0 - DV))
            </p>
          </div>
          <p className="text-xs text-[#7C6A6D]">
            The calculated score is bounded strictly between 0.0 (Failed) and 100.0 (Pristine Condition), ensuring non-negative mathematical stability for the RL observation state.
          </p>
        </div>

        <div className="glass-surface-static p-6 space-y-4">
          <div className="flex items-center gap-2 text-[#D2410D]">
            <ShieldCheck className="w-5 h-5" />
            <h3 className="font-heading font-bold text-base text-[#2A1418]">
              Gymnasium MDP Maintenance Environment
            </h3>
          </div>
          <div className="p-4 rounded-xl bg-[#FBF7F6] border border-[#9B4A57]/15 font-mono text-xs space-y-2 text-[#2A1418]">
            <p className="text-[#D2410D] font-semibold">// Observation Space (Box[5])</p>
            <p className="bg-white p-2.5 rounded-lg border border-[#9B4A57]/10">
              S = [PCI (0-100), Digs (0-50), Patches (0-50), Cracks (0-50), Age (0-30)]
            </p>
            <p className="text-[#D2410D] font-semibold">// Action Space (Discrete[3])</p>
            <p className="bg-white p-2.5 rounded-lg border border-[#9B4A57]/10">
              A = {'{0: Do Nothing, 1: Preventive Repair (-15 cost), 2: Resurface (-60 cost)}'}
            </p>
            <p className="text-[#D2410D] font-semibold">// Reward Function</p>
            <p className="bg-white p-2.5 rounded-lg border border-[#9B4A57]/10">
              R = (PCI / 10.0) - ActionCost - (50.0 if PCI &lt; 40 else 0)
            </p>
          </div>
          <p className="text-xs text-[#7C6A6D]">
            Trained for 50,000 steps using Stable-Baselines3 Deep Q-Network policy with experience replay buffer.
          </p>
        </div>
      </div>
    </div>
  );
};
