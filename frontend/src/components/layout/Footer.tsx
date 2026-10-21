import React from 'react';
import { Activity, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-16 border-t border-[#9B4A57]/10 bg-white/70 backdrop-blur-md py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#9B4A57] flex items-center justify-center text-white text-xs font-bold shadow-sm">
              <Activity className="w-4 h-4 stroke-[2.2]" />
            </div>
            <div>
              <p className="font-heading font-semibold text-sm text-[#2A1418]">
                RoadDetector AI Platform
              </p>
              <p className="text-xs text-[#7C6A6D]">
                Autonomous Visual Inspection & Reinforcement Learning Maintenance Scheduling
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-[#7C6A6D]">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              ASTM D6433 Standard Compliant
            </span>
            <span>•</span>
            <span>OpenCV & YOLOv8</span>
            <span>•</span>
            <span>Gymnasium DQN</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
