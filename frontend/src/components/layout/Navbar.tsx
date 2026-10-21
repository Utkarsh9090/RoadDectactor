import React, { useState } from 'react';
import { Activity, Layers, Menu, Sliders, Video, X } from 'lucide-react';
import type { BackendHealth } from '../../types';

interface NavbarProps {
  currentTab: 'inspection' | 'simulator' | 'architecture';
  onSelectTab: (tab: 'inspection' | 'simulator' | 'architecture') => void;
  backendHealth: BackendHealth;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onSelectTab, backendHealth }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'inspection' as const, label: 'Video Inspection', icon: Video },
    { id: 'simulator' as const, label: 'What-If Simulator', icon: Sliders },
    { id: 'architecture' as const, label: 'Architecture & Engine', icon: Layers },
  ];

  return (
    <header className="sticky top-0 z-50 glass-nav">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo & Platform Title */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => onSelectTab('inspection')}>
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-[#9B4A57] to-[#7A3441] flex items-center justify-center text-white shadow-md shadow-[#9B4A57]/20 border border-[#9B4A57]/30">
              <Activity className="w-6 h-6 stroke-[2.2]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-heading font-bold text-xl tracking-tight text-[#2A1418]">
                  RoadDetector
                </span>
                <span className="bg-[#D2410D] text-white text-[10px] font-heading font-bold px-2 py-0.5 rounded-full tracking-wider uppercase">
                  AI
                </span>
              </div>
              <p className="text-xs text-[#7C6A6D] font-body tracking-normal hidden sm:block">
                Pavement Defect & Predictive Maintenance Platform
              </p>
            </div>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-2 bg-[#FBF7F6] p-1.5 rounded-2xl border border-[#9B4A57]/10">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-white text-[#9B4A57] shadow-sm shadow-[#421418]/5 font-semibold'
                      : 'text-[#7C6A6D] hover:text-[#2A1418] hover:bg-white/60'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-[#9B4A57]' : 'text-[#7C6A6D]'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Backend Status Diagnostics Badge */}
          <div className="hidden lg:flex items-center gap-3">
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/90 border border-[#9B4A57]/15 shadow-sm text-xs font-medium">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  backendHealth.status === 'ok'
                    ? 'bg-emerald-500 animate-pulse'
                    : 'bg-amber-400'
                }`}
              />
              <span className="text-[#2A1418] font-medium">
                {backendHealth.status === 'ok' ? 'FastAPI & YOLO Online' : 'Local Fallback Engine'}
              </span>
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-xl text-[#2A1418] hover:bg-white/80 focus:outline-none"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden glass-surface mx-4 my-2 p-4 space-y-2 border border-[#9B4A57]/20 shadow-xl animate-fade-in">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onSelectTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-[#9B4A57] text-white font-semibold'
                    : 'text-[#2A1418] hover:bg-[#9B4A57]/10'
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </button>
            );
          })}
          <div className="pt-3 border-t border-[#9B4A57]/10 flex items-center justify-between text-xs text-[#7C6A6D]">
            <span>FastAPI Server Status</span>
            <span className="font-semibold text-[#9B4A57]">
              {backendHealth.status === 'ok' ? 'Connected (Port 8000)' : 'Standalone Mode'}
            </span>
          </div>
        </div>
      )}
    </header>
  );
};
