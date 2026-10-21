import React, { useState, useEffect } from 'react';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { VideoInputSelector } from './components/inspection/VideoInputSelector';
import { PciGauge } from './components/dashboard/PciGauge';
import { DefectBarChart } from './components/dashboard/DefectBarChart';
import { ActionAlertBanner } from './components/dashboard/ActionAlertBanner';
import { SegmentTable } from './components/dashboard/SegmentTable';
import { WhatIfSimulator } from './components/simulator/WhatIfSimulator';
import { SystemArchitecture } from './components/architecture/SystemArchitecture';
import { checkBackendHealth, analyzeVideo, analyzeStoredVideo } from './lib/api';
import type { BackendHealth, InspectionReport } from './types';
import { Activity, AlertCircle, CheckCircle2, Film } from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<'inspection' | 'simulator' | 'architecture'>('inspection');
  const [backendHealth, setBackendHealth] = useState<BackendHealth>({
    status: 'checking',
    detector_ready: false,
    rl_policy_ready: false,
  });

  const [selectedFile, setSelectedFile] = useState<File | Blob | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [capturedPhotoUrl, setCapturedPhotoUrl] = useState<string | null>(null);
  const [isStoredVideo, setIsStoredVideo] = useState<boolean>(false);

  const [roadAge, setRoadAge] = useState<number>(3.5);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [inspectionReport, setInspectionReport] = useState<InspectionReport | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Check backend health on initial mount
  useEffect(() => {
    checkBackendHealth().then((health) => setBackendHealth(health));
    const interval = setInterval(() => {
      checkBackendHealth().then((health) => setBackendHealth(health));
    }, 20000);
    return () => clearInterval(interval);
  }, []);

  const handleVideoSelected = (file: File | Blob, filename: string, previewUrl: string, isStored: boolean = false) => {
    setSelectedFile(file);
    setSelectedFileName(filename);
    setVideoPreviewUrl(previewUrl);
    setIsStoredVideo(isStored);
    setCapturedPhotoUrl(null);
    setErrorMessage(null);
  };

  const handlePhotoSelected = (imageBlob: Blob) => {
    const url = URL.createObjectURL(imageBlob);
    setCapturedPhotoUrl(url);
    setSelectedFileName(`snapshot_${Date.now()}.jpg`);
    setSelectedFile(imageBlob);
    setIsStoredVideo(false);
    setVideoPreviewUrl(null);
    setErrorMessage(null);
  };

  const handleRunInspection = async () => {
    if (!selectedFile || !selectedFileName) {
      setErrorMessage('Please upload a video or record a clip before running inspection.');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(15);
    setErrorMessage(null);

    try {
      let report: InspectionReport;
      // If it's a project-stored video, execute fast server-side analysis
      if (
        isStoredVideo ||
        selectedFileName.includes('4695859') ||
        selectedFileName.includes('16433509') ||
        selectedFileName.includes('20258332')
      ) {
        report = await analyzeStoredVideo(
          selectedFileName,
          roadAge,
          (progress) => setProcessingProgress(progress)
        );
      } else {
        report = await analyzeVideo(
          selectedFile,
          selectedFileName,
          roadAge,
          (progress) => setProcessingProgress(progress)
        );
      }
      setInspectionReport(report);
    } catch (err: any) {
      console.error('Inspection execution error:', err);
      setErrorMessage(err.message || 'An error occurred during pavement analysis.');
    } finally {
      setIsProcessing(false);
      setProcessingProgress(100);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#FBF7F6] text-[#2A1418]">
      {/* Formal Header / Navbar */}
      <Navbar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        backendHealth={backendHealth}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* TAB 1: Inspection & Analysis */}
        {currentTab === 'inspection' && (
          <div className="space-y-8 animate-fade-in">
            {/* Hero / Overview Banner */}
            <div className="glass-surface-static p-6 sm:p-8 relative overflow-hidden">
              <div className="max-w-3xl">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#9B4A57]/10 text-[#9B4A57] text-xs font-heading font-semibold uppercase tracking-wider mb-3">
                  <Activity className="w-3.5 h-3.5" />
                  <span>ASTM D6433 & Deep Q-Network Engine</span>
                </div>
                <h1 className="text-3xl sm:text-4xl font-heading font-bold text-[#2A1418] tracking-tight">
                  Intelligent Road Defect Detection & Predictive Maintenance
                </h1>
                <p className="mt-2 text-base text-[#7C6A6D] leading-relaxed">
                  Real-time video ingestion with YOLOv8 spatial defect tracking, automated Pavement Condition Index (PCI) health scoring, and Reinforcement Learning maintenance intervention scheduling.
                </p>
              </div>

              {/* Decorative Subtle Background Glow */}
              <div className="absolute right-0 top-0 -mt-12 -mr-12 w-64 h-64 rounded-full bg-gradient-to-br from-[#9B4A57]/10 to-[#D2410D]/5 blur-3xl pointer-events-none" />
            </div>

            {/* Error Message Toast */}
            {errorMessage && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-sm flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Two-Column Input & Preview Stage */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Input Selection Component */}
              <div className="lg:col-span-5">
                <VideoInputSelector
                  onVideoSelected={handleVideoSelected}
                  onPhotoSelected={handlePhotoSelected}
                  onRunInspection={handleRunInspection}
                  isProcessing={isProcessing}
                  roadAge={roadAge}
                  onRoadAgeChange={setRoadAge}
                  selectedFileName={selectedFileName}
                />
              </div>

              {/* Right Column: Video / Photo Preview & Status Card */}
              <div className="lg:col-span-7 glass-surface-static p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-[#9B4A57]/10 pb-3">
                  <div>
                    <span className="eyebrow-label">Step 2 • Inspection Stage</span>
                    <h3 className="font-heading font-bold text-base text-[#2A1418] mt-0.5">
                      Visual Ingestion Monitor
                    </h3>
                  </div>
                  {isProcessing && (
                    <span className="text-xs font-semibold text-[#D2410D] bg-[#D2410D]/10 px-2.5 py-1 rounded-full animate-pulse">
                      Processing: {processingProgress}%
                    </span>
                  )}
                </div>

                {/* Media Preview Box */}
                <div className="relative rounded-2xl overflow-hidden bg-[#2A1418] aspect-video flex items-center justify-center border border-[#9B4A57]/20 shadow-md">
                  {videoPreviewUrl ? (
                    <video
                      src={videoPreviewUrl}
                      controls
                      className="w-full h-full object-contain"
                    />
                  ) : capturedPhotoUrl ? (
                    <img
                      src={capturedPhotoUrl}
                      alt="Road surface snapshot"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-center p-8 text-white/70">
                      <Film className="w-10 h-10 mx-auto mb-2 text-[#9B4A57]" />
                      <p className="font-heading font-semibold text-sm text-white">
                        No video or camera feed loaded
                      </p>
                      <p className="text-xs text-white/60 mt-1 max-w-sm">
                        Select a file or start recording in Step 1 to preview your road segment and run YOLO inference.
                      </p>
                    </div>
                  )}

                  {/* Processing Progress Overlay */}
                  {isProcessing && (
                    <div className="absolute inset-0 bg-[#2A1418]/80 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center text-white">
                      <div className="w-12 h-12 rounded-full border-4 border-[#9B4A57] border-t-[#D2410D] animate-spin mb-4" />
                      <h4 className="font-heading font-bold text-lg text-white">
                        Analyzing Road Frames...
                      </h4>
                      <p className="text-xs text-white/70 mt-1 max-w-xs">
                        Detecting utility cuts, patches, cracks, potholes, and building state vector.
                      </p>
                      <div className="w-64 h-2 bg-white/20 rounded-full mt-4 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-[#9B4A57] to-[#D2410D] transition-all duration-300"
                          style={{ width: `${processingProgress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Diagnostics / Specification Ribbon */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
                  <div className="p-2.5 rounded-xl bg-white border border-[#9B4A57]/15">
                    <span className="text-[#7C6A6D] block text-[10px] uppercase">Sampling Rate</span>
                    <span className="font-heading font-bold text-[#2A1418]">Every 5th Frame</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-white border border-[#9B4A57]/15">
                    <span className="text-[#7C6A6D] block text-[10px] uppercase">Segment Size</span>
                    <span className="font-heading font-bold text-[#2A1418]">100 Frames (100m)</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-white border border-[#9B4A57]/15">
                    <span className="text-[#7C6A6D] block text-[10px] uppercase">Model Weights</span>
                    <span className="font-heading font-bold text-[#9B4A57]">YOLOv8 + DQN</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-white border border-[#9B4A57]/15">
                    <span className="text-[#7C6A6D] block text-[10px] uppercase">Health Standard</span>
                    <span className="font-heading font-bold text-[#D2410D]">ASTM D6433</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Inspection Results Section */}
            {inspectionReport && (
              <div className="space-y-6 pt-4 animate-fade-in">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="eyebrow-label">Inspection Outputs</span>
                    <h2 className="text-2xl font-heading font-bold text-[#2A1418] mt-1">
                      Pavement Condition & Maintenance Directive
                    </h2>
                  </div>
                  <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs px-3 py-1 rounded-full font-semibold flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Analysis Complete
                  </span>
                </div>

                {/* Directive Banner */}
                <ActionAlertBanner
                  pciScore={inspectionReport.summary.overall_pci}
                  recommendedAction={inspectionReport.summary.recommended_action}
                  totalDefectsCount={
                    inspectionReport.summary.total_defects.digs +
                    inspectionReport.summary.total_defects.patches +
                    inspectionReport.summary.total_defects.cracks +
                    inspectionReport.summary.total_defects.potholes
                  }
                />

                {/* PCI Gauge & Defect Breakdown Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-5">
                    <PciGauge
                      score={inspectionReport.summary.overall_pci}
                      conditionCategory={inspectionReport.summary.condition_category || 'Fair'}
                    />
                  </div>
                  <div className="lg:col-span-7">
                    <DefectBarChart defects={inspectionReport.summary.total_defects} />
                  </div>
                </div>

                {/* Granular Segment Breakdown Table */}
                <SegmentTable
                  segments={inspectionReport.segment_details}
                  filename={inspectionReport.filename}
                />
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Interactive What-If Simulator */}
        {currentTab === 'simulator' && (
          <div className="animate-fade-in">
            <WhatIfSimulator />
          </div>
        )}

        {/* TAB 3: System Architecture & Specifications */}
        {currentTab === 'architecture' && (
          <div className="animate-fade-in">
            <SystemArchitecture />
          </div>
        )}
      </main>

      {/* Formal Footer */}
      <Footer />
    </div>
  );
};

export default App;
