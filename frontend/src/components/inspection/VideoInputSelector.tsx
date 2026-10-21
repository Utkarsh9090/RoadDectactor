import React, { useState, useRef, useEffect } from 'react';
import { Camera, CheckCircle2, Film, RefreshCw, StopCircle, Upload, Video, Play, Sparkles } from 'lucide-react';
import { fetchStoredVideos } from '../../lib/api';
import type { StoredVideoInfo } from '../../types';

interface VideoInputSelectorProps {
  onVideoSelected: (file: File | Blob, filename: string, previewUrl: string, isStored?: boolean) => void;
  onPhotoSelected: (imageBlob: Blob) => void;
  onRunInspection: () => void;
  isProcessing: boolean;
  roadAge: number;
  onRoadAgeChange: (age: number) => void;
  selectedFileName: string | null;
}

export const VideoInputSelector: React.FC<VideoInputSelectorProps> = ({
  onVideoSelected,
  onPhotoSelected,
  onRunInspection,
  isProcessing,
  roadAge,
  onRoadAgeChange,
  selectedFileName,
}) => {
  const [sourceMode, setSourceMode] = useState<'upload' | 'record' | 'snapshot' | 'sample'>('upload');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [storedVideos, setStoredVideos] = useState<StoredVideoInfo[]>([]);
  const [isLoadingVideos, setIsLoadingVideos] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoStreamRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);

  // Fetch stored videos catalog on component mount
  useEffect(() => {
    setIsLoadingVideos(true);
    fetchStoredVideos().then((videos) => {
      setStoredVideos(videos);
      setIsLoadingVideos(false);
    });
  }, []);

  // Stop camera when unmounting or switching modes
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const handleStartCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      mediaStreamRef.current = stream;
      if (videoStreamRef.current) {
        videoStreamRef.current.srcObject = stream;
        videoStreamRef.current.play();
      }
      setIsCameraActive(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraError('Unable to access camera. Please allow camera permissions in your browser.');
    }
  };

  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setIsCameraActive(false);
    setIsRecording(false);
    setRecordingSeconds(0);
  };

  const handleStartRecording = () => {
    if (!mediaStreamRef.current) return;
    recordedChunksRef.current = [];
    setRecordingSeconds(0);

    try {
      const mimeTypes = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'];
      let selectedMime = 'video/webm';
      for (const m of mimeTypes) {
        if (MediaRecorder.isTypeSupported(m)) {
          selectedMime = m;
          break;
        }
      }

      const recorder = new MediaRecorder(mediaStreamRef.current, { mimeType: selectedMime });
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: selectedMime });
        const url = URL.createObjectURL(blob);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(11, 19);
        const name = `camera_recording_${timestamp}.webm`;
        onVideoSelected(blob, name, url);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setIsRecording(true);

      timerIntervalRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Failed to start MediaRecorder:', err);
      setCameraError('Failed to initialize video recording.');
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setIsRecording(false);
  };

  const handleCaptureSnapshot = () => {
    if (!videoStreamRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoStreamRef.current.videoWidth || 640;
    canvas.height = videoStreamRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoStreamRef.current, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          onPhotoSelected(blob);
        }
      }, 'image/jpeg', 0.95);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      onVideoSelected(file, file.name, url);
    }
  };

  const handleSelectStoredVideo = (video: StoredVideoInfo) => {
    const fakeBlob = new Blob([video.filename], { type: 'video/mp4' });
    onVideoSelected(fakeBlob, video.filename, video.stream_url, true);
  };

  return (
    <div className="glass-surface-static p-6 space-y-6">
      {/* Eyebrow and Section Header */}
      <div>
        <span className="eyebrow-label">Step 1 • Input Selection</span>
        <h2 className="text-xl font-heading font-bold text-[#2A1418] mt-1">
          Source Video & Road Parameters
        </h2>
        <p className="text-sm text-[#7C6A6D] mt-0.5">
          Upload an inspection video from your computer or record live via camera.
        </p>
      </div>

      {/* Mode Navigation Tabs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-1.5 rounded-xl bg-[#FBF7F6] border border-[#9B4A57]/15">
        <button
          type="button"
          onClick={() => {
            stopCamera();
            setSourceMode('upload');
          }}
          className={`flex flex-col items-center justify-center p-2.5 rounded-lg text-xs font-medium transition-all ${
            sourceMode === 'upload'
              ? 'bg-white text-[#9B4A57] shadow-sm font-semibold border border-[#9B4A57]/20'
              : 'text-[#7C6A6D] hover:text-[#2A1418]'
          }`}
        >
          <Upload className="w-4 h-4 mb-1" />
          <span>Upload File</span>
        </button>

        <button
          type="button"
          onClick={() => {
            setSourceMode('record');
            handleStartCamera();
          }}
          className={`flex flex-col items-center justify-center p-2.5 rounded-lg text-xs font-medium transition-all ${
            sourceMode === 'record'
              ? 'bg-white text-[#9B4A57] shadow-sm font-semibold border border-[#9B4A57]/20'
              : 'text-[#7C6A6D] hover:text-[#2A1418]'
          }`}
        >
          <Video className="w-4 h-4 mb-1" />
          <span>Record Live</span>
        </button>

        <button
          type="button"
          onClick={() => {
            setSourceMode('snapshot');
            handleStartCamera();
          }}
          className={`flex flex-col items-center justify-center p-2.5 rounded-lg text-xs font-medium transition-all ${
            sourceMode === 'snapshot'
              ? 'bg-white text-[#9B4A57] shadow-sm font-semibold border border-[#9B4A57]/20'
              : 'text-[#7C6A6D] hover:text-[#2A1418]'
          }`}
        >
          <Camera className="w-4 h-4 mb-1" />
          <span>Snap Photo</span>
        </button>

        <button
          type="button"
          onClick={() => {
            stopCamera();
            setSourceMode('sample');
            if (storedVideos.length > 0 && !selectedFileName) {
              handleSelectStoredVideo(storedVideos[0]);
            }
          }}
          className={`flex flex-col items-center justify-center p-2.5 rounded-lg text-xs font-medium transition-all ${
            sourceMode === 'sample'
              ? 'bg-white text-[#9B4A57] shadow-sm font-semibold border border-[#9B4A57]/20'
              : 'text-[#7C6A6D] hover:text-[#2A1418]'
          }`}
        >
          <Film className="w-4 h-4 mb-1" />
          <span>Real Road Videos</span>
        </button>
      </div>

      {/* Mode A: File Upload */}
      {sourceMode === 'upload' && (
        <div className="space-y-4">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
            className="hidden"
          />
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-[#9B4A57]/25 hover:border-[#9B4A57] rounded-2xl p-7 text-center cursor-pointer bg-white/60 hover:bg-white transition-all group"
          >
            <div className="w-12 h-12 rounded-full bg-[#9B4A57]/10 text-[#9B4A57] flex items-center justify-center mx-auto mb-3 group-hover:scale-105 transition-transform">
              <Upload className="w-6 h-6" />
            </div>
            <p className="font-heading font-semibold text-[#2A1418] text-sm">
              Click to upload inspection video
            </p>
            <p className="text-xs text-[#7C6A6D] mt-1">
              Supports MP4, MOV, AVI (H.264 / Dashcam 30-60 FPS)
            </p>
          </div>
        </div>
      )}

      {/* Mode B: Real-Time Live Camera Recording */}
      {sourceMode === 'record' && (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-[#2A1418] aspect-video flex items-center justify-center border border-[#9B4A57]/20 shadow-inner">
            <video
              ref={videoStreamRef}
              playsInline
              muted
              autoPlay
              className="w-full h-full object-cover"
            />

            {!isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-white/80 p-4 text-center">
                <Video className="w-8 h-8 mb-2 text-[#9B4A57]" />
                <p className="text-sm">Camera inactive</p>
                <button
                  type="button"
                  onClick={handleStartCamera}
                  className="mt-3 btn-outline text-xs bg-white/20 text-white border-white/30"
                >
                  Enable Camera
                </button>
              </div>
            )}

            {isRecording && (
              <div className="absolute top-4 left-4 bg-red-600/90 text-white text-xs px-3 py-1.5 rounded-full font-semibold flex items-center gap-2 shadow-lg animate-pulse">
                <span className="w-2.5 h-2.5 rounded-full bg-white animate-ping" />
                <span>REC {String(Math.floor(recordingSeconds / 60)).padStart(2, '0')}:{String(recordingSeconds % 60).padStart(2, '0')}</span>
              </div>
            )}
          </div>

          {cameraError && (
            <p className="text-xs text-red-600 bg-red-50 p-2.5 rounded-lg border border-red-200">
              {cameraError}
            </p>
          )}

          <div className="flex items-center gap-3">
            {!isRecording ? (
              <button
                type="button"
                onClick={handleStartRecording}
                disabled={!isCameraActive}
                className="btn-primary w-full text-sm font-medium py-3 disabled:opacity-50"
              >
                <span className="w-3 h-3 rounded-full bg-white" />
                Start Real-Time Recording
              </button>
            ) : (
              <button
                type="button"
                onClick={handleStopRecording}
                className="w-full bg-[#7A3441] hover:bg-[#2A1418] text-white font-medium py-3 rounded-xl flex items-center justify-center gap-2 transition-all shadow-md text-sm"
              >
                <StopCircle className="w-5 h-5 text-red-400" />
                Stop & Save Video Clip
              </button>
            )}
          </div>
        </div>
      )}

      {/* Mode C: Snap Photo Snapshot */}
      {sourceMode === 'snapshot' && (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-[#2A1418] aspect-video flex items-center justify-center border border-[#9B4A57]/20 shadow-inner">
            <video
              ref={videoStreamRef}
              playsInline
              muted
              autoPlay
              className="w-full h-full object-cover"
            />
          </div>

          <button
            type="button"
            onClick={handleCaptureSnapshot}
            disabled={!isCameraActive}
            className="btn-secondary w-full text-sm font-medium py-3 disabled:opacity-50"
          >
            <Camera className="w-5 h-5" />
            Capture Road Surface Photo
          </button>
        </div>
      )}

      {/* Mode D: Real Project Videos Catalog */}
      {sourceMode === 'sample' && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-[#7C6A6D] px-1">
            <span className="font-semibold text-[#2A1418] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#D2410D]" />
              Project Real Highway & Urban Datasets
            </span>
            <span>{storedVideos.length} Available</span>
          </div>

          {isLoadingVideos ? (
            <div className="p-6 text-center text-[#7C6A6D] text-xs">
              <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-[#9B4A57]" />
              <span>Loading video catalog...</span>
            </div>
          ) : (
            <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
              {storedVideos.map((video) => {
                const isSelected = selectedFileName === video.filename;
                return (
                  <div
                    key={video.id}
                    onClick={() => handleSelectStoredVideo(video)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer text-left ${
                      isSelected
                        ? 'bg-white border-[#9B4A57] ring-2 ring-[#9B4A57]/20 shadow-md'
                        : 'bg-white/70 hover:bg-white border-[#9B4A57]/15 hover:border-[#9B4A57]/40 shadow-sm'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-heading font-semibold text-xs text-[#2A1418]">
                            {video.title}
                          </span>
                          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-[#9B4A57]/10 text-[#9B4A57]">
                            {video.category}
                          </span>
                        </div>
                        <p className="text-[11px] text-[#7C6A6D] mt-1 leading-snug">
                          {video.description}
                        </p>
                      </div>
                      <div className="shrink-0 mt-0.5">
                        {isSelected ? (
                          <span className="flex items-center gap-1 text-[11px] font-semibold text-[#9B4A57] bg-[#9B4A57]/10 px-2 py-0.5 rounded">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            Active
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-[11px] font-medium text-[#7C6A6D] hover:text-[#9B4A57] px-2 py-0.5 rounded bg-gray-50 border border-gray-200">
                            <Play className="w-3 h-3" />
                            Select
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Metadata Specs Bar */}
                    <div className="flex flex-wrap items-center gap-2 mt-2.5 pt-2 border-t border-[#9B4A57]/10 text-[10px] text-[#7C6A6D]">
                      <span className="font-mono bg-[#FBF7F6] px-1.5 py-0.5 rounded border border-[#9B4A57]/10">
                        {video.resolution}
                      </span>
                      <span className="font-mono bg-[#FBF7F6] px-1.5 py-0.5 rounded border border-[#9B4A57]/10">
                        {video.fps} FPS
                      </span>
                      <span className="font-mono bg-[#FBF7F6] px-1.5 py-0.5 rounded border border-[#9B4A57]/10">
                        {video.duration_seconds}s ({video.total_frames} frames)
                      </span>
                      <span className="font-mono bg-[#FBF7F6] px-1.5 py-0.5 rounded border border-[#9B4A57]/10">
                        {video.size_mb} MB
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Active File Status Notification */}
      {selectedFileName && (
        <div className="flex items-center justify-between p-3.5 rounded-xl bg-white border border-[#9B4A57]/20 shadow-sm text-xs">
          <div className="flex items-center gap-2.5 text-[#2A1418]">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <div className="truncate max-w-[200px] sm:max-w-[280px]">
              <span className="font-semibold text-[#2A1418]">{selectedFileName}</span>
              <span className="text-[#7C6A6D] ml-1.5">ready for inspection</span>
            </div>
          </div>
          <span className="bg-[#9B4A57]/10 text-[#9B4A57] font-semibold px-2 py-0.5 rounded text-[11px]">
            Selected
          </span>
        </div>
      )}

      {/* Road Physical Age Slider */}
      <div className="pt-2 border-t border-[#9B4A57]/10 space-y-2">
        <div className="flex justify-between items-center text-xs">
          <label htmlFor="roadAgeSlider" className="font-semibold text-[#2A1418]">
            Estimated Road Surface Age
          </label>
          <span className="text-xs font-heading font-bold text-[#9B4A57] bg-[#9B4A57]/10 px-2 py-0.5 rounded-md">
            {roadAge.toFixed(1)} Years
          </span>
        </div>
        <input
          id="roadAgeSlider"
          type="range"
          min="0.0"
          max="15.0"
          step="0.5"
          value={roadAge}
          onChange={(e) => onRoadAgeChange(parseFloat(e.target.value))}
          className="w-full accent-[#9B4A57] cursor-pointer"
        />
        <div className="flex justify-between text-[11px] text-[#7C6A6D]">
          <span>New (0 yrs)</span>
          <span>Mid-Life (5-7 yrs)</span>
          <span>Old (15 yrs)</span>
        </div>
      </div>

      {/* Primary Action Button (Blood Orange Accent #D2410D) */}
      <button
        type="button"
        onClick={onRunInspection}
        disabled={isProcessing || !selectedFileName}
        className="w-full btn-primary text-base py-3.5 shadow-lg shadow-[#D2410D]/25 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
      >
        {isProcessing ? (
          <>
            <RefreshCw className="w-5 h-5 animate-spin" />
            <span>Analyzing Pavement via YOLO & RL...</span>
          </>
        ) : (
          <>
            <span>Run AI Road Inspection</span>
            <span className="text-white/70">→</span>
          </>
        )}
      </button>
    </div>
  );
};
