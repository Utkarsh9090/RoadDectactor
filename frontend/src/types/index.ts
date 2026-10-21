export interface DefectCounts {
  digs: number;
  patches: number;
  cracks: number;
  potholes: number;
}

export interface SegmentDetail {
  segment_id: number;
  frame_start: number;
  frame_end: number;
  defects: DefectCounts;
  pci_score: number;
  condition_category: 'Good' | 'Satisfactory' | 'Fair' | 'Poor' | 'Very Poor' | 'Serious' | 'Failed';
  rl_recommended_action: number;
  rl_action_name: string;
}

export interface InspectionSummary {
  overall_pci: number;
  recommended_action: string;
  total_defects: DefectCounts;
  condition_category?: string;
}

export interface InspectionReport {
  status: string;
  filename: string;
  total_segments_analyzed: number;
  summary: InspectionSummary;
  segment_details: SegmentDetail[];
  video_url?: string;
}

export interface PredictActionRequest {
  pci: number;
  digs: number;
  patches: number;
  cracks: number;
  road_age_years: number;
}

export interface PredictActionResponse {
  status: string;
  state_vector: number[];
  pci_score: number;
  condition_category: string;
  recommended_action_code: number;
  recommended_action: string;
}

export interface BackendHealth {
  status: string;
  detector_ready: boolean;
  rl_policy_ready: boolean;
  version?: string;
}

export interface StoredVideoInfo {
  id: string;
  filename: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
  resolution: string;
  fps: number;
  duration_seconds: number;
  total_frames: number;
  size_mb: number;
  stream_url: string;
}
