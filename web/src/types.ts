export type GrainStatus = "ordinary" | "thin" | "uncertain" | "false_positive";

export interface Grain {
  id: number;
  bbox: [number, number, number, number];
  area: number;
  intergrowth_type: string;
  gray_ratio: number;
  status: GrainStatus;
  conf_ordinary: number;
  conf_thin: number;
}

export interface Counts {
  total_k: number;
  ordinary_l: number;
  thin_j: number;
  uncertain: number;
  false_positive: number;
}

export interface Metrics {
  sulfide_percent: number;
  ordinary_percent: number;
  thin_percent: number;
  talc_percent: number | null;
  talc_available: boolean;
  grain_count: number;
  ordinary_count: number;
  thin_count: number;
  uncertain_count: number;
  false_positive_count: number;
}

export interface AnalysisResult {
  result_id: string;
  mode: string;
  sort_label_ru: string;
  sort_code: string;
  conclusion: string;
  explanation: string;
  talc_percent: number | null;
  talc_available: boolean;
  sulfide_percent: number;
  ordinary_percent: number;
  thin_percent: number;
  grain_count: number;
  grains: Grain[];
  counts: Counts;
  metrics: Metrics;
  image_url: string | null;
  talc_layer_url: string | null;
  talc_display_url: string | null;
  type_layer_url: string | null;
  labels_url: string | null;
  pdf_url: string | null;
  csv_url: string | null;
  original_width: number;
  original_height: number;
}

export type LayerMode = "overview" | "talc" | "type";
