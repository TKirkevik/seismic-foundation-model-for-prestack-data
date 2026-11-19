export interface HealthStatus {
  status: string
  device: string
  device_name: string
  torch_version: string
  mps_fallback: boolean
  weights_available: boolean
}

export interface EmbeddingSetInfo {
  id: string
  label: string
  has_patch_embeddings: boolean
  has_offset_summary: boolean
  patch_grid_shape: [number, number, number]
  patch_size: number
  il_patch_start: number[]
  xl_patch_start: number[]
  t_patch_start: number[]
  inline_center: number[]
  xline_center: number[]
  time_center: number[]
  embedded_offsets: number[]
  available_modes: string[]
  available_attributes: string[]
}

export interface SurveySummary {
  id: string
  name: string
  sgz_path: string
  is_4d: boolean
  shape: [number, number, number, number] // [n_il, n_xl, n_samples, n_offsets]
  ilines: number[]
  xlines: number[]
  samples: number[]
  offsets: number[]
  default_offset_index: number
  amplitude_clip_default: number
  embedding_sets: EmbeddingSetInfo[]
  horizon_ids: string[]
}

export type HorizonKind = 'horizon' | 'top' | 'base'

export interface HorizonSurfaceInfo {
  id: string
  label: string
  kind: HorizonKind
  color: string
  grid_shape: [number, number] // [n_il, n_xl]
  twt_min: number
  twt_max: number
  coverage: number
  twt_b64: string
}

export interface WellPick {
  well: string
  formation: string
  il_idx: number
  xl_idx: number
  twt: number
  tvdss: number
  qualifier: string
}

export interface HorizonsResponse {
  survey_id: string
  horizons: HorizonSurfaceInfo[]
  well_picks: WellPick[]
  meta: Record<string, unknown>
}

export type SectionAxis = 'inline' | 'xline' | 'sample'
export type GatherLineAxis = 'xline' | 'inline'

export interface SectionResponseData {
  axis: SectionAxis
  index: number
  offsetIndex: number
  shape: [number, number]
  data: Float32Array
}

export interface GatherLocation {
  il_idx: number
  xl_idx: number
  il: number
  xl: number
}

export interface GatherPanelResponseData {
  shape: [number, number, number] // [actual_count, n_offsets, n_samples]
  locations: GatherLocation[]
  data: Float32Array
}

export interface FeaturePick {
  il_idx: number
  xl_idx: number
  sample_idx: number
  offset_idx: number
}

export interface SimilarityRequest {
  embedding_set_id: string
  il_idx: number
  xl_idx: number
  sample_idx: number
  offset_idx: number
  mode: string
  whiten: boolean
}

export interface SimilarityMatch {
  patch_idx: [number, number, number]
  il_idx: number
  xl_idx: number
  sample_idx: number
  il: number
  xl: number
  sample: number
  score: number
}

export interface TrajectoryData {
  available: boolean
  offsets: number[]
  c1: number[]
  c2: number[]
  slope: number | null
  r2: number | null
  axis_var: [number, number] | null
}

export interface SimilarityResponse {
  pick: {
    il_idx: number
    xl_idx: number
    sample_idx: number
    offset_idx: number
    patch_idx: [number, number, number]
    patch_center: { il: number; xl: number; sample: number }
    embedded_offset_used: number | null
  }
  mode: string
  whiten: boolean
  grid_shape: [number, number, number]
  patch_size: number
  il_patch_start: number[]
  xl_patch_start: number[]
  t_patch_start: number[]
  similarity_min: number
  similarity_max: number
  similarity_b64: string
  top_matches: SimilarityMatch[]
  trajectory: TrajectoryData
}

export interface AttributeResponse {
  attribute: string
  grid_shape: [number, number, number]
  patch_size: number
  il_patch_start: number[]
  xl_patch_start: number[]
  t_patch_start: number[]
  value_min: number
  value_max: number
  values_b64: string
}

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface JobInfo {
  id: string
  type?: string
  kind?: string
  title?: string
  survey_id?: string
  status: JobStatus | string
  progress?: number
  stage?: string
  message?: string
  error?: string | null
  logs?: string[]
  log_tail?: string
  created_at?: string
  updated_at?: string
}

export interface DemoEmbeddingsRequest {
  survey_id: string
  embedding_set_id?: string
}

export interface EmbedJobRequest {
  survey_id: string
  embedding_set_id?: string
  offsets?: string
  num_overlap_patches?: number
  batch_size?: number
  auto_summarize?: boolean
}

export interface SummarizeJobRequest {
  survey_id: string
  embedding_set_id?: string
  exclude_zero_offset?: boolean
}

export type ViewerType = 'section' | 'cube' | 'gather'
export type LayoutMode =
  | 'focus-section'
  | 'focus-cube'
  | 'focus-gather'
  | 'split-2h'
  | 'split-2v'
  | 'workbench-3'

export type SeismicColormapId = 'rwb' | 'grayscale' | 'bone'
export type OverlayColormapId = 'inferno' | 'turbo' | 'viridis' | 'hot'
export type OverlaySource = 'similarity' | 'attribute' | 'off'
