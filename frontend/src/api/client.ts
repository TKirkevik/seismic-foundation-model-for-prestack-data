import type {
  AttributeResponse,
  DemoEmbeddingsRequest,
  EmbedJobRequest,
  FocusTrajectoryResponse,
  GatherLineAxis,
  GatherLocation,
  GatherPanelResponseData,
  HealthStatus,
  HorizonsResponse,
  JobInfo,
  SectionAxis,
  SectionResponseData,
  SimilarityRequest,
  SimilarityResponse,
  SummarizeJobRequest,
  SurveySummary,
} from './types'

const slotControllers = new Map<string, AbortController>()

function getSlotSignal(slot: string): AbortSignal {
  const prev = slotControllers.get(slot)
  if (prev) {
    prev.abort()
  }
  const ctrl = new AbortController()
  slotControllers.set(slot, ctrl)
  return ctrl.signal
}

export function abortSlot(slot: string): void {
  const prev = slotControllers.get(slot)
  if (prev) {
    prev.abort()
    slotControllers.delete(slot)
  }
}

export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

export function decodeFloat32Base64(b64: string): Float32Array {
  if (!b64) return new Float32Array(0)
  const binary = atob(b64)
  const len = binary.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Float32Array(bytes.buffer, 0, Math.floor(len / 4))
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch('/api/health')
  return parseJsonOrThrow<HealthStatus>(res)
}

export async function fetchSurveys(): Promise<SurveySummary[]> {
  const res = await fetch('/api/surveys')
  return parseJsonOrThrow<SurveySummary[]>(res)
}

export async function rescanSurveys(): Promise<SurveySummary[]> {
  const res = await fetch('/api/surveys/rescan', { method: 'POST' })
  return parseJsonOrThrow<SurveySummary[]>(res)
}

export async function fetchSection(
  surveyId: string,
  axis: SectionAxis,
  index: number,
  offsetIndex: number,
  slot = `section:${axis}`,
): Promise<SectionResponseData> {
  const signal = getSlotSignal(slot)
  const params = new URLSearchParams({
    axis,
    index: String(index),
    offset_index: String(offsetIndex),
  })
  const res = await fetch(`/api/surveys/${encodeURIComponent(surveyId)}/section?${params.toString()}`, {
    signal,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  const shapeHeader = res.headers.get('X-Shape') || ''
  const parts = shapeHeader.split(',').map((s) => parseInt(s.trim(), 10))
  const buf = await res.arrayBuffer()
  const data = new Float32Array(buf)
  const shape: [number, number] =
    parts.length === 2 && !Number.isNaN(parts[0]) && !Number.isNaN(parts[1])
      ? [parts[0], parts[1]]
      : [1, data.length]

  return {
    axis,
    index,
    offsetIndex,
    shape,
    data,
  }
}

export async function fetchGatherPanel(
  surveyId: string,
  lineAxis: GatherLineAxis,
  ilStartIdx: number,
  xlStartIdx: number,
  increment: number,
  count: number,
  slot = 'gather-panel',
): Promise<GatherPanelResponseData> {
  const signal = getSlotSignal(slot)
  const params = new URLSearchParams({
    line_axis: lineAxis,
    il_start_idx: String(ilStartIdx),
    xl_start_idx: String(xlStartIdx),
    increment: String(increment),
    count: String(count),
  })
  const res = await fetch(
    `/api/surveys/${encodeURIComponent(surveyId)}/gather-panel?${params.toString()}`,
    { signal },
  )
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  const shapeHeader = res.headers.get('X-Shape') || ''
  const parts = shapeHeader.split(',').map((s) => parseInt(s.trim(), 10))
  const locationsHeader = res.headers.get('X-Locations') || '[]'
  let locations: GatherLocation[] = []
  try {
    locations = JSON.parse(locationsHeader) as GatherLocation[]
  } catch {
    locations = []
  }

  const buf = await res.arrayBuffer()
  const data = new Float32Array(buf)
  const shape: [number, number, number] =
    parts.length === 3 && parts.every((p) => !Number.isNaN(p))
      ? [parts[0], parts[1], parts[2]]
      : [locations.length || 1, 1, data.length]

  return {
    shape,
    locations,
    data,
  }
}

export async function fetchSimilarity(
  surveyId: string,
  payload: SimilarityRequest,
  slot = 'similarity',
): Promise<SimilarityResponse> {
  const signal = getSlotSignal(slot)
  const res = await fetch(`/api/surveys/${encodeURIComponent(surveyId)}/similarity`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })
  return parseJsonOrThrow<SimilarityResponse>(res)
}

export async function fetchFocusTrajectory(
  surveyId: string,
  params: {
    embedding_set_id: string
    il_idx: number
    xl_idx: number
    sample_idx: number
    offset_idx: number
  },
  slot = 'trajectory',
): Promise<FocusTrajectoryResponse> {
  const signal = getSlotSignal(slot)
  const qs = new URLSearchParams({
    embedding_set_id: params.embedding_set_id,
    il_idx: String(params.il_idx),
    xl_idx: String(params.xl_idx),
    sample_idx: String(params.sample_idx),
    offset_idx: String(params.offset_idx),
  })
  const res = await fetch(
    `/api/surveys/${encodeURIComponent(surveyId)}/trajectory?${qs.toString()}`,
    { signal },
  )
  return parseJsonOrThrow<FocusTrajectoryResponse>(res)
}

export async function fetchAttribute(
  surveyId: string,
  embeddingSetId: string,
  attribute: string,
  slot = 'attribute',
): Promise<AttributeResponse> {
  const signal = getSlotSignal(slot)
  const params = new URLSearchParams({
    embedding_set_id: embeddingSetId,
    attribute,
  })
  const res = await fetch(
    `/api/surveys/${encodeURIComponent(surveyId)}/attribute?${params.toString()}`,
    { signal },
  )
  return parseJsonOrThrow<AttributeResponse>(res)
}

export async function fetchHorizons(surveyId: string, slot = 'horizons'): Promise<HorizonsResponse> {
  const signal = getSlotSignal(slot)
  const res = await fetch(`/api/surveys/${encodeURIComponent(surveyId)}/horizons`, { signal })
  return parseJsonOrThrow<HorizonsResponse>(res)
}

export async function fetchJobs(): Promise<JobInfo[]> {
  const res = await fetch('/api/jobs')
  const data = await parseJsonOrThrow<JobInfo[] | { jobs: JobInfo[] }>(res)
  return Array.isArray(data) ? data : (data.jobs ?? [])
}

export async function startDemoEmbeddingsJob(payload: DemoEmbeddingsRequest): Promise<JobInfo> {
  const res = await fetch('/api/jobs/demo-embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      survey_id: payload.survey_id,
      ...(payload.tag !== undefined ? { tag: payload.tag } : {}),
      ...(payload.offset_stride !== undefined ? { offset_stride: payload.offset_stride } : {}),
    }),
  })
  return parseJsonOrThrow<JobInfo>(res)
}

export async function startEmbedJob(payload: EmbedJobRequest): Promise<JobInfo> {
  const res = await fetch('/api/jobs/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      survey_id: payload.survey_id,
      ...(payload.tag !== undefined ? { tag: payload.tag } : {}),
      ...(payload.offsets !== undefined ? { offsets: payload.offsets } : {}),
      ...(payload.num_overlap_patches !== undefined
        ? { num_overlap_patches: payload.num_overlap_patches }
        : {}),
      ...(payload.batch_size !== undefined ? { batch_size: payload.batch_size } : {}),
      ...(payload.auto_summarize !== undefined ? { auto_summarize: payload.auto_summarize } : {}),
    }),
  })
  return parseJsonOrThrow<JobInfo>(res)
}

export async function startSummarizeJob(payload: SummarizeJobRequest): Promise<JobInfo> {
  const res = await fetch('/api/jobs/summarize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      survey_id: payload.survey_id,
      ...(payload.tag !== undefined ? { tag: payload.tag } : {}),
      ...(payload.exclude_zero_offset !== undefined
        ? { exclude_zero_offset: payload.exclude_zero_offset }
        : {}),
    }),
  })
  return parseJsonOrThrow<JobInfo>(res)
}

export async function cancelJob(jobId: string): Promise<JobInfo> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  })
  return parseJsonOrThrow<JobInfo>(res)
}
