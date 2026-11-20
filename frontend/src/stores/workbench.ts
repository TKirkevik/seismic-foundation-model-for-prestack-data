import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'
import {
  abortSlot,
  cancelJob,
  decodeFloat32Base64,
  fetchAttribute,
  fetchFocusTrajectory,
  fetchGatherPanel,
  fetchHealth,
  fetchHorizons,
  fetchJobs,
  fetchSection,
  fetchSimilarity,
  fetchSurveys,
  isAbortError,
  rescanSurveys,
  startDemoEmbeddingsJob,
  startEmbedJob,
  startSummarizeJob,
} from '../api/client'
import type {
  AttributeResponse,
  DemoEmbeddingsRequest,
  EmbedJobRequest,
  EmbeddingSetInfo,
  FeatureGeometryMode,
  FeaturePlaneAxis,
  FeatureRole,
  FocusTrajectoryResponse,
  GatherLineAxis,
  GatherPanelResponseData,
  HealthStatus,
  HorizonKind,
  JobInfo,
  LayoutMode,
  OverlayColormapId,
  OverlaySource,
  PickCoord,
  PickedFeature,
  PickedFeaturePayload,
  SectionAxis,
  SectionResponseData,
  SeismicColormapId,
  SimilarityResponse,
  SummarizeJobRequest,
  SurveySummary,
  ViewerType,
  WellPick,
} from '../api/types'

export interface ActiveOverlayVolume {
  kind: 'similarity' | 'attribute'
  label: string
  data: Float32Array
  gridShape: [number, number, number]
  patchSize: number
  ilPatchStart: number[]
  xlPatchStart: number[]
  tPatchStart: number[]
  min: number
  max: number
  threshold: number
}

export interface HorizonGrid {
  id: string
  label: string
  kind: HorizonKind
  color: string
  twt: Float32Array // (n_il, n_xl) ms, NaN where undefined
  twtMin: number
  twtMax: number
  coverage: number
}

export interface ReservoirEnvelope {
  label: string
  color: string
  top: Float32Array
  base: Float32Array
}

const MAX_SECTION_CACHE = 48
const FEATURES_STORAGE_PREFIX = 'prestack-workbench:features:v1:'

function generateFeatureId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `feat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

export const useWorkbenchStore = defineStore('workbench', () => {
  // System & surveys
  const health = ref<HealthStatus | null>(null)
  const surveys = ref<SurveySummary[]>([])
  const activeSurveyId = ref<string | null>(null)
  const loadingSurveys = ref(false)
  const apiError = ref<string | null>(null)

  // Synchronized 4D cursor
  const ilIdx = ref(0)
  const xlIdx = ref(0)
  const sampleIdx = ref(0)
  const offsetIndex = ref(0)

  // Display settings
  const seismicColormap = ref<SeismicColormapId>('rwb')
  const overlayColormap = ref<OverlayColormapId>('inferno')
  const globalGain = ref(1.0)
  const overlayInterp = ref<'smooth' | 'patch'>('smooth')
  const overlayOpacity = ref(0.72)

  // Layout & UI state
  const layoutMode = ref<LayoutMode>('workbench-3')
  const paneViews = ref<{ a: ViewerType; b: ViewerType; c: ViewerType }>({
    a: 'section',
    b: 'cube',
    c: 'gather',
  })
  const maximizedPane = ref<'a' | 'b' | 'c' | null>(null)
  const sidebarCollapsed = ref(false)
  const jobsModalOpen = ref(false)

  // Section slices & cache
  const sectionCache = new Map<string, SectionResponseData>()
  const inFlightSections = new Map<SectionAxis, string>()
  const sections = ref<Record<SectionAxis, SectionResponseData | null>>({
    inline: null,
    xline: null,
    sample: null,
  })
  const sectionLoading = ref<Record<SectionAxis, boolean>>({
    inline: false,
    xline: false,
    sample: false,
  })

  // Gather panel
  const gatherLineAxis = ref<GatherLineAxis>('xline')
  const gatherIlStartIdx = ref(0)
  const gatherXlStartIdx = ref(0)
  const gatherIncrement = ref(10)
  const gatherCount = ref(8)
  const gatherWiggle = ref(false)
  const gatherHighlightActiveOffset = ref(true)
  const gatherSyncOffsetOnPick = ref(true)
  const gatherPanelData = ref<GatherPanelResponseData | null>(null)
  const gatherLoading = ref(false)

  // Latent exploration & Multi-Pick Feature Picking
  const activeEmbeddingSetId = ref<string | null>(null)
  const overlaySource = ref<OverlaySource>('similarity')
  const similarityMode = ref<string>('active_offset')
  const whitenChannels = ref(false)
  const similarityThreshold = ref(0.55)
  const pickMode = ref(true)
  const activeGeometryMode = ref<FeatureGeometryMode>('single')
  const activeFeatureRole = ref<FeatureRole>('include')
  const exclusionWeight = ref(0.75)

  const pickedFeatures = ref<PickedFeature[]>([])
  const selectedFeatureId = ref<string | null>(null)
  const activeDrawingFeatureId = ref<string | null>(null)
  const focusPick = ref<PickCoord | null>(null)
  const focusTrajectory = ref<FocusTrajectoryResponse | null>(null)
  const hoveredPickRef = ref<{ featureId: string; pickIndex: number } | null>(null)

  const similarityResult = ref<SimilarityResponse | null>(null)
  const similarityVolume = ref<Float32Array | null>(null)
  const similarityLoading = ref(false)
  const similarityDirty = ref(false)

  let similarityReqSeq = 0
  let moveThrottleTimer: number | null = null
  let weightThrottleTimer: number | null = null
  let trajectoryDebounceTimer: number | null = null

  const selectedAttribute = ref<string>('')
  const attributeThreshold = ref(0)
  const attributeResult = ref<AttributeResponse | null>(null)
  const attributeVolume = ref<Float32Array | null>(null)
  const attributeLoading = ref(false)

  // Ground truth (interpreted horizons + reservoir model converted to time)
  const horizons = ref<HorizonGrid[]>([])
  const wellPicks = ref<WellPick[]>([])
  const horizonsMeta = ref<Record<string, unknown>>({})
  const horizonsLoading = ref(false)
  const showReservoir = ref(false)
  const reservoirShowModel = ref(true)
  const reservoirShowHorizons = ref(true)
  const reservoirShowWells = ref(true)
  const reservoirOpacity = ref(0.35)

  // Jobs orchestrator
  const jobs = ref<JobInfo[]>([])
  const jobsLoading = ref(false)
  let pollTimer: number | null = null
  const previouslyRunningIds = new Set<string>()

  // Computed helpers
  const activeSurvey = computed<SurveySummary | null>(() => {
    if (!surveys.value.length) return null
    return surveys.value.find((s) => s.id === activeSurveyId.value) ?? surveys.value[0]
  })

  const activeEmbeddingSet = computed<EmbeddingSetInfo | null>(() => {
    const survey = activeSurvey.value
    if (!survey || !survey.embedding_sets.length) return null
    return (
      survey.embedding_sets.find((e) => e.id === activeEmbeddingSetId.value) ??
      survey.embedding_sets[0]
    )
  })

  const effectiveClipAmplitude = computed<number>(() => {
    const base = activeSurvey.value?.amplitude_clip_default || 1.0
    const gain = Math.max(0.05, globalGain.value)
    return base / gain
  })

  const currentInlineLabel = computed(() => activeSurvey.value?.ilines[ilIdx.value] ?? ilIdx.value)
  const currentXlineLabel = computed(() => activeSurvey.value?.xlines[xlIdx.value] ?? xlIdx.value)
  const currentSampleLabel = computed(() => activeSurvey.value?.samples[sampleIdx.value] ?? sampleIdx.value)
  const currentOffsetLabel = computed(
    () => activeSurvey.value?.offsets[offsetIndex.value] ?? offsetIndex.value,
  )

  const activeOverlay = computed<ActiveOverlayVolume | null>(() => {
    if (overlaySource.value === 'off') return null
    if (
      overlaySource.value === 'similarity' &&
      similarityResult.value &&
      similarityVolume.value &&
      similarityVolume.value.length > 0
    ) {
      return {
        kind: 'similarity',
        label: `Similarity (${similarityResult.value.mode})`,
        data: similarityVolume.value,
        gridShape: similarityResult.value.grid_shape,
        patchSize: similarityResult.value.patch_size,
        ilPatchStart: similarityResult.value.il_patch_start,
        xlPatchStart: similarityResult.value.xl_patch_start,
        tPatchStart: similarityResult.value.t_patch_start,
        min: similarityResult.value.similarity_min,
        max: similarityResult.value.similarity_max,
        threshold: similarityThreshold.value,
      }
    }
    if (
      overlaySource.value === 'attribute' &&
      attributeResult.value &&
      attributeVolume.value &&
      attributeVolume.value.length > 0
    ) {
      return {
        kind: 'attribute',
        label: attributeResult.value.attribute,
        data: attributeVolume.value,
        gridShape: attributeResult.value.grid_shape,
        patchSize: attributeResult.value.patch_size,
        ilPatchStart: attributeResult.value.il_patch_start,
        xlPatchStart: attributeResult.value.xl_patch_start,
        tPatchStart: attributeResult.value.t_patch_start,
        min: attributeResult.value.value_min,
        max: attributeResult.value.value_max,
        threshold: attributeThreshold.value,
      }
    }
    return null
  })

  const hasActiveJobs = computed(() =>
    jobs.value.some((j) => j.status === 'queued' || j.status === 'running'),
  )

  const hasHorizons = computed(() => (activeSurvey.value?.horizon_ids.length ?? 0) > 0)

  const sampleOrigin = computed(() => activeSurvey.value?.samples[0] ?? 0)
  const sampleRate = computed(() => {
    const s = activeSurvey.value?.samples
    return s && s.length > 1 ? s[1] - s[0] : 1
  })

  /** Fractional sample index of a two-way time in ms. */
  function twtToSample(twt: number): number {
    return (twt - sampleOrigin.value) / sampleRate.value
  }

  const reservoirEnvelope = computed<ReservoirEnvelope | null>(() => {
    if (!showReservoir.value || !reservoirShowModel.value) return null
    const top = horizons.value.find((h) => h.kind === 'top')
    const base = horizons.value.find((h) => h.kind === 'base')
    if (!top || !base) return null
    return { label: top.label, color: top.color, top: top.twt, base: base.twt }
  })

  const interpretedHorizons = computed<HorizonGrid[]>(() => {
    if (!showReservoir.value || !reservoirShowHorizons.value) return []
    return horizons.value.filter((h) => h.kind === 'horizon')
  })

  const visibleWellPicks = computed<WellPick[]>(() => {
    if (!showReservoir.value || !reservoirShowWells.value) return []
    return wellPicks.value.filter((p) => p.formation.startsWith('Hugin'))
  })

  const activeDrawingFeature = computed<PickedFeature | null>(() => {
    if (!activeDrawingFeatureId.value) return null
    return (
      pickedFeatures.value.find(
        (f) => f.id === activeDrawingFeatureId.value && f.status === 'in_progress',
      ) ?? null
    )
  })

  const enabledIncludeFeatures = computed<PickedFeature[]>(() =>
    pickedFeatures.value.filter(
      (f) => f.enabled && f.role === 'include' && f.picks.length >= 1,
    ),
  )

  const enabledExcludeFeatures = computed<PickedFeature[]>(() =>
    pickedFeatures.value.filter(
      (f) => f.enabled && f.role === 'exclude' && f.picks.length >= 1,
    ),
  )

  const hasOnlyExcludeFeatures = computed(
    () => enabledIncludeFeatures.value.length === 0 && enabledExcludeFeatures.value.length > 0,
  )

  const hasQueryableFeatures = computed(() => enabledIncludeFeatures.value.length > 0)

  function getAxisIndex(axis: SectionAxis): number {
    if (axis === 'inline') return ilIdx.value
    if (axis === 'xline') return xlIdx.value
    return sampleIdx.value
  }

  function clampIndex(val: number, maxExclusive: number): number {
    if (!Number.isFinite(val)) return 0
    if (!maxExclusive || !Number.isFinite(maxExclusive) || maxExclusive <= 0) return 0
    return Math.max(0, Math.min(maxExclusive - 1, Math.round(val)))
  }

  function applyEmbeddingSetDefaults(embSet: EmbeddingSetInfo | null) {
    if (!embSet) {
      activeEmbeddingSetId.value = null
      return
    }
    activeEmbeddingSetId.value = embSet.id
    if (embSet.available_modes.length > 0) {
      if (!embSet.available_modes.includes(similarityMode.value)) {
        similarityMode.value = embSet.available_modes.includes('active_offset')
          ? 'active_offset'
          : embSet.available_modes[0]
      }
    }
    if (embSet.available_attributes.length > 0) {
      if (!embSet.available_attributes.includes(selectedAttribute.value)) {
        selectedAttribute.value = embSet.available_attributes[0]
      }
    } else {
      selectedAttribute.value = ''
    }
  }

  function clearPendingFeatureTimers() {
    if (moveThrottleTimer !== null) {
      window.clearTimeout(moveThrottleTimer)
      moveThrottleTimer = null
    }
    if (weightThrottleTimer !== null) {
      window.clearTimeout(weightThrottleTimer)
      weightThrottleTimer = null
    }
    if (trajectoryDebounceTimer !== null) {
      window.clearTimeout(trajectoryDebounceTimer)
      trajectoryDebounceTimer = null
    }
  }

  function clearSimilarityState() {
    similarityReqSeq++
    clearPendingFeatureTimers()
    abortSlot('similarity')
    similarityLoading.value = false
    similarityResult.value = null
    similarityVolume.value = null
  }

  function saveFeaturesToStorage() {
    const sid = activeSurveyId.value
    if (!sid) return
    try {
      const payload = {
        features: pickedFeatures.value,
        exclusionWeight: exclusionWeight.value,
      }
      localStorage.setItem(`${FEATURES_STORAGE_PREFIX}${sid}`, JSON.stringify(payload))
    } catch {
      // ignore storage quota/availability errors
    }
  }

  function loadFeaturesFromStorage(survey: SurveySummary) {
    const [nIl, nXl, nSamples] = survey.shape
    pickedFeatures.value = []
    selectedFeatureId.value = null
    activeDrawingFeatureId.value = null
    hoveredPickRef.value = null
    focusPick.value = null
    focusTrajectory.value = null
    exclusionWeight.value = 0.75

    try {
      const raw = localStorage.getItem(`${FEATURES_STORAGE_PREFIX}${survey.id}`)
      if (!raw) return
      const parsed = JSON.parse(raw) as {
        features?: unknown[]
        exclusionWeight?: unknown
      }
      if (
        typeof parsed.exclusionWeight === 'number' &&
        Number.isFinite(parsed.exclusionWeight)
      ) {
        exclusionWeight.value = Math.max(0, Math.min(2, parsed.exclusionWeight))
      }
      if (!Array.isArray(parsed.features)) return

      const loaded: PickedFeature[] = []
      for (const item of parsed.features) {
        if (!item || typeof item !== 'object') continue
        const f = item as Partial<PickedFeature>
        const planeAxis: FeaturePlaneAxis = f.plane_axis === 'xline' ? 'xline' : 'inline'
        const role: FeatureRole = f.role === 'exclude' ? 'exclude' : 'include'
        const geom: FeatureGeometryMode =
          f.geometry_mode === 'points' ||
          f.geometry_mode === 'polyline' ||
          f.geometry_mode === 'polygon'
            ? f.geometry_mode
            : 'single'
        const planeMax = planeAxis === 'inline' ? nIl : nXl
        const planeIndex = clampIndex(Number(f.plane_index ?? 0), planeMax)
        const rawPicks = Array.isArray(f.picks) ? f.picks : []
        const sanitizedPicks: PickCoord[] = []
        for (const p of rawPicks) {
          if (!p || typeof p !== 'object') continue
          const pc = p as Partial<PickCoord>
          sanitizedPicks.push({
            il_idx:
              planeAxis === 'inline'
                ? planeIndex
                : clampIndex(Number(pc.il_idx ?? 0), nIl),
            xl_idx:
              planeAxis === 'xline'
                ? planeIndex
                : clampIndex(Number(pc.xl_idx ?? 0), nXl),
            sample_idx: clampIndex(Number(pc.sample_idx ?? 0), nSamples),
          })
        }
        if (sanitizedPicks.length === 0) continue

        loaded.push({
          id: typeof f.id === 'string' && f.id ? f.id : generateFeatureId(),
          name:
            typeof f.name === 'string' && f.name
              ? f.name
              : buildFeatureName(role, geom, loaded.length + 1),
          role,
          plane_axis: planeAxis,
          plane_index: planeIndex,
          geometry_mode: geom,
          picks: geom === 'single' ? sanitizedPicks.slice(0, 1) : sanitizedPicks,
          enabled: typeof f.enabled === 'boolean' ? f.enabled : true,
          status: 'completed',
          createdAt: typeof f.createdAt === 'number' ? f.createdAt : Date.now(),
        })
      }

      pickedFeatures.value = loaded
      const firstEnabled = loaded.find((f) => f.enabled && f.picks.length > 0) ?? loaded[0]
      if (firstEnabled) {
        selectedFeatureId.value = firstEnabled.id
        focusPick.value = { ...firstEnabled.picks[firstEnabled.picks.length - 1] }
      }
    } catch {
      pickedFeatures.value = []
    }
  }

  function buildFeatureName(
    role: FeatureRole,
    geom: FeatureGeometryMode,
    indexHint?: number,
  ): string {
    const roleLabel = role === 'exclude' ? 'Exclude' : 'Include'
    const geomLabel =
      geom === 'single'
        ? 'Pick'
        : geom === 'points'
          ? 'Points'
          : geom === 'polyline'
            ? 'Polyline'
            : 'Polygon'
    const n = indexHint ?? pickedFeatures.value.length + 1
    return `${roleLabel} ${geomLabel} #${n}`
  }

  function finishActiveDrawing(options?: { skipSave?: boolean }) {
    const id = activeDrawingFeatureId.value
    if (!id) return
    const idx = pickedFeatures.value.findIndex((f) => f.id === id)
    if (idx !== -1) {
      const feat = pickedFeatures.value[idx]
      if (feat.picks.length === 0) {
        pickedFeatures.value.splice(idx, 1)
        if (selectedFeatureId.value === id) {
          selectedFeatureId.value = pickedFeatures.value[0]?.id ?? null
        }
      } else {
        feat.status = 'completed'
      }
    }
    activeDrawingFeatureId.value = null
    if (!options?.skipSave) {
      saveFeaturesToStorage()
    }
  }

  function selectSurvey(surveyId: string) {
    const target = surveys.value.find((s) => s.id === surveyId)
    if (!target) return

    finishActiveDrawing({ skipSave: true })
    activeSurveyId.value = target.id
    const [nIl, nXl, nSamples] = target.shape
    const nOffsets = target.shape[3] ?? 1

    ilIdx.value = clampIndex(Math.floor(nIl / 2), nIl)
    xlIdx.value = clampIndex(Math.floor(nXl / 2), nXl)
    sampleIdx.value = clampIndex(Math.floor(nSamples / 2), nSamples)
    offsetIndex.value = clampIndex(
      target.default_offset_index ?? Math.floor(nOffsets / 2),
      nOffsets,
    )

    gatherIlStartIdx.value = ilIdx.value
    gatherXlStartIdx.value = clampIndex(Math.max(0, xlIdx.value - 35), nXl)

    sectionCache.clear()
    sections.value = { inline: null, xline: null, sample: null }
    gatherPanelData.value = null
    clearSimilarityState()
    similarityDirty.value = false
    attributeResult.value = null
    attributeVolume.value = null
    horizons.value = []
    wellPicks.value = []
    horizonsMeta.value = {}

    const preferredSet =
      target.embedding_sets.find((e) => e.id === 'ncs3d') ?? target.embedding_sets[0] ?? null
    applyEmbeddingSetDefaults(preferredSet)

    loadFeaturesFromStorage(target)
    if (preferredSet && enabledIncludeFeatures.value.length > 0) {
      if (overlaySource.value === 'off') {
        overlaySource.value = 'similarity'
      }
      if (overlaySource.value === 'similarity') {
        void runSimilarityQuery()
      } else if (focusPick.value) {
        void refreshFocusTrajectory()
      }
    } else if (preferredSet && focusPick.value) {
      void refreshFocusTrajectory()
    }

    void refreshActiveViews()
    void loadHorizons()
  }

  async function initialize() {
    loadingSurveys.value = true
    apiError.value = null
    try {
      const [h, sList] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchSurveys(),
      ])
      health.value = h
      surveys.value = sList
      if (sList.length > 0) {
        selectSurvey(sList[0].id)
      }
      await refreshJobs()
    } catch (err) {
      apiError.value = err instanceof Error ? err.message : String(err)
    } finally {
      loadingSurveys.value = false
    }
  }

  async function rescanAndRefreshSurveys() {
    loadingSurveys.value = true
    apiError.value = null
    try {
      const prevId = activeSurveyId.value
      const sList = await rescanSurveys()
      surveys.value = sList
      if (sList.length > 0) {
        const match = sList.find((s) => s.id === prevId)
        if (match) {
          // Keep current coordinates, refresh embedding sets
          activeSurveyId.value = match.id
          const curSet =
            match.embedding_sets.find((e) => e.id === activeEmbeddingSetId.value) ??
            match.embedding_sets.find((e) => e.id === 'ncs3d') ??
            match.embedding_sets[0] ??
            null
          applyEmbeddingSetDefaults(curSet)
          if (match.horizon_ids.length !== horizons.value.length) void loadHorizons()
          if (
            overlaySource.value === 'attribute' &&
            (!curSet || curSet.available_attributes.length === 0)
          ) {
            overlaySource.value = curSet && curSet.available_modes.length > 0 ? 'similarity' : 'off'
          }
          if (overlaySource.value === 'similarity' && hasQueryableFeatures.value && curSet) {
            void runSimilarityQuery()
          } else if (overlaySource.value === 'attribute' && selectedAttribute.value && curSet) {
            void loadAttribute()
          }
        } else {
          selectSurvey(sList[0].id)
        }
      }
    } catch (err) {
      apiError.value = err instanceof Error ? err.message : String(err)
    } finally {
      loadingSurveys.value = false
    }
  }

  async function ensureSection(axis: SectionAxis, indexOverride?: number): Promise<void> {
    const survey = activeSurvey.value
    if (!survey) return
    const idx = indexOverride !== undefined ? indexOverride : getAxisIndex(axis)
    const offIdx = offsetIndex.value
    const cacheKey = `${survey.id}:${axis}:${idx}:${offIdx}`

    const cached = sectionCache.get(cacheKey)
    if (cached) {
      // Touch LRU
      sectionCache.delete(cacheKey)
      sectionCache.set(cacheKey, cached)
      abortSlot(`section:${axis}`)
      inFlightSections.delete(axis)
      sectionLoading.value = { ...sectionLoading.value, [axis]: false }
      if (idx === getAxisIndex(axis)) {
        sections.value = { ...sections.value, [axis]: cached }
      }
      return
    }

    if (inFlightSections.get(axis) === cacheKey) {
      return
    }

    inFlightSections.set(axis, cacheKey)
    sectionLoading.value = { ...sectionLoading.value, [axis]: true }

    try {
      const result = await fetchSection(survey.id, axis, idx, offIdx, `section:${axis}`)
      sectionCache.set(cacheKey, result)
      if (sectionCache.size > MAX_SECTION_CACHE) {
        const firstKey = sectionCache.keys().next().value
        if (firstKey) sectionCache.delete(firstKey)
      }
      // Apply only if still aligned with current cursor + offset
      if (
        activeSurvey.value?.id === survey.id &&
        idx === getAxisIndex(axis) &&
        offsetIndex.value === offIdx
      ) {
        sections.value = { ...sections.value, [axis]: result }
      }
    } catch (err) {
      if (!isAbortError(err)) {
        apiError.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      if (inFlightSections.get(axis) === cacheKey) {
        inFlightSections.delete(axis)
        sectionLoading.value = { ...sectionLoading.value, [axis]: false }
      }
    }
  }

  function isViewerVisible(type: ViewerType): boolean {
    if (layoutMode.value === 'focus-section') return type === 'section'
    if (layoutMode.value === 'focus-cube') return type === 'cube'
    if (layoutMode.value === 'focus-gather') return type === 'gather'
    if (maximizedPane.value) {
      return paneViews.value[maximizedPane.value] === type
    }
    if (layoutMode.value === 'split-2h' || layoutMode.value === 'split-2v') {
      return paneViews.value.a === type || paneViews.value.b === type
    }
    return (
      paneViews.value.a === type ||
      paneViews.value.b === type ||
      paneViews.value.c === type
    )
  }

  async function refreshActiveViews() {
    const tasks: Promise<void>[] = []
    if (isViewerVisible('cube')) {
      tasks.push(ensureSection('inline'), ensureSection('xline'), ensureSection('sample'))
    } else if (isViewerVisible('section')) {
      tasks.push(ensureSection('inline'), ensureSection('xline'), ensureSection('sample'))
    }
    if (isViewerVisible('gather') || !gatherPanelData.value) {
      tasks.push(loadGatherPanel())
    }
    await Promise.all(tasks)
  }

  function setCursor(
    next: Partial<{ ilIdx: number; xlIdx: number; sampleIdx: number; offsetIndex: number }>,
    options?: {
      skipSimilarity?: boolean
      preserveDrawingOnPlane?: { planeAxis: FeaturePlaneAxis; planeIndex: number }
    },
  ) {
    const survey = activeSurvey.value
    if (!survey) return
    const [nIl, nXl, nSamples] = survey.shape
    const nOffsets = survey.shape[3] ?? 1

    let ilChanged = false
    let xlChanged = false
    let sampleChanged = false
    let offsetChanged = false

    if (next.ilIdx !== undefined) {
      const clamped = clampIndex(next.ilIdx, nIl)
      if (clamped !== ilIdx.value) {
        ilIdx.value = clamped
        ilChanged = true
      }
    }
    if (next.xlIdx !== undefined) {
      const clamped = clampIndex(next.xlIdx, nXl)
      if (clamped !== xlIdx.value) {
        xlIdx.value = clamped
        xlChanged = true
      }
    }
    if (next.sampleIdx !== undefined) {
      const clamped = clampIndex(next.sampleIdx, nSamples)
      if (clamped !== sampleIdx.value) {
        sampleIdx.value = clamped
        sampleChanged = true
      }
    }
    if (next.offsetIndex !== undefined) {
      const clamped = clampIndex(next.offsetIndex, nOffsets)
      if (clamped !== offsetIndex.value) {
        offsetIndex.value = clamped
        offsetChanged = true
      }
    }

    // Auto-finish active drawing if cursor leaves its locked plane
    const drawing = activeDrawingFeature.value
    if (drawing) {
      const preserve = options?.preserveDrawingOnPlane
      const preserved =
        preserve &&
        preserve.planeAxis === drawing.plane_axis &&
        preserve.planeIndex === drawing.plane_index
      if (!preserved) {
        if (drawing.plane_axis === 'inline' && ilIdx.value !== drawing.plane_index) {
          finishActiveDrawing()
        } else if (drawing.plane_axis === 'xline' && xlIdx.value !== drawing.plane_index) {
          finishActiveDrawing()
        }
      }
    }

    if (offsetChanged) {
      void Promise.all([
        ensureSection('inline'),
        ensureSection('xline'),
        ensureSection('sample'),
      ])
      if (!options?.skipSimilarity) {
        if (hasQueryableFeatures.value && similarityMode.value === 'active_offset') {
          if (overlaySource.value === 'similarity') {
            void runSimilarityQuery()
          } else {
            clearSimilarityState()
            similarityDirty.value = true
            if (focusPick.value) {
              void refreshFocusTrajectory()
            }
          }
        } else if (focusPick.value) {
          void refreshFocusTrajectory()
        }
      }
    } else {
      const promises: Promise<void>[] = []
      if (ilChanged) promises.push(ensureSection('inline'))
      if (xlChanged) promises.push(ensureSection('xline'))
      if (sampleChanged) promises.push(ensureSection('sample'))
      if (promises.length > 0) void Promise.all(promises)
    }
  }

  async function loadGatherPanel() {
    const survey = activeSurvey.value
    if (!survey || !survey.is_4d) return
    const [nIl, nXl] = survey.shape
    gatherIlStartIdx.value = clampIndex(gatherIlStartIdx.value, nIl)
    gatherXlStartIdx.value = clampIndex(gatherXlStartIdx.value, nXl)
    gatherIncrement.value = Math.max(1, Math.min(50, Math.round(gatherIncrement.value)))
    gatherCount.value = Math.max(1, Math.min(30, Math.round(gatherCount.value)))

    gatherLoading.value = true
    try {
      const data = await fetchGatherPanel(
        survey.id,
        gatherLineAxis.value,
        gatherIlStartIdx.value,
        gatherXlStartIdx.value,
        gatherIncrement.value,
        gatherCount.value,
      )
      gatherPanelData.value = data
    } catch (err) {
      if (!isAbortError(err)) {
        apiError.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      gatherLoading.value = false
    }
  }

  function useCursorForGatherStart() {
    gatherIlStartIdx.value = ilIdx.value
    gatherXlStartIdx.value = xlIdx.value
    void loadGatherPanel()
  }

  function makeCoplanarCoord(
    planeAxis: FeaturePlaneAxis,
    planeIndex: number,
    horizIdx: number,
    sampleIdxVal: number,
  ): PickCoord {
    const survey = activeSurvey.value
    const [nIl, nXl, nSamples] = survey ? survey.shape : [1, 1, 1]
    if (planeAxis === 'inline') {
      return {
        il_idx: clampIndex(planeIndex, nIl),
        xl_idx: clampIndex(horizIdx, nXl),
        sample_idx: clampIndex(sampleIdxVal, nSamples),
      }
    }
    return {
      il_idx: clampIndex(horizIdx, nIl),
      xl_idx: clampIndex(planeIndex, nXl),
      sample_idx: clampIndex(sampleIdxVal, nSamples),
    }
  }

  function startNewFeatureSlot() {
    finishActiveDrawing()
    selectedFeatureId.value = null
  }

  function setGeometryMode(mode: FeatureGeometryMode) {
    if (activeGeometryMode.value === mode) return
    finishActiveDrawing()
    activeGeometryMode.value = mode
  }

  function setFeatureRole(role: FeatureRole) {
    if (activeFeatureRole.value === role) return
    finishActiveDrawing()
    activeFeatureRole.value = role
  }

  watch(activeGeometryMode, () => {
    if (
      activeDrawingFeature.value &&
      activeDrawingFeature.value.geometry_mode !== activeGeometryMode.value
    ) {
      finishActiveDrawing()
    }
  })

  watch(activeFeatureRole, () => {
    if (
      activeDrawingFeature.value &&
      activeDrawingFeature.value.role !== activeFeatureRole.value
    ) {
      finishActiveDrawing()
    }
  })

  async function refreshFocusTrajectory() {
    const survey = activeSurvey.value
    const embSet = activeEmbeddingSet.value
    const fp = focusPick.value
    if (!survey || !embSet || !fp) {
      focusTrajectory.value = null
      return
    }
    try {
      const resp = await fetchFocusTrajectory(survey.id, {
        embedding_set_id: embSet.id,
        il_idx: fp.il_idx,
        xl_idx: fp.xl_idx,
        sample_idx: fp.sample_idx,
        offset_idx: offsetIndex.value,
      })
      focusTrajectory.value = resp
    } catch (err) {
      if (!isAbortError(err)) {
        // Non-fatal: keep existing trajectory or null
      }
    }
  }

  function scheduleFocusTrajectoryRefresh() {
    if (trajectoryDebounceTimer !== null) {
      window.clearTimeout(trajectoryDebounceTimer)
    }
    trajectoryDebounceTimer = window.setTimeout(() => {
      trajectoryDebounceTimer = null
      void refreshFocusTrajectory()
    }, 60)
  }

  function triggerSimilarityOrFallback() {
    if (!hasQueryableFeatures.value) {
      clearSimilarityState()
      similarityDirty.value = false
      if (focusPick.value) {
        void refreshFocusTrajectory()
      } else {
        focusTrajectory.value = null
      }
      return
    }
    if (activeEmbeddingSet.value) {
      if (overlaySource.value === 'off') {
        overlaySource.value = 'similarity'
      }
      if (overlaySource.value === 'similarity') {
        void runSimilarityQuery()
      } else {
        clearSimilarityState()
        similarityDirty.value = true
        if (focusPick.value) {
          void refreshFocusTrajectory()
        }
      }
    }
  }

  async function placePickOnPlane(
    planeAxis: FeaturePlaneAxis,
    planeIndex: number,
    horizIdx: number,
    sampleIdxVal: number,
    options?: { isDoubleClick?: boolean },
  ) {
    const survey = activeSurvey.value
    if (!survey) return
    const [nIl, nXl] = survey.shape
    const clampedPlane = clampIndex(planeIndex, planeAxis === 'inline' ? nIl : nXl)
    const coord = makeCoplanarCoord(planeAxis, clampedPlane, horizIdx, sampleIdxVal)

    setCursor(
      {
        ilIdx: coord.il_idx,
        xlIdx: coord.xl_idx,
        sampleIdx: coord.sample_idx,
      },
      {
        skipSimilarity: true,
        preserveDrawingOnPlane: { planeAxis, planeIndex: clampedPlane },
      },
    )

    if (activeGeometryMode.value === 'single') {
      finishActiveDrawing({ skipSave: true })
      const existing = selectedFeatureId.value
        ? pickedFeatures.value.find((f) => f.id === selectedFeatureId.value)
        : undefined
      if (
        existing &&
        existing.geometry_mode === 'single' &&
        existing.role === activeFeatureRole.value
      ) {
        existing.plane_axis = planeAxis
        existing.plane_index = clampedPlane
        existing.picks = [coord]
        existing.enabled = true
        existing.status = 'completed'
      } else {
        const id = generateFeatureId()
        const feat: PickedFeature = {
          id,
          name: buildFeatureName(activeFeatureRole.value, 'single'),
          role: activeFeatureRole.value,
          plane_axis: planeAxis,
          plane_index: clampedPlane,
          geometry_mode: 'single',
          picks: [coord],
          enabled: true,
          status: 'completed',
          createdAt: Date.now(),
        }
        pickedFeatures.value.push(feat)
        selectedFeatureId.value = id
      }
    } else {
      const currentDrawing = activeDrawingFeature.value
      if (
        currentDrawing &&
        (currentDrawing.plane_axis !== planeAxis ||
          currentDrawing.plane_index !== clampedPlane ||
          currentDrawing.geometry_mode !== activeGeometryMode.value ||
          currentDrawing.role !== activeFeatureRole.value)
      ) {
        finishActiveDrawing({ skipSave: true })
      }

      const active = activeDrawingFeature.value
      if (!active) {
        if (!options?.isDoubleClick) {
          const id = generateFeatureId()
          const feat: PickedFeature = {
            id,
            name: buildFeatureName(activeFeatureRole.value, activeGeometryMode.value),
            role: activeFeatureRole.value,
            plane_axis: planeAxis,
            plane_index: clampedPlane,
            geometry_mode: activeGeometryMode.value,
            picks: [coord],
            enabled: true,
            status: 'in_progress',
            createdAt: Date.now(),
          }
          pickedFeatures.value.push(feat)
          activeDrawingFeatureId.value = id
          selectedFeatureId.value = id
        }
      } else {
        const last = active.picks[active.picks.length - 1]
        const isExactDup =
          last &&
          last.il_idx === coord.il_idx &&
          last.xl_idx === coord.xl_idx &&
          last.sample_idx === coord.sample_idx
        const isNearDoubleClickDup =
          Boolean(options?.isDoubleClick) &&
          last &&
          Math.abs(last.il_idx - coord.il_idx) +
            Math.abs(last.xl_idx - coord.xl_idx) +
            Math.abs(last.sample_idx - coord.sample_idx) <=
            2

        if (!isExactDup && !isNearDoubleClickDup) {
          active.picks.push(coord)
        }
        if (options?.isDoubleClick) {
          finishActiveDrawing({ skipSave: true })
        }
      }
    }

    focusPick.value = { ...coord }
    saveFeaturesToStorage()
    triggerSimilarityOrFallback()
  }

  function movePick(
    featureId: string,
    pickIndex: number,
    horizIdx: number,
    sampleIdxVal: number,
    options?: { commit?: boolean },
  ) {
    const feat = pickedFeatures.value.find((f) => f.id === featureId)
    if (!feat || pickIndex < 0 || pickIndex >= feat.picks.length) return

    const coord = makeCoplanarCoord(
      feat.plane_axis,
      feat.plane_index,
      horizIdx,
      sampleIdxVal,
    )
    feat.picks[pickIndex] = coord
    focusPick.value = { ...coord }

    if (options?.commit) {
      if (moveThrottleTimer !== null) {
        window.clearTimeout(moveThrottleTimer)
        moveThrottleTimer = null
      }
      saveFeaturesToStorage()
      triggerSimilarityOrFallback()
      return
    }

    if (moveThrottleTimer === null) {
      moveThrottleTimer = window.setTimeout(() => {
        moveThrottleTimer = null
        triggerSimilarityOrFallback()
      }, 80)
    }
  }

  function insertPickOnSegment(
    featureId: string,
    afterPickIndex: number,
    horizIdx: number,
    sampleIdxVal: number,
  ) {
    const feat = pickedFeatures.value.find((f) => f.id === featureId)
    if (!feat || feat.picks.length < 2) return
    const insertAt = Math.max(1, Math.min(feat.picks.length, afterPickIndex + 1))
    const coord = makeCoplanarCoord(
      feat.plane_axis,
      feat.plane_index,
      horizIdx,
      sampleIdxVal,
    )
    feat.picks.splice(insertAt, 0, coord)
    selectedFeatureId.value = feat.id
    focusPick.value = { ...coord }
    saveFeaturesToStorage()
    triggerSimilarityOrFallback()
  }

  function deletePick(featureId: string, pickIndex: number) {
    const idx = pickedFeatures.value.findIndex((f) => f.id === featureId)
    if (idx === -1) return
    const feat = pickedFeatures.value[idx]
    if (pickIndex < 0 || pickIndex >= feat.picks.length) return

    const removed = feat.picks[pickIndex]
    feat.picks.splice(pickIndex, 1)

    if (hoveredPickRef.value?.featureId === featureId) {
      if (hoveredPickRef.value.pickIndex === pickIndex) {
        hoveredPickRef.value = null
      } else if (hoveredPickRef.value.pickIndex > pickIndex) {
        hoveredPickRef.value = {
          featureId,
          pickIndex: hoveredPickRef.value.pickIndex - 1,
        }
      }
    }

    if (feat.picks.length === 0) {
      if (activeDrawingFeatureId.value === featureId) {
        activeDrawingFeatureId.value = null
      }
      pickedFeatures.value.splice(idx, 1)
      if (selectedFeatureId.value === featureId) {
        selectedFeatureId.value = pickedFeatures.value[0]?.id ?? null
      }
    }

    if (
      focusPick.value &&
      focusPick.value.il_idx === removed.il_idx &&
      focusPick.value.xl_idx === removed.xl_idx &&
      focusPick.value.sample_idx === removed.sample_idx
    ) {
      const fallback =
        feat.picks[Math.min(pickIndex, feat.picks.length - 1)] ??
        enabledIncludeFeatures.value[0]?.picks[0] ??
        pickedFeatures.value[0]?.picks[0] ??
        null
      focusPick.value = fallback ? { ...fallback } : null
    }

    saveFeaturesToStorage()
    triggerSimilarityOrFallback()
  }

  function toggleFeatureRole(id: string) {
    const feat = pickedFeatures.value.find((f) => f.id === id)
    if (!feat) return
    feat.role = feat.role === 'include' ? 'exclude' : 'include'
    saveFeaturesToStorage()
    triggerSimilarityOrFallback()
  }

  function toggleFeatureEnabled(id: string) {
    const feat = pickedFeatures.value.find((f) => f.id === id)
    if (!feat) return
    feat.enabled = !feat.enabled
    if (!feat.enabled) {
      if (activeDrawingFeatureId.value === id) {
        finishActiveDrawing({ skipSave: true })
      }
      if (hoveredPickRef.value?.featureId === id) {
        hoveredPickRef.value = null
      }
    }
    saveFeaturesToStorage()
    triggerSimilarityOrFallback()
  }

  function deleteFeature(id: string) {
    const idx = pickedFeatures.value.findIndex((f) => f.id === id)
    if (idx === -1) return
    if (activeDrawingFeatureId.value === id) {
      activeDrawingFeatureId.value = null
    }
    if (hoveredPickRef.value?.featureId === id) {
      hoveredPickRef.value = null
    }
    pickedFeatures.value.splice(idx, 1)
    if (selectedFeatureId.value === id) {
      selectedFeatureId.value = pickedFeatures.value[0]?.id ?? null
    }
    const nextFocus =
      enabledIncludeFeatures.value[0]?.picks[0] ?? pickedFeatures.value[0]?.picks[0] ?? null
    focusPick.value = nextFocus ? { ...nextFocus } : null
    saveFeaturesToStorage()
    triggerSimilarityOrFallback()
  }

  function clearAllFeatures() {
    finishActiveDrawing({ skipSave: true })
    pickedFeatures.value = []
    selectedFeatureId.value = null
    activeDrawingFeatureId.value = null
    hoveredPickRef.value = null
    focusPick.value = null
    focusTrajectory.value = null
    clearSimilarityState()
    similarityDirty.value = false
    saveFeaturesToStorage()
  }

  function jumpToFeature(id: string) {
    const feat = pickedFeatures.value.find((f) => f.id === id)
    if (!feat || feat.picks.length === 0) return
    selectedFeatureId.value = id
    const sumIl = feat.picks.reduce((acc, p) => acc + p.il_idx, 0)
    const sumXl = feat.picks.reduce((acc, p) => acc + p.xl_idx, 0)
    const sumS = feat.picks.reduce((acc, p) => acc + p.sample_idx, 0)
    const n = feat.picks.length
    const targetIl = feat.plane_axis === 'inline' ? feat.plane_index : Math.round(sumIl / n)
    const targetXl = feat.plane_axis === 'xline' ? feat.plane_index : Math.round(sumXl / n)
    const targetSample = Math.round(sumS / n)

    setCursor(
      { ilIdx: targetIl, xlIdx: targetXl, sampleIdx: targetSample },
      {
        skipSimilarity: true,
        preserveDrawingOnPlane: { planeAxis: feat.plane_axis, planeIndex: feat.plane_index },
      },
    )
    focusPick.value = { ...feat.picks[0] }
    void refreshFocusTrajectory()
  }

  function setExclusionWeight(val: number) {
    const clamped = Math.max(0, Math.min(2, Math.round(val * 100) / 100))
    exclusionWeight.value = clamped
    saveFeaturesToStorage()
    if (!hasQueryableFeatures.value) return
    if (overlaySource.value !== 'similarity') {
      clearSimilarityState()
      similarityDirty.value = true
      if (focusPick.value) {
        void refreshFocusTrajectory()
      }
      return
    }
    if (weightThrottleTimer !== null) {
      window.clearTimeout(weightThrottleTimer)
    }
    weightThrottleTimer = window.setTimeout(() => {
      weightThrottleTimer = null
      if (overlaySource.value === 'similarity') {
        void runSimilarityQuery()
      }
    }, 90)
  }

  function onSimilarityParamsChanged() {
    if (!hasQueryableFeatures.value) {
      clearSimilarityState()
      similarityDirty.value = false
      return
    }
    if (overlaySource.value === 'similarity') {
      void runSimilarityQuery()
    } else {
      clearSimilarityState()
      similarityDirty.value = true
      if (focusPick.value) {
        void refreshFocusTrajectory()
      }
    }
  }

  function setHoveredPick(refVal: { featureId: string; pickIndex: number } | null) {
    const prev = hoveredPickRef.value
    if (
      prev?.featureId === refVal?.featureId &&
      prev?.pickIndex === refVal?.pickIndex
    ) {
      return
    }
    hoveredPickRef.value = refVal
    if (refVal) {
      const feat = pickedFeatures.value.find((f) => f.id === refVal.featureId)
      const pick = feat?.picks[refVal.pickIndex]
      if (pick) {
        focusPick.value = { ...pick }
        scheduleFocusTrajectoryRefresh()
      }
    }
  }

  async function runSimilarityQuery() {
    const survey = activeSurvey.value
    const embSet = activeEmbeddingSet.value
    if (!survey || !embSet || !hasQueryableFeatures.value) {
      clearSimilarityState()
      similarityDirty.value = false
      return
    }

    const enabledPayload: PickedFeaturePayload[] = pickedFeatures.value
      .filter((f) => f.enabled && f.picks.length >= 1)
      .map((f) => ({
        id: f.id,
        role: f.role,
        plane_axis: f.plane_axis,
        plane_index: f.plane_index,
        geometry_mode: f.geometry_mode,
        picks: f.picks,
      }))

    const fallbackPick = enabledIncludeFeatures.value[0]?.picks[0]
    const anchor = focusPick.value ?? fallbackPick
    if (!anchor) {
      clearSimilarityState()
      similarityDirty.value = false
      return
    }

    const reqId = ++similarityReqSeq
    similarityDirty.value = false
    similarityLoading.value = true
    apiError.value = null
    try {
      const resp = await fetchSimilarity(survey.id, {
        embedding_set_id: embSet.id,
        offset_idx: offsetIndex.value,
        mode: similarityMode.value,
        whiten: whitenChannels.value,
        exclusion_weight: exclusionWeight.value,
        features: enabledPayload,
        focus_pick: anchor,
        il_idx: anchor.il_idx,
        xl_idx: anchor.xl_idx,
        sample_idx: anchor.sample_idx,
      })
      if (reqId !== similarityReqSeq) return
      similarityResult.value = resp
      similarityVolume.value = decodeFloat32Base64(resp.similarity_b64)
      focusTrajectory.value = {
        pick: resp.pick,
        trajectory: resp.trajectory,
      }
    } catch (err) {
      if (reqId === similarityReqSeq && !isAbortError(err)) {
        apiError.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      if (reqId === similarityReqSeq) {
        similarityLoading.value = false
      }
    }
  }

  async function loadAttribute(attrName?: string) {
    const survey = activeSurvey.value
    const embSet = activeEmbeddingSet.value
    if (!survey || !embSet) return

    const attr = attrName ?? selectedAttribute.value
    if (!attr) return
    selectedAttribute.value = attr

    attributeLoading.value = true
    apiError.value = null
    try {
      const resp = await fetchAttribute(survey.id, embSet.id, attr)
      attributeResult.value = resp
      attributeVolume.value = decodeFloat32Base64(resp.values_b64)
      attributeThreshold.value = resp.value_min
    } catch (err) {
      if (!isAbortError(err)) {
        apiError.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      attributeLoading.value = false
    }
  }

  async function loadHorizons() {
    const survey = activeSurvey.value
    if (!survey || survey.horizon_ids.length === 0) {
      horizons.value = []
      wellPicks.value = []
      horizonsMeta.value = {}
      return
    }
    horizonsLoading.value = true
    try {
      const resp = await fetchHorizons(survey.id)
      if (activeSurvey.value?.id !== survey.id) return
      horizons.value = resp.horizons.map((h) => ({
        id: h.id,
        label: h.label,
        kind: h.kind,
        color: h.color,
        twt: decodeFloat32Base64(h.twt_b64),
        twtMin: h.twt_min,
        twtMax: h.twt_max,
        coverage: h.coverage,
      }))
      wellPicks.value = resp.well_picks
      horizonsMeta.value = resp.meta
    } catch (err) {
      if (!isAbortError(err)) {
        apiError.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      horizonsLoading.value = false
    }
  }

  function toggleReservoir(force?: boolean) {
    showReservoir.value = force !== undefined ? force : !showReservoir.value
  }

  function selectEmbeddingSet(setId: string) {
    const survey = activeSurvey.value
    if (!survey) return
    const embSet = survey.embedding_sets.find((e) => e.id === setId) ?? null
    clearSimilarityState()
    similarityDirty.value = false
    attributeResult.value = null
    attributeVolume.value = null
    focusTrajectory.value = null
    applyEmbeddingSetDefaults(embSet)

    if (
      overlaySource.value === 'attribute' &&
      (!embSet || embSet.available_attributes.length === 0)
    ) {
      overlaySource.value = embSet && embSet.available_modes.length > 0 ? 'similarity' : 'off'
    }

    if (overlaySource.value === 'similarity' && hasQueryableFeatures.value && embSet) {
      void runSimilarityQuery()
    } else if (overlaySource.value === 'attribute' && selectedAttribute.value && embSet) {
      void loadAttribute()
    } else if (focusPick.value && embSet) {
      void refreshFocusTrajectory()
    }
  }

  function setOverlaySource(source: OverlaySource) {
    overlaySource.value = source
    if (
      source === 'similarity' &&
      hasQueryableFeatures.value &&
      (similarityDirty.value || !similarityVolume.value)
    ) {
      void runSimilarityQuery()
    } else if (source === 'attribute' && selectedAttribute.value && !attributeVolume.value) {
      void loadAttribute()
    }
  }

  // Layout controls
  function setLayoutMode(mode: LayoutMode) {
    layoutMode.value = mode
    maximizedPane.value = null
    void refreshActiveViews()
  }

  function setPaneView(pane: 'a' | 'b' | 'c', view: ViewerType) {
    paneViews.value = { ...paneViews.value, [pane]: view }
    void refreshActiveViews()
  }

  function toggleMaximizePane(pane: 'a' | 'b' | 'c') {
    maximizedPane.value = maximizedPane.value === pane ? null : pane
    void refreshActiveViews()
  }

  // Jobs management
  function syncJobPolling() {
    if (hasActiveJobs.value) {
      if (pollTimer === null) {
        pollTimer = window.setInterval(() => {
          void refreshJobs()
        }, 1500)
      }
    } else if (pollTimer !== null) {
      window.clearInterval(pollTimer)
      pollTimer = null
    }
  }

  async function refreshJobs() {
    jobsLoading.value = true
    try {
      const list = await fetchJobs()
      let anyJustCompleted = false
      for (const job of list) {
        if (job.status === 'completed' && previouslyRunningIds.has(job.id)) {
          anyJustCompleted = true
        }
      }
      previouslyRunningIds.clear()
      for (const job of list) {
        if (job.status === 'queued' || job.status === 'running') {
          previouslyRunningIds.add(job.id)
        }
      }
      jobs.value = list
      syncJobPolling()
      if (anyJustCompleted) {
        await rescanAndRefreshSurveys()
      }
    } catch {
      // Don't flood banner if jobs endpoint temporarily fails
    } finally {
      jobsLoading.value = false
    }
  }

  async function launchDemoJob(payload: DemoEmbeddingsRequest) {
    const job = await startDemoEmbeddingsJob(payload)
    previouslyRunningIds.add(job.id)
    await refreshJobs()
    return job
  }

  async function launchEmbedJob(payload: EmbedJobRequest) {
    const job = await startEmbedJob(payload)
    previouslyRunningIds.add(job.id)
    await refreshJobs()
    return job
  }

  async function launchSummarizeJob(payload: SummarizeJobRequest) {
    const job = await startSummarizeJob(payload)
    previouslyRunningIds.add(job.id)
    await refreshJobs()
    return job
  }

  async function cancelRunningJob(jobId: string) {
    await cancelJob(jobId)
    await refreshJobs()
  }

  return {
    // State
    health,
    surveys,
    activeSurveyId,
    loadingSurveys,
    apiError,
    ilIdx,
    xlIdx,
    sampleIdx,
    offsetIndex,
    seismicColormap,
    overlayColormap,
    globalGain,
    overlayInterp,
    overlayOpacity,
    layoutMode,
    paneViews,
    maximizedPane,
    sidebarCollapsed,
    jobsModalOpen,
    sections,
    sectionLoading,
    gatherLineAxis,
    gatherIlStartIdx,
    gatherXlStartIdx,
    gatherIncrement,
    gatherCount,
    gatherWiggle,
    gatherHighlightActiveOffset,
    gatherSyncOffsetOnPick,
    gatherPanelData,
    gatherLoading,
    activeEmbeddingSetId,
    overlaySource,
    similarityMode,
    whitenChannels,
    similarityThreshold,
    pickMode,
    activeGeometryMode,
    activeFeatureRole,
    exclusionWeight,
    pickedFeatures,
    selectedFeatureId,
    activeDrawingFeatureId,
    focusPick,
    focusTrajectory,
    hoveredPickRef,
    similarityResult,
    similarityVolume,
    similarityLoading,
    similarityDirty,
    selectedAttribute,
    attributeThreshold,
    attributeResult,
    attributeVolume,
    attributeLoading,
    horizons,
    wellPicks,
    horizonsMeta,
    horizonsLoading,
    showReservoir,
    reservoirShowModel,
    reservoirShowHorizons,
    reservoirShowWells,
    reservoirOpacity,
    jobs,
    jobsLoading,

    // Getters
    activeSurvey,
    activeEmbeddingSet,
    effectiveClipAmplitude,
    currentInlineLabel,
    currentXlineLabel,
    currentSampleLabel,
    currentOffsetLabel,
    activeOverlay,
    hasActiveJobs,
    hasHorizons,
    sampleOrigin,
    sampleRate,
    reservoirEnvelope,
    interpretedHorizons,
    visibleWellPicks,
    activeDrawingFeature,
    enabledIncludeFeatures,
    enabledExcludeFeatures,
    hasOnlyExcludeFeatures,
    hasQueryableFeatures,

    // Actions
    initialize,
    rescanAndRefreshSurveys,
    selectSurvey,
    ensureSection,
    refreshActiveViews,
    setCursor,
    loadGatherPanel,
    useCursorForGatherStart,
    startNewFeatureSlot,
    setGeometryMode,
    setFeatureRole,
    placePickOnPlane,
    finishActiveDrawing,
    movePick,
    insertPickOnSegment,
    deletePick,
    toggleFeatureRole,
    toggleFeatureEnabled,
    deleteFeature,
    clearAllFeatures,
    jumpToFeature,
    setExclusionWeight,
    onSimilarityParamsChanged,
    setHoveredPick,
    refreshFocusTrajectory,
    runSimilarityQuery,
    loadAttribute,
    loadHorizons,
    toggleReservoir,
    twtToSample,
    selectEmbeddingSet,
    setOverlaySource,
    setLayoutMode,
    setPaneView,
    toggleMaximizePane,
    refreshJobs,
    launchDemoJob,
    launchEmbedJob,
    launchSummarizeJob,
    cancelRunningJob,
  }
})
