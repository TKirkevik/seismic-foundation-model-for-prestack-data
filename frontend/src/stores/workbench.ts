import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import {
  cancelJob,
  decodeFloat32Base64,
  fetchAttribute,
  fetchGatherPanel,
  fetchHealth,
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
  FeaturePick,
  GatherLineAxis,
  GatherPanelResponseData,
  HealthStatus,
  JobInfo,
  LayoutMode,
  OverlayColormapId,
  OverlaySource,
  SectionAxis,
  SectionResponseData,
  SeismicColormapId,
  SimilarityResponse,
  SummarizeJobRequest,
  SurveySummary,
  ViewerType,
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

const MAX_SECTION_CACHE = 48

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

  // Latent exploration & Feature Pick
  const activeEmbeddingSetId = ref<string | null>(null)
  const overlaySource = ref<OverlaySource>('similarity')
  const similarityMode = ref<string>('active_offset')
  const whitenChannels = ref(false)
  const similarityThreshold = ref(0.55)
  const pickMode = ref(true)
  const featurePick = ref<FeaturePick | null>(null)
  const similarityResult = ref<SimilarityResponse | null>(null)
  const similarityVolume = ref<Float32Array | null>(null)
  const similarityLoading = ref(false)

  const selectedAttribute = ref<string>('')
  const attributeThreshold = ref(0)
  const attributeResult = ref<AttributeResponse | null>(null)
  const attributeVolume = ref<Float32Array | null>(null)
  const attributeLoading = ref(false)

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

  function getAxisIndex(axis: SectionAxis): number {
    if (axis === 'inline') return ilIdx.value
    if (axis === 'xline') return xlIdx.value
    return sampleIdx.value
  }

  function clampIndex(val: number, maxExclusive: number): number {
    if (maxExclusive <= 0) return 0
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

  function selectSurvey(surveyId: string) {
    const target = surveys.value.find((s) => s.id === surveyId)
    if (!target) return

    activeSurveyId.value = target.id
    const [nIl, nXl, nSamples, nOffsets] = target.shape

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
    featurePick.value = null
    similarityResult.value = null
    similarityVolume.value = null
    attributeResult.value = null
    attributeVolume.value = null

    const preferredSet =
      target.embedding_sets.find((e) => e.id === 'ncs3d') ?? target.embedding_sets[0] ?? null
    applyEmbeddingSetDefaults(preferredSet)

    void refreshActiveViews()
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
  ) {
    const survey = activeSurvey.value
    if (!survey) return
    const [nIl, nXl, nSamples, nOffsets] = survey.shape

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

    if (offsetChanged) {
      void Promise.all([
        ensureSection('inline'),
        ensureSection('xline'),
        ensureSection('sample'),
      ])
      if (
        featurePick.value &&
        overlaySource.value === 'similarity' &&
        similarityMode.value === 'active_offset'
      ) {
        void runSimilarityQuery()
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
    if (!survey) return
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

  async function pickFeature(
    il: number,
    xl: number,
    sample: number,
    offIdx?: number,
    syncCursor = true,
  ) {
    const survey = activeSurvey.value
    if (!survey) return
    const [nIl, nXl, nSamples, nOffsets] = survey.shape

    const cIl = clampIndex(il, nIl)
    const cXl = clampIndex(xl, nXl)
    const cSample = clampIndex(sample, nSamples)
    const cOff = clampIndex(offIdx !== undefined ? offIdx : offsetIndex.value, nOffsets)

    featurePick.value = {
      il_idx: cIl,
      xl_idx: cXl,
      sample_idx: cSample,
      offset_idx: cOff,
    }

    if (syncCursor) {
      setCursor({
        ilIdx: cIl,
        xlIdx: cXl,
        sampleIdx: cSample,
        offsetIndex: offIdx !== undefined ? cOff : undefined,
      })
    }

    if (activeEmbeddingSet.value) {
      if (overlaySource.value === 'off') {
        overlaySource.value = 'similarity'
      }
      if (overlaySource.value === 'similarity') {
        await runSimilarityQuery()
      }
    }
  }

  async function runSimilarityQuery() {
    const survey = activeSurvey.value
    const embSet = activeEmbeddingSet.value
    const pick = featurePick.value
    if (!survey || !embSet || !pick) return

    similarityLoading.value = true
    apiError.value = null
    try {
      const resp = await fetchSimilarity(survey.id, {
        embedding_set_id: embSet.id,
        il_idx: pick.il_idx,
        xl_idx: pick.xl_idx,
        sample_idx: pick.sample_idx,
        offset_idx:
          similarityMode.value === 'active_offset' ? offsetIndex.value : pick.offset_idx,
        mode: similarityMode.value,
        whiten: whitenChannels.value,
      })
      similarityResult.value = resp
      similarityVolume.value = decodeFloat32Base64(resp.similarity_b64)
    } catch (err) {
      if (!isAbortError(err)) {
        apiError.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      similarityLoading.value = false
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

  function selectEmbeddingSet(setId: string) {
    const survey = activeSurvey.value
    if (!survey) return
    const embSet = survey.embedding_sets.find((e) => e.id === setId) ?? null
    applyEmbeddingSetDefaults(embSet)
    if (overlaySource.value === 'similarity' && featurePick.value) {
      void runSimilarityQuery()
    } else if (overlaySource.value === 'attribute' && selectedAttribute.value) {
      void loadAttribute()
    }
  }

  function setOverlaySource(source: OverlaySource) {
    overlaySource.value = source
    if (source === 'similarity' && featurePick.value && !similarityVolume.value) {
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
    featurePick,
    similarityResult,
    similarityVolume,
    similarityLoading,
    selectedAttribute,
    attributeThreshold,
    attributeResult,
    attributeVolume,
    attributeLoading,
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

    // Actions
    initialize,
    rescanAndRefreshSurveys,
    selectSurvey,
    ensureSection,
    refreshActiveViews,
    setCursor,
    loadGatherPanel,
    useCursorForGatherStart,
    pickFeature,
    runSimilarityQuery,
    loadAttribute,
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
