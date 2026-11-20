<script setup lang="ts">
import { computed } from 'vue'
import {
  Cpu,
  Database,
  Eye,
  EyeOff,
  Layers,
  Plus,
  RefreshCw,
  Sparkles,
  Target,
  Trash2,
} from 'lucide-vue-next'
import type { FeatureGeometryMode, PickedFeature } from '../api/types'
import { useWorkbenchStore } from '../stores/workbench'
import { colormapGradientCss, getOverlayLUT } from '../utils/colormaps'
import TrajectoryCard from './TrajectoryCard.vue'

const store = useWorkbenchStore()

const survey = computed(() => store.activeSurvey)
const embSet = computed(() => store.activeEmbeddingSet)

const RESERVOIR_LAYERS = [
  {
    id: 'model',
    label: 'Reservoir model (Eclipse grid, top/base)',
    color: '#4ade80',
    get: () => store.reservoirShowModel,
    set: (v: boolean) => (store.reservoirShowModel = v),
  },
  {
    id: 'horizons',
    label: 'Top Hugin interpreted horizon',
    color: '#f472b6',
    get: () => store.reservoirShowHorizons,
    set: (v: boolean) => (store.reservoirShowHorizons = v),
  },
  {
    id: 'wells',
    label: 'Well picks (Top/Base Hugin)',
    color: '#fde047',
    get: () => store.reservoirShowWells,
    set: (v: boolean) => (store.reservoirShowWells = v),
  },
]

const reservoirMetaLine = computed(() => {
  const m = store.horizonsMeta
  const vInt = m.hugin_interval_velocity_mps
  const vAvg = m.v_avg_median_mps
  const rms = m.qc_rms_ms
  const parts: string[] = []
  if (typeof vInt === 'number') parts.push(`V_int Hugin ${Math.round(vInt)} m/s`)
  if (typeof vAvg === 'number') parts.push(`V_avg to top ${Math.round(vAvg)} m/s`)
  if (typeof rms === 'number') parts.push(`well tie RMS ${rms.toFixed(1)} ms`)
  return parts.join(' · ')
})

const MODE_LABELS: Record<string, string> = {
  active_offset: 'Active Offset',
  zero_offset: 'Zero-Offset',
  offset_direction: 'Offset Direction (AVO)',
  offset_curvature: 'Offset Curvature',
  combined: 'Combined Trajectory',
  combined_trajectory: 'Combined Trajectory',
}

const ATTR_LABELS: Record<string, string> = {
  offset_slope: 'Offset Slope ‖b‖',
  offset_fit_r2: 'Linear Fit R²',
  offset_axis_var_1: 'Axis 1 Variance',
  offset_axis_var_2: 'Axis 2 Curvature Variance',
  offset_axis_var_plane: '2D Plane Total Variance',
}

function formatModeLabel(mode: string): string {
  return MODE_LABELS[mode] ?? mode.replace(/_/g, ' ')
}

function formatAttrLabel(attr: string): string {
  return ATTR_LABELS[attr] ?? attr.replace(/_/g, ' ')
}

const overlayGradient = computed(() =>
  colormapGradientCss(getOverlayLUT(store.overlayColormap)),
)

const attrStep = computed(() => {
  const res = store.attributeResult
  if (!res) return 0.01
  const span = Math.max(1e-6, res.value_max - res.value_min)
  return Number((span / 100).toPrecision(2)) || 0.001
})

function findOffsetIndexNear(targetAbsFraction: number): number {
  const s = survey.value
  if (!s || !s.offsets.length) return 0
  let maxAbs = 0
  for (const o of s.offsets) {
    const a = Math.abs(o)
    if (a > maxAbs) maxAbs = a
  }
  const target = maxAbs * targetAbsFraction
  let bestIdx = 0
  let bestDiff = Infinity
  for (let i = 0; i < s.offsets.length; i++) {
    const d = Math.abs(Math.abs(s.offsets[i]) - target)
    // Prefer positive offset on tie
    if (d < bestDiff || (d === bestDiff && s.offsets[i] >= 0)) {
      bestDiff = d
      bestIdx = i
    }
  }
  return bestIdx
}

function jumpOffsetPreset(kind: 'near' | 'mid' | 'far') {
  if (kind === 'near') store.setCursor({ offsetIndex: findOffsetIndexNear(0) })
  else if (kind === 'mid') store.setCursor({ offsetIndex: findOffsetIndexNear(0.5) })
  else store.setCursor({ offsetIndex: findOffsetIndexNear(1.0) })
}

function onSurveyChange(e: Event) {
  const id = (e.target as HTMLSelectElement).value
  store.selectSurvey(id)
}

function onEmbeddingSetChange(e: Event) {
  const id = (e.target as HTMLSelectElement).value
  store.selectEmbeddingSet(id)
}

function onModeChange() {
  store.onSimilarityParamsChanged()
}

function onWhitenChange() {
  store.onSimilarityParamsChanged()
}

const GEOM_MODES: Array<{ id: FeatureGeometryMode; label: string }> = [
  { id: 'single', label: 'Single' },
  { id: 'points', label: 'Points' },
  { id: 'polyline', label: 'Polyline' },
  { id: 'polygon', label: 'Polygon' },
]

function formatPlaneBadge(feat: PickedFeature): string {
  const s = survey.value
  if (feat.plane_axis === 'inline') {
    const val = s?.ilines[feat.plane_index] ?? feat.plane_index
    return `IL ${val}`
  }
  const val = s?.xlines[feat.plane_index] ?? feat.plane_index
  return `XL ${val}`
}

function formatGeomLabel(mode: FeatureGeometryMode): string {
  if (mode === 'single') return 'Single'
  if (mode === 'points') return 'Points'
  if (mode === 'polyline') return 'Polyline'
  return 'Polygon'
}

function onAttributeChange() {
  if (store.selectedAttribute) {
    void store.loadAttribute(store.selectedAttribute)
  }
}

function jumpToMatch(ilIdx: number, xlIdx: number, sampleIdx: number) {
  store.setCursor({ ilIdx, xlIdx, sampleIdx })
}
</script>

<template>
  <aside
    class="w-80 shrink-0 h-full bg-slate-950 border-r border-slate-800/90 flex flex-col overflow-hidden"
  >
    <div class="flex-1 overflow-y-auto p-3 space-y-4">
      <!-- Survey & Common-Offset Volume -->
      <section class="space-y-2.5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-300">
            <Database class="w-3.5 h-3.5 text-cyan-400" />
            <span>Survey &amp; Common-Offset</span>
          </div>
          <button
            type="button"
            class="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800 transition-colors"
            title="Rescan data/ for surveys and embeddings"
            @click="store.rescanAndRefreshSurveys()"
          >
            <RefreshCw class="w-3 h-3" :class="{ 'animate-spin': store.loadingSurveys }" />
            <span>Rescan</span>
          </button>
        </div>

        <select
          :value="store.activeSurveyId ?? ''"
          class="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
          :disabled="store.surveys.length === 0"
          @change="onSurveyChange"
        >
          <option v-if="store.surveys.length === 0" value="">No surveys found</option>
          <option v-for="s in store.surveys" :key="s.id" :value="s.id">
            {{ s.name }}
          </option>
        </select>

        <!-- Common-Offset Volume Scrubber -->
        <div v-if="survey" class="rounded border border-slate-800/90 bg-slate-900/50 p-2 space-y-2">
          <div class="flex items-center justify-between text-[11px]">
            <span class="text-slate-400">Common-Offset Volume</span>
            <span class="font-mono-num text-amber-300 font-semibold">
              {{ store.currentOffsetLabel }} m
              <span class="text-slate-500 font-normal">
                (#{{ store.offsetIndex }}/{{ Math.max(0, survey.offsets.length - 1) }})
              </span>
            </span>
          </div>

          <input
            :value="store.offsetIndex"
            type="range"
            min="0"
            :max="Math.max(0, survey.offsets.length - 1)"
            step="1"
            class="ws-slider w-full"
            @input="store.setCursor({ offsetIndex: Number(($event.target as HTMLInputElement).value) })"
          />

          <div class="grid grid-cols-3 gap-1.5">
            <button
              type="button"
              class="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-950 border border-slate-800 text-slate-300 hover:border-amber-500/50 hover:text-amber-200 transition-colors"
              @click="jumpOffsetPreset('near')"
            >
              Near ~0m
            </button>
            <button
              type="button"
              class="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-950 border border-slate-800 text-slate-300 hover:border-amber-500/50 hover:text-amber-200 transition-colors"
              @click="jumpOffsetPreset('mid')"
            >
              Mid
            </button>
            <button
              type="button"
              class="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-950 border border-slate-800 text-slate-300 hover:border-amber-500/50 hover:text-amber-200 transition-colors"
              @click="jumpOffsetPreset('far')"
            >
              Far
            </button>
          </div>
        </div>

        <!-- Global Display Controls -->
        <div class="rounded border border-slate-800/90 bg-slate-900/50 p-2 space-y-2 text-[11px]">
          <div class="flex items-center justify-between gap-2">
            <span class="text-slate-400">Global Gain</span>
            <input
              v-model.number="store.globalGain"
              type="range"
              min="0.2"
              max="4"
              step="0.05"
              class="ws-slider flex-1"
            />
            <span class="font-mono-num text-slate-200 w-9 text-right">
              {{ store.globalGain.toFixed(1) }}×
            </span>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <label class="space-y-1">
              <span class="block text-[10px] text-slate-500">Seismic Colormap</span>
              <select
                v-model="store.seismicColormap"
                class="w-full bg-slate-950 border border-slate-800 rounded px-1.5 py-1 text-[11px] text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value="rwb">Seismic RWB</option>
                <option value="grayscale">Grayscale</option>
                <option value="bone">Bone</option>
              </select>
            </label>

            <label class="space-y-1">
              <span class="block text-[10px] text-slate-500">Overlay Colormap</span>
              <select
                v-model="store.overlayColormap"
                class="w-full bg-slate-950 border border-slate-800 rounded px-1.5 py-1 text-[11px] text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                <option value="inferno">Inferno</option>
                <option value="turbo">Turbo</option>
                <option value="viridis">Viridis</option>
                <option value="hot">Hot</option>
              </select>
            </label>
          </div>

          <div class="flex items-center justify-between pt-0.5">
            <span class="text-[10px] text-slate-500">Overlay Sampling</span>
            <div class="flex rounded bg-slate-950 border border-slate-800 p-0.5 text-[10px]">
              <button
                type="button"
                class="px-2 py-0.5 rounded transition-colors"
                :class="
                  store.overlayInterp === 'smooth'
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                "
                @click="store.overlayInterp = 'smooth'"
              >
                Smooth
              </button>
              <button
                type="button"
                class="px-2 py-0.5 rounded transition-colors"
                :class="
                  store.overlayInterp === 'patch'
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                "
                @click="store.overlayInterp = 'patch'"
              >
                Patch Grid
              </button>
            </div>
          </div>
        </div>
      </section>

      <hr class="border-slate-800/80" />

      <!-- Ground Truth: Volve reservoir in PP time -->
      <section v-if="store.hasHorizons" class="space-y-2.5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-300">
            <Layers class="w-3.5 h-3.5 text-emerald-400" />
            <span>Ground Truth</span>
          </div>
          <button
            type="button"
            class="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border transition-colors"
            :class="
              store.showReservoir
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-200'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'
            "
            title="Toggle the Hugin reservoir answer in all viewers"
            @click="store.toggleReservoir()"
          >
            <Eye v-if="store.showReservoir" class="w-3 h-3" />
            <EyeOff v-else class="w-3 h-3" />
            <span>{{ store.showReservoir ? 'Shown' : 'Hidden' }}</span>
          </button>
        </div>

        <div class="rounded border border-slate-800/90 bg-slate-900/50 p-2 space-y-2 text-[11px]">
          <label
            v-for="layer in RESERVOIR_LAYERS"
            :key="layer.id"
            class="flex items-center gap-2 cursor-pointer"
            :class="store.showReservoir ? 'text-slate-200' : 'text-slate-500'"
          >
            <input
              :checked="layer.get()"
              type="checkbox"
              class="accent-emerald-400 rounded"
              :disabled="!store.showReservoir"
              @change="layer.set(($event.target as HTMLInputElement).checked)"
            />
            <span class="w-3 h-0.5 rounded" :style="{ background: layer.color }" />
            <span>{{ layer.label }}</span>
          </label>

          <div class="space-y-1 pt-1 border-t border-slate-800/80">
            <div class="flex items-center justify-between">
              <span class="text-slate-400">Fill Opacity</span>
              <span class="font-mono-num text-slate-200">
                {{ Math.round(store.reservoirOpacity * 100) }}%
              </span>
            </div>
            <input
              v-model.number="store.reservoirOpacity"
              type="range"
              min="0"
              max="0.8"
              step="0.02"
              class="ws-slider w-full"
              :disabled="!store.showReservoir"
            />
          </div>

          <p v-if="reservoirMetaLine" class="text-[10px] text-slate-500 leading-snug">
            {{ reservoirMetaLine }}
          </p>
          <p class="text-[10px] text-slate-600 leading-snug">
            Volve Eclipse grid (depth) tied to the interpreted Top Hugin TWT horizon; base from
            the well-derived Hugin interval velocity. Equinor Volve data licence.
          </p>
        </div>
      </section>

      <hr v-if="store.hasHorizons" class="border-slate-800/80" />

      <!-- Latent Exploration -->
      <section class="space-y-2.5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-300">
            <Sparkles class="w-3.5 h-3.5 text-cyan-400" />
            <span>Latent Exploration</span>
          </div>
          <button
            type="button"
            class="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-slate-900 border border-slate-800 text-cyan-300 hover:bg-slate-800 transition-colors"
            @click="store.jobsModalOpen = true"
          >
            <Cpu class="w-3 h-3" />
            <span>Embed / Jobs</span>
          </button>
        </div>

        <!-- Embedding Set Selector -->
        <div v-if="survey && survey.embedding_sets.length > 0" class="space-y-2">
          <label class="block space-y-1">
            <span class="text-[10px] text-slate-500">Embedding Lattice</span>
            <select
              :value="store.activeEmbeddingSetId ?? ''"
              class="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              @change="onEmbeddingSetChange"
            >
              <option v-for="es in survey.embedding_sets" :key="es.id" :value="es.id">
                {{ es.label }} ({{ es.patch_grid_shape.join('×') }})
              </option>
            </select>
          </label>

          <!-- Overlay Source Toggle -->
          <div class="grid grid-cols-3 rounded bg-slate-900 border border-slate-800 p-0.5 text-[11px]">
            <button
              type="button"
              class="py-1 rounded font-medium transition-colors"
              :class="
                store.overlaySource === 'similarity'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 border border-transparent'
              "
              @click="store.setOverlaySource('similarity')"
            >
              Similarity
            </button>
            <button
              type="button"
              class="py-1 rounded font-medium transition-colors disabled:opacity-40"
              :disabled="!embSet || embSet.available_attributes.length === 0"
              :class="
                store.overlaySource === 'attribute'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 border border-transparent'
              "
              @click="store.setOverlaySource('attribute')"
            >
              Attribute
            </button>
            <button
              type="button"
              class="py-1 rounded font-medium transition-colors"
              :class="
                store.overlaySource === 'off'
                  ? 'bg-slate-800 text-slate-200'
                  : 'text-slate-400 hover:text-slate-200'
              "
              @click="store.setOverlaySource('off')"
            >
              Off
            </button>
          </div>

          <!-- Similarity Mode Controls -->
          <div
            v-if="store.overlaySource === 'similarity' && embSet"
            class="rounded border border-slate-800/90 bg-slate-900/50 p-2 space-y-2 text-[11px]"
          >
            <label class="block space-y-1">
              <span class="text-[10px] text-slate-500">Similarity Mode</span>
              <select
                v-model="store.similarityMode"
                class="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
                @change="onModeChange"
              >
                <option v-for="m in embSet.available_modes" :key="m" :value="m">
                  {{ formatModeLabel(m) }}
                </option>
              </select>
            </label>

            <label class="flex items-center gap-2 text-slate-300 cursor-pointer">
              <input
                v-model="store.whitenChannels"
                type="checkbox"
                class="accent-cyan-400 rounded"
                @change="onWhitenChange"
              />
              <span>Whiten Channels</span>
            </label>

            <div class="space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-slate-400">Similarity Threshold</span>
                <span class="font-mono-num text-cyan-300">
                  {{ store.similarityThreshold.toFixed(2) }}
                </span>
              </div>
              <input
                v-model.number="store.similarityThreshold"
                type="range"
                min="-0.5"
                max="0.99"
                step="0.01"
                class="ws-slider w-full"
              />
            </div>

            <div class="space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-slate-400">Overlay Opacity</span>
                <span class="font-mono-num text-slate-200">
                  {{ Math.round(store.overlayOpacity * 100) }}%
                </span>
              </div>
              <input
                v-model.number="store.overlayOpacity"
                type="range"
                min="0"
                max="1"
                step="0.02"
                class="ws-slider w-full"
              />
            </div>

            <!-- Guard & Threshold Notices -->
            <div
              v-if="store.hasOnlyExcludeFeatures"
              class="rounded bg-amber-950/60 border border-amber-500/40 px-2 py-1.5 text-[10px] text-amber-200"
            >
              Enable or add at least one Inclusion (+) feature to compute similarity.
            </div>
            <div
              v-else-if="
                store.similarityResult &&
                store.similarityResult.similarity_max < store.similarityThreshold
              "
              class="rounded bg-amber-950/60 border border-amber-500/40 px-2 py-1.5 text-[10px] text-amber-200"
            >
              Max similarity ({{ store.similarityResult.similarity_max.toFixed(2) }}) is below
              threshold ({{ store.similarityThreshold.toFixed(2) }}) — lower threshold to view.
            </div>
          </div>

          <!-- Feature Picking & Refinement Panel -->
          <div class="rounded border border-slate-800/90 bg-slate-900/50 p-2 space-y-2 text-[11px]">
            <div class="flex items-center justify-between">
              <span class="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-slate-300">
                <Target class="w-3 h-3 text-cyan-400" />
                <span>Feature Picking &amp; Refinement</span>
              </span>
              <button
                v-if="store.activeGeometryMode === 'single'"
                type="button"
                class="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] bg-slate-950 border border-slate-800 text-cyan-300 hover:border-cyan-500/50 transition-colors"
                title="Start a new Single pick instead of replacing the selected one"
                @click="store.startNewFeatureSlot()"
              >
                <Plus class="w-2.5 h-2.5" />
                <span>New</span>
              </button>
            </div>

            <!-- Geometry Mode segmented buttons -->
            <div class="grid grid-cols-4 rounded bg-slate-950 border border-slate-800 p-0.5 text-[10px]">
              <button
                v-for="gm in GEOM_MODES"
                :key="gm.id"
                type="button"
                class="py-1 rounded font-medium transition-colors"
                :class="
                  store.activeGeometryMode === gm.id
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                "
                @click="store.setGeometryMode(gm.id)"
              >
                {{ gm.label }}
              </button>
            </div>

            <!-- Role segmented toggle -->
            <div class="grid grid-cols-2 rounded bg-slate-950 border border-slate-800 p-0.5 text-[10px]">
              <button
                type="button"
                class="py-1 rounded font-medium transition-colors"
                :class="
                  store.activeFeatureRole === 'include'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                "
                @click="store.setFeatureRole('include')"
              >
                Include (+)
              </button>
              <button
                type="button"
                class="py-1 rounded font-medium transition-colors"
                :class="
                  store.activeFeatureRole === 'exclude'
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                "
                @click="store.setFeatureRole('exclude')"
              >
                Exclude (−)
              </button>
            </div>

            <!-- Active drawing banner -->
            <div
              v-if="store.activeDrawingFeature"
              class="flex items-center justify-between gap-2 rounded bg-amber-500/15 border border-amber-500/40 px-2 py-1 text-[10px] text-amber-200"
            >
              <span class="truncate">
                Drawing {{ formatGeomLabel(store.activeDrawingFeature.geometry_mode) }} on
                {{ formatPlaneBadge(store.activeDrawingFeature) }} ({{
                  store.activeDrawingFeature.picks.length
                }}
                picks) — Double-click or Finish
              </span>
              <button
                type="button"
                class="shrink-0 px-1.5 py-0.5 rounded bg-amber-500/30 hover:bg-amber-500/50 text-amber-100 font-semibold"
                @click="store.finishActiveDrawing()"
              >
                Finish
              </button>
            </div>

            <!-- Exclusion Weight (alpha) -->
            <div class="space-y-1 pt-0.5">
              <div class="flex items-center justify-between text-[10px]">
                <span class="text-slate-400">Exclusion Weight (α)</span>
                <span class="font-mono-num text-rose-300">
                  {{ store.exclusionWeight.toFixed(2) }}
                </span>
              </div>
              <input
                :value="store.exclusionWeight"
                type="range"
                min="0"
                max="2"
                step="0.05"
                class="ws-slider w-full"
                @input="store.setExclusionWeight(Number(($event.target as HTMLInputElement).value))"
              />
              <div
                v-if="
                  store.similarityResult &&
                  store.similarityResult.included_patch_count !== undefined
                "
                class="font-mono-num text-[9px] text-slate-400 flex items-center justify-between"
              >
                <span>
                  {{ store.similarityResult.included_patch_count }} included ·
                  {{ store.similarityResult.excluded_patch_count ?? 0 }} excluded patches
                </span>
                <span v-if="store.similarityResult.overlap_resolved_count">
                  {{ store.similarityResult.overlap_resolved_count }} overlap
                </span>
              </div>
            </div>

            <!-- Picked Features List -->
            <div class="space-y-1 pt-1 border-t border-slate-800/80">
              <div class="flex items-center justify-between text-[10px]">
                <span class="text-slate-400">
                  Picked Features ({{ store.pickedFeatures.length }})
                </span>
                <button
                  v-if="store.pickedFeatures.length > 0"
                  type="button"
                  class="text-slate-500 hover:text-rose-300 transition-colors"
                  @click="store.clearAllFeatures()"
                >
                  Clear All
                </button>
              </div>

              <div
                v-if="store.pickedFeatures.length > 0"
                class="max-h-40 overflow-y-auto space-y-1 pr-0.5"
              >
                <div
                  v-for="feat in store.pickedFeatures"
                  :key="feat.id"
                  class="flex items-center gap-1.5 px-1.5 py-1 rounded border text-[10px] transition-colors"
                  :class="
                    store.selectedFeatureId === feat.id
                      ? 'bg-slate-900 border-cyan-500/50'
                      : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700'
                  "
                >
                  <!-- Role flip pill -->
                  <button
                    type="button"
                    class="w-5 h-5 rounded font-bold flex items-center justify-center shrink-0 border transition-colors"
                    :class="
                      feat.role === 'include'
                        ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30'
                        : 'bg-rose-500/20 border-rose-500/40 text-rose-300 hover:bg-rose-500/30'
                    "
                    title="Click to flip Include / Exclude"
                    @click.stop="store.toggleFeatureRole(feat.id)"
                  >
                    {{ feat.role === 'include' ? '+' : '−' }}
                  </button>

                  <!-- Row jump button -->
                  <button
                    type="button"
                    class="flex-1 min-w-0 text-left flex items-center justify-between gap-1"
                    :class="feat.enabled ? 'text-slate-200' : 'text-slate-500 line-through'"
                    @click="store.jumpToFeature(feat.id)"
                  >
                    <span class="truncate font-medium">
                      {{ formatGeomLabel(feat.geometry_mode) }}
                      <span class="text-slate-500 font-mono-num">
                        · {{ feat.picks.length }}p
                      </span>
                    </span>
                    <span class="font-mono-num text-[9px] px-1 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300 shrink-0">
                      {{ formatPlaneBadge(feat) }}
                    </span>
                  </button>

                  <!-- Enable / Mute -->
                  <button
                    type="button"
                    class="p-0.5 rounded text-slate-400 hover:text-slate-100 shrink-0"
                    :title="feat.enabled ? 'Mute feature' : 'Enable feature'"
                    @click.stop="store.toggleFeatureEnabled(feat.id)"
                  >
                    <Eye v-if="feat.enabled" class="w-3 h-3" />
                    <EyeOff v-else class="w-3 h-3 text-slate-600" />
                  </button>

                  <!-- Delete -->
                  <button
                    type="button"
                    class="p-0.5 rounded text-slate-500 hover:text-rose-400 shrink-0"
                    title="Delete feature"
                    @click.stop="store.deleteFeature(feat.id)"
                  >
                    <Trash2 class="w-3 h-3" />
                  </button>
                </div>
              </div>
              <div v-else class="text-[10px] text-slate-500 py-1">
                No features picked yet.
              </div>
            </div>

            <div class="text-[9px] text-slate-500 leading-relaxed pt-0.5 border-t border-slate-800/60">
              Click IL/XL plane to pick · Double-click to close · Drag handle to move ·
              Middle-click handle to delete · Click line to insert
            </div>
          </div>

          <!-- Precomputed Attribute Controls -->
          <div
            v-if="store.overlaySource === 'attribute' && embSet"
            class="rounded border border-slate-800/90 bg-slate-900/50 p-2 space-y-2 text-[11px]"
          >
            <label class="block space-y-1">
              <span class="text-[10px] text-slate-500">Precomputed Attribute</span>
              <select
                v-model="store.selectedAttribute"
                class="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
                @change="onAttributeChange"
              >
                <option v-for="attr in embSet.available_attributes" :key="attr" :value="attr">
                  {{ formatAttrLabel(attr) }}
                </option>
              </select>
            </label>

            <div v-if="store.attributeResult" class="space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-slate-400">Threshold</span>
                <span class="font-mono-num text-cyan-300">
                  {{ store.attributeThreshold.toPrecision(3) }}
                </span>
              </div>
              <input
                v-model.number="store.attributeThreshold"
                type="range"
                :min="store.attributeResult.value_min"
                :max="store.attributeResult.value_max"
                :step="attrStep"
                class="ws-slider w-full"
              />
            </div>

            <div class="space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-slate-400">Overlay Opacity</span>
                <span class="font-mono-num text-slate-200">
                  {{ Math.round(store.overlayOpacity * 100) }}%
                </span>
              </div>
              <input
                v-model.number="store.overlayOpacity"
                type="range"
                min="0"
                max="1"
                step="0.02"
                class="ws-slider w-full"
              />
            </div>
          </div>

          <!-- Colorbar Legend -->
          <div
            v-if="store.activeOverlay"
            class="rounded border border-slate-800/80 bg-slate-900/40 px-2 py-1.5 space-y-1"
          >
            <div
              class="h-2 rounded-sm border border-slate-700/60"
              :style="{ background: overlayGradient }"
            />
            <div class="flex items-center justify-between font-mono-num text-[9px] text-slate-400">
              <span>{{ Math.min(store.activeOverlay.threshold, store.activeOverlay.max).toPrecision(3) }}</span>
              <span class="text-slate-500">{{ store.activeOverlay.label }}</span>
              <span>{{ store.activeOverlay.max.toPrecision(3) }}</span>
            </div>
          </div>
        </div>

        <!-- No embeddings empty card -->
        <div
          v-else
          class="rounded border border-dashed border-slate-800 bg-slate-900/40 p-3 text-center space-y-2"
        >
          <p class="text-[11px] text-slate-400">
            No Embedding Lattice generated for this Survey yet.
          </p>
          <button
            type="button"
            class="w-full py-1.5 rounded text-xs font-medium bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25 transition-colors"
            @click="store.jobsModalOpen = true"
          >
            Generate Demo / NCS Embeddings
          </button>
        </div>

        <!-- Offset Trajectory Card -->
        <TrajectoryCard />

        <!-- Top Analogue Matches -->
        <div
          v-if="store.similarityResult && store.similarityResult.top_matches.length > 0"
          class="rounded border border-slate-800 bg-slate-950/90 p-2 space-y-1.5"
        >
          <div class="flex items-center justify-between">
            <span class="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
              Top Analogue Matches
            </span>
            <span class="font-mono-num text-[10px] text-slate-500">
              {{ store.similarityResult.top_matches.length }} patches
            </span>
          </div>

          <div class="max-h-48 overflow-y-auto space-y-1 pr-0.5">
            <button
              v-for="(m, idx) in store.similarityResult.top_matches"
              :key="`${m.patch_idx.join('-')}-${idx}`"
              type="button"
              class="w-full flex items-center justify-between px-2 py-1 rounded text-[11px] bg-slate-900/70 hover:bg-slate-800 border border-slate-800/80 hover:border-cyan-500/40 transition-colors text-left"
              @click="jumpToMatch(m.il_idx, m.xl_idx, m.sample_idx)"
            >
              <span class="font-mono-num text-slate-300">
                <span class="text-slate-500 mr-1">#{{ idx + 1 }}</span>
                IL {{ m.il }} · XL {{ m.xl }} · {{ m.sample }}ms
              </span>
              <span class="font-mono-num text-emerald-300 font-semibold">
                {{ m.score.toFixed(3) }}
              </span>
            </button>
          </div>
        </div>
      </section>
    </div>
  </aside>
</template>
