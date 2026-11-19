<script setup lang="ts">
import { computed } from 'vue'
import { Cpu, Database, RefreshCw, Sparkles, Target } from 'lucide-vue-next'
import { useWorkbenchStore } from '../stores/workbench'
import { colormapGradientCss, getOverlayLUT } from '../utils/colormaps'
import TrajectoryCard from './TrajectoryCard.vue'

const store = useWorkbenchStore()

const survey = computed(() => store.activeSurvey)
const embSet = computed(() => store.activeEmbeddingSet)

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
  if (store.featurePick && store.overlaySource === 'similarity') {
    void store.runSimilarityQuery()
  }
}

function onWhitenChange() {
  if (store.featurePick && store.overlaySource === 'similarity') {
    void store.runSimilarityQuery()
  }
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
                min="0"
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

            <!-- Active Feature Pick Status -->
            <div class="pt-1 border-t border-slate-800/80 text-[10px]">
              <div v-if="store.featurePick" class="flex items-center justify-between text-slate-300">
                <span class="flex items-center gap-1 text-cyan-300">
                  <Target class="w-3 h-3" />
                  Feature Pick
                </span>
                <span class="font-mono-num">
                  IL {{ survey.ilines[store.featurePick.il_idx] ?? store.featurePick.il_idx }} ·
                  XL {{ survey.xlines[store.featurePick.xl_idx] ?? store.featurePick.xl_idx }} ·
                  {{ survey.samples[store.featurePick.sample_idx] ?? store.featurePick.sample_idx }}ms
                </span>
              </div>
              <div v-else class="text-slate-500">
                Click on any Section or Gather to set a Feature Pick.
              </div>
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
              <span>{{ store.activeOverlay.threshold.toPrecision(3) }}</span>
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
