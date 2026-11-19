<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Cpu, Play, RefreshCw, Square, Terminal, X, Zap } from 'lucide-vue-next'
import type { JobInfo } from '../api/types'
import { useWorkbenchStore } from '../stores/workbench'

const store = useWorkbenchStore()

const offsetsSlice = ref('0:100:10')
const numOverlapPatches = ref(0)
const batchSize = ref(8)
const autoSummarize = ref(true)
const excludeZeroOffset = ref(false)
const summarizeSetId = ref('ncs3d')
const launchError = ref<string | null>(null)
const launching = ref(false)

const survey = computed(() => store.activeSurvey)
const embeddingSets = computed(() => survey.value?.embedding_sets ?? [])

function jobTitle(job: JobInfo): string {
  return job.title || job.type || job.kind || job.id
}

function jobProgressPct(job: JobInfo): number {
  const p = job.progress ?? 0
  if (p <= 1 && p > 0) return Math.round(p * 100)
  return Math.max(0, Math.min(100, Math.round(p)))
}

function jobLogsText(job: JobInfo): string {
  if (job.log_tail) return job.log_tail
  if (job.logs && job.logs.length > 0) return job.logs.slice(-14).join('\n')
  return ''
}

function statusBadgeClass(status: string): string {
  if (status === 'completed') return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
  if (status === 'running') return 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40'
  if (status === 'queued') return 'bg-amber-500/15 text-amber-300 border-amber-500/40'
  if (status === 'failed') return 'bg-rose-500/15 text-rose-300 border-rose-500/40'
  return 'bg-slate-800 text-slate-300 border-slate-700'
}

async function runDemo() {
  if (!survey.value) return
  launching.value = true
  launchError.value = null
  try {
    await store.launchDemoJob({ survey_id: survey.value.id })
  } catch (err) {
    launchError.value = err instanceof Error ? err.message : String(err)
  } finally {
    launching.value = false
  }
}

async function runEmbed() {
  if (!survey.value) return
  launching.value = true
  launchError.value = null
  try {
    await store.launchEmbedJob({
      survey_id: survey.value.id,
      offsets: offsetsSlice.value.trim() || undefined,
      num_overlap_patches: numOverlapPatches.value,
      batch_size: batchSize.value,
      auto_summarize: autoSummarize.value,
    })
  } catch (err) {
    launchError.value = err instanceof Error ? err.message : String(err)
  } finally {
    launching.value = false
  }
}

async function runSummarize() {
  if (!survey.value) return
  launching.value = true
  launchError.value = null
  try {
    await store.launchSummarizeJob({
      survey_id: survey.value.id,
      embedding_set_id: summarizeSetId.value || store.activeEmbeddingSetId || undefined,
      exclude_zero_offset: excludeZeroOffset.value,
    })
  } catch (err) {
    launchError.value = err instanceof Error ? err.message : String(err)
  } finally {
    launching.value = false
  }
}

onMounted(() => {
  void store.refreshJobs()
  if (store.activeEmbeddingSetId) {
    summarizeSetId.value = store.activeEmbeddingSetId
  }
})
</script>

<template>
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4"
    @click.self="store.jobsModalOpen = false"
  >
    <div
      class="w-full max-w-4xl max-h-[88vh] bg-slate-950 border border-slate-800 rounded-lg shadow-2xl flex flex-col overflow-hidden"
    >
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 bg-slate-900/90 border-b border-slate-800">
        <div class="flex items-center gap-2.5">
          <Cpu class="w-4 h-4 text-cyan-400" />
          <div>
            <h2 class="text-sm font-semibold text-slate-100">
              Embedding &amp; Trajectory Pipeline Orchestrator
            </h2>
            <p class="text-[11px] text-slate-400">
              Survey: <span class="font-mono-num text-cyan-300">{{ survey?.name ?? 'None' }}</span>
              <span v-if="store.health" class="ml-2 text-slate-500">
                · {{ store.health.device.toUpperCase() }}: {{ store.health.device_name }}
                · Weights:
                <span :class="store.health.weights_available ? 'text-emerald-400' : 'text-amber-400'">
                  {{ store.health.weights_available ? 'Ready' : 'Missing' }}
                </span>
              </span>
            </p>
          </div>
        </div>

        <div class="flex items-center gap-2">
          <button
            type="button"
            class="flex items-center gap-1 px-2 py-1 rounded text-xs bg-slate-950 border border-slate-800 text-slate-300 hover:bg-slate-800"
            @click="store.refreshJobs()"
          >
            <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': store.jobsLoading }" />
            <span>Refresh</span>
          </button>
          <button
            type="button"
            class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800"
            @click="store.jobsModalOpen = false"
          >
            <X class="w-4 h-4" />
          </button>
        </div>
      </div>

      <!-- Body -->
      <div class="flex-1 overflow-y-auto p-4 space-y-4">
        <div
          v-if="launchError"
          class="rounded border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-300"
        >
          {{ launchError }}
        </div>

        <!-- 3 Pipeline Launchers -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <!-- 1. Fast Demo Embeddings -->
          <div class="rounded border border-slate-800 bg-slate-900/50 p-3 flex flex-col justify-between space-y-3">
            <div class="space-y-1">
              <div class="flex items-center gap-1.5 text-xs font-semibold text-amber-300">
                <Zap class="w-3.5 h-3.5" />
                <span>Fast Demo Embeddings</span>
              </div>
              <p class="text-[11px] text-slate-400 leading-relaxed">
                Generates a lightweight synthetic Embedding Lattice &amp; Offset Trajectory summary in seconds for immediate UI exploration without GPU inference.
              </p>
            </div>
            <button
              type="button"
              class="w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium bg-amber-500/15 border border-amber-500/40 text-amber-200 hover:bg-amber-500/25 disabled:opacity-40 transition-colors"
              :disabled="!survey || launching"
              @click="runDemo"
            >
              <Play class="w-3.5 h-3.5" />
              <span>Run Demo Generator</span>
            </button>
          </div>

          <!-- 2. NCS-3D Patch Embedder -->
          <div class="rounded border border-slate-800 bg-slate-900/50 p-3 flex flex-col justify-between space-y-2.5">
            <div class="space-y-2">
              <div class="flex items-center gap-1.5 text-xs font-semibold text-cyan-300">
                <Cpu class="w-3.5 h-3.5" />
                <span>NCS-3D Patch Embedder</span>
              </div>

              <div class="space-y-1.5 text-[11px]">
                <div>
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-slate-400">Offsets slice</span>
                    <div class="flex gap-1">
                      <button
                        type="button"
                        class="px-1.5 py-0.2 rounded bg-slate-950 border border-slate-800 text-[10px] text-slate-300 hover:border-cyan-500/40"
                        @click="offsetsSlice = '0:100:10'"
                      >
                        Fast (1/10)
                      </button>
                      <button
                        type="button"
                        class="px-1.5 py-0.2 rounded bg-slate-950 border border-slate-800 text-[10px] text-slate-300 hover:border-cyan-500/40"
                        @click="offsetsSlice = '0:100:1'"
                      >
                        Full
                      </button>
                    </div>
                  </div>
                  <input
                    v-model="offsetsSlice"
                    type="text"
                    placeholder="0:100:10"
                    class="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 font-mono-num text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
                  />
                </div>

                <div class="grid grid-cols-2 gap-2">
                  <label class="space-y-0.5">
                    <span class="text-[10px] text-slate-500">Overlap Patches</span>
                    <input
                      v-model.number="numOverlapPatches"
                      type="number"
                      min="0"
                      max="13"
                      class="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 font-mono-num text-xs text-slate-100"
                    />
                  </label>
                  <label class="space-y-0.5">
                    <span class="text-[10px] text-slate-500">Batch Size</span>
                    <input
                      v-model.number="batchSize"
                      type="number"
                      min="1"
                      max="64"
                      class="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 font-mono-num text-xs text-slate-100"
                    />
                  </label>
                </div>

                <label class="flex items-center gap-2 text-slate-300 cursor-pointer pt-0.5">
                  <input v-model="autoSummarize" type="checkbox" class="accent-cyan-400" />
                  <span>Auto-summarize Offset Trajectory</span>
                </label>
              </div>
            </div>

            <button
              type="button"
              class="w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium bg-cyan-500/15 border border-cyan-500/40 text-cyan-200 hover:bg-cyan-500/25 disabled:opacity-40 transition-colors"
              :disabled="!survey || launching"
              @click="runEmbed"
            >
              <Play class="w-3.5 h-3.5" />
              <span>Launch NCS-3D Embedder</span>
            </button>
          </div>

          <!-- 3. Offset Trajectory Summarizer -->
          <div class="rounded border border-slate-800 bg-slate-900/50 p-3 flex flex-col justify-between space-y-2.5">
            <div class="space-y-2">
              <div class="flex items-center gap-1.5 text-xs font-semibold text-emerald-300">
                <Terminal class="w-3.5 h-3.5" />
                <span>Offset Trajectory Summarizer</span>
              </div>
              <p class="text-[11px] text-slate-400 leading-relaxed">
                Computes zero-offset anchor, primary offset direction, and curvature basis from an existing Embedding Lattice.
              </p>

              <div class="space-y-1.5 text-[11px]">
                <label class="block space-y-0.5">
                  <span class="text-[10px] text-slate-500">Target Embedding Set</span>
                  <select
                    v-model="summarizeSetId"
                    class="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-100"
                  >
                    <option v-for="es in embeddingSets" :key="es.id" :value="es.id">
                      {{ es.label }}
                    </option>
                    <option v-if="embeddingSets.length === 0" value="ncs3d">ncs3d</option>
                  </select>
                </label>

                <label class="flex items-center gap-2 text-slate-300 cursor-pointer">
                  <input v-model="excludeZeroOffset" type="checkbox" class="accent-emerald-400" />
                  <span>Exclude ±50m near-zero offsets</span>
                </label>
              </div>
            </div>

            <button
              type="button"
              class="w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium bg-emerald-500/15 border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/25 disabled:opacity-40 transition-colors"
              :disabled="!survey || launching"
              @click="runSummarize"
            >
              <Play class="w-3.5 h-3.5" />
              <span>Run Summarizer</span>
            </button>
          </div>
        </div>

        <!-- Job Queue & Logs -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Background Jobs ({{ store.jobs.length }})
            </h3>
            <span v-if="store.hasActiveJobs" class="text-[11px] text-cyan-300 font-mono-num">
              Polling every 1.5s…
            </span>
          </div>

          <div v-if="store.jobs.length === 0" class="rounded border border-slate-800/80 bg-slate-900/30 p-4 text-center text-xs text-slate-500">
            No background pipeline jobs recorded in this backend session.
          </div>

          <div v-else class="space-y-2">
            <div
              v-for="job in store.jobs"
              :key="job.id"
              class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-2"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="flex items-center gap-2 min-w-0">
                  <span
                    class="px-1.5 py-0.5 rounded text-[10px] font-mono-num uppercase border"
                    :class="statusBadgeClass(job.status)"
                  >
                    {{ job.status }}
                  </span>
                  <span class="text-xs font-medium text-slate-100 truncate">
                    {{ jobTitle(job) }}
                  </span>
                  <span class="font-mono-num text-[10px] text-slate-500">
                    {{ job.id }}
                  </span>
                </div>

                <div class="flex items-center gap-2 shrink-0">
                  <span class="font-mono-num text-[11px] text-slate-300">
                    {{ jobProgressPct(job) }}%
                  </span>
                  <button
                    v-if="job.status === 'running' || job.status === 'queued'"
                    type="button"
                    class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-rose-500/15 border border-rose-500/40 text-rose-300 hover:bg-rose-500/25"
                    @click="store.cancelRunningJob(job.id)"
                  >
                    <Square class="w-3 h-3" />
                    <span>Cancel</span>
                  </button>
                </div>
              </div>

              <!-- Progress Bar -->
              <div class="h-1.5 w-full rounded-full bg-slate-950 overflow-hidden border border-slate-800/80">
                <div
                  class="h-full transition-all duration-300"
                  :class="
                    job.status === 'failed'
                      ? 'bg-rose-500'
                      : job.status === 'completed'
                        ? 'bg-emerald-500'
                        : 'bg-cyan-400'
                  "
                  :style="{ width: `${jobProgressPct(job)}%` }"
                />
              </div>

              <div v-if="job.stage || job.message || job.error" class="text-[11px]">
                <span v-if="job.stage" class="text-cyan-300 mr-1.5">[{{ job.stage }}]</span>
                <span v-if="job.message" class="text-slate-300">{{ job.message }}</span>
                <span v-if="job.error" class="text-rose-300 block mt-1">{{ job.error }}</span>
              </div>

              <pre
                v-if="jobLogsText(job)"
                class="max-h-32 overflow-y-auto rounded bg-slate-950 border border-slate-800/90 p-2 font-mono-num text-[10px] text-slate-400 whitespace-pre-wrap leading-relaxed"
              >{{ jobLogsText(job) }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
