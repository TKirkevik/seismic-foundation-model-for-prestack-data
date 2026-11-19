<script setup lang="ts">
import { computed, onMounted } from 'vue'
import {
  Activity,
  AlertTriangle,
  Box,
  Columns2,
  Cpu,
  LayoutGrid,
  PanelLeftClose,
  PanelLeftOpen,
  Rows2,
  Square,
  X,
} from 'lucide-vue-next'
import type { LayoutMode } from './api/types'
import ControlSidebar from './components/ControlSidebar.vue'
import JobsModal from './components/JobsModal.vue'
import WorkspaceLayout from './components/WorkspaceLayout.vue'
import { useWorkbenchStore } from './stores/workbench'

const store = useWorkbenchStore()

const survey = computed(() => store.activeSurvey)

const LAYOUT_BUTTONS: Array<{ id: LayoutMode; label: string; group: 'focus' | 'split' }> = [
  { id: 'focus-section', label: 'Section', group: 'focus' },
  { id: 'focus-cube', label: '3D Cube', group: 'focus' },
  { id: 'focus-gather', label: 'Gather Panel', group: 'focus' },
  { id: 'split-2h', label: 'Split 2H', group: 'split' },
  { id: 'split-2v', label: 'Split 2V', group: 'split' },
  { id: 'workbench-3', label: 'Workbench 3-Pane', group: 'split' },
]

onMounted(() => {
  void store.initialize()
})
</script>

<template>
  <div class="flex flex-col w-screen h-screen bg-slate-950 text-slate-100 overflow-hidden">
    <!-- Top Workstation Bar -->
    <header
      class="h-11 px-3 bg-slate-950 border-b border-slate-800/90 flex items-center gap-3 shrink-0"
    >
      <!-- Sidebar toggle + Title -->
      <div class="flex items-center gap-2 shrink-0">
        <button
          type="button"
          class="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 hover:border-slate-700 transition-colors"
          :title="store.sidebarCollapsed ? 'Expand controls sidebar' : 'Collapse controls sidebar'"
          @click="store.sidebarCollapsed = !store.sidebarCollapsed"
        >
          <PanelLeftOpen v-if="store.sidebarCollapsed" class="w-4 h-4" />
          <PanelLeftClose v-else class="w-4 h-4" />
        </button>

        <div class="flex items-center gap-2">
          <Activity class="w-4 h-4 text-cyan-400" />
          <span class="text-xs font-semibold tracking-wide text-slate-100 hidden md:inline">
            Pre-Stack Latent Workbench
          </span>
        </div>

        <span
          v-if="survey"
          class="hidden xl:inline-block font-mono-num text-[10px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400"
        >
          {{ survey.shape.join(' × ') }}
        </span>
      </div>

      <!-- Synchronized 4D Coordinate Readout -->
      <div
        v-if="survey"
        class="hidden lg:flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-900/90 border border-slate-800 font-mono-num text-[11px]"
      >
        <span class="text-slate-400">
          IL <b class="text-cyan-300">{{ store.currentInlineLabel }}</b>
        </span>
        <span class="text-slate-700">·</span>
        <span class="text-slate-400">
          XL <b class="text-cyan-300">{{ store.currentXlineLabel }}</b>
        </span>
        <span class="text-slate-700">·</span>
        <span class="text-slate-400">
          T <b class="text-cyan-300">{{ store.currentSampleLabel }}</b> ms
        </span>
        <span class="text-slate-700">·</span>
        <span class="text-slate-400">
          Off <b class="text-amber-300">{{ store.currentOffsetLabel }}</b> m
        </span>
      </div>

      <!-- Flexible Layout Switcher (Single-View Focus Tabs + Split Layouts) -->
      <div class="flex items-center gap-1.5 mx-auto">
        <div class="flex items-center rounded bg-slate-900 border border-slate-800 p-0.5">
          <button
            v-for="btn in LAYOUT_BUTTONS.filter((b) => b.group === 'focus')"
            :key="btn.id"
            type="button"
            class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium transition-colors"
            :class="
              store.layoutMode === btn.id && !store.maximizedPane
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            "
            @click="store.setLayoutMode(btn.id)"
          >
            <Square v-if="btn.id === 'focus-section'" class="w-3 h-3" />
            <Box v-else-if="btn.id === 'focus-cube'" class="w-3 h-3" />
            <Activity v-else class="w-3 h-3" />
            <span>{{ btn.label }}</span>
          </button>
        </div>

        <div class="flex items-center rounded bg-slate-900 border border-slate-800 p-0.5">
          <button
            v-for="btn in LAYOUT_BUTTONS.filter((b) => b.group === 'split')"
            :key="btn.id"
            type="button"
            class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium transition-colors"
            :class="
              store.layoutMode === btn.id && !store.maximizedPane
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            "
            @click="store.setLayoutMode(btn.id)"
          >
            <Columns2 v-if="btn.id === 'split-2h'" class="w-3 h-3" />
            <Rows2 v-else-if="btn.id === 'split-2v'" class="w-3 h-3" />
            <LayoutGrid v-else class="w-3 h-3" />
            <span class="hidden sm:inline">{{ btn.label }}</span>
          </button>
        </div>
      </div>

      <!-- Right Status: Device Badge + Jobs Button -->
      <div class="flex items-center gap-2 shrink-0">
        <div
          v-if="store.health"
          class="hidden xl:flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono-num text-[10px] text-slate-300"
          :title="`PyTorch ${store.health.torch_version} | Weights: ${store.health.weights_available}`"
        >
          <span
            class="w-1.5 h-1.5 rounded-full"
            :class="store.health.status === 'ok' ? 'bg-emerald-400' : 'bg-amber-400'"
          />
          <span>{{ store.health.device.toUpperCase() }}: {{ store.health.device_name }}</span>
        </div>

        <button
          type="button"
          class="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border transition-colors"
          :class="
            store.hasActiveJobs
              ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-200'
              : 'bg-slate-900 border-slate-800 text-slate-200 hover:bg-slate-800'
          "
          @click="store.jobsModalOpen = true"
        >
          <Cpu class="w-3.5 h-3.5" :class="{ 'animate-spin text-cyan-300': store.hasActiveJobs }" />
          <span>Jobs</span>
          <span
            v-if="store.hasActiveJobs"
            class="font-mono-num text-[10px] px-1 rounded bg-cyan-500/30 text-cyan-100"
          >
            {{ store.jobs.filter((j) => j.status === 'running' || j.status === 'queued').length }}
          </span>
        </button>
      </div>
    </header>

    <!-- Non-blocking API error banner -->
    <div
      v-if="store.apiError"
      class="px-3 py-1.5 bg-rose-950/70 border-b border-rose-500/40 text-xs text-rose-200 flex items-center justify-between shrink-0"
    >
      <div class="flex items-center gap-2">
        <AlertTriangle class="w-3.5 h-3.5 text-rose-400 shrink-0" />
        <span>{{ store.apiError }}</span>
      </div>
      <button
        type="button"
        class="p-0.5 rounded hover:bg-rose-900/60"
        @click="store.apiError = null"
      >
        <X class="w-3.5 h-3.5" />
      </button>
    </div>

    <!-- Main Body: Collapsible Sidebar + Flexible Workspace -->
    <div class="flex flex-1 min-h-0 overflow-hidden">
      <ControlSidebar v-if="!store.sidebarCollapsed" />
      <WorkspaceLayout />
    </div>

    <!-- Pipeline Jobs Modal -->
    <JobsModal v-if="store.jobsModalOpen" />
  </div>
</template>
