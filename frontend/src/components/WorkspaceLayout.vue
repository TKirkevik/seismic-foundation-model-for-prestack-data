<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { Maximize2, Minimize2 } from 'lucide-vue-next'
import type { ViewerType } from '../api/types'
import { useWorkbenchStore } from '../stores/workbench'
import GatherPanelViewer from './GatherPanelViewer.vue'
import SectionViewer from './SectionViewer.vue'
import Volume3DViewer from './Volume3DViewer.vue'

const store = useWorkbenchStore()

const rootRef = ref<HTMLDivElement | null>(null)

// Resizable splitter fractions
const split2H = ref(0.5)
const split2V = ref(0.58)
const split3TopH = ref(0.5)
const split3V = ref(0.62)

type DragTarget = 'split2H' | 'split2V' | 'split3TopH' | 'split3V' | null
let activeDrag: DragTarget = null

const VIEW_OPTIONS: Array<{ id: ViewerType; label: string }> = [
  { id: 'section', label: 'Section Viewer' },
  { id: 'cube', label: '3D Cube Viewer' },
  { id: 'gather', label: 'Gather Panel' },
]

function startDrag(target: DragTarget, e: PointerEvent) {
  e.preventDefault()
  activeDrag = target
  window.addEventListener('pointermove', onDragMove)
  window.addEventListener('pointerup', onDragEnd)
}

function onDragMove(e: PointerEvent) {
  if (!activeDrag || !rootRef.value) return
  const rect = rootRef.value.getBoundingClientRect()
  const rx = Math.max(0.18, Math.min(0.82, (e.clientX - rect.left) / Math.max(1, rect.width)))
  const ry = Math.max(0.18, Math.min(0.82, (e.clientY - rect.top) / Math.max(1, rect.height)))

  if (activeDrag === 'split2H') split2H.value = rx
  else if (activeDrag === 'split2V') split2V.value = ry
  else if (activeDrag === 'split3TopH') split3TopH.value = rx
  else if (activeDrag === 'split3V') split3V.value = ry
}

function onDragEnd() {
  activeDrag = null
  window.removeEventListener('pointermove', onDragMove)
  window.removeEventListener('pointerup', onDragEnd)
}

onBeforeUnmount(() => {
  onDragEnd()
})

const focusViewer = computed<ViewerType | null>(() => {
  if (store.layoutMode === 'focus-section') return 'section'
  if (store.layoutMode === 'focus-cube') return 'cube'
  if (store.layoutMode === 'focus-gather') return 'gather'
  return null
})

function onSelectPaneView(pane: 'a' | 'b' | 'c', e: Event) {
  const view = (e.target as HTMLSelectElement).value as ViewerType
  store.setPaneView(pane, view)
}

function onFocusViewSwitch(e: Event) {
  const view = (e.target as HTMLSelectElement).value as ViewerType
  if (view === 'section') store.setLayoutMode('focus-section')
  else if (view === 'cube') store.setLayoutMode('focus-cube')
  else store.setLayoutMode('focus-gather')
}
</script>

<template>
  <div ref="rootRef" class="flex-1 min-w-0 min-h-0 h-full bg-slate-950 overflow-hidden relative">
    <!-- 1. Single-View Focus Mode -->
    <div v-if="focusViewer" class="flex flex-col w-full h-full">
      <div
        class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0"
      >
        <div class="flex items-center gap-2">
          <span class="w-1.5 h-1.5 rounded-full bg-cyan-400" />
          <select
            :value="focusViewer"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onFocusViewSwitch"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <span class="text-[10px] font-mono-num text-slate-500">100% Focus</span>
        </div>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          title="Return to Workbench 3-Pane"
          @click="store.setLayoutMode('workbench-3')"
        >
          <Minimize2 class="w-3.5 h-3.5" />
          <span>Split View</span>
        </button>
      </div>

      <div class="flex-1 min-h-0">
        <SectionViewer v-if="focusViewer === 'section'" initial-axis="inline" />
        <Volume3DViewer v-else-if="focusViewer === 'cube'" />
        <GatherPanelViewer v-else />
      </div>
    </div>

    <!-- 2. Maximized Pane from a Split Layout -->
    <div v-else-if="store.maximizedPane" class="flex flex-col w-full h-full">
      <div
        class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0"
      >
        <div class="flex items-center gap-2">
          <span class="w-1.5 h-1.5 rounded-full bg-cyan-400" />
          <select
            :value="store.paneViews[store.maximizedPane]"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onSelectPaneView(store.maximizedPane, $event)"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <span class="text-[10px] font-mono-num text-cyan-400/80">Maximized</span>
        </div>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25 transition-colors"
          title="Restore split layout"
          @click="store.toggleMaximizePane(store.maximizedPane)"
        >
          <Minimize2 class="w-3.5 h-3.5" />
          <span>Restore</span>
        </button>
      </div>

      <div class="flex-1 min-h-0">
        <SectionViewer
          v-if="store.paneViews[store.maximizedPane] === 'section'"
          :initial-axis="store.maximizedPane === 'b' ? 'xline' : 'inline'"
        />
        <Volume3DViewer v-else-if="store.paneViews[store.maximizedPane] === 'cube'" />
        <GatherPanelViewer v-else />
      </div>
    </div>

    <!-- 3. Split 2H (Side-by-Side) -->
    <div v-else-if="store.layoutMode === 'split-2h'" class="flex w-full h-full min-h-0">
      <div class="flex flex-col min-w-0 min-h-0 h-full" :style="{ width: `${split2H * 100}%` }">
        <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
          <select
            :value="store.paneViews.a"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onSelectPaneView('a', $event)"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <button
            type="button"
            class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            title="Maximize pane"
            @click="store.toggleMaximizePane('a')"
          >
            <Maximize2 class="w-3.5 h-3.5" />
          </button>
        </div>
        <div class="flex-1 min-h-0">
          <SectionViewer v-if="store.paneViews.a === 'section'" initial-axis="inline" />
          <Volume3DViewer v-else-if="store.paneViews.a === 'cube'" />
          <GatherPanelViewer v-else />
        </div>
      </div>

      <div
        class="w-1.5 shrink-0 bg-slate-900 hover:bg-cyan-500/60 border-x border-slate-800 cursor-col-resize transition-colors"
        @pointerdown="startDrag('split2H', $event)"
      />

      <div class="flex flex-col flex-1 min-w-0 min-h-0 h-full">
        <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
          <select
            :value="store.paneViews.b"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onSelectPaneView('b', $event)"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <button
            type="button"
            class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            title="Maximize pane"
            @click="store.toggleMaximizePane('b')"
          >
            <Maximize2 class="w-3.5 h-3.5" />
          </button>
        </div>
        <div class="flex-1 min-h-0">
          <SectionViewer v-if="store.paneViews.b === 'section'" initial-axis="xline" />
          <Volume3DViewer v-else-if="store.paneViews.b === 'cube'" />
          <GatherPanelViewer v-else />
        </div>
      </div>
    </div>

    <!-- 4. Split 2V (Top / Bottom) -->
    <div v-else-if="store.layoutMode === 'split-2v'" class="flex flex-col w-full h-full min-h-0">
      <div class="flex flex-col min-w-0 min-h-0 w-full" :style="{ height: `${split2V * 100}%` }">
        <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
          <select
            :value="store.paneViews.a"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onSelectPaneView('a', $event)"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <button
            type="button"
            class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            title="Maximize pane"
            @click="store.toggleMaximizePane('a')"
          >
            <Maximize2 class="w-3.5 h-3.5" />
          </button>
        </div>
        <div class="flex-1 min-h-0">
          <SectionViewer v-if="store.paneViews.a === 'section'" initial-axis="inline" />
          <Volume3DViewer v-else-if="store.paneViews.a === 'cube'" />
          <GatherPanelViewer v-else />
        </div>
      </div>

      <div
        class="h-1.5 shrink-0 bg-slate-900 hover:bg-cyan-500/60 border-y border-slate-800 cursor-row-resize transition-colors"
        @pointerdown="startDrag('split2V', $event)"
      />

      <div class="flex flex-col flex-1 min-w-0 min-h-0 w-full">
        <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
          <select
            :value="store.paneViews.c"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onSelectPaneView('c', $event)"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <button
            type="button"
            class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            title="Maximize pane"
            @click="store.toggleMaximizePane('c')"
          >
            <Maximize2 class="w-3.5 h-3.5" />
          </button>
        </div>
        <div class="flex-1 min-h-0">
          <SectionViewer v-if="store.paneViews.c === 'section'" initial-axis="sample" />
          <Volume3DViewer v-else-if="store.paneViews.c === 'cube'" />
          <GatherPanelViewer v-else />
        </div>
      </div>
    </div>

    <!-- 5. Workbench 3-Pane (Top-Left + Top-Right + Bottom Full-Width) -->
    <div v-else class="flex flex-col w-full h-full min-h-0">
      <!-- Top Row -->
      <div class="flex w-full min-h-0" :style="{ height: `${split3V * 100}%` }">
        <!-- Top-Left Pane A -->
        <div class="flex flex-col min-w-0 min-h-0 h-full" :style="{ width: `${split3TopH * 100}%` }">
          <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
            <select
              :value="store.paneViews.a"
              class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
              @change="onSelectPaneView('a', $event)"
            >
              <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
                {{ opt.label }}
              </option>
            </select>
            <button
              type="button"
              class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
              title="Maximize pane"
              @click="store.toggleMaximizePane('a')"
            >
              <Maximize2 class="w-3.5 h-3.5" />
            </button>
          </div>
          <div class="flex-1 min-h-0">
            <SectionViewer v-if="store.paneViews.a === 'section'" initial-axis="inline" />
            <Volume3DViewer v-else-if="store.paneViews.a === 'cube'" />
            <GatherPanelViewer v-else />
          </div>
        </div>

        <div
          class="w-1.5 shrink-0 bg-slate-900 hover:bg-cyan-500/60 border-x border-slate-800 cursor-col-resize transition-colors"
          @pointerdown="startDrag('split3TopH', $event)"
        />

        <!-- Top-Right Pane B -->
        <div class="flex flex-col flex-1 min-w-0 min-h-0 h-full">
          <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
            <select
              :value="store.paneViews.b"
              class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
              @change="onSelectPaneView('b', $event)"
            >
              <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
                {{ opt.label }}
              </option>
            </select>
            <button
              type="button"
              class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
              title="Maximize pane"
              @click="store.toggleMaximizePane('b')"
            >
              <Maximize2 class="w-3.5 h-3.5" />
            </button>
          </div>
          <div class="flex-1 min-h-0">
            <SectionViewer v-if="store.paneViews.b === 'section'" initial-axis="xline" />
            <Volume3DViewer v-else-if="store.paneViews.b === 'cube'" />
            <GatherPanelViewer v-else />
          </div>
        </div>
      </div>

      <div
        class="h-1.5 shrink-0 bg-slate-900 hover:bg-cyan-500/60 border-y border-slate-800 cursor-row-resize transition-colors"
        @pointerdown="startDrag('split3V', $event)"
      />

      <!-- Bottom Full-Width Strip Pane C -->
      <div class="flex flex-col flex-1 min-w-0 min-h-0 w-full">
        <div class="h-8 px-2.5 bg-slate-900/95 border-b border-slate-800 flex items-center justify-between shrink-0">
          <select
            :value="store.paneViews.c"
            class="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-xs font-medium text-slate-100 focus:outline-none focus:border-cyan-500"
            @change="onSelectPaneView('c', $event)"
          >
            <option v-for="opt in VIEW_OPTIONS" :key="opt.id" :value="opt.id">
              {{ opt.label }}
            </option>
          </select>
          <button
            type="button"
            class="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            title="Maximize pane"
            @click="store.toggleMaximizePane('c')"
          >
            <Maximize2 class="w-3.5 h-3.5" />
          </button>
        </div>
        <div class="flex-1 min-h-0">
          <SectionViewer v-if="store.paneViews.c === 'section'" initial-axis="sample" />
          <Volume3DViewer v-else-if="store.paneViews.c === 'cube'" />
          <GatherPanelViewer v-else />
        </div>
      </div>
    </div>
  </div>
</template>
