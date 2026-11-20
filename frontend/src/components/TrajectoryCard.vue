<script setup lang="ts">
import { computed, ref } from 'vue'
import { useWorkbenchStore } from '../stores/workbench'

const store = useWorkbenchStore()

const hoveredIdx = ref<number | null>(null)

const trajectory = computed(
  () => store.focusTrajectory?.trajectory ?? store.similarityResult?.trajectory ?? null,
)
const survey = computed(() => store.activeSurvey)

const focusBadgeLabel = computed(() => {
  const fp = store.focusPick
  const s = survey.value
  if (!fp || !s) return null
  const il = s.ilines[fp.il_idx] ?? fp.il_idx
  const xl = s.xlines[fp.xl_idx] ?? fp.xl_idx
  const t = s.samples[fp.sample_idx] ?? fp.sample_idx
  return `IL ${il} · XL ${xl} · ${t}ms`
})

const SVG_W = 280
const SVG_H = 186
const PAD_L = 34
const PAD_R = 14
const PAD_T = 14
const PAD_B = 26

const bounds = computed(() => {
  const tr = trajectory.value
  if (!tr || !tr.available || tr.c1.length === 0) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1 }
  }
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (let i = 0; i < tr.c1.length; i++) {
    const x = tr.c1[i]
    const y = tr.c2[i]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  // Include origin slightly if close, and add 12% margin
  const spanX = Math.max(0.1, maxX - minX)
  const spanY = Math.max(0.1, maxY - minY)
  return {
    minX: minX - spanX * 0.14,
    maxX: maxX + spanX * 0.14,
    minY: minY - spanY * 0.16,
    maxY: maxY + spanY * 0.16,
  }
})

function mapX(x: number): number {
  const { minX, maxX } = bounds.value
  return PAD_L + ((x - minX) / Math.max(1e-6, maxX - minX)) * (SVG_W - PAD_L - PAD_R)
}

function mapY(y: number): number {
  const { minY, maxY } = bounds.value
  return SVG_H - PAD_B - ((y - minY) / Math.max(1e-6, maxY - minY)) * (SVG_H - PAD_T - PAD_B)
}

function offsetColor(off: number, minOff: number, maxOff: number): string {
  const span = Math.max(1, maxOff - minOff)
  const t = Math.max(0, Math.min(1, (off - minOff) / span))
  // Negative offset (cyan/blue) -> near zero (white/emerald) -> positive far (amber/rose)
  if (t < 0.5) {
    const u = t * 2
    const r = Math.round(56 + (226 - 56) * u)
    const g = Math.round(189 + (232 - 189) * u)
    const b = Math.round(248 + (240 - 248) * u)
    return `rgb(${r},${g},${b})`
  } else {
    const u = (t - 0.5) * 2
    const r = Math.round(226 + (249 - 226) * u)
    const g = Math.round(232 + (115 - 232) * u)
    const b = Math.round(240 + (22 - 240) * u)
    return `rgb(${r},${g},${b})`
  }
}

const points = computed(() => {
  const tr = trajectory.value
  if (!tr || !tr.available || !tr.offsets.length) return []
  const minOff = Math.min(...tr.offsets)
  const maxOff = Math.max(...tr.offsets)
  const activeOff = store.currentOffsetLabel

  let closestIdx = 0
  let closestDist = Infinity
  for (let i = 0; i < tr.offsets.length; i++) {
    const d = Math.abs(tr.offsets[i] - activeOff)
    if (d < closestDist) {
      closestDist = d
      closestIdx = i
    }
  }

  return tr.offsets.map((off, idx) => ({
    idx,
    off,
    c1: tr.c1[idx] ?? 0,
    c2: tr.c2[idx] ?? 0,
    sx: mapX(tr.c1[idx] ?? 0),
    sy: mapY(tr.c2[idx] ?? 0),
    color: offsetColor(off, minOff, maxOff),
    isActive: idx === closestIdx,
  }))
})

const polylinePoints = computed(() =>
  points.value.map((p) => `${p.sx.toFixed(1)},${p.sy.toFixed(1)}`).join(' '),
)

const zeroAxes = computed(() => {
  const { minX, maxX, minY, maxY } = bounds.value
  return {
    showX0: minX <= 0 && maxX >= 0,
    x0: mapX(0),
    showY0: minY <= 0 && maxY >= 0,
    y0: mapY(0),
  }
})

const planeVarPct = computed(() => {
  const av = trajectory.value?.axis_var
  if (!av) return null
  return (av[0] + av[1]) * 100
})

const hoveredPoint = computed(() => {
  if (hoveredIdx.value === null) return null
  return points.value[hoveredIdx.value] ?? null
})

function jumpToOffset(offVal: number) {
  const s = survey.value
  if (!s || !s.offsets.length) return
  let bestIdx = 0
  let bestDist = Infinity
  for (let i = 0; i < s.offsets.length; i++) {
    const d = Math.abs(s.offsets[i] - offVal)
    if (d < bestDist) {
      bestDist = d
      bestIdx = i
    }
  }
  store.setCursor({ offsetIndex: bestIdx })
}
</script>

<template>
  <div class="rounded border border-slate-800 bg-slate-950/90 p-2.5 space-y-2">
    <div class="flex items-center justify-between gap-2">
      <span class="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
        Offset Trajectory
      </span>
      <span
        v-if="focusBadgeLabel"
        class="font-mono-num text-[9px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-cyan-300 truncate"
      >
        {{ focusBadgeLabel }}
      </span>
      <span
        v-else-if="trajectory?.available"
        class="font-mono-num text-[10px] text-cyan-300"
      >
        {{ trajectory.offsets.length }} offsets
      </span>
    </div>

    <template v-if="trajectory?.available && points.length > 0">
      <!-- Metrics Strip -->
      <div class="grid grid-cols-3 gap-1.5 text-center">
        <div class="rounded bg-slate-900/90 border border-slate-800/80 px-1.5 py-1">
          <div class="text-[9px] uppercase text-slate-500">Slope ‖b‖</div>
          <div class="font-mono-num text-[11px] font-semibold text-slate-100">
            {{ trajectory.slope !== null ? trajectory.slope.toExponential(2) : '—' }}
          </div>
        </div>
        <div class="rounded bg-slate-900/90 border border-slate-800/80 px-1.5 py-1">
          <div class="text-[9px] uppercase text-slate-500">Linear R²</div>
          <div class="font-mono-num text-[11px] font-semibold text-cyan-300">
            {{ trajectory.r2 !== null ? trajectory.r2.toFixed(2) : '—' }}
          </div>
        </div>
        <div class="rounded bg-slate-900/90 border border-slate-800/80 px-1.5 py-1">
          <div class="text-[9px] uppercase text-slate-500">Plane Var</div>
          <div class="font-mono-num text-[11px] font-semibold text-emerald-300">
            {{ planeVarPct !== null ? `${planeVarPct.toFixed(0)}%` : '—' }}
          </div>
        </div>
      </div>

      <!-- 2D Scatter + Path Plot -->
      <div class="relative rounded border border-slate-800/80 bg-slate-900/50 overflow-hidden">
        <svg
          :viewBox="`0 0 ${SVG_W} ${SVG_H}`"
          class="w-full h-auto block"
          @mouseleave="hoveredIdx = null"
        >
          <!-- Plot frame -->
          <rect
            :x="PAD_L"
            :y="PAD_T"
            :width="SVG_W - PAD_L - PAD_R"
            :height="SVG_H - PAD_T - PAD_B"
            fill="#020617"
            stroke="#1e293b"
            stroke-width="1"
          />

          <!-- Zero axes -->
          <line
            v-if="zeroAxes.showX0"
            :x1="zeroAxes.x0"
            :y1="PAD_T"
            :x2="zeroAxes.x0"
            :y2="SVG_H - PAD_B"
            stroke="#1e293b"
            stroke-dasharray="3 3"
            stroke-width="1"
          />
          <line
            v-if="zeroAxes.showY0"
            :x1="PAD_L"
            :y1="zeroAxes.y0"
            :x2="SVG_W - PAD_R"
            :y2="zeroAxes.y0"
            stroke="#1e293b"
            stroke-dasharray="3 3"
            stroke-width="1"
          />

          <!-- Trajectory path -->
          <polyline
            :points="polylinePoints"
            fill="none"
            stroke="rgba(148, 163, 184, 0.35)"
            stroke-width="1.25"
          />

          <!-- Points -->
          <g v-for="pt in points" :key="pt.idx">
            <circle
              :cx="pt.sx"
              :cy="pt.sy"
              :r="pt.isActive ? 5.2 : hoveredIdx === pt.idx ? 4.5 : 3"
              :fill="pt.color"
              :stroke="pt.isActive ? '#22d3ee' : '#020617'"
              :stroke-width="pt.isActive ? 2 : 0.75"
              class="cursor-pointer transition-all"
              @mouseenter="hoveredIdx = pt.idx"
              @click="jumpToOffset(pt.off)"
            />
          </g>

          <!-- Active point reticle ring -->
          <template v-for="pt in points" :key="`ring-${pt.idx}`">
            <circle
              v-if="pt.isActive"
              :cx="pt.sx"
              :cy="pt.sy"
              r="8.5"
              fill="none"
              stroke="#22d3ee"
              stroke-opacity="0.65"
              stroke-width="1"
              pointer-events="none"
            />
          </template>

          <!-- Axis labels -->
          <text
            :x="(PAD_L + SVG_W - PAD_R) / 2"
            :y="SVG_H - 6"
            text-anchor="middle"
            fill="#64748b"
            font-size="9"
            font-family="ui-monospace, monospace"
          >
            c₁ (Offset Direction)
          </text>
          <text
            x="10"
            :y="(PAD_T + SVG_H - PAD_B) / 2"
            text-anchor="middle"
            fill="#64748b"
            font-size="9"
            font-family="ui-monospace, monospace"
            :transform="`rotate(-90 10 ${(PAD_T + SVG_H - PAD_B) / 2})`"
          >
            c₂ (Curvature)
          </text>

          <!-- Numeric corner bounds -->
          <text
            :x="PAD_L + 2"
            :y="SVG_H - PAD_B + 10"
            fill="#475569"
            font-size="8"
            font-family="ui-monospace, monospace"
          >
            {{ bounds.minX.toFixed(1) }}
          </text>
          <text
            :x="SVG_W - PAD_R"
            :y="SVG_H - PAD_B + 10"
            text-anchor="end"
            fill="#475569"
            font-size="8"
            font-family="ui-monospace, monospace"
          >
            {{ bounds.maxX.toFixed(1) }}
          </text>
          <text
            :x="PAD_L - 3"
            :y="PAD_T + 7"
            text-anchor="end"
            fill="#475569"
            font-size="8"
            font-family="ui-monospace, monospace"
          >
            {{ bounds.maxY.toFixed(1) }}
          </text>
          <text
            :x="PAD_L - 3"
            :y="SVG_H - PAD_B"
            text-anchor="end"
            fill="#475569"
            font-size="8"
            font-family="ui-monospace, monospace"
          >
            {{ bounds.minY.toFixed(1) }}
          </text>
        </svg>

        <!-- Hover mini badge -->
        <div
          v-if="hoveredPoint"
          class="px-2 py-1 border-t border-slate-800 bg-slate-950 text-[10px] font-mono-num text-slate-300 flex items-center justify-between"
        >
          <span>Offset <b class="text-amber-300">{{ hoveredPoint.off }} m</b></span>
          <span>c₁={{ hoveredPoint.c1.toFixed(2) }}, c₂={{ hoveredPoint.c2.toFixed(2) }}</span>
        </div>
        <div
          v-else
          class="px-2 py-1 border-t border-slate-800 bg-slate-950 text-[10px] text-slate-500 flex items-center justify-between"
        >
          <span>Click any point to jump offset</span>
          <span class="font-mono-num text-cyan-300">{{ store.currentOffsetLabel }} m</span>
        </div>
      </div>
    </template>

    <div v-else class="text-[11px] text-slate-500 py-2">
      {{
        store.focusPick
          ? 'No Offset Trajectory summary available for this embedding set.'
          : 'Pick a Feature on an Inline or Crossline section to inspect its Offset Trajectory.'
      }}
    </div>
  </div>
</template>
