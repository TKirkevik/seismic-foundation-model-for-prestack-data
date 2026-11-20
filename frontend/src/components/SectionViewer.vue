<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Crosshair, RotateCcw } from 'lucide-vue-next'
import type { FeatureGeometryMode, PickedFeature, SectionAxis } from '../api/types'
import { useWorkbenchStore } from '../stores/workbench'
import { getOverlayLUT, getSeismicLUT } from '../utils/colormaps'
import {
  EDGE_HIT_RADIUS_PX,
  HANDLE_HIT_RADIUS_PX,
  ROLE_COLORS,
  hitTestEdges,
  hitTestHandles,
  type EdgeCandidate,
  type HandleCandidate,
} from '../utils/featureGeometry'
import { buildAxisLookup } from '../utils/interpolation'

const props = withDefaults(
  defineProps<{
    initialAxis?: SectionAxis
  }>(),
  {
    initialAxis: 'inline',
  },
)

const store = useWorkbenchStore()
const axis = ref<SectionAxis>(props.initialAxis)
const localGain = ref(1.0)

const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)

// Offscreen canvases for 1:1 rasterization
const seismicOffscreen = document.createElement('canvas')
const overlayOffscreen = document.createElement('canvas')
let hasSeismicImage = false
let hasOverlayImage = false

// Zoom & pan state in plot-relative normalized/pixel space
const zoom = ref(1.0)
const panX = ref(0)
const panY = ref(0)

// Pointer interaction state
let isPanning = false
let panStartClientX = 0
let panStartClientY = 0
let panOriginX = 0
let panOriginY = 0
let pointerDownMoved = false

const draggingHandle = ref<{ featureId: string; pickIndex: number } | null>(null)
let lastPrimaryClickTime = 0
let lastClickX = 0
let lastClickY = 0
let justFinishedUntil = 0

const hoverCanvasPos = ref<{ x: number; y: number } | null>(null)
const hoveredEdge = ref<{ featureId: string; segmentIndex: number } | null>(null)

const GEOM_OPTIONS: Array<{ id: FeatureGeometryMode; label: string }> = [
  { id: 'single', label: '1-Pick' },
  { id: 'points', label: 'Points' },
  { id: 'polyline', label: 'Line' },
  { id: 'polygon', label: 'Polygon' },
]

// Hover state
const hoverInfo = ref<{
  il: number
  xl: number
  sample: number
  amp: number
  sim: number | null
} | null>(null)

const MARGIN_LEFT = 56
const MARGIN_TOP = 26
const MARGIN_RIGHT = 12
const MARGIN_BOTTOM = 12

const survey = computed(() => store.activeSurvey)

const sliceMax = computed(() => {
  if (!survey.value) return 0
  const [nIl, nXl, nSamples] = survey.value.shape
  if (axis.value === 'inline') return Math.max(0, nIl - 1)
  if (axis.value === 'xline') return Math.max(0, nXl - 1)
  return Math.max(0, nSamples - 1)
})

const sliceIndex = computed({
  get() {
    if (axis.value === 'inline') return store.ilIdx
    if (axis.value === 'xline') return store.xlIdx
    return store.sampleIdx
  },
  set(val: number) {
    if (axis.value === 'inline') store.setCursor({ ilIdx: val })
    else if (axis.value === 'xline') store.setCursor({ xlIdx: val })
    else store.setCursor({ sampleIdx: val })
  },
})

const sliceReadout = computed(() => {
  if (!survey.value) return '—'
  if (axis.value === 'inline') {
    return `IL ${survey.value.ilines[store.ilIdx] ?? store.ilIdx}`
  }
  if (axis.value === 'xline') {
    return `XL ${survey.value.xlines[store.xlIdx] ?? store.xlIdx}`
  }
  return `${survey.value.samples[store.sampleIdx] ?? store.sampleIdx} ms`
})

const currentSection = computed(() => store.sections[axis.value])
const isLoading = computed(() => store.sectionLoading[axis.value])

const isDrawingOnThisPlane = computed(() => {
  const d = store.activeDrawingFeature
  return (
    d !== null &&
    (axis.value === 'inline' || axis.value === 'xline') &&
    d.plane_axis === axis.value &&
    d.plane_index === sliceIndex.value
  )
})

const canvasCursorClass = computed(() => {
  if (draggingHandle.value) return 'cursor-grabbing'
  if (store.hoveredPickRef && !isDrawingOnThisPlane.value) return 'cursor-grab'
  if (hoveredEdge.value && !isDrawingOnThisPlane.value && store.pickMode) return 'cursor-copy'
  return 'cursor-crosshair'
})

function getDataDims(): [number, number] {
  const sec = currentSection.value
  if (sec) {
    if (sec.axis === 'sample') {
      // shape is [n_il, n_xl], horizontal X = n_xl, vertical Y = n_il
      return [sec.shape[1], sec.shape[0]]
    }
    // inline: [n_xl, n_samples] -> W=n_xl, H=n_samples
    // xline: [n_il, n_samples] -> W=n_il, H=n_samples
    return [sec.shape[0], sec.shape[1]]
  }
  if (!survey.value) return [1, 1]
  const [nIl, nXl, nSamples] = survey.value.shape
  if (axis.value === 'inline') return [nXl, nSamples]
  if (axis.value === 'xline') return [nIl, nSamples]
  return [nXl, nIl]
}

function rebuildSeismicTexture() {
  const sec = currentSection.value
  if (!sec || !sec.data.length) {
    hasSeismicImage = false
    return
  }
  const [w, h] = getDataDims()
  if (w <= 0 || h <= 0) {
    hasSeismicImage = false
    return
  }
  seismicOffscreen.width = w
  seismicOffscreen.height = h
  const ctx = seismicOffscreen.getContext('2d')
  if (!ctx) return

  const img = ctx.createImageData(w, h)
  const px = img.data
  const data = sec.data
  const lut = getSeismicLUT(store.seismicColormap)
  const clip = Math.max(1e-6, store.effectiveClipAmplitude / Math.max(0.05, localGain.value))
  const invTwoClip = 1 / (2 * clip)

  if (sec.axis === 'sample') {
    // val = data[y * n_xl + x] -> already row-major (y, x)
    const total = w * h
    for (let i = 0; i < total; i++) {
      const v = data[i]
      let t = (v + clip) * invTwoClip
      if (t < 0) t = 0
      else if (t > 1) t = 1
      const cIdx = ((t * 255) | 0) * 4
      const o = i * 4
      px[o] = lut[cIdx]
      px[o + 1] = lut[cIdx + 1]
      px[o + 2] = lut[cIdx + 2]
      px[o + 3] = 255
    }
  } else {
    // inline: shape [n_xl, n_samples], val = data[x * h + y]
    // xline: shape [n_il, n_samples], val = data[x * h + y]
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w
      for (let x = 0; x < w; x++) {
        const v = data[x * h + y]
        let t = (v + clip) * invTwoClip
        if (t < 0) t = 0
        else if (t > 1) t = 1
        const cIdx = ((t * 255) | 0) * 4
        const o = (rowOffset + x) * 4
        px[o] = lut[cIdx]
        px[o + 1] = lut[cIdx + 1]
        px[o + 2] = lut[cIdx + 2]
        px[o + 3] = 255
      }
    }
  }

  ctx.putImageData(img, 0, 0)
  hasSeismicImage = true
}

function rebuildOverlayTexture() {
  const overlay = store.activeOverlay
  const s = survey.value
  if (!overlay || !s) {
    hasOverlayImage = false
    return
  }

  const [w, h] = getDataDims()
  if (w <= 0 || h <= 0) {
    hasOverlayImage = false
    return
  }

  overlayOffscreen.width = w
  overlayOffscreen.height = h
  const ctx = overlayOffscreen.getContext('2d')
  if (!ctx) return

  const img = ctx.createImageData(w, h)
  const px = img.data
  const lut = getOverlayLUT(store.overlayColormap)
  const smooth = store.overlayInterp === 'smooth'
  const [nIlP, nXlP, nTP] = overlay.gridShape
  const strideIl = nXlP * nTP
  const d = overlay.data
  const thresh = overlay.threshold
  const normMin = Math.min(thresh, overlay.max - 1e-4)
  const normSpan = Math.max(1e-5, overlay.max - normMin)

  const [nIl, nXl, nSamples] = s.shape
  const ilLook = buildAxisLookup(nIl, nIlP, overlay.patchSize, overlay.ilPatchStart, smooth)
  const xlLook = buildAxisLookup(nXl, nXlP, overlay.patchSize, overlay.xlPatchStart, smooth)
  const tLook = buildAxisLookup(nSamples, nTP, overlay.patchSize, overlay.tPatchStart, smooth)

  if (axis.value === 'inline') {
    const fixedIl = Math.min(nIl - 1, Math.max(0, store.ilIdx))
    const i0 = ilLook.i0[fixedIl]
    const i1 = ilLook.i1[fixedIl]
    const wi = ilLook.w1[fixedIl]
    const base0 = i0 * strideIl
    const base1 = i1 * strideIl

    for (let y = 0; y < h; y++) {
      const k0 = tLook.i0[y]
      const k1 = tLook.i1[y]
      const wk = tLook.w1[y]
      const rowOffset = y * w

      for (let x = 0; x < w; x++) {
        const j0 = xlLook.i0[x]
        const j1 = xlLook.i1[x]
        const wj = xlLook.w1[x]

        const v00 = d[base0 + j0 * nTP + k0] + (d[base0 + j0 * nTP + k1] - d[base0 + j0 * nTP + k0]) * wk
        const v01 = d[base0 + j1 * nTP + k0] + (d[base0 + j1 * nTP + k1] - d[base0 + j1 * nTP + k0]) * wk
        const v10 = d[base1 + j0 * nTP + k0] + (d[base1 + j0 * nTP + k1] - d[base1 + j0 * nTP + k0]) * wk
        const v11 = d[base1 + j1 * nTP + k0] + (d[base1 + j1 * nTP + k1] - d[base1 + j1 * nTP + k0]) * wk

        const v0 = v00 + (v01 - v00) * wj
        const v1 = v10 + (v11 - v10) * wj
        const val = v0 + (v1 - v0) * wi

        if (val >= thresh) {
          let norm = (val - normMin) / normSpan
          if (norm < 0) norm = 0
          else if (norm > 1) norm = 1
          const cIdx = ((norm * 255) | 0) * 4
          const o = (rowOffset + x) * 4
          px[o] = lut[cIdx]
          px[o + 1] = lut[cIdx + 1]
          px[o + 2] = lut[cIdx + 2]
          px[o + 3] = 255
        }
      }
    }
  } else if (axis.value === 'xline') {
    const fixedXl = Math.min(nXl - 1, Math.max(0, store.xlIdx))
    const j0 = xlLook.i0[fixedXl]
    const j1 = xlLook.i1[fixedXl]
    const wj = xlLook.w1[fixedXl]

    for (let y = 0; y < h; y++) {
      const k0 = tLook.i0[y]
      const k1 = tLook.i1[y]
      const wk = tLook.w1[y]
      const rowOffset = y * w

      for (let x = 0; x < w; x++) {
        const i0 = ilLook.i0[x]
        const i1 = ilLook.i1[x]
        const wi = ilLook.w1[x]
        const b0 = i0 * strideIl
        const b1 = i1 * strideIl

        const v00 = d[b0 + j0 * nTP + k0] + (d[b0 + j0 * nTP + k1] - d[b0 + j0 * nTP + k0]) * wk
        const v01 = d[b0 + j1 * nTP + k0] + (d[b0 + j1 * nTP + k1] - d[b0 + j1 * nTP + k0]) * wk
        const v10 = d[b1 + j0 * nTP + k0] + (d[b1 + j0 * nTP + k1] - d[b1 + j0 * nTP + k0]) * wk
        const v11 = d[b1 + j1 * nTP + k0] + (d[b1 + j1 * nTP + k1] - d[b1 + j1 * nTP + k0]) * wk

        const v0 = v00 + (v01 - v00) * wj
        const v1 = v10 + (v11 - v10) * wj
        const val = v0 + (v1 - v0) * wi

        if (val >= thresh) {
          let norm = (val - normMin) / normSpan
          if (norm < 0) norm = 0
          else if (norm > 1) norm = 1
          const cIdx = ((norm * 255) | 0) * 4
          const o = (rowOffset + x) * 4
          px[o] = lut[cIdx]
          px[o + 1] = lut[cIdx + 1]
          px[o + 2] = lut[cIdx + 2]
          px[o + 3] = 255
        }
      }
    }
  } else {
    const fixedT = Math.min(nSamples - 1, Math.max(0, store.sampleIdx))
    const k0 = tLook.i0[fixedT]
    const k1 = tLook.i1[fixedT]
    const wk = tLook.w1[fixedT]

    for (let y = 0; y < h; y++) {
      const i0 = ilLook.i0[y]
      const i1 = ilLook.i1[y]
      const wi = ilLook.w1[y]
      const b0 = i0 * strideIl
      const b1 = i1 * strideIl
      const rowOffset = y * w

      for (let x = 0; x < w; x++) {
        const j0 = xlLook.i0[x]
        const j1 = xlLook.i1[x]
        const wj = xlLook.w1[x]

        const v00 = d[b0 + j0 * nTP + k0] + (d[b0 + j0 * nTP + k1] - d[b0 + j0 * nTP + k0]) * wk
        const v01 = d[b0 + j1 * nTP + k0] + (d[b0 + j1 * nTP + k1] - d[b0 + j1 * nTP + k0]) * wk
        const v10 = d[b1 + j0 * nTP + k0] + (d[b1 + j0 * nTP + k1] - d[b1 + j0 * nTP + k0]) * wk
        const v11 = d[b1 + j1 * nTP + k0] + (d[b1 + j1 * nTP + k1] - d[b1 + j1 * nTP + k0]) * wk

        const v0 = v00 + (v01 - v00) * wj
        const v1 = v10 + (v11 - v10) * wj
        const val = v0 + (v1 - v0) * wi

        if (val >= thresh) {
          let norm = (val - normMin) / normSpan
          if (norm < 0) norm = 0
          else if (norm > 1) norm = 1
          const cIdx = ((norm * 255) | 0) * 4
          const o = (rowOffset + x) * 4
          px[o] = lut[cIdx]
          px[o + 1] = lut[cIdx + 1]
          px[o + 2] = lut[cIdx + 2]
          px[o + 3] = 255
        }
      }
    }
  }

  ctx.putImageData(img, 0, 0)
  hasOverlayImage = true
}

function getViewportRect(cssW: number, cssH: number) {
  const plotX = MARGIN_LEFT
  const plotY = MARGIN_TOP
  const plotW = Math.max(10, cssW - MARGIN_LEFT - MARGIN_RIGHT)
  const plotH = Math.max(10, cssH - MARGIN_TOP - MARGIN_BOTTOM)

  const baseW = plotW
  const baseH = plotH
  const drawW = baseW * zoom.value
  const drawH = baseH * zoom.value
  const drawX = plotX + (plotW - drawW) * 0.5 + panX.value
  const drawY = plotY + (plotH - drawH) * 0.5 + panY.value

  return { plotX, plotY, plotW, plotH, drawX, drawY, drawW, drawH }
}

function renderCanvas() {
  const canvas = canvasRef.value
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  const cssW = canvas.width / dpr
  const cssH = canvas.height / dpr
  if (cssW <= 0 || cssH <= 0) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.save()
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  // Background
  ctx.fillStyle = '#020617'
  ctx.fillRect(0, 0, cssW, cssH)

  const { plotX, plotY, plotW, plotH, drawX, drawY, drawW, drawH } = getViewportRect(cssW, cssH)
  const [dataW, dataH] = getDataDims()

  // Plot background
  ctx.fillStyle = '#090d16'
  ctx.fillRect(plotX, plotY, plotW, plotH)

  // Clip to plot area
  ctx.save()
  ctx.beginPath()
  ctx.rect(plotX, plotY, plotW, plotH)
  ctx.clip()

  if (hasSeismicImage) {
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'medium'
    ctx.drawImage(seismicOffscreen, drawX, drawY, drawW, drawH)
  }

  if (hasOverlayImage && store.activeOverlay) {
    ctx.save()
    ctx.globalAlpha = store.overlayOpacity
    ctx.imageSmoothingEnabled = store.overlayInterp === 'smooth'
    ctx.drawImage(overlayOffscreen, drawX, drawY, drawW, drawH)
    ctx.restore()
  }

  const toScreenX = (gx: number) => drawX + ((gx + 0.5) / dataW) * drawW
  const toScreenY = (gy: number) => drawY + ((gy + 0.5) / dataH) * drawH

  // Gather panel location markers
  const gp = store.gatherPanelData
  if (gp && gp.locations.length > 0) {
    ctx.save()
    if (axis.value === 'sample') {
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.75)'
      ctx.lineWidth = 1.25
      ctx.beginPath()
      gp.locations.forEach((loc, idx) => {
        const sx = toScreenX(loc.xl_idx)
        const sy = toScreenY(loc.il_idx)
        if (idx === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      ctx.fillStyle = '#fbbf24'
      for (const loc of gp.locations) {
        const sx = toScreenX(loc.xl_idx)
        const sy = toScreenY(loc.il_idx)
        ctx.fillRect(sx - 2.5, sy - 2.5, 5, 5)
      }
    } else if (axis.value === 'inline') {
      for (const loc of gp.locations) {
        if (Math.abs(loc.il_idx - store.ilIdx) <= 2) {
          const sx = toScreenX(loc.xl_idx)
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.45)'
          ctx.setLineDash([3, 4])
          ctx.beginPath()
          ctx.moveTo(sx, plotY)
          ctx.lineTo(sx, plotY + plotH)
          ctx.stroke()
          ctx.setLineDash([])
          ctx.fillStyle = '#fbbf24'
          ctx.beginPath()
          ctx.moveTo(sx, plotY + 8)
          ctx.lineTo(sx - 4, plotY + 1)
          ctx.lineTo(sx + 4, plotY + 1)
          ctx.closePath()
          ctx.fill()
        }
      }
    } else {
      for (const loc of gp.locations) {
        if (Math.abs(loc.xl_idx - store.xlIdx) <= 2) {
          const sx = toScreenX(loc.il_idx)
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.45)'
          ctx.setLineDash([3, 4])
          ctx.beginPath()
          ctx.moveTo(sx, plotY)
          ctx.lineTo(sx, plotY + plotH)
          ctx.stroke()
          ctx.setLineDash([])
          ctx.fillStyle = '#fbbf24'
          ctx.beginPath()
          ctx.moveTo(sx, plotY + 8)
          ctx.lineTo(sx - 4, plotY + 1)
          ctx.lineTo(sx + 4, plotY + 1)
          ctx.closePath()
          ctx.fill()
        }
      }
    }
    ctx.restore()
  }

  // Orthogonal synchronized cursor crosshair
  if (survey.value) {
    const curX =
      axis.value === 'inline'
        ? store.xlIdx
        : axis.value === 'xline'
          ? store.ilIdx
          : store.xlIdx
    const curY =
      axis.value === 'sample' ? store.ilIdx : store.sampleIdx

    const cx = toScreenX(curX)
    const cy = toScreenY(curY)

    ctx.save()
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.72)'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(cx, plotY)
    ctx.lineTo(cx, plotY + plotH)
    ctx.moveTo(plotX, cy)
    ctx.lineTo(plotX + plotW, cy)
    ctx.stroke()
    ctx.restore()
  }

  // Picked Features overlay (coplanarity guard: only features on this axis)
  if (survey.value && (axis.value === 'inline' || axis.value === 'xline')) {
    const curPlaneAxis = axis.value
    const curSliceIdx = sliceIndex.value
    const getHorizIdx = (f: PickedFeature, pIdx: number) =>
      curPlaneAxis === 'inline' ? f.picks[pIdx].xl_idx : f.picks[pIdx].il_idx

    for (const feat of store.pickedFeatures) {
      if (!feat.enabled || feat.picks.length === 0) continue
      if (feat.plane_axis !== curPlaneAxis) continue

      const delta = Math.abs(feat.plane_index - curSliceIdx)
      if (delta > 8) continue

      const colors = ROLE_COLORS[feat.role]
      const onPlane = delta === 0
      const isSelected = feat.id === store.selectedFeatureId
      const isDrawing = feat.status === 'in_progress' && onPlane

      const pts = feat.picks.map((p, idx) => ({
        x: toScreenX(getHorizIdx(feat, idx)),
        y: toScreenY(p.sample_idx),
      }))

      ctx.save()
      if (!onPlane) {
        ctx.globalAlpha = 0.4
        ctx.setLineDash([4, 4])
      }

      // Draw geometry body
      if (feat.geometry_mode === 'polygon') {
        if (pts.length >= 3) {
          ctx.beginPath()
          ctx.moveTo(pts[0].x, pts[0].y)
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
          if (!isDrawing) ctx.closePath()
          if (onPlane) {
            ctx.fillStyle = colors.fill
            ctx.fill('evenodd')
          }
          ctx.strokeStyle = colors.primary
          ctx.lineWidth = isSelected && onPlane ? 2.4 : 1.8
          ctx.stroke()
        } else if (pts.length === 2) {
          ctx.beginPath()
          ctx.moveTo(pts[0].x, pts[0].y)
          ctx.lineTo(pts[1].x, pts[1].y)
          ctx.strokeStyle = colors.primary
          ctx.lineWidth = isSelected && onPlane ? 2.4 : 1.8
          ctx.stroke()
        }

        // Rubber-band segment while drawing
        if (isDrawing && hoverCanvasPos.value && pts.length >= 1) {
          ctx.save()
          ctx.setLineDash([6, 3])
          ctx.strokeStyle = colors.primary
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(pts[pts.length - 1].x, pts[pts.length - 1].y)
          ctx.lineTo(hoverCanvasPos.value.x, hoverCanvasPos.value.y)
          if (pts.length >= 2) {
            ctx.lineTo(pts[0].x, pts[0].y)
          }
          ctx.stroke()
          ctx.restore()
        }
      } else if (feat.geometry_mode === 'polyline') {
        if (pts.length >= 2) {
          ctx.beginPath()
          ctx.moveTo(pts[0].x, pts[0].y)
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
          ctx.strokeStyle = colors.primary
          ctx.lineWidth = isSelected && onPlane ? 2.4 : 1.8
          ctx.stroke()
        }
        if (isDrawing && hoverCanvasPos.value && pts.length >= 1) {
          ctx.save()
          ctx.setLineDash([6, 3])
          ctx.strokeStyle = colors.primary
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(pts[pts.length - 1].x, pts[pts.length - 1].y)
          ctx.lineTo(hoverCanvasPos.value.x, hoverCanvasPos.value.y)
          ctx.stroke()
          ctx.restore()
        }
      }

      // Draw handles / reticles
      ctx.setLineDash([])
      for (let i = 0; i < pts.length; i++) {
        const pt = pts[i]
        const isHovered =
          onPlane &&
          store.hoveredPickRef?.featureId === feat.id &&
          store.hoveredPickRef?.pickIndex === i
        const isDragged =
          onPlane &&
          draggingHandle.value?.featureId === feat.id &&
          draggingHandle.value?.pickIndex === i

        if (feat.geometry_mode === 'single') {
          const r = isHovered || isDragged ? 10 : onPlane ? 8 : 6
          ctx.strokeStyle = isHovered || isDragged ? '#ffffff' : colors.primary
          ctx.lineWidth = isHovered || isDragged ? 2.2 : onPlane ? 1.8 : 1.2
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2)
          ctx.stroke()

          ctx.beginPath()
          ctx.moveTo(pt.x - 13, pt.y)
          ctx.lineTo(pt.x - 4, pt.y)
          ctx.moveTo(pt.x + 4, pt.y)
          ctx.lineTo(pt.x + 13, pt.y)
          ctx.moveTo(pt.x, pt.y - 13)
          ctx.lineTo(pt.x, pt.y - 4)
          ctx.moveTo(pt.x, pt.y + 4)
          ctx.lineTo(pt.x, pt.y + 13)
          ctx.stroke()

          ctx.fillStyle = isHovered || isDragged ? '#fbbf24' : colors.primary
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, isHovered || isDragged ? 3.2 : 2.5, 0, Math.PI * 2)
          ctx.fill()
        } else {
          const r = isHovered || isDragged ? 7.5 : onPlane ? 5 : 3.5
          ctx.fillStyle = isHovered || isDragged ? '#fbbf24' : '#020617'
          ctx.strokeStyle = isHovered || isDragged ? '#ffffff' : colors.primary
          ctx.lineWidth = isHovered || isDragged ? 2.2 : 1.75
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()

          if (onPlane) {
            ctx.fillStyle = isHovered || isDragged ? '#ffffff' : colors.primary
            ctx.beginPath()
            ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }

      ctx.restore()
    }
  }

  ctx.restore() // end plot clip

  // Plot border
  ctx.strokeStyle = '#1e293b'
  ctx.lineWidth = 1
  ctx.strokeRect(plotX + 0.5, plotY + 0.5, plotW - 1, plotH - 1)

  // Rulers & ticks
  drawRulers(ctx, plotX, plotY, plotW, plotH, drawX, drawY, drawW, drawH, dataW, dataH)

  ctx.restore()
}

function drawRulers(
  ctx: CanvasRenderingContext2D,
  plotX: number,
  plotY: number,
  plotW: number,
  plotH: number,
  drawX: number,
  drawY: number,
  drawW: number,
  drawH: number,
  dataW: number,
  dataH: number,
) {
  const s = survey.value
  if (!s) return

  ctx.save()
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.fillStyle = '#94a3b8'
  ctx.strokeStyle = '#334155'
  ctx.lineWidth = 1

  const xLabels = axis.value === 'xline' ? s.ilines : s.xlines
  const yLabels = axis.value === 'sample' ? s.ilines : s.samples
  const xPrefix = axis.value === 'xline' ? 'IL' : 'XL'
  const yUnit = axis.value === 'sample' ? 'IL' : 'ms'

  // Top X ticks
  const numXTicks = Math.max(3, Math.floor(plotW / 110))
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  for (let i = 0; i <= numXTicks; i++) {
    const sx = plotX + (i / numXTicks) * plotW
    const frac = (sx - drawX) / drawW
    const idx = Math.round(frac * dataW - 0.5)
    if (idx < 0 || idx >= dataW) continue
    ctx.beginPath()
    ctx.moveTo(sx, plotY)
    ctx.lineTo(sx, plotY - 4)
    ctx.stroke()
    const lbl = xLabels[idx] ?? idx
    ctx.fillText(`${xPrefix} ${lbl}`, sx, plotY - 6)
  }

  // Left Y ticks
  const numYTicks = Math.max(4, Math.floor(plotH / 70))
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  for (let i = 0; i <= numYTicks; i++) {
    const sy = plotY + (i / numYTicks) * plotH
    const frac = (sy - drawY) / drawH
    const idx = Math.round(frac * dataH - 0.5)
    if (idx < 0 || idx >= dataH) continue
    ctx.beginPath()
    ctx.moveTo(plotX, sy)
    ctx.lineTo(plotX - 4, sy)
    ctx.stroke()
    const lbl = yLabels[idx] ?? idx
    ctx.fillText(axis.value === 'sample' ? `${lbl}` : `${lbl}`, plotX - 7, sy)
  }

  // Axis unit badge top-left corner
  ctx.textAlign = 'left'
  ctx.textBaseline = 'bottom'
  ctx.fillStyle = '#64748b'
  ctx.fillText(yUnit, 8, plotY - 6)

  ctx.restore()
}

let rafPending = false
function scheduleRender() {
  if (rafPending) return
  rafPending = true
  requestAnimationFrame(() => {
    rafPending = false
    renderCanvas()
  })
}

function resizeCanvas() {
  const container = containerRef.value
  const canvas = canvasRef.value
  if (!container || !canvas) return
  const rect = container.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  const w = Math.max(1, Math.floor(rect.width))
  const h = Math.max(1, Math.floor(rect.height))
  if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
    canvas.width = Math.floor(w * dpr)
    canvas.height = Math.floor(h * dpr)
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
  }
  scheduleRender()
}

function screenToData(
  clientX: number,
  clientY: number,
  clamp = false,
): { gx: number; gy: number } | null {
  const canvas = canvasRef.value
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  const mx = clientX - rect.left
  const my = clientY - rect.top
  const { plotX, plotY, plotW, plotH, drawX, drawY, drawW, drawH } = getViewportRect(
    rect.width,
    rect.height,
  )
  if (!clamp && (mx < plotX || mx > plotX + plotW || my < plotY || my > plotY + plotH)) {
    return null
  }
  const [dataW, dataH] = getDataDims()
  let gx = Math.floor(((mx - drawX) / drawW) * dataW)
  let gy = Math.floor(((my - drawY) / drawH) * dataH)
  if (clamp) {
    gx = Math.max(0, Math.min(dataW - 1, gx))
    gy = Math.max(0, Math.min(dataH - 1, gy))
    return { gx, gy }
  }
  if (gx < 0 || gx >= dataW || gy < 0 || gy >= dataH) return null
  return { gx, gy }
}

function getPlaneHandleAndEdgeCandidates() {
  const canvas = canvasRef.value
  if (!canvas || (axis.value !== 'inline' && axis.value !== 'xline')) {
    return { handles: [] as HandleCandidate[], edges: [] as EdgeCandidate[] }
  }
  const rect = canvas.getBoundingClientRect()
  const { drawX, drawY, drawW, drawH } = getViewportRect(rect.width, rect.height)
  const [dataW, dataH] = getDataDims()
  const toScreenX = (gx: number) => drawX + ((gx + 0.5) / dataW) * drawW
  const toScreenY = (gy: number) => drawY + ((gy + 0.5) / dataH) * drawH

  const curAxis = axis.value
  const curSlice = sliceIndex.value
  const handles: HandleCandidate[] = []
  const edges: EdgeCandidate[] = []

  for (const feat of store.pickedFeatures) {
    if (!feat.enabled || feat.picks.length === 0) continue
    if (feat.plane_axis !== curAxis || feat.plane_index !== curSlice) continue

    const pts = feat.picks.map((p) => ({
      x: toScreenX(curAxis === 'inline' ? p.xl_idx : p.il_idx),
      y: toScreenY(p.sample_idx),
    }))

    for (let i = 0; i < pts.length; i++) {
      handles.push({
        featureId: feat.id,
        pickIndex: i,
        x: pts[i].x,
        y: pts[i].y,
      })
    }

    if (
      feat.status === 'completed' &&
      (feat.geometry_mode === 'polyline' || feat.geometry_mode === 'polygon') &&
      pts.length >= 2
    ) {
      edges.push({
        featureId: feat.id,
        geometryMode: feat.geometry_mode,
        points: pts,
      })
    }
  }

  return { handles, edges }
}

function onMouseDown(e: MouseEvent) {
  if (e.button === 1) {
    e.preventDefault()
  }
}

function onPointerDown(e: PointerEvent) {
  const canvas = canvasRef.value
  if (!canvas) return
  if (e.button === 1) e.preventDefault()
  canvas.setPointerCapture(e.pointerId)
  pointerDownMoved = false

  const rect = canvas.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const { handles } = getPlaneHandleAndEdgeCandidates()
  const hitHandle = hitTestHandles(mx, my, handles, HANDLE_HIT_RADIUS_PX)

  if (isDrawingOnThisPlane.value) {
    if (e.button === 1 && hitHandle) {
      e.preventDefault()
      store.deletePick(hitHandle.featureId, hitHandle.pickIndex)
      return
    }
    if (e.button === 1 || e.button === 2 || e.altKey) {
      isPanning = true
    }
    panStartClientX = e.clientX
    panStartClientY = e.clientY
    panOriginX = panX.value
    panOriginY = panY.value
    return
  }

  if (hitHandle) {
    if (e.button === 0) {
      draggingHandle.value = { featureId: hitHandle.featureId, pickIndex: hitHandle.pickIndex }
      store.selectedFeatureId = hitHandle.featureId
      store.setHoveredPick(draggingHandle.value)
      return
    }
    if (e.button === 1) {
      e.preventDefault()
      store.deletePick(hitHandle.featureId, hitHandle.pickIndex)
      return
    }
  }

  if (e.button === 1 || e.button === 2 || e.altKey) {
    isPanning = true
    panStartClientX = e.clientX
    panStartClientY = e.clientY
    panOriginX = panX.value
    panOriginY = panY.value
  } else if (e.button === 0) {
    panStartClientX = e.clientX
    panStartClientY = e.clientY
    panOriginX = panX.value
    panOriginY = panY.value
  }
}

function onPointerMove(e: PointerEvent) {
  const canvas = canvasRef.value
  if (canvas) {
    const rect = canvas.getBoundingClientRect()
    hoverCanvasPos.value = { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  if (draggingHandle.value) {
    const clamped = screenToData(e.clientX, e.clientY, true)
    if (clamped) {
      store.movePick(
        draggingHandle.value.featureId,
        draggingHandle.value.pickIndex,
        clamped.gx,
        clamped.gy,
        { commit: false },
      )
    }
    scheduleRender()
    return
  }

  if (isPanning) {
    const dx = e.clientX - panStartClientX
    const dy = e.clientY - panStartClientY
    if (Math.hypot(dx, dy) > 3) pointerDownMoved = true
    panX.value = panOriginX + dx
    panY.value = panOriginY + dy
    scheduleRender()
    return
  }

  if (e.buttons === 1 && !isDrawingOnThisPlane.value) {
    const dx = e.clientX - panStartClientX
    const dy = e.clientY - panStartClientY
    if (Math.hypot(dx, dy) > 4) {
      isPanning = true
      pointerDownMoved = true
      panX.value = panOriginX + dx
      panY.value = panOriginY + dy
      scheduleRender()
      return
    }
  }

  if (e.buttons === 0 && canvas) {
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    if (!isDrawingOnThisPlane.value) {
      const { handles, edges } = getPlaneHandleAndEdgeCandidates()
      const hitHandle = hitTestHandles(mx, my, handles, HANDLE_HIT_RADIUS_PX)
      if (hitHandle) {
        store.setHoveredPick({ featureId: hitHandle.featureId, pickIndex: hitHandle.pickIndex })
        hoveredEdge.value = null
      } else {
        store.setHoveredPick(null)
        const hitEdge = hitTestEdges(mx, my, edges, EDGE_HIT_RADIUS_PX)
        hoveredEdge.value = hitEdge
          ? { featureId: hitEdge.featureId, segmentIndex: hitEdge.segmentIndex }
          : null
      }
    } else {
      store.setHoveredPick(null)
      hoveredEdge.value = null
      scheduleRender()
    }
  }

  updateHover(e.clientX, e.clientY)
}

function onPointerUp(e: PointerEvent) {
  const canvas = canvasRef.value
  if (canvas && canvas.hasPointerCapture(e.pointerId)) {
    canvas.releasePointerCapture(e.pointerId)
  }

  if (draggingHandle.value) {
    const clamped = screenToData(e.clientX, e.clientY, true)
    if (clamped) {
      store.movePick(
        draggingHandle.value.featureId,
        draggingHandle.value.pickIndex,
        clamped.gx,
        clamped.gy,
        { commit: true },
      )
    }
    draggingHandle.value = null
    scheduleRender()
    return
  }

  if (isPanning) {
    isPanning = false
    if (pointerDownMoved) return
  }

  if (e.button !== 0 || pointerDownMoved) return

  const now = performance.now()
  if (now < justFinishedUntil) {
    return
  }

  const coords = screenToData(e.clientX, e.clientY)
  if (!coords || !survey.value) return

  const onExactVerticalPlane = axis.value === 'inline' || axis.value === 'xline'

  if (isDrawingOnThisPlane.value && onExactVerticalPlane) {
    const isDoubleClick =
      e.detail >= 2 ||
      (now - lastPrimaryClickTime < 320 &&
        Math.hypot(e.clientX - lastClickX, e.clientY - lastClickY) < 8)

    if (isDoubleClick) {
      justFinishedUntil = now + 260
      lastPrimaryClickTime = 0
      void store.placePickOnPlane(
        axis.value as 'inline' | 'xline',
        sliceIndex.value,
        coords.gx,
        coords.gy,
        { isDoubleClick: true },
      )
    } else {
      lastPrimaryClickTime = now
      lastClickX = e.clientX
      lastClickY = e.clientY
      void store.placePickOnPlane(
        axis.value as 'inline' | 'xline',
        sliceIndex.value,
        coords.gx,
        coords.gy,
      )
    }
    return
  }

  if (onExactVerticalPlane && store.pickMode && !e.shiftKey && canvas) {
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const { edges } = getPlaneHandleAndEdgeCandidates()
    const hitEdge = hitTestEdges(mx, my, edges, EDGE_HIT_RADIUS_PX)
    if (hitEdge) {
      store.insertPickOnSegment(
        hitEdge.featureId,
        hitEdge.segmentIndex,
        coords.gx,
        coords.gy,
      )
      return
    }

    lastPrimaryClickTime = now
    lastClickX = e.clientX
    lastClickY = e.clientY
    void store.placePickOnPlane(
      axis.value as 'inline' | 'xline',
      sliceIndex.value,
      coords.gx,
      coords.gy,
    )
    return
  }

  // Cursor-only navigation (Time Slice, Shift+Click, or Pick mode off)
  let targetIl = store.ilIdx
  let targetXl = store.xlIdx
  let targetSample = store.sampleIdx
  if (axis.value === 'inline') {
    targetXl = coords.gx
    targetSample = coords.gy
  } else if (axis.value === 'xline') {
    targetIl = coords.gx
    targetSample = coords.gy
  } else {
    targetXl = coords.gx
    targetIl = coords.gy
  }
  store.setCursor({ ilIdx: targetIl, xlIdx: targetXl, sampleIdx: targetSample })
}

function onPointerCancel(e: PointerEvent) {
  const canvas = canvasRef.value
  if (canvas && canvas.hasPointerCapture(e.pointerId)) {
    canvas.releasePointerCapture(e.pointerId)
  }
  if (draggingHandle.value) {
    const clamped = screenToData(e.clientX, e.clientY, true)
    if (clamped) {
      store.movePick(
        draggingHandle.value.featureId,
        draggingHandle.value.pickIndex,
        clamped.gx,
        clamped.gy,
        { commit: true },
      )
    }
    draggingHandle.value = null
  }
  isPanning = false
  scheduleRender()
}

function onPointerLeave() {
  hoverInfo.value = null
  hoverCanvasPos.value = null
  hoveredEdge.value = null
  if (!draggingHandle.value) {
    store.setHoveredPick(null)
  }
  if (isDrawingOnThisPlane.value) {
    scheduleRender()
  }
}

function updateHover(clientX: number, clientY: number) {
  const coords = screenToData(clientX, clientY)
  const s = survey.value
  const sec = currentSection.value
  if (!coords || !s || !sec) {
    hoverInfo.value = null
    return
  }

  const [dataW, dataH] = getDataDims()
  let ilI = store.ilIdx
  let xlI = store.xlIdx
  let sampI = store.sampleIdx
  let amp = 0

  if (axis.value === 'inline') {
    xlI = coords.gx
    sampI = coords.gy
    amp = sec.data[coords.gx * dataH + coords.gy] ?? 0
  } else if (axis.value === 'xline') {
    ilI = coords.gx
    sampI = coords.gy
    amp = sec.data[coords.gx * dataH + coords.gy] ?? 0
  } else {
    xlI = coords.gx
    ilI = coords.gy
    amp = sec.data[coords.gy * dataW + coords.gx] ?? 0
  }

  let sim: number | null = null
  const ov = store.activeOverlay
  if (ov) {
    const smooth = store.overlayInterp === 'smooth'
    const [nIlP, nXlP, nTP] = ov.gridShape
    const ilLook = buildAxisLookup(s.shape[0], nIlP, ov.patchSize, ov.ilPatchStart, smooth)
    const xlLook = buildAxisLookup(s.shape[1], nXlP, ov.patchSize, ov.xlPatchStart, smooth)
    const tLook = buildAxisLookup(s.shape[2], nTP, ov.patchSize, ov.tPatchStart, smooth)
    const i0 = ilLook.i0[ilI]
    const i1 = ilLook.i1[ilI]
    const wi = ilLook.w1[ilI]
    const j0 = xlLook.i0[xlI]
    const j1 = xlLook.i1[xlI]
    const wj = xlLook.w1[xlI]
    const k0 = tLook.i0[sampI]
    const k1 = tLook.i1[sampI]
    const wk = tLook.w1[sampI]
    const stride = nXlP * nTP
    const d = ov.data
    const c00 = d[i0 * stride + j0 * nTP + k0] * (1 - wk) + d[i0 * stride + j0 * nTP + k1] * wk
    const c01 = d[i0 * stride + j1 * nTP + k0] * (1 - wk) + d[i0 * stride + j1 * nTP + k1] * wk
    const c10 = d[i1 * stride + j0 * nTP + k0] * (1 - wk) + d[i1 * stride + j0 * nTP + k1] * wk
    const c11 = d[i1 * stride + j1 * nTP + k0] * (1 - wk) + d[i1 * stride + j1 * nTP + k1] * wk
    sim = (c00 * (1 - wj) + c01 * wj) * (1 - wi) + (c10 * (1 - wj) + c11 * wj) * wi
  }

  hoverInfo.value = {
    il: s.ilines[ilI] ?? ilI,
    xl: s.xlines[xlI] ?? xlI,
    sample: s.samples[sampI] ?? sampI,
    amp,
    sim,
  }
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top

  const { plotX, plotY, plotW, plotH, drawX, drawY, drawW, drawH } = getViewportRect(
    rect.width,
    rect.height,
  )

  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  const nextZoom = Math.max(0.5, Math.min(24, zoom.value * factor))
  if (nextZoom === zoom.value) return

  // Anchor zoom at cursor
  const relX = (mx - drawX) / drawW
  const relY = (my - drawY) / drawH
  const nextW = plotW * nextZoom
  const nextH = plotH * nextZoom
  const nextDrawX = mx - relX * nextW
  const nextDrawY = my - relY * nextH

  panX.value = nextDrawX - (plotX + (plotW - nextW) * 0.5)
  panY.value = nextDrawY - (plotY + (plotH - nextH) * 0.5)
  zoom.value = nextZoom
  scheduleRender()
}

function resetZoom() {
  zoom.value = 1.0
  panX.value = 0
  panY.value = 0
  scheduleRender()
}

let resizeObs: ResizeObserver | null = null

onMounted(() => {
  void store.ensureSection(axis.value)
  resizeCanvas()
  if (containerRef.value) {
    resizeObs = new ResizeObserver(() => resizeCanvas())
    resizeObs.observe(containerRef.value)
  }
})

onBeforeUnmount(() => {
  resizeObs?.disconnect()
})

watch(axis, (nextAxis) => {
  resetZoom()
  void store.ensureSection(nextAxis)
})

watch(
  [
    currentSection,
    () => store.seismicColormap,
    () => store.effectiveClipAmplitude,
    localGain,
  ],
  () => {
    rebuildSeismicTexture()
    scheduleRender()
  },
  { immediate: true },
)

watch(
  [
    () => store.activeOverlay,
    () => store.overlayColormap,
    () => store.overlayInterp,
    () => store.ilIdx,
    () => store.xlIdx,
    () => store.sampleIdx,
    axis,
  ],
  () => {
    rebuildOverlayTexture()
    scheduleRender()
  },
  { immediate: true },
)

watch(
  [
    () => store.overlayOpacity,
    () => store.pickedFeatures,
    () => store.selectedFeatureId,
    () => store.hoveredPickRef,
    () => store.gatherPanelData,
  ],
  () => {
    scheduleRender()
  },
  { deep: true },
)
</script>

<template>
  <div class="flex flex-col h-full w-full bg-slate-950 select-none overflow-hidden">
    <!-- Pane Sub-header Controls -->
    <div
      class="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-2.5 py-1.5 bg-slate-900/80 border-b border-slate-800/90 text-xs shrink-0"
    >
      <!-- Axis selector -->
      <div class="flex items-center rounded bg-slate-950 border border-slate-800 p-0.5">
        <button
          v-for="opt in [
            { id: 'inline', label: 'Inline' },
            { id: 'xline', label: 'Crossline' },
            { id: 'sample', label: 'Time Slice' },
          ] as const"
          :key="opt.id"
          type="button"
          class="px-2 py-0.5 rounded text-[11px] font-medium transition-colors"
          :class="
            axis === opt.id
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200 border border-transparent'
          "
          @click="axis = opt.id"
        >
          {{ opt.label }}
        </button>
      </div>

      <!-- Slice Scrubber -->
      <div class="flex items-center gap-2 min-w-[190px] flex-1 max-w-xs">
        <input
          v-model.number="sliceIndex"
          type="range"
          min="0"
          :max="sliceMax"
          step="1"
          class="ws-slider flex-1"
          :disabled="!survey"
        />
        <span
          class="font-mono-num text-[11px] px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-cyan-300 min-w-[74px] text-center"
        >
          {{ sliceReadout }}
        </span>
      </div>

      <!-- Local Gain -->
      <div class="flex items-center gap-1.5">
        <span class="text-[11px] text-slate-400">Gain</span>
        <input
          v-model.number="localGain"
          type="range"
          min="0.2"
          max="4"
          step="0.05"
          class="ws-slider w-16"
        />
        <span class="font-mono-num text-[11px] text-slate-300 w-9 text-right">
          {{ localGain.toFixed(1) }}×
        </span>
      </div>

      <div class="flex items-center gap-1.5 ml-auto flex-wrap">
        <!-- Quick Geometry Mode & Role Pills (on vertical sections) -->
        <div
          v-if="axis !== 'sample'"
          class="hidden md:flex items-center rounded bg-slate-950 border border-slate-800 p-0.5"
        >
          <button
            v-for="opt in GEOM_OPTIONS"
            :key="opt.id"
            type="button"
            class="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors"
            :class="
              store.activeGeometryMode === opt.id
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            "
            @click="store.setGeometryMode(opt.id)"
          >
            {{ opt.label }}
          </button>
        </div>

        <div
          v-if="axis !== 'sample'"
          class="hidden md:flex items-center rounded bg-slate-950 border border-slate-800 p-0.5"
        >
          <button
            type="button"
            class="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors"
            :class="
              store.activeFeatureRole === 'include'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            "
            @click="store.setFeatureRole('include')"
          >
            + Inc
          </button>
          <button
            type="button"
            class="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors"
            :class="
              store.activeFeatureRole === 'exclude'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            "
            @click="store.setFeatureRole('exclude')"
          >
            − Exc
          </button>
        </div>

        <!-- Pick Feature vs Cursor mode -->
        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border transition-colors"
          :disabled="axis === 'sample'"
          :class="
            axis === 'sample'
              ? 'bg-slate-950/60 border-slate-800/70 text-slate-600 cursor-not-allowed'
              : store.pickMode
                ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-300'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
          "
          :title="
            axis === 'sample'
              ? 'Picking is available on Inline and Crossline vertical sections'
              : 'Left-click picks a Feature (Shift+Click moves cursor only)'
          "
          @click="store.pickMode = !store.pickMode"
        >
          <Crosshair class="w-3 h-3" />
          <span>{{ axis === 'sample' ? 'Cursor Only' : 'Pick' }}</span>
        </button>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-slate-950 border border-slate-800 text-slate-300 hover:bg-slate-800 transition-colors"
          title="Reset zoom and pan"
          @click="resetZoom"
        >
          <RotateCcw class="w-3 h-3" />
          <span>Reset</span>
        </button>
      </div>
    </div>

    <!-- Canvas Area -->
    <div ref="containerRef" class="relative flex-1 min-h-0 overflow-hidden bg-slate-950">
      <canvas
        ref="canvasRef"
        class="block w-full h-full"
        :class="canvasCursorClass"
        @mousedown="onMouseDown"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointercancel="onPointerCancel"
        @pointerleave="onPointerLeave"
        @wheel="onWheel"
        @auxclick.prevent
        @contextmenu.prevent
      />

      <!-- Drawing in-progress badge -->
      <div
        v-if="isDrawingOnThisPlane"
        class="absolute top-8 left-16 z-10 flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-950/90 border border-amber-500/50 text-amber-200 text-[10px] shadow-lg"
      >
        <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
        <span>Drawing… (Double-click or Enter to finish)</span>
        <button
          type="button"
          class="px-1.5 py-0.2 rounded bg-amber-500/30 hover:bg-amber-500/50 text-amber-100 font-semibold"
          @click="store.finishActiveDrawing()"
        >
          Finish
        </button>
      </div>

      <!-- Loading badge -->
      <div
        v-if="isLoading"
        class="absolute top-2 right-3 px-2 py-0.5 rounded bg-slate-900/90 border border-slate-700/80 text-[10px] font-mono-num text-cyan-300 pointer-events-none"
      >
        Loading section…
      </div>

      <!-- Hover Readout -->
      <div
        v-if="hoverInfo"
        class="absolute bottom-2.5 left-16 px-2.5 py-1 rounded bg-slate-950/90 border border-slate-800 text-[11px] font-mono-num text-slate-200 pointer-events-none flex items-center gap-2.5 shadow-lg"
      >
        <span>IL <b class="text-cyan-300">{{ hoverInfo.il }}</b></span>
        <span>XL <b class="text-cyan-300">{{ hoverInfo.xl }}</b></span>
        <span>T <b class="text-cyan-300">{{ hoverInfo.sample }}</b> ms</span>
        <span>Off <b class="text-amber-300">{{ store.currentOffsetLabel }}</b> m</span>
        <span>Amp <b class="text-slate-100">{{ hoverInfo.amp.toFixed(3) }}</b></span>
        <span v-if="hoverInfo.sim !== null">
          {{ store.overlaySource === 'attribute' ? 'Attr' : 'Sim' }}
          <b class="text-emerald-300">{{ hoverInfo.sim.toFixed(3) }}</b>
        </span>
      </div>

      <!-- Empty state -->
      <div
        v-if="!survey"
        class="absolute inset-0 flex items-center justify-center text-xs text-slate-500"
      >
        No survey loaded
      </div>
    </div>
  </div>
</template>
