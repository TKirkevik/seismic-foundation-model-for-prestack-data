<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Crosshair, RefreshCw, RotateCcw } from 'lucide-vue-next'
import { useWorkbenchStore } from '../stores/workbench'
import { getOverlayLUT, getSeismicLUT } from '../utils/colormaps'
import { buildAxisLookup } from '../utils/interpolation'

const store = useWorkbenchStore()
const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)

const zoomX = ref(1.0)
const zoomY = ref(1.0)
const panX = ref(0)
const panY = ref(0)

let isPanning = false
let panStartClientX = 0
let panStartClientY = 0
let panOriginX = 0
let panOriginY = 0
let pointerDownMoved = false

// Offscreen buffers per gather
const seismicCanvases: HTMLCanvasElement[] = []
const overlayCanvases: HTMLCanvasElement[] = []

const hoverInfo = ref<{
  gatherIdx: number
  il: number
  xl: number
  offset: number
  sample: number
  amp: number
  sim: number | null
} | null>(null)

const MARGIN_LEFT = 56
const MARGIN_TOP = 40
const MARGIN_RIGHT = 12
const MARGIN_BOTTOM = 12
const GUTTER = 8

const survey = computed(() => store.activeSurvey)
const gp = computed(() => store.gatherPanelData)

const startIlHeader = computed(
  () => survey.value?.ilines[store.gatherIlStartIdx] ?? store.gatherIlStartIdx,
)
const startXlHeader = computed(
  () => survey.value?.xlines[store.gatherXlStartIdx] ?? store.gatherXlStartIdx,
)

function ensureOffscreenPair(idx: number) {
  while (seismicCanvases.length <= idx) {
    seismicCanvases.push(document.createElement('canvas'))
    overlayCanvases.push(document.createElement('canvas'))
  }
}

function rebuildGatherSeismicTextures() {
  const dataObj = gp.value
  if (!dataObj || !dataObj.data.length) return
  const [actualCount, nOffsets, nSamples] = dataObj.shape
  if (actualCount <= 0 || nOffsets <= 0 || nSamples <= 0) return

  const lut = getSeismicLUT(store.seismicColormap)
  const clip = Math.max(1e-6, store.effectiveClipAmplitude)
  const invTwoClip = 1 / (2 * clip)
  const d = dataObj.data

  for (let g = 0; g < actualCount; g++) {
    ensureOffscreenPair(g)
    const c = seismicCanvases[g]
    c.width = nOffsets
    c.height = nSamples
    const ctx = c.getContext('2d')
    if (!ctx) continue

    const img = ctx.createImageData(nOffsets, nSamples)
    const px = img.data
    const gBase = g * nOffsets * nSamples

    for (let y = 0; y < nSamples; y++) {
      const rowOff = y * nOffsets
      for (let x = 0; x < nOffsets; x++) {
        const val = d[gBase + x * nSamples + y]
        let t = (val + clip) * invTwoClip
        if (t < 0) t = 0
        else if (t > 1) t = 1
        const cIdx = ((t * 255) | 0) * 4
        const o = (rowOff + x) * 4
        px[o] = lut[cIdx]
        px[o + 1] = lut[cIdx + 1]
        px[o + 2] = lut[cIdx + 2]
        px[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }
}

function rebuildGatherOverlayTextures() {
  const dataObj = gp.value
  const ov = store.activeOverlay
  const s = survey.value
  if (!dataObj || !ov || !s) return

  const [actualCount, nOffsets, nSamples] = dataObj.shape
  if (actualCount <= 0 || nOffsets <= 0 || nSamples <= 0) return

  const lut = getOverlayLUT(store.overlayColormap)
  const smooth = store.overlayInterp === 'smooth'
  const [nIlP, nXlP, nTP] = ov.gridShape
  const strideIl = nXlP * nTP
  const d = ov.data
  const thresh = ov.threshold
  const normMin = Math.min(thresh, ov.max - 1e-4)
  const normSpan = Math.max(1e-5, ov.max - normMin)

  const [nIl, nXl] = s.shape
  const ilLook = buildAxisLookup(nIl, nIlP, ov.patchSize, ov.ilPatchStart, smooth)
  const xlLook = buildAxisLookup(nXl, nXlP, ov.patchSize, ov.xlPatchStart, smooth)
  const tLook = buildAxisLookup(nSamples, nTP, ov.patchSize, ov.tPatchStart, smooth)

  for (let g = 0; g < actualCount; g++) {
    ensureOffscreenPair(g)
    const c = overlayCanvases[g]
    c.width = 1
    c.height = nSamples
    const ctx = c.getContext('2d')
    if (!ctx) continue

    const loc = dataObj.locations[g]
    if (!loc) continue

    const ilI = Math.min(nIl - 1, Math.max(0, loc.il_idx))
    const xlI = Math.min(nXl - 1, Math.max(0, loc.xl_idx))

    const i0 = ilLook.i0[ilI]
    const i1 = ilLook.i1[ilI]
    const wi = ilLook.w1[ilI]
    const j0 = xlLook.i0[xlI]
    const j1 = xlLook.i1[xlI]
    const wj = xlLook.w1[xlI]

    const img = ctx.createImageData(1, nSamples)
    const px = img.data

    const b00 = i0 * strideIl + j0 * nTP
    const b01 = i0 * strideIl + j1 * nTP
    const b10 = i1 * strideIl + j0 * nTP
    const b11 = i1 * strideIl + j1 * nTP

    for (let y = 0; y < nSamples; y++) {
      const k0 = tLook.i0[y]
      const k1 = tLook.i1[y]
      const wk = tLook.w1[y]

      const v00 = d[b00 + k0] + (d[b00 + k1] - d[b00 + k0]) * wk
      const v01 = d[b01 + k0] + (d[b01 + k1] - d[b01 + k0]) * wk
      const v10 = d[b10 + k0] + (d[b10 + k1] - d[b10 + k0]) * wk
      const v11 = d[b11 + k0] + (d[b11 + k1] - d[b11 + k0]) * wk

      const v0 = v00 + (v01 - v00) * wj
      const v1 = v10 + (v11 - v10) * wj
      const val = v0 + (v1 - v0) * wi

      if (val >= thresh) {
        let norm = (val - normMin) / normSpan
        if (norm < 0) norm = 0
        else if (norm > 1) norm = 1
        const cIdx = ((norm * 255) | 0) * 4
        const o = y * 4
        px[o] = lut[cIdx]
        px[o + 1] = lut[cIdx + 1]
        px[o + 2] = lut[cIdx + 2]
        px[o + 3] = 255
      }
    }

    ctx.putImageData(img, 0, 0)
  }
}

function getLayoutGeom(cssW: number, cssH: number) {
  const plotX = MARGIN_LEFT
  const plotY = MARGIN_TOP
  const plotW = Math.max(10, cssW - MARGIN_LEFT - MARGIN_RIGHT)
  const plotH = Math.max(10, cssH - MARGIN_TOP - MARGIN_BOTTOM)

  const count = gp.value?.shape[0] ?? 1
  const totalContentW = plotW * zoomX.value
  const totalContentH = plotH * zoomY.value
  const startX = plotX + panX.value
  const startY = plotY + panY.value

  const totalGutter = GUTTER * Math.max(0, count - 1)
  const gatherW = Math.max(8, (totalContentW - totalGutter) / Math.max(1, count))

  return { plotX, plotY, plotW, plotH, startX, startY, gatherW, totalContentH, count }
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
  ctx.fillStyle = '#020617'
  ctx.fillRect(0, 0, cssW, cssH)

  const dataObj = gp.value
  const s = survey.value
  if (!dataObj || !s) {
    ctx.restore()
    return
  }

  const [actualCount, nOffsets, nSamples] = dataObj.shape
  const { plotX, plotY, plotW, plotH, startX, startY, gatherW, totalContentH } = getLayoutGeom(
    cssW,
    cssH,
  )

  // Plot background
  ctx.fillStyle = '#090d16'
  ctx.fillRect(plotX, plotY, plotW, plotH)

  // Draw gathers inside plot clip
  ctx.save()
  ctx.beginPath()
  ctx.rect(plotX, plotY, plotW, plotH)
  ctx.clip()

  const clipAmp = Math.max(1e-6, store.effectiveClipAmplitude)

  for (let g = 0; g < actualCount; g++) {
    const gx = startX + g * (gatherW + GUTTER)
    if (gx + gatherW < plotX || gx > plotX + plotW) continue

    const sc = seismicCanvases[g]
    if (sc) {
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(sc, gx, startY, gatherW, totalContentH)
    }

    if (store.activeOverlay && overlayCanvases[g]) {
      ctx.save()
      ctx.globalAlpha = store.overlayOpacity
      ctx.imageSmoothingEnabled = store.overlayInterp === 'smooth'
      ctx.drawImage(overlayCanvases[g], gx, startY, gatherW, totalContentH)
      ctx.restore()
    }

    // Optional wiggle overlay
    if (store.gatherWiggle && nOffsets > 0 && nSamples > 0) {
      const traceStep = Math.max(1, Math.floor(nOffsets / Math.min(20, Math.max(6, Math.floor(gatherW / 9)))))
      const traceAmpPx = (gatherW / nOffsets) * traceStep * 0.45
      const sampleStep = Math.max(1, Math.floor(nSamples / Math.min(450, Math.max(120, Math.floor(totalContentH)))))
      const gBase = g * nOffsets * nSamples
      const d = dataObj.data

      ctx.save()
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.55)'
      ctx.lineWidth = 0.75
      for (let off = 0; off < nOffsets; off += traceStep) {
        const baseX = gx + ((off + 0.5) / nOffsets) * gatherW
        ctx.beginPath()
        for (let samp = 0; samp < nSamples; samp += sampleStep) {
          const v = Math.max(-1, Math.min(1, d[gBase + off * nSamples + samp] / clipAmp))
          const wx = baseX + v * traceAmpPx
          const wy = startY + ((samp + 0.5) / nSamples) * totalContentH
          if (samp === 0) ctx.moveTo(wx, wy)
          else ctx.lineTo(wx, wy)
        }
        ctx.stroke()
      }
      ctx.restore()
    }

    // Active offset hairline
    if (store.gatherHighlightActiveOffset && nOffsets > 0) {
      const ox = gx + ((store.offsetIndex + 0.5) / nOffsets) * gatherW
      ctx.save()
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)'
      ctx.lineWidth = 1.1
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(ox, startY)
      ctx.lineTo(ox, startY + totalContentH)
      ctx.stroke()
      ctx.restore()
    }

    // Feature pick reticle if matching gather location
    const loc = dataObj.locations[g]
    if (store.featurePick && loc) {
      const fp = store.featurePick
      const dist = Math.hypot(fp.il_idx - loc.il_idx, fp.xl_idx - loc.xl_idx)
      if (dist <= 3) {
        const fx = gx + ((fp.offset_idx + 0.5) / nOffsets) * gatherW
        const fy = startY + ((fp.sample_idx + 0.5) / nSamples) * totalContentH
        const exact = dist === 0
        ctx.save()
        ctx.strokeStyle = exact ? '#22d3ee' : 'rgba(251, 191, 36, 0.8)'
        ctx.lineWidth = exact ? 2 : 1.25
        if (!exact) ctx.setLineDash([3, 2])
        ctx.beginPath()
        ctx.arc(fx, fy, 7, 0, Math.PI * 2)
        ctx.stroke()
        ctx.fillStyle = exact ? '#22d3ee' : '#fbbf24'
        ctx.beginPath()
        ctx.arc(fx, fy, 2.2, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      }
    }

    // Crisp border around each gather
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    ctx.strokeRect(gx + 0.5, startY + 0.5, gatherW - 1, totalContentH - 1)
  }

  // Horizontal time cursor line across entire panel
  const cy = startY + ((store.sampleIdx + 0.5) / Math.max(1, nSamples)) * totalContentH
  ctx.save()
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 4])
  ctx.beginPath()
  ctx.moveTo(plotX, cy)
  ctx.lineTo(plotX + plotW, cy)
  ctx.stroke()
  ctx.restore()

  ctx.restore() // end plot clip

  // Top headers per gather (clipped horizontally to plotX..plotX+plotW)
  ctx.save()
  ctx.beginPath()
  ctx.rect(plotX, 0, plotW, MARGIN_TOP)
  ctx.clip()

  const minOff = s.offsets[0] ?? 0
  const maxOff = s.offsets[nOffsets - 1] ?? nOffsets - 1

  for (let g = 0; g < actualCount; g++) {
    const gx = startX + g * (gatherW + GUTTER)
    if (gx + gatherW < plotX || gx > plotX + plotW) continue
    const loc = dataObj.locations[g]
    const cx = gx + gatherW * 0.5

    // Gather header pill
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(gx, 4, gatherW, 18)
    ctx.strokeStyle = '#1e293b'
    ctx.strokeRect(gx + 0.5, 4.5, gatherW - 1, 17)

    ctx.fillStyle = '#e2e8f0'
    ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    const label = loc ? `${loc.il}/${loc.xl}` : `#${g + 1}`
    if (gatherW >= 54) {
      ctx.fillText(label, cx, 13)
    } else if (gatherW >= 24) {
      ctx.fillText(`${g + 1}`, cx, 13)
    }

    // Offset sub-ticks
    if (gatherW >= 72) {
      ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'
      ctx.fillStyle = '#64748b'
      ctx.textAlign = 'left'
      ctx.fillText(`${minOff}m`, gx + 2, 31)
      ctx.textAlign = 'right'
      ctx.fillText(`${maxOff}m`, gx + gatherW - 2, 31)
    }
  }
  ctx.restore()

  // Left Time Ruler
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, plotY, MARGIN_LEFT, plotH)
  ctx.clip()
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.fillStyle = '#94a3b8'
  ctx.strokeStyle = '#334155'
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'

  const numTicks = Math.max(4, Math.floor(plotH / 70))
  for (let i = 0; i <= numTicks; i++) {
    const sy = plotY + (i / numTicks) * plotH
    const frac = (sy - startY) / totalContentH
    const sIdx = Math.round(frac * nSamples - 0.5)
    if (sIdx < 0 || sIdx >= nSamples) continue
    ctx.beginPath()
    ctx.moveTo(plotX, sy)
    ctx.lineTo(plotX - 4, sy)
    ctx.stroke()
    ctx.fillText(`${s.samples[sIdx] ?? sIdx}`, plotX - 7, sy)
  }
  ctx.restore()

  // Corner unit label
  ctx.save()
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.fillStyle = '#64748b'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'bottom'
  ctx.fillText('ms', 8, plotY - 6)
  ctx.restore()

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

function screenToGatherCoords(
  clientX: number,
  clientY: number,
): { gatherIdx: number; offIdx: number; sampleIdx: number } | null {
  const canvas = canvasRef.value
  const dataObj = gp.value
  if (!canvas || !dataObj) return null
  const rect = canvas.getBoundingClientRect()
  const mx = clientX - rect.left
  const my = clientY - rect.top
  const { plotX, plotY, plotW, plotH, startX, startY, gatherW, totalContentH } = getLayoutGeom(
    rect.width,
    rect.height,
  )
  if (mx < plotX || mx > plotX + plotW || my < plotY || my > plotY + plotH) return null

  const [actualCount, nOffsets, nSamples] = dataObj.shape
  const sampleIdx = Math.floor(((my - startY) / totalContentH) * nSamples)
  if (sampleIdx < 0 || sampleIdx >= nSamples) return null

  const stride = gatherW + GUTTER
  const relX = mx - startX
  if (relX < 0) return null
  const gatherIdx = Math.floor(relX / stride)
  if (gatherIdx < 0 || gatherIdx >= actualCount) return null
  const inGatherX = relX - gatherIdx * stride
  if (inGatherX < 0 || inGatherX > gatherW) return null

  const offIdx = Math.floor((inGatherX / gatherW) * nOffsets)
  if (offIdx < 0 || offIdx >= nOffsets) return null

  return { gatherIdx, offIdx, sampleIdx }
}

function onPointerDown(e: PointerEvent) {
  const canvas = canvasRef.value
  if (!canvas) return
  canvas.setPointerCapture(e.pointerId)
  pointerDownMoved = false
  panStartClientX = e.clientX
  panStartClientY = e.clientY
  panOriginX = panX.value
  panOriginY = panY.value
  if (e.button === 1 || e.button === 2 || e.altKey) {
    isPanning = true
  }
}

function onPointerMove(e: PointerEvent) {
  if (isPanning) {
    const dx = e.clientX - panStartClientX
    const dy = e.clientY - panStartClientY
    if (Math.hypot(dx, dy) > 3) pointerDownMoved = true
    panX.value = panOriginX + dx
    panY.value = panOriginY + dy
    scheduleRender()
    return
  }
  if (e.buttons === 1) {
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
  updateHover(e.clientX, e.clientY)
}

function onPointerUp(e: PointerEvent) {
  const canvas = canvasRef.value
  if (canvas && canvas.hasPointerCapture(e.pointerId)) {
    canvas.releasePointerCapture(e.pointerId)
  }
  if (isPanning) {
    isPanning = false
    if (pointerDownMoved) return
  }
  if (e.button !== 0 || pointerDownMoved) return

  const hit = screenToGatherCoords(e.clientX, e.clientY)
  if (!hit || !gp.value) return
  const loc = gp.value.locations[hit.gatherIdx]
  if (!loc) return

  const offForPick = store.gatherSyncOffsetOnPick ? hit.offIdx : undefined
  if (store.pickMode && !e.shiftKey) {
    void store.pickFeature(loc.il_idx, loc.xl_idx, hit.sampleIdx, offForPick ?? hit.offIdx, true)
    if (store.gatherSyncOffsetOnPick) {
      store.setCursor({ offsetIndex: hit.offIdx })
    }
  } else {
    store.setCursor({
      ilIdx: loc.il_idx,
      xlIdx: loc.xl_idx,
      sampleIdx: hit.sampleIdx,
      offsetIndex: store.gatherSyncOffsetOnPick ? hit.offIdx : undefined,
    })
  }
}

function updateHover(clientX: number, clientY: number) {
  const hit = screenToGatherCoords(clientX, clientY)
  const dataObj = gp.value
  const s = survey.value
  if (!hit || !dataObj || !s) {
    hoverInfo.value = null
    return
  }

  const loc = dataObj.locations[hit.gatherIdx]
  if (!loc) {
    hoverInfo.value = null
    return
  }
  const [, nOffsets, nSamples] = dataObj.shape
  const amp = dataObj.data[(hit.gatherIdx * nOffsets + hit.offIdx) * nSamples + hit.sampleIdx] ?? 0

  let sim: number | null = null
  const ov = store.activeOverlay
  if (ov) {
    const smooth = store.overlayInterp === 'smooth'
    const [nIlP, nXlP, nTP] = ov.gridShape
    const ilLook = buildAxisLookup(s.shape[0], nIlP, ov.patchSize, ov.ilPatchStart, smooth)
    const xlLook = buildAxisLookup(s.shape[1], nXlP, ov.patchSize, ov.xlPatchStart, smooth)
    const tLook = buildAxisLookup(s.shape[2], nTP, ov.patchSize, ov.tPatchStart, smooth)
    const i0 = ilLook.i0[loc.il_idx]
    const i1 = ilLook.i1[loc.il_idx]
    const wi = ilLook.w1[loc.il_idx]
    const j0 = xlLook.i0[loc.xl_idx]
    const j1 = xlLook.i1[loc.xl_idx]
    const wj = xlLook.w1[loc.xl_idx]
    const k0 = tLook.i0[hit.sampleIdx]
    const k1 = tLook.i1[hit.sampleIdx]
    const wk = tLook.w1[hit.sampleIdx]
    const stride = nXlP * nTP
    const d = ov.data
    const c00 = d[i0 * stride + j0 * nTP + k0] * (1 - wk) + d[i0 * stride + j0 * nTP + k1] * wk
    const c01 = d[i0 * stride + j1 * nTP + k0] * (1 - wk) + d[i0 * stride + j1 * nTP + k1] * wk
    const c10 = d[i1 * stride + j0 * nTP + k0] * (1 - wk) + d[i1 * stride + j0 * nTP + k1] * wk
    const c11 = d[i1 * stride + j1 * nTP + k0] * (1 - wk) + d[i1 * stride + j1 * nTP + k1] * wk
    sim = (c00 * (1 - wj) + c01 * wj) * (1 - wi) + (c10 * (1 - wj) + c11 * wj) * wi
  }

  hoverInfo.value = {
    gatherIdx: hit.gatherIdx + 1,
    il: loc.il,
    xl: loc.xl,
    offset: s.offsets[hit.offIdx] ?? hit.offIdx,
    sample: s.samples[hit.sampleIdx] ?? hit.sampleIdx,
    amp,
    sim,
  }
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  zoomX.value = Math.max(0.8, Math.min(12, zoomX.value * factor))
  zoomY.value = Math.max(0.8, Math.min(12, zoomY.value * factor))
  scheduleRender()
}

function resetZoom() {
  zoomX.value = 1.0
  zoomY.value = 1.0
  panX.value = 0
  panY.value = 0
  scheduleRender()
}

let resizeObs: ResizeObserver | null = null

onMounted(() => {
  if (!store.gatherPanelData && store.activeSurvey) {
    void store.loadGatherPanel()
  }
  resizeCanvas()
  if (containerRef.value) {
    resizeObs = new ResizeObserver(() => resizeCanvas())
    resizeObs.observe(containerRef.value)
  }
})

onBeforeUnmount(() => {
  resizeObs?.disconnect()
})

watch(
  [gp, () => store.seismicColormap, () => store.effectiveClipAmplitude],
  () => {
    rebuildGatherSeismicTextures()
    rebuildGatherOverlayTextures()
    scheduleRender()
  },
  { immediate: true },
)

watch(
  [() => store.activeOverlay, () => store.overlayColormap, () => store.overlayInterp],
  () => {
    rebuildGatherOverlayTextures()
    scheduleRender()
  },
  { immediate: true },
)

watch(
  [
    () => store.overlayOpacity,
    () => store.offsetIndex,
    () => store.sampleIdx,
    () => store.featurePick,
    () => store.gatherWiggle,
    () => store.gatherHighlightActiveOffset,
  ],
  () => {
    scheduleRender()
  },
)
</script>

<template>
  <div class="flex flex-col h-full w-full bg-slate-950 select-none overflow-hidden">
    <!-- Dedicated Gather Panel Controls -->
    <div
      class="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 px-2.5 py-1.5 bg-slate-900/80 border-b border-slate-800/90 text-[11px] shrink-0"
    >
      <!-- Line direction -->
      <div class="flex items-center gap-1">
        <span class="text-slate-400">Line</span>
        <select
          v-model="store.gatherLineAxis"
          class="bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 text-slate-200 focus:outline-none focus:border-cyan-500"
          @change="store.loadGatherPanel()"
        >
          <option value="xline">Step XL (along Inline)</option>
          <option value="inline">Step IL (along Crossline)</option>
        </select>
      </div>

      <!-- Start IL / XL -->
      <div class="flex items-center gap-1">
        <span class="text-slate-400">IL idx</span>
        <input
          v-model.number="store.gatherIlStartIdx"
          type="number"
          min="0"
          :max="Math.max(0, (survey?.shape[0] ?? 1) - 1)"
          class="w-14 bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 font-mono-num text-slate-200 focus:outline-none focus:border-cyan-500"
          @change="store.loadGatherPanel()"
        />
        <span class="font-mono-num text-[10px] text-slate-500">({{ startIlHeader }})</span>
      </div>

      <div class="flex items-center gap-1">
        <span class="text-slate-400">XL idx</span>
        <input
          v-model.number="store.gatherXlStartIdx"
          type="number"
          min="0"
          :max="Math.max(0, (survey?.shape[1] ?? 1) - 1)"
          class="w-14 bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 font-mono-num text-slate-200 focus:outline-none focus:border-cyan-500"
          @change="store.loadGatherPanel()"
        />
        <span class="font-mono-num text-[10px] text-slate-500">({{ startXlHeader }})</span>
      </div>

      <button
        type="button"
        class="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-cyan-300 hover:bg-slate-800 transition-colors"
        title="Anchor Gather Panel start at current cursor (IL, XL)"
        @click="store.useCursorForGatherStart()"
      >
        <Crosshair class="w-3 h-3" />
        <span>Use Current Cursor</span>
      </button>

      <!-- Increment & Count -->
      <div class="flex items-center gap-1">
        <span class="text-slate-400">Inc</span>
        <input
          v-model.number="store.gatherIncrement"
          type="number"
          min="1"
          max="50"
          class="w-12 bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 font-mono-num text-slate-200 focus:outline-none focus:border-cyan-500"
          @change="store.loadGatherPanel()"
        />
      </div>

      <div class="flex items-center gap-1">
        <span class="text-slate-400">Gathers N</span>
        <input
          v-model.number="store.gatherCount"
          type="number"
          min="1"
          max="30"
          class="w-12 bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 font-mono-num text-slate-200 focus:outline-none focus:border-cyan-500"
          @change="store.loadGatherPanel()"
        />
      </div>

      <button
        type="button"
        class="flex items-center gap-1 px-2 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25 transition-colors"
        @click="store.loadGatherPanel()"
      >
        <RefreshCw class="w-3 h-3" :class="{ 'animate-spin': store.gatherLoading }" />
        <span>Fetch</span>
      </button>

      <!-- Checkboxes & Reset -->
      <div class="flex items-center gap-2.5 ml-auto">
        <label class="flex items-center gap-1 text-slate-300 cursor-pointer">
          <input
            v-model="store.gatherWiggle"
            type="checkbox"
            class="accent-cyan-400 rounded"
          />
          <span>Wiggle Overlay</span>
        </label>

        <label class="flex items-center gap-1 text-slate-300 cursor-pointer">
          <input
            v-model="store.gatherHighlightActiveOffset"
            type="checkbox"
            class="accent-cyan-400 rounded"
          />
          <span>Highlight Active Offset</span>
        </label>

        <label class="hidden xl:flex items-center gap-1 text-slate-400 cursor-pointer">
          <input
            v-model="store.gatherSyncOffsetOnPick"
            type="checkbox"
            class="accent-cyan-400 rounded"
          />
          <span>Sync Offset</span>
        </label>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300 hover:bg-slate-800 transition-colors"
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
        class="block w-full h-full cursor-crosshair"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointerleave="hoverInfo = null"
        @wheel="onWheel"
        @contextmenu.prevent
      />

      <div
        v-if="store.gatherLoading"
        class="absolute top-2 right-3 px-2 py-0.5 rounded bg-slate-900/90 border border-slate-700/80 text-[10px] font-mono-num text-cyan-300 pointer-events-none"
      >
        Loading gathers…
      </div>

      <!-- Hover tooltip -->
      <div
        v-if="hoverInfo"
        class="absolute bottom-2.5 left-16 px-2.5 py-1 rounded bg-slate-950/90 border border-slate-800 text-[11px] font-mono-num text-slate-200 pointer-events-none flex items-center gap-2.5 shadow-lg"
      >
        <span>Gather <b class="text-amber-300">#{{ hoverInfo.gatherIdx }}</b>:</span>
        <span>IL <b class="text-cyan-300">{{ hoverInfo.il }}</b></span>
        <span>XL <b class="text-cyan-300">{{ hoverInfo.xl }}</b></span>
        <span>Offset <b class="text-amber-300">{{ hoverInfo.offset }}</b> m</span>
        <span>T <b class="text-cyan-300">{{ hoverInfo.sample }}</b> ms</span>
        <span>Amp <b class="text-slate-100">{{ hoverInfo.amp.toFixed(3) }}</b></span>
        <span v-if="hoverInfo.sim !== null">
          {{ store.overlaySource === 'attribute' ? 'Attr' : 'Sim' }}
          <b class="text-emerald-300">{{ hoverInfo.sim.toFixed(3) }}</b>
        </span>
      </div>

      <div
        v-if="!survey"
        class="absolute inset-0 flex items-center justify-center text-xs text-slate-500"
      >
        No survey loaded
      </div>
    </div>
  </div>
</template>
