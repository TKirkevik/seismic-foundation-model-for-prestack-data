import type { WellPick } from '../api/types'
import type { HorizonGrid, ReservoirEnvelope } from '../stores/workbench'

export function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.replace(/(.)/g, '$1$1') : h, 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}

export interface SectionLineContext {
  ctx: CanvasRenderingContext2D
  /** 'inline' => x runs over crossline indices; 'xline' => x runs over inline indices */
  axis: 'inline' | 'xline'
  fixedIdx: number
  nIl: number
  nXl: number
  twtToSample: (twt: number) => number
  /** continuous grid x (bin index, 0.5 = bin centre) -> screen x */
  toScreenX: (gx: number) => number
  /** continuous sample index -> screen y */
  toScreenY: (gy: number) => number
  opacity: number
}

function sectionValue(grid: Float32Array, c: SectionLineContext, x: number): number {
  return c.axis === 'inline' ? grid[c.fixedIdx * c.nXl + x] : grid[x * c.nXl + c.fixedIdx]
}

function tracePolyline(
  c: SectionLineContext,
  grid: Float32Array,
  onSegment: (xs: number[], ys: number[]) => void,
) {
  const n = c.axis === 'inline' ? c.nXl : c.nIl
  let xs: number[] = []
  let ys: number[] = []
  for (let x = 0; x < n; x++) {
    const v = sectionValue(grid, c, x)
    if (Number.isFinite(v)) {
      xs.push(c.toScreenX(x))
      ys.push(c.toScreenY(c.twtToSample(v)))
    } else if (xs.length) {
      onSegment(xs, ys)
      xs = []
      ys = []
    }
  }
  if (xs.length) onSegment(xs, ys)
}

function strokePolyline(ctx: CanvasRenderingContext2D, xs: number[], ys: number[]) {
  ctx.beginPath()
  for (let i = 0; i < xs.length; i++) {
    if (i === 0) ctx.moveTo(xs[i], ys[i])
    else ctx.lineTo(xs[i], ys[i])
  }
  ctx.stroke()
}

/** Fill the reservoir interval and stroke top/base along a vertical section. */
export function drawEnvelopeSection(c: SectionLineContext, env: ReservoirEnvelope) {
  const { ctx } = c
  const n = c.axis === 'inline' ? c.nXl : c.nIl
  ctx.save()
  ctx.fillStyle = hexToRgba(env.color, c.opacity)
  let run: Array<[number, number, number]> = [] // [sx, yTop, yBase]
  const flush = () => {
    if (run.length < 1) return
    ctx.beginPath()
    run.forEach(([sx, yt], i) => (i === 0 ? ctx.moveTo(sx, yt) : ctx.lineTo(sx, yt)))
    for (let i = run.length - 1; i >= 0; i--) ctx.lineTo(run[i][0], run[i][2])
    ctx.closePath()
    ctx.fill()
    run = []
  }
  for (let x = 0; x < n; x++) {
    const t = sectionValue(env.top, c, x)
    const b = sectionValue(env.base, c, x)
    if (Number.isFinite(t) && Number.isFinite(b)) {
      const sx = c.toScreenX(x)
      run.push([sx, c.toScreenY(c.twtToSample(t)), c.toScreenY(c.twtToSample(b))])
    } else {
      flush()
    }
  }
  flush()

  ctx.strokeStyle = hexToRgba(env.color, Math.min(1, c.opacity + 0.5))
  ctx.lineWidth = 1.5
  tracePolyline(c, env.top, (xs, ys) => strokePolyline(ctx, xs, ys))
  ctx.lineWidth = 1
  ctx.setLineDash([5, 3])
  tracePolyline(c, env.base, (xs, ys) => strokePolyline(ctx, xs, ys))
  ctx.restore()
}

export function drawHorizonSection(c: SectionLineContext, hz: HorizonGrid) {
  const { ctx } = c
  ctx.save()
  ctx.strokeStyle = hexToRgba(hz.color, 0.95)
  ctx.lineWidth = 1.25
  ctx.setLineDash([2, 3])
  tracePolyline(c, hz.twt, (xs, ys) => strokePolyline(ctx, xs, ys))
  ctx.restore()
}

/** Well pick markers on a vertical section: solid when on the plane, faded within `near` bins. */
export function drawWellPicksSection(c: SectionLineContext, picks: WellPick[], near = 6) {
  const { ctx } = c
  ctx.save()
  ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.textBaseline = 'middle'
  for (const p of picks) {
    const along = c.axis === 'inline' ? p.xl_idx : p.il_idx
    const across = c.axis === 'inline' ? p.il_idx : p.xl_idx
    const d = Math.abs(across - c.fixedIdx)
    if (d > near) continue
    const sx = c.toScreenX(along)
    const sy = c.toScreenY(c.twtToSample(p.twt))
    const alpha = d === 0 ? 1 : 0.35 + 0.5 * (1 - d / near)
    const isTop = p.formation.includes('Top')
    ctx.strokeStyle = `rgba(253, 224, 71, ${alpha})`
    ctx.fillStyle = `rgba(253, 224, 71, ${alpha})`
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(sx - 6, sy)
    ctx.lineTo(sx + 6, sy)
    ctx.stroke()
    if (isTop) {
      ctx.beginPath()
      ctx.moveTo(sx, sy - 5)
      ctx.lineTo(sx - 3, sy)
      ctx.lineTo(sx + 3, sy)
      ctx.closePath()
      ctx.fill()
    }
    if (d <= 1 && isTop) {
      ctx.textAlign = 'left'
      ctx.fillText(p.well, sx + 8, sy - 6)
    }
  }
  ctx.restore()
}

export interface TimeSliceContext {
  ctx: CanvasRenderingContext2D
  nIl: number
  nXl: number
  /** two-way time (ms) of the displayed slice */
  twt: number
  /** ms half-width within which a horizon counts as intersecting the slice */
  halfWindowMs: number
  /** continuous crossline index -> screen x */
  toScreenX: (gx: number) => number
  /** continuous inline index -> screen y */
  toScreenY: (gy: number) => number
  opacity: number
}

/**
 * Paint a (n_xl x n_il) RGBA image: filled where the slice time is inside the reservoir
 * interval, and horizon outcrops where |twt - t| <= halfWindow. Returns false if nothing to draw.
 */
export function paintTimeSliceImage(
  canvas: HTMLCanvasElement,
  nIl: number,
  nXl: number,
  twt: number,
  halfWindowMs: number,
  env: ReservoirEnvelope | null,
  horizons: HorizonGrid[],
): boolean {
  if (!env && horizons.length === 0) return false
  canvas.width = nXl
  canvas.height = nIl
  const ctx = canvas.getContext('2d')
  if (!ctx) return false
  const img = ctx.createImageData(nXl, nIl)
  const px = img.data
  const parse = (hex: string) => {
    const n = parseInt(hex.replace('#', ''), 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  }
  const envRgb = env ? parse(env.color) : null
  const hzRgb = horizons.map((h) => parse(h.color))
  let any = false
  for (let i = 0; i < nIl; i++) {
    for (let j = 0; j < nXl; j++) {
      const g = i * nXl + j
      const o = g * 4
      if (env && envRgb) {
        const t = env.top[g]
        const b = env.base[g]
        if (Number.isFinite(t) && Number.isFinite(b) && twt >= t && twt <= b) {
          px[o] = envRgb[0]
          px[o + 1] = envRgb[1]
          px[o + 2] = envRgb[2]
          px[o + 3] = 255
          any = true
        }
      }
      for (let h = 0; h < horizons.length; h++) {
        const v = horizons[h].twt[g]
        if (Number.isFinite(v) && Math.abs(v - twt) <= halfWindowMs) {
          px[o] = hzRgb[h][0]
          px[o + 1] = hzRgb[h][1]
          px[o + 2] = hzRgb[h][2]
          px[o + 3] = 255
          any = true
        }
      }
    }
  }
  if (any) ctx.putImageData(img, 0, 0)
  return any
}

export function drawWellPicksTimeSlice(c: TimeSliceContext, picks: WellPick[], windowMs: number) {
  const { ctx } = c
  ctx.save()
  ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'bottom'
  for (const p of picks) {
    if (!p.formation.includes('Top')) continue
    const sx = c.toScreenX(p.xl_idx)
    const sy = c.toScreenY(p.il_idx)
    const close = Math.abs(p.twt - c.twt) <= windowMs
    ctx.strokeStyle = close ? 'rgba(253, 224, 71, 1)' : 'rgba(253, 224, 71, 0.45)'
    ctx.lineWidth = close ? 1.5 : 1
    ctx.beginPath()
    ctx.arc(sx, sy, close ? 4 : 2.5, 0, Math.PI * 2)
    ctx.stroke()
    if (close) {
      ctx.fillStyle = 'rgba(253, 224, 71, 0.95)'
      ctx.fillText(p.well, sx + 6, sy - 2)
    }
  }
  ctx.restore()
}

/** Reservoir top/base TWT at a bin, or null when undefined there. */
export function envelopeAt(
  env: ReservoirEnvelope,
  nXl: number,
  ilIdx: number,
  xlIdx: number,
): { top: number; base: number } | null {
  const g = ilIdx * nXl + xlIdx
  const t = env.top[g]
  const b = env.base[g]
  return Number.isFinite(t) && Number.isFinite(b) ? { top: t, base: b } : null
}
