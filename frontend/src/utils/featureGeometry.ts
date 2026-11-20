import type { FeatureGeometryMode, FeatureRole } from '../api/types'

export const HANDLE_HIT_RADIUS_PX = 9
export const EDGE_HIT_RADIUS_PX = 7

export interface RoleColorTokens {
  primary: string
  primaryHex: number
  fill: string
  ghost: string
  haloHex: number
}

export const ROLE_COLORS: Record<FeatureRole, RoleColorTokens> = {
  include: {
    primary: '#22d3ee',
    primaryHex: 0x22d3ee,
    fill: 'rgba(34, 211, 238, 0.16)',
    ghost: 'rgba(34, 211, 238, 0.45)',
    haloHex: 0xfbbf24,
  },
  exclude: {
    primary: '#fb7185',
    primaryHex: 0xfb7185,
    fill: 'rgba(251, 113, 133, 0.16)',
    ghost: 'rgba(251, 113, 133, 0.45)',
    haloHex: 0xfb7185,
  },
}

export interface ScreenPoint {
  x: number
  y: number
}

export interface HandleCandidate {
  featureId: string
  pickIndex: number
  x: number
  y: number
}

export interface HandleHitResult {
  featureId: string
  pickIndex: number
  distance: number
}

export interface EdgeCandidate {
  featureId: string
  geometryMode: FeatureGeometryMode
  points: ScreenPoint[]
}

export interface EdgeHitResult {
  featureId: string
  segmentIndex: number
  distance: number
  projX: number
  projY: number
}

export function pointDistance(ax: number, ay: number, bx: number, by: number): number {
  return Math.hypot(ax - bx, ay - by)
}

export function pointToSegmentDistance(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): { distance: number; projX: number; projY: number; t: number } {
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy
  if (lenSq <= 1e-9) {
    return {
      distance: Math.hypot(px - ax, py - ay),
      projX: ax,
      projY: ay,
      t: 0,
    }
  }
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq))
  const projX = ax + t * dx
  const projY = ay + t * dy
  return {
    distance: Math.hypot(px - projX, py - projY),
    projX,
    projY,
    t,
  }
}

export function hitTestHandles(
  mx: number,
  my: number,
  candidates: HandleCandidate[],
  radiusPx = HANDLE_HIT_RADIUS_PX,
): HandleHitResult | null {
  let best: HandleHitResult | null = null
  for (const c of candidates) {
    const d = pointDistance(mx, my, c.x, c.y)
    if (d <= radiusPx && (!best || d < best.distance)) {
      best = {
        featureId: c.featureId,
        pickIndex: c.pickIndex,
        distance: d,
      }
    }
  }
  return best
}

export function hitTestEdges(
  mx: number,
  my: number,
  candidates: EdgeCandidate[],
  radiusPx = EDGE_HIT_RADIUS_PX,
): EdgeHitResult | null {
  let best: EdgeHitResult | null = null
  for (const c of candidates) {
    const pts = c.points
    const n = pts.length
    if (n < 2) continue
    if (c.geometryMode !== 'polyline' && c.geometryMode !== 'polygon') continue

    // Completed polyline: segments 0..n-2
    // Completed polygon: if n >= 3, closed segments 0..n-1; if n === 2, single segment 0..1
    const segCount = c.geometryMode === 'polygon' && n >= 3 ? n : n - 1
    for (let i = 0; i < segCount; i++) {
      const a = pts[i]
      const b = pts[(i + 1) % n]
      const res = pointToSegmentDistance(mx, my, a.x, a.y, b.x, b.y)
      if (res.distance <= radiusPx && (!best || res.distance < best.distance)) {
        best = {
          featureId: c.featureId,
          segmentIndex: i,
          distance: res.distance,
          projX: res.projX,
          projY: res.projY,
        }
      }
    }
  }
  return best
}

/**
 * Matches backend `_find_patch_idx(starts, patch_size, idx)`:
 * First hit where `starts[p] <= idx < starts[p] + patchSize`, else nearest center `starts[p] + patchSize / 2`.
 */
export function findPatchIdx(starts: number[], patchSize: number, idx: number): number {
  if (!starts || starts.length === 0) return 0
  for (let p = 0; p < starts.length; p++) {
    const s = starts[p]
    if (s <= idx && idx < s + patchSize) {
      return p
    }
  }
  let bestP = 0
  let bestDist = Infinity
  const half = patchSize * 0.5
  for (let p = 0; p < starts.length; p++) {
    const center = starts[p] + half
    const d = Math.abs(center - idx)
    if (d < bestDist) {
      bestDist = d
      bestP = p
    }
  }
  return bestP
}
