export interface PatchVolumeSpec {
  data: Float32Array
  gridShape: [number, number, number]
  patchSize: number
  ilPatchStart?: number[]
  xlPatchStart?: number[]
  tPatchStart?: number[]
}

export interface AxisLookup {
  i0: Uint16Array
  i1: Uint16Array
  w1: Float32Array
}

function computeCenters(gridLen: number, patchSize: number, patchStarts?: number[]): Float32Array {
  const centers = new Float32Array(gridLen)
  const half = patchSize * 0.5
  if (patchStarts && patchStarts.length >= gridLen) {
    for (let p = 0; p < gridLen; p++) {
      centers[p] = patchStarts[p] + half
    }
  } else {
    for (let p = 0; p < gridLen; p++) {
      centers[p] = p * patchSize + half
    }
  }
  return centers
}

export function buildAxisLookup(
  sampleCount: number,
  gridLen: number,
  patchSize: number,
  patchStarts?: number[],
  smooth = true,
): AxisLookup {
  const i0 = new Uint16Array(sampleCount)
  const i1 = new Uint16Array(sampleCount)
  const w1 = new Float32Array(sampleCount)

  if (gridLen <= 0) {
    return { i0, i1, w1 }
  }
  if (gridLen === 1) {
    return { i0, i1, w1 }
  }

  const centers = computeCenters(gridLen, patchSize, patchStarts)

  if (!smooth) {
    let p = 0
    for (let s = 0; s < sampleCount; s++) {
      const sc = s + 0.5
      while (p + 1 < gridLen && Math.abs(centers[p + 1] - sc) < Math.abs(centers[p] - sc)) {
        p++
      }
      i0[s] = p
      i1[s] = p
      w1[s] = 0
    }
    return { i0, i1, w1 }
  }

  let p = 0
  for (let s = 0; s < sampleCount; s++) {
    const sc = s + 0.5
    if (sc <= centers[0]) {
      i0[s] = 0
      i1[s] = 0
      w1[s] = 0
      continue
    }
    if (sc >= centers[gridLen - 1]) {
      const last = gridLen - 1
      i0[s] = last
      i1[s] = last
      w1[s] = 0
      continue
    }
    while (p + 1 < gridLen - 1 && centers[p + 1] < sc) {
      p++
    }
    const c0 = centers[p]
    const c1 = centers[p + 1]
    const span = c1 - c0
    i0[s] = p
    i1[s] = p + 1
    w1[s] = span > 1e-6 ? Math.max(0, Math.min(1, (sc - c0) / span)) : 0
  }

  return { i0, i1, w1 }
}

function findBracket(
  sc: number,
  gridLen: number,
  patchSize: number,
  patchStarts?: number[],
  smooth = true,
): [number, number, number] {
  if (gridLen <= 1) return [0, 0, 0]
  const half = patchSize * 0.5
  const centerAt = (p: number) =>
    patchStarts && patchStarts.length > p ? patchStarts[p] + half : p * patchSize + half

  if (!smooth) {
    let best = 0
    let bestDist = Math.abs(centerAt(0) - sc)
    for (let p = 1; p < gridLen; p++) {
      const d = Math.abs(centerAt(p) - sc)
      if (d < bestDist) {
        best = p
        bestDist = d
      }
    }
    return [best, best, 0]
  }

  const first = centerAt(0)
  if (sc <= first) return [0, 0, 0]
  const lastIdx = gridLen - 1
  const last = centerAt(lastIdx)
  if (sc >= last) return [lastIdx, lastIdx, 0]

  let lo = 0
  let hi = lastIdx
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (centerAt(mid) <= sc) lo = mid
    else hi = mid
  }
  const c0 = centerAt(lo)
  const c1 = centerAt(hi)
  const span = c1 - c0
  const w = span > 1e-6 ? Math.max(0, Math.min(1, (sc - c0) / span)) : 0
  return [lo, hi, w]
}

export function samplePatchVolume(
  vol: PatchVolumeSpec,
  ilIdx: number,
  xlIdx: number,
  sampleIdx: number,
  smooth = true,
): number {
  const [nIlP, nXlP, nTP] = vol.gridShape
  if (nIlP === 0 || nXlP === 0 || nTP === 0 || vol.data.length === 0) return NaN

  const [i0, i1, wi] = findBracket(ilIdx + 0.5, nIlP, vol.patchSize, vol.ilPatchStart, smooth)
  const [j0, j1, wj] = findBracket(xlIdx + 0.5, nXlP, vol.patchSize, vol.xlPatchStart, smooth)
  const [k0, k1, wk] = findBracket(sampleIdx + 0.5, nTP, vol.patchSize, vol.tPatchStart, smooth)

  const strideIl = nXlP * nTP
  const d = vol.data

  const c000 = d[i0 * strideIl + j0 * nTP + k0]
  const c001 = d[i0 * strideIl + j0 * nTP + k1]
  const c010 = d[i0 * strideIl + j1 * nTP + k0]
  const c011 = d[i0 * strideIl + j1 * nTP + k1]
  const c100 = d[i1 * strideIl + j0 * nTP + k0]
  const c101 = d[i1 * strideIl + j0 * nTP + k1]
  const c110 = d[i1 * strideIl + j1 * nTP + k0]
  const c111 = d[i1 * strideIl + j1 * nTP + k1]

  const c00 = c000 + (c001 - c000) * wk
  const c01 = c010 + (c011 - c010) * wk
  const c10 = c100 + (c101 - c100) * wk
  const c11 = c110 + (c111 - c110) * wk

  const c0 = c00 + (c01 - c00) * wj
  const c1 = c10 + (c11 - c10) * wj

  return c0 + (c1 - c0) * wi
}
