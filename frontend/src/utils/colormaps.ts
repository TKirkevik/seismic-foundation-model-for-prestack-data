import type { OverlayColormapId, SeismicColormapId } from '../api/types'

export type ColorLUT = Uint8ClampedArray // 256 * 4 bytes [r,g,b,a,...]

type RGBStop = [number, number, number]

function buildLUT(stops: RGBStop[]): ColorLUT {
  const lut = new Uint8ClampedArray(256 * 4)
  const lastIdx = stops.length - 1
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const scaled = t * lastIdx
    const idx0 = Math.min(lastIdx - 1, Math.floor(scaled))
    const f = scaled - idx0
    const c0 = stops[idx0]
    const c1 = stops[idx0 + 1]
    const off = i * 4
    lut[off] = Math.round(c0[0] + (c1[0] - c0[0]) * f)
    lut[off + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * f)
    lut[off + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * f)
    lut[off + 3] = 255
  }
  return lut
}

// Diverging Blue - White - Red seismic amplitude colormap
const RWB_LUT = buildLUT([
  [8, 48, 107],
  [33, 113, 181],
  [107, 174, 214],
  [247, 250, 252],
  [251, 106, 74],
  [203, 24, 29],
  [103, 0, 13],
])

// Classic high-contrast seismic grayscale
const GRAYSCALE_LUT = buildLUT([
  [8, 10, 14],
  [128, 132, 138],
  [248, 250, 252],
])

// Cool steel-bone seismic grayscale
const BONE_LUT = buildLUT([
  [5, 8, 15],
  [68, 82, 104],
  [146, 174, 192],
  [242, 248, 252],
])

// Perceptually uniform Inferno
const INFERNO_LUT = buildLUT([
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
])

// Turbo rainbow-like perceptually smooth LUT
const TURBO_LUT = buildLUT([
  [48, 18, 59],
  [65, 69, 171],
  [41, 143, 232],
  [26, 206, 172],
  [124, 238, 86],
  [226, 208, 49],
  [249, 125, 29],
  [199, 42, 19],
  [122, 4, 3],
])

// Viridis
const VIRIDIS_LUT = buildLUT([
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
])

// Hot
const HOT_LUT = buildLUT([
  [18, 0, 0],
  [140, 0, 0],
  [230, 62, 0],
  [255, 165, 0],
  [255, 235, 95],
  [255, 255, 245],
])

export const SEISMIC_LUTS: Record<SeismicColormapId, ColorLUT> = {
  rwb: RWB_LUT,
  grayscale: GRAYSCALE_LUT,
  bone: BONE_LUT,
}

export const OVERLAY_LUTS: Record<OverlayColormapId, ColorLUT> = {
  inferno: INFERNO_LUT,
  turbo: TURBO_LUT,
  viridis: VIRIDIS_LUT,
  hot: HOT_LUT,
}

export function getSeismicLUT(id: SeismicColormapId): ColorLUT {
  return SEISMIC_LUTS[id] ?? RWB_LUT
}

export function getOverlayLUT(id: OverlayColormapId): ColorLUT {
  return OVERLAY_LUTS[id] ?? INFERNO_LUT
}

export function sampleLUT(lut: ColorLUT, norm01: number): [number, number, number] {
  const clamped = norm01 <= 0 ? 0 : norm01 >= 1 ? 1 : norm01
  const idx = Math.min(255, Math.max(0, Math.round(clamped * 255))) * 4
  return [lut[idx], lut[idx + 1], lut[idx + 2]]
}

export function colormapGradientCss(lut: ColorLUT): string {
  const stops: string[] = []
  for (let i = 0; i <= 8; i++) {
    const idx = Math.round((i / 8) * 255) * 4
    stops.push(`rgb(${lut[idx]}, ${lut[idx + 1]}, ${lut[idx + 2]}) ${(i / 8) * 100}%`)
  }
  return `linear-gradient(to right, ${stops.join(', ')})`
}
