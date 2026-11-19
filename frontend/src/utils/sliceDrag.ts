// Geometry for dragging the inline / crossline / time-slice planes in the 3D cube viewer.
// Mirrors Volume3DViewer's indexToWorld(): inline runs along world Z, crossline along X,
// and samples along Y with time increasing downwards.

export type SliceAxis = 'inline' | 'xline' | 'sample'

export interface SliceDims {
  nIl: number
  nXl: number
  nSamples: number
  sizeX: number
  sizeY: number
  sizeZ: number
}

/** Unit world direction each slice plane moves along (its normal). */
export const SLICE_AXIS_DIR: Record<SliceAxis, [number, number, number]> = {
  inline: [0, 0, 1],
  xline: [1, 0, 0],
  sample: [0, 1, 0],
}

function axisCount(axis: SliceAxis, d: SliceDims): number {
  return Math.max(1, axis === 'inline' ? d.nIl : axis === 'xline' ? d.nXl : d.nSamples)
}

/** World coordinate along the slice's axis for a slice index (cell centre). */
export function sliceIndexToWorld(axis: SliceAxis, index: number, d: SliceDims): number {
  const n = axisCount(axis, d)
  const frac = (index + 0.5) / n
  if (axis === 'inline') return (frac - 0.5) * d.sizeZ
  if (axis === 'xline') return (frac - 0.5) * d.sizeX
  return (0.5 - frac) * d.sizeY
}

/** Nearest slice index for a world coordinate along the slice's axis, clamped to the volume. */
export function worldToSliceIndex(axis: SliceAxis, coord: number, d: SliceDims): number {
  const n = axisCount(axis, d)
  let frac: number
  if (axis === 'inline') frac = coord / d.sizeZ + 0.5
  else if (axis === 'xline') frac = coord / d.sizeX + 0.5
  else frac = 0.5 - coord / d.sizeY
  const idx = Math.round(frac * n - 0.5)
  return Math.min(n - 1, Math.max(0, idx))
}

/**
 * Normal of the surface the pointer is projected onto while dragging a slice: it contains
 * the slice's axis and faces the camera as much as possible, so screen motion maps smoothly
 * to motion along the axis at any view angle. Returns null when the camera looks (almost)
 * straight along the axis, where dragging along it is ill-defined.
 */
export function dragSurfaceNormal(
  axis: SliceAxis,
  cameraDir: [number, number, number],
): [number, number, number] | null {
  const a = SLICE_AXIS_DIR[axis]
  const along = cameraDir[0] * a[0] + cameraDir[1] * a[1] + cameraDir[2] * a[2]
  const v: [number, number, number] = [
    cameraDir[0] - along * a[0],
    cameraDir[1] - along * a[1],
    cameraDir[2] - along * a[2],
  ]
  const len = Math.hypot(v[0], v[1], v[2])
  const camLen = Math.hypot(cameraDir[0], cameraDir[1], cameraDir[2])
  if (camLen === 0 || len / camLen < 0.05) return null
  return [v[0] / len, v[1] / len, v[2] / len]
}
