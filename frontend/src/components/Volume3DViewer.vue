<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Box, Clock, Eye, Layers, RotateCcw } from 'lucide-vue-next'
import type { FeaturePlaneAxis, SectionAxis, SectionResponseData } from '../api/types'
import { useWorkbenchStore } from '../stores/workbench'
import { getOverlayLUT, getSeismicLUT, sampleLUT } from '../utils/colormaps'
import {
  dragSurfaceNormal,
  sliceIndexToWorld,
  worldToSliceIndex,
  type SliceAxis,
} from '../utils/sliceDrag'
import {
  EDGE_HIT_RADIUS_PX,
  HANDLE_HIT_RADIUS_PX,
  ROLE_COLORS,
  findPatchIdx,
  hitTestEdges,
  hitTestHandles,
  type EdgeCandidate,
  type HandleCandidate,
} from '../utils/featureGeometry'

const store = useWorkbenchStore()
const containerRef = ref<HTMLDivElement | null>(null)

const showSlices = ref(true)
// The time slice is loaded with the other sections but hidden until the user turns it on;
// it starts off on every page load and survey change. "Slices" stays the master switch.
const showTimeSlice = ref(false)
const showCloud = ref(true)
const showGatherCurtain = ref(true)
const showBox = ref(true)
const hoveredPatchScore = ref<{ il: number; xl: number; sample: number; val: number } | null>(null)

const survey = computed(() => store.activeSurvey)
const cloudVisibleCount = ref(0)

// Three.js core objects
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let renderer: THREE.WebGLRenderer | null = null
let controls: OrbitControls | null = null
let animFrameId: number | null = null
let resizeObs: ResizeObserver | null = null

// Scene groups & meshes
let boxHelper: THREE.LineSegments | null = null
let inlineMesh: THREE.Mesh | null = null
let xlineMesh: THREE.Mesh | null = null
let sampleMesh: THREE.Mesh | null = null
let inlineTex: THREE.CanvasTexture | null = null
let xlineTex: THREE.CanvasTexture | null = null
let sampleTex: THREE.CanvasTexture | null = null

const inlineCanvas = document.createElement('canvas')
const xlineCanvas = document.createElement('canvas')
const sampleCanvas = document.createElement('canvas')

let gatherGroup: THREE.Group | null = null
let pickGroup: THREE.Group | null = null
let reservoirGroup: THREE.Group | null = null
let pickGroupTextures: THREE.Texture[] = []
let cloudMesh: THREE.InstancedMesh | null = null
let cloudMaxCapacity = 0
let cloudGeometryKey = ''
let visiblePatchCoords = new Int32Array(0) // [il_idx, xl_idx, sample_idx] per visible instance
let visiblePatchValues = new Float32Array(0)

const raycaster = new THREE.Raycaster()
const occlusionRaycaster = new THREE.Raycaster()
const pointerNdc = new THREE.Vector2()
let pointerDownX = 0
let pointerDownY = 0

// Shift+drag on a slice plane moves that slice along its axis. The plane follows the
// pointer live; the shared cursor (and hence section fetches) is updated on release.
interface SliceDragState {
  axis: SliceAxis
  mesh: THREE.Mesh
  surface: THREE.Plane
  pointerId: number
  index: number
  startIndex: number
}
let sliceDrag: SliceDragState | null = null
const sliceDragReadout = ref<string | null>(null)
const dragHit = new THREE.Vector3()

let draggingHandle3D: {
  featureId: string
  pickIndex: number
  planeAxis: FeaturePlaneAxis
  plane: THREE.Plane
} | null = null
let middleDeleting3D = false
let lastPrimaryClickTime3D = 0
let lastClickX3D = 0
let lastClickY3D = 0
let justFinishedUntil3D = 0
const cursor3DStyle = ref<'default' | 'grab' | 'grabbing' | 'copy'>('default')

function getWorldDims() {
  const s = survey.value
  if (!s) return { sizeX: 2.0, sizeY: 1.6, sizeZ: 2.0, nIl: 1, nXl: 1, nSamples: 1 }
  const [nIl, nXl, nSamples] = s.shape
  const maxH = Math.max(1, nIl, nXl)
  const sizeX = (nXl / maxH) * 2.0
  const sizeZ = (nIl / maxH) * 2.0
  const sizeY = 1.6
  return { sizeX, sizeY, sizeZ, nIl, nXl, nSamples }
}

function indexToWorld(ilIdx: number, xlIdx: number, sampleIdx: number): THREE.Vector3 {
  const { sizeX, sizeY, sizeZ, nIl, nXl, nSamples } = getWorldDims()
  const wx = ((xlIdx + 0.5) / Math.max(1, nXl) - 0.5) * sizeX
  const wy = (0.5 - (sampleIdx + 0.5) / Math.max(1, nSamples)) * sizeY
  const wz = ((ilIdx + 0.5) / Math.max(1, nIl) - 0.5) * sizeZ
  return new THREE.Vector3(wx, wy, wz)
}

function worldToIndex(pt: THREE.Vector3): { ilIdx: number; xlIdx: number; sampleIdx: number } {
  const { sizeX, sizeY, sizeZ, nIl, nXl, nSamples } = getWorldDims()
  const xlIdx = Math.max(
    0,
    Math.min(nXl - 1, Math.floor((pt.x / sizeX + 0.5) * nXl)),
  )
  const sampleIdx = Math.max(
    0,
    Math.min(nSamples - 1, Math.floor((0.5 - pt.y / sizeY) * nSamples)),
  )
  const ilIdx = Math.max(
    0,
    Math.min(nIl - 1, Math.floor((pt.z / sizeZ + 0.5) * nIl)),
  )
  return { ilIdx, xlIdx, sampleIdx }
}

function getLockedThreePlane(planeAxis: FeaturePlaneAxis, planeIndex: number): THREE.Plane {
  if (planeAxis === 'inline') {
    const wp = indexToWorld(planeIndex, 0, 0)
    return new THREE.Plane(new THREE.Vector3(0, 0, 1), -wp.z)
  }
  const wp = indexToWorld(0, planeIndex, 0)
  return new THREE.Plane(new THREE.Vector3(1, 0, 0), -wp.x)
}

function initThree() {
  const container = containerRef.value
  if (!container) return

  scene = new THREE.Scene()
  scene.background = new THREE.Color('#020617')

  const rect = container.getBoundingClientRect()
  const w = Math.max(1, rect.width)
  const h = Math.max(1, rect.height)

  camera = new THREE.PerspectiveCamera(42, w / h, 0.05, 100)
  camera.position.set(2.45, 1.45, 2.55)

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  renderer.setSize(w, h)
  container.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.minDistance = 0.7
  controls.maxDistance = 12
  controls.target.set(0, 0, 0)
  controls.update()

  // Lighting for 3D similarity voxels
  const ambient = new THREE.AmbientLight(0xffffff, 0.75)
  scene.add(ambient)
  const dir1 = new THREE.DirectionalLight(0xffffff, 0.65)
  dir1.position.set(3, 5, 4)
  scene.add(dir1)
  const dir2 = new THREE.DirectionalLight(0x94a3b8, 0.35)
  dir2.position.set(-3, -2, -4)
  scene.add(dir2)

  gatherGroup = new THREE.Group()
  pickGroup = new THREE.Group()
  reservoirGroup = new THREE.Group()
  scene.add(gatherGroup)
  scene.add(pickGroup)
  scene.add(reservoirGroup)

  rebuildSurveyGeometry()
  updateSliceTextures()
  updateSlicePositions()
  updateGatherCurtain()
  updateFeaturePickMarker()
  updateCloud()
  updateReservoir()

  const animate = () => {
    animFrameId = requestAnimationFrame(animate)
    controls?.update()
    if (renderer && scene && camera) {
      renderer.render(scene, camera)
    }
  }
  animate()

  // Capture phase on the container runs before OrbitControls' listeners on the canvas,
  // so a shift-drag that grabs a slice never starts an orbit/pan.
  container.addEventListener('pointerdown', onSliceDragStart, { capture: true })
  container.addEventListener('pointermove', onSliceDragMove, { capture: true })
  container.addEventListener('pointerup', onSliceDragEnd, { capture: true })
  container.addEventListener('pointercancel', onSliceDragEnd, { capture: true })

  renderer.domElement.addEventListener('mousedown', onMouseDown)
  renderer.domElement.addEventListener('auxclick', onAuxClick)
  renderer.domElement.addEventListener('pointerdown', onPointerDown, { capture: true })
  renderer.domElement.addEventListener('pointermove', onPointerMove)
  renderer.domElement.addEventListener('pointerup', onPointerUp)
  renderer.domElement.addEventListener('pointercancel', onPointerCancel)
  renderer.domElement.addEventListener('pointerleave', onPointerLeave)
  window.addEventListener('blur', onWindowBlur)
}

function disposeObject(obj: THREE.Object3D | null) {
  if (!obj) return
  obj.traverse((child) => {
    const mesh = child as THREE.Mesh
    if (mesh.geometry) mesh.geometry.dispose()
    if (mesh.material) {
      if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose())
      else mesh.material.dispose()
    }
  })
}

function rebuildSurveyGeometry() {
  if (!scene) return

  if (boxHelper) {
    scene.remove(boxHelper)
    boxHelper.geometry.dispose()
    ;(boxHelper.material as THREE.Material).dispose()
    boxHelper = null
  }
  if (inlineMesh) {
    scene.remove(inlineMesh)
    disposeObject(inlineMesh)
    inlineMesh = null
  }
  if (xlineMesh) {
    scene.remove(xlineMesh)
    disposeObject(xlineMesh)
    xlineMesh = null
  }
  if (sampleMesh) {
    scene.remove(sampleMesh)
    disposeObject(sampleMesh)
    sampleMesh = null
  }
  inlineTex?.dispose()
  xlineTex?.dispose()
  sampleTex?.dispose()
  inlineTex = null
  xlineTex = null
  sampleTex = null

  if (!survey.value) return
  const { sizeX, sizeY, sizeZ } = getWorldDims()

  // Bounding wireframe box
  const boxGeo = new THREE.BoxGeometry(sizeX, sizeY, sizeZ)
  const edgesGeo = new THREE.EdgesGeometry(boxGeo)
  boxGeo.dispose()
  const boxMat = new THREE.LineBasicMaterial({
    color: 0x334155,
    transparent: true,
    opacity: 0.9,
  })
  boxHelper = new THREE.LineSegments(edgesGeo, boxMat)
  boxHelper.visible = showBox.value
  scene.add(boxHelper)

  // Inline plane (constant Z = ilIdx, spans X=crossline, Y=sample)
  inlineTex = new THREE.CanvasTexture(inlineCanvas)
  inlineTex.colorSpace = THREE.SRGBColorSpace
  inlineTex.minFilter = THREE.LinearFilter
  inlineTex.magFilter = THREE.LinearFilter
  const inlineMat = new THREE.MeshBasicMaterial({
    map: inlineTex,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  })
  inlineMesh = new THREE.Mesh(new THREE.PlaneGeometry(sizeX, sizeY), inlineMat)
  inlineMesh.visible = showSlices.value
  scene.add(inlineMesh)

  // Crossline plane (constant X = xlIdx, spans Z=inline, Y=sample)
  xlineTex = new THREE.CanvasTexture(xlineCanvas)
  xlineTex.colorSpace = THREE.SRGBColorSpace
  xlineTex.minFilter = THREE.LinearFilter
  xlineTex.magFilter = THREE.LinearFilter
  const xlineMat = new THREE.MeshBasicMaterial({
    map: xlineTex,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  })
  xlineMesh = new THREE.Mesh(new THREE.PlaneGeometry(sizeZ, sizeY), xlineMat)
  xlineMesh.rotation.y = -Math.PI / 2
  xlineMesh.visible = showSlices.value
  scene.add(xlineMesh)

  // Time Slice plane (constant Y = sampleIdx, spans X=crossline, Z=inline)
  sampleTex = new THREE.CanvasTexture(sampleCanvas)
  sampleTex.colorSpace = THREE.SRGBColorSpace
  sampleTex.minFilter = THREE.LinearFilter
  sampleTex.magFilter = THREE.LinearFilter
  const sampleMat = new THREE.MeshBasicMaterial({
    map: sampleTex,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  })
  sampleMesh = new THREE.Mesh(new THREE.PlaneGeometry(sizeX, sizeZ), sampleMat)
  sampleMesh.rotation.x = -Math.PI / 2
  sampleMesh.visible = showSlices.value && showTimeSlice.value
  scene.add(sampleMesh)
}

function paintSectionCanvas(
  canvas: HTMLCanvasElement,
  sec: SectionResponseData | null,
  axis: SectionAxis,
) {
  if (!sec || !sec.data.length) return false
  let w: number
  let h: number
  if (axis === 'sample') {
    h = sec.shape[0]
    w = sec.shape[1]
  } else {
    w = sec.shape[0]
    h = sec.shape[1]
  }
  if (w <= 0 || h <= 0) return false

  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return false

  const img = ctx.createImageData(w, h)
  const px = img.data
  const data = sec.data
  const lut = getSeismicLUT(store.seismicColormap)
  const clip = Math.max(1e-6, store.effectiveClipAmplitude)
  const invTwoClip = 1 / (2 * clip)

  if (axis === 'sample') {
    const total = w * h
    for (let i = 0; i < total; i++) {
      let t = (data[i] + clip) * invTwoClip
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
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w
      for (let x = 0; x < w; x++) {
        let t = (data[x * h + y] + clip) * invTwoClip
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
  return true
}

// three.js fixes a texture's GPU storage size at its first upload; if the canvas is
// resized afterwards (e.g. the first frame uploaded the blank 300x150 default before the
// section fetch finished), later uploads fail silently and the plane stays black.
// Disposing frees the old storage so the next render reallocates it at the new size.
function repaintSlice(
  canvas: HTMLCanvasElement,
  tex: THREE.CanvasTexture | null,
  sec: SectionResponseData | null,
  axis: SectionAxis,
) {
  const prevW = canvas.width
  const prevH = canvas.height
  if (!paintSectionCanvas(canvas, sec, axis) || !tex) return
  if (canvas.width !== prevW || canvas.height !== prevH) tex.dispose()
  tex.needsUpdate = true
}

function updateSliceTextures() {
  repaintSlice(inlineCanvas, inlineTex, store.sections.inline, 'inline')
  repaintSlice(xlineCanvas, xlineTex, store.sections.xline, 'xline')
  repaintSlice(sampleCanvas, sampleTex, store.sections.sample, 'sample')
}

function updateSlicePositions() {
  const pos = indexToWorld(store.ilIdx, store.xlIdx, store.sampleIdx)
  if (inlineMesh) inlineMesh.position.z = pos.z
  if (xlineMesh) xlineMesh.position.x = pos.x
  if (sampleMesh) sampleMesh.position.y = pos.y
}

function updateGatherCurtain() {
  if (!gatherGroup) return
  while (gatherGroup.children.length > 0) {
    const child = gatherGroup.children[0]
    gatherGroup.remove(child)
    disposeObject(child)
  }

  gatherGroup.visible = showGatherCurtain.value
  const gp = store.gatherPanelData
  if (!gp || !gp.locations.length || !survey.value) return

  const { sizeY } = getWorldDims()
  const topY = sizeY * 0.5 + 0.02
  const botY = -sizeY * 0.5

  const railPoints: THREE.Vector3[] = []
  const markerGeo = new THREE.SphereGeometry(0.018, 10, 10)
  const markerMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 })
  const lineMat = new THREE.LineBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.65,
  })

  for (const loc of gp.locations) {
    const p = indexToWorld(loc.il_idx, loc.xl_idx, 0)
    const topPt = new THREE.Vector3(p.x, topY, p.z)
    const botPt = new THREE.Vector3(p.x, botY, p.z)
    railPoints.push(topPt)

    const vGeo = new THREE.BufferGeometry().setFromPoints([topPt, botPt])
    gatherGroup.add(new THREE.Line(vGeo, lineMat))

    const dot = new THREE.Mesh(markerGeo, markerMat)
    dot.position.copy(topPt)
    gatherGroup.add(dot)
  }

  if (railPoints.length > 1) {
    const railGeo = new THREE.BufferGeometry().setFromPoints(railPoints)
    const railMat = new THREE.LineBasicMaterial({ color: 0xf59e0b })
    gatherGroup.add(new THREE.Line(railGeo, railMat))
  }
}

function addPatchWireframeBox(
  group: THREE.Group,
  ilIdx: number,
  xlIdx: number,
  sampleIdx: number,
  colorHex: number,
) {
  const { sizeX, sizeY, sizeZ, nIl, nXl, nSamples } = getWorldDims()
  const emb = store.activeEmbeddingSet
  const ov = store.activeOverlay
  const patchSize = ov?.patchSize ?? emb?.patch_size ?? 16
  const ilStarts = ov?.ilPatchStart ?? emb?.il_patch_start ?? []
  const xlStarts = ov?.xlPatchStart ?? emb?.xl_patch_start ?? []
  const tStarts = ov?.tPatchStart ?? emb?.t_patch_start ?? []

  let boxPos: THREE.Vector3
  if (ilStarts.length > 0 && xlStarts.length > 0 && tStarts.length > 0) {
    const pi = findPatchIdx(ilStarts, patchSize, ilIdx)
    const pj = findPatchIdx(xlStarts, patchSize, xlIdx)
    const pk = findPatchIdx(tStarts, patchSize, sampleIdx)
    const half = patchSize * 0.5
    boxPos = indexToWorld(
      Math.min(nIl - 1, (ilStarts[pi] ?? pi * patchSize) + half - 0.5),
      Math.min(nXl - 1, (xlStarts[pj] ?? pj * patchSize) + half - 0.5),
      Math.min(nSamples - 1, (tStarts[pk] ?? pk * patchSize) + half - 0.5),
    )
  } else {
    boxPos = indexToWorld(ilIdx, xlIdx, sampleIdx)
  }

  const pw = (patchSize / Math.max(1, nXl)) * sizeX
  const ph = (patchSize / Math.max(1, nSamples)) * sizeY
  const pd = (patchSize / Math.max(1, nIl)) * sizeZ
  const boxGeo = new THREE.BoxGeometry(pw, ph, pd)
  const edges = new THREE.EdgesGeometry(boxGeo)
  boxGeo.dispose()
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.9,
    }),
  )
  line.position.copy(boxPos)
  group.add(line)
}

function disposePickGroupTextures() {
  for (const tex of pickGroupTextures) {
    tex.dispose()
  }
  pickGroupTextures = []
}

function updateFeaturePickMarker() {
  if (!pickGroup) return
  disposePickGroupTextures()
  while (pickGroup.children.length > 0) {
    const child = pickGroup.children[0]
    pickGroup.remove(child)
    disposeObject(child)
  }

  if (!survey.value) return

  const hovered = store.hoveredPickRef

  for (const feat of store.pickedFeatures) {
    if (!feat.enabled || feat.picks.length === 0) continue

    const colors = ROLE_COLORS[feat.role]
    const onActiveSlice =
      (feat.plane_axis === 'inline' && feat.plane_index === store.ilIdx) ||
      (feat.plane_axis === 'xline' && feat.plane_index === store.xlIdx)
    const lineOpacity = onActiveSlice ? 1.0 : 0.82
    const fillOpacity = onActiveSlice ? 0.22 : 0.14

    const worldPts = feat.picks.map((p) => indexToWorld(p.il_idx, p.xl_idx, p.sample_idx))

    // 3D sphere handles
    for (let i = 0; i < worldPts.length; i++) {
      const isHovered =
        hovered?.featureId === feat.id && hovered?.pickIndex === i
      const radius = isHovered ? 0.036 : 0.024
      const sphereGeo = new THREE.SphereGeometry(radius, 14, 14)
      const sphereMat = new THREE.MeshBasicMaterial({
        color: isHovered ? 0xffffff : colors.primaryHex,
      })
      const sphere = new THREE.Mesh(sphereGeo, sphereMat)
      sphere.position.copy(worldPts[i])
      pickGroup.add(sphere)

      if (isHovered || feat.geometry_mode === 'single') {
        const haloGeo = new THREE.SphereGeometry(isHovered ? 0.054 : 0.038, 14, 14)
        const haloMat = new THREE.MeshBasicMaterial({
          color: colors.haloHex,
          transparent: true,
          opacity: isHovered ? 0.45 : 0.28,
          depthWrite: false,
        })
        const halo = new THREE.Mesh(haloGeo, haloMat)
        halo.position.copy(worldPts[i])
        pickGroup.add(halo)
      }

      // 16x16x16 Patch Wireframe Box only for hovered pick or enabled 'single' features
      if (isHovered || feat.geometry_mode === 'single') {
        const p = feat.picks[i]
        addPatchWireframeBox(pickGroup, p.il_idx, p.xl_idx, p.sample_idx, colors.primaryHex)
      }
    }

    // Polyline or Polygon edges
    if (
      (feat.geometry_mode === 'polyline' || feat.geometry_mode === 'polygon') &&
      worldPts.length >= 2
    ) {
      const linePts = [...worldPts]
      if (feat.geometry_mode === 'polygon' && feat.status === 'completed' && worldPts.length >= 3) {
        linePts.push(worldPts[0])
      }
      const lineGeo = new THREE.BufferGeometry().setFromPoints(linePts)
      const lineMat = new THREE.LineBasicMaterial({
        color: colors.primaryHex,
        transparent: true,
        opacity: lineOpacity,
      })
      const line = new THREE.Line(lineGeo, lineMat)
      line.renderOrder = 3
      pickGroup.add(line)
    }

    // Planar Polygon interior mesh (even-odd parity via 2D canvas texture)
    if (feat.geometry_mode === 'polygon' && worldPts.length >= 3) {
      let minU = Infinity
      let maxU = -Infinity
      let minV = Infinity
      let maxV = -Infinity
      for (const wp of worldPts) {
        const u = feat.plane_axis === 'inline' ? wp.x : wp.z
        const v = wp.y
        if (u < minU) minU = u
        if (u > maxU) maxU = u
        if (v < minV) minV = v
        if (v > maxV) maxV = v
      }
      const spanU = Math.max(1e-4, maxU - minU)
      const spanV = Math.max(1e-4, maxV - minV)
      const centerU = (minU + maxU) * 0.5
      const centerV = (minV + maxV) * 0.5

      const TEX_SIZE = 256
      const polyCanvas = document.createElement('canvas')
      polyCanvas.width = TEX_SIZE
      polyCanvas.height = TEX_SIZE
      const pctx = polyCanvas.getContext('2d')
      if (pctx) {
        pctx.clearRect(0, 0, TEX_SIZE, TEX_SIZE)
        pctx.beginPath()
        for (let i = 0; i < worldPts.length; i++) {
          const wp = worldPts[i]
          const u = feat.plane_axis === 'inline' ? wp.x : wp.z
          const v = wp.y
          const px = ((u - minU) / spanU) * TEX_SIZE
          const py = ((maxV - v) / spanV) * TEX_SIZE
          if (i === 0) pctx.moveTo(px, py)
          else pctx.lineTo(px, py)
        }
        pctx.closePath()
        pctx.fillStyle = colors.primary
        pctx.fill('evenodd')
      }

      const polyTex = new THREE.CanvasTexture(polyCanvas)
      polyTex.colorSpace = THREE.SRGBColorSpace
      polyTex.minFilter = THREE.LinearFilter
      polyTex.magFilter = THREE.LinearFilter
      pickGroupTextures.push(polyTex)

      const polyGeo = new THREE.PlaneGeometry(spanU, spanV)
      const polyMat = new THREE.MeshBasicMaterial({
        map: polyTex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: fillOpacity,
        depthWrite: false,
      })
      const mesh = new THREE.Mesh(polyGeo, polyMat)
      mesh.renderOrder = 2
      if (feat.plane_axis === 'inline') {
        mesh.position.set(centerU, centerV, worldPts[0].z)
      } else {
        mesh.rotation.y = -Math.PI / 2
        mesh.position.set(worldPts[0].x, centerV, centerU)
      }
      pickGroup.add(mesh)
    }
  }
}

function disposeCloudMesh() {
  if (!cloudMesh) return
  scene?.remove(cloudMesh)
  cloudMesh.geometry.dispose()
  if (Array.isArray(cloudMesh.material)) {
    cloudMesh.material.forEach((m) => m.dispose())
  } else {
    cloudMesh.material.dispose()
  }
  cloudMesh.dispose()
  cloudMesh = null
  cloudMaxCapacity = 0
  cloudGeometryKey = ''
}

/** Triangulated surface through the finite bins of a (n_il, n_xl) TWT grid. */
function buildSurfaceMesh(
  twt: Float32Array,
  color: string,
  opacity: number,
  wireframe = false,
): THREE.Mesh | null {
  const { sizeX, sizeY, sizeZ, nIl, nXl, nSamples } = getWorldDims()
  const vertexOf = new Int32Array(nIl * nXl).fill(-1)
  const positions: number[] = []
  for (let i = 0; i < nIl; i++) {
    for (let j = 0; j < nXl; j++) {
      const v = twt[i * nXl + j]
      if (!Number.isFinite(v)) continue
      vertexOf[i * nXl + j] = positions.length / 3
      const s = store.twtToSample(v)
      positions.push(
        ((j + 0.5) / nXl - 0.5) * sizeX,
        (0.5 - (s + 0.5) / nSamples) * sizeY,
        ((i + 0.5) / nIl - 0.5) * sizeZ,
      )
    }
  }
  if (positions.length === 0) return null
  const indices: number[] = []
  for (let i = 0; i < nIl - 1; i++) {
    for (let j = 0; j < nXl - 1; j++) {
      const a = vertexOf[i * nXl + j]
      const b = vertexOf[i * nXl + j + 1]
      const c = vertexOf[(i + 1) * nXl + j]
      const d = vertexOf[(i + 1) * nXl + j + 1]
      if (a < 0 || b < 0 || c < 0 || d < 0) continue
      indices.push(a, b, d, a, d, c)
    }
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geo.setIndex(indices)
  geo.computeVertexNormals()
  const mat = new THREE.MeshLambertMaterial({
    color: new THREE.Color(color),
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false,
    wireframe,
  })
  return new THREE.Mesh(geo, mat)
}

function updateReservoir() {
  if (!reservoirGroup) return
  while (reservoirGroup.children.length > 0) {
    const child = reservoirGroup.children[0]
    reservoirGroup.remove(child)
    disposeObject(child)
  }
  if (!survey.value || !store.showReservoir) return

  const env = store.reservoirEnvelope
  if (env) {
    const top = buildSurfaceMesh(env.top, env.color, Math.min(0.95, store.reservoirOpacity + 0.35))
    const base = buildSurfaceMesh(env.base, env.color, Math.min(0.8, store.reservoirOpacity + 0.1))
    if (top) reservoirGroup.add(top)
    if (base) reservoirGroup.add(base)
  }
  for (const hz of store.interpretedHorizons) {
    const mesh = buildSurfaceMesh(hz.twt, hz.color, 0.55, true)
    if (mesh) reservoirGroup.add(mesh)
  }

  const picks = store.visibleWellPicks
  if (picks.length) {
    const geo = new THREE.SphereGeometry(0.012, 8, 8)
    const mat = new THREE.MeshBasicMaterial({ color: 0xfde047 })
    for (const p of picks) {
      const dot = new THREE.Mesh(geo, mat)
      dot.position.copy(indexToWorld(p.il_idx, p.xl_idx, store.twtToSample(p.twt)))
      reservoirGroup.add(dot)
    }
  }
}

function ensureCloudMesh(capacity: number) {
  if (!scene || !survey.value) return
  const { sizeX, sizeY, sizeZ, nIl, nXl, nSamples } = getWorldDims()
  const patchSize = store.activeOverlay?.patchSize ?? 16
  const nextKey = `${sizeX}:${sizeY}:${sizeZ}:${nIl}:${nXl}:${nSamples}:${patchSize}`
  if (cloudMesh && cloudMaxCapacity >= capacity && cloudGeometryKey === nextKey) return

  disposeCloudMesh()

  const vw = Math.max(0.015, (patchSize / Math.max(1, nXl)) * sizeX * 0.74)
  const vh = Math.max(0.012, (patchSize / Math.max(1, nSamples)) * sizeY * 0.74)
  const vd = Math.max(0.015, (patchSize / Math.max(1, nIl)) * sizeZ * 0.74)

  const geo = new THREE.BoxGeometry(vw, vh, vd)
  const mat = new THREE.MeshLambertMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.88,
  })

  cloudMesh = new THREE.InstancedMesh(geo, mat, Math.max(1, capacity))
  cloudMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  cloudMesh.frustumCulled = false
  cloudMaxCapacity = capacity
  cloudGeometryKey = nextKey
  visiblePatchCoords = new Int32Array(capacity * 3)
  visiblePatchValues = new Float32Array(capacity)
  scene.add(cloudMesh)
}

function updateCloud() {
  const ov = store.activeOverlay
  const s = survey.value
  if (!ov || !s || !showCloud.value) {
    if (cloudMesh) cloudMesh.visible = false
    cloudVisibleCount.value = 0
    return
  }

  const [nIlP, nXlP, nTP] = ov.gridShape
  const total = nIlP * nXlP * nTP
  if (total === 0 || ov.data.length === 0) {
    if (cloudMesh) cloudMesh.visible = false
    cloudVisibleCount.value = 0
    return
  }

  ensureCloudMesh(total)
  if (!cloudMesh) return

  const lut = getOverlayLUT(store.overlayColormap)
  const thresh = ov.threshold
  const normMin = Math.min(thresh, ov.max - 1e-4)
  const normSpan = Math.max(1e-5, ov.max - normMin)
  const half = ov.patchSize * 0.5
  const [nIl, nXl, nSamples] = s.shape

  const dummy = new THREE.Object3D()
  const color = new THREE.Color()
  let count = 0
  const strideIl = nXlP * nTP
  const d = ov.data

  for (let pi = 0; pi < nIlP; pi++) {
    const ilStart = ov.ilPatchStart[pi] ?? pi * ov.patchSize
    const ilCoord = Math.min(nIl - 1, ilStart + half - 0.5)
    const ilCenter = Math.min(nIl - 1, Math.max(0, Math.round(ilCoord)))
    for (let pj = 0; pj < nXlP; pj++) {
      const xlStart = ov.xlPatchStart[pj] ?? pj * ov.patchSize
      const xlCoord = Math.min(nXl - 1, xlStart + half - 0.5)
      const xlCenter = Math.min(nXl - 1, Math.max(0, Math.round(xlCoord)))
      for (let pk = 0; pk < nTP; pk++) {
        const val = d[pi * strideIl + pj * nTP + pk]
        if (val < thresh || Number.isNaN(val)) continue

        const tStart = ov.tPatchStart[pk] ?? pk * ov.patchSize
        const tCoord = Math.min(nSamples - 1, tStart + half - 0.5)
        const tCenter = Math.min(nSamples - 1, Math.max(0, Math.round(tCoord)))
        const wp = indexToWorld(ilCoord, xlCoord, tCoord)
        dummy.position.copy(wp)

        let norm = (val - normMin) / normSpan
        if (norm < 0) norm = 0
        else if (norm > 1) norm = 1
        const scale = 0.65 + 0.5 * norm
        dummy.scale.set(scale, scale, scale)
        dummy.updateMatrix()

        cloudMesh.setMatrixAt(count, dummy.matrix)
        const [r, g, b] = sampleLUT(lut, norm)
        color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace)
        cloudMesh.setColorAt(count, color)

        visiblePatchCoords[count * 3] = ilCenter
        visiblePatchCoords[count * 3 + 1] = xlCenter
        visiblePatchCoords[count * 3 + 2] = tCenter
        visiblePatchValues[count] = val
        count++
      }
    }
  }

  cloudMesh.count = count
  cloudMesh.visible = count > 0
  cloudMesh.instanceMatrix.needsUpdate = true
  if (cloudMesh.instanceColor) cloudMesh.instanceColor.needsUpdate = true
  cloudVisibleCount.value = count
}

function getHitCloudInstance(clientX: number, clientY: number): number | null {
  if (!renderer || !camera || !cloudMesh || !cloudMesh.visible || cloudMesh.count === 0) {
    return null
  }
  const rect = renderer.domElement.getBoundingClientRect()
  pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1
  pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1
  raycaster.setFromCamera(pointerNdc, camera)
  const hits = raycaster.intersectObject(cloudMesh, false)
  if (hits.length > 0 && hits[0].instanceId !== undefined) {
    return hits[0].instanceId
  }
  return null
}

function setPointerNdc(clientX: number, clientY: number) {
  if (!renderer) return
  const rect = renderer.domElement.getBoundingClientRect()
  pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1
  pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1
}

function currentSliceIndex(axis: SliceAxis): number {
  if (axis === 'inline') return store.ilIdx
  if (axis === 'xline') return store.xlIdx
  return store.sampleIdx
}

function sliceLabel(axis: SliceAxis, idx: number): string {
  const s = survey.value
  if (axis === 'inline') return `IL ${s?.ilines[idx] ?? idx}`
  if (axis === 'xline') return `XL ${s?.xlines[idx] ?? idx}`
  return `T ${s?.samples[idx] ?? idx} ms`
}

function onSliceDragStart(e: PointerEvent) {
  if (!e.shiftKey || e.button !== 0 || !showSlices.value) return
  if (!renderer || !camera || !controls || !survey.value || e.target !== renderer.domElement) return
  const candidates: [THREE.Mesh, SliceAxis][] = []
  if (inlineMesh?.visible) candidates.push([inlineMesh, 'inline'])
  if (xlineMesh?.visible) candidates.push([xlineMesh, 'xline'])
  if (sampleMesh?.visible) candidates.push([sampleMesh, 'sample'])
  if (!candidates.length) return

  setPointerNdc(e.clientX, e.clientY)
  raycaster.setFromCamera(pointerNdc, camera)
  const hit = raycaster.intersectObjects(
    candidates.map(([m]) => m),
    false,
  )[0]
  if (!hit) return
  const axis = candidates.find(([m]) => m === hit.object)![1]
  const camDir = camera.getWorldDirection(new THREE.Vector3())
  const n = dragSurfaceNormal(axis, [camDir.x, camDir.y, camDir.z])
  if (!n) return // looking straight down this slice's axis: leave the gesture to OrbitControls

  e.preventDefault()
  e.stopPropagation()
  controls.enabled = false
  renderer.domElement.setPointerCapture(e.pointerId)
  cursor3DStyle.value = 'grabbing'
  const index = currentSliceIndex(axis)
  sliceDrag = {
    axis,
    mesh: hit.object as THREE.Mesh,
    surface: new THREE.Plane().setFromNormalAndCoplanarPoint(new THREE.Vector3(...n), hit.point),
    pointerId: e.pointerId,
    index,
    startIndex: index,
  }
  sliceDragReadout.value = sliceLabel(axis, index)
}

function onSliceDragMove(e: PointerEvent) {
  if (!sliceDrag || e.pointerId !== sliceDrag.pointerId || !camera) return
  e.stopPropagation()
  setPointerNdc(e.clientX, e.clientY)
  raycaster.setFromCamera(pointerNdc, camera)
  if (!raycaster.ray.intersectPlane(sliceDrag.surface, dragHit)) return
  const { axis, mesh } = sliceDrag
  const coord = axis === 'inline' ? dragHit.z : axis === 'xline' ? dragHit.x : dragHit.y
  const dims = getWorldDims()
  const index = worldToSliceIndex(axis, coord, dims)
  if (index === sliceDrag.index) return
  sliceDrag.index = index
  const w = sliceIndexToWorld(axis, index, dims)
  if (axis === 'inline') mesh.position.z = w
  else if (axis === 'xline') mesh.position.x = w
  else mesh.position.y = w
  sliceDragReadout.value = sliceLabel(axis, index)
}

function onSliceDragEnd(e: PointerEvent) {
  if (!sliceDrag || e.pointerId !== sliceDrag.pointerId) return
  e.stopPropagation()
  const { axis, index, startIndex } = sliceDrag
  sliceDrag = null
  sliceDragReadout.value = null
  if (controls) controls.enabled = true
  if (renderer?.domElement.hasPointerCapture(e.pointerId)) {
    renderer.domElement.releasePointerCapture(e.pointerId)
  }
  cursor3DStyle.value = 'default'
  if (e.type === 'pointercancel' || index === startIndex) {
    updateSlicePositions()
    return
  }
  if (axis === 'inline') store.setCursor({ ilIdx: index })
  else if (axis === 'xline') store.setCursor({ xlIdx: index })
  else store.setCursor({ sampleIdx: index })
}

function getVisibleSliceMeshes(): THREE.Mesh[] {
  const meshes: THREE.Mesh[] = []
  if (!showSlices.value) return meshes
  if (inlineMesh && inlineMesh.visible) meshes.push(inlineMesh)
  if (xlineMesh && xlineMesh.visible) meshes.push(xlineMesh)
  if (sampleMesh && sampleMesh.visible) meshes.push(sampleMesh)
  return meshes
}

function isWorldPointOccludedBySlices(worldPos: THREE.Vector3): boolean {
  if (!camera || !showSlices.value) return false
  const sliceMeshes = getVisibleSliceMeshes()
  if (sliceMeshes.length === 0) return false
  const dir = worldPos.clone().sub(camera.position)
  const dist = dir.length()
  if (dist <= 1e-6) return false
  dir.divideScalar(dist)
  occlusionRaycaster.set(camera.position, dir)
  const hits = occlusionRaycaster.intersectObjects(sliceMeshes, false)
  return hits.length > 0 && hits[0].distance < dist - 0.015
}

function getProjected3DCandidates(rect: DOMRect) {
  const handles: HandleCandidate[] = []
  const edges: EdgeCandidate[] = []
  if (!camera || !survey.value) return { handles, edges }

  for (const feat of store.pickedFeatures) {
    if (!feat.enabled || feat.picks.length === 0) continue

    const screenPts: Array<{ x: number; y: number } | null> = []
    for (let i = 0; i < feat.picks.length; i++) {
      const p = feat.picks[i]
      const wp = indexToWorld(p.il_idx, p.xl_idx, p.sample_idx)
      const ndc = wp.clone().project(camera)
      if (ndc.z < -1 || ndc.z > 1) {
        screenPts.push(null)
        continue
      }
      if (isWorldPointOccludedBySlices(wp)) {
        screenPts.push(null)
        continue
      }
      const sx = ((ndc.x + 1) * 0.5) * rect.width
      const sy = ((1 - ndc.y) * 0.5) * rect.height
      screenPts.push({ x: sx, y: sy })
      handles.push({
        featureId: feat.id,
        pickIndex: i,
        x: sx,
        y: sy,
      })
    }

    if (
      feat.status === 'completed' &&
      (feat.geometry_mode === 'polyline' || feat.geometry_mode === 'polygon') &&
      screenPts.length >= 2 &&
      screenPts.every((pt) => pt !== null)
    ) {
      edges.push({
        featureId: feat.id,
        geometryMode: feat.geometry_mode,
        points: screenPts as Array<{ x: number; y: number }>,
      })
    }
  }

  return { handles, edges }
}

function updateRaycasterFromClient(clientX: number, clientY: number): DOMRect | null {
  if (!renderer || !camera) return null
  const rect = renderer.domElement.getBoundingClientRect()
  pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1
  pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1
  raycaster.setFromCamera(pointerNdc, camera)
  return rect
}

function onMouseDown(e: MouseEvent) {
  if (e.button === 1) {
    e.preventDefault()
  }
}

function onAuxClick(e: MouseEvent) {
  e.preventDefault()
}

function restoreOrbitControls() {
  if (controls) controls.enabled = true
}

function onPointerDown(e: PointerEvent) {
  pointerDownX = e.clientX
  pointerDownY = e.clientY
  if (e.button === 1) e.preventDefault()

  if (!renderer || !camera || !survey.value) return
  const rect = renderer.domElement.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top

  const isDrawing = store.activeDrawingFeature !== null
  const { handles } = getProjected3DCandidates(rect)
  const hitHandle = hitTestHandles(mx, my, handles, HANDLE_HIT_RADIUS_PX)

  if (hitHandle) {
    if (e.button === 1) {
      e.preventDefault()
      e.stopImmediatePropagation()
      middleDeleting3D = true
      try {
        renderer.domElement.setPointerCapture(e.pointerId)
      } catch {
        // ignore if pointerId is not active
      }
      if (controls) controls.enabled = false
      store.deletePick(hitHandle.featureId, hitHandle.pickIndex)
      return
    }
    if (!isDrawing && e.button === 0) {
      const feat = store.pickedFeatures.find((f) => f.id === hitHandle.featureId)
      if (feat) {
        e.stopImmediatePropagation()
        try {
          renderer.domElement.setPointerCapture(e.pointerId)
        } catch {
          // ignore if pointerId is not active
        }
        if (controls) controls.enabled = false
        draggingHandle3D = {
          featureId: feat.id,
          pickIndex: hitHandle.pickIndex,
          planeAxis: feat.plane_axis,
          plane: getLockedThreePlane(feat.plane_axis, feat.plane_index),
        }
        store.selectedFeatureId = feat.id
        store.setHoveredPick({ featureId: feat.id, pickIndex: hitHandle.pickIndex })
        cursor3DStyle.value = 'grabbing'
        return
      }
    }
  }
}

function onPointerMove(e: PointerEvent) {
  if (!renderer || !camera || !survey.value) return

  if (draggingHandle3D && (e.buttons & 1) === 0) {
    updateRaycasterFromClient(e.clientX, e.clientY)
    const hitPt = new THREE.Vector3()
    if (raycaster.ray.intersectPlane(draggingHandle3D.plane, hitPt)) {
      const idx = worldToIndex(hitPt)
      const horiz = draggingHandle3D.planeAxis === 'inline' ? idx.xlIdx : idx.ilIdx
      store.movePick(
        draggingHandle3D.featureId,
        draggingHandle3D.pickIndex,
        horiz,
        idx.sampleIdx,
        { commit: true },
      )
    }
    draggingHandle3D = null
    restoreOrbitControls()
    cursor3DStyle.value = 'default'
    return
  }

  if (draggingHandle3D) {
    updateRaycasterFromClient(e.clientX, e.clientY)
    const hitPt = new THREE.Vector3()
    if (raycaster.ray.intersectPlane(draggingHandle3D.plane, hitPt)) {
      const idx = worldToIndex(hitPt)
      const horiz = draggingHandle3D.planeAxis === 'inline' ? idx.xlIdx : idx.ilIdx
      store.movePick(
        draggingHandle3D.featureId,
        draggingHandle3D.pickIndex,
        horiz,
        idx.sampleIdx,
        { commit: false },
      )
    }
    return
  }

  if (e.buttons !== 0) {
    hoveredPatchScore.value = null
    return
  }

  const rect = renderer.domElement.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top

  if (!store.activeDrawingFeature) {
    const { handles, edges } = getProjected3DCandidates(rect)
    const hitHandle = hitTestHandles(mx, my, handles, HANDLE_HIT_RADIUS_PX)
    if (hitHandle) {
      store.setHoveredPick({ featureId: hitHandle.featureId, pickIndex: hitHandle.pickIndex })
      cursor3DStyle.value = 'grab'
      hoveredPatchScore.value = null
      return
    }
    store.setHoveredPick(null)
    if (store.pickMode && !e.shiftKey) {
      const hitEdge = hitTestEdges(mx, my, edges, EDGE_HIT_RADIUS_PX)
      if (hitEdge) {
        const feat = store.pickedFeatures.find((f) => f.id === hitEdge.featureId)
        if (feat) {
          updateRaycasterFromClient(e.clientX, e.clientY)
          const plane = getLockedThreePlane(feat.plane_axis, feat.plane_index)
          const hitPt = new THREE.Vector3()
          if (
            raycaster.ray.intersectPlane(plane, hitPt) &&
            !isWorldPointOccludedBySlices(hitPt)
          ) {
            cursor3DStyle.value = 'copy'
            hoveredPatchScore.value = null
            return
          }
        }
      }
    }
  } else {
    store.setHoveredPick(null)
  }

  cursor3DStyle.value = 'default'

  const id = getHitCloudInstance(e.clientX, e.clientY)
  if (id !== null) {
    const ilIdx = visiblePatchCoords[id * 3]
    const xlIdx = visiblePatchCoords[id * 3 + 1]
    const tIdx = visiblePatchCoords[id * 3 + 2]
    hoveredPatchScore.value = {
      il: survey.value.ilines[ilIdx] ?? ilIdx,
      xl: survey.value.xlines[xlIdx] ?? xlIdx,
      sample: survey.value.samples[tIdx] ?? tIdx,
      val: visiblePatchValues[id],
    }
  } else {
    hoveredPatchScore.value = null
  }
}

function onPointerUp(e: PointerEvent) {
  if (renderer?.domElement.hasPointerCapture(e.pointerId)) {
    renderer.domElement.releasePointerCapture(e.pointerId)
  }

  if (middleDeleting3D) {
    middleDeleting3D = false
    restoreOrbitControls()
    return
  }

  if (draggingHandle3D) {
    updateRaycasterFromClient(e.clientX, e.clientY)
    const hitPt = new THREE.Vector3()
    if (raycaster.ray.intersectPlane(draggingHandle3D.plane, hitPt)) {
      const idx = worldToIndex(hitPt)
      const horiz = draggingHandle3D.planeAxis === 'inline' ? idx.xlIdx : idx.ilIdx
      store.movePick(
        draggingHandle3D.featureId,
        draggingHandle3D.pickIndex,
        horiz,
        idx.sampleIdx,
        { commit: true },
      )
    }
    draggingHandle3D = null
    restoreOrbitControls()
    cursor3DStyle.value = 'default'
    return
  }

  restoreOrbitControls()

  if (e.button !== 0) return
  if (Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY) > 5) return
  if (!renderer || !camera || !survey.value) return

  const now = performance.now()
  if (now < justFinishedUntil3D) return

  const rect = updateRaycasterFromClient(e.clientX, e.clientY)
  if (!rect) return

  if (store.pickMode && !e.shiftKey && showSlices.value) {
    const active = store.activeDrawingFeature
    if (active) {
      const activeMesh =
        active.plane_axis === 'inline' &&
        active.plane_index === store.ilIdx &&
        inlineMesh?.visible
          ? inlineMesh
          : active.plane_axis === 'xline' &&
              active.plane_index === store.xlIdx &&
              xlineMesh?.visible
            ? xlineMesh
            : null

      if (activeMesh) {
        const hits = raycaster.intersectObject(activeMesh, false)
        if (hits.length > 0) {
          const idx = worldToIndex(hits[0].point)
          const isDoubleClick =
            e.detail >= 2 ||
            (now - lastPrimaryClickTime3D < 320 &&
              Math.hypot(e.clientX - lastClickX3D, e.clientY - lastClickY3D) < 8)
          if (isDoubleClick) {
            justFinishedUntil3D = now + 260
            lastPrimaryClickTime3D = 0
          } else {
            lastPrimaryClickTime3D = now
            lastClickX3D = e.clientX
            lastClickY3D = e.clientY
          }
          const horiz = active.plane_axis === 'inline' ? idx.xlIdx : idx.ilIdx
          void store.placePickOnPlane(
            active.plane_axis,
            active.plane_index,
            horiz,
            idx.sampleIdx,
            { isDoubleClick },
          )
          return
        }
      }
    }

    if (!active) {
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const { edges } = getProjected3DCandidates(rect)
      const hitEdge = hitTestEdges(mx, my, edges, EDGE_HIT_RADIUS_PX)
      if (hitEdge) {
        const feat = store.pickedFeatures.find((f) => f.id === hitEdge.featureId)
        if (feat) {
          const plane = getLockedThreePlane(feat.plane_axis, feat.plane_index)
          const hitPt = new THREE.Vector3()
          if (
            raycaster.ray.intersectPlane(plane, hitPt) &&
            !isWorldPointOccludedBySlices(hitPt)
          ) {
            const idx = worldToIndex(hitPt)
            const horiz = feat.plane_axis === 'inline' ? idx.xlIdx : idx.ilIdx
            store.insertPickOnSegment(feat.id, hitEdge.segmentIndex, horiz, idx.sampleIdx)
            return
          }
        }
      }
    }

    const sliceMeshes = getVisibleSliceMeshes()
    const hits = raycaster.intersectObjects(sliceMeshes, false)
    if (hits.length > 0) {
      const hit = hits[0]
      const idx = worldToIndex(hit.point)

      if (hit.object === inlineMesh) {
        lastPrimaryClickTime3D = now
        lastClickX3D = e.clientX
        lastClickY3D = e.clientY
        void store.placePickOnPlane('inline', store.ilIdx, idx.xlIdx, idx.sampleIdx)
        return
      }
      if (hit.object === xlineMesh) {
        lastPrimaryClickTime3D = now
        lastClickX3D = e.clientX
        lastClickY3D = e.clientY
        void store.placePickOnPlane('xline', store.xlIdx, idx.ilIdx, idx.sampleIdx)
        return
      }
      if (hit.object === sampleMesh) {
        store.setCursor({
          ilIdx: idx.ilIdx,
          xlIdx: idx.xlIdx,
          sampleIdx: store.sampleIdx,
        })
        return
      }
    }
  }

  // Fallback / cursor-only: raycast cloudMesh then slices
  const id = getHitCloudInstance(e.clientX, e.clientY)
  if (id !== null) {
    const ilIdx = visiblePatchCoords[id * 3]
    const xlIdx = visiblePatchCoords[id * 3 + 1]
    const tIdx = visiblePatchCoords[id * 3 + 2]
    store.setCursor({ ilIdx, xlIdx, sampleIdx: tIdx })
    return
  }

  const sliceMeshes = getVisibleSliceMeshes()
  if (sliceMeshes.length > 0) {
    const hits = raycaster.intersectObjects(sliceMeshes, false)
    if (hits.length > 0) {
      const idx = worldToIndex(hits[0].point)
      store.setCursor({
        ilIdx: hits[0].object === inlineMesh ? store.ilIdx : idx.ilIdx,
        xlIdx: hits[0].object === xlineMesh ? store.xlIdx : idx.xlIdx,
        sampleIdx: hits[0].object === sampleMesh ? store.sampleIdx : idx.sampleIdx,
      })
    }
  }
}

function onPointerCancel(e: PointerEvent) {
  if (renderer?.domElement.hasPointerCapture(e.pointerId)) {
    renderer.domElement.releasePointerCapture(e.pointerId)
  }
  draggingHandle3D = null
  middleDeleting3D = false
  restoreOrbitControls()
  cursor3DStyle.value = 'default'
}

function onPointerLeave() {
  hoveredPatchScore.value = null
  if (!draggingHandle3D) {
    store.setHoveredPick(null)
    cursor3DStyle.value = 'default'
  }
}

function onWindowBlur() {
  draggingHandle3D = null
  middleDeleting3D = false
  restoreOrbitControls()
  cursor3DStyle.value = 'default'
}

function resetCamera() {
  if (!camera || !controls) return
  camera.position.set(2.45, 1.45, 2.55)
  controls.target.set(0, 0, 0)
  controls.update()
}

function handleResize() {
  const container = containerRef.value
  if (!container || !renderer || !camera) return
  const rect = container.getBoundingClientRect()
  const w = Math.max(1, Math.floor(rect.width))
  const h = Math.max(1, Math.floor(rect.height))
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  renderer.setSize(w, h)
}

onMounted(() => {
  void Promise.all([
    store.ensureSection('inline'),
    store.ensureSection('xline'),
    store.ensureSection('sample'),
  ])
  initThree()
  if (containerRef.value) {
    resizeObs = new ResizeObserver(() => handleResize())
    resizeObs.observe(containerRef.value)
  }
})

onBeforeUnmount(() => {
  resizeObs?.disconnect()
  const container = containerRef.value
  if (container) {
    container.removeEventListener('pointerdown', onSliceDragStart, { capture: true })
    container.removeEventListener('pointermove', onSliceDragMove, { capture: true })
    container.removeEventListener('pointerup', onSliceDragEnd, { capture: true })
    container.removeEventListener('pointercancel', onSliceDragEnd, { capture: true })
  }
  if (animFrameId !== null) cancelAnimationFrame(animFrameId)
  restoreOrbitControls()
  window.removeEventListener('blur', onWindowBlur)
  if (renderer?.domElement) {
    renderer.domElement.removeEventListener('mousedown', onMouseDown)
    renderer.domElement.removeEventListener('auxclick', onAuxClick)
    renderer.domElement.removeEventListener('pointerdown', onPointerDown, { capture: true })
    renderer.domElement.removeEventListener('pointermove', onPointerMove)
    renderer.domElement.removeEventListener('pointerup', onPointerUp)
    renderer.domElement.removeEventListener('pointercancel', onPointerCancel)
    renderer.domElement.removeEventListener('pointerleave', onPointerLeave)
    renderer.domElement.remove()
  }
  controls?.dispose()
  disposePickGroupTextures()
  disposeCloudMesh()
  disposeObject(scene)
  inlineTex?.dispose()
  xlineTex?.dispose()
  sampleTex?.dispose()
  renderer?.dispose()
})

watch(
  () => store.activeSurveyId,
  () => {
    showTimeSlice.value = false
    rebuildSurveyGeometry()
    updateSlicePositions()
    updateGatherCurtain()
    updateFeaturePickMarker()
    updateCloud()
    updateReservoir()
  },
)

watch(
  [
    () => store.showReservoir,
    () => store.reservoirEnvelope,
    () => store.interpretedHorizons,
    () => store.visibleWellPicks,
    () => store.reservoirOpacity,
  ],
  () => {
    updateReservoir()
  },
)

watch(
  [
    () => store.sections.inline,
    () => store.sections.xline,
    () => store.sections.sample,
    () => store.seismicColormap,
    () => store.effectiveClipAmplitude,
  ],
  () => {
    updateSliceTextures()
  },
)

watch(
  [() => store.ilIdx, () => store.xlIdx, () => store.sampleIdx],
  () => {
    updateSlicePositions()
  },
)

watch(
  () => store.gatherPanelData,
  () => {
    updateGatherCurtain()
  },
)

watch(
  [
    () => store.pickedFeatures,
    () => store.hoveredPickRef,
    () => store.ilIdx,
    () => store.xlIdx,
    () => store.activeSurveyId,
    () => store.activeEmbeddingSet,
    () => store.activeOverlay,
  ],
  () => {
    updateFeaturePickMarker()
  },
  { deep: true },
)

watch(
  [() => store.activeOverlay, () => store.overlayColormap, showCloud],
  () => {
    updateCloud()
  },
)

watch([showSlices, showTimeSlice], ([slices, timeSlice]) => {
  if (inlineMesh) inlineMesh.visible = slices
  if (xlineMesh) xlineMesh.visible = slices
  if (sampleMesh) sampleMesh.visible = slices && timeSlice
})

watch(showGatherCurtain, (v) => {
  if (gatherGroup) gatherGroup.visible = v
})

watch(showBox, (v) => {
  if (boxHelper) boxHelper.visible = v
})
</script>

<template>
  <div class="relative flex flex-col h-full w-full bg-slate-950 select-none overflow-hidden">
    <!-- Sub-header controls -->
    <div
      class="flex flex-wrap items-center gap-2 px-2.5 py-1.5 bg-slate-900/80 border-b border-slate-800/90 text-xs shrink-0"
    >
      <div class="flex items-center gap-1">
        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] border transition-colors"
          :class="
            showSlices
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
              : 'bg-slate-950 border-slate-800 text-slate-400'
          "
          @click="showSlices = !showSlices"
        >
          <Layers class="w-3 h-3" />
          <span>Slices</span>
        </button>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] border transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          :class="
            showTimeSlice
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
              : 'bg-slate-950 border-slate-800 text-slate-400'
          "
          :disabled="!showSlices"
          :aria-pressed="showTimeSlice"
          :title="showSlices ? 'Show or hide the time slice' : 'Turn on Slices to use the time slice'"
          @click="showTimeSlice = !showTimeSlice"
        >
          <Clock class="w-3 h-3" />
          <span>Time slice</span>
        </button>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] border transition-colors"
          :class="
            showCloud
              ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
              : 'bg-slate-950 border-slate-800 text-slate-400'
          "
          @click="showCloud = !showCloud"
        >
          <Eye class="w-3 h-3" />
          <span>3D Cloud</span>
          <span
            v-if="store.activeOverlay"
            class="font-mono-num text-[10px] text-cyan-200/80 ml-0.5"
          >
            ({{ cloudVisibleCount }})
          </span>
        </button>

        <button
          type="button"
          class="px-2 py-0.5 rounded text-[11px] border transition-colors"
          :class="
            showGatherCurtain
              ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
              : 'bg-slate-950 border-slate-800 text-slate-400'
          "
          @click="showGatherCurtain = !showGatherCurtain"
        >
          Gather Curtain
        </button>

        <button
          v-if="store.hasHorizons"
          type="button"
          class="px-2 py-0.5 rounded text-[11px] border transition-colors"
          :class="
            store.showReservoir
              ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
              : 'bg-slate-950 border-slate-800 text-slate-400'
          "
          title="Show the Volve Hugin reservoir model and interpreted horizons (ground truth)"
          @click="store.toggleReservoir()"
        >
          Reservoir
        </button>

        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] border transition-colors"
          :class="
            showBox
              ? 'bg-slate-800 border-slate-700 text-slate-200'
              : 'bg-slate-950 border-slate-800 text-slate-400'
          "
          @click="showBox = !showBox"
        >
          <Box class="w-3 h-3" />
          <span>Box</span>
        </button>
      </div>

      <div class="ml-auto flex items-center gap-2">
        <span class="hidden sm:inline text-[10px] text-slate-500">
          Drag to orbit · Shift+drag slice to move · Scroll to zoom · Click patch to jump
        </span>
        <button
          type="button"
          class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-slate-950 border border-slate-800 text-slate-300 hover:bg-slate-800 transition-colors"
          @click="resetCamera"
        >
          <RotateCcw class="w-3 h-3" />
          <span>Reset Cam</span>
        </button>
      </div>
    </div>

    <!-- Three.js Mount -->
    <div
      ref="containerRef"
      class="relative flex-1 min-h-0 overflow-hidden bg-slate-950"
      :style="{ cursor: cursor3DStyle }"
    >
      <!-- Slice drag readout -->
      <div
        v-if="sliceDragReadout"
        class="absolute top-2.5 left-3 px-2.5 py-1 rounded bg-slate-950/90 border border-cyan-500/40 text-[11px] font-mono-num text-cyan-200 pointer-events-none shadow-lg"
      >
        {{ sliceDragReadout }}
      </div>

      <!-- Hover voxel badge -->
      <div
        v-if="hoveredPatchScore"
        class="absolute bottom-2.5 left-3 px-2.5 py-1 rounded bg-slate-950/90 border border-slate-800 text-[11px] font-mono-num text-slate-200 pointer-events-none flex items-center gap-2.5 shadow-lg"
      >
        <span>IL <b class="text-cyan-300">{{ hoveredPatchScore.il }}</b></span>
        <span>XL <b class="text-cyan-300">{{ hoveredPatchScore.xl }}</b></span>
        <span>T <b class="text-cyan-300">{{ hoveredPatchScore.sample }}</b> ms</span>
        <span>Val <b class="text-emerald-300">{{ hoveredPatchScore.val.toFixed(3) }}</b></span>
      </div>

      <div
        v-if="!survey"
        class="absolute inset-0 flex items-center justify-center text-xs text-slate-500 pointer-events-none"
      >
        No survey loaded
      </div>
    </div>
  </div>
</template>
