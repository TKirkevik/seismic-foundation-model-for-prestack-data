<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Box, Eye, Layers, RotateCcw } from 'lucide-vue-next'
import type { SectionAxis, SectionResponseData } from '../api/types'
import { useWorkbenchStore } from '../stores/workbench'
import { getOverlayLUT, getSeismicLUT, sampleLUT } from '../utils/colormaps'

const store = useWorkbenchStore()
const containerRef = ref<HTMLDivElement | null>(null)

const showSlices = ref(true)
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
let cloudMesh: THREE.InstancedMesh | null = null
let cloudMaxCapacity = 0
let cloudGeometryKey = ''
let visiblePatchCoords = new Int32Array(0) // [il_idx, xl_idx, sample_idx] per visible instance
let visiblePatchValues = new Float32Array(0)

const raycaster = new THREE.Raycaster()
const pointerNdc = new THREE.Vector2()
let pointerDownX = 0
let pointerDownY = 0

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
  scene.add(gatherGroup)
  scene.add(pickGroup)

  rebuildSurveyGeometry()
  updateSliceTextures()
  updateSlicePositions()
  updateGatherCurtain()
  updateFeaturePickMarker()
  updateCloud()

  const animate = () => {
    animFrameId = requestAnimationFrame(animate)
    controls?.update()
    if (renderer && scene && camera) {
      renderer.render(scene, camera)
    }
  }
  animate()

  renderer.domElement.addEventListener('pointerdown', onPointerDown)
  renderer.domElement.addEventListener('pointermove', onPointerMove)
  renderer.domElement.addEventListener('pointerup', onPointerUp)
  renderer.domElement.addEventListener('pointerleave', () => {
    hoveredPatchScore.value = null
  })
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
  })
  sampleMesh = new THREE.Mesh(new THREE.PlaneGeometry(sizeX, sizeZ), sampleMat)
  sampleMesh.rotation.x = -Math.PI / 2
  sampleMesh.visible = showSlices.value
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

function updateFeaturePickMarker() {
  if (!pickGroup) return
  while (pickGroup.children.length > 0) {
    const child = pickGroup.children[0]
    pickGroup.remove(child)
    disposeObject(child)
  }

  const fp = store.featurePick
  if (!fp || !survey.value) return

  const { sizeX, sizeY, sizeZ, nIl, nXl, nSamples } = getWorldDims()
  const patchSize = store.activeOverlay?.patchSize ?? store.activeEmbeddingSet?.patch_size ?? 16
  const pos = indexToWorld(fp.il_idx, fp.xl_idx, fp.sample_idx)

  // Glowing core sphere
  const sphereGeo = new THREE.SphereGeometry(0.032, 16, 16)
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee })
  const sphere = new THREE.Mesh(sphereGeo, sphereMat)
  sphere.position.copy(pos)
  pickGroup.add(sphere)

  // Outer halo sphere
  const haloGeo = new THREE.SphereGeometry(0.052, 16, 16)
  const haloMat = new THREE.MeshBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
  })
  const halo = new THREE.Mesh(haloGeo, haloMat)
  halo.position.copy(pos)
  pickGroup.add(halo)

  // Patch wireframe box (snapped to the queried patch when available)
  const simRes = store.similarityResult
  const simMatchesPick =
    simRes !== null &&
    simRes.pick.il_idx === fp.il_idx &&
    simRes.pick.xl_idx === fp.xl_idx &&
    simRes.pick.sample_idx === fp.sample_idx
  const boxPatchSize = simMatchesPick ? simRes.patch_size || patchSize : patchSize
  let boxPos = pos
  if (simMatchesPick && simRes.pick.patch_idx.length >= 3) {
    const [pi, pj, pk] = simRes.pick.patch_idx
    const half = boxPatchSize * 0.5
    const ilStart = simRes.il_patch_start[pi] ?? pi * boxPatchSize
    const xlStart = simRes.xl_patch_start[pj] ?? pj * boxPatchSize
    const tStart = simRes.t_patch_start[pk] ?? pk * boxPatchSize
    boxPos = indexToWorld(
      Math.min(nIl - 1, ilStart + half - 0.5),
      Math.min(nXl - 1, xlStart + half - 0.5),
      Math.min(nSamples - 1, tStart + half - 0.5),
    )
  }

  const pw = (boxPatchSize / Math.max(1, nXl)) * sizeX
  const ph = (boxPatchSize / Math.max(1, nSamples)) * sizeY
  const pd = (boxPatchSize / Math.max(1, nIl)) * sizeZ
  const boxGeo = new THREE.BoxGeometry(pw, ph, pd)
  const edges = new THREE.EdgesGeometry(boxGeo)
  boxGeo.dispose()
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color: 0x22d3ee }),
  )
  line.position.copy(boxPos)
  pickGroup.add(line)
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

function onPointerDown(e: PointerEvent) {
  pointerDownX = e.clientX
  pointerDownY = e.clientY
}

function onPointerMove(e: PointerEvent) {
  if (e.buttons !== 0 || !survey.value) {
    hoveredPatchScore.value = null
    return
  }
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
  if (e.button !== 0) return
  if (Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY) > 5) return
  const id = getHitCloudInstance(e.clientX, e.clientY)
  if (id === null) return
  const ilIdx = visiblePatchCoords[id * 3]
  const xlIdx = visiblePatchCoords[id * 3 + 1]
  const tIdx = visiblePatchCoords[id * 3 + 2]
  store.setCursor({ ilIdx, xlIdx, sampleIdx: tIdx })
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
  if (animFrameId !== null) cancelAnimationFrame(animFrameId)
  if (renderer?.domElement) {
    renderer.domElement.removeEventListener('pointerdown', onPointerDown)
    renderer.domElement.removeEventListener('pointermove', onPointerMove)
    renderer.domElement.removeEventListener('pointerup', onPointerUp)
    renderer.domElement.remove()
  }
  controls?.dispose()
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
    rebuildSurveyGeometry()
    updateSlicePositions()
    updateGatherCurtain()
    updateFeaturePickMarker()
    updateCloud()
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
  [() => store.featurePick, () => store.similarityResult],
  () => {
    updateFeaturePickMarker()
  },
)

watch(
  [() => store.activeOverlay, () => store.overlayColormap, showCloud],
  () => {
    updateCloud()
  },
)

watch(showSlices, (v) => {
  if (inlineMesh) inlineMesh.visible = v
  if (xlineMesh) xlineMesh.visible = v
  if (sampleMesh) sampleMesh.visible = v
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
          Drag to orbit · Scroll to zoom · Click patch to jump
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
    <div ref="containerRef" class="relative flex-1 min-h-0 overflow-hidden bg-slate-950">
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
