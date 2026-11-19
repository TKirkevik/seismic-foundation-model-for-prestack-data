# Pre-Stack Seismic Latent Exploration

Interactive geophysical exploration of 4D pre-stack seismic surveys using 3D foundation-model representations to discover analogous subsurface features across space and offset.

## Language

### Seismic Geometry

**Survey**:
A co-registered 4D pre-stack seismic dataset defined on a shared `(Inline, Crossline, Sample)` grid across ordered `Offset` bins or angle ranges.
_Avoid_: Project, raw data, cube set

**Common-Offset Volume**:
A single 3D seismic volume `(Inline, Crossline, Sample)` extracted at one discrete offset value within a Survey.
_Avoid_: Stack, sub-cube, partial volume

**Section**:
A 2D slice through a single Offset Cube along a constant Inline, Crossline, or Sample plane.
_Avoid_: Profile, cut, slice view

**Gather**:
A 2D pre-stack slice `(Offset, Sample)` at a single `(Inline, Crossline)` coordinate showing amplitude variation across offsets.
_Avoid_: Trace fan, CMP panel, offset trace

**Gather Panel**:
An ordered sequence of Gathers sampled along an Inline or Crossline from a starting `(Inline, Crossline)` coordinate at a fixed spatial increment for a specified number of locations.
_Avoid_: Pre-stack section, supergather, gather strip

### Latent Space & Interpretation

**Patch**:
A localized `16×16×16` 3D sub-volume within a Common-Offset Volume represented by a single vector in latent space.
_Avoid_: Token, voxel block, tile, window

**Embedding Lattice**:
The spatial grid of Patch vectors aligned to the Survey coordinate system across offsets.
_Avoid_: Feature map, vector store, latent cube

**Offset Trajectory**:
The path a single spatial Patch traces through latent space across offsets, summarized by its zero-offset anchor, primary offset direction (linear trend with `|offset|`), and offset curvature (leading nonlinear residual axis).
_Avoid_: AVO curve, offset path, latent drift

**Feature Pick**:
A user-selected `(Inline, Crossline, Sample)` coordinate (with optional offset scope) used as the query anchor in latent space.
_Avoid_: Seed, marker, click point, label

**Similarity Volume**:
A 3D scalar field over the Survey grid quantifying latent-space similarity between every Patch and a Feature Pick—evaluated at a selected offset, at zero-offset, or along the Offset Trajectory (`offset_direction` / `offset_curvature`).
_Avoid_: Heatmap cube, correlation volume, distance map

### Ground Truth

**Horizon**:
A `(Inline, Crossline)` grid of two-way time (ms) for one geological surface, `NaN` where undefined. Served per Survey from `<survey>_horizons.npz` with a `kind` of `horizon` (interpreted), `top` or `base` (the two bounding surfaces of the Reservoir Envelope).
_Avoid_: Surface map, pick grid, marker

**Reservoir Envelope**:
The interval between the `top` and `base` Horizons of the Volve Eclipse reservoir model (Hugin Fm.), converted from depth to PP time by tying the top to the interpreted Top Hugin horizon and the base to the well-derived Hugin interval velocity. Toggled as one layer across every viewer.
_Avoid_: Answer key, truth mask, reservoir body

**Well Pick**:
A formation top observed in a wellbore (`well`, `formation`, TVDSS, TWT) placed at its own `(Inline, Crossline)` bin; drawn as a marker when the current Section or Time Slice passes near it.
_Avoid_: Well top, marker, formation marker
