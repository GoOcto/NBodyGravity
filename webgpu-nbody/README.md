# WebGPU N-Body Gravity Simulation

A modern, high-performance N-body gravity simulation using WebGPU compute shaders. This is a complete rewrite of the original DirectX 10 implementation using cutting-edge web technologies.

**🎮 [Try the Live Demo](https://demo.goocto.com/webgpu-nbody/)**

![N-Body Simulation](https://img.shields.io/badge/WebGPU-Enabled-brightgreen)
![Compute Shaders](https://img.shields.io/badge/WGSL-Compute%20Shaders-blue)

## 🚀 Features

- **Two pages, one shared control-panel flow**: `index.html` (3D) and
  `index-2d.html` (2D) each expose an in-page dropdown to switch physics
  backend/algorithm without navigating away — a "Demo" dropdown still
  switches between the two pages themselves.
  - **`index.html` (3D)**: `CPU direct (O(N²))` vs. `GPU FMM octree`,
    switched live via a state-preserving backend swap (GPU state is read
    back only before entering CPU mode or resizing the particle set; CPU
    state is uploaded before entering GPU mode).
  - **`index-2d.html` (2D)**: `Grid-Based — CPU`, `Grid-Based — GPU
    Compute` (multi-resolution grid + separable-blur approximation), and
    `Particle-Mesh — GPU FFT Poisson` (an exact 2D field solve). Switching
    algorithms here tears down and reconstructs the simulation object
    (the three algorithms' internal GPU pipelines are too different to
    share live state) but carries over every current control-panel value.
- **Real-time N-Body Physics**: Direct interactions remain available for
  near-field cells while distant cells use octree multipoles (3D/GPU FMM)
  or grid/FFT-based field solves (2D)
- **Interactive Controls**: particle count, gravity strength, time scale,
  damping, restitution (2D), boundary handling (2D), and GPU FMM opening
  angle (theta) / softening (3D) — all live, all shared via `js/common.js`
  so the same setting behaves identically across algorithms that support it
- **Orbital Camera** (3D only): Mouse/touch controls for 360° viewing
- **Cross-Platform**: Runs in any WebGPU-compatible browser
- **High Performance**: GPU-accelerated compute shaders for maximum efficiency

## 🌐 Browser Requirements

WebGPU is still experimental. You'll need:

### Chrome/Chromium (Recommended)
- Chrome 113+ or Edge 113+
- Enable WebGPU: `chrome://flags/#enable-unsafe-webgpu`

### Firefox
- Firefox Nightly
- Enable WebGPU: `about:config` → `dom.webgpu.enabled = true`

### Safari
- Safari Technology Preview
- WebGPU should be enabled by default

## 🛠️ Quick Start

### Option 1: Python Server (Recommended)
```bash
cd webgpu-nbody
python -m http.server 8080
# Or if you have Python 3:
python3 -m http.server 8080
```

### Option 2: Node.js Server
```bash
cd webgpu-nbody
npm install
npm run dev-node
```

### Option 3: Any HTTP Server
Any local HTTP server will work. HTTPS is required for WebGPU in some browsers.

Then open: http://localhost:8080

## 🎮 Controls

### `index.html` (3D)
- **Mouse Drag**: Orbit around the simulation
- **Mouse Wheel**: Zoom in/out
- **Touch**: Mobile-friendly orbital controls
- **Particle Count**: 100–40,000
- **Gravity Strength**: Control gravitational force intensity
- **Time Scale**: Speed up or slow down the simulation
- **Damping**: Add velocity damping to stabilize the system
- **Physics backend**: `CPU direct (O(N²))` or `GPU FMM octree`, switchable live
- **Octree Opening Angle (θ) / Softening** (GPU FMM only, disabled in CPU
  mode): live per-frame tuning of the FMM's accuracy/performance trade-off
  and force softening — the octree's fixed depth (4 levels) is not exposed
  since changing it requires rewriting the traversal shader
- **Reset**: Reinitialize particle positions and velocities

The backend selector does not reset particles. If GPU initialization or a mode
transition fails, the current backend remains active and the error is shown in
the status line; there is no silent fallback.

### `index-2d.html` (2D)
- **Algorithm**: `Grid-Based — CPU`, `Grid-Based — GPU Compute`, or
  `Particle-Mesh — GPU FFT Poisson` — switching reconstructs the simulation
  (see Features above) but preserves every other control's current value
- **Particle Count**: up to 1,000,000 (CPU grid) or 4,000,000 (either GPU algorithm)
- **Gravity Strength**, **Time Scale** (log-scale slider, 0.001x–1000x)
- **Damping**: has no effect in `Particle-Mesh` mode (its integrator has no
  damping term by design) — the control is disabled with a note in that mode
- **Restitution**: bounce elasticity at domain boundaries
- **Initial Orbital Speed**: scales the circular velocity particles are
  seeded with (0 = no spin, radial collapse; 1 = default). Always follows a
  `Math.sqrt(.../radius)` law so nearer particles orbit faster — the same
  shape used by `index.html`'s 3D demo — and starts every algorithm at the
  same speed for the same Particle Count/Gravity Strength, fixing a
  previous bug where the three algorithms started at different, radius-
  independent speeds. Changing it immediately regenerates the disc.
- **Boundary Handling**: `Clamping (Bounce)`, `Periodic (Wrap)`, or `Deletion (Cull)`
- **Reset**: Reinitialize particle positions and velocities

Both pages share a "Demo" dropdown to jump between the 3D and 2D pages.

## 🏗️ Architecture

### File Layout
```
webgpu-nbody/
├── index.html          # 3D page (CPU direct / GPU FMM octree)
├── index-2d.html        # 2D page (Grid CPU / Grid GPU / Particle-Mesh GPU)
├── styles.css
├── js/
│   ├── common.js         # shared constants + helpers (WebGPU init, dispatch
│   │                      # sizing, shader loading, boundary-mode constants,
│   │                      # grid resolution/particle-count tiers/fixed
│   │                      # timestep, control-panel binding/slider-sync
│   │                      # helpers) used by every simulation/app file below
│   ├── camera.js          # orbital camera controller (3D only)
│   ├── gl-matrix.js       # minimal mat4/vec3 helpers (3D only)
│   ├── sim-3d.js          # SimpleNBodySimulation: CPU direct O(N²) + GPU FMM
│   │                      # octree, with a live, state-preserving mode switch
│   ├── sim-grid-cpu.js    # GridNBodySimulation: CPU multi-resolution grid + blur
│   ├── sim-grid-gpu.js    # GpuGridNBodySimulation: same algorithm, GPU compute
│   ├── sim-pm-gpu.js      # PmGpuNBodySimulation: GPU particle-mesh FFT Poisson solver
│   ├── app-3d.js          # App shell for index.html (controls, RAF loop, stats)
│   └── app-2d.js          # App shell for index-2d.html (controls, RAF loop,
│                          # stats, and the tear-down/recreate algorithm switch)
└── shaders/               # every WGSL shader lives directly here (flattened,
                            # single directory, no unused/orphaned files)
```

Each `sim-*.js` file owns a fully self-contained simulation class (particle
state, WebGPU pipelines/bind groups, integration, rendering) so the
algorithm-specific logic — where these demos genuinely diverge — stays
isolated to one file and one shader set per algorithm, while anything that
*should* be identical across algorithms (domain size, mass scaling,
boundary-mode codes, damping/restitution/orbital-speed defaults, the shared
mass-grid resolution, the fixed physics timestep, particle-count tiers,
WebGPU bootstrap, shader loading, control-panel wiring/slider-sync helpers)
is centralized in `js/common.js` and can no longer silently drift apart
between files. Each `sim-*.js` file's own top-of-file block is left with
only the constants that are genuinely specific to that one algorithm (e.g.
`FORCE_SCALE`, LOD level count, blur kernel radius, FFT sub-stepping cap).
Where a value is exposed as a control-panel slider (e.g. Particle Count's
min/max/step, which differs a lot between the CPU-only grid algorithm and
the two GPU algorithms), `common.js`'s `applySliderRange()` keeps the
slider's bounds in sync with whichever model/algorithm is active — called
on initial page load and again on every algorithm/backend switch.

### Compute Shaders (WGSL) — 3D (FMM octree)
- `octree-leaf.wgsl`: Assigns particles to fixed-depth leaf cells on the GPU
- `octree-leaf-multipole.wgsl`: Builds leaf mass/center-of-mass multipoles
- `octree-aggregate.wgsl`: Reduces leaf multipoles through the hierarchy
- `fmm-force-compute.wgsl`: Traverses the GPU octree, using direct leaf interactions and accepted distant multipoles
- `fmm-integrate-compute.wgsl`: Updates particle positions and velocities
- `octree-clear.wgsl`: Clears per-frame multipole accumulators

### Render Pipeline (WGSL) — 3D
- `particle-vertex-quad.wgsl`: Transforms particle positions to screen space
- `particle-fragment-quad.wgsl`: Renders particles with color-coded velocities

### JavaScript Classes
- `SimpleNBodySimulation` (`js/sim-3d.js`): CPU solver, GPU compute passes, resource management, and state synchronization
- `GridNBodySimulation` / `GpuGridNBodySimulation` (`js/sim-grid-cpu.js` / `js/sim-grid-gpu.js`): multi-resolution grid + separable-blur 2D approximation, CPU and GPU compute variants
- `PmGpuNBodySimulation` (`js/sim-pm-gpu.js`): 2D particle-mesh FFT Poisson solver (see below)
- `CameraController` (`js/camera.js`): Orbital camera with smooth mouse/touch controls (3D only)
- `App` (`js/app-3d.js`, `js/app-2d.js`): per-page application lifecycle, control-panel wiring, and UI management

## 🔧 Technical Details

### Simulation Algorithm
The CPU backend retains the direct pairwise O(N²) calculation. The GPU backend
executes separate WebGPU compute passes to clear a fixed four-level octree,
assign particles to leaves, build monopole mass/center-of-mass data bottom-up,
traverse the hierarchy with a fixed opening criterion, and integrate particles.
Near leaves use direct particle interactions; accepted distant nodes use their
mass multipole. Both backends apply the same damping and boundary conditions.

The internal GPU accuracy settings are fixed (depth 4, opening threshold 0.65,
softening 0.01); they are not exposed as UI tuning controls.

### Performance Optimizations
- **Compute Workgroups**: Uses 64-thread workgroups
- **Structured Buffers**: Shared particle layout for rendering and both solvers
- **Separate passes**: Octree construction, multipole aggregation, force traversal, and integration are distinct compute passes
- **GPU readback**: A staging buffer synchronizes state only when switching back to the CPU backend or resizing the particle set
- **GPU frame back-pressure**: GPU simulation and render submissions wait for the previous frame to complete, preventing an unbounded queue backlog

The Compute Encode and Render Encode/Submit values in the UI are CPU-side
timings for command encoding and `queue.submit()`. They do not measure GPU
execution time. GPU-to-CPU readback is not performed every frame; it occurs
only when switching from GPU to CPU or resizing the particle set.

### WebGPU Features Used
- Compute pipelines for physics simulation
- Render pipelines for particle visualization  
- Storage buffers for particle data
- Uniform buffers for simulation parameters
- Bind groups for resource management

## 🧮 Particle-Mesh GPU Demo (`index-2d.html`'s "Particle-Mesh — GPU FFT Poisson" algorithm, `js/sim-pm-gpu.js`)

Alongside the grid-based blur approximation (`index-2d.html`'s "Grid-Based —
GPU Compute" algorithm, `js/sim-grid-gpu.js`), the repo also has a
physically-grounded **Particle-Mesh (PM)** GPU algorithm. Where the
blur-based algorithm approximates gravity with a hand-tuned separable-blur
kernel across a multi-resolution grid hierarchy, the PM algorithm actually
solves the field equation:

1. **Cloud-in-Cell (CIC) mass deposit** — each particle bilinearly spreads
   its mass across its 4 surrounding grid cells (`pm-mass-scatter-cic.wgsl`),
   rather than the nearest-grid-point deposit the blur demo uses.
2. **2D FFT** of the single (no LOD hierarchy) mass grid, via GPU compute
   shaders implementing a from-scratch, ping-pong, bit-reversal +
   Cooley-Tukey butterfly FFT (`pm-fft-*.wgsl`).
3. **The true 2D Poisson equation is solved exactly** in the frequency
   domain — multiplying by the Green's function `1/|k|²` and zeroing the
   undefined k=0 (mean-density) term, the standard periodic-Poisson trick
   (`pm-poisson-greens.wgsl`) — rather than approximated by a blur kernel.
   This models genuine 2D gravity: a logarithmic potential, i.e. a force
   that decays as `1/r`, not the 3D-style `1/r²` analog the blur demo's
   kernel was tuned to mimic.
4. An **inverse 2D FFT** transforms back to a real potential field
   (`pm-potential-extract.wgsl`), whose gradient (reusing the blur demo's
   `gradient.wgsl`/`accumulate-gradient.wgsl` unmodified) gives each
   particle's acceleration via the same matching bilinear interpolation
   used for the CIC deposit (avoiding self-force artifacts).
5. **No velocity damping** — `pm-integrate.wgsl` is a plain symplectic
   Euler integrator with no damping multiply, unlike the blur demo's
   `integrate.wgsl`.
6. **Fixed internal integration sub-stepping, independent of playback
   speed** — the physics step size (`FIXED_DT`) never changes; a
   real-time accumulator decides how many fixed-size steps to run per
   rendered frame (capped, so extreme Time Scale values fall behind
   rather than taking one giant, unstable step or unboundedly stalling
   the frame). This decouples numerical stability from the Time Scale
   slider entirely.

Because the FFT Poisson solve is inherently periodic, gravity always wraps
at the domain edge (mass near one edge gravitationally interacts with the
opposite edge) regardless of the selected particle `Boundary Handling`
mode — that setting only affects what happens to a particle's own
position/velocity at the edge, not the field solve itself.

**Initial orbital velocities** are also computed on the GPU rather than
from an analytic formula: `initializeParticles()` uploads the random disc
with zero velocity, runs one field-solve pass to get the *actual*
acceleration field for that specific mass distribution, then
`pm-init-circular-velocity.wgsl` converts each particle's local
(radially-inward) acceleration into the exact tangential speed needed for
a circular orbit (`v = sqrt(r · |accel|)`) — no CPU readback, and no
guessing at the disc's enclosed-mass profile.

## 🎨 Customization

### Modify Physics Parameters
Edit the defaults in the constructor of the relevant `js/sim-*.js` class (e.g. `js/sim-3d.js`'s `SimpleNBodySimulation`, shared constants live in `js/common.js`):
```javascript
this.gravityStrength = 1.0;  // Gravitational constant
this.damping = 0.999;        // Velocity damping factor
this.timeScale = 1.0;        // Simulation speed multiplier
```

### Change Visual Appearance
Modify `shaders/particle-fragment-quad.wgsl` (3D) or `shaders/colorize.wgsl` (2D) to adjust:
- Particle colors
- Size scaling
- Transparency effects
- Glow effects

### Add New Initial Conditions
Edit `initializeParticles()` in the relevant `js/sim-*.js` file to create:
- Galaxy formations
- Binary systems  
- Clustered configurations
- Custom patterns

## 🚧 Comparison with Original DirectX Version

| Feature | DirectX 10 | WebGPU |
|---------|------------|---------|
| **Platform** | Windows only | Cross-platform |
| **Compute** | Render-to-texture hack | Native compute shaders |
| **Language** | C++ with HLSL | JavaScript with WGSL |
| **Dependencies** | DXUT, DirectX SDK | None (web browser) |
| **Development** | Visual Studio required | Any text editor |
| **Distribution** | Compiled executable | Web-based (instant access) |

## 🔮 Future Enhancements

- **Higher-order multipoles**: Add quadrupole terms to the existing monopole path
- **Compute Performance Analysis**: GPU timing and profiling
- **Advanced Rendering**: Bloom effects, trails, and procedural backgrounds
- **Physics Presets**: Pre-configured galaxy, solar system, and cluster simulations
- **Data Export**: Save/load simulation states and export particle data

## 📝 License

MIT License - see the original project for details.

## 🤝 Contributing

This is a modernization of the original DirectX N-body simulation. Contributions welcome for:
- Performance optimizations
- Visual enhancements  
- New physics features
- Browser compatibility improvements
- Mobile optimization

---

**Note**: This WebGPU implementation demonstrates how modern web technologies can match and exceed the performance of native graphics applications while being more accessible and portable.