// Shared helpers/constants used by both N-Body demo families (3D
// direct/FMM in sim-3d.js, and 2D grid/PM-FFT in sim-grid-cpu.js /
// sim-grid-gpu.js / sim-pm-gpu.js) so behavior that ought to be identical
// across backends can't quietly drift apart the way it did before this
// module existed.

export const WORKGROUP_SIZE = 64;

// Rounds up n/workgroupSize, used by every compute-shader demo to size its
// dispatchWorkgroups() calls.
export function dispatchCount(n, workgroupSize = WORKGROUP_SIZE) {
    return Math.max(1, Math.ceil(n / workgroupSize));
}

// Requests a WebGPU adapter/device and configures the canvas context.
// Shared so adapter/device error handling and canvas configuration behave
// identically across every demo.
export async function initWebGPU(canvas) {
    if (!navigator.gpu) {
        throw new Error('WebGPU is not supported by this browser.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('No appropriate GPUAdapter found.');
    }
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');
    if (!context) {
        throw new Error('Unable to create a WebGPU canvas context.');
    }
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat });
    return { device, context, canvasFormat };
}

async function loadShaderSource(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Unable to load shader ${url} (${response.status}).`);
    }
    return await response.text();
}

// Fetches and compiles a WGSL shader module, surfacing compilation errors
// (rather than silently producing a pipeline that fails later) when the
// device supports getCompilationInfo().
export async function loadShaderModule(device, url) {
    const code = await loadShaderSource(url);
    const module = device.createShaderModule({ code });
    if (module.getCompilationInfo) {
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter((message) => message.type === 'error');
        if (errors.length > 0) {
            throw new Error(`${url}: ${errors.map((message) => message.message).join('; ')}`);
        }
    }
    return module;
}

// Formats a time-scale multiplier (0.001x .. 1000x) with roughly 3
// significant figures, used by every log-scale Time Scale slider's label.
export function formatTimeScale(scale) {
    if (!(scale > 0)) return '0x';
    const digits = Math.max(0, Math.min(6, 2 - Math.floor(Math.log10(scale))));
    return `${scale.toFixed(digits)}x`;
}

// Computes the 6 vertices (position.xy in NDC, uv.xy) of a full-canvas quad
// that "cover-fits" a square (1:1) texture into a canvas of arbitrary
// aspect ratio: the quad always fills the entire viewport with no
// letterboxing, cropping whichever axis overhangs rather than stretching
// the content. Used by every 2D grid/PM heatmap demo.
export function computeCoverQuadVertices(canvasAspect) {
    let uvHalfW, uvHalfH;
    if (canvasAspect >= 1) {
        uvHalfW = 0.5;
        uvHalfH = 0.5 / canvasAspect;
    } else {
        uvHalfW = 0.5 * canvasAspect;
        uvHalfH = 0.5;
    }

    const u0 = 0.5 - uvHalfW, u1 = 0.5 + uvHalfW;
    const v0 = 0.5 - uvHalfH, v1 = 0.5 + uvHalfH;

    return new Float32Array([
        -1, -1, u0, v0,
         1, -1, u1, v0,
        -1,  1, u0, v1,
         1, -1, u1, v0,
         1,  1, u1, v1,
        -1,  1, u0, v1,
    ]);
}

// ---- 2D grid/PM shared domain + tuning constants ---------------------
// These were previously copy-pasted (identically) into all three 2D demo
// files; centralizing them here means they can no longer drift apart.

export const DOMAIN_HALF_SIZE = 50;
export const MASS_VISUAL_SCALE = 2.0;
// Fixed-point scale for atomic mass-scatter/deposit passes (WGSL has no
// atomic<f32>). Must be large enough for sub-integer mass precision but
// small enough that heavily-populated cells don't overflow i32.
export const MASS_FIXED_POINT_SCALE = 65536;
export const DEFAULT_DAMPING = 0.999;
export const DEFAULT_RESTITUTION = 1.0;
export const ALIVE_READBACK_INTERVAL_FRAMES = 20;

export const BOUNDARY_MODE_WRAP = 'wrap';
export const BOUNDARY_MODE_BOUNCE = 'bounce';
export const BOUNDARY_MODE_DELETE = 'delete';
export const DEFAULT_BOUNDARY_MODE = BOUNDARY_MODE_BOUNCE;
export const BOUNDARY_MODE_CODES = { bounce: 0, wrap: 1, delete: 2 };

// ---- 2D grid/PM shared structural + timing constants --------------------
// The mass-accumulation grid resolution (1024x1024) was previously
// copy-pasted identically into grid-cpu.js/grid-gpu.js (as the base LOD
// level) and pm-gpu.js (as its single FFT grid) — centralized so a future
// resolution change can't happen in only one of the three.
export const GRID_RESOLUTION = 1024; // must be a power of two
export const DOMAIN_SIZE = 2 * DOMAIN_HALF_SIZE;

// Fixed internal physics timestep, in simulation-time seconds. Was
// previously the literal 0.016 copy-pasted into every simulation file
// (2D and 3D). grid-cpu.js/grid-gpu.js multiply it directly by timeScale
// each frame; pm-gpu.js instead accumulates real time and takes 0+
// FIXED_DT-sized sub-steps per rendered frame (see its stepSimulation) so
// its stability never depends on timeScale.
export const FIXED_DT = 0.016;

// ---- 2D particle-count tiers ---------------------------------------------
// grid-cpu.js runs entirely on the CPU (typed arrays, JS loops) so it caps
// out much lower than the two GPU-compute algorithms, which share an
// identical, much higher cap/step/range.
export const MAX_CPU_GRID_PARTICLES = 1000000;
export const CPU_GRID_PARTICLE_COUNT_RANGE = { min: 100, max: MAX_CPU_GRID_PARTICLES, step: 100 };
export const DEFAULT_CPU_GRID_PARTICLE_COUNT = 250000;

export const MAX_GPU_PARTICLES = 4000000;
export const GPU_PARTICLE_COUNT_RANGE = { min: 0, max: MAX_GPU_PARTICLES, step: 100000 };
export const DEFAULT_GPU_PARTICLE_COUNT = 1000000;

// Wraps a coordinate into [-DOMAIN_HALF_SIZE, DOMAIN_HALF_SIZE) for
// periodic boundaries. Handles arbitrarily large/negative values (not just
// single overshoots) via modulo arithmetic.
export function wrapCoordinate(v) {
    const size = 2 * DOMAIN_HALF_SIZE;
    let wrapped = (v + DOMAIN_HALF_SIZE) % size;
    if (wrapped < 0) wrapped += size;
    return wrapped - DOMAIN_HALF_SIZE;
}

// ---- Shared 2D disc initial-condition helpers ---------------------------
// All three 2D algorithms (grid-cpu, grid-gpu, pm-gpu) seed particles in a
// random annulus/disc and give them a circular orbital velocity so the
// system starts in a roughly stable, rotating configuration instead of
// immediately collapsing under its own gravity. This speed ALWAYS follows
// an inverse-square-root law in radius (Math.sqrt of a quantity divided by
// radius) so nearer particles orbit faster than farther ones — the same
// "closer = faster" Keplerian shape already used by the 3D demo — and it
// deliberately does NOT use any algorithm's own force-integration
// FORCE_SCALE tuning constant, so switching between algorithms with the
// same Particle Count/Gravity Strength/Orbital Speed always starts at the
// same speed instead of drifting apart the way it used to (grid-cpu and
// grid-gpu previously each multiplied by their own 10x-different
// FORCE_SCALE, and both previously had a bug where the radius terms
// canceled out entirely, giving every particle the SAME speed regardless
// of radius).
export const DEFAULT_ORBITAL_SPEED = 1.0;
export const ORBITAL_SPEED_RANGE = { min: 0, max: 3, step: 0.05 };
// Baseline tuning constant (an assumed "central mass contribution per
// particle" times an empirical damping-down factor so the default Orbital
// Speed of 1.0 produces stable-looking orbits at the default Gravity
// Strength) — a heuristic initial condition, intentionally independent of
// any algorithm's own force-integration scale.
const ORBITAL_SPEED_TUNING = 0.0036;

// Returns the circular orbital speed (not a velocity vector) for a particle
// at the given radius, used directly by grid-cpu.js/grid-gpu.js. PM-GPU
// instead derives its circular speed from the actual GPU-computed
// acceleration field (see pm-init-circular-velocity.wgsl) but still scales
// the result by the same orbitalSpeed factor for a consistent control.
export function discOrbitalSpeed(radius, particleCount, gravityStrength, orbitalSpeed) {
    return orbitalSpeed * Math.sqrt(gravityStrength * ORBITAL_SPEED_TUNING * particleCount / radius);
}

// ---- Generic control-panel binding helpers -----------------------------
// Wires an <input type="range"> (or any input with a numeric .value) to a
// value-label element and a setter callback, so every demo's slider wiring
// updates its label and invokes its setter the same way.
export function bindRange(input, valueEl, { parse = parseFloat, format = (v) => String(v), onInput } = {}) {
    if (!input) return;
    const apply = (value) => {
        if (valueEl) valueEl.textContent = format(value);
        if (onInput) onInput(value);
    };
    input.addEventListener('input', (e) => apply(parse(e.target.value)));
}

export function bindSelect(select, onChange) {
    if (!select) return;
    select.addEventListener('change', (e) => onChange(e.target.value));
}

// Applies a {min,max,step} range object and a default value to a range
// <input> (and its paired value-label element), keeping the slider's
// bounds in sync with whichever model/algorithm is currently active. Call
// this any time the active simulation's own range constants should take
// over the control — initial page load, an algorithm/backend switch, etc.
// — instead of hand-setting .min/.max/.step/.value in each app file.
export function applySliderRange(input, valueEl, range, defaultValue, format = (v) => String(v)) {
    if (!input) return;
    input.min = range.min;
    input.max = range.max;
    input.step = range.step;
    input.value = defaultValue;
    if (valueEl) valueEl.textContent = format(defaultValue);
}
