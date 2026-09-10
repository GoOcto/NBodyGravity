// Particle-Mesh (PM) engine: Cloud-in-Cell (CIC) mass deposit. Unlike
// mass-scatter.wgsl's nearest-grid-point (NGP) deposit (floor to a single
// cell), CIC bilinearly spreads each particle's mass across the 4 cells
// whose centers surround it — the same cell-center convention
// accumulate-gradient.wgsl uses to sample the force field back, so
// deposit and force-sampling use *matching* interpolation (standard PM
// practice, avoids self-force/grid-heating artifacts). WGSL has no
// atomic<f32>, so mass is still scaled into fixed-point i32 and
// atomically added (mass-resolve.wgsl converts it back to f32).
//
// The FFT Poisson solve (pm-poisson-greens.wgsl) is inherently periodic,
// so deposition always wraps at the domain edge — independent of
// sim.boundaryMode, which only governs how particle position/velocity
// are handled during integration (pm-integrate.wgsl).

struct SimParams {
    forceMultiplier: f32,
    damping: f32,           // unused by the PM engine; see pm-integrate.wgsl
    dt: f32,
    domainHalfSize: f32,

    restitution: f32,
    boundaryMode: u32,
    particleCount: u32,
    massFixedPointScale: f32,

    massVisualScale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<uniform> level: LevelParams;
@group(0) @binding(2) var<storage, read> posX: array<f32>;
@group(0) @binding(3) var<storage, read> posY: array<f32>;
@group(0) @binding(4) var<storage, read> massArr: array<f32>;
@group(0) @binding(5) var<storage, read> alive: array<u32>;
@group(0) @binding(6) var<storage, read_write> atomicMass: array<atomic<i32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.particleCount) {
        return;
    }
    if (alive[i] == 0u) {
        return;
    }

    let dim = level.dim;
    let dimI = i32(dim);

    // Same (grid-coordinate - 0.5) cell-center convention as
    // accumulate-gradient.wgsl's bilinear force sampling.
    let gx = (posX[i] + sim.domainHalfSize) / level.cellSize - 0.5;
    let gy = (posY[i] + sim.domainHalfSize) / level.cellSize - 0.5;

    var ix0 = i32(floor(gx));
    var iy0 = i32(floor(gy));
    let fx = gx - f32(ix0);
    let fy = gy - f32(iy0);

    ix0 = ((ix0 % dimI) + dimI) % dimI;
    iy0 = ((iy0 % dimI) + dimI) % dimI;
    let ix1 = (ix0 + 1) % dimI;
    let iy1 = (iy0 + 1) % dimI;

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let m = massArr[i] * sim.massFixedPointScale;

    atomicAdd(&atomicMass[u32(iy0) * dim + u32(ix0)], i32(round(m * w00)));
    atomicAdd(&atomicMass[u32(iy0) * dim + u32(ix1)], i32(round(m * w10)));
    atomicAdd(&atomicMass[u32(iy1) * dim + u32(ix0)], i32(round(m * w01)));
    atomicAdd(&atomicMass[u32(iy1) * dim + u32(ix1)], i32(round(m * w11)));
}
