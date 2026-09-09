// Particle-to-grid mass scatter (2D-optimized.md Step 2.1), base grid level
// only. WGSL has no atomic<f32>, so mass is scaled into a fixed-point i32
// (see SimParams.massFixedPointScale) and atomically added; mass-resolve.wgsl
// converts the result back to a plain f32 buffer for the rest of the pipeline.

struct SimParams {
    forceMultiplier: f32,   // gravityStrength * FORCE_SCALE
    damping: f32,
    dt: f32,
    domainHalfSize: f32,

    restitution: f32,
    boundaryMode: u32,      // 0 = bounce, 1 = wrap, 2 = delete
    particleCount: u32,     // number of particle slots to consider this frame
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
    var cx = i32(floor((posX[i] + sim.domainHalfSize) / level.cellSize));
    var cy = i32(floor((posY[i] + sim.domainHalfSize) / level.cellSize));
    cx = clamp(cx, 0, dimI - 1);
    cy = clamp(cy, 0, dimI - 1);

    let cellIndex = u32(cy) * dim + u32(cx);
    let fixedMass = i32(round(massArr[i] * sim.massFixedPointScale));
    atomicAdd(&atomicMass[cellIndex], fixedMass);
}
