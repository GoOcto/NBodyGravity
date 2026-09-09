// Converts the atomic fixed-point base mass grid back into a plain f32
// buffer (levels[0].rawMass) for the rest of the pipeline (downsample,
// blur, gradients, colorize).

struct SimParams {
    forceMultiplier: f32,
    damping: f32,
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
@group(0) @binding(2) var<storage, read_write> atomicMass: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> rawMass: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cellCount = level.dim * level.dim;
    if (idx >= cellCount) {
        return;
    }
    let fixedMass = atomicLoad(&atomicMass[idx]);
    rawMass[idx] = f32(fixedMass) / sim.massFixedPointScale;
}
