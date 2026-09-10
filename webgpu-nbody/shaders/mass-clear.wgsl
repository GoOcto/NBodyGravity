// Clears the atomic fixed-point mass accumulation buffer for one grid level
// to zero before this frame's particle-to-grid scatter (mass-scatter.wgsl).

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> level: LevelParams;
@group(0) @binding(1) var<storage, read_write> atomicMass: array<atomic<i32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cellCount = level.dim * level.dim;
    if (idx >= cellCount) {
        return;
    }
    atomicStore(&atomicMass[idx], 0);
}
