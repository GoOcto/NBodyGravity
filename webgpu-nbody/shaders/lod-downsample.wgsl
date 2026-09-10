// Builds one coarser LOD level (2D-optimized.md Step 2.2) by summing 2x2
// blocks of the previous (finer) level's raw mass grid. `level` describes
// the *current* (coarser, being-written) level; the previous level's
// dimension is always level.dim * 2.

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> level: LevelParams;
@group(0) @binding(1) var<storage, read> prevMass: array<f32>;
@group(0) @binding(2) var<storage, read_write> curMass: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = level.dim;
    let cellCount = dim * dim;
    if (idx >= cellCount) {
        return;
    }

    let x = idx % dim;
    let y = idx / dim;
    let prevDim = dim * 2u;
    let px0 = x * 2u;
    let py0 = y * 2u;
    let px1 = px0 + 1u;
    let py1 = py0 + 1u;

    let sum =
        prevMass[py0 * prevDim + px0] + prevMass[py0 * prevDim + px1] +
        prevMass[py1 * prevDim + px0] + prevMass[py1 * prevDim + px1];

    curMass[idx] = sum;
}
