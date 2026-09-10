// Computes the gradient (2D-optimized.md Step 2.4) of a blurred mass field
// via central differences. When level.wrap is set, neighbor lookups wrap
// around the domain edge (periodic boundaries); otherwise edge cells are
// clamped (replicated), matching a hard-box domain.

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> level: LevelParams;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradX: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradY: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = level.dim;
    let cellCount = dim * dim;
    if (idx >= cellCount) {
        return;
    }

    let dimI = i32(dim);
    let x = i32(idx % dim);
    let y = i32(idx / dim);
    let wrap = level.wrap != 0u;

    var xLeft: i32;
    var xRight: i32;
    var yUp: i32;
    var yDown: i32;

    if (wrap) {
        xLeft = ((x - 1) % dimI + dimI) % dimI;
        xRight = (x + 1) % dimI;
        yUp = ((y - 1) % dimI + dimI) % dimI;
        yDown = (y + 1) % dimI;
    } else {
        xLeft = clamp(x - 1, 0, dimI - 1);
        xRight = clamp(x + 1, 0, dimI - 1);
        yUp = clamp(y - 1, 0, dimI - 1);
        yDown = clamp(y + 1, 0, dimI - 1);
    }

    let invTwoCell = 1.0 / (2.0 * level.cellSize);
    let row = u32(y) * dim;
    let rowUp = u32(yUp) * dim;
    let rowDown = u32(yDown) * dim;

    gradX[row + u32(x)] = (field[row + u32(xRight)] - field[row + u32(xLeft)]) * invTwoCell;
    gradY[row + u32(x)] = (field[rowDown + u32(x)] - field[rowUp + u32(x)]) * invTwoCell;
}
