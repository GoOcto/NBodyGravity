// Separable blur (2D-optimized.md Step 2.3), horizontal pass. Spreads mass
// outward using a fixed 7-tap kernel approximating gravitational falloff
// (1/(1+|r|)), matching buildBlurKernel(3) in main-optimized-gpu.js.
// When level.wrap is set (periodic boundaries), edge taps wrap around
// instead of clamping to the edge cell.

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> level: LevelParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

const RADIUS: i32 = 3;
// Sum of 1/(1+|k|) for k in [-3, 3]: matches buildBlurKernel(3)'s normalization.
const KERNEL_SUM: f32 = 3.16666667;

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
    let row = u32(y) * dim;
    let wrap = level.wrap != 0u;

    var acc: f32 = 0.0;
    for (var k = -RADIUS; k <= RADIUS; k = k + 1) {
        var sx = x + k;
        if (wrap) {
            sx = ((sx % dimI) + dimI) % dimI;
        } else {
            sx = clamp(sx, 0, dimI - 1);
        }
        let weight = (1.0 / (1.0 + f32(abs(k)))) / KERNEL_SUM;
        acc = acc + src[row + u32(sx)] * weight;
    }

    dst[row + u32(x)] = acc;
}
