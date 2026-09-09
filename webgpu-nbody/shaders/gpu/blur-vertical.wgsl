// Separable blur (2D-optimized.md Step 2.3), vertical pass. Reads the
// horizontal pass's output (blurTemp) and writes the final blurred field
// for this level.

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
    let wrap = level.wrap != 0u;

    var acc: f32 = 0.0;
    for (var k = -RADIUS; k <= RADIUS; k = k + 1) {
        var sy = y + k;
        if (wrap) {
            sy = ((sy % dimI) + dimI) % dimI;
        } else {
            sy = clamp(sy, 0, dimI - 1);
        }
        let weight = (1.0 / (1.0 + f32(abs(k)))) / KERNEL_SUM;
        acc = acc + src[u32(sy) * dim + u32(x)] * weight;
    }

    dst[u32(y) * dim + u32(x)] = acc;
}
