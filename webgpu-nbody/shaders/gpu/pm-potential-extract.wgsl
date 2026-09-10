// Particle-Mesh (PM) engine: final step of the inverse 2D FFT. Extracts
// the real component of the (nominally real, up to floating-point
// round-off) inverse-transformed pseudo-potential and applies the
// 1/(dim*dim) normalization that a forward-then-inverse FFT pair
// requires (this shader is only run once, after both inverse-direction
// row-FFT + transpose passes, rather than splitting the 1/dim
// normalization across each pass). Writes into a plain f32 buffer so the
// unmodified gradient.wgsl can consume it exactly like the old blur
// pipeline's per-level "field" buffer.

struct ExtractParams {
    dim: u32,
    invCellCount: f32,   // 1 / (dim * dim)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ExtractParams;
@group(0) @binding(1) var<storage, read> data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cellCount = params.dim * params.dim;
    if (idx >= cellCount) {
        return;
    }
    field[idx] = data[idx].x * params.invCellCount;
}
