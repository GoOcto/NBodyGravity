// Particle-Mesh (PM) engine: complex dim x dim matrix transpose. Used to
// turn the "row FFT" primitive (pm-fft-bitreverse.wgsl + repeated
// pm-fft-butterfly.wgsl dispatches) into a full 2D FFT: rowFFT, transpose,
// rowFFT, transpose. Always reads from one fixed buffer and writes to the
// other (see main-pm-gpu.js's fixed buffer-role comments), so the same
// bind group is reused for every transpose call in the pipeline.

struct FftDimParams {
    dim: u32,
    logDim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FftDimParams;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = params.dim;
    let cellCount = dim * dim;
    if (idx >= cellCount) {
        return;
    }

    let y = idx / dim;
    let x = idx % dim;
    dst[x * dim + y] = src[idx];
}
