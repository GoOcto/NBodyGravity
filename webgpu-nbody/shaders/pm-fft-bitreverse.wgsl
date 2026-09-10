// Particle-Mesh (PM) engine: per-row bit-reversal permutation, the first
// step of an iterative (non-recursive) Cooley-Tukey FFT. Operates on a
// dim x dim complex buffer, permuting each row of length `dim`
// independently — used both for the "along x" pass and, after
// pm-fft-transpose.wgsl swaps the axes, for the "along y" pass. The same
// permutation is used for both forward and inverse transforms (bit
// reversal doesn't depend on transform direction).
//
// Algorithm validated against a naive O(N^2) DFT and round-tripped through
// forward+inverse before being transcribed to WGSL (see session notes).

struct FftDimParams {
    dim: u32,
    logDim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FftDimParams;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

fn bitReverse(value: u32, bitCount: u32) -> u32 {
    var r: u32 = 0u;
    var v: u32 = value;
    for (var b: u32 = 0u; b < bitCount; b = b + 1u) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = params.dim;
    let cellCount = dim * dim;
    if (idx >= cellCount) {
        return;
    }

    let row = idx / dim;
    let col = idx % dim;
    let reversedCol = bitReverse(col, params.logDim);

    dst[row * dim + reversedCol] = src[idx];
}
