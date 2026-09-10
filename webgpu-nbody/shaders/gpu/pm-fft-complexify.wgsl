// Particle-Mesh (PM) engine: copies the resolved real-valued mass grid
// (levels[0].rawMass, produced by the shared mass-clear/pm-mass-scatter-cic/
// mass-resolve trio) into a complex (re, im) working buffer as the input to
// the forward 2D FFT (pm-fft-bitreverse.wgsl / pm-fft-butterfly.wgsl /
// pm-fft-transpose.wgsl).

struct FftDimParams {
    dim: u32,
    logDim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FftDimParams;
@group(0) @binding(1) var<storage, read> rawMass: array<f32>;
@group(0) @binding(2) var<storage, read_write> complexOut: array<vec2<f32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cellCount = params.dim * params.dim;
    if (idx >= cellCount) {
        return;
    }
    complexOut[idx] = vec2<f32>(rawMass[idx], 0.0);
}
