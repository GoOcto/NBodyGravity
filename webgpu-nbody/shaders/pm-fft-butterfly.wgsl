// Particle-Mesh (PM) engine: one Cooley-Tukey butterfly stage of an
// iterative, in-place, per-row complex FFT. Dispatched once per stage
// (log2(dim) times) after pm-fft-bitreverse.wgsl has permuted the input;
// each dispatch handles every row's stage-`len` butterflies in parallel.
// Because every thread reads/writes a disjoint (idx0, idx1) pair, this is
// safe to do in-place within a single dispatch (no cross-thread hazard).
//
// `sign` selects the transform direction: -1.0 for forward, +1.0 for
// inverse (the 1/(dim*dim) inverse normalization is applied later, in
// pm-potential-extract.wgsl, once after both inverse-direction row-FFT
// passes).

struct FftStageParams {
    dim: u32,
    half: u32,   // len / 2 for this stage
    len: u32,    // 2^stage
    sign: f32,   // -1.0 forward, +1.0 inverse
}

@group(0) @binding(0) var<uniform> params: FftStageParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = params.dim;
    let pairsPerRow = dim / 2u;
    let totalPairs = dim * pairsPerRow;
    if (idx >= totalPairs) {
        return;
    }

    let row = idx / pairsPerRow;
    let i = idx % pairsPerRow;

    let half = params.half;
    let len = params.len;
    let groupIndex = i / half;
    let indexInGroup = i % half;
    let start = groupIndex * len;

    let base = row * dim;
    let idx0 = base + start + indexInGroup;
    let idx1 = idx0 + half;

    let angle = params.sign * 2.0 * PI * f32(indexInGroup) / f32(len);
    let wr = cos(angle);
    let wi = sin(angle);

    let u = data[idx0];
    let v = data[idx1];

    let tr = wr * v.x - wi * v.y;
    let ti = wr * v.y + wi * v.x;

    data[idx0] = vec2<f32>(u.x + tr, u.y + ti);
    data[idx1] = vec2<f32>(u.x - tr, u.y - ti);
}
