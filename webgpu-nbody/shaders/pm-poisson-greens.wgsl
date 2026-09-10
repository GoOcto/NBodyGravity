// Particle-Mesh (PM) engine: solves the 2D Poisson equation in the
// frequency domain. For a periodic domain, ∇²Φ = 2πGΣ becomes, per
// Fourier mode, Φ̂(k) = -2πG·Σ̂(k)/|k|² (with the k=0 term set to zero —
// the standard periodic-Poisson trick, since a net non-zero mean density
// has no periodic solution; it also removes the otherwise-undefined
// 1/0 term).
//
// This shader instead computes Ψ̂ = +ρ̂(k)/|k|² (no minus sign, and the
// 2πG / cell-area constants folded into the UI-tunable forceMultiplier
// applied later in pm-integrate.wgsl). Ψ = -Φ is a "mass-is-a-hill"
// pseudo-potential: unlike the real potential Φ (which is a *well* at
// mass concentrations), Ψ is a *hill*, so the reused, unmodified
// gradient.wgsl / accumulate-gradient.wgsl / pm-integrate.wgsl — which
// treat "field" as something particles should climb toward (force ∝
// +∇field, matching the old blur pipeline's convention where blurred
// mass was literally a hill) — still produce attractive gravity without
// needing a sign flip anywhere downstream.
//
// k is mapped to signed cycles-per-domain-length frequencies (standard
// FFT ordering: indices 0..dim/2 are non-negative frequencies,
// dim/2+1..dim-1 map to negative frequencies), scaled to physical units
// by 2π/domainSize so the force law doesn't implicitly depend on grid
// resolution.

struct GreensParams {
    dim: u32,
    domainSize: f32,   // 2 * DOMAIN_HALF_SIZE
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: GreensParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;

const TWO_PI: f32 = 6.28318530717958647692;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = params.dim;
    let cellCount = dim * dim;
    if (idx >= cellCount) {
        return;
    }

    let dimI = i32(dim);
    let x = i32(idx % dim);
    let y = idx / dim;

    var kx = x;
    if (kx > dimI / 2) {
        kx = kx - dimI;
    }
    var ky = i32(y);
    if (ky > dimI / 2) {
        ky = ky - dimI;
    }

    if (kx == 0 && ky == 0) {
        data[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let kxPhys = TWO_PI * f32(kx) / params.domainSize;
    let kyPhys = TWO_PI * f32(ky) / params.domainSize;
    let kSq = kxPhys * kxPhys + kyPhys * kyPhys;

    let factor = 1.0 / kSq;
    data[idx] = data[idx] * factor;
}
