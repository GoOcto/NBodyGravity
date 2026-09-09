// Colorizes the raw (unblurred) base mass grid directly into the heatmap
// storage texture on the GPU — no CPU readback/upload roundtrip. The color
// ramp (black -> dark blue -> red -> orange -> white) and the exponential
// saturating brightness curve must match buildColorLUT()/MASS_VISUAL_SCALE
// in main-optimized-gpu.js.

struct SimParams {
    forceMultiplier: f32,
    damping: f32,
    dt: f32,
    domainHalfSize: f32,

    restitution: f32,
    boundaryMode: u32,
    particleCount: u32,
    massFixedPointScale: f32,

    massVisualScale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<uniform> level: LevelParams;
@group(0) @binding(2) var<storage, read> rawMass: array<f32>;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

fn colorRamp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(  0.0,   0.0,   0.0);
    let c1 = vec3<f32>( 20.0,  20.0, 100.0) / 255.0;
    let c2 = vec3<f32>(150.0,  20.0,  20.0) / 255.0;
    let c3 = vec3<f32>(220.0, 150.0,   0.0) / 255.0;
    let c4 = vec3<f32>(255.0, 255.0, 255.0) / 255.0;

    if (t < 0.3) {
        return mix(c0, c1, t / 0.25);
    } else if (t < 0.6) {
        return mix(c1, c2, (t - 0.25) / 0.25);
    } else if (t < 0.9) {
        return mix(c2, c3, (t - 0.5) / 0.25);
    } else {
        return mix(c3, c4, (t - 0.75) / 0.25);
    }
}

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

    let mass = rawMass[idx];
    let t = clamp(1.0 - exp(-mass / sim.massVisualScale), 0.0, 1.0);
    let color = colorRamp(t);

    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}
