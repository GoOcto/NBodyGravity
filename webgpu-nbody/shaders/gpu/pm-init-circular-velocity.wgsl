// Particle-Mesh (PM) engine: one-time initialization helper. Rather than
// an empirically-derived circular-velocity formula (which would need to
// guess at the disc's enclosed-mass profile), this shader computes the
// exact tangential velocity needed for a circular orbit from the ACTUAL
// simulated acceleration field — main-pm-gpu.js's initializeParticles()
// uploads particle positions with zero velocity, runs one field-solve
// pass (CIC deposit -> FFT Poisson solve -> gradient) to populate
// accelX/accelY for that initial distribution, then dispatches this
// shader once to convert "radially-inward acceleration" into "circular
// tangential velocity" (v = sqrt(r * |accel|), rotated 90° from the
// radial direction) directly on the GPU, with no CPU readback roundtrip.

struct SimParams {
    forceMultiplier: f32,
    damping: f32,           // unused (see pm-integrate.wgsl)
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

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<storage, read> posX: array<f32>;
@group(0) @binding(2) var<storage, read> posY: array<f32>;
@group(0) @binding(3) var<storage, read> accelX: array<f32>;
@group(0) @binding(4) var<storage, read> accelY: array<f32>;
@group(0) @binding(5) var<storage, read> alive: array<u32>;
@group(0) @binding(6) var<storage, read_write> velX: array<f32>;
@group(0) @binding(7) var<storage, read_write> velY: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.particleCount) {
        return;
    }
    if (alive[i] == 0u) {
        velX[i] = 0.0;
        velY[i] = 0.0;
        return;
    }

    let x = posX[i];
    let y = posY[i];
    let r = sqrt(x * x + y * y);
    if (r < 1e-5) {
        velX[i] = 0.0;
        velY[i] = 0.0;
        return;
    }

    let ax = accelX[i] * sim.forceMultiplier;
    let ay = accelY[i] * sim.forceMultiplier;
    let accelMag = sqrt(ax * ax + ay * ay);
    let speed = sqrt(r * accelMag);

    // Tangent direction (-y/r, x/r) is the same counter-clockwise
    // convention as the disc's angular sampling.
    velX[i] = -speed * (y / r);
    velY[i] = speed * (x / r);
}
