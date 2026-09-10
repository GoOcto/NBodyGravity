// Particle-Mesh (PM) engine: final integration step, symplectic Euler,
// with NO velocity damping (unlike integrate.wgsl's `* sim.damping`).
// `damping` is still declared in the SimParams struct below purely so its
// byte layout matches the shared SimParams uniform buffer written once by
// main-pm-gpu.js and also consumed as-is by reused shaders
// (mass-resolve.wgsl, accumulate-gradient.wgsl, colorize.wgsl) — it is
// never read here.
//
// Boundary handling mirrors integrate.wgsl's sim.boundaryMode:
//   0 = bounce   - clamp to the edge, invert the offending velocity axis.
//   1 = wrap     - periodic: wrap the position to the opposite edge.
//   2 = delete   - cull: mark the particle dead and decrement aliveCounter.
// Note this is independent of the FFT Poisson solve's own (always
// periodic) field boundary condition — see pm-mass-scatter-cic.wgsl.

struct SimParams {
    forceMultiplier: f32,
    damping: f32,           // unused (see comment above)
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

const BOUNDARY_BOUNCE: u32 = 0u;
const BOUNDARY_WRAP: u32 = 1u;
const BOUNDARY_DELETE: u32 = 2u;

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read_write> velX: array<f32>;
@group(0) @binding(4) var<storage, read_write> velY: array<f32>;
@group(0) @binding(5) var<storage, read> accelX: array<f32>;
@group(0) @binding(6) var<storage, read> accelY: array<f32>;
@group(0) @binding(7) var<storage, read_write> alive: array<u32>;
@group(0) @binding(8) var<storage, read_write> aliveCounter: array<atomic<i32>>;

fn wrapCoord(v: f32, half: f32) -> f32 {
    let size = 2.0 * half;
    var w = (v + half) % size;
    if (w < 0.0) {
        w = w + size;
    }
    return w - half;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.particleCount) {
        return;
    }
    if (alive[i] == 0u) {
        return;
    }

    let px = posX[i];
    let py = posY[i];

    let accX = accelX[i] * sim.forceMultiplier;
    let accY = accelY[i] * sim.forceMultiplier;

    var vx = velX[i] + accX * sim.dt;
    var vy = velY[i] + accY * sim.dt;

    var nx = px + vx * sim.dt;
    var ny = py + vy * sim.dt;

    let half = sim.domainHalfSize;

    if (sim.boundaryMode == BOUNDARY_WRAP) {
        nx = wrapCoord(nx, half);
        ny = wrapCoord(ny, half);
    } else if (sim.boundaryMode == BOUNDARY_DELETE) {
        if (abs(nx) > half || abs(ny) > half) {
            alive[i] = 0u;
            atomicSub(&aliveCounter[0], 1);
            return;
        }
    } else {
        if (abs(nx) > half) {
            nx = sign(nx) * half;
            vx = vx * -sim.restitution;
        }
        if (abs(ny) > half) {
            ny = sign(ny) * half;
            vy = vy * -sim.restitution;
        }
    }

    velX[i] = vx;
    velY[i] = vy;
    posX[i] = nx;
    posY[i] = ny;
}
