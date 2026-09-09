struct Particle {
    position: vec3<f32>,
    mass: f32,
    velocity: vec3<f32>,
    _padding: f32,
}

struct SimParams {
    particleCount: u32,
    nodeCount: u32,
    deltaTime: f32,
    gravityStrength: f32,
    damping: f32,
    theta: f32,
    softening: f32,
    maxDepth: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particleLeaves: array<u32>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn assignLeaves(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.particleCount) {
        return;
    }

    let position = clamp(particles[index].position, vec3<f32>(-49.999), vec3<f32>(49.999));
    let cellSize = 100.0 / 16.0;
    let cell = clamp(vec3<u32>(floor((position + vec3<f32>(50.0)) / cellSize)), vec3<u32>(0u), vec3<u32>(15u));
    var leaf = 0u;
    for (var level = 0u; level < 4u; level++) {
        let octant = ((cell.x >> (3u - level)) & 1u)
            | (((cell.y >> (3u - level)) & 1u) << 1u)
            | (((cell.z >> (3u - level)) & 1u) << 2u);
        leaf = (leaf << 3u) | octant;
    }
    particleLeaves[index] = leaf;
}
