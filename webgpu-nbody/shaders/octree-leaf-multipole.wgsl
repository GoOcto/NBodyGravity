struct Particle {
    position: vec3<f32>,
    mass: f32,
    velocity: vec3<f32>,
    _padding: f32,
}

struct OctreeNode {
    bounds: vec4<f32>,
    massCom: vec4<f32>,
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
@group(0) @binding(1) var<storage, read> particleLeaves: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodes: array<OctreeNode>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn buildLeafMultipoles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let leaf = global_id.x;
    if (leaf >= 4096u) {
        return;
    }

    var mass = 0.0;
    var weightedPosition = vec3<f32>(0.0);
    for (var particleIndex = 0u; particleIndex < params.particleCount; particleIndex++) {
        if (particleLeaves[particleIndex] == leaf) {
            let particle = particles[particleIndex];
            mass += particle.mass;
            weightedPosition += particle.position * particle.mass;
        }
    }

    let nodeIndex = 585u + leaf;
    var node = nodes[nodeIndex];
    if (mass > 0.0) {
        node.massCom = vec4<f32>(weightedPosition / mass, mass);
    } else {
        node.massCom = vec4<f32>(0.0);
    }
    nodes[nodeIndex] = node;
}
