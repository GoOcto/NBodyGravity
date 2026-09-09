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
@group(0) @binding(2) var<storage, read> nodes: array<OctreeNode>;
@group(0) @binding(3) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

const LEVEL_OFFSETS: array<u32, 5> = array<u32, 5>(0u, 1u, 9u, 73u, 585u);

fn nodeLevel(nodeIndex: u32) -> u32 {
    if (nodeIndex >= 585u) { return 4u; }
    if (nodeIndex >= 73u) { return 3u; }
    if (nodeIndex >= 9u) { return 2u; }
    if (nodeIndex >= 1u) { return 1u; }
    return 0u;
}

fn parentIndex(nodeIndex: u32, level: u32) -> u32 {
    let local = nodeIndex - LEVEL_OFFSETS[level];
    return LEVEL_OFFSETS[level - 1u] + local / 8u;
}

fn ancestorAccepted(nodeIndex: u32, level: u32, position: vec3<f32>) -> bool {
    var current = nodeIndex;
    var currentLevel = level;
    loop {
        if (currentLevel == 0u) {
            break;
        }
        current = parentIndex(current, currentLevel);
        currentLevel -= 1u;
        let ancestor = nodes[current];
        if (ancestor.massCom.w <= 0.0) {
            continue;
        }
        let offset = ancestor.massCom.xyz - position;
        let distance = sqrt(dot(offset, offset) + params.softening * params.softening);
        if (ancestor.bounds.w / distance < params.theta) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn computeFmmForces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.particleCount) {
        return;
    }

    let particle = particles[index];
    var totalForce = vec3<f32>(0.0);
    for (var nodeIndex = 0u; nodeIndex < 4681u; nodeIndex++) {
        let node = nodes[nodeIndex];
        if (node.massCom.w <= 0.0 || ancestorAccepted(nodeIndex, nodeLevel(nodeIndex), particle.position)) {
            continue;
        }

        let offset = node.massCom.xyz - particle.position;
        let distanceSquared = dot(offset, offset);
        let distance = sqrt(distanceSquared + params.softening * params.softening);
        let level = nodeLevel(nodeIndex);
        if (level == 4u) {
            for (var otherIndex = 0u; otherIndex < params.particleCount; otherIndex++) {
                if (otherIndex == index || particleLeaves[otherIndex] != nodeIndex - 585u) {
                    continue;
                }
                let other = particles[otherIndex];
                let directOffset = other.position - particle.position;
                let directDistanceSquared = dot(directOffset, directOffset) + params.softening * params.softening;
                let directDistance = sqrt(directDistanceSquared);
                totalForce += params.gravityStrength * particle.mass * other.mass
                    * directOffset / (directDistanceSquared * directDistance);
            }
        } else if (node.bounds.w / distance < params.theta
            && any(abs(particle.position - node.bounds.xyz) > vec3<f32>(node.bounds.w))) {
            totalForce += params.gravityStrength * particle.mass * node.massCom.w
                * offset / (distance * distance * distance);
        }
    }
    forces[index] = vec4<f32>(totalForce, 0.0);
}
