// Force calculation compute shader
// Each thread processes one particle and calculates gravitational forces from all other particles

struct Particle {
    position: vec3<f32>,
    mass: f32,
    velocity: vec3<f32>,
    _padding: f32,
}

struct SimParams {
    particleCount: u32,
    deltaTime: f32,
    gravityStrength: f32,
    damping: f32,
}

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forcesOut: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.particleCount) {
        return;
    }
    
    let particle = particlesIn[index];
    var totalForce = vec3<f32>(0.0, 0.0, 0.0);
    
    // Calculate gravitational force from all other particles
    for (var i = 0u; i < params.particleCount; i++) {
        if (i == index) {
            continue;
        }
        
        let other = particlesIn[i];
        let r = other.position - particle.position;
        let distanceSquared = dot(r, r);
        
        // Add small epsilon to prevent singularity when particles are very close
        let epsilon = 0.01;
        let distance = sqrt(distanceSquared + epsilon * epsilon);
        
        // F = G * m1 * m2 / r^2, but we'll use simplified masses
        let forceMagnitude = params.gravityStrength * particle.mass * other.mass / distanceSquared;
        let forceDirection = normalize(r);
        
        totalForce += forceMagnitude * forceDirection;
    }
    
    forcesOut[index] = totalForce;
}