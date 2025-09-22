// Particle integration compute shader
// Updates particle positions and velocities based on calculated forces

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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> forces: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn integrateParticles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.particleCount) {
        return;
    }
    
    var particle = particles[index];
    let force = forces[index];
    
    // F = ma, so a = F/m
    let acceleration = force / particle.mass;
    
    // Verlet integration for better stability
    // v(t+dt) = v(t) + a(t) * dt
    // x(t+dt) = x(t) + v(t+dt) * dt
    
    particle.velocity += acceleration * params.deltaTime;
    particle.velocity *= params.damping; // Apply damping to prevent runaway velocities
    particle.position += particle.velocity * params.deltaTime;
    
    // Simple boundary conditions - bounce off edges
    let boundarySize = 50.0;
    let restitution = 0.8;
    
    if (abs(particle.position.x) > boundarySize) {
        particle.position.x = sign(particle.position.x) * boundarySize;
        particle.velocity.x *= -restitution;
    }
    if (abs(particle.position.y) > boundarySize) {
        particle.position.y = sign(particle.position.y) * boundarySize;
        particle.velocity.y *= -restitution;
    }
    if (abs(particle.position.z) > boundarySize) {
        particle.position.z = sign(particle.position.z) * boundarySize;
        particle.velocity.z *= -restitution;
    }
    
    particles[index] = particle;
}