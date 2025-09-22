// Vertex shader for particle rendering
// Transforms particle positions from world space to screen space

struct Particle {
    position: vec3<f32>,
    mass: f32,
    velocity: vec3<f32>,
    _padding: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) size: f32,
}

struct Uniforms {
    viewProjectionMatrix: mat4x4<f32>,
    cameraPosition: vec3<f32>,
    time: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let particle = particles[vertexIndex];
    
    var output: VertexOutput;
    output.position = uniforms.viewProjectionMatrix * vec4<f32>(particle.position, 1.0);
    
    // Color particles based on velocity magnitude for visual appeal
    let speed = length(particle.velocity);
    let normalizedSpeed = clamp(speed / 10.0, 0.0, 1.0);
    
    // Color gradient from blue (slow) to red (fast)
    output.color = vec3<f32>(
        normalizedSpeed,                    // Red component
        0.3 + 0.7 * (1.0 - normalizedSpeed), // Green component  
        1.0 - normalizedSpeed               // Blue component
    );
    
    // Size based on mass and distance from camera
    let distance = length(particle.position - uniforms.cameraPosition);
    output.size = clamp(particle.mass * 100.0 / (distance * 0.1), 2.0, 20.0);
    
    return output;
}