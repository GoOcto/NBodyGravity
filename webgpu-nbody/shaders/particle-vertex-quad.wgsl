// Vertex shader for particle rendering using instanced quads
// Each particle becomes a small quad instead of a point

struct Particle {
    position: vec3<f32>,
    mass: f32,
    velocity: vec3<f32>,
    _padding: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct Uniforms {
    viewProjectionMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    cameraPosition: vec3<f32>,
    time: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

// Quad vertices in local space
var<private> QUAD_VERTICES: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), // Bottom left
    vec2<f32>( 1.0, -1.0), // Bottom right
    vec2<f32>(-1.0,  1.0), // Top left
    vec2<f32>( 1.0, -1.0), // Bottom right
    vec2<f32>( 1.0,  1.0), // Top right
    vec2<f32>(-1.0,  1.0)  // Top left
);

var<private> QUAD_UVS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), // Bottom left
    vec2<f32>(1.0, 0.0), // Bottom right
    vec2<f32>(0.0, 1.0), // Top left
    vec2<f32>(1.0, 0.0), // Bottom right
    vec2<f32>(1.0, 1.0), // Top right
    vec2<f32>(0.0, 1.0)  // Top left
);

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let particleIndex = vertexIndex / 6u;
    let quadVertexIndex = vertexIndex % 6u;
    
    let particle = particles[particleIndex];
    let quadVertex = QUAD_VERTICES[quadVertexIndex];
    let uv = QUAD_UVS[quadVertexIndex];
    
    // Calculate particle size based on mass
    let baseSize = 0.5; // Base size in world units
    let size = baseSize * particle.mass;
    
    // Create billboard quad that always faces camera
    // Extract right and up vectors from the view matrix (camera space axes)
    let rightVector = vec3<f32>(uniforms.viewMatrix[0][0], uniforms.viewMatrix[1][0], uniforms.viewMatrix[2][0]);
    let upVector = vec3<f32>(uniforms.viewMatrix[0][1], uniforms.viewMatrix[1][1], uniforms.viewMatrix[2][1]);
    
    // Create the billboard quad using camera-aligned axes
    let offset = rightVector * (quadVertex.x * size) + upVector * (quadVertex.y * size);
    let worldPos = particle.position + offset;
    
    var output: VertexOutput;
    output.position = uniforms.viewProjectionMatrix * vec4<f32>(worldPos, 1.0);
    
    // Color particles based on velocity magnitude
    let speed = length(particle.velocity);
    let normalizedSpeed = clamp(speed / 10.0, 0.0, 1.0);
    
    // Color gradient from blue (slow) to red (fast)
    output.color = vec3<f32>(
        normalizedSpeed,                      // Red component
        0.3 + 0.7 * (1.0 - normalizedSpeed), // Green component  
        1.0 - normalizedSpeed                 // Blue component
    );
    
    output.uv = uv;
    
    return output;
}