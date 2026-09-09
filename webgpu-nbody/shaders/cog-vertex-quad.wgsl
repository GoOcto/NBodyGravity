// Vertex shader for rendering the Center of Gravity (COG) marker
// Billboards a single quad at the COG world-space position

struct COGData {
    position: vec3<f32>,
    size: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Uniforms {
    viewProjectionMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    cameraPosition: vec3<f32>,
    time: f32,
}

@group(0) @binding(0) var<storage, read> cog: COGData;
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
    let quadVertex = QUAD_VERTICES[vertexIndex];
    let uv = QUAD_UVS[vertexIndex];

    let size = cog.size;

    // Create billboard quad that always faces camera
    let rightVector = vec3<f32>(uniforms.viewMatrix[0][0], uniforms.viewMatrix[1][0], uniforms.viewMatrix[2][0]);
    let upVector = vec3<f32>(uniforms.viewMatrix[0][1], uniforms.viewMatrix[1][1], uniforms.viewMatrix[2][1]);

    let offset = rightVector * (quadVertex.x * size) + upVector * (quadVertex.y * size);
    let worldPos = cog.position + offset;

    var output: VertexOutput;
    output.position = uniforms.viewProjectionMatrix * vec4<f32>(worldPos, 1.0);
    output.uv = uv;

    return output;
}
