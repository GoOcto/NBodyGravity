// Vertex shader for the grid mass-accumulation heatmap.
// Draws a full-canvas quad directly in clip space (no camera/projection —
// there is no 3D content, just a flat color-mapped texture). The quad's
// vertex positions/UVs are precomputed on the CPU each frame
// (computeCoverQuadVertices in main-optimized.js) to "cover-fit" the square
// grid texture into the canvas: the texture always fills the entire
// viewport, cropping whichever axis overhangs rather than stretching it.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

