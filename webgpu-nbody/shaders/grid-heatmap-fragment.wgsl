// Fragment shader for the grid mass-accumulation heatmap.
// The texture already contains the final black -> dark blue -> red ->
// orange -> white colors (computed on the CPU from the base mass grid),
// so this simply samples it.

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var gridSampler: sampler;
@group(0) @binding(1) var gridTexture: texture_2d<f32>;

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    return textureSample(gridTexture, gridSampler, input.uv);
}
