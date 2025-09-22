// Fragment shader for particle rendering
// Creates smooth circular particles with falloff

struct FragmentInput {
    @location(0) color: vec3<f32>,
    @location(1) size: f32,
}

@fragment
fn fs_main(input: FragmentInput, @builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // Debug: Make particles very bright and visible
    let debugColor = vec3<f32>(1.0, 1.0, 1.0); // Pure white for visibility
    
    // Make particles fully opaque and bright
    return vec4<f32>(debugColor, 1.0);
}