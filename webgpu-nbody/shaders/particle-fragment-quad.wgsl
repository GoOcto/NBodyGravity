// Fragment shader for quad-based particles
// Creates smooth circular particles

struct FragmentInput {
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Create circular particle
    let center = vec2<f32>(0.5, 0.5);
    let distance = length(input.uv - center);
    
    // Smooth circular falloff
    let alpha = smoothstep(0.5, 0.3, distance);
    
    // Make sure particles are bright and visible
    let brightness = 2.0;
    let finalColor = input.color * brightness;
    
    return vec4<f32>(finalColor, alpha);
}