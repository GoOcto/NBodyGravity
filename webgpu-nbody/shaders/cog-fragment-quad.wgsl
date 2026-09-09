// Fragment shader for the Center of Gravity (COG) marker
// Draws a white empty (outlined) circle with a white plus sign in the middle

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let p = input.uv - center;
    let distance = length(p);

    // Ring (empty circle outline)
    let ringOuter = 0.5;
    let ringInner = 0.4;
    let ringAlpha = smoothstep(ringOuter, ringOuter - 0.03, distance) * smoothstep(ringInner - 0.03, ringInner, distance);

    // Plus sign through the middle
    let armHalfLength = 0.35;
    let armHalfThickness = 0.035;
    let horizontalArm = step(abs(p.y), armHalfThickness) * step(abs(p.x), armHalfLength);
    let verticalArm = step(abs(p.x), armHalfThickness) * step(abs(p.y), armHalfLength);
    let crossAlpha = max(horizontalArm, verticalArm);

    let alpha = max(ringAlpha, crossAlpha);

    let white = vec3<f32>(1.0, 1.0, 1.0);

    return vec4<f32>(white, alpha);
}
