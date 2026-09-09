struct OctreeNode {
    bounds: vec4<f32>,
    massCom: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<OctreeNode>;

@compute @workgroup_size(64)
fn clearNodes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= 4681u) {
        return;
    }
    nodes[global_id.x].massCom = vec4<f32>(0.0);
}
