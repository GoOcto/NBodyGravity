struct OctreeNode {
    bounds: vec4<f32>,
    massCom: vec4<f32>,
}

struct LevelParams {
    level: u32,
    parentStart: u32,
    parentCount: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<OctreeNode>;
@group(0) @binding(1) var<uniform> levelParams: LevelParams;

@compute @workgroup_size(64)
fn aggregateLevel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let localParent = global_id.x;
    if (localParent >= levelParams.parentCount) {
        return;
    }

    let parentIndex = levelParams.parentStart + localParent;
    let childStart = parentIndex * 8u + 1u;
    var mass = 0.0;
    var weightedPosition = vec3<f32>(0.0);
    for (var child = 0u; child < 8u; child++) {
        let childMassCom = nodes[childStart + child].massCom;
        mass += childMassCom.w;
        weightedPosition += childMassCom.xyz * childMassCom.w;
    }

    var node = nodes[parentIndex];
    if (mass > 0.0) {
        node.massCom = vec4<f32>(weightedPosition / mass, mass);
    } else {
        node.massCom = vec4<f32>(0.0);
    }
    nodes[parentIndex] = node;
}
