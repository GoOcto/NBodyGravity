// Multi-resolution force sampling (2D-optimized.md Step 2.4), one LOD level
// per dispatch. Each particle bilinearly samples this level's gradient
// field and adds the contribution into the running accelX/accelY totals
// (levels[0]'s dispatch, isFirstLevel=1, initializes them instead of
// adding). integrate.wgsl later scales the summed gradient by
// gravityStrength * FORCE_SCALE and performs the actual integration.

struct SimParams {
    forceMultiplier: f32,
    damping: f32,
    dt: f32,
    domainHalfSize: f32,

    restitution: f32,
    boundaryMode: u32,
    particleCount: u32,
    massFixedPointScale: f32,

    massVisualScale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct LevelParams {
    dim: u32,
    cellSize: f32,
    wrap: u32,
    isFirstLevel: u32,
}

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<uniform> level: LevelParams;
@group(0) @binding(2) var<storage, read> posX: array<f32>;
@group(0) @binding(3) var<storage, read> posY: array<f32>;
@group(0) @binding(4) var<storage, read> alive: array<u32>;
@group(0) @binding(5) var<storage, read> gradX: array<f32>;
@group(0) @binding(6) var<storage, read> gradY: array<f32>;
@group(0) @binding(7) var<storage, read_write> accelX: array<f32>;
@group(0) @binding(8) var<storage, read_write> accelY: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.particleCount) {
        return;
    }

    if (alive[i] == 0u) {
        if (level.isFirstLevel != 0u) {
            accelX[i] = 0.0;
            accelY[i] = 0.0;
        }
        return;
    }

    let dim = level.dim;
    let dimI = i32(dim);
    let dimF = f32(dim);
    let wrap = level.wrap != 0u;

    var gx = (posX[i] + sim.domainHalfSize) / level.cellSize - 0.5;
    var gy = (posY[i] + sim.domainHalfSize) / level.cellSize - 0.5;

    var ix0: i32;
    var iy0: i32;
    var ix1: i32;
    var iy1: i32;
    var fx: f32;
    var fy: f32;

    if (wrap) {
        gx = ((gx % dimF) + dimF) % dimF;
        gy = ((gy % dimF) + dimF) % dimF;
        ix0 = i32(floor(gx));
        iy0 = i32(floor(gy));
        fx = gx - f32(ix0);
        fy = gy - f32(iy0);
        ix1 = (ix0 + 1) % dimI;
        iy1 = (iy0 + 1) % dimI;
    } else {
        let maxCoord = dimF - 1.0;
        if (gx < 0.0) {
            gx = 0.0;
        } else if (gx > maxCoord) {
            gx = maxCoord;
        }
        if (gy < 0.0) {
            gy = 0.0;
        } else if (gy > maxCoord) {
            gy = maxCoord;
        }
        ix0 = i32(floor(gx));
        iy0 = i32(floor(gy));
        ix1 = min(ix0 + 1, dimI - 1);
        iy1 = min(iy0 + 1, dimI - 1);
        fx = gx - f32(ix0);
        fy = gy - f32(iy0);
    }

    let i00 = u32(iy0) * dim + u32(ix0);
    let i10 = u32(iy0) * dim + u32(ix1);
    let i01 = u32(iy1) * dim + u32(ix0);
    let i11 = u32(iy1) * dim + u32(ix1);

    let gx00 = gradX[i00];
    let gx10 = gradX[i10];
    let gx01 = gradX[i01];
    let gx11 = gradX[i11];
    let topX = gx00 + (gx10 - gx00) * fx;
    let bottomX = gx01 + (gx11 - gx01) * fx;
    let sampledGradX = topX + (bottomX - topX) * fy;

    let gy00 = gradY[i00];
    let gy10 = gradY[i10];
    let gy01 = gradY[i01];
    let gy11 = gradY[i11];
    let topY = gy00 + (gy10 - gy00) * fx;
    let bottomY = gy01 + (gy11 - gy01) * fx;
    let sampledGradY = topY + (bottomY - topY) * fy;

    if (level.isFirstLevel != 0u) {
        accelX[i] = sampledGradX;
        accelY[i] = sampledGradY;
    } else {
        accelX[i] = accelX[i] + sampledGradX;
        accelY[i] = accelY[i] + sampledGradY;
    }
}
