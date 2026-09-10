// Multi-resolution grid-based 2D N-body gravity simulation.
//
// Implements the algorithm described in ../2D-optimized.md:
//   1. Bin particle mass into a high-resolution base grid ("particle-to-grid").
//   2. Build a pyramid of coarser LOD grids by summing 2x2 blocks (mipmap-style).
//   3. Spread mass outward on every LOD level with a separable 1D blur.
//   4. Compute the gradient of each blurred field (central differences).
//   5. Each particle samples the gradient from every LOD level and sums the
//      contributions to approximate near + far gravitational pull, then
//      integrates with symplectic Euler.
//
// Physics is 2D: all particles live in the XY plane (z is always 0). Physics
// runs on the CPU (typed arrays, no WebGPU calls) so it can scale to tens of
// thousands of particles. Rendering draws the mass-accumulation grid itself
// as a color-mapped heatmap texture (rather than individual particles), so
// there is no 3D content and no camera — the quad is cover-fit directly to
// the canvas each frame (see computeCoverQuadVertices).

const MAX_PARTICLES = 1000000;
const DOMAIN_HALF_SIZE = 50;
const BASE_GRID_SIZE = 1024; // must be a power of two
const NUM_LOD_LEVELS = 5;   // 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16
const BLUR_KERNEL_RADIUS = 3;
const FORCE_SCALE = 0.02;    // overall tuning constant, see 2D-optimized.md
const DAMPING = 0.999;
const RESTITUTION = 1;
const FIXED_DT = 0.016;

// Boundary handling modes (see 2D-optimized.md discussion + follow-up request):
//   'wrap'   - periodic boundaries: particles and the grid/blur wrap around.
//   'bounce' - hard box: particles are clamped and bounce with restitution.
//   'delete' - particles that leave the domain are culled (swap-removed).
const BOUNDARY_MODE_WRAP = 'wrap';
const BOUNDARY_MODE_BOUNCE = 'bounce';
const BOUNDARY_MODE_DELETE = 'delete';
const DEFAULT_BOUNDARY_MODE = BOUNDARY_MODE_BOUNCE;

// Wraps a coordinate into [-DOMAIN_HALF_SIZE, DOMAIN_HALF_SIZE) for periodic
// boundaries. Handles arbitrarily large/negative values (not just single
// overshoots) via modulo arithmetic.
function wrapCoordinate(v) {
    const size = 2 * DOMAIN_HALF_SIZE;
    let wrapped = (v + DOMAIN_HALF_SIZE) % size;
    if (wrapped < 0) wrapped += size;
    return wrapped - DOMAIN_HALF_SIZE;
}

// How much accumulated mass in a single base-grid cell counts as "full
// brightness" (t = 1) in the heatmap. Tuned so a handful of overlapping
// particles saturate towards white rather than requiring hundreds.
const MASS_VISUAL_SCALE = 2.0;
const COLOR_LUT_SIZE = 256;

// Custom 1D blur kernel approximating gravitational falloff (1/(1+|r|)),
// normalized to sum to 1. Applied separably (horizontal pass, then vertical).
function buildBlurKernel(radius) {
    const weights = [];
    let sum = 0;
    for (let r = -radius; r <= radius; r++) {
        const w = 1 / (1 + Math.abs(r));
        weights.push(w);
        sum += w;
    }
    return weights.map(w => w / sum);
}

const BLUR_KERNEL = buildBlurKernel(BLUR_KERNEL_RADIUS);

// approximating how a cell's accumulated mass ("heat") should look as it
// grows from empty to densely packed with particles.
const COLOR_STOPS = [
    { t: 0.00, color: [  0,   0,   0] },
    { t: 0.12, color: [ 10,  15,  45] },
    { t: 0.25, color: [ 60,  10,  65] },
    { t: 0.40, color: [160,  10,  30] },
    { t: 0.55, color: [230,  60,  10] },
    { t: 0.70, color: [255, 140,   0] },
    { t: 0.85, color: [255, 215,   0] },
    { t: 0.95, color: [255, 245, 200] },
    { t: 1.00, color: [255, 255, 255] },
];

function buildColorLUT(size) {
    const lut = new Uint8Array(size * 3);
    for (let i = 0; i < size; i++) {
        const t = i / (size - 1);
        let color = COLOR_STOPS[COLOR_STOPS.length - 1].color;
        for (let s = 0; s < COLOR_STOPS.length - 1; s++) {
            const a = COLOR_STOPS[s];
            const b = COLOR_STOPS[s + 1];
            if (t >= a.t && t <= b.t) {
                const localT = (t - a.t) / (b.t - a.t);
                color = [
                    a.color[0] + (b.color[0] - a.color[0]) * localT,
                    a.color[1] + (b.color[1] - a.color[1]) * localT,
                    a.color[2] + (b.color[2] - a.color[2]) * localT,
                ];
                break;
            }
        }
        lut[i * 3 + 0] = color[0];
        lut[i * 3 + 1] = color[1];
        lut[i * 3 + 2] = color[2];
    }
    return lut;
}

const COLOR_LUT = buildColorLUT(COLOR_LUT_SIZE);

// Computes the 6 vertices (position.xy in NDC, uv.xy) of a full-canvas quad
// that "cover-fits" the square (1:1) grid texture into a canvas of arbitrary
// aspect ratio: the quad always fills the entire viewport with no
// letterboxing, cropping whichever axis overhangs rather than stretching
// the content (so the texture itself is never distorted).
function computeCoverQuadVertices(canvasAspect) {
    let uvHalfW, uvHalfH;
    if (canvasAspect >= 1) {
        // Wider than tall (or square): full texture width visible, crop top/bottom.
        uvHalfW = 0.5;
        uvHalfH = 0.5 / canvasAspect;
    } else {
        // Taller than wide: full texture height visible, crop left/right.
        uvHalfW = 0.5 * canvasAspect;
        uvHalfH = 0.5;
    }

    const u0 = 0.5 - uvHalfW, u1 = 0.5 + uvHalfW;
    const v0 = 0.5 - uvHalfH, v1 = 0.5 + uvHalfH;

    // Two triangles covering NDC [-1,1] x [-1,1], each vertex is (x, y, u, v).
    return new Float32Array([
        -1, -1, u0, v0,
         1, -1, u1, v0,
        -1,  1, u0, v1,
         1, -1, u1, v0,
         1,  1, u1, v1,
        -1,  1, u0, v1,
    ]);
}

// Formats a time-scale multiplier (0.001x .. 1000x) with roughly 3
// significant figures, used by the log-scale Time Scale slider's label.
function formatTimeScale(scale) {
    if (!(scale > 0)) return '0x';
    const digits = Math.max(0, Math.min(6, 2 - Math.floor(Math.log10(scale))));
    return `${scale.toFixed(digits)}x`;
}

class GridNBodySimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;

        // Simulation parameters
        this.particleCount = 5;
        this.gravityStrength = 1.0;
        this.timeScale = 1.0;
        this.boundaryMode = DEFAULT_BOUNDARY_MODE;

        // Particle data (CPU-based, structure-of-arrays for performance at high N)
        this.posX = new Float32Array(MAX_PARTICLES);
        this.posY = new Float32Array(MAX_PARTICLES);
        this.velX = new Float32Array(MAX_PARTICLES);
        this.velY = new Float32Array(MAX_PARTICLES);
        this.mass = new Float32Array(MAX_PARTICLES);

        // Number of particles currently alive (<= particleCount). Only
        // shrinks in 'delete' boundary mode, where out-of-bounds particles
        // are swap-removed rather than simulated forever.
        this.aliveCount = 0;

        // Multi-resolution grid hierarchy
        this.levels = [];
        this.allocateGrids();

        // Heatmap image buffer (RGBA bytes) rebuilt from the base mass grid
        // every frame and uploaded to gridTexture for rendering.
        const baseDim = this.levels[0].dim;
        this.gridImageBuffer = new Uint8Array(baseDim * baseDim * 4);

        // WebGPU resources (render only)
        this.gridTexture = null;
        this.gridTextureView = null;
        this.gridSampler = null;
        this.quadVertexBuffer = null;

        // Render pipeline
        this.heatmapPipeline = null;
        this.heatmapBindGroup = null;

        // Performance tracking
        this.frameCount = 0;
        this.lastTime = 0;
        this.fps = 0;
        this.computeTime = 0;
        this.renderTime = 0;
    }

    async init() {
        // Initialize WebGPU (render only)
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No appropriate GPUAdapter found');
        }

        this.device = await adapter.requestDevice();

        this.context = this.canvas.getContext('webgpu');
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: canvasFormat,
        });

        this.initializeParticles();
        await this.initResources();
        await this.initHeatmapPipeline();
    }

    // ---- Grid allocation -------------------------------------------------

    allocateGrids() {
        this.levels = [];
        let dim = BASE_GRID_SIZE;
        for (let level = 0; level < NUM_LOD_LEVELS; level++) {
            const cells = dim * dim;
            this.levels.push({
                dim,
                cellSize: (2 * DOMAIN_HALF_SIZE) / dim,
                mass: new Float32Array(cells),
                temp: new Float32Array(cells),
                field: new Float32Array(cells),
                gradX: new Float32Array(cells),
                gradY: new Float32Array(cells),
            });
            dim = dim >> 1;
        }
    }

    // ---- Particle initialization ------------------------------------------

    initializeParticles() {
        // Seed particles in a 2D disc (annulus, avoiding the exact center to
        // sidestep a degenerate zero-radius velocity sample) with circular
        // orbital velocity, similar in spirit to a flat galaxy disc.
        const minRadius = 0.2;
        const maxRadius = DOMAIN_HALF_SIZE * 0.6;

        for (let i = 0; i < this.particleCount; i++) {
            const radius = minRadius + Math.random() * (maxRadius - minRadius);
            const theta = Math.random() * Math.PI * 2;

            this.posX[i] = radius * Math.cos(theta);
            this.posY[i] = radius * Math.sin(theta);
            this.mass[i] = 0.6 + Math.random() * 0.8;

            const enclosedMassEstimate = this.particleCount * 0.6 * (radius / maxRadius);
            const speed = Math.sqrt(this.gravityStrength * FORCE_SCALE * enclosedMassEstimate / radius) * 0.3;

            this.velX[i] = -speed * Math.sin(theta);
            this.velY[i] = speed * Math.cos(theta);
        }

        this.aliveCount = this.particleCount;
    }

    // ---- Physics: Step 2.1 Mass accumulation ------------------------------

    accumulateMass() {
        const base = this.levels[0];
        base.mass.fill(0);

        const dim = base.dim;
        const cellSize = base.cellSize;

        for (let i = 0; i < this.aliveCount; i++) {
            let cx = Math.floor((this.posX[i] + DOMAIN_HALF_SIZE) / cellSize);
            let cy = Math.floor((this.posY[i] + DOMAIN_HALF_SIZE) / cellSize);
            if (cx < 0) cx = 0; else if (cx >= dim) cx = dim - 1;
            if (cy < 0) cy = 0; else if (cy >= dim) cy = dim - 1;
            base.mass[cy * dim + cx] += this.mass[i];
        }
    }

    // ---- Physics: Step 2.2 Downsample LOD pyramid -------------------------

    buildLODs() {
        for (let level = 1; level < this.levels.length; level++) {
            const prev = this.levels[level - 1];
            const cur = this.levels[level];
            const prevDim = prev.dim;
            const dim = cur.dim;

            for (let y = 0; y < dim; y++) {
                const py0 = 2 * y;
                const py1 = py0 + 1;
                for (let x = 0; x < dim; x++) {
                    const px0 = 2 * x;
                    const px1 = px0 + 1;
                    const sum =
                        prev.mass[py0 * prevDim + px0] + prev.mass[py0 * prevDim + px1] +
                        prev.mass[py1 * prevDim + px0] + prev.mass[py1 * prevDim + px1];
                    cur.mass[y * dim + x] = sum;
                }
            }
        }
    }

    // ---- Physics: Step 2.3 Separable blur ----------------------------------

    blurGrids() {
        const wrap = this.boundaryMode === BOUNDARY_MODE_WRAP;
        for (const level of this.levels) {
            this.blurSeparable(level.mass, level.temp, level.field, level.dim, wrap);
        }
    }

    blurSeparable(src, temp, dst, dim, wrap) {
        const radius = BLUR_KERNEL_RADIUS;

        // Horizontal pass: src -> temp
        for (let y = 0; y < dim; y++) {
            const row = y * dim;
            for (let x = 0; x < dim; x++) {
                let acc = 0;
                for (let k = -radius; k <= radius; k++) {
                    let sx = x + k;
                    if (wrap) {
                        sx = ((sx % dim) + dim) % dim;
                    } else if (sx < 0) {
                        sx = 0;
                    } else if (sx >= dim) {
                        sx = dim - 1;
                    }
                    acc += src[row + sx] * BLUR_KERNEL[k + radius];
                }
                temp[row + x] = acc;
            }
        }

        // Vertical pass: temp -> dst
        for (let y = 0; y < dim; y++) {
            for (let x = 0; x < dim; x++) {
                let acc = 0;
                for (let k = -radius; k <= radius; k++) {
                    let sy = y + k;
                    if (wrap) {
                        sy = ((sy % dim) + dim) % dim;
                    } else if (sy < 0) {
                        sy = 0;
                    } else if (sy >= dim) {
                        sy = dim - 1;
                    }
                    acc += temp[sy * dim + x] * BLUR_KERNEL[k + radius];
                }
                dst[y * dim + x] = acc;
            }
        }
    }

    // ---- Physics: Step 2.4 Gradient fields ---------------------------------

    computeGradients() {
        const wrap = this.boundaryMode === BOUNDARY_MODE_WRAP;

        for (const level of this.levels) {
            const { field, gradX, gradY, dim, cellSize } = level;
            const invTwoCell = 1 / (2 * cellSize);

            for (let y = 0; y < dim; y++) {
                const row = y * dim;
                let yUp, yDown;
                if (wrap) {
                    yUp = ((y - 1 + dim) % dim) * dim;
                    yDown = ((y + 1) % dim) * dim;
                } else {
                    yUp = (y > 0 ? y - 1 : 0) * dim;
                    yDown = (y < dim - 1 ? y + 1 : dim - 1) * dim;
                }

                for (let x = 0; x < dim; x++) {
                    let xLeft, xRight;
                    if (wrap) {
                        xLeft = (x - 1 + dim) % dim;
                        xRight = (x + 1) % dim;
                    } else {
                        xLeft = x > 0 ? x - 1 : 0;
                        xRight = x < dim - 1 ? x + 1 : dim - 1;
                    }

                    gradX[row + x] = (field[row + xRight] - field[row + xLeft]) * invTwoCell;
                    gradY[row + x] = (field[yDown + x] - field[yUp + x]) * invTwoCell;
                }
            }
        }
    }

    // ---- Physics: Step 2.4/2.5 Force sampling + integration ----------------

    sampleBilinear(grid, dim, cellSize, px, py, wrap) {
        // Convert world position to fractional grid coordinates (cell centers
        // sit at integer coordinates + 0.5, hence the -0.5 offset).
        let gx = (px + DOMAIN_HALF_SIZE) / cellSize - 0.5;
        let gy = (py + DOMAIN_HALF_SIZE) / cellSize - 0.5;

        let ix0, iy0, ix1, iy1, fx, fy;
        if (wrap) {
            gx = ((gx % dim) + dim) % dim;
            gy = ((gy % dim) + dim) % dim;
            ix0 = Math.floor(gx);
            iy0 = Math.floor(gy);
            fx = gx - ix0;
            fy = gy - iy0;
            ix1 = (ix0 + 1) % dim;
            iy1 = (iy0 + 1) % dim;
        } else {
            const maxCoord = dim - 1;
            if (gx < 0) gx = 0; else if (gx > maxCoord) gx = maxCoord;
            if (gy < 0) gy = 0; else if (gy > maxCoord) gy = maxCoord;

            ix0 = Math.floor(gx);
            iy0 = Math.floor(gy);
            ix1 = Math.min(ix0 + 1, maxCoord);
            iy1 = Math.min(iy0 + 1, maxCoord);
            fx = gx - ix0;
            fy = gy - iy0;
        }

        const v00 = grid[iy0 * dim + ix0];
        const v10 = grid[iy0 * dim + ix1];
        const v01 = grid[iy1 * dim + ix0];
        const v11 = grid[iy1 * dim + ix1];

        const top = v00 + (v10 - v00) * fx;
        const bottom = v01 + (v11 - v01) * fx;
        return top + (bottom - top) * fy;
    }

    computeForcesAndIntegrate(dt) {
        const wrap = this.boundaryMode === BOUNDARY_MODE_WRAP;
        const mode = this.boundaryMode;

        for (let i = 0; i < this.aliveCount; i++) {
            const px = this.posX[i];
            const py = this.posY[i];

            let accX = 0;
            let accY = 0;

            // Multi-resolution sampling: sum gradients from every LOD level.
            // Finer levels naturally dominate at short range (smaller cell
            // spacing means larger gradients for the same mass difference);
            // coarser levels contribute the long-range, low-frequency pull.
            for (const level of this.levels) {
                accX += this.sampleBilinear(level.gradX, level.dim, level.cellSize, px, py, wrap);
                accY += this.sampleBilinear(level.gradY, level.dim, level.cellSize, px, py, wrap);
            }

            accX *= this.gravityStrength * FORCE_SCALE;
            accY *= this.gravityStrength * FORCE_SCALE;

            let vx = (this.velX[i] + accX * dt) * DAMPING;
            let vy = (this.velY[i] + accY * dt) * DAMPING;

            let nx = px + vx * dt;
            let ny = py + vy * dt;

            if (mode === BOUNDARY_MODE_WRAP) {
                // Periodic boundaries: particles that exit one side re-enter
                // on the opposite side (grid/blur wrap-around is handled in
                // blurSeparable/computeGradients/sampleBilinear above).
                nx = wrapCoordinate(nx);
                ny = wrapCoordinate(ny);
            } else if (mode === BOUNDARY_MODE_DELETE) {
                // Culling: particles beyond the domain are destroyed. Swap
                // the last alive particle into this slot and shrink
                // aliveCount, then reprocess this index (the swapped-in
                // particle hasn't been integrated yet this frame). Forces
                // come from the grid, not other particles, so recomputing
                // out of order is still correct.
                if (Math.abs(nx) > DOMAIN_HALF_SIZE || Math.abs(ny) > DOMAIN_HALF_SIZE) {
                    this.aliveCount--;
                    const last = this.aliveCount;
                    this.posX[i] = this.posX[last];
                    this.posY[i] = this.posY[last];
                    this.velX[i] = this.velX[last];
                    this.velY[i] = this.velY[last];
                    this.mass[i] = this.mass[last];
                    i--;
                    continue;
                }
            } else {
                // Clamping (bounce): hard box, invert velocity with damping.
                if (Math.abs(nx) > DOMAIN_HALF_SIZE) {
                    nx = Math.sign(nx) * DOMAIN_HALF_SIZE;
                    vx *= -RESTITUTION;
                }
                if (Math.abs(ny) > DOMAIN_HALF_SIZE) {
                    ny = Math.sign(ny) * DOMAIN_HALF_SIZE;
                    vy *= -RESTITUTION;
                }
            }

            this.velX[i] = vx;
            this.velY[i] = vy;
            this.posX[i] = nx;
            this.posY[i] = ny;
        }
    }

    updatePhysics(deltaTime) {
        this.accumulateMass();
        this.buildLODs();
        this.blurGrids();
        this.computeGradients();
        this.computeForcesAndIntegrate(deltaTime);
    }

    // ---- WebGPU resources / rendering --------------------------------------

    async initResources() {
        const dim = this.levels[0].dim;

        // Grid texture: one texel per base-grid cell, colorized on the CPU
        // each frame from the raw (unblurred) mass accumulation (2D-optimized.md
        // Step 2.1) and uploaded here for rendering.
        this.gridTexture = this.device.createTexture({
            size: [dim, dim],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.gridTextureView = this.gridTexture.createView();

        this.gridSampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Vertex buffer for the cover-fit full-canvas quad: 6 vertices,
        // each (x, y, u, v) as float32 — rebuilt every frame from the
        // current canvas aspect ratio (see computeCoverQuadVertices).
        this.quadVertexBuffer = this.device.createBuffer({
            size: 6 * 4 * 4, // 6 vertices * 4 floats * 4 bytes
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
    }

    async loadShader(url) {
        const response = await fetch(url);
        return await response.text();
    }

    async initHeatmapPipeline() {
        const vertexShader = await this.loadShader('./shaders/grid-heatmap-vertex.wgsl');
        const fragmentShader = await this.loadShader('./shaders/grid-heatmap-fragment.wgsl');

        this.heatmapPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: vertexShader }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 4 * 4, // (x, y, u, v) floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 2 * 4, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: this.device.createShaderModule({ code: fragmentShader }),
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        this.heatmapBindGroup = this.device.createBindGroup({
            layout: this.heatmapPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.gridSampler },
                { binding: 1, resource: this.gridTextureView },
            ],
        });
    }

    // Rebuilds the heatmap image from the base mass grid and uploads it to
    // gridTexture. Uses a per-cell exponential saturating curve (rather than
    // normalizing by a global maximum) so a cell's color is stable frame to
    // frame and doesn't flicker as unrelated cells gain or lose mass.
    updateGridImage() {
        const base = this.levels[0];
        const dim = base.dim;
        const massGrid = base.mass;
        const buffer = this.gridImageBuffer;
        const lut = COLOR_LUT;
        const maxLutIndex = COLOR_LUT_SIZE - 1;

        for (let i = 0, cellCount = dim * dim; i < cellCount; i++) {
            const t = 1 - Math.exp(-massGrid[i] / MASS_VISUAL_SCALE);
            const lutIndex = Math.min(maxLutIndex, (t * maxLutIndex) | 0);
            const c = lutIndex * 3;
            const p = i * 4;
            buffer[p + 0] = lut[c + 0];
            buffer[p + 1] = lut[c + 1];
            buffer[p + 2] = lut[c + 2];
            buffer[p + 3] = 255;
        }

        this.device.queue.writeTexture(
            { texture: this.gridTexture },
            buffer,
            { bytesPerRow: dim * 4, rowsPerImage: dim },
            { width: dim, height: dim }
        );
    }

    // Recomputes the cover-fit quad vertices for the current canvas size and
    // uploads them. Cheap (6 vertices), so it's safe to call every frame,
    // which keeps it correct across canvas resizes without extra wiring.
    updateQuadVertexBuffer() {
        const aspect = this.canvas.width / this.canvas.height;
        const vertexData = computeCoverQuadVertices(aspect);
        this.device.queue.writeBuffer(this.quadVertexBuffer, 0, vertexData);
    }

    render(time) {
        const computeStart = performance.now();
        this.updatePhysics(FIXED_DT * this.timeScale);
        this.updateGridImage();
        const computeEnd = performance.now();
        this.computeTime = computeEnd - computeStart;

        this.updateQuadVertexBuffer();

        const renderStart = performance.now();
        const commandEncoder = this.device.createCommandEncoder();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.heatmapPipeline);
        renderPass.setBindGroup(0, this.heatmapBindGroup);
        renderPass.setVertexBuffer(0, this.quadVertexBuffer);
        renderPass.draw(6);

        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        const renderEnd = performance.now();
        this.renderTime = renderEnd - renderStart;

        // Update performance stats
        this.frameCount++;
        if (time - this.lastTime > 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (time - this.lastTime));
            this.frameCount = 0;
            this.lastTime = time;
        }
    }

    setParticleCount(count) {
        // The heatmap texture size is fixed (one texel per base-grid cell)
        // regardless of particle count, so no GPU resources need recreating.
        this.particleCount = Math.floor(Math.min(count, MAX_PARTICLES));
        this.initializeParticles();
    }

    setGravityStrength(strength) { this.gravityStrength = strength; }

    setTimeScale(scale) { this.timeScale = scale; }

    setBoundaryMode(mode) {
        // Mode switches reset the simulation so 'delete' mode's culled
        // particle count (aliveCount) doesn't linger into another mode.
        this.boundaryMode = mode;
        this.initializeParticles();
    }

    resetSimulation() { this.initializeParticles(); }
}

// App wiring for index.html controls
class App {
    constructor() {
        this.simulation = null;
        this.animationId = null;
    }

    async init() {
        const canvas = document.getElementById('canvas');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const controls = document.getElementById('controls');

        try {
            this.resizeCanvas(canvas);
            window.addEventListener('resize', () => this.resizeCanvas(canvas));

            this.simulation = new GridNBodySimulation(canvas);
            await this.simulation.init();

            loading.style.display = 'none';
            controls.style.display = 'block';

            this.setupControls();
            this.animate();

        } catch (err) {
            console.error('Failed to initialize:', err);
            loading.style.display = 'none';
            error.style.display = 'block';
            document.getElementById('errorMessage').textContent = err.message || String(err);
        }
    }

    resizeCanvas(canvas) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * devicePixelRatio;
        canvas.height = rect.height * devicePixelRatio;
        // No camera aspect to update — render() reads canvas.width/height
        // directly every frame to cover-fit the heatmap quad.
    }

    setupControls() {
        const particleCountSlider = document.getElementById('particleCount');
        const gravitySlider = document.getElementById('gravity');
        const timeScaleSlider = document.getElementById('timeScale');
        const boundaryModeSelect = document.getElementById('boundaryMode');
        const resetBtn = document.getElementById('resetBtn');

        const particleCountValue = document.getElementById('particleCountValue');
        const gravityValue = document.getElementById('gravityValue');
        const timeScaleValue = document.getElementById('timeScaleValue');

        // The simulation constructor defaults to a small particleCount so it
        // starts up fast even if this script is embedded elsewhere; sync it
        // to whatever this page's slider markup declares as its starting
        // value before the render loop begins.
        this.simulation.setParticleCount(parseInt(particleCountSlider.value, 10));

        // index-optimized.html already sets the slider range appropriate for
        // this grid-based solver, so no overrides here.
        particleCountSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            particleCountValue.textContent = value;
            this.simulation.setParticleCount(value);
        });

        gravitySlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            gravityValue.textContent = value.toFixed(1);
            this.simulation.setGravityStrength(value);
        });

        // Time Scale is a log-scale slider (exponent in [-3, 3]) so a single
        // control can usefully span 0.001x to 1000x.
        timeScaleSlider.addEventListener('input', (e) => {
            const exponent = parseFloat(e.target.value);
            const scale = Math.pow(10, exponent);
            timeScaleValue.textContent = formatTimeScale(scale);
            this.simulation.setTimeScale(scale);
        });

        boundaryModeSelect.addEventListener('change', (e) => {
            this.simulation.setBoundaryMode(e.target.value);
        });

        resetBtn.addEventListener('click', () => {
            this.simulation.resetSimulation();
        });
    }

    animate() {
        const time = performance.now();

        if (this.simulation) {
            this.simulation.render(time);

            document.getElementById('fps').textContent = this.simulation.fps;
            document.getElementById('computeTime').textContent = this.simulation.computeTime.toFixed(3);
            document.getElementById('renderTime').textContent = this.simulation.renderTime.toFixed(3);
            document.getElementById('activeParticles').textContent = this.simulation.aliveCount;
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

export { App, GridNBodySimulation };

// Start the application (guarded so this module can be imported in Node,
// e.g. for testing the physics, without a DOM/browser environment).
if (typeof document !== 'undefined') {
    const app = new App();
    app.init();
}