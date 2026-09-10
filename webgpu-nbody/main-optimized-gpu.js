// Multi-resolution grid-based 2D N-body gravity simulation — GPU compute
// variant. Implements the same algorithm as main-optimized.js (see
// ../2D-optimized.md) but runs the majority of the per-frame work as WebGPU
// compute shaders instead of CPU JavaScript:
//   1. Mass scatter (2D-optimized.md Step 2.1) — atomic fixed-point add into
//      the base grid (shaders/gpu/mass-scatter.wgsl), then resolved to f32
//      (mass-resolve.wgsl). WGSL has no atomic<f32>, hence the fixed-point
//      trick (see MASS_FIXED_POINT_SCALE).
//   2. LOD pyramid (Step 2.2) — lod-downsample.wgsl, one dispatch per level.
//   3. Separable blur (Step 2.3) — blur-horizontal.wgsl / blur-vertical.wgsl,
//      one dispatch pair per level.
//   4. Gradient fields (Step 2.4) — gradient.wgsl, one dispatch per level.
//   5. Multi-level force sampling — accumulate-gradient.wgsl, one dispatch
//      per level, summing into per-particle accelX/accelY buffers.
//   6. Integration + boundary handling (Step 2.5) — integrate.wgsl (wrap /
//      bounce / delete, see BOUNDARY_MODE_* below).
//   7. Heatmap colorization — colorize.wgsl writes directly into a storage
//      texture (no CPU readback/upload roundtrip at all).
//
// Particle state (position/velocity/mass/alive flag) lives entirely in GPU
// storage buffers; it is only uploaded from the CPU once at startup and
// whenever particle count / boundary mode / reset changes. Every steady
// -state frame is CPU→GPU uniform writes (tiny) + compute dispatches +
// a render pass — no per-frame particle or grid readback.
//
// 'delete' boundary mode does not physically compact the particle buffers
// like the CPU version's swap-remove (that would need GPU stream
// compaction / prefix sums, disproportionate to this demo's scope).
// Instead, dead particles carry an `alive` flag and are skipped early in
// the scatter/accumulate/integrate shaders — see integrate.wgsl for
// details. The "Active Particles" stat is refreshed via a small,
// non-blocking periodic GPU→CPU readback of a single atomic counter.

const MAX_PARTICLES = 4000000;
const DOMAIN_HALF_SIZE = 50;
const BASE_GRID_SIZE = 1024; // must be a power of two
const NUM_LOD_LEVELS = 9;    // 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
const FORCE_SCALE = 0.002;     // overall tuning constant, see 2D-optimized.md
const DAMPING = 0.999;
const RESTITUTION = 1;
const FIXED_DT = 0.016;
const MASS_VISUAL_SCALE = 2.0;
// Fixed-point scale for the atomic mass-scatter pass (WGSL has no
// atomic<f32>). Must be large enough for sub-integer mass precision but
// small enough that heavily-populated cells don't overflow i32.
const MASS_FIXED_POINT_SCALE = 65536;
const WORKGROUP_SIZE = 64;

const BOUNDARY_MODE_CODES = { bounce: 0, wrap: 1, delete: 2 };
const DEFAULT_BOUNDARY_MODE = 'bounce';

const ALIVE_READBACK_INTERVAL_FRAMES = 20;

function dispatchCount(n) {
    return Math.max(1, Math.ceil(n / WORKGROUP_SIZE));
}

// Computes the 6 vertices (position.xy in NDC, uv.xy) of a full-canvas quad
// that "cover-fits" the square (1:1) grid texture into a canvas of arbitrary
// aspect ratio: the quad always fills the entire viewport with no
// letterboxing, cropping whichever axis overhangs rather than stretching
// the content (so the texture itself is never distorted). Identical to the
// CPU version's helper of the same name.
function computeCoverQuadVertices(canvasAspect) {
    let uvHalfW, uvHalfH;
    if (canvasAspect >= 1) {
        uvHalfW = 0.5;
        uvHalfH = 0.5 / canvasAspect;
    } else {
        uvHalfW = 0.5 * canvasAspect;
        uvHalfH = 0.5;
    }

    const u0 = 0.5 - uvHalfW, u1 = 0.5 + uvHalfW;
    const v0 = 0.5 - uvHalfH, v1 = 0.5 + uvHalfH;

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

class GpuGridNBodySimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;

        this.particleCount = 5;
        this.gravityStrength = 1.0;
        this.timeScale = 1.0;
        this.boundaryMode = DEFAULT_BOUNDARY_MODE;

        // Last known alive count, refreshed periodically via an async GPU
        // readback (see maybeReadbackAliveCount). Optimistically set to
        // particleCount immediately after any reset.
        this.aliveCount = 0;
        this.aliveReadbackPending = false;
        this.framesSinceAliveReadback = 0;

        // Level descriptors (dim, cellSize); the actual GPU buffers for each
        // level are created in createBuffers() as this.levelBuffers.
        this.levels = [];
        let dim = BASE_GRID_SIZE;
        for (let i = 0; i < NUM_LOD_LEVELS; i++) {
            this.levels.push({ dim, cellSize: (2 * DOMAIN_HALF_SIZE) / dim });
            dim = dim >> 1;
        }
        this.levelBuffers = [];

        this.frameCount = 0;
        this.lastTime = 0;
        this.fps = 0;
        this.computeTime = 0;
        this.renderTime = 0;
    }

    async init() {
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

        this.createBuffers();
        this.writeLevelParams();
        await this.createPipelines();
        this.createBindGroups();
        this.initializeParticles();
    }

    // ---- GPU resource setup -------------------------------------------------

    createBuffers() {
        const dev = this.device;
        const STORAGE = GPUBufferUsage.STORAGE;
        const COPY_DST = GPUBufferUsage.COPY_DST;
        const COPY_SRC = GPUBufferUsage.COPY_SRC;
        const UNIFORM = GPUBufferUsage.UNIFORM;
        const VERTEX = GPUBufferUsage.VERTEX;

        // Particle state (structure-of-arrays), sized for the maximum
        // particle count so changing particleCount never needs to recreate
        // GPU resources — only re-upload/re-dispatch a different range.
        this.posXBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.posYBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.velXBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.velYBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.massBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.aliveBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.accelXBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE });
        this.accelYBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE });

        this.aliveCounterBuf = dev.createBuffer({ size: 4, usage: STORAGE | COPY_DST | COPY_SRC });
        this.aliveCounterStagingBuf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | COPY_DST });

        // Per-level grid buffers.
        this.levelBuffers = this.levels.map((level) => {
            const cellCount = level.dim * level.dim;
            return {
                rawMass: dev.createBuffer({ size: cellCount * 4, usage: STORAGE }),
                field: dev.createBuffer({ size: cellCount * 4, usage: STORAGE }),
                gradX: dev.createBuffer({ size: cellCount * 4, usage: STORAGE }),
                gradY: dev.createBuffer({ size: cellCount * 4, usage: STORAGE }),
            };
        });

        const baseCellCount = this.levels[0].dim * this.levels[0].dim;
        this.atomicMassBuf = dev.createBuffer({ size: baseCellCount * 4, usage: STORAGE });
        // Shared horizontal-blur scratch buffer, reused across levels
        // (sized for the largest/base level; smaller levels just use a
        // leading sub-range).
        this.blurTempBuf = dev.createBuffer({ size: baseCellCount * 4, usage: STORAGE });

        // Uniforms.
        this.simParamsBuf = dev.createBuffer({ size: 48, usage: UNIFORM | COPY_DST });
        this.levelParamsBufs = this.levels.map(() => dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST }));

        // Heatmap texture: written directly by colorize.wgsl (storage
        // texture), sampled directly by the render pass (texture binding) —
        // no CPU involvement in between.
        const dim0 = this.levels[0].dim;
        this.gridTexture = dev.createTexture({
            size: [dim0, dim0],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gridTextureView = this.gridTexture.createView();
        this.gridSampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        this.quadVertexBuffer = dev.createBuffer({ size: 6 * 4 * 4, usage: VERTEX | COPY_DST });
    }

    // Writes the (mostly static) per-level uniform buffers: dim, cellSize,
    // wrap flag, and whether this is the first LOD level (used by
    // accumulate-gradient.wgsl to initialize vs. accumulate). Called once at
    // startup and again whenever boundaryMode changes (wrap flag only).
    writeLevelParams() {
        const wrap = this.boundaryMode === 'wrap' ? 1 : 0;
        for (let i = 0; i < this.levels.length; i++) {
            const level = this.levels[i];
            const data = new ArrayBuffer(16);
            new Uint32Array(data, 0, 1)[0] = level.dim;
            new Float32Array(data, 4, 1)[0] = level.cellSize;
            new Uint32Array(data, 8, 1)[0] = wrap;
            new Uint32Array(data, 12, 1)[0] = i === 0 ? 1 : 0;
            this.device.queue.writeBuffer(this.levelParamsBufs[i], 0, data);
        }
    }

    writeSimParams() {
        const data = new ArrayBuffer(48);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32[0] = this.gravityStrength * FORCE_SCALE; // forceMultiplier
        f32[1] = DAMPING;
        f32[2] = FIXED_DT * this.timeScale;
        f32[3] = DOMAIN_HALF_SIZE;
        f32[4] = RESTITUTION;
        u32[5] = BOUNDARY_MODE_CODES[this.boundaryMode];
        u32[6] = this.particleCount;
        f32[7] = MASS_FIXED_POINT_SCALE;
        f32[8] = MASS_VISUAL_SCALE;
        this.device.queue.writeBuffer(this.simParamsBuf, 0, data);
    }

    async loadShader(url) {
        const response = await fetch(url);
        return await response.text();
    }

    async createPipelines() {
        const dev = this.device;
        const [
            clearSrc, scatterSrc, resolveSrc, downsampleSrc,
            blurHSrc, blurVSrc, gradientSrc, accumSrc, integrateSrc, colorizeSrc,
            heatmapVertexSrc, heatmapFragmentSrc,
        ] = await Promise.all([
            this.loadShader('./shaders/gpu/mass-clear.wgsl'),
            this.loadShader('./shaders/gpu/mass-scatter.wgsl'),
            this.loadShader('./shaders/gpu/mass-resolve.wgsl'),
            this.loadShader('./shaders/gpu/lod-downsample.wgsl'),
            this.loadShader('./shaders/gpu/blur-horizontal.wgsl'),
            this.loadShader('./shaders/gpu/blur-vertical.wgsl'),
            this.loadShader('./shaders/gpu/gradient.wgsl'),
            this.loadShader('./shaders/gpu/accumulate-gradient.wgsl'),
            this.loadShader('./shaders/gpu/integrate.wgsl'),
            this.loadShader('./shaders/gpu/colorize.wgsl'),
            this.loadShader('./shaders/grid-heatmap-vertex.wgsl'),
            this.loadShader('./shaders/grid-heatmap-fragment.wgsl'),
        ]);

        const makeComputePipeline = (code) => dev.createComputePipeline({
            layout: 'auto',
            compute: { module: dev.createShaderModule({ code }), entryPoint: 'cs_main' },
        });

        this.clearPipeline = makeComputePipeline(clearSrc);
        this.scatterPipeline = makeComputePipeline(scatterSrc);
        this.resolvePipeline = makeComputePipeline(resolveSrc);
        this.downsamplePipeline = makeComputePipeline(downsampleSrc);
        this.blurHPipeline = makeComputePipeline(blurHSrc);
        this.blurVPipeline = makeComputePipeline(blurVSrc);
        this.gradientPipeline = makeComputePipeline(gradientSrc);
        this.accumulatePipeline = makeComputePipeline(accumSrc);
        this.integratePipeline = makeComputePipeline(integrateSrc);
        this.colorizePipeline = makeComputePipeline(colorizeSrc);

        this.renderPipeline = dev.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: dev.createShaderModule({ code: heatmapVertexSrc }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 4 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 2 * 4, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: dev.createShaderModule({ code: heatmapFragmentSrc }),
                entryPoint: 'fs_main',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    createBindGroups() {
        const dev = this.device;
        const L = this.levelBuffers;
        const P = this.levelParamsBufs;

        this.clearBindGroup = dev.createBindGroup({
            layout: this.clearPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: P[0] } },
                { binding: 1, resource: { buffer: this.atomicMassBuf } },
            ],
        });

        this.scatterBindGroup = dev.createBindGroup({
            layout: this.scatterPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: P[0] } },
                { binding: 2, resource: { buffer: this.posXBuf } },
                { binding: 3, resource: { buffer: this.posYBuf } },
                { binding: 4, resource: { buffer: this.massBuf } },
                { binding: 5, resource: { buffer: this.aliveBuf } },
                { binding: 6, resource: { buffer: this.atomicMassBuf } },
            ],
        });

        this.resolveBindGroup = dev.createBindGroup({
            layout: this.resolvePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: P[0] } },
                { binding: 2, resource: { buffer: this.atomicMassBuf } },
                { binding: 3, resource: { buffer: L[0].rawMass } },
            ],
        });

        this.downsampleBindGroups = [];
        for (let i = 1; i < this.levels.length; i++) {
            this.downsampleBindGroups.push(dev.createBindGroup({
                layout: this.downsamplePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: P[i] } },
                    { binding: 1, resource: { buffer: L[i - 1].rawMass } },
                    { binding: 2, resource: { buffer: L[i].rawMass } },
                ],
            }));
        }

        this.blurHBindGroups = [];
        this.blurVBindGroups = [];
        this.gradientBindGroups = [];
        this.accumulateBindGroups = [];
        for (let i = 0; i < this.levels.length; i++) {
            this.blurHBindGroups.push(dev.createBindGroup({
                layout: this.blurHPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: P[i] } },
                    { binding: 1, resource: { buffer: L[i].rawMass } },
                    { binding: 2, resource: { buffer: this.blurTempBuf } },
                ],
            }));
            this.blurVBindGroups.push(dev.createBindGroup({
                layout: this.blurVPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: P[i] } },
                    { binding: 1, resource: { buffer: this.blurTempBuf } },
                    { binding: 2, resource: { buffer: L[i].field } },
                ],
            }));
            this.gradientBindGroups.push(dev.createBindGroup({
                layout: this.gradientPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: P[i] } },
                    { binding: 1, resource: { buffer: L[i].field } },
                    { binding: 2, resource: { buffer: L[i].gradX } },
                    { binding: 3, resource: { buffer: L[i].gradY } },
                ],
            }));
            this.accumulateBindGroups.push(dev.createBindGroup({
                layout: this.accumulatePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.simParamsBuf } },
                    { binding: 1, resource: { buffer: P[i] } },
                    { binding: 2, resource: { buffer: this.posXBuf } },
                    { binding: 3, resource: { buffer: this.posYBuf } },
                    { binding: 4, resource: { buffer: this.aliveBuf } },
                    { binding: 5, resource: { buffer: L[i].gradX } },
                    { binding: 6, resource: { buffer: L[i].gradY } },
                    { binding: 7, resource: { buffer: this.accelXBuf } },
                    { binding: 8, resource: { buffer: this.accelYBuf } },
                ],
            }));
        }

        this.integrateBindGroup = dev.createBindGroup({
            layout: this.integratePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: this.posXBuf } },
                { binding: 2, resource: { buffer: this.posYBuf } },
                { binding: 3, resource: { buffer: this.velXBuf } },
                { binding: 4, resource: { buffer: this.velYBuf } },
                { binding: 5, resource: { buffer: this.accelXBuf } },
                { binding: 6, resource: { buffer: this.accelYBuf } },
                { binding: 7, resource: { buffer: this.aliveBuf } },
                { binding: 8, resource: { buffer: this.aliveCounterBuf } },
            ],
        });

        this.colorizeBindGroup = dev.createBindGroup({
            layout: this.colorizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: P[0] } },
                { binding: 2, resource: { buffer: L[0].rawMass } },
                { binding: 3, resource: this.gridTextureView },
            ],
        });

        this.renderBindGroup = dev.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.gridSampler },
                { binding: 1, resource: this.gridTextureView },
            ],
        });
    }

    // ---- Particle initialization / parameter updates -----------------------

    // Generates initial particle state on the CPU (same disc distribution as
    // main-optimized.js) and uploads it once. This is the only time particle
    // data crosses the CPU/GPU boundary in bulk; every subsequent frame it
    // stays resident in GPU buffers.
    initializeParticles() {
        const n = this.particleCount;
        const posX = new Float32Array(n);
        const posY = new Float32Array(n);
        const velX = new Float32Array(n);
        const velY = new Float32Array(n);
        const mass = new Float32Array(n);
        const alive = new Uint32Array(n).fill(1);

        const minRadius = 0.2;
        const maxRadius = DOMAIN_HALF_SIZE * 0.6;

        for (let i = 0; i < n; i++) {
            const radius = minRadius + Math.random() * (maxRadius - minRadius);
            const theta = Math.random() * Math.PI * 2;

            posX[i] = radius * Math.cos(theta);
            posY[i] = radius * Math.sin(theta);
            mass[i] = 0.6 + Math.random() * 0.8;

            const enclosedMassEstimate = n * 0.6 * (radius / maxRadius);
            const speed = Math.sqrt(this.gravityStrength * FORCE_SCALE * enclosedMassEstimate / radius) * 0.3;

            velX[i] = -speed * Math.sin(theta);
            velY[i] =  speed * Math.cos(theta);
        }

        const dev = this.device;
        dev.queue.writeBuffer(this.posXBuf, 0, posX);
        dev.queue.writeBuffer(this.posYBuf, 0, posY);
        dev.queue.writeBuffer(this.velXBuf, 0, velX);
        dev.queue.writeBuffer(this.velYBuf, 0, velY);
        dev.queue.writeBuffer(this.massBuf, 0, mass);
        dev.queue.writeBuffer(this.aliveBuf, 0, alive);
        dev.queue.writeBuffer(this.aliveCounterBuf, 0, new Int32Array([n]));

        this.aliveCount = n;
    }

    setParticleCount(count) {
        this.particleCount = Math.floor(Math.min(count, MAX_PARTICLES));
        this.initializeParticles();
    }

    setGravityStrength(strength) { this.gravityStrength = strength; }

    setTimeScale(scale) { this.timeScale = scale; }

    setBoundaryMode(mode) {
        this.boundaryMode = mode;
        this.writeLevelParams(); // refresh the wrap flag baked into level uniforms
        this.initializeParticles();
    }

    resetSimulation() { this.initializeParticles(); }

    // ---- Per-frame rendering ------------------------------------------------

    updateQuadVertexBuffer() {
        const aspect = this.canvas.width / this.canvas.height;
        const vertexData = computeCoverQuadVertices(aspect);
        this.device.queue.writeBuffer(this.quadVertexBuffer, 0, vertexData);
    }

    // Non-blocking readback of the single atomic alive-particle counter, so
    // the "Active Particles" stat stays reasonably fresh in 'delete' mode
    // without stalling the frame (only issued every N frames, and skipped
    // entirely while a previous readback is still in flight).
    maybeReadbackAliveCount() {
        this.framesSinceAliveReadback++;
        if (this.aliveReadbackPending || this.framesSinceAliveReadback < ALIVE_READBACK_INTERVAL_FRAMES) {
            return;
        }
        this.framesSinceAliveReadback = 0;
        this.aliveReadbackPending = true;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.aliveCounterBuf, 0, this.aliveCounterStagingBuf, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        this.aliveCounterStagingBuf.mapAsync(GPUMapMode.READ).then(() => {
            const value = new Int32Array(this.aliveCounterStagingBuf.getMappedRange())[0];
            this.aliveCount = value;
            this.aliveCounterStagingBuf.unmap();
            this.aliveReadbackPending = false;
        }).catch(() => {
            this.aliveReadbackPending = false;
        });
    }

    render(time) {
        const computeStart = performance.now();

        this.writeSimParams();

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();

        const baseCellCount = this.levels[0].dim * this.levels[0].dim;
        const particleDispatch = dispatchCount(this.particleCount);

        pass.setPipeline(this.clearPipeline);
        pass.setBindGroup(0, this.clearBindGroup);
        pass.dispatchWorkgroups(dispatchCount(baseCellCount));

        pass.setPipeline(this.scatterPipeline);
        pass.setBindGroup(0, this.scatterBindGroup);
        pass.dispatchWorkgroups(particleDispatch);

        pass.setPipeline(this.resolvePipeline);
        pass.setBindGroup(0, this.resolveBindGroup);
        pass.dispatchWorkgroups(dispatchCount(baseCellCount));

        pass.setPipeline(this.downsamplePipeline);
        for (let i = 1; i < this.levels.length; i++) {
            const cellCount = this.levels[i].dim * this.levels[i].dim;
            pass.setBindGroup(0, this.downsampleBindGroups[i - 1]);
            pass.dispatchWorkgroups(dispatchCount(cellCount));
        }

        // NOTE: blurTempBuf is shared/reused across levels as scratch space
        // for the horizontal->vertical blur handoff. Each level's vertical
        // pass MUST run immediately after that level's horizontal pass —
        // otherwise a later level's horizontal dispatch overwrites the
        // buffer before the earlier level's vertical pass has read it,
        // reinterpreting a smaller level's raw floats as if they belonged
        // to a larger level's row layout (this caused the reported
        // deterministic "pinch point" artifacts in the lower ~quarter of
        // the domain, aliased at power-of-two column fractions like 0.25/0.75).
        for (let i = 0; i < this.levels.length; i++) {
            const cellCount = this.levels[i].dim * this.levels[i].dim;
            const d = dispatchCount(cellCount);

            pass.setPipeline(this.blurHPipeline);
            pass.setBindGroup(0, this.blurHBindGroups[i]);
            pass.dispatchWorkgroups(d);

            pass.setPipeline(this.blurVPipeline);
            pass.setBindGroup(0, this.blurVBindGroups[i]);
            pass.dispatchWorkgroups(d);
        }

        pass.setPipeline(this.gradientPipeline);
        for (let i = 0; i < this.levels.length; i++) {
            const cellCount = this.levels[i].dim * this.levels[i].dim;
            pass.setBindGroup(0, this.gradientBindGroups[i]);
            pass.dispatchWorkgroups(dispatchCount(cellCount));
        }

        pass.setPipeline(this.accumulatePipeline);
        for (let i = 0; i < this.levels.length; i++) {
            pass.setBindGroup(0, this.accumulateBindGroups[i]);
            pass.dispatchWorkgroups(particleDispatch);
        }

        pass.setPipeline(this.integratePipeline);
        pass.setBindGroup(0, this.integrateBindGroup);
        pass.dispatchWorkgroups(particleDispatch);

        pass.setPipeline(this.colorizePipeline);
        pass.setBindGroup(0, this.colorizeBindGroup);
        pass.dispatchWorkgroups(dispatchCount(baseCellCount));

        pass.end();

        const computeEnd = performance.now();
        this.computeTime = computeEnd - computeStart;

        this.updateQuadVertexBuffer();

        const renderStart = performance.now();

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.setVertexBuffer(0, this.quadVertexBuffer);
        renderPass.draw(6);
        renderPass.end();

        this.device.queue.submit([encoder.finish()]);
        const renderEnd = performance.now();
        this.renderTime = renderEnd - renderStart;

        this.maybeReadbackAliveCount();

        this.frameCount++;
        if (time - this.lastTime > 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (time - this.lastTime));
            this.frameCount = 0;
            this.lastTime = time;
        }
    }
}

// App wiring for index-optimized-gpu.html controls (same DOM ids as the CPU
// version's index.html so both pages/scripts are interchangeable).
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

            this.simulation = new GpuGridNBodySimulation(canvas);
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

export { App, GpuGridNBodySimulation };

if (typeof document !== 'undefined') {
    const app = new App();
    app.init();
}
