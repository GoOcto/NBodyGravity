// Particle-Mesh (PM) 2D N-body gravity simulation — GPU compute variant.
// Unlike main-optimized-gpu.js's multi-resolution grid + separable-blur
// approximation (2D-optimized.md), this is a textbook particle-mesh
// method:
//   1. Cloud-in-Cell (CIC) mass deposit (shaders/gpu/pm-mass-scatter-cic.wgsl)
//      bilinearly spreads each particle's mass across its 4 surrounding
//      cells, then mass-resolve.wgsl converts the atomic fixed-point grid
//      to plain f32 (both the atomic trick and mass-resolve.wgsl are
//      shared, unmodified, with main-optimized-gpu.js).
//   2. A single dim x dim 2D FFT (pm-fft-complexify / pm-fft-bitreverse /
//      pm-fft-butterfly / pm-fft-transpose.wgsl) transforms the mass grid
//      into the frequency domain.
//   3. The true 2D Poisson equation is solved exactly, per Fourier mode,
//      by multiplying by the Green's function 1/|k|² (pm-poisson-greens.wgsl),
//      zeroing the undefined k=0 (mean density) term — the standard
//      periodic-Poisson trick.
//   4. An inverse 2D FFT (same primitives) transforms back to a real
//      pseudo-potential field (pm-potential-extract.wgsl).
//   5. gradient.wgsl and accumulate-gradient.wgsl (shared, unmodified)
//      compute and bilinearly sample the field's gradient at each
//      particle position — matching interpolation with the CIC deposit,
//      avoiding self-force artifacts.
//   6. pm-integrate.wgsl integrates with NO velocity damping (unlike
//      integrate.wgsl's damping multiply).
//   7. colorize.wgsl (shared, unmodified) writes the heatmap texture.
//
// Steps 1-6 run once per FIXED_DT-sized physics sub-step; a real-time
// accumulator decides how many sub-steps to run per rendered frame (see
// render() below), so simulation stability never depends on the
// timeScale UI control — only on how many stable, fixed-size steps are
// taken.
//
// Because the FFT Poisson solve is inherently periodic, gravity always
// wraps at the domain edge (mass near one edge gravitationally
// interacts with the opposite edge) regardless of which particle
// boundaryMode (bounce/wrap/delete) is selected — that setting only
// controls what happens to a particle's own position/velocity when it
// reaches the edge, not the field solve itself.

const MAX_PARTICLES = 4000000;
const DOMAIN_HALF_SIZE = 50;
const DOMAIN_SIZE = 2 * DOMAIN_HALF_SIZE;
const GRID_DIM = 1024; // must be a power of two
const LOG_GRID_DIM = Math.log2(GRID_DIM);
const CELL_SIZE = DOMAIN_SIZE / GRID_DIM;
const CELL_COUNT = GRID_DIM * GRID_DIM;

// Overall tuning constant folding in the Poisson equation's 2πG constant,
// cell-area normalization, etc. — empirically tuned (see README) rather
// than derived, same spirit as main-optimized-gpu.js's FORCE_SCALE. Unlike
// the blur pipeline's bounded, normalized kernel, the FFT Poisson solve's
// raw gradient magnitude scales with total mass and the domain's physical
// size, so this constant is far smaller than the blur demo's.
const FORCE_SCALE = 0.15;

const RESTITUTION = 1;
// Fixed internal integration step, in simulation-time seconds. This NEVER
// scales with timeScale (see stepsForFrame) — that's the whole point of
// decoupling stability from playback speed.
const FIXED_DT = 0.016;
// Upper bound on how many FIXED_DT sub-steps run per rendered frame. At
// extreme timeScale values (or a slow/backgrounded tab), playback falls
// behind the requested multiplier rather than taking one giant unstable
// step or pegging the GPU with an unbounded number of steps.
const MAX_SUBSTEPS_PER_FRAME = 4;

const MASS_VISUAL_SCALE = 2.0;
// Fixed-point scale for the atomic CIC mass-deposit pass (WGSL has no
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
// aspect ratio. Identical to the CPU/blur-GPU versions' helper of the same
// name.
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

class PmGpuNBodySimulation {
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

        // Fixed-timestep sub-stepping state (see stepsForFrame).
        this.timeAccumulator = 0;
        this.lastFrameTimeMs = null;
        this.lastSubsteps = 0;

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
        this.writeStaticParams();
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
        // GPU resources.
        // COPY_SRC is included (unlike velX/velY/mass/alive) so tooling can
        // read back particle positions for validation/debugging without
        // needing to recreate the buffers.
        this.posXBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST | COPY_SRC });
        this.posYBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST | COPY_SRC });
        this.velXBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.velYBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.massBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.aliveBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE | COPY_DST });
        this.accelXBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE });
        this.accelYBuf = dev.createBuffer({ size: MAX_PARTICLES * 4, usage: STORAGE });

        this.aliveCounterBuf = dev.createBuffer({ size: 4, usage: STORAGE | COPY_DST | COPY_SRC });
        this.aliveCounterStagingBuf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | COPY_DST });

        // Single-grid buffers (no LOD hierarchy).
        this.atomicMassBuf = dev.createBuffer({ size: CELL_COUNT * 4, usage: STORAGE });
        this.rawMassBuf = dev.createBuffer({ size: CELL_COUNT * 4, usage: STORAGE });
        this.fieldBuf = dev.createBuffer({ size: CELL_COUNT * 4, usage: STORAGE });
        this.gradXBuf = dev.createBuffer({ size: CELL_COUNT * 4, usage: STORAGE });
        this.gradYBuf = dev.createBuffer({ size: CELL_COUNT * 4, usage: STORAGE });

        // Complex (re, im) FFT ping-pong buffers — vec2<f32> per cell (8
        // bytes). Buffer roles are fixed throughout the whole forward+
        // Poisson+inverse pipeline (see encodeRowFft/encodePhysicsStep):
        // bit-reversal always reads fftBufA and writes fftBufB; butterfly
        // stages always operate in-place on fftBufB; transpose always
        // reads fftBufB and writes fftBufA. This lets every stage reuse a
        // single pair of bind groups across all 4 row-FFT applications
        // (forward-x, forward-y, inverse-x, inverse-y).
        this.fftBufA = dev.createBuffer({ size: CELL_COUNT * 8, usage: STORAGE });
        this.fftBufB = dev.createBuffer({ size: CELL_COUNT * 8, usage: STORAGE });

        // Uniforms.
        this.simParamsBuf = dev.createBuffer({ size: 48, usage: UNIFORM | COPY_DST });
        this.levelParamsBuf = dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
        this.fftDimParamsBuf = dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
        this.greensParamsBuf = dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
        this.extractParamsBuf = dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
        // One small uniform buffer per FFT stage/direction (log2(dim)
        // forward + log2(dim) inverse). Written once at startup and never
        // again — using per-stage buffers (rather than repeatedly
        // rewriting one shared buffer with queue.writeBuffer) avoids the
        // classic WebGPU pitfall where all writeBuffer calls made before a
        // single queue.submit() land before any of that submission's
        // dispatches execute, which would leave every stage seeing only
        // the last-written values.
        this.fftStageParamsBufsForward = [];
        this.fftStageParamsBufsInverse = [];
        for (let s = 0; s < LOG_GRID_DIM; s++) {
            this.fftStageParamsBufsForward.push(dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST }));
            this.fftStageParamsBufsInverse.push(dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST }));
        }

        // Heatmap texture: written directly by colorize.wgsl (storage
        // texture), sampled directly by the render pass (texture binding).
        this.gridTexture = dev.createTexture({
            size: [GRID_DIM, GRID_DIM],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gridTextureView = this.gridTexture.createView();
        this.gridSampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        this.quadVertexBuffer = dev.createBuffer({ size: 6 * 4 * 4, usage: VERTEX | COPY_DST });
    }

    // Writes every uniform buffer whose contents never change after
    // startup: LevelParams (wrap is always 1 — periodic — because the FFT
    // Poisson solve is inherently periodic, independent of boundaryMode),
    // the FFT dimension params, the per-stage FFT params, the Green's
    // function params, and the potential-extract normalization params.
    writeStaticParams() {
        const dev = this.device;

        {
            const data = new ArrayBuffer(16);
            new Uint32Array(data, 0, 1)[0] = GRID_DIM;
            new Float32Array(data, 4, 1)[0] = CELL_SIZE;
            new Uint32Array(data, 8, 1)[0] = 1; // wrap: always periodic
            new Uint32Array(data, 12, 1)[0] = 1; // isFirstLevel: only one level
            dev.queue.writeBuffer(this.levelParamsBuf, 0, data);
        }

        {
            const data = new ArrayBuffer(16);
            new Uint32Array(data, 0, 1)[0] = GRID_DIM;
            new Uint32Array(data, 4, 1)[0] = LOG_GRID_DIM;
            dev.queue.writeBuffer(this.fftDimParamsBuf, 0, data);
        }

        for (let stage = 0; stage < LOG_GRID_DIM; stage++) {
            const len = 1 << (stage + 1);
            const half = len >> 1;

            const fwd = new ArrayBuffer(16);
            new Uint32Array(fwd, 0, 1)[0] = GRID_DIM;
            new Uint32Array(fwd, 4, 1)[0] = half;
            new Uint32Array(fwd, 8, 1)[0] = len;
            new Float32Array(fwd, 12, 1)[0] = -1.0;
            dev.queue.writeBuffer(this.fftStageParamsBufsForward[stage], 0, fwd);

            const inv = new ArrayBuffer(16);
            new Uint32Array(inv, 0, 1)[0] = GRID_DIM;
            new Uint32Array(inv, 4, 1)[0] = half;
            new Uint32Array(inv, 8, 1)[0] = len;
            new Float32Array(inv, 12, 1)[0] = 1.0;
            dev.queue.writeBuffer(this.fftStageParamsBufsInverse[stage], 0, inv);
        }

        {
            const data = new ArrayBuffer(16);
            new Uint32Array(data, 0, 1)[0] = GRID_DIM;
            new Float32Array(data, 4, 1)[0] = DOMAIN_SIZE;
            dev.queue.writeBuffer(this.greensParamsBuf, 0, data);
        }

        {
            const data = new ArrayBuffer(16);
            new Uint32Array(data, 0, 1)[0] = GRID_DIM;
            new Float32Array(data, 4, 1)[0] = 1.0 / CELL_COUNT;
            dev.queue.writeBuffer(this.extractParamsBuf, 0, data);
        }
    }

    // Writes the per-frame SimParams uniform (forceMultiplier/dt depend on
    // the caller-supplied dt, which is always FIXED_DT — see
    // stepSimulation). The unused `damping` slot exists only so this
    // buffer's byte layout matches what mass-resolve.wgsl,
    // accumulate-gradient.wgsl, and colorize.wgsl (all shared, unmodified,
    // with main-optimized-gpu.js) expect.
    writeSimParams(dt) {
        const data = new ArrayBuffer(48);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32[0] = this.gravityStrength * FORCE_SCALE; // forceMultiplier
        f32[1] = 1.0; // damping: unused by pm-integrate.wgsl
        f32[2] = dt;
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
            clearSrc, scatterCicSrc, resolveSrc,
            complexifySrc, bitreverseSrc, butterflySrc, transposeSrc,
            greensSrc, extractSrc,
            gradientSrc, accumSrc, integrateSrc, initVelocitySrc, colorizeSrc,
            heatmapVertexSrc, heatmapFragmentSrc,
        ] = await Promise.all([
            this.loadShader('./shaders/gpu/mass-clear.wgsl'),
            this.loadShader('./shaders/gpu/pm-mass-scatter-cic.wgsl'),
            this.loadShader('./shaders/gpu/mass-resolve.wgsl'),
            this.loadShader('./shaders/gpu/pm-fft-complexify.wgsl'),
            this.loadShader('./shaders/gpu/pm-fft-bitreverse.wgsl'),
            this.loadShader('./shaders/gpu/pm-fft-butterfly.wgsl'),
            this.loadShader('./shaders/gpu/pm-fft-transpose.wgsl'),
            this.loadShader('./shaders/gpu/pm-poisson-greens.wgsl'),
            this.loadShader('./shaders/gpu/pm-potential-extract.wgsl'),
            this.loadShader('./shaders/gpu/gradient.wgsl'),
            this.loadShader('./shaders/gpu/accumulate-gradient.wgsl'),
            this.loadShader('./shaders/gpu/pm-integrate.wgsl'),
            this.loadShader('./shaders/gpu/pm-init-circular-velocity.wgsl'),
            this.loadShader('./shaders/gpu/colorize.wgsl'),
            this.loadShader('./shaders/grid-heatmap-vertex.wgsl'),
            this.loadShader('./shaders/grid-heatmap-fragment.wgsl'),
        ]);

        const makeComputePipeline = (code) => dev.createComputePipeline({
            layout: 'auto',
            compute: { module: dev.createShaderModule({ code }), entryPoint: 'cs_main' },
        });

        this.clearPipeline = makeComputePipeline(clearSrc);
        this.scatterCicPipeline = makeComputePipeline(scatterCicSrc);
        this.resolvePipeline = makeComputePipeline(resolveSrc);
        this.complexifyPipeline = makeComputePipeline(complexifySrc);
        this.bitreversePipeline = makeComputePipeline(bitreverseSrc);
        this.butterflyPipeline = makeComputePipeline(butterflySrc);
        this.transposePipeline = makeComputePipeline(transposeSrc);
        this.greensPipeline = makeComputePipeline(greensSrc);
        this.extractPipeline = makeComputePipeline(extractSrc);
        this.gradientPipeline = makeComputePipeline(gradientSrc);
        this.accumulatePipeline = makeComputePipeline(accumSrc);
        this.integratePipeline = makeComputePipeline(integrateSrc);
        this.initVelocityPipeline = makeComputePipeline(initVelocitySrc);
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

        this.clearBindGroup = dev.createBindGroup({
            layout: this.clearPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.levelParamsBuf } },
                { binding: 1, resource: { buffer: this.atomicMassBuf } },
            ],
        });

        this.scatterBindGroup = dev.createBindGroup({
            layout: this.scatterCicPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: this.levelParamsBuf } },
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
                { binding: 1, resource: { buffer: this.levelParamsBuf } },
                { binding: 2, resource: { buffer: this.atomicMassBuf } },
                { binding: 3, resource: { buffer: this.rawMassBuf } },
            ],
        });

        this.complexifyBindGroup = dev.createBindGroup({
            layout: this.complexifyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.fftDimParamsBuf } },
                { binding: 1, resource: { buffer: this.rawMassBuf } },
                { binding: 2, resource: { buffer: this.fftBufA } },
            ],
        });

        // Fixed buffer roles across the whole pipeline (see createBuffers'
        // comment): bit-reversal always A->B, transpose always B->A. Both
        // bind groups are reused for every one of the 4 row-FFT
        // applications (forward-x, forward-y, inverse-x, inverse-y).
        this.bitreverseBindGroup = dev.createBindGroup({
            layout: this.bitreversePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.fftDimParamsBuf } },
                { binding: 1, resource: { buffer: this.fftBufA } },
                { binding: 2, resource: { buffer: this.fftBufB } },
            ],
        });
        this.transposeBindGroup = dev.createBindGroup({
            layout: this.transposePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.fftDimParamsBuf } },
                { binding: 1, resource: { buffer: this.fftBufB } },
                { binding: 2, resource: { buffer: this.fftBufA } },
            ],
        });

        this.butterflyBindGroupsForward = this.fftStageParamsBufsForward.map((buf) => dev.createBindGroup({
            layout: this.butterflyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: buf } },
                { binding: 1, resource: { buffer: this.fftBufB } },
            ],
        }));
        this.butterflyBindGroupsInverse = this.fftStageParamsBufsInverse.map((buf) => dev.createBindGroup({
            layout: this.butterflyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: buf } },
                { binding: 1, resource: { buffer: this.fftBufB } },
            ],
        }));

        this.greensBindGroup = dev.createBindGroup({
            layout: this.greensPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.greensParamsBuf } },
                { binding: 1, resource: { buffer: this.fftBufA } },
            ],
        });

        this.extractBindGroup = dev.createBindGroup({
            layout: this.extractPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.extractParamsBuf } },
                { binding: 1, resource: { buffer: this.fftBufA } },
                { binding: 2, resource: { buffer: this.fieldBuf } },
            ],
        });

        this.gradientBindGroup = dev.createBindGroup({
            layout: this.gradientPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.levelParamsBuf } },
                { binding: 1, resource: { buffer: this.fieldBuf } },
                { binding: 2, resource: { buffer: this.gradXBuf } },
                { binding: 3, resource: { buffer: this.gradYBuf } },
            ],
        });

        this.accumulateBindGroup = dev.createBindGroup({
            layout: this.accumulatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: this.levelParamsBuf } },
                { binding: 2, resource: { buffer: this.posXBuf } },
                { binding: 3, resource: { buffer: this.posYBuf } },
                { binding: 4, resource: { buffer: this.aliveBuf } },
                { binding: 5, resource: { buffer: this.gradXBuf } },
                { binding: 6, resource: { buffer: this.gradYBuf } },
                { binding: 7, resource: { buffer: this.accelXBuf } },
                { binding: 8, resource: { buffer: this.accelYBuf } },
            ],
        });

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

        this.initVelocityBindGroup = dev.createBindGroup({
            layout: this.initVelocityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: this.posXBuf } },
                { binding: 2, resource: { buffer: this.posYBuf } },
                { binding: 3, resource: { buffer: this.accelXBuf } },
                { binding: 4, resource: { buffer: this.accelYBuf } },
                { binding: 5, resource: { buffer: this.aliveBuf } },
                { binding: 6, resource: { buffer: this.velXBuf } },
                { binding: 7, resource: { buffer: this.velYBuf } },
            ],
        });

        this.colorizeBindGroup = dev.createBindGroup({
            layout: this.colorizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuf } },
                { binding: 1, resource: { buffer: this.levelParamsBuf } },
                { binding: 2, resource: { buffer: this.rawMassBuf } },
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

    // Generates initial particle state on the CPU (same disc distribution
    // approach as the other demos) and uploads it once.
    // Generates initial particle state on the CPU (same disc distribution
    // approach as the other demos): random positions/masses are uploaded
    // with zero velocity, then a one-time GPU field solve computes the
    // ACTUAL acceleration field for that distribution, from which
    // pm-init-circular-velocity.wgsl derives circular orbital velocities
    // directly on the GPU (see that shader's comment) — this is exact for
    // whatever mass profile the random disc actually has, rather than an
    // approximate analytic formula.
    initializeParticles() {
        const n = this.particleCount;
        const posX = new Float32Array(n);
        const posY = new Float32Array(n);
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
        }

        const dev = this.device;
        dev.queue.writeBuffer(this.posXBuf, 0, posX);
        dev.queue.writeBuffer(this.posYBuf, 0, posY);
        dev.queue.writeBuffer(this.velXBuf, 0, new Float32Array(n));
        dev.queue.writeBuffer(this.velYBuf, 0, new Float32Array(n));
        dev.queue.writeBuffer(this.massBuf, 0, mass);
        dev.queue.writeBuffer(this.aliveBuf, 0, alive);
        dev.queue.writeBuffer(this.aliveCounterBuf, 0, new Int32Array([n]));

        this.writeSimParams(FIXED_DT);
        const encoder = dev.createCommandEncoder();
        const pass = encoder.beginComputePass();
        this.encodeFieldSolve(pass);
        pass.setPipeline(this.initVelocityPipeline);
        pass.setBindGroup(0, this.initVelocityBindGroup);
        pass.dispatchWorkgroups(dispatchCount(n));
        pass.end();
        dev.queue.submit([encoder.finish()]);

        this.aliveCount = n;
        // Restart the sub-step accumulator so a reset doesn't inherit a
        // large backlog from before.
        this.timeAccumulator = 0;
    }

    setParticleCount(count) {
        this.particleCount = Math.floor(Math.min(count, MAX_PARTICLES));
        this.initializeParticles();
    }

    setGravityStrength(strength) { this.gravityStrength = strength; }

    setTimeScale(scale) { this.timeScale = scale; }

    setBoundaryMode(mode) {
        this.boundaryMode = mode;
        this.initializeParticles();
    }

    resetSimulation() { this.initializeParticles(); }

    // ---- Per-substep physics pipeline ---------------------------------------

    // Encodes one full "row FFT" application: per-row bit-reversal
    // permutation, log2(dim) in-place butterfly stages, then a transpose.
    // Buffer roles are fixed (see createBuffers), so this is identical
    // code whether it's being used for the "along x" or (post-transpose)
    // "along y" direction.
    encodeRowFft(pass, forward) {
        const cellDispatch = dispatchCount(CELL_COUNT);
        const pairDispatch = dispatchCount(CELL_COUNT / 2);
        const stageBindGroups = forward ? this.butterflyBindGroupsForward : this.butterflyBindGroupsInverse;

        pass.setPipeline(this.bitreversePipeline);
        pass.setBindGroup(0, this.bitreverseBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        pass.setPipeline(this.butterflyPipeline);
        for (let stage = 0; stage < LOG_GRID_DIM; stage++) {
            pass.setBindGroup(0, stageBindGroups[stage]);
            pass.dispatchWorkgroups(pairDispatch);
        }

        pass.setPipeline(this.transposePipeline);
        pass.setBindGroup(0, this.transposeBindGroup);
        pass.dispatchWorkgroups(cellDispatch);
    }

    // Encodes the field solve (CIC deposit -> FFT Poisson solve -> force
    // sampling), populating accelX/accelY for the CURRENT particle
    // positions, but without integrating. Used both by encodePhysicsStep
    // (immediately followed by integration) and by initializeParticles
    // (to compute the real acceleration field for a fresh disc, from
    // which pm-init-circular-velocity.wgsl derives circular velocities —
    // see that shader's comment).
    encodeFieldSolve(pass) {
        const cellDispatch = dispatchCount(CELL_COUNT);
        const particleDispatch = dispatchCount(this.particleCount);

        pass.setPipeline(this.clearPipeline);
        pass.setBindGroup(0, this.clearBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        pass.setPipeline(this.scatterCicPipeline);
        pass.setBindGroup(0, this.scatterBindGroup);
        pass.dispatchWorkgroups(particleDispatch);

        pass.setPipeline(this.resolvePipeline);
        pass.setBindGroup(0, this.resolveBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        pass.setPipeline(this.complexifyPipeline);
        pass.setBindGroup(0, this.complexifyBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        // Forward 2D FFT: rowFFT (along x) -> transpose -> rowFFT (along
        // the now-transposed y) -> transpose = full 2D FFT, landing back
        // in fftBufA (see createBuffers' fixed buffer-role comment).
        this.encodeRowFft(pass, true);
        this.encodeRowFft(pass, true);

        pass.setPipeline(this.greensPipeline);
        pass.setBindGroup(0, this.greensBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        // Inverse 2D FFT, same structure, landing back in fftBufA.
        this.encodeRowFft(pass, false);
        this.encodeRowFft(pass, false);

        pass.setPipeline(this.extractPipeline);
        pass.setBindGroup(0, this.extractBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        pass.setPipeline(this.gradientPipeline);
        pass.setBindGroup(0, this.gradientBindGroup);
        pass.dispatchWorkgroups(cellDispatch);

        pass.setPipeline(this.accumulatePipeline);
        pass.setBindGroup(0, this.accumulateBindGroup);
        pass.dispatchWorkgroups(particleDispatch);
    }

    // Encodes one full fixed-size physics step (field solve + integration)
    // into an already-open compute pass, using the SimParams uniform
    // already written for this dt.
    encodePhysicsStep(pass) {
        this.encodeFieldSolve(pass);

        pass.setPipeline(this.integratePipeline);
        pass.setBindGroup(0, this.integrateBindGroup);
        pass.dispatchWorkgroups(dispatchCount(this.particleCount));
    }

    // ---- Per-frame rendering ------------------------------------------------

    updateQuadVertexBuffer() {
        const aspect = this.canvas.width / this.canvas.height;
        const vertexData = computeCoverQuadVertices(aspect);
        this.device.queue.writeBuffer(this.quadVertexBuffer, 0, vertexData);
    }

    // Non-blocking readback of the single atomic alive-particle counter, so
    // the "Active Particles" stat stays reasonably fresh in 'delete' mode
    // without stalling the frame.
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

        // Fixed-timestep sub-stepping: real elapsed time (scaled by
        // timeScale) accumulates simulation-time debt, which is paid off
        // in FIXED_DT-sized, always-stable installments. timeScale only
        // changes how many steps run this frame, never their size.
        if (this.lastFrameTimeMs == null) {
            this.lastFrameTimeMs = time;
        }
        let realDtSeconds = (time - this.lastFrameTimeMs) / 1000;
        this.lastFrameTimeMs = time;
        // Clamp huge real-time gaps (e.g. a backgrounded tab) so resuming
        // doesn't demand an enormous number of catch-up steps.
        realDtSeconds = Math.min(realDtSeconds, 0.25);

        this.timeAccumulator += realDtSeconds * this.timeScale;
        const maxAccumulated = MAX_SUBSTEPS_PER_FRAME * FIXED_DT;
        if (this.timeAccumulator > maxAccumulated) {
            this.timeAccumulator = maxAccumulated;
        }

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();

        let substeps = 0;
        while (this.timeAccumulator >= FIXED_DT && substeps < MAX_SUBSTEPS_PER_FRAME) {
            this.writeSimParams(FIXED_DT);
            this.encodePhysicsStep(pass);
            this.timeAccumulator -= FIXED_DT;
            substeps++;
        }
        this.lastSubsteps = substeps;

        // Colorize once per rendered frame (using whichever sub-step ran
        // last), not once per sub-step.
        pass.setPipeline(this.colorizePipeline);
        pass.setBindGroup(0, this.colorizeBindGroup);
        pass.dispatchWorkgroups(dispatchCount(CELL_COUNT));

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

// App wiring for index-pm-gpu.html controls (same DOM ids as the other
// demos so pages/scripts are interchangeable).
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

            this.simulation = new PmGpuNBodySimulation(canvas);
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
        // starts up fast; sync it to whatever this page's slider markup
        // declares as its starting value before the render loop begins.
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
            const substepsEl = document.getElementById('substeps');
            if (substepsEl) substepsEl.textContent = this.simulation.lastSubsteps;
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

export { App, PmGpuNBodySimulation };

if (typeof document !== 'undefined') {
    const app = new App();
    app.init();
}
