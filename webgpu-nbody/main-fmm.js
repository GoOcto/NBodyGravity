import { CameraController } from './camera.js';
import { mat4, vec3 } from './gl-matrix.js';

const WORKGROUP_SIZE = 64;
const PARTICLE_FLOATS = 8;
const PARTICLE_BYTES = PARTICLE_FLOATS * 4;
const BOUNDARY_SIZE = 50;
const OCTREE_DEPTH = 4;
const OCTREE_THETA = 0.65;
const OCTREE_SOFTENING = 0.01;
const CPU_DISTANCE_EPSILON = 0.01;
const OCTREE_LEVEL_OFFSETS = [0, 1, 9, 73, 585];
const OCTREE_LEAF_COUNT = 1 << (OCTREE_DEPTH * 3);
const OCTREE_NODE_COUNT = OCTREE_LEVEL_OFFSETS[OCTREE_DEPTH] + OCTREE_LEAF_COUNT;

class SimpleNBodySimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;
        this.canvasFormat = null;

        this.particleCount = 500;
        this.gravityStrength = 1.0;
        this.timeScale = 1.0;
        this.damping = 0.999;
        this.mode = 'cpu';
        this.switchingMode = false;
        this.reconfiguring = false;
        this.onGpuError = null;

        // This is the canonical state used by the CPU path and for lossless mode switches.
        this.particleData = new Float32Array(0);
        this.cpuAccel = new Float32Array(0);

        this.particleBuffer = null;
        this.forceBuffer = null;
        this.nodeBuffer = null;
        this.particleLeafBuffer = null;
        this.paramBuffer = null;
        this.readbackBuffer = null;
        this.aggregateLevelBuffers = [];

        this.clearPipeline = null;
        this.leafPipeline = null;
        this.leafMultipolePipeline = null;
        this.aggregatePipeline = null;
        this.fmmForcePipeline = null;
        this.fmmIntegratePipeline = null;
        this.renderPipeline = null;

        this.clearBindGroup = null;
        this.leafBindGroup = null;
        this.leafMultipoleBindGroup = null;
        this.aggregateBindGroups = [];
        this.fmmForceBindGroup = null;
        this.fmmIntegrateBindGroup = null;
        this.renderBindGroup = null;

        this.camera = {
            position: vec3.fromValues(0, 0, 100),
            target: vec3.fromValues(0, 0, 0),
            up: vec3.fromValues(0, 1, 0),
            fovy: Math.PI / 4,
            aspect: 1,
            near: 0.1,
            far: 1000
        };
        this.cameraController = null;

        this.frameCount = 0;
        this.lastTime = 0;
        this.fps = 0;
        this.computeTime = 0;
        this.renderTime = 0;
        this.gpuFrameInFlight = false;
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported by this browser.');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter is available.');
        }

        this.device = await adapter.requestDevice();
        this.device.addEventListener('uncapturederror', (event) => {
            const error = event.error || new Error('Uncaptured WebGPU error.');
            console.error('WebGPU error:', error);
            if (this.onGpuError) this.onGpuError(error);
        });
        this.device.lost.then((info) => {
            if (info.reason !== 'destroyed') {
                console.error(`WebGPU device lost: ${info.message}`);
            }
        });

        this.context = this.canvas.getContext('webgpu');
        if (!this.context) {
            throw new Error('Unable to create a WebGPU canvas context.');
        }
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
        });

        this.camera.aspect = this.canvas.width / this.canvas.height;
        this.initializeParticles();
        await this.initPipelines();
        this.createResources();
        this.createBindGroups();
        this.cameraController = new CameraController(this.camera, this.canvas);
    }

    async loadShader(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Unable to load shader ${url} (${response.status}).`);
        }
        const code = await response.text();
        const module = this.device.createShaderModule({ code });
        if (module.getCompilationInfo) {
            const info = await module.getCompilationInfo();
            const errors = info.messages.filter((message) => message.type === 'error');
            if (errors.length > 0) {
                throw new Error(`${url}: ${errors.map((message) => message.message).join('; ')}`);
            }
        }
        return module;
    }

    async initPipelines() {
        const [
            clearShader,
            leafShader,
            leafMultipoleShader,
            aggregateShader,
            fmmForceShader,
            fmmIntegrateShader,
            vertexShader,
            fragmentShader,
        ] = await Promise.all([
            this.loadShader('./shaders/octree-clear.wgsl'),
            this.loadShader('./shaders/octree-leaf.wgsl'),
            this.loadShader('./shaders/octree-leaf-multipole.wgsl'),
            this.loadShader('./shaders/octree-aggregate.wgsl'),
            this.loadShader('./shaders/fmm-force-compute.wgsl'),
            this.loadShader('./shaders/fmm-integrate-compute.wgsl'),
            this.loadShader('./shaders/particle-vertex-quad.wgsl'),
            this.loadShader('./shaders/particle-fragment-quad.wgsl'),
        ]);

        this.clearPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: clearShader, entryPoint: 'clearNodes' },
        });
        this.leafPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: leafShader, entryPoint: 'assignLeaves' },
        });
        this.leafMultipolePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: leafMultipoleShader, entryPoint: 'buildLeafMultipoles' },
        });
        this.aggregatePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: aggregateShader, entryPoint: 'aggregateLevel' },
        });
        this.fmmForcePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: fmmForceShader, entryPoint: 'computeFmmForces' },
        });
        this.fmmIntegratePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: fmmIntegrateShader, entryPoint: 'integrateFmmParticles' },
        });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: vertexShader,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: fragmentShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.canvasFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                    },
                }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    initializeParticles() {
        this.particleData = new Float32Array(this.particleCount * PARTICLE_FLOATS);
        this.cpuAccel = new Float32Array(this.particleCount * 3);

        for (let i = 0; i < this.particleCount; i++) {
            const offset = i * PARTICLE_FLOATS;
            const radius = Math.random() * 20 + 5;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);

            this.particleData[offset] = radius * Math.sin(phi) * Math.cos(theta);
            this.particleData[offset + 1] = radius * Math.sin(phi) * Math.sin(theta);
            this.particleData[offset + 2] = radius * Math.cos(phi);
            this.particleData[offset + 3] = 1.0; //Math.random() * 0.5 + 0.5;

            const speed = Math.sqrt(this.gravityStrength * 100 / radius) * 0.3;
            this.particleData[offset + 4] = -speed * Math.sin(theta);
            this.particleData[offset + 5] = speed * Math.cos(theta);
            this.particleData[offset + 6] = 0;
            this.particleData[offset + 7] = 0;
        }

        if (this.particleCount > 0) {
            this.particleData[0] = 0;
            this.particleData[1] = 0;
            this.particleData[2] = 0;
            this.particleData[3] = 2;
            this.particleData[4] = 0;
            this.particleData[5] = 0;
            this.particleData[6] = 0;
            this.particleData[7] = 0;
        }
    }

    createResources() {
        const particleBytes = Math.max(PARTICLE_BYTES, this.particleData.byteLength);
        this.particleBuffer = this.device.createBuffer({
            size: particleBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.forceBuffer = this.device.createBuffer({
            size: Math.max(16, this.particleCount * 16),
            usage: GPUBufferUsage.STORAGE,
        });
        this.nodeBuffer = this.device.createBuffer({
            size: OCTREE_NODE_COUNT * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.particleLeafBuffer = this.device.createBuffer({
            size: Math.max(4, this.particleCount * 4),
            usage: GPUBufferUsage.STORAGE,
        });
        this.paramBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.readbackBuffer = this.device.createBuffer({
            size: particleBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        for (let level = 0; level < OCTREE_DEPTH; level++) {
            this.aggregateLevelBuffers[level] = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const levelData = new Uint32Array([
                level,
                OCTREE_LEVEL_OFFSETS[level],
                8 ** level,
                0,
            ]);
            this.device.queue.writeBuffer(this.aggregateLevelBuffers[level], 0, levelData);
        }

        this.device.queue.writeBuffer(this.nodeBuffer, 0, this.createNodeBounds());
        this.device.queue.writeBuffer(this.particleBuffer, 0, this.particleData);
    }

    createNodeBounds() {
        const bounds = new Float32Array(OCTREE_NODE_COUNT * 8);
        for (let level = 0; level <= OCTREE_DEPTH; level++) {
            const count = 8 ** level;
            const start = OCTREE_LEVEL_OFFSETS[level];
            const halfSize = BOUNDARY_SIZE / (2 ** level);
            for (let local = 0; local < count; local++) {
                let x = 0;
                let y = 0;
                let z = 0;
                for (let bit = level - 1; bit >= 0; bit--) {
                    const octant = (local >> (bit * 3)) & 7;
                    const childHalf = BOUNDARY_SIZE / (2 ** (bit + 1));
                    x += (octant & 1) ? childHalf : -childHalf;
                    y += (octant & 2) ? childHalf : -childHalf;
                    z += (octant & 4) ? childHalf : -childHalf;
                }
                const offset = (start + local) * 8;
                bounds[offset] = x;
                bounds[offset + 1] = y;
                bounds[offset + 2] = z;
                bounds[offset + 3] = halfSize;
            }
        }
        return bounds;
    }

    createBindGroups() {
        this.clearBindGroup = this.device.createBindGroup({
            layout: this.clearPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.nodeBuffer } }],
        });
        this.leafBindGroup = this.device.createBindGroup({
            layout: this.leafPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.particleLeafBuffer } },
                { binding: 2, resource: { buffer: this.paramBuffer } },
            ],
        });
        this.leafMultipoleBindGroup = this.device.createBindGroup({
            layout: this.leafMultipolePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.particleLeafBuffer } },
                { binding: 2, resource: { buffer: this.nodeBuffer } },
                { binding: 3, resource: { buffer: this.paramBuffer } },
            ],
        });
        this.aggregateBindGroups = this.aggregateLevelBuffers.map((levelBuffer) => this.device.createBindGroup({
            layout: this.aggregatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.nodeBuffer } },
                { binding: 1, resource: { buffer: levelBuffer } },
            ],
        }));
        this.fmmForceBindGroup = this.device.createBindGroup({
            layout: this.fmmForcePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.particleLeafBuffer } },
                { binding: 2, resource: { buffer: this.nodeBuffer } },
                { binding: 3, resource: { buffer: this.forceBuffer } },
                { binding: 4, resource: { buffer: this.paramBuffer } },
            ],
        });
        this.fmmIntegrateBindGroup = this.device.createBindGroup({
            layout: this.fmmIntegratePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.forceBuffer } },
                { binding: 2, resource: { buffer: this.paramBuffer } },
            ],
        });
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer || this.createUniformBuffer() } },
            ],
        });
    }

    createUniformBuffer() {
        this.uniformBuffer = this.device.createBuffer({
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        return this.uniformBuffer;
    }

    writeParams(deltaTime) {
        const data = new ArrayBuffer(32);
        const view = new DataView(data);
        view.setUint32(0, this.particleCount, true);
        view.setUint32(4, OCTREE_NODE_COUNT, true);
        view.setFloat32(8, deltaTime * this.timeScale, true);
        view.setFloat32(12, this.gravityStrength, true);
        view.setFloat32(16, this.damping, true);
        view.setFloat32(20, OCTREE_THETA, true);
        view.setFloat32(24, OCTREE_SOFTENING, true);
        view.setUint32(28, OCTREE_DEPTH, true);
        this.device.queue.writeBuffer(this.paramBuffer, 0, data);
    }

    updateUniforms(time) {
        const viewMatrix = mat4.create();
        const projMatrix = mat4.create();
        const viewProjMatrix = mat4.create();
        mat4.lookAt(viewMatrix, this.camera.position, this.camera.target, this.camera.up);
        mat4.perspective(projMatrix, this.camera.fovy, this.camera.aspect, this.camera.near, this.camera.far);
        mat4.multiply(viewProjMatrix, projMatrix, viewMatrix);

        const uniformData = new Float32Array(36);
        uniformData.set(viewProjMatrix, 0);
        uniformData.set(viewMatrix, 16);
        uniformData.set(this.camera.position, 32);
        uniformData[35] = time;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    updateCpuPhysics(deltaTime) {
        const dt = deltaTime * this.timeScale;
        //this.cpuForces.fill(0);
		this.cpuAccel.fill(0);

        for (let i = 0; i < this.particleCount; i++) {
            const iOffset = i * PARTICLE_FLOATS;
            const ix = this.particleData[iOffset];
            const iy = this.particleData[iOffset + 1];
            const iz = this.particleData[iOffset + 2];
            const im = this.particleData[iOffset + 3];
            for (let j = 0; j < this.particleCount; j++) {
                if (i === j) continue;
                const jOffset = j * PARTICLE_FLOATS;
                const dx = this.particleData[jOffset] - ix;
                const dy = this.particleData[jOffset + 1] - iy;
                const dz = this.particleData[jOffset + 2] - iz;
                const distanceSquared = dx * dx + dy * dy + dz * dz + CPU_DISTANCE_EPSILON;
                const distance = Math.sqrt(distanceSquared);
                // const force = this.gravityStrength * im * this.particleData[jOffset + 3] / distanceSquared;
                const accel = this.gravityStrength * this.particleData[jOffset + 3] / distanceSquared;
                const forceOffset = i * 3;
                this.cpuAccel[forceOffset] += accel * dx / distance;
                this.cpuAccel[forceOffset + 1] += accel * dy / distance;
                this.cpuAccel[forceOffset + 2] += accel * dz / distance;
            }
        }

        for (let i = 0; i < this.particleCount; i++) {
            const offset = i * PARTICLE_FLOATS;
            const forceOffset = i * 3;
            const mass = Math.max(this.particleData[offset + 3], 0.0001);
            this.particleData[offset + 4] += this.cpuAccel[forceOffset] * dt;
            this.particleData[offset + 5] += this.cpuAccel[forceOffset + 1] * dt;
            this.particleData[offset + 6] += this.cpuAccel[forceOffset + 2] * dt;
            this.particleData[offset + 4] *= this.damping;
            this.particleData[offset + 5] *= this.damping;
            this.particleData[offset + 6] *= this.damping;
            this.particleData[offset] += this.particleData[offset + 4] * dt;
            this.particleData[offset + 1] += this.particleData[offset + 5] * dt;
            this.particleData[offset + 2] += this.particleData[offset + 6] * dt;
            this.applyBoundary(offset);
        }
        this.device.queue.writeBuffer(this.particleBuffer, 0, this.particleData);
    }

    applyBoundary(offset) {
        for (let axis = 0; axis < 3; axis++) {
            if (Math.abs(this.particleData[offset + axis]) > BOUNDARY_SIZE) {
                this.particleData[offset + axis] = Math.sign(this.particleData[offset + axis]) * BOUNDARY_SIZE;
                this.particleData[offset + 4 + axis] *= -0.8;
            }
        }
    }

    encodeFmmCompute(commandEncoder) {
        const clearPass = commandEncoder.beginComputePass();
        clearPass.setPipeline(this.clearPipeline);
        clearPass.setBindGroup(0, this.clearBindGroup);
        clearPass.dispatchWorkgroups(Math.ceil(OCTREE_NODE_COUNT / WORKGROUP_SIZE));
        clearPass.end();

        const leafPass = commandEncoder.beginComputePass();
        leafPass.setPipeline(this.leafPipeline);
        leafPass.setBindGroup(0, this.leafBindGroup);
        leafPass.dispatchWorkgroups(Math.ceil(this.particleCount / WORKGROUP_SIZE));
        leafPass.end();

        const multipolePass = commandEncoder.beginComputePass();
        multipolePass.setPipeline(this.leafMultipolePipeline);
        multipolePass.setBindGroup(0, this.leafMultipoleBindGroup);
        multipolePass.dispatchWorkgroups(Math.ceil(OCTREE_LEAF_COUNT / WORKGROUP_SIZE));
        multipolePass.end();

        for (let level = OCTREE_DEPTH - 1; level >= 0; level--) {
            const aggregatePass = commandEncoder.beginComputePass();
            aggregatePass.setPipeline(this.aggregatePipeline);
            aggregatePass.setBindGroup(0, this.aggregateBindGroups[level]);
            aggregatePass.dispatchWorkgroups(Math.ceil((8 ** level) / WORKGROUP_SIZE));
            aggregatePass.end();
        }

        const forcePass = commandEncoder.beginComputePass();
        forcePass.setPipeline(this.fmmForcePipeline);
        forcePass.setBindGroup(0, this.fmmForceBindGroup);
        forcePass.dispatchWorkgroups(Math.ceil(this.particleCount / WORKGROUP_SIZE));
        forcePass.end();

        const integratePass = commandEncoder.beginComputePass();
        integratePass.setPipeline(this.fmmIntegratePipeline);
        integratePass.setBindGroup(0, this.fmmIntegrateBindGroup);
        integratePass.dispatchWorkgroups(Math.ceil(this.particleCount / WORKGROUP_SIZE));
        integratePass.end();
    }

    render(time) {
        if (!this.cameraController || this.switchingMode || this.reconfiguring) return;
        if (this.mode === 'gpu' && this.gpuFrameInFlight) return;
        this.cameraController.update();
        const computeStart = performance.now();
        this.updateUniforms(time);
        const commandEncoder = this.device.createCommandEncoder();

        if (this.mode === 'cpu') {
            this.updateCpuPhysics(0.016);
        } else if (this.mode === 'gpu') {
            this.writeParams(0.016);
            this.encodeFmmCompute(commandEncoder);
        } else {
            throw new Error(`Unknown simulation mode: ${this.mode}`);
        }

        const computeEnd = performance.now();
        this.computeTime = computeEnd - computeStart;

        const renderStart = performance.now();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(this.particleCount * 6);
        renderPass.end();
        const commandBuffer = commandEncoder.finish();
        if (this.mode === 'gpu') {
            this.gpuFrameInFlight = true;
            try {
                this.device.queue.submit([commandBuffer]);
                this.device.queue.onSubmittedWorkDone().then(
                    () => {
                        this.gpuFrameInFlight = false;
                    },
                    (error) => {
                        this.gpuFrameInFlight = false;
                        console.error('GPU frame completion failed:', error);
                        if (this.onGpuError) this.onGpuError(error);
                    }
                );
            } catch (error) {
                this.gpuFrameInFlight = false;
                throw error;
            }
        } else {
            this.device.queue.submit([commandBuffer]);
        }
        this.renderTime = performance.now() - renderStart;

        this.frameCount++;
        if (time - this.lastTime > 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (time - this.lastTime));
            this.frameCount = 0;
            this.lastTime = time;
        }
    }

    async syncGpuToCpu() {
        await this.device.queue.onSubmittedWorkDone();
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.particleBuffer, 0, this.readbackBuffer, 0, this.particleData.byteLength);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.readbackBuffer.mapAsync(GPUMapMode.READ);
        const mapped = this.readbackBuffer.getMappedRange();
        this.particleData = new Float32Array(mapped.slice(0));
        this.readbackBuffer.unmap();
    }

    async setMode(mode) {
        if (mode !== 'cpu' && mode !== 'gpu') {
            throw new Error(`Unsupported simulation mode: ${mode}`);
        }
        if (mode === this.mode) return;

        this.switchingMode = true;
        try {
            if (mode === 'cpu') {
                await this.syncGpuToCpu();
            } else {
                if (!this.fmmForcePipeline || !this.nodeBuffer) {
                    throw new Error('GPU FMM resources are not available; CPU mode remains active.');
                }
                this.device.queue.writeBuffer(this.particleBuffer, 0, this.particleData);
            }
            this.mode = mode;
        } finally {
            this.switchingMode = false;
        }
    }

    async setParticleCount(count) {
        if (this.reconfiguring) return;
        this.reconfiguring = true;
        try {
            if (this.mode === 'gpu') await this.syncGpuToCpu();
            this.particleCount = Math.floor(Math.max(100, Math.min(count, 4000)));
            this.initializeParticles();
            this.createResources();
            this.createBindGroups();
        } finally {
            this.reconfiguring = false;
        }
    }

    setGravityStrength(strength) { this.gravityStrength = strength; }
    setTimeScale(scale) { this.timeScale = scale; }
    setDamping(damping) { this.damping = damping; }

    resetSimulation() {
        this.initializeParticles();
        this.device.queue.writeBuffer(this.particleBuffer, 0, this.particleData);
    }
}

class App {
    constructor() {
        this.simulation = null;
        this.animationId = null;
    }

    async init() {
        const canvas = document.getElementById('canvas');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const errorMessage = document.getElementById('errorMessage');
        const controls = document.getElementById('controls');

        try {
            this.resizeCanvas(canvas);
            window.addEventListener('resize', () => this.resizeCanvas(canvas));
            this.simulation = new SimpleNBodySimulation(canvas);
            await this.simulation.init();
            loading.style.display = 'none';
            controls.style.display = 'block';
            this.setupControls();
            this.animate();
        } catch (err) {
            console.error('Failed to initialize:', err);
            loading.style.display = 'none';
            errorMessage.textContent = err instanceof Error ? err.message : String(err);
            error.style.display = 'block';
        }
    }

    resizeCanvas(canvas) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * devicePixelRatio;
        canvas.height = rect.height * devicePixelRatio;
        if (this.simulation) this.simulation.camera.aspect = canvas.width / canvas.height;
    }

    setupControls() {
        const modeSelect = document.getElementById('simulationMode');
        const modeStatus = document.getElementById('modeStatus');
        const particleCountSlider = document.getElementById('particleCount');
        const gravitySlider = document.getElementById('gravity');
        const timeScaleSlider = document.getElementById('timeScale');
        const dampingSlider = document.getElementById('damping');
        const resetBtn = document.getElementById('resetBtn');

        const particleCountValue = document.getElementById('particleCountValue');
        const gravityValue = document.getElementById('gravityValue');
        const timeScaleValue = document.getElementById('timeScaleValue');
        const dampingValue = document.getElementById('dampingValue');

        this.simulation.onGpuError = (error) => {
            modeStatus.textContent = `WebGPU error: ${error.message || error}`;
            modeStatus.style.color = '#ff6b6b';
        };

        particleCountSlider.value = this.simulation.particleCount;
        particleCountValue.textContent = this.simulation.particleCount;
        modeSelect.value = this.simulation.mode;

        modeSelect.addEventListener('change', async (event) => {
            const requestedMode = event.target.value;
            modeStatus.textContent = `Switching to ${requestedMode === 'gpu' ? 'GPU FMM octree' : 'CPU direct (O(N²))'}…`;
            modeStatus.style.color = '#ffd166';
            try {
                await this.simulation.setMode(requestedMode);
                modeStatus.textContent = requestedMode === 'gpu' ? 'GPU FMM octree' : 'CPU direct (O(N²))';
                modeStatus.style.color = '#9be7a5';
            } catch (err) {
                modeSelect.value = this.simulation.mode;
                modeStatus.textContent = `Mode switch failed: ${err.message || err}`;
                modeStatus.style.color = '#ff6b6b';
            }
        });

        particleCountSlider.addEventListener('input', (event) => {
            const value = parseInt(event.target.value, 10);
            particleCountValue.textContent = value;
            this.simulation.setParticleCount(value).catch((err) => {
                modeStatus.textContent = `Particle resize failed: ${err.message || err}`;
                modeStatus.style.color = '#ff6b6b';
            });
        });
        gravitySlider.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            gravityValue.textContent = value.toFixed(1);
            this.simulation.setGravityStrength(value);
        });
        timeScaleSlider.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            timeScaleValue.textContent = value.toFixed(1);
            this.simulation.setTimeScale(value);
        });
        dampingSlider.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            dampingValue.textContent = value.toFixed(3);
            this.simulation.setDamping(value);
        });
        resetBtn.addEventListener('click', () => this.simulation.resetSimulation());
    }

    animate() {
        const time = performance.now();
        if (this.simulation) {
            this.simulation.render(time);
            document.getElementById('fps').textContent = this.simulation.fps;
            document.getElementById('computeTime').textContent = this.simulation.computeTime.toFixed(3);
            document.getElementById('renderTime').textContent = this.simulation.renderTime.toFixed(3);
        }
        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

const app = new App();
app.init();
