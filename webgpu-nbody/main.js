// WebGPU N-Body Gravity Simulation
// Modern compute shader implementation

import { CameraController } from './camera.js';
import { mat4, vec3 } from './gl-matrix.js';

class NBodySimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;

        // Simulation parameters
        this.particleCount = 1000;
        this.gravityStrength = 1.0;
        this.timeScale = 1.0;
        this.damping = 0.999;

        // WebGPU resources
        this.particleBuffer = null;
        this.forceBuffer = null;
        this.uniformBuffer = null;
        this.paramBuffer = null;

        // Compute pipelines
        this.forceComputePipeline = null;
        this.integrateComputePipeline = null;

        // Render pipeline
        this.renderPipeline = null;

        // Bind groups
        this.forceBindGroup = null;
        this.integrateBindGroup = null;
        this.renderBindGroup = null;

        // Camera
        this.camera = {
            position: vec3.fromValues(0, 0, 100),
            target: vec3.fromValues(0, 0, 0),
            up: vec3.fromValues(0, 1, 0),
            fovy: Math.PI / 4,
            aspect: 1,
            near: 0.1,
            far: 1000
        };

        // Camera controller
        this.cameraController = null;

        // Performance tracking
        this.frameCount = 0;
        this.lastTime = 0;
        this.fps = 0;
        this.computeTime = 0;
        this.renderTime = 0;
    }

    async init() {
        // Initialize WebGPU
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

        // Update camera aspect ratio
        this.camera.aspect = this.canvas.width / this.canvas.height;

        await this.initResources();
        await this.initComputePipelines();
        await this.initRenderPipeline();

        this.initializeParticles();

        // Initialize camera controller
        this.cameraController = new CameraController(this.camera, this.canvas);
    }

    async loadShader(url) {
        const response = await fetch(url);
        return await response.text();
    }

    async initResources() {
        // Create particle buffer (position, mass, velocity)
        const particleBufferSize = this.particleCount * 8 * 4; // 8 floats per particle, 4 bytes per float
        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Create force buffer
        const forceBufferSize = this.particleCount * 3 * 4; // 3 floats per force vector
        this.forceBuffer = this.device.createBuffer({
            size: forceBufferSize,
            usage: GPUBufferUsage.STORAGE,
        });

        // Create uniform buffer for camera matrices
        this.uniformBuffer = this.device.createBuffer({
            size: 80, // mat4 (64 bytes) + vec3 (12 bytes) + float (4 bytes)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create parameter buffer for simulation parameters
        this.paramBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    async initComputePipelines() {
        // Load compute shaders
        const forceShader = await this.loadShader('./shaders/force-compute.wgsl');
        const integrateShader = await this.loadShader('./shaders/integrate-compute.wgsl');

        // Create compute pipelines
        this.forceComputePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: forceShader }),
                entryPoint: 'computeForces',
            },
        });

        this.integrateComputePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: integrateShader }),
                entryPoint: 'integrateParticles',
            },
        });

        // Create bind groups for compute shaders
        this.forceBindGroup = this.device.createBindGroup({
            layout: this.forceComputePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.forceBuffer } },
                { binding: 2, resource: { buffer: this.paramBuffer } },
            ],
        });

        this.integrateBindGroup = this.device.createBindGroup({
            layout: this.integrateComputePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.forceBuffer } },
                { binding: 2, resource: { buffer: this.paramBuffer } },
            ],
        });
    }

    async initRenderPipeline() {
        // Use quad-based rendering instead of points
        const vertexShader = await this.loadShader('./shaders/particle-vertex-quad.wgsl');
        const fragmentShader = await this.loadShader('./shaders/particle-fragment-quad.wgsl');

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: vertexShader }),
                entryPoint: 'vs_main',
            },
            fragment: {
                module: this.device.createShaderModule({ code: fragmentShader }),
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-list', // Changed from point-list to triangle-list
            },
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });
    }

    initializeParticles() {
        const particles = new Float32Array(this.particleCount * 8);

        for (let i = 0; i < this.particleCount; i++) {
            const offset = i * 8;

            // Position (tighter distribution for better visibility)
            const radius = Math.random() * 20 + 5;  // Reduced range
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);

            particles[offset + 0] = radius * Math.sin(phi) * Math.cos(theta); // x
            particles[offset + 1] = radius * Math.sin(phi) * Math.sin(theta); // y
            particles[offset + 2] = radius * Math.cos(phi);                   // z
            particles[offset + 3] = Math.random() * 0.5 + 0.5; // mass

            // Velocity (initial orbital velocity)
            const speed = Math.sqrt(this.gravityStrength * 100 / radius) * 0.3;
            particles[offset + 4] = -speed * Math.sin(theta); // vx
            particles[offset + 5] = speed * Math.cos(theta);  // vy
            particles[offset + 6] = 0; // vz
            particles[offset + 7] = 0; // padding
        }

        // Add a few test particles at known positions for debugging
        if (this.particleCount > 0) {
            particles[0] = 0;   // x: center
            particles[1] = 0;   // y: center  
            particles[2] = 0;   // z: center
            particles[3] = 1.0; // mass
            particles[4] = 0;   // vx
            particles[5] = 0;   // vy
            particles[6] = 0;   // vz
            particles[7] = 0;   // padding
        }

        this.device.queue.writeBuffer(this.particleBuffer, 0, particles);
    }

    updateUniforms(time) {
        // Update camera matrices
        const viewMatrix = mat4.create();
        const projMatrix = mat4.create();
        const viewProjMatrix = mat4.create();

        mat4.lookAt(viewMatrix, this.camera.position, this.camera.target, this.camera.up);
        mat4.perspective(projMatrix, this.camera.fovy, this.camera.aspect, this.camera.near, this.camera.far);
        mat4.multiply(viewProjMatrix, projMatrix, viewMatrix);

        const uniformData = new Float32Array(20);
        uniformData.set(viewProjMatrix, 0);
        uniformData.set(this.camera.position, 16);
        uniformData[19] = time;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // Update simulation parameters
        const paramData = new Float32Array([
            this.particleCount,
            0.016 * this.timeScale, // deltaTime (assuming 60 FPS)
            this.gravityStrength,
            this.damping
        ]);

        this.device.queue.writeBuffer(this.paramBuffer, 0, paramData);
    }

    render(time) {
        // Update camera controller
        if (this.cameraController) {
            this.cameraController.update();
        }

        this.updateUniforms(time);

        const commandEncoder = this.device.createCommandEncoder();

        // Compute pass - calculate forces
        const computePass1 = commandEncoder.beginComputePass();
        computePass1.setPipeline(this.forceComputePipeline);
        computePass1.setBindGroup(0, this.forceBindGroup);
        computePass1.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
        computePass1.end();

        // Compute pass - integrate particles
        const computePass2 = commandEncoder.beginComputePass();
        computePass2.setPipeline(this.integrateComputePipeline);
        computePass2.setBindGroup(0, this.integrateBindGroup);
        computePass2.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
        computePass2.end();

        // Render pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(this.particleCount * 6); // 6 vertices per particle quad
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        // Update performance stats
        this.frameCount++;
        if (time - this.lastTime > 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (time - this.lastTime));
            this.frameCount = 0;
            this.lastTime = time;
        }
    }

    setParticleCount(count) {
        this.particleCount = Math.floor(count);
        // Re-initialize resources with new particle count
        this.initResources().then(() => {
            this.initComputePipelines().then(() => {
                this.initRenderPipeline().then(() => {
                    this.initializeParticles();
                });
            });
        });
    }

    setGravityStrength(strength) {
        this.gravityStrength = strength;
    }

    setTimeScale(scale) {
        this.timeScale = scale;
    }

    setDamping(damping) {
        this.damping = damping;
    }

    resetSimulation() {
        this.initializeParticles();
    }
}

// Main application
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
            // Resize canvas
            this.resizeCanvas(canvas);
            window.addEventListener('resize', () => this.resizeCanvas(canvas));

            // Initialize simulation
            this.simulation = new NBodySimulation(canvas);
            await this.simulation.init();

            // Hide loading, show controls
            loading.style.display = 'none';
            controls.style.display = 'block';

            // Setup controls
            this.setupControls();

            // Start render loop
            this.animate();

        } catch (err) {
            console.error('Failed to initialize WebGPU:', err);
            loading.style.display = 'none';
            error.style.display = 'block';
        }
    }

    resizeCanvas(canvas) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * devicePixelRatio;
        canvas.height = rect.height * devicePixelRatio;

        if (this.simulation) {
            this.simulation.camera.aspect = canvas.width / canvas.height;
        }
    }

    setupControls() {
        const particleCountSlider = document.getElementById('particleCount');
        const gravitySlider = document.getElementById('gravity');
        const timeScaleSlider = document.getElementById('timeScale');
        const dampingSlider = document.getElementById('damping');
        const resetBtn = document.getElementById('resetBtn');

        const particleCountValue = document.getElementById('particleCountValue');
        const gravityValue = document.getElementById('gravityValue');
        const timeScaleValue = document.getElementById('timeScaleValue');
        const dampingValue = document.getElementById('dampingValue');

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

        timeScaleSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            timeScaleValue.textContent = value.toFixed(1);
            this.simulation.setTimeScale(value);
        });

        dampingSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            dampingValue.textContent = value.toFixed(3);
            this.simulation.setDamping(value);
        });

        resetBtn.addEventListener('click', () => {
            this.simulation.resetSimulation();
        });
    }

    animate() {
        const time = performance.now();

        if (this.simulation) {
            this.simulation.render(time);

            // Update stats
            document.getElementById('fps').textContent = this.simulation.fps;
            document.getElementById('computeTime').textContent = this.simulation.computeTime.toFixed(2);
            document.getElementById('renderTime').textContent = this.simulation.renderTime.toFixed(2);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

// Start the application
const app = new App();
app.init();