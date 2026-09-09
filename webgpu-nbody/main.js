// Simplified N-Body simulation without compute shaders
// Uses JavaScript for physics to avoid GPU driver crashes

import { CameraController } from './camera.js';
import { mat4, vec3 } from './gl-matrix.js';

class SimpleNBodySimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;

        // Simulation parameters
        this.particleCount = 500; // Reduced for CPU simulation
        this.gravityStrength = 1.0;
        this.timeScale = 1.0;
        this.damping = 0.999;

        // Particle data (CPU-based)
        this.particles = [];

        // WebGPU resources (render only)
        this.particleBuffer = null;
        this.uniformBuffer = null;

        // Render pipeline
        this.renderPipeline = null;
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

        this.cameraController = null;

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

        this.camera.aspect = this.canvas.width / this.canvas.height;

        this.initializeParticles();
        await this.initResources();
        await this.initRenderPipeline();

        this.cameraController = new CameraController(this.camera, this.canvas);
    }

    initializeParticles() {
        this.particles = [];

        for (let i = 0; i < this.particleCount; i++) {
            // Position (random distribution in a sphere)
            const radius = Math.random() * 20 + 5;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);

            const particle = {
                position: [
                    radius * Math.sin(phi) * Math.cos(theta),
                    radius * Math.sin(phi) * Math.sin(theta),
                    radius * Math.cos(phi)
                ],
                mass: Math.random() * 0.5 + 0.5,
                velocity: [0, 0, 0]
            };

            // Initial orbital velocity
            const speed = Math.sqrt(this.gravityStrength * 100 / radius) * 0.3;
            particle.velocity[0] = -speed * Math.sin(theta);
            particle.velocity[1] = speed * Math.cos(theta);
            particle.velocity[2] = 0;

            this.particles.push(particle);
        }

        // Add one particle at center for reference
        if (this.particles.length > 0) {
            this.particles[0].position = [0, 0, 0];
            this.particles[0].velocity = [0, 0, 0];
            this.particles[0].mass = 2.0;
        }
    }

    async initResources() {
        // Create particle buffer for rendering only
        const particleBufferSize = this.particleCount * 8 * 4;
        this.particleBuffer = this.device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Create uniform buffer for camera matrices
        this.uniformBuffer = this.device.createBuffer({
            size: 144, // viewProj (64) + view (64) + cameraPos (12) + time (4)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.updateParticleBuffer();
    }

    async loadShader(url) {
        const response = await fetch(url);
        return await response.text();
    }

    async initRenderPipeline() {
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
                topology: 'triangle-list',
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

    updatePhysics(deltaTime) {
        const dt = deltaTime * this.timeScale;

        // Calculate forces (CPU-based)
        const forces = [];
        for (let i = 0; i < this.particleCount; i++) {
            forces[i] = [0, 0, 0];

            const particle = this.particles[i];

            for (let j = 0; j < this.particleCount; j++) {
                if (i === j) continue;

                const other = this.particles[j];
                const dx = other.position[0] - particle.position[0];
                const dy = other.position[1] - particle.position[1];
                const dz = other.position[2] - particle.position[2];

                const distanceSquared = dx * dx + dy * dy + dz * dz + 0.01; // epsilon
                const distance = Math.sqrt(distanceSquared);

                const force = this.gravityStrength * particle.mass * other.mass / distanceSquared;

                forces[i][0] += force * dx / distance;
                forces[i][1] += force * dy / distance;
                forces[i][2] += force * dz / distance;
            }
        }

        // Integrate particles
        for (let i = 0; i < this.particleCount; i++) {
            const particle = this.particles[i];
            const force = forces[i];

            // Update velocity
            particle.velocity[0] += (force[0] / particle.mass) * dt;
            particle.velocity[1] += (force[1] / particle.mass) * dt;
            particle.velocity[2] += (force[2] / particle.mass) * dt;

            // Apply damping
            particle.velocity[0] *= this.damping;
            particle.velocity[1] *= this.damping;
            particle.velocity[2] *= this.damping;

            // Update position
            particle.position[0] += particle.velocity[0] * dt;
            particle.position[1] += particle.velocity[1] * dt;
            particle.position[2] += particle.velocity[2] * dt;

            // Boundary conditions
            const boundarySize = 50.0;
            const restitution = 0.8;

            if (Math.abs(particle.position[0]) > boundarySize) {
                particle.position[0] = Math.sign(particle.position[0]) * boundarySize;
                particle.velocity[0] *= -restitution;
            }
            if (Math.abs(particle.position[1]) > boundarySize) {
                particle.position[1] = Math.sign(particle.position[1]) * boundarySize;
                particle.velocity[1] *= -restitution;
            }
            if (Math.abs(particle.position[2]) > boundarySize) {
                particle.position[2] = Math.sign(particle.position[2]) * boundarySize;
                particle.velocity[2] *= -restitution;
            }
        }
    }

    updateParticleBuffer() {
        const particleData = new Float32Array(this.particleCount * 8);

        for (let i = 0; i < this.particleCount; i++) {
            const offset = i * 8;
            const particle = this.particles[i];

            particleData[offset + 0] = particle.position[0];
            particleData[offset + 1] = particle.position[1];
            particleData[offset + 2] = particle.position[2];
            particleData[offset + 3] = particle.mass;
            particleData[offset + 4] = particle.velocity[0];
            particleData[offset + 5] = particle.velocity[1];
            particleData[offset + 6] = particle.velocity[2];
            particleData[offset + 7] = 0; // padding
        }

        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);
    }

    updateUniforms(time) {
        const viewMatrix = mat4.create();
        const projMatrix = mat4.create();
        const viewProjMatrix = mat4.create();

        mat4.lookAt(viewMatrix, this.camera.position, this.camera.target, this.camera.up);
        mat4.perspective(projMatrix, this.camera.fovy, this.camera.aspect, this.camera.near, this.camera.far);
        mat4.multiply(viewProjMatrix, projMatrix, viewMatrix);

        const uniformData = new Float32Array(36); // 16 + 16 + 3 + 1 = 36 floats
        uniformData.set(viewProjMatrix, 0);      // 16 floats (64 bytes)
        uniformData.set(viewMatrix, 16);         // 16 floats (64 bytes) 
        uniformData.set(this.camera.position, 32); // 3 floats (12 bytes)
        uniformData[35] = time;                  // 1 float (4 bytes)

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    render(time) {
        if (this.cameraController) {
            this.cameraController.update();
        }

        const computeStart = performance.now();
        this.updatePhysics(0.016); // Fixed timestep
        this.updateParticleBuffer();
        const computeEnd = performance.now();
        this.computeTime = computeEnd - computeStart;

        this.updateUniforms(time);

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

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(this.particleCount * 6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        const renderEnd = performance.now();
        this.renderTime = renderEnd - renderStart;

        // Debug: Log timing occasionally
        if (this.frameCount % 60 === 0) {
            console.log(`Compute: ${this.computeTime.toFixed(3)}ms, Render: ${this.renderTime.toFixed(3)}ms, Particles: ${this.particleCount}`);
        }

        // Update performance stats
        this.frameCount++;
        if (time - this.lastTime > 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (time - this.lastTime));
            this.frameCount = 0;
            this.lastTime = time;
        }
    }

    setParticleCount(count) {
        this.particleCount = Math.floor(Math.min(count, 4000)); // Cap at 4000 for CPU
        this.initializeParticles();
        this.initResources().then(() => {
            this.initRenderPipeline();
        });
    }

    setGravityStrength(strength) { this.gravityStrength = strength; }
    setTimeScale(scale) { this.timeScale = scale; }
    setDamping(damping) { this.damping = damping; }
    resetSimulation() { this.initializeParticles(); }
}

// Use the stable version
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

            // Use stable CPU-based simulation
            this.simulation = new SimpleNBodySimulation(canvas);
            await this.simulation.init();

            loading.style.display = 'none';
            controls.style.display = 'block';

            this.setupControls();
            this.animate();

        } catch (err) {
            console.error('Failed to initialize:', err);
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

        // Limit particle count for CPU simulation
        particleCountSlider.max = 4000;
        particleCountSlider.value = 500;
        particleCountValue.textContent = 500;

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

            document.getElementById('fps').textContent = this.simulation.fps;
            document.getElementById('computeTime').textContent = this.simulation.computeTime.toFixed(3);
            document.getElementById('renderTime').textContent = this.simulation.renderTime.toFixed(3);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

// Start the application
const app = new App();
app.init();