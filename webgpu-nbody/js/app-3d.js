// App shell for index.html — 3D N-Body Gravity Simulation.
// Wires the shared control panel to SimpleNBodySimulation (js/sim-3d.js),
// which itself owns the CPU direct O(N^2) / GPU FMM octree backend switch.
import {
    SimpleNBodySimulation, PARTICLE_COUNT_RANGE, DEFAULT_PARTICLE_COUNT,
    OCTREE_THETA_RANGE, OCTREE_SOFTENING_RANGE, DEFAULT_OCTREE_THETA, DEFAULT_OCTREE_SOFTENING,
} from './sim-3d.js';
import { bindRange, bindSelect, applySliderRange } from './common.js';

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
            this.applyInitialControlValues();
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

    // Sets the slider min/max/default from sim-3d.js's shared ranges before
    // the simulation is constructed, so the markup never needs to hardcode
    // (and risk drifting from) these values.
    applyInitialControlValues() {
        applySliderRange(
            document.getElementById('particleCount'), document.getElementById('particleCountValue'),
            PARTICLE_COUNT_RANGE, DEFAULT_PARTICLE_COUNT,
        );
        applySliderRange(
            document.getElementById('octreeTheta'), document.getElementById('octreeThetaValue'),
            OCTREE_THETA_RANGE, DEFAULT_OCTREE_THETA, (v) => v.toFixed(2),
        );
        applySliderRange(
            document.getElementById('octreeSoftening'), document.getElementById('octreeSofteningValue'),
            OCTREE_SOFTENING_RANGE, DEFAULT_OCTREE_SOFTENING, (v) => v.toFixed(3),
        );
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
        const gpuOnlyControls = document.querySelectorAll('[data-gpu-only]');

        const updateGpuOnlyControlsEnabled = () => {
            const gpuActive = this.simulation.mode === 'gpu';
            gpuOnlyControls.forEach((el) => { el.disabled = !gpuActive; });
        };

        this.simulation.onGpuError = (err) => {
            modeStatus.textContent = `WebGPU error: ${err.message || err}`;
            modeStatus.style.color = '#ff6b6b';
        };

        modeSelect.value = this.simulation.mode;
        updateGpuOnlyControlsEnabled();

        bindSelect(modeSelect, async (requestedMode) => {
            modeStatus.textContent = `Switching to ${requestedMode === 'gpu' ? 'GPU FMM octree' : 'CPU direct (O(N²))'}…`;
            modeStatus.style.color = '#ffd166';
            try {
                await this.simulation.setMode(requestedMode);
                modeStatus.textContent = requestedMode === 'gpu' ? 'GPU FMM octree' : 'CPU direct (O(N²))';
                modeStatus.style.color = '#9be7a5';
                updateGpuOnlyControlsEnabled();
            } catch (err) {
                modeSelect.value = this.simulation.mode;
                modeStatus.textContent = `Mode switch failed: ${err.message || err}`;
                modeStatus.style.color = '#ff6b6b';
            }
        });

        bindRange(document.getElementById('particleCount'), document.getElementById('particleCountValue'), {
            parse: (v) => parseInt(v, 10),
            onInput: (value) => {
                this.simulation.setParticleCount(value).catch((err) => {
                    modeStatus.textContent = `Particle resize failed: ${err.message || err}`;
                    modeStatus.style.color = '#ff6b6b';
                });
            },
        });

        bindRange(document.getElementById('gravity'), document.getElementById('gravityValue'), {
            format: (v) => v.toFixed(1),
            onInput: (value) => this.simulation.setGravityStrength(value),
        });

        bindRange(document.getElementById('timeScale'), document.getElementById('timeScaleValue'), {
            format: (v) => v.toFixed(1),
            onInput: (value) => this.simulation.setTimeScale(value),
        });

        bindRange(document.getElementById('damping'), document.getElementById('dampingValue'), {
            format: (v) => v.toFixed(3),
            onInput: (value) => this.simulation.setDamping(value),
        });

        bindRange(document.getElementById('octreeTheta'), document.getElementById('octreeThetaValue'), {
            format: (v) => v.toFixed(2),
            onInput: (value) => this.simulation.setOctreeTheta(value),
        });

        bindRange(document.getElementById('octreeSoftening'), document.getElementById('octreeSofteningValue'), {
            format: (v) => v.toFixed(3),
            onInput: (value) => this.simulation.setOctreeSoftening(value),
        });

        document.getElementById('resetBtn').addEventListener('click', () => this.simulation.resetSimulation());
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
