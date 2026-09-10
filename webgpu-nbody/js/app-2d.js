// App shell for index-2d.html — 2D N-Body Gravity Simulation.
// Wires the shared control panel to one of three interchangeable
// algorithm classes (GridNBodySimulation / GpuGridNBodySimulation /
// PmGpuNBodySimulation). Unlike sim-3d.js's single class with a live
// CPU/GPU backend switch, these three algorithms are different enough
// (CPU vs. two very different GPU pipelines) that switching between them
// tears down the old simulation and constructs a fresh one, rather than
// transferring live GPU state — see the plan notes for why.
import { GridNBodySimulation, PARTICLE_COUNT_RANGE as CPU_GRID_RANGE, DEFAULT_PARTICLE_COUNT as CPU_GRID_DEFAULT } from './sim-grid-cpu.js';
import { GpuGridNBodySimulation, PARTICLE_COUNT_RANGE as GPU_GRID_RANGE, DEFAULT_PARTICLE_COUNT as GPU_GRID_DEFAULT } from './sim-grid-gpu.js';
import { PmGpuNBodySimulation, PARTICLE_COUNT_RANGE as PM_GPU_RANGE, DEFAULT_PARTICLE_COUNT as PM_GPU_DEFAULT } from './sim-pm-gpu.js';
import { bindRange, bindSelect, applySliderRange, formatTimeScale, DEFAULT_DAMPING, DEFAULT_RESTITUTION, DEFAULT_BOUNDARY_MODE, DEFAULT_ORBITAL_SPEED, ORBITAL_SPEED_RANGE } from './common.js';

// Registry describing each selectable algorithm: its class, its
// particle-count range/default (these differ a lot: CPU grid tops out at
// 1M, both GPU pipelines scale to 4M), and whether it supports the
// Damping control (PM-FFT's integrator has none, by design).
const ALGORITHMS = {
    'grid-cpu': {
        label: 'Grid-Based — CPU',
        Simulation: GridNBodySimulation,
        particleCountRange: CPU_GRID_RANGE,
        defaultParticleCount: CPU_GRID_DEFAULT,
        supportsDamping: true,
    },
    'grid-gpu': {
        label: 'Grid-Based — GPU Compute',
        Simulation: GpuGridNBodySimulation,
        particleCountRange: GPU_GRID_RANGE,
        defaultParticleCount: GPU_GRID_DEFAULT,
        supportsDamping: true,
    },
    'pm-gpu': {
        label: 'Particle-Mesh — GPU FFT Poisson',
        Simulation: PmGpuNBodySimulation,
        particleCountRange: PM_GPU_RANGE,
        defaultParticleCount: PM_GPU_DEFAULT,
        supportsDamping: false,
    },
};

class App {
    constructor() {
        this.simulation = null;
        this.algorithmKey = null;
        this.animationId = null;
        this.canvas = null;
    }

    async init() {
        this.canvas = document.getElementById('canvas');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const errorMessage = document.getElementById('errorMessage');
        const controls = document.getElementById('controls');

        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        document.getElementById('damping').value = DEFAULT_DAMPING;
        document.getElementById('dampingValue').textContent = DEFAULT_DAMPING.toFixed(3);
        document.getElementById('restitution').value = DEFAULT_RESTITUTION;
        document.getElementById('restitutionValue').textContent = DEFAULT_RESTITUTION.toFixed(2);
        applySliderRange(
            document.getElementById('orbitalSpeed'), document.getElementById('orbitalSpeedValue'),
            ORBITAL_SPEED_RANGE, DEFAULT_ORBITAL_SPEED, (v) => v.toFixed(2),
        );
        document.getElementById('boundaryMode').value = DEFAULT_BOUNDARY_MODE;

        this.setupStaticControls();

        try {
            await this.switchAlgorithm(document.getElementById('algorithm').value, { firstLoad: true });
            loading.style.display = 'none';
            controls.style.display = 'block';
            this.animate();
        } catch (err) {
            console.error('Failed to initialize:', err);
            loading.style.display = 'none';
            errorMessage.textContent = err instanceof Error ? err.message : String(err);
            error.style.display = 'block';
        }
    }

    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
    }

    // Tears down the current simulation (if any) and constructs+initializes
    // the requested algorithm, applying every current control-panel value
    // to it so switching algorithms feels seamless even though the
    // underlying object is brand new.
    async switchAlgorithm(key, { firstLoad = false } = {}) {
        const algorithmStatus = document.getElementById('algorithmStatus');
        const config = ALGORITHMS[key];
        if (!config) throw new Error(`Unknown algorithm: ${key}`);

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        algorithmStatus.textContent = `Loading ${config.label}…`;
        algorithmStatus.style.color = '#ffd166';

        const particleCountSlider = document.getElementById('particleCount');
        const particleCountValueEl = document.getElementById('particleCountValue');
        // Preserve the user's particle count across a switch when it fits
        // within the new algorithm's range; otherwise fall back to that
        // algorithm's own default (e.g. switching from GPU's 4M range down
        // to CPU grid's 1M cap).
        const previousParticleCount = parseInt(particleCountSlider.value, 10);
        applySliderRange(particleCountSlider, particleCountValueEl, config.particleCountRange, config.defaultParticleCount);
        const requestedParticleCount = firstLoad
            ? config.defaultParticleCount
            : Math.min(previousParticleCount, config.particleCountRange.max);
        particleCountSlider.value = requestedParticleCount;
        particleCountValueEl.textContent = requestedParticleCount;

        const dampingSlider = document.getElementById('damping');
        dampingSlider.disabled = !config.supportsDamping;
        document.getElementById('dampingNote').style.display = config.supportsDamping ? 'none' : 'block';

        const simulation = new config.Simulation(this.canvas);
        await simulation.init();

        simulation.setParticleCount(requestedParticleCount);
        simulation.setGravityStrength(parseFloat(document.getElementById('gravity').value));
        const exponent = parseFloat(document.getElementById('timeScale').value);
        simulation.setTimeScale(Math.pow(10, exponent));
        simulation.setDamping(parseFloat(dampingSlider.value));
        simulation.setRestitution(parseFloat(document.getElementById('restitution').value));
        simulation.setOrbitalSpeed(parseFloat(document.getElementById('orbitalSpeed').value));
        simulation.setBoundaryMode(document.getElementById('boundaryMode').value);

        this.simulation = simulation;
        this.algorithmKey = key;

        const substepsRow = document.getElementById('substepsRow');
        if (substepsRow) substepsRow.style.display = key === 'pm-gpu' ? 'block' : 'none';

        algorithmStatus.textContent = config.label;
        algorithmStatus.style.color = '#9be7a5';

        this.animate();
    }

    setupStaticControls() {
        const algorithmSelect = document.getElementById('algorithm');
        bindSelect(algorithmSelect, (key) => {
            this.switchAlgorithm(key).catch((err) => {
                const algorithmStatus = document.getElementById('algorithmStatus');
                algorithmStatus.textContent = `Algorithm switch failed: ${err.message || err}`;
                algorithmStatus.style.color = '#ff6b6b';
            });
        });

        bindRange(document.getElementById('particleCount'), document.getElementById('particleCountValue'), {
            parse: (v) => parseInt(v, 10),
            onInput: (value) => this.simulation && this.simulation.setParticleCount(value),
        });

        bindRange(document.getElementById('gravity'), document.getElementById('gravityValue'), {
            format: (v) => v.toFixed(1),
            onInput: (value) => this.simulation && this.simulation.setGravityStrength(value),
        });

        bindRange(document.getElementById('timeScale'), document.getElementById('timeScaleValue'), {
            format: (exponent) => formatTimeScale(Math.pow(10, exponent)),
            onInput: (exponent) => this.simulation && this.simulation.setTimeScale(Math.pow(10, exponent)),
        });

        bindRange(document.getElementById('damping'), document.getElementById('dampingValue'), {
            format: (v) => v.toFixed(3),
            onInput: (value) => this.simulation && this.simulation.setDamping(value),
        });

        bindRange(document.getElementById('restitution'), document.getElementById('restitutionValue'), {
            format: (v) => v.toFixed(2),
            onInput: (value) => this.simulation && this.simulation.setRestitution(value),
        });

        bindRange(document.getElementById('orbitalSpeed'), document.getElementById('orbitalSpeedValue'), {
            format: (v) => v.toFixed(2),
            onInput: (value) => this.simulation && this.simulation.setOrbitalSpeed(value),
        });

        bindSelect(document.getElementById('boundaryMode'), (mode) => {
            this.simulation && this.simulation.setBoundaryMode(mode);
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.simulation && this.simulation.resetSimulation();
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
            if (this.algorithmKey === 'pm-gpu') {
                const substepsEl = document.getElementById('substeps');
                if (substepsEl) substepsEl.textContent = this.simulation.lastSubsteps;
            }
        }
        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

const app = new App();
app.init();
