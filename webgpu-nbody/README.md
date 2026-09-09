# WebGPU N-Body Gravity Simulation

A modern, high-performance N-body gravity simulation using WebGPU compute shaders. This is a complete rewrite of the original DirectX 10 implementation using cutting-edge web technologies.

**🎮 [Try the Live Demo](https://demo.goocto.com/webgpu-nbody/)**

![N-Body Simulation](https://img.shields.io/badge/WebGPU-Enabled-brightgreen)
![Compute Shaders](https://img.shields.io/badge/WGSL-Compute%20Shaders-blue)

## 🚀 Features

- **Two live physics backends**: Switch between the existing CPU direct O(N²) solver and a GPU FMM-style octree solver
- **State-preserving switching**: GPU state is read back only before entering CPU mode or resizing the particle set; CPU state is uploaded before entering GPU mode
- **Real-time N-Body Physics**: Direct interactions remain available for near-field cells while distant cells use octree multipoles
- **Interactive Controls**: Adjust gravity strength, particle count, time scale, and damping in real-time
- **Orbital Camera**: Mouse/touch controls for 360° viewing
- **Cross-Platform**: Runs in any WebGPU-compatible browser
- **High Performance**: GPU-accelerated compute shaders for maximum efficiency

## 🌐 Browser Requirements

WebGPU is still experimental. You'll need:

### Chrome/Chromium (Recommended)
- Chrome 113+ or Edge 113+
- Enable WebGPU: `chrome://flags/#enable-unsafe-webgpu`

### Firefox
- Firefox Nightly
- Enable WebGPU: `about:config` → `dom.webgpu.enabled = true`

### Safari
- Safari Technology Preview
- WebGPU should be enabled by default

## 🛠️ Quick Start

### Option 1: Python Server (Recommended)
```bash
cd webgpu-nbody
python -m http.server 8080
# Or if you have Python 3:
python3 -m http.server 8080
```

### Option 2: Node.js Server
```bash
cd webgpu-nbody
npm install
npm run dev-node
```

### Option 3: Any HTTP Server
Any local HTTP server will work. HTTPS is required for WebGPU in some browsers.

Then open: http://localhost:8080

## 🎮 Controls

- **Mouse Drag**: Orbit around the simulation
- **Mouse Wheel**: Zoom in/out
- **Touch**: Mobile-friendly orbital controls
- **Particle Count**: Adjust the number of particles (100-4,000)
- **Gravity Strength**: Control gravitational force intensity
- **Time Scale**: Speed up or slow down the simulation
- **Damping**: Add velocity damping to stabilize the system
- **Physics backend**: Choose `CPU direct (O(N²))` or `GPU FMM octree` while running
- **Reset**: Reinitialize particle positions and velocities

The backend selector does not reset particles. If GPU initialization or a mode
transition fails, the current backend remains active and the error is shown in
the status line; there is no silent fallback.

## 🏗️ Architecture

### Compute Shaders (WGSL)
- `octree-leaf.wgsl`: Assigns particles to fixed-depth leaf cells on the GPU
- `octree-leaf-multipole.wgsl`: Builds leaf mass/center-of-mass multipoles
- `octree-aggregate.wgsl`: Reduces leaf multipoles through the hierarchy
- `fmm-force-compute.wgsl`: Traverses the GPU octree, using direct leaf interactions and accepted distant multipoles
- `fmm-integrate-compute.wgsl`: Updates particle positions and velocities
- `octree-clear.wgsl`: Clears per-frame multipole accumulators

### Render Pipeline (WGSL)
- `particle-vertex-quad.wgsl`: Transforms particle positions to screen space
- `particle-fragment-quad.wgsl`: Renders particles with color-coded velocities

### JavaScript Classes
- `SimpleNBodySimulation`: CPU solver, GPU compute passes, resource management, and state synchronization
- `CameraController`: Orbital camera with smooth mouse/touch controls
- `App`: Application lifecycle and UI management

## 🔧 Technical Details

### Simulation Algorithm
The CPU backend retains the direct pairwise O(N²) calculation. The GPU backend
executes separate WebGPU compute passes to clear a fixed four-level octree,
assign particles to leaves, build monopole mass/center-of-mass data bottom-up,
traverse the hierarchy with a fixed opening criterion, and integrate particles.
Near leaves use direct particle interactions; accepted distant nodes use their
mass multipole. Both backends apply the same damping and boundary conditions.

The internal GPU accuracy settings are fixed (depth 4, opening threshold 0.65,
softening 0.01); they are not exposed as UI tuning controls.

### Performance Optimizations
- **Compute Workgroups**: Uses 64-thread workgroups
- **Structured Buffers**: Shared particle layout for rendering and both solvers
- **Separate passes**: Octree construction, multipole aggregation, force traversal, and integration are distinct compute passes
- **GPU readback**: A staging buffer synchronizes state only when switching back to the CPU backend or resizing the particle set
- **GPU frame back-pressure**: GPU simulation and render submissions wait for the previous frame to complete, preventing an unbounded queue backlog

The Compute Encode and Render Encode/Submit values in the UI are CPU-side
timings for command encoding and `queue.submit()`. They do not measure GPU
execution time. GPU-to-CPU readback is not performed every frame; it occurs
only when switching from GPU to CPU or resizing the particle set.

### WebGPU Features Used
- Compute pipelines for physics simulation
- Render pipelines for particle visualization  
- Storage buffers for particle data
- Uniform buffers for simulation parameters
- Bind groups for resource management

## 🎨 Customization

### Modify Physics Parameters
Edit the initial values in `main-stable.js`:
```javascript
this.gravityStrength = 1.0;  // Gravitational constant
this.damping = 0.999;        // Velocity damping factor
this.timeScale = 1.0;        // Simulation speed multiplier
```

### Change Visual Appearance
Modify `particle-fragment.wgsl` to adjust:
- Particle colors
- Size scaling
- Transparency effects
- Glow effects

### Add New Initial Conditions
Edit `initializeParticles()` in `main-stable.js` to create:
- Galaxy formations
- Binary systems  
- Clustered configurations
- Custom patterns

## 🚧 Comparison with Original DirectX Version

| Feature | DirectX 10 | WebGPU |
|---------|------------|---------|
| **Platform** | Windows only | Cross-platform |
| **Compute** | Render-to-texture hack | Native compute shaders |
| **Language** | C++ with HLSL | JavaScript with WGSL |
| **Dependencies** | DXUT, DirectX SDK | None (web browser) |
| **Development** | Visual Studio required | Any text editor |
| **Distribution** | Compiled executable | Web-based (instant access) |

## 🔮 Future Enhancements

- **Higher-order multipoles**: Add quadrupole terms to the existing monopole path
- **Compute Performance Analysis**: GPU timing and profiling
- **Advanced Rendering**: Bloom effects, trails, and procedural backgrounds
- **Physics Presets**: Pre-configured galaxy, solar system, and cluster simulations
- **Data Export**: Save/load simulation states and export particle data

## 📝 License

MIT License - see the original project for details.

## 🤝 Contributing

This is a modernization of the original DirectX N-body simulation. Contributions welcome for:
- Performance optimizations
- Visual enhancements  
- New physics features
- Browser compatibility improvements
- Mobile optimization

---

**Note**: This WebGPU implementation demonstrates how modern web technologies can match and exceed the performance of native graphics applications while being more accessible and portable.