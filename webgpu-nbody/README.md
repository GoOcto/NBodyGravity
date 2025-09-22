# WebGPU N-Body Gravity Simulation

A modern, high-performance N-body gravity simulation using WebGPU compute shaders. This is a complete rewrite of the original DirectX 10 implementation using cutting-edge web technologies.

![N-Body Simulation](https://img.shields.io/badge/WebGPU-Enabled-brightgreen)
![Compute Shaders](https://img.shields.io/badge/WGSL-Compute%20Shaders-blue)

## 🚀 Features

- **Pure WebGPU Implementation**: Uses modern compute shaders for physics simulation
- **Real-time N-Body Physics**: Accurate gravitational force calculations between up to 10,000 particles
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
- **Particle Count**: Adjust the number of particles (100-10,000)
- **Gravity Strength**: Control gravitational force intensity
- **Time Scale**: Speed up or slow down the simulation
- **Damping**: Add velocity damping to stabilize the system
- **Reset**: Reinitialize particle positions and velocities

## 🏗️ Architecture

### Compute Shaders (WGSL)
- `force-compute.wgsl`: Calculates gravitational forces between all particles
- `integrate-compute.wgsl`: Updates particle positions and velocities using Verlet integration

### Render Pipeline (WGSL)  
- `particle-vertex.wgsl`: Transforms particle positions to screen space
- `particle-fragment.wgsl`: Renders particles with color-coded velocities

### JavaScript Classes
- `NBodySimulation`: Main simulation engine and WebGPU setup
- `CameraController`: Orbital camera with smooth mouse/touch controls
- `App`: Application lifecycle and UI management

## 🔧 Technical Details

### Simulation Algorithm
1. **Force Calculation**: Each particle calculates gravitational forces from all other particles using compute shaders
2. **Numerical Integration**: Verlet integration updates particle positions and velocities
3. **Boundary Conditions**: Particles bounce off invisible walls with energy loss
4. **Visualization**: Particles are color-coded by velocity magnitude (blue = slow, red = fast)

### Performance Optimizations
- **Compute Workgroups**: Uses 64-thread workgroups for optimal GPU utilization
- **Structured Buffers**: Efficient GPU memory layout for particle data
- **Double Buffering**: Separates force calculation and integration passes
- **Adaptive Quality**: Frame rate monitoring for performance adjustment

### WebGPU Features Used
- Compute pipelines for physics simulation
- Render pipelines for particle visualization  
- Storage buffers for particle data
- Uniform buffers for simulation parameters
- Bind groups for resource management

## 🎨 Customization

### Modify Physics Parameters
Edit the initial values in `main.js`:
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
Edit `initializeParticles()` in `main.js` to create:
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

- **Spatial Partitioning**: Octree/grid-based optimization for larger particle counts
- **Barnes-Hut Algorithm**: O(N log N) approximation for massive simulations  
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