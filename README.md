# N-Body Gravity Simulation (DirectX 10)

A real-time gravitational N-body simulation built in November 2010 using DirectX 10 and the DXUT framework.

## Overview

This project simulates gravitational interactions between particles using Newton's law of universal gravitation. Each particle attracts every other particle, creating complex orbital dynamics and emergent behaviors like galaxy formation and gravitational slingshots.

## Technical Details

- **Platform**: Windows (DirectX 10 required)
- **Framework**: Microsoft DXUT (DirectX Utility Toolkit)
- **Language**: C++
- **Graphics**: DirectX 10 with HLSL shaders
- **Physics**: CPU-based N-body simulation (O(N²) complexity)

## Files

- `NBodyGravity.cpp` - Main simulation logic and DirectX rendering
- `NBodyGravity.fx` - HLSL shaders for particle rendering
- `NBodyGravity_2010.sln` - Visual Studio 2010 solution
- `DXUT/` - DirectX Utility Toolkit framework

## Compatibility Warning

⚠️ **This is legacy DirectX 10 code from 2010.** Installing and running this on modern Windows (Win10/Win11) may not work due to deprecated DirectX SDK dependencies and compatibility issues.

## Modern Alternative

For a working, cross-platform version of this simulation, see the **WebGPU implementation** in the `webgpu-nbody/` directory. It provides the same gravitational physics with modern web technology and improved camera controls.

## Building (Historical)

Originally built with:
- Visual Studio 2010
- DirectX SDK (June 2010)
- Windows 7/Vista

*Note: Modern build instructions are not provided as this version is no longer maintained.*



