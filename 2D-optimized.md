# Multi-Resolution Grid-Based Gravity Simulation

This document describes a highly scalable, grid-based N-body gravity simulation algorithm designed to handle extremely large particle counts in real time. The algorithm diverges from traditional $O(N^2)$ direct integration or tree-based (Barnes-Hut) methods by leveraging spatial grids, multiple levels of detail (LODs), and separable image blurs to approximate gravitational potential fields.

## 1. Algorithm Overview

The core idea is to treat individual particles not as point masses that interact directly with every other particle, but as density contributions to a continuous field. By discretizing space into a grid and spreading the accumulated mass using a blur, the algorithm computes a scalar potential field (or force field) from which individual particles can simply sample their local gradients to determine acceleration. 

To account for both local accuracy and distant, large-scale gravitational effects without an enormous performance cost, the algorithm uses multiple resolutions (Levels of Detail) of the mass grid.

## 2. Core Steps of the Algorithm

### Step 2.1: Mass Accumulation (Particle-to-Grid)
In the first step, the continuous domain is divided into a discrete 2D uniform grid. 
* Iterate through every particle in the system.
* Determine which grid cell the particle's $(x, y)$ coordinate falls into.
* Add the particle's mass to that specific grid cell.
* The output is a high-resolution 2D array representing the mass density of the system.

### Step 2.2: Generating Levels of Detail (Downsampling)
To handle distant gravitational forces efficiently, the algorithm builds a hierarchy of grids, conceptually similar to an image mipmap or a multigrid hierarchy. 
* Starting with the base (highest resolution) grid, group cells into $2 	imes 2$ blocks.
* Sum the mass of these blocks to form a single cell in the next coarser grid (LOD 1).
* Repeat this process to generate further levels (e.g., LOD 2, LOD 3). 
* The base level handles accurate local forces, while the coarse levels handle rough approximations of distant massive clusters.

### Step 2.3: Mass Spreading via Separable Blur
The most computationally intensive part of gravity is that mass affects distant points. Instead of having each grid cell sample all surrounding cells within a large radius (which scales poorly), the algorithm "spreads" the mass outward using a blur filter.
* **Separable Convolution:** To optimize this, the 2D blur is separated into a 1D horizontal pass followed by a 1D vertical pass. 
* **Kernel Weighting:** The weights of the blur kernel are chosen to approximate gravitational decay. While a standard Gaussian blur spreads values, a custom kernel tailored to mimic the $1/r^2$ force law (or $1/r$ potential) provides a better physical approximation.
* This blur is applied to *every* LOD grid. The result is a set of "potential fields" where the value at any cell represents the gravitational influence from the surrounding area.

### Step 2.4: Force Calculation (Gradient Sampling)
With the blurred mass grids calculated, the interactions between particles are decoupled. A particle no longer needs to know about other particles; it only needs to read the environment grids.
* For a given particle, compute the local gradient (the rate of change in the horizontal and vertical directions) of the blurred mass field at the particle's position. 
* **Multi-Resolution Sampling:** The particle samples the gradient from the highest resolution grid for local forces. It then samples the gradients from the progressively coarser grids to account for forces from distant objects. 
* The total gravitational force vector on the particle is the weighted sum of these sampled gradients.

### Step 2.5: Integration (Movement)
Once the force (acceleration) vector is known, the particle's state is updated using a numerical integrator (e.g., Symplectic Euler or Velocity Verlet).
* `Velocity_new = Velocity_old + Force * delta_time`
* `Position_new = Position_old + Velocity_new * delta_time`

## 3. Advantages and Edge Cases

### Scalability
By completely decoupling particle-to-particle interactions, the time complexity shifts from $O(N^2)$ or $O(N \log N)$ to $O(N + G)$, where $N$ is the number of particles and $G$ is the total number of grid cells across all LODs. 

### Approximations and Artifacts
* **Dampening Effect:** Because local masses are binned into the same cell, intra-cell (or "intra-pixel") forces are severely underestimated. Two particles in the exact same grid cell may exert zero force on each other.
* **Grid Artifacts:** "Invisible walls" or grid-aligned artifacts can occur due to discrete down-sampling. If a cluster of particles crosses a cell boundary, the sudden shift in how its mass is represented in coarser LODs can cause sudden changes in the force field.
* **Blur Error:** Using a separable horizontal + vertical blur is mathematically not perfectly radially symmetric. It causes a slightly square-shaped gravitational well compared to a true circular radial decay, introducing small directional biases.

## 4. Implementation Agnosticism
This structure maps perfectly to modern parallel computing architectures. While heavily utilized in GPU compute shaders (where grid cells map to textures/buffers and blur passes are standard image processing techniques), it is equally viable on multi-core CPUs. The mass accumulation can be done via atomic additions or spatial sorting, and the separable blur is highly cache-friendly for CPU vectorization (SIMD).