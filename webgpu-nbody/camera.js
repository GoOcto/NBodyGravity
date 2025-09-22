// Camera controls for WebGPU N-Body simulation
export class CameraController {
    constructor(camera, canvas) {
        this.camera = camera;
        this.canvas = canvas;

        // Mouse interaction state
        this.isMouseDown = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Orbital camera state - use rotation matrix for view-relative rotations
        this.distance = 100;
        this.targetDistance = this.distance;

        // Camera orientation as 3x3 rotation matrix (starts as identity)
        this.rotMatrix = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ];

        // Smooth camera movement
        this.smoothing = 0.1;

        this.setupEventListeners();
        this.updateCameraPosition();
    }

    setupEventListeners() {
        // Mouse controls
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));

        // Touch controls for mobile
        this.canvas.addEventListener('touchstart', (e) => this.onTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.onTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.onTouchEnd(e));

        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    onMouseDown(e) {
        this.isMouseDown = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }

    onMouseMove(e) {
        if (!this.isMouseDown) return;

        const deltaX = e.clientX - this.lastMouseX;
        const deltaY = e.clientY - this.lastMouseY;

        // View-relative rotations using current camera axes
        const rotationSpeed = 0.01;

        // Left/Right: rotate around camera's current up vector
        if (Math.abs(deltaX) > 0.001) {
            this.rotateAroundCameraUp(-deltaX * rotationSpeed);
        }

        // Up/Down: rotate around camera's current right vector
        if (Math.abs(deltaY) > 0.001) {
            this.rotateAroundCameraRight(-deltaY * rotationSpeed);
        }

        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;

        this.updateCameraPosition();
    }

    onMouseUp(e) {
        this.isMouseDown = false;
        this.canvas.style.cursor = 'grab';
    }

    onWheel(e) {
        e.preventDefault();

        const zoomSpeed = 0.1;
        const zoomDelta = e.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;

        this.targetDistance = Math.max(10, Math.min(500, this.targetDistance * zoomDelta));
    }

    onTouchStart(e) {
        if (e.touches.length === 1) {
            this.isMouseDown = true;
            this.lastMouseX = e.touches[0].clientX;
            this.lastMouseY = e.touches[0].clientY;
        }
    }

    onTouchMove(e) {
        e.preventDefault();

        if (e.touches.length === 1 && this.isMouseDown) {
            const deltaX = e.touches[0].clientX - this.lastMouseX;
            const deltaY = e.touches[0].clientY - this.lastMouseY;

            // View-relative rotations using current camera axes (same as mouse)
            const rotationSpeed = 0.01;

            // Left/Right: rotate around camera's current up vector
            if (Math.abs(deltaX) > 0.001) {
                this.rotateAroundCameraUp(-deltaX * rotationSpeed);
            }

            // Up/Down: rotate around camera's current right vector
            if (Math.abs(deltaY) > 0.001) {
                this.rotateAroundCameraRight(-deltaY * rotationSpeed);
            }

            this.lastMouseX = e.touches[0].clientX;
            this.lastMouseY = e.touches[0].clientY;

            this.updateCameraPosition();
        }
    }

    onTouchEnd(e) {
        this.isMouseDown = false;
    }

    update() {
        // Smooth distance interpolation
        this.distance += (this.targetDistance - this.distance) * this.smoothing;
        this.updateCameraPosition();
    }

    // Rotate around camera's current up vector (left/right movement)
    rotateAroundCameraUp(angle) {
        // Get current camera up vector from rotation matrix
        const upX = this.rotMatrix[1]; // Second column, first row
        const upY = this.rotMatrix[4]; // Second column, second row
        const upZ = this.rotMatrix[7]; // Second column, third row

        // Create rotation matrix around this up vector
        this.rotateAroundAxis(upX, upY, upZ, angle);
    }

    // Rotate around camera's current right vector (up/down movement)  
    rotateAroundCameraRight(angle) {
        // Get current camera right vector from rotation matrix
        const rightX = this.rotMatrix[0]; // First column, first row
        const rightY = this.rotMatrix[3]; // First column, second row
        const rightZ = this.rotMatrix[6]; // First column, third row

        // Create rotation matrix around this right vector
        this.rotateAroundAxis(rightX, rightY, rightZ, angle);
    }

    // Rotate the camera's orientation matrix around an arbitrary axis using Rodrigues' formula
    rotateAroundAxis(axisX, axisY, axisZ, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const oneMinusCos = 1 - cos;

        // Rodrigues rotation matrix
        const rotMatrix = [
            cos + axisX * axisX * oneMinusCos,
            axisX * axisY * oneMinusCos - axisZ * sin,
            axisX * axisZ * oneMinusCos + axisY * sin,

            axisY * axisX * oneMinusCos + axisZ * sin,
            cos + axisY * axisY * oneMinusCos,
            axisY * axisZ * oneMinusCos - axisX * sin,

            axisZ * axisX * oneMinusCos - axisY * sin,
            axisZ * axisY * oneMinusCos + axisX * sin,
            cos + axisZ * axisZ * oneMinusCos
        ];

        // Apply rotation: newMatrix = rotMatrix * currentMatrix
        this.rotMatrix = this.multiply3x3(rotMatrix, this.rotMatrix);
    }

    // 3x3 matrix multiplication
    multiply3x3(a, b) {
        return [
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
        ];
    }

    updateCameraPosition() {
        // Calculate camera position: apply rotation matrix to [0, 0, distance]
        // This puts camera at distance along the rotated Z-axis
        const x = this.rotMatrix[2] * this.distance;  // Z-axis X component
        const y = this.rotMatrix[5] * this.distance;  // Z-axis Y component
        const z = this.rotMatrix[8] * this.distance;  // Z-axis Z component

        this.camera.position[0] = x;
        this.camera.position[1] = y;
        this.camera.position[2] = z;

        // Set camera up vector to the current Y-axis of rotation matrix
        this.camera.up[0] = this.rotMatrix[1]; // Y-axis X component
        this.camera.up[1] = this.rotMatrix[4]; // Y-axis Y component
        this.camera.up[2] = this.rotMatrix[7]; // Y-axis Z component
    }

    reset() {
        this.distance = 100;
        this.targetDistance = 100;
        // Reset to identity rotation matrix (looking down -Z axis)
        this.rotMatrix = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ];
        this.updateCameraPosition();
    }
}