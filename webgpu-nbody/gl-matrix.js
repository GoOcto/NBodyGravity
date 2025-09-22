// Simplified gl-matrix for WebGPU N-Body simulation
// Only includes the functions we need

export const vec3 = {
    fromValues: (x, y, z) => new Float32Array([x, y, z]),

    length: (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]),

    normalize: (out, a) => {
        const len = vec3.length(a);
        if (len > 0) {
            out[0] = a[0] / len;
            out[1] = a[1] / len;
            out[2] = a[2] / len;
        }
        return out;
    },

    subtract: (out, a, b) => {
        out[0] = a[0] - b[0];
        out[1] = a[1] - b[1];
        out[2] = a[2] - b[2];
        return out;
    }
};

export const mat4 = {
    create: () => new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]),

    lookAt: (out, eye, center, up) => {
        const eyex = eye[0];
        const eyey = eye[1];
        const eyez = eye[2];
        const upx = up[0];
        const upy = up[1];
        const upz = up[2];
        const centerx = center[0];
        const centery = center[1];
        const centerz = center[2];

        if (Math.abs(eyex - centerx) < 0.000001 &&
            Math.abs(eyey - centery) < 0.000001 &&
            Math.abs(eyez - centerz) < 0.000001) {
            return mat4.identity(out);
        }

        let z0 = eyex - centerx;
        let z1 = eyey - centery;
        let z2 = eyez - centerz;

        let len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        z0 *= len;
        z1 *= len;
        z2 *= len;

        let x0 = upy * z2 - upz * z1;
        let x1 = upz * z0 - upx * z2;
        let x2 = upx * z1 - upy * z0;
        len = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
        if (!len) {
            x0 = 0;
            x1 = 0;
            x2 = 0;
        } else {
            len = 1 / len;
            x0 *= len;
            x1 *= len;
            x2 *= len;
        }

        let y0 = z1 * x2 - z2 * x1;
        let y1 = z2 * x0 - z0 * x2;
        let y2 = z0 * x1 - z1 * x0;

        len = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
        if (!len) {
            y0 = 0;
            y1 = 0;
            y2 = 0;
        } else {
            len = 1 / len;
            y0 *= len;
            y1 *= len;
            y2 *= len;
        }

        out[0] = x0;
        out[1] = y0;
        out[2] = z0;
        out[3] = 0;
        out[4] = x1;
        out[5] = y1;
        out[6] = z1;
        out[7] = 0;
        out[8] = x2;
        out[9] = y2;
        out[10] = z2;
        out[11] = 0;
        out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
        out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
        out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
        out[15] = 1;

        return out;
    },

    perspective: (out, fovy, aspect, near, far) => {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);

        out[0] = f / aspect;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 0;
        out[5] = f;
        out[6] = 0;
        out[7] = 0;
        out[8] = 0;
        out[9] = 0;
        out[10] = (far + near) * nf;
        out[11] = -1;
        out[12] = 0;
        out[13] = 0;
        out[14] = 2 * far * near * nf;
        out[15] = 0;

        return out;
    },

    multiply: (out, a, b) => {
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

        let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
        out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
        out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
        out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
        out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

        b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
        out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
        out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
        out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
        out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

        b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
        out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
        out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
        out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
        out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

        b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
        out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
        out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
        out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
        out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

        return out;
    },

    identity: (out) => {
        out[0] = 1;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 0;
        out[5] = 1;
        out[6] = 0;
        out[7] = 0;
        out[8] = 0;
        out[9] = 0;
        out[10] = 1;
        out[11] = 0;
        out[12] = 0;
        out[13] = 0;
        out[14] = 0;
        out[15] = 1;
        return out;
    }
};