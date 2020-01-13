/** Three seperate kernels that when called in sequence calculate the orizontal
 * diffusion.
 *
 * Reuquired macros:
 * - GRID_ARGS
 * - INDEX
 * - NEIGHBOR
 */
// Laplace Kernel
template<typename value_t>
__global__
void hdiff_unfused_lap(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, value_t *lap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x - 1; // ref implementation starts at i = -1
    const int j = blockIdx.y * blockDim.y + threadIdx.y - 1;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(i >= info.max_coord.x ||
        j >= info.max_coord.y ||
        k >= info.max_coord.z) {
        return;
    }
    const int idx = INDEX(i, j, k);
    lap[idx] = 
        4 * in[idx] 
        - (in[NEIGHBOR(idx, -1, 0, 0)] 
            + in[NEIGHBOR(idx, +1, 0, 0)] 
            + in[NEIGHBOR(idx, 0, -1, 0)] 
            + in[NEIGHBOR(idx, 0, +1, 0)]);
}

// Flx Kernel
template<typename value_t>
__global__
void hdiff_unfused_flx(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, const value_t *lap, value_t *flx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x - 1;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(i >= info.max_coord.x-1 ||
        j >= info.max_coord.y ||
        k >= info.max_coord.z) {
        return;
    }
    const int idx = INDEX(i, j, k);
    flx[idx] = lap[NEIGHBOR(idx, +1, 0, 0)] - lap[idx];
    if (flx[idx] * (in[NEIGHBOR(idx, +1, 0, 0)] - in[idx]) > 0) {
        flx[idx] = 0.;
    }
}

// Fly Kernel
template<typename value_t>
__global__
void hdiff_unfused_fly(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, const value_t *lap, value_t *fly) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y - 1;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(i >= info.max_coord.x ||
        j >= info.max_coord.y-1 ||
        k >= info.max_coord.z) {
            return;
    }
    const int idx = INDEX(i, j, k);
    fly[idx] = lap[NEIGHBOR(idx, 0, +1, 0)] - lap[idx];
    if (fly[idx] * (in[NEIGHBOR(idx, 0, +1, 0)] - in[idx]) > 0) {
        fly[idx] = 0.;
    }
}

// Output kernel
template<typename value_t>
__global__
void hdiff_unfused_out(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, const value_t *coeff, const value_t *flx, const value_t *fly, value_t *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(i >= info.max_coord.x ||
        j >= info.max_coord.y ||
        k >= info.max_coord.z) {
            return;
    }
    const int idx = INDEX(i, j, k);
    out[idx] =
        in[idx] -
        coeff[idx] * (flx[idx] - flx[NEIGHBOR(idx, -1, 0, 0)] +
                                        fly[idx] - fly[NEIGHBOR(idx, 0, -1, 0)]);

}
