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
    const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1; // ref implementation starts at i = -1
    const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
    if(i >= info.max_coord.x ||
        j >= info.max_coord.y ||
        k >= info.max_coord.z) {
        return;
    }
    lap[INDEX(i, j, k)] = 
        4 * in[INDEX(i, j, k)] 
        - (in[NEIGHBOR(i, j, k, -1, 0, 0)] 
            + in[NEIGHBOR(i, j, k, +1, 0, 0)] 
            + in[NEIGHBOR(i, j, k, 0, -1, 0)] 
            + in[NEIGHBOR(i, j, k, 0, +1, 0)]);
}

// Flx Kernel
template<typename value_t>
__global__
void hdiff_unfused_flx(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, const value_t *lap, value_t *flx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
    if(i >= info.max_coord.x-1 ||
        j >= info.max_coord.y ||
        k >= info.max_coord.z) {
        return;
    }
    flx[INDEX(i, j, k)] = lap[NEIGHBOR(i, j, k, +1, 0, 0)] - lap[INDEX(i, j, k)];
    if (flx[INDEX(i, j, k)] * (in[NEIGHBOR(i, j, k, +1, 0, 0)] - in[INDEX(i, j, k)]) > 0) {
        flx[INDEX(i, j, k)] = 0.;
    }
}

// Fly Kernel
template<typename value_t>
__global__
void hdiff_unfused_fly(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, const value_t *lap, value_t *fly) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
    if(i >= info.max_coord.x ||
        j >= info.max_coord.y-1 ||
        k >= info.max_coord.z) {
            return;
        }
    fly[INDEX(i, j, k)] = lap[NEIGHBOR(i, j, k, 0, +1, 0)] - lap[INDEX(i, j, k)];
    if (fly[INDEX(i, j, k)] * (in[NEIGHBOR(i, j, k, 0, +1, 0)] - in[INDEX(i, j, k)]) > 0) {
        fly[INDEX(i, j, k)] = 0.;
    }
}

// Output kernel
template<typename value_t>
__global__
void hdiff_unfused_out(HdiffCudaBase::Info info, GRID_ARGS const value_t *in, const value_t *coeff, const value_t *flx, const value_t *fly, value_t *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
    if(i >= info.max_coord.x ||
        j >= info.max_coord.y ||
        k >= info.max_coord.z) {
            return;
        }
    out[INDEX(i, j, k)] =
        in[INDEX(i, j, k)] -
        coeff[INDEX(i, j, k)] * (flx[INDEX(i, j, k)] - flx[NEIGHBOR(i, j, k, -1, 0, 0)] +
                                        fly[INDEX(i, j, k)] - fly[NEIGHBOR(i, j, k, 0, -1, 0)]);

}
