/** Like kernel_shared, but uses a k loop and in each loop iteration
 * threads are synchronized. This reduces the amount of memory required,
 * as only the lap/flx/fly for one k-level is stored in shared memory. 
 *
 * Required macros:
 *  - GRID_INFO
 *  - INDEX
 *  - NEIGHBOR
 *  - SMEM_INDEX
 *  - SMEM_NEIGHBOR */
template<typename value_t>
__global__
void hdiff_shared_kloop(const HdiffCudaBase::Info info,
                        GRID_ARGS
                        const value_t *in,
                        value_t *out,
                        const value_t *coeff) {
    
    // Global grid position
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(i >= info.max_coord.x || j >= info.max_coord.y) {
        return;
    }

    const int idx = INDEX(i, j, k);
    const int smem_idx = SMEM_INDEX((int)threadIdx.x, (int)threadIdx.y, 0);

    // Shared memory
    extern __shared__ char smem[];

    // Local grids holding results for laplace, flx and fly calcualted by other threads
    value_t *local_lap = (value_t*)smem;
    value_t *local_flx = &local_lap[blockDim.x*blockDim.y];
    value_t *local_fly = &local_flx[blockDim.x*blockDim.y];
    
    // K-loop
    for(int k = info.halo.z; k < info.max_coord.z; k++) {

        // Calculate own laplace
        const value_t lap_ij = 4 * in[NEIGHBOR(idx, 0, 0, 0)] 
            - in[NEIGHBOR(idx, -1, 0, 0)] - in[NEIGHBOR(idx, +1, 0, 0)]
            - in[NEIGHBOR(idx, 0, -1, 0)] - in[NEIGHBOR(idx, 0, +1, 0)];
        local_lap[smem_idx] = lap_ij;

        // Sync threads to enable access to their laplace calculations
        __syncthreads();

        value_t lap_ipj;
        if(threadIdx.x == blockDim.x-1 || i == info.max_coord.x-1) {
            // rightmost in block, need to compute right dependency ourselves
            lap_ipj = 4 * in[NEIGHBOR(idx, +1, 0, 0)]
                - in[NEIGHBOR(idx, 0, 0, 0)] - in[NEIGHBOR(idx, +2, 0, 0)]
                - in[NEIGHBOR(idx, +1, -1, 0)] - in[NEIGHBOR(idx, +1, +1, 0)];
        } else {
            lap_ipj = local_lap[SMEM_NEIGHBOR(idx,0, +1, 0, 0)];
        }

        value_t lap_ijp;
        if(threadIdx.y == blockDim.y-1 || j == info.max_coord.y-1) {
            lap_ijp = 4 * in[NEIGHBOR(idx, 0, +1, 0)]
                - in[NEIGHBOR(idx, -1, +1, 0)] - in[NEIGHBOR(idx, +1, +1, 0)]
                - in[NEIGHBOR(idx, 0, 0, 0)] - in[NEIGHBOR(idx, 0, +2, 0)];
        } else {
            lap_ijp = local_lap[SMEM_NEIGHBOR(idx,0, 0, +1, 0)];
        }

        // Own flx/fly calculation
        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[NEIGHBOR(idx, +1, 0, 0)] - in[idx]) > 0 ? 0 : flx_ij;
        local_flx[smem_idx] = flx_ij;

        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[NEIGHBOR(idx, 0, +1, 0)] - in[idx]) > 0 ? 0 : fly_ij;
        local_fly[smem_idx] = fly_ij;

        // Make flx/fly available to other threads by synchronizing
        __syncthreads();

        value_t flx_imj;
        if(threadIdx.x == 0) {
            // leftmost in block, need to compute left dependency ourselves
            value_t lap_imj = 4 * in[NEIGHBOR(idx, -1, 0, 0)]
                - in[NEIGHBOR(idx, -2, 0, 0)] - in[NEIGHBOR(idx, 0, 0, 0)]
                - in[NEIGHBOR(idx, -1, -1, 0)] - in[NEIGHBOR(idx, -1, +1, 0)];
            flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[idx] - in[NEIGHBOR(idx, -1, 0, 0)]) > 0 ? 0 : flx_imj;
        } else {
            flx_imj = local_flx[SMEM_NEIGHBOR(idx,0, -1, 0, 0)];
        }

        value_t fly_ijm;
        if(threadIdx.y == 0) {
            // need to also calculate lap for j - 1 as we are at boundary
            value_t lap_ijm = 4 * in[NEIGHBOR(idx, 0, -1, 0)]
                    - in[NEIGHBOR(idx, -1, -1, 0)] - in[NEIGHBOR(idx, +1, -1, 0)]
                    - in[NEIGHBOR(idx, 0, -2, 0)] - in[NEIGHBOR(idx, 0, 0, 0)];
            fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[idx] - in[NEIGHBOR(idx, 0, -1, 0)]) > 0 ? 0 : fly_ijm;
        } else {
            fly_ijm = local_fly[SMEM_NEIGHBOR(idx,0, 0, -1, 0)];
        }

        out[idx] = in[idx]
                - coeff[idx] * (flx_ij - flx_imj + fly_ij - fly_ijm);
    }
}