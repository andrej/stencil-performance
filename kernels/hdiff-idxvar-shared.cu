/** A designated kernel invocation (at k=0) loads the neighborship relation
 * for the given x and y coordinates of this kernel. The other kernel
 * invocations at higher levels rely on shared memory to access the
 * neighborship information.
 *
 * Required macros:
 * - GRID_ARGS
 * - INDEX
 * - NEIGHBOR_OF_INDEX
 * - K_STEP
 */
template<typename value_t>
__global__
void hdiff_idxvar_shared(const HdiffCudaBase::Info info,
                         GRID_ARGS
                         const value_t *in,
                         value_t *out,
                         const value_t *coeff) {
    const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
    const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
    if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
        return;
    }
    
    extern __shared__ char smem[]; // stores four neighbors of cell i at smem[i*4]
    int * __restrict__ idxvars = (int *)smem;
    const int local_idx = (threadIdx.x + threadIdx.y*blockDim.x) * 12;
    const int global_idx_2d = INDEX(i, j, 0);

    if(k % blockDim.z == 0) {
        // We are the thread responsible for looking up neighbor info
        /*  0 -1 */ idxvars[local_idx+0] = NEIGHBOR_OF_INDEX(global_idx_2d, 0, -1, 0);
        /*  0 -2 */ idxvars[local_idx+1] = NEIGHBOR_OF_INDEX(idxvars[local_idx+0], 0, -1, 0);
        /* -1  0 */ idxvars[local_idx+2] = NEIGHBOR_OF_INDEX(global_idx_2d, -1, 0, 0);
        /* -1 -1 */ idxvars[local_idx+3] = NEIGHBOR_OF_INDEX(idxvars[local_idx+2], 0, -1, 0);
        /* -2  0 */ idxvars[local_idx+4] = NEIGHBOR_OF_INDEX(idxvars[local_idx+2], -1, 0, 0);
        /*  0 +1 */ idxvars[local_idx+5] = NEIGHBOR_OF_INDEX(global_idx_2d, 0, +1, 0);
        /*  0 +2 */ idxvars[local_idx+6] = NEIGHBOR_OF_INDEX(idxvars[local_idx+5], 0, +1, 0);
        /* +1  0 */ idxvars[local_idx+7] = NEIGHBOR_OF_INDEX(global_idx_2d, +1, 0, 0);
        /* +1 +1 */ idxvars[local_idx+8] = NEIGHBOR_OF_INDEX(idxvars[local_idx+7], 0, +1, 0);
        /* +2  0 */ idxvars[local_idx+9] = NEIGHBOR_OF_INDEX(idxvars[local_idx+7], +1, 0, 0);
        /* -1 +1 */ idxvars[local_idx+10]= NEIGHBOR_OF_INDEX(idxvars[local_idx+2], 0, +1, 0);
        /* +1 -1 */ idxvars[local_idx+11]= NEIGHBOR_OF_INDEX(idxvars[local_idx+7], 0, -1, 0);
    }
    
    __syncthreads();
    const int k_step = K_STEP;
    const int n_0_0_0       = global_idx_2d + k_step;
    const int n_0_n1_0      = idxvars[local_idx+0] + k_step;
    const int n_0_n2_0      = idxvars[local_idx+1] + k_step;
    const int n_n1_0_0      = idxvars[local_idx+2] + k_step;
    const int n_n1_n1_0     = idxvars[local_idx+3] + k_step;
    const int n_n2_0_0      = idxvars[local_idx+4] + k_step;
    //const int n_n2_n1_0     = idxvars_n_n2_n1_0 + k_step;
    const int n_0_p1_0      = idxvars[local_idx+5] + k_step;
    const int n_0_p2_0      = idxvars[local_idx+6] + k_step;
    const int n_p1_0_0      = idxvars[local_idx+7] + k_step;
    const int n_p1_p1_0     = idxvars[local_idx+8] + k_step;
    const int n_p2_0_0      = idxvars[local_idx+9] + k_step;
    //const int n_p2_p1_0     = idxvars_n_p2_p1_0 + k_step;
    const int n_n1_p1_0     = idxvars[local_idx+10] + k_step;
    const int n_p1_n1_0     = idxvars[local_idx+11] + k_step;

    const value_t lap_ij = 
        4 * in[n_0_0_0] 
        - in[n_n1_0_0] - in[n_p1_0_0]
        - in[n_0_n1_0] - in[n_0_p1_0];
    const value_t lap_imj = 
        4 * in[n_n1_0_0]
        - in[n_n2_0_0] - in[n_0_0_0]
        - in[n_n1_n1_0] - in[n_n1_p1_0];
    const value_t lap_ipj =
        4 * in[n_p1_0_0]
        - in[n_0_0_0] - in[n_p2_0_0]
        - in[n_p1_n1_0] - in[n_p1_p1_0];
    const value_t lap_ijm =
        4 * in[n_0_n1_0]
        - in[n_n1_n1_0] - in[n_p1_n1_0]
        - in[n_0_n2_0] - in[n_0_0_0];
    const value_t lap_ijp =
        4 * in[n_0_p1_0]
        - in[n_n1_p1_0] - in[n_p1_p1_0]
        - in[n_0_0_0] - in[n_0_p2_0];

    value_t flx_ij = lap_ipj - lap_ij;
    flx_ij = flx_ij * (in[n_p1_0_0] - in[n_0_0_0]) > 0 ? 0 : flx_ij;

    value_t flx_imj = lap_ij - lap_imj;
    flx_imj = flx_imj * (in[n_0_0_0] - in[n_n1_0_0]) > 0 ? 0 : flx_imj;

    value_t fly_ij = lap_ijp - lap_ij;
    fly_ij = fly_ij * (in[n_0_p1_0] - in[n_0_0_0]) > 0 ? 0 : fly_ij;

    value_t fly_ijm = lap_ij - lap_ijm;
    fly_ijm = fly_ijm * (in[n_0_0_0] - in[n_0_n1_0]) > 0 ? 0 : fly_ijm;

    out[n_0_0_0] =
        in[n_0_0_0]
        - coeff[n_0_0_0] * (flx_ij - flx_imj + fly_ij - fly_ijm);

}