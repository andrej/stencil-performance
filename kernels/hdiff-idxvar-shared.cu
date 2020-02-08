/** A designated kernel invocation (at k=0) loads the neighborship relation
 * for the given x and y coordinates of this kernel. The other kernel
 * invocations at higher levels rely on shared memory to access the
 * neighborship information.
 *
 * Required macros:
 * - GRID_ARGS
 * - INDEX
 * - NEIGHBOR
 * - K_STEP
 */
#define HDIFF_IDXVAR_SHARED_SMEM_SZ_PER_THREAD 17
template<typename value_t>
__global__
void hdiff_idxvar_shared(const coord3 max_coord,
                         GRID_ARGS
                         const value_t *in,
                         value_t *out,
                         const value_t *coeff) {
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    const int k = threadIdx.z + blockIdx.z*blockDim.z;
    if(!(IS_IN_BOUNDS(i, j, k))) {
        return;
    }
    
    extern __shared__ char smem[]; // stores four neighbors of cell i at smem[i*4]
    const int local_idx = (threadIdx.x + threadIdx.y*blockDim.x) * HDIFF_IDXVAR_SHARED_SMEM_SZ_PER_THREAD;
    int * __restrict__ idxvars = &((int *)smem)[local_idx];
    const bool is_first = k % blockDim.z == 0;
    const int k_step = K_STEP;
    int n_0_0_0, n_0_n1_0, n_0_n2_0, n_n1_0_0, n_n1_n1_0, n_n2_0_0, n_0_p1_0, n_0_p2_0, n_p1_0_0, n_p1_p1_0, n_p2_0_0, n_n1_p1_0, n_p1_n1_0; 
    n_0_0_0 = INDEX(i, j, 0);
    if(is_first) {
        // We are the thread responsible for looking up neighbor info
        /*  0 -1 */ n_0_n1_0 =  idxvars[0] = NEIGHBOR(n_0_0_0, 0, -1, 0);
        /* -1  0 */ n_n1_0_0 =  idxvars[1] = NEIGHBOR(n_0_0_0, -1, 0, 0);
        /*  0 +1 */ n_0_p1_0 =  idxvars[2] = NEIGHBOR(n_0_0_0, 0, +1, 0);
        /* +1  0 */ n_p1_0_0 =  idxvars[3] = NEIGHBOR(n_0_0_0, +1, 0, 0);
        #ifdef CHASING
            /*  0 -2 */ n_0_n2_0 =  idxvars[4] = NEIGHBOR(n_0_n1_0, 0, -1, 0);
            /* -1 -1 */ n_n1_n1_0 = idxvars[5] = NEIGHBOR(n_n1_0_0, 0, -1, 0);
            /* -2  0 */ n_n2_0_0 =  idxvars[6] = NEIGHBOR(n_n1_0_0, -1, 0, 0);
            /*  0 +2 */ n_0_p2_0 =  idxvars[7] = NEIGHBOR(n_0_p1_0, 0, +1, 0);
            /* +1 +1 */ n_p1_p1_0 = idxvars[8] = NEIGHBOR(n_p1_0_0, 0, +1, 0);
            /* +2  0 */ n_p2_0_0 =  idxvars[9] = NEIGHBOR(n_p1_0_0, +1, 0, 0);
            /* -1 +1 */ n_n1_p1_0 = idxvars[10]= NEIGHBOR(n_n1_0_0, 0, +1, 0);
            /* +1 -1 */ n_p1_n1_0 = idxvars[11]= NEIGHBOR(n_p1_0_0, 0, -1, 0);
        #else
            /*  0 -2 */ n_0_n2_0 =  idxvars[4] = DOUBLE_NEIGHBOR(n_0_0_0, 0, -1, 0, 0, -1, 0);
            /* -1 -1 */ n_n1_n1_0 = idxvars[5] = DOUBLE_NEIGHBOR(n_0_0_0, -1, 0, 0, 0, -1, 0);
            /* -2  0 */ n_n2_0_0 =  idxvars[6] = DOUBLE_NEIGHBOR(n_0_0_0, -1, 0, 0, -1, 0, 0);
            /*  0 +2 */ n_0_p2_0 =  idxvars[7] = DOUBLE_NEIGHBOR(n_0_0_0, 0, +1, 0, 0, +1, 0);
            /* +1 +1 */ n_p1_p1_0 = idxvars[8] = DOUBLE_NEIGHBOR(n_0_0_0, +1, 0, 0, 0, +1, 0);
            /* +2  0 */ n_p2_0_0 =  idxvars[9] = DOUBLE_NEIGHBOR(n_0_0_0, +1, 0, 0, +1, 0, 0);
            /* -1 +1 */ n_n1_p1_0 = idxvars[10]= DOUBLE_NEIGHBOR(n_0_0_0, -1, 0, 0, 0, +1, 0);
            /* +1 -1 */ n_p1_n1_0 = idxvars[11]= DOUBLE_NEIGHBOR(n_0_0_0, +1, 0, 0, 0, -1, 0);
        #endif
    }

    __syncthreads();

    if(!is_first) {
        n_0_n1_0 = idxvars[0];
        n_n1_0_0 = idxvars[1];
        n_0_p1_0 = idxvars[2];
        n_p1_0_0 = idxvars[3];
        n_0_n2_0 = idxvars[4];
        n_n1_n1_0 = idxvars[5];
        n_n2_0_0 = idxvars[6];
        n_0_p2_0 = idxvars[7];
        n_p1_p1_0 = idxvars[8];
        n_p2_0_0 = idxvars[9];
        n_n1_p1_0 = idxvars[10];
        n_p1_n1_0 = idxvars[11];
    }
    
    n_0_0_0 += k_step;
    n_0_n1_0 += k_step;
    n_0_n2_0 += k_step;
    n_n1_0_0 += k_step;
    n_n1_n1_0 += k_step;
    n_n2_0_0 += k_step;
    n_0_p1_0 += k_step;
    n_0_p2_0 += k_step;
    n_p1_0_0 += k_step;
    n_p1_p1_0 += k_step;
    n_p2_0_0 += k_step;
    n_n1_p1_0 += k_step;
    n_p1_n1_0 += k_step;

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