    /** Kernels cooperating/sharing intermediate results through
     * unified memory.
     *
     * Shared intermediate results: In this variant, each kernel first
     * computes its values for lap, flx and fly (at its i, j, k) position.
     * Threads are synchronized at appropriate points so they can access each
     * others results as needed.
     * 
     * Note that at the block boundaries, some threads still need to compute
     * their dependencies, as __synchthreads() only synchronizes within the
     * block, not entire grid.
     * 
     * This can be thought of as a kind of sequential version, but only on the
     * thread-level. 
     *
     * Required macros:
     * - GRID_ARGS
     * - INDEX
     * - NEIGHBOR */
     template<typename value_t>
     __global__
     void hdiff_coop(const HdiffCudaBase::Info info,
                     GRID_ARGS
                     const value_t *in,
                     value_t *out,
                     const value_t *coeff,
                     value_t *lap,
                     value_t *flx,
                     value_t *fly) {
         
         // Global grid position
         const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
         const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
         const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
         
         if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
             return;
         }
 
         // Calculate own laplace
         value_t lap_ij = 4 * in[NEIGHBOR(i, j, k, 0, 0, 0)] 
             - in[NEIGHBOR(i, j, k, -1, 0, 0)] - in[NEIGHBOR(i, j, k, +1, 0, 0)]
             - in[NEIGHBOR(i, j, k, 0, -1, 0)] - in[NEIGHBOR(i, j, k, 0, +1, 0)];
         lap[INDEX(i, j, k)] = lap_ij;
 
         // Sync threads to enable access to their laplace calculations
         __syncthreads();
 
         value_t lap_ipj;
         if(threadIdx.x == blockDim.x-1) {
             // rightmost in block, need to compute right dependency ourselves
             lap_ipj = 4 * in[NEIGHBOR(i, j, k, +1, 0, 0)]
                 - in[NEIGHBOR(i, j, k, 0, 0, 0)] - in[NEIGHBOR(i, j, k, +2, 0, 0)]
                 - in[NEIGHBOR(i, j, k, +1, -1, 0)] - in[NEIGHBOR(i, j, k, +1, +1, 0)];
         } else {
             lap_ipj = lap[NEIGHBOR(i, j, k, +1, 0, 0)];
         }
 
         value_t lap_ijp;
         if(threadIdx.y == blockDim.y-1) {
             lap_ijp = 4 * in[NEIGHBOR(i, j, k, 0, +1, 0)]
                 - in[NEIGHBOR(i, j, k, -1, +1, 0)] - in[NEIGHBOR(i, j, k, +1, +1, 0)]
                 - in[NEIGHBOR(i, j, k, 0, 0, 0)] - in[NEIGHBOR(i, j, k, 0, +2, 0)];
         } else {
             lap_ijp = lap[NEIGHBOR(i, j, k, 0, +1, 0)];
         }
 
         // Own flx/fly calculation
         value_t flx_ij = lap_ipj - lap_ij;
         flx_ij = flx_ij * (in[NEIGHBOR(i, j, k, +1, 0, 0)] - in[INDEX(i, j, k)]) > 0 ? 0 : flx_ij;
         flx[INDEX(i, j, k)] = flx_ij;
 
         value_t fly_ij = lap_ijp - lap_ij;
         fly_ij = fly_ij * (in[NEIGHBOR(i, j, k, 0, +1, 0)] - in[INDEX(i, j, k)]) > 0 ? 0 : fly_ij;
         fly[INDEX(i, j, k)] = fly_ij;
 
         // Make flx/fly available to other threads by synchronizing
         __syncthreads();
 
         value_t flx_imj;
         if(threadIdx.x == 0) {
             // leftmost in block, need to compute left dependency ourselves
             value_t lap_imj = 4 * in[NEIGHBOR(i, j, k, -1, 0, 0)]
                 - in[NEIGHBOR(i, j, k, -2, 0, 0)] - in[NEIGHBOR(i, j, k, 0, 0, 0)]
                 - in[NEIGHBOR(i, j, k, -1, -1, 0)] - in[NEIGHBOR(i, j, k, -1, +1, 0)];
             flx_imj = lap_ij - lap_imj;
             flx_imj = flx_imj * (in[INDEX(i, j, k)] - in[NEIGHBOR(i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
         } else {
             flx_imj = flx[NEIGHBOR(i, j, k, -1, 0, 0)];
         }
 
         value_t fly_ijm;
         if(threadIdx.y == 0) {
             // need to also calculate lap for j - 1 as we are at boundary
             value_t lap_ijm = 4 * in[NEIGHBOR(i, j, k, 0, -1, 0)]
                     - in[NEIGHBOR(i, j, k, -1, -1, 0)] - in[NEIGHBOR(i, j, k, +1, -1, 0)]
                     - in[NEIGHBOR(i, j, k, 0, -2, 0)] - in[NEIGHBOR(i, j, k, 0, 0, 0)];
             fly_ijm = lap_ij - lap_ijm;
             fly_ijm = fly_ijm * (in[INDEX(i, j, k)] - in[NEIGHBOR(i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;
         } else {
             fly_ijm = fly[NEIGHBOR(i, j, k, 0, -1, 0)];
         }
 
         out[INDEX(i, j, k)] = in[INDEX(i, j, k)]
                 - coeff[INDEX(i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
 
     }