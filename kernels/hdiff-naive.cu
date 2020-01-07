/* Horizontal Diffusion Kernel
 * Naive: Every thread computes all of its dependencies (i.e. required 
 * neighboring values are computed as well.)
 * Fused: All intermediate computations (lap, flx, fly) are computed within
 * one kernel.
 *
 * Required Macro Definitions when using this kernel:
 * - GRID_ARGS; arguments that need to be passed to the kernel that are
 *   specific to the grid used. Will get passed ot INDEX and NEIGHBOR macros.
 * - INDEX(x, y, z)
 * - NEIGHBOR(x, y, z, x_offs, y_offs, z_offs) <-- note that this macro must support +2 -2 neighbors as well!
 */
template<typename value_t>
__global__
void hdiff_naive(HdiffCudaBase::Info info,
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

    const value_t lap_ij = 
        4 * in[NEIGHBOR(i, j, k, 0, 0, 0)] 
        - in[NEIGHBOR(i, j, k, -1, 0, 0)] - in[NEIGHBOR(i, j, k, +1, 0, 0)]
        - in[NEIGHBOR(i, j, k, 0, -1, 0)] - in[NEIGHBOR(i, j, k, 0, +1, 0)];
    const value_t lap_imj = 
        4 * in[NEIGHBOR(i, j, k, -1, 0, 0)]
        - in[NEIGHBOR(i, j, k, -2, 0, 0)] - in[NEIGHBOR(i, j, k, 0, 0, 0)]
        - in[NEIGHBOR(i, j, k, -1, -1, 0)] - in[NEIGHBOR(i, j, k, -1, +1, 0)];
    const value_t lap_ipj =
        4 * in[NEIGHBOR(i, j, k, +1, 0, 0)]
        - in[NEIGHBOR(i, j, k, 0, 0, 0)] - in[NEIGHBOR(i, j, k, +2, 0, 0)]
        - in[NEIGHBOR(i, j, k, +1, -1, 0)] - in[NEIGHBOR(i, j, k, +1, +1, 0)];
    const value_t lap_ijm =
        4 * in[NEIGHBOR(i, j, k, 0, -1, 0)]
        - in[NEIGHBOR(i, j, k, -1, -1, 0)] - in[NEIGHBOR(i, j, k, +1, -1, 0)]
        - in[NEIGHBOR(i, j, k, 0, -2, 0)] - in[NEIGHBOR(i, j, k, 0, 0, 0)];
    const value_t lap_ijp =
        4 * in[NEIGHBOR(i, j, k, 0, +1, 0)]
        - in[NEIGHBOR(i, j, k, -1, +1, 0)] - in[NEIGHBOR(i, j, k, +1, +1, 0)]
        - in[NEIGHBOR(i, j, k, 0, 0, 0)] - in[NEIGHBOR(i, j, k, 0, +2, 0)];

    value_t flx_ij = lap_ipj - lap_ij;
    flx_ij = flx_ij * (in[NEIGHBOR(i, j, k, +1, 0, 0)] - in[INDEX(i, j, k)]) > 0 ? 0 : flx_ij;

    value_t flx_imj = lap_ij - lap_imj;
    flx_imj = flx_imj * (in[INDEX(i, j, k)] - in[NEIGHBOR(i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;

    value_t fly_ij = lap_ijp - lap_ij;
    fly_ij = fly_ij * (in[NEIGHBOR(i, j, k, 0, +1, 0)] - in[INDEX(i, j, k)]) > 0 ? 0 : fly_ij;

    value_t fly_ijm = lap_ij - lap_ijm;
    fly_ijm = fly_ijm * (in[INDEX(i, j, k)] - in[NEIGHBOR(i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;

    out[INDEX(i, j, k)] =
        in[INDEX(i, j, k)]
        - coeff[INDEX(i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);

}