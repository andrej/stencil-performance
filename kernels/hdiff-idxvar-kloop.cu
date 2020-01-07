/** K-Loop using variables to store indices (cache)
 * This kernel makes use of the regularity of the grid in the Z-direction.
 * Instead of naively resolving the neighborship relations at each k-step,
 * The locations of the neighboring cells are calculated at one level and
 * then reused, with the constant (regular) Z-step at each k-iteration.
 *
 * Required macros:
 * - GRID_ARGS
 * - INDEX
 * - NEIGHBOR_OF_INDEX
 * - NEXT_Z_NEIGHBOR_OF_INDEX
 */
template<typename value_t>
__global__
void hdiff_idxvar_kloop(const HdiffCudaBase::Info info,
                        GRID_ARGS
                        const value_t *in,
                        value_t *out,
                        const value_t *coeff) {

    const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
    if(i >= info.max_coord.x || j >= info.max_coord.y) {
        return;
    }

    /** Store index offsets for the current x and y coordinate, so we do
    * not have to recalculate them in every k-iteration. Instead, with
    * each iteration, the k-stride is simply added once -- thus making
    * use of the regularity of the grid in z-direction. 
    * idx of neighbor X Y Z = n_X_Y_Z with p for positive offset and 
    * n for negative offset. */
    int n_0_0_0       = INDEX(i, j, 0);
    int n_0_n1_0      = NEIGHBOR_OF_INDEX(n_0_0_0, 0, -1, 0);
    int n_0_n2_0      = NEIGHBOR_OF_INDEX(n_0_n1_0, 0, -1, 0);
    int n_n1_0_0      = NEIGHBOR_OF_INDEX(n_0_0_0, -1, 0, 0);
    int n_n1_n1_0     = NEIGHBOR_OF_INDEX(n_n1_0_0, 0, -1, 0);
    int n_n2_0_0      = NEIGHBOR_OF_INDEX(n_n1_0_0, -1, 0, 0);
    int n_0_p1_0      = NEIGHBOR_OF_INDEX(n_0_0_0, 0, +1, 0);
    int n_0_p2_0      = NEIGHBOR_OF_INDEX(n_0_p1_0, 0, +1, 0);
    int n_p1_0_0      = NEIGHBOR_OF_INDEX(n_0_0_0, +1, 0, 0);
    int n_p1_p1_0     = NEIGHBOR_OF_INDEX(n_p1_0_0, 0, +1, 0);
    int n_p2_0_0      = NEIGHBOR_OF_INDEX(n_p1_0_0, +1, 0, 0);
    int n_n1_p1_0     = NEIGHBOR_OF_INDEX(n_n1_0_0, 0, +1, 0);
    int n_p1_n1_0     = NEIGHBOR_OF_INDEX(n_p1_0_0, 0, -1, 0);

    for (int k = info.halo.z; k < info.max_coord.z; k++) {

        value_t lap_ij = 
            4 * in[n_0_0_0] 
            - in[n_n1_0_0] - in[n_p1_0_0]
            - in[n_0_n1_0] - in[n_0_p1_0];
        value_t lap_imj = 
            4 * in[n_n1_0_0]
            - in[n_n2_0_0] - in[n_0_0_0]
            - in[n_n1_n1_0] - in[n_n1_p1_0];
        value_t lap_ipj =
            4 * in[n_p1_0_0]
            - in[n_0_0_0] - in[n_p2_0_0]
            - in[n_p1_n1_0] - in[n_p1_p1_0];
        value_t lap_ijm =
            4 * in[n_0_n1_0]
            - in[n_n1_n1_0] - in[n_p1_n1_0]
            - in[n_0_n2_0] - in[n_0_0_0];
        value_t lap_ijp =
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
        

        // Make use of regularity in Z-direciton: neighbors are exactly the
        // same, just one Z-stride apart.
        n_0_0_0       = NEXT_Z_NEIGHBOR_OF_INDEX(n_0_0_0);
        n_0_n1_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_0_n1_0);
        n_0_n2_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_0_n2_0);
        n_n1_0_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_n1_0_0);
        n_n1_n1_0     = NEXT_Z_NEIGHBOR_OF_INDEX(n_n1_n1_0);
        n_n2_0_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_n2_0_0);
        n_0_p1_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_0_p1_0);
        n_0_p2_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_0_p2_0);
        n_p1_0_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_p1_0_0);
        n_p1_p1_0     = NEXT_Z_NEIGHBOR_OF_INDEX(n_p1_p1_0);
        n_p2_0_0      = NEXT_Z_NEIGHBOR_OF_INDEX(n_p2_0_0);
        n_n1_p1_0     = NEXT_Z_NEIGHBOR_OF_INDEX(n_n1_p1_0);
        n_p1_n1_0     = NEXT_Z_NEIGHBOR_OF_INDEX(n_p1_n1_0);

    }

}