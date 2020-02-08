/** Looks up neighbors in beginning and stores them in index variables,
 * instead of recomputing them for every access using the INDEX() and 
 * NEIGHBOR() macros. This is primarily an optimization for the unstructured
 * case.
 * 
 * Required macros:
 *  - GRID_ARGS
 *  - INDEX
 *  - NEIGHBOR
 *  - K_STEP
 */
template<typename value_t>
__global__
void hdiff_idxvar(const coord3 max_coord,
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
    
    int n_0_0_0       = INDEX(i, j, 0);
    int n_0_n1_0      = NEIGHBOR(n_0_0_0, 0, -1, 0); /* left */
    int n_n1_0_0      = NEIGHBOR(n_0_0_0, -1, 0, 0); /* top */
    int n_0_p1_0      = NEIGHBOR(n_0_0_0, 0, +1, 0); /* right */
    int n_p1_0_0      = NEIGHBOR(n_0_0_0, +1, 0, 0); /* bottom */
    #ifdef CHASING
        int n_0_n2_0      = NEIGHBOR(n_0_n1_0, 0, -1, 0); /* 2 top */
        int n_n1_n1_0     = NEIGHBOR(n_n1_0_0, 0, -1, 0); /* top left */
        int n_n2_0_0      = NEIGHBOR(n_n1_0_0, -1, 0, 0); /* 2 left */
        int n_0_p2_0      = NEIGHBOR(n_0_p1_0, 0, +1, 0); /* 2 right */
        int n_p1_p1_0     = NEIGHBOR(n_p1_0_0, 0, +1, 0); /* bottom right */
        int n_p2_0_0      = NEIGHBOR(n_p1_0_0, +1, 0, 0); /* 2 right */
        int n_n1_p1_0     = NEIGHBOR(n_n1_0_0, 0, +1, 0); /* bottom left */
        int n_p1_n1_0     = NEIGHBOR(n_p1_0_0, 0, -1, 0); /* top right */
    #else
        int n_0_n2_0      = DOUBLE_NEIGHBOR(n_0_0_0, 0, -1, 0, 0, -1, 0); /* 2 top */
        int n_n1_n1_0     = DOUBLE_NEIGHBOR(n_0_0_0, 0, -1, 0, -1, 0, 0); /* top left */
        int n_n2_0_0      = DOUBLE_NEIGHBOR(n_0_0_0, -1, 0, 0, -1, 0, 0); /* 2 left */
        int n_0_p2_0      = DOUBLE_NEIGHBOR(n_0_0_0, 0, +1, 0, 0, +1, 0); /* 2 bottom */
        int n_p1_p1_0     = DOUBLE_NEIGHBOR(n_0_0_0, 0, +1, 0, +1, 0, 0); /* bottom right */
        int n_p2_0_0      = DOUBLE_NEIGHBOR(n_0_0_0, +1, 0, 0, +1, 0, 0); /* 2 right */
        int n_n1_p1_0     = DOUBLE_NEIGHBOR(n_0_0_0, -1, 0, 0, 0, +1, 0); /* bottom left */
        int n_p1_n1_0     = DOUBLE_NEIGHBOR(n_0_0_0,  0, -1, 0, +1, 0, 0); /* top right */
    #endif

    const int k_step  = K_STEP;
    n_0_0_0           += k_step;
    n_0_n1_0          += k_step; /* left */
    n_0_n2_0          += k_step; /* 2 left */
    n_n1_0_0          += k_step; /* top */
    n_n1_n1_0         += k_step; /* top left */
    n_n2_0_0          += k_step; /* 2 top */
    n_0_p1_0          += k_step; /* right */
    n_0_p2_0          += k_step; /* 2 right */
    n_p1_0_0          += k_step; /* bottom */
    n_p1_p1_0         += k_step; /* bottom right */
    n_p2_0_0          += k_step; /* 2 bottom */
    n_n1_p1_0         += k_step; /* top right */
    n_p1_n1_0         += k_step; /* bottom left */

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