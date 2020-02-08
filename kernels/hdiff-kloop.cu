template<typename value_t>
__global__
void hdiff_kloop(const coord3 max_coord,
                 GRID_ARGS
                 const value_t *in,
                 value_t *out,
                 const value_t *coeff) {

    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(!(IS_IN_BOUNDS(i, j, 0))) {
        return;
    }
    
    for(int k = 0; k < max_coord.z; k++) {
        int idx = INDEX(i, j, k);
        const value_t lap_ij = 
            4 * in[NEIGHBOR(idx, 0, 0, 0)] 
            - in[NEIGHBOR(idx, -1, 0, 0)] - in[NEIGHBOR(idx, +1, 0, 0)]
            - in[NEIGHBOR(idx, 0, -1, 0)] - in[NEIGHBOR(idx, 0, +1, 0)];
        const value_t lap_imj = 
            4 * in[NEIGHBOR(idx, -1, 0, 0)]
            - in[NEIGHBOR(idx, -2, 0, 0)] - in[NEIGHBOR(idx, 0, 0, 0)]
            - in[NEIGHBOR(idx, -1, -1, 0)] - in[NEIGHBOR(idx, -1, +1, 0)];
        const value_t lap_ipj =
            4 * in[NEIGHBOR(idx, +1, 0, 0)]
            - in[NEIGHBOR(idx, 0, 0, 0)] - in[NEIGHBOR(idx, +2, 0, 0)]
            - in[NEIGHBOR(idx, +1, -1, 0)] - in[NEIGHBOR(idx, +1, +1, 0)];
        const value_t lap_ijm =
            4 * in[NEIGHBOR(idx, 0, -1, 0)]
            - in[NEIGHBOR(idx, -1, -1, 0)] - in[NEIGHBOR(idx, +1, -1, 0)]
            - in[NEIGHBOR(idx, 0, -2, 0)] - in[NEIGHBOR(idx, 0, 0, 0)];
        const value_t lap_ijp =
            4 * in[NEIGHBOR(idx, 0, +1, 0)]
            - in[NEIGHBOR(idx, -1, +1, 0)] - in[NEIGHBOR(idx, +1, +1, 0)]
            - in[NEIGHBOR(idx, 0, 0, 0)] - in[NEIGHBOR(idx, 0, +2, 0)];

        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[NEIGHBOR(idx, +1, 0, 0)] - in[idx]) > 0 ? 0 : flx_ij;
        value_t flx_imj = lap_ij - lap_imj;
        flx_imj = flx_imj * (in[idx] - in[NEIGHBOR(idx, -1, 0, 0)]) > 0 ? 0 : flx_imj;
        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[NEIGHBOR(idx, 0, +1, 0)] - in[idx]) > 0 ? 0 : fly_ij;
        value_t fly_ijm = lap_ij - lap_ijm;
        fly_ijm = fly_ijm * (in[idx] - in[NEIGHBOR(idx, 0, -1, 0)]) > 0 ? 0 : fly_ijm;

        out[idx] =
            in[idx]
            - coeff[idx] * (flx_ij - flx_imj + fly_ij - fly_ijm);
    }
}