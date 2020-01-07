#define LAP_OFFS(in, i, j, k, i_, j_, k_) \
    4 * in[NEIGHBOR(i, j, k, i_, j_, k_)] \
    - in[NEIGHBOR(i, j, k, -1+(i_), j_, k_)] \
    - in[NEIGHBOR(i, j, k, +1+(i_), j_, k_)] \
    - in[NEIGHBOR(i, j, k, i_, -1+(j_), k_)] \
    - in[NEIGHBOR(i, j, k, i_, +1+(j_), k_)]

template<typename value_t>
__global__
void laplap_naive(GRID_ARGS const coord3 halo, const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + halo.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z + halo.z;
    if(i >= max_coord.x || j >= max_coord.y || k >= max_coord.z) {
        return;
    }
    const value_t lap_center       = LAP_OFFS(in, i, j, k,  0,  0, 0);
    const value_t lap_left         = LAP_OFFS(in, i, j, k, -1,  0, 0);
    const value_t lap_right        = LAP_OFFS(in, i, j, k, +1,  0, 0);
    const value_t lap_top          = LAP_OFFS(in, i, j, k,  0, -1, 0);
    const value_t lap_bottom       = LAP_OFFS(in, i, j, k,  0, +1, 0);
    out[INDEX(i, j, k)] = 4 * lap_center
                          - lap_left
                          - lap_right
                          - lap_top
                          - lap_bottom;
}