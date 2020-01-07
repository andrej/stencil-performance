template<typename value_t>
__global__
void lap(GRID_ARGS coord3 halo, coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + halo.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z + halo.z;
    if(i >= max_coord.x || j >= max_coord.y || k >= max_coord.z) {
        return;
    }
    out[INDEX(i, j, k)] = 4 * in[INDEX(i, j, k)]
                          - in[NEIGHBOR(i, j, k, -1, 0, 0)]
                          - in[NEIGHBOR(i, j, k, +1, 0, 0)]
                          - in[NEIGHBOR(i, j, k, 0, -1, 0)]
                          - in[NEIGHBOR(i, j, k, 0, +1, 0)];
}