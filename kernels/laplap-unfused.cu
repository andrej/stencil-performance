template<typename value_t>
__global__
void lap(GRID_ARGS coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(!(IS_IN_BOUNDS(i, j, k))) {
        return;
    }
    const int idx = INDEX(i, j, k);
    out[idx] = 4 * in[idx]
                          - in[NEIGHBOR(idx, -1, 0, 0)]
                          - in[NEIGHBOR(idx, +1, 0, 0)]
                          - in[NEIGHBOR(idx, 0, -1, 0)]
                          - in[NEIGHBOR(idx, 0, +1, 0)];
}