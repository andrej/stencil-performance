template<typename value_t>
__global__
void laplap_idxvar(GRID_ARGS const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(!(IS_IN_BOUNDS(i, j, k))) {
        return;
    }

    const int center        = INDEX(i, j, k);
    const int left          = NEIGHBOR(center, -1,  0, 0);
    const int leftleft      = DOUBLE_NEIGHBOR(center, -1,  0, 0, -1,  0,  0);
    const int topleft       = DOUBLE_NEIGHBOR(center,  0, -1, 0, -1,  0,  0);
    const int bottomleft    = DOUBLE_NEIGHBOR(center,  0, +1, 0, -1,  0,  0);

    const int right         = NEIGHBOR(center, +1,  0, 0);
    const int topright      = DOUBLE_NEIGHBOR(center,  0, -1, 0, +1,  0,  0);
    const int rightright    = DOUBLE_NEIGHBOR(center, +1,  0, 0, +1,  0,  0);
    const int bottomright   = DOUBLE_NEIGHBOR(center,  0, +1, 0, +1,  0,  0);

    const int top           = NEIGHBOR(center,  0, -1, 0);
    const int toptop        = DOUBLE_NEIGHBOR(center,  0, -1, 0,  0, -1,  0);

    const int bottom        = NEIGHBOR(center,  0, +1, 0);
    const int bottombottom  = DOUBLE_NEIGHBOR(center,  0, +1, 0,  0, +1,  0);


    const value_t lap_center= 4 * in[center]
                                - in[left]
                                - in[right]
                                - in[top]
                                - in[bottom];
    const value_t lap_left  = 4 * in[left]
                                - in[leftleft]
                                - in[center]
                                - in[topleft]
                                - in[bottomleft];
    const value_t lap_right = 4 * in[right]
                                - in[center]
                                - in[rightright]
                                - in[topright]
                                - in[bottomright];
    const value_t lap_top   = 4 * in[top]
                                - in[topleft]
                                - in[topright]
                                - in[toptop]
                                - in[center];
    const value_t lap_bottom= 4 * in[bottom]
                                - in[bottomleft]
                                - in[bottomright]
                                - in[center]
                                - in[bottombottom];
    out[center]             = 4 * lap_center
                                - lap_left
                                - lap_right
                                - lap_top
                                - lap_bottom;
}