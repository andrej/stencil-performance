template<typename value_t>
__global__
void laplap_idxvar_kloop_sliced(GRID_ARGS const int k_per_thread, const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k_start = (blockIdx.z*blockDim.z + threadIdx.z) * k_per_thread;
    if(!(IS_IN_BOUNDS(i, j, k_start))) {
        return;
    }

    const int k_stop = (k_start + k_per_thread < max_coord.z ? k_start + k_per_thread : max_coord.z); 

    int center        = INDEX(i, j, 0);
    int left          = NEIGHBOR(center, -1,  0, 0);
    int right         = NEIGHBOR(center, +1,  0, 0);
    int top           = NEIGHBOR(center,  0, -1, 0);
    int bottom        = NEIGHBOR(center,  0, +1, 0);
    #ifdef CHASING
        int leftleft      = NEIGHBOR(left, -1,  0, 0);
        int topleft       = NEIGHBOR(top,  -1,  0,  0);
        int bottomleft    = NEIGHBOR(bottom, -1,  0,  0);
        int topright      = NEIGHBOR(top,  +1,  0,  0);
        int rightright    = NEIGHBOR(right, +1,  0,  0);
        int bottomright   = NEIGHBOR(bottom, +1,  0,  0);
        int toptop        = NEIGHBOR(top, 0, -1,  0);
        int bottombottom  = NEIGHBOR(bottom, 0, +1,  0);
    #else
        int leftleft      = DOUBLE_NEIGHBOR(center, -1,  0, 0, -1,  0,  0);
        int topleft       = DOUBLE_NEIGHBOR(center,  0, -1, 0, -1,  0,  0);
        int bottomleft    = DOUBLE_NEIGHBOR(center,  0, +1, 0, -1,  0,  0);
        int topright      = DOUBLE_NEIGHBOR(center,  0, -1, 0, +1,  0,  0);
        int rightright    = DOUBLE_NEIGHBOR(center, +1,  0, 0, +1,  0,  0);
        int bottomright   = DOUBLE_NEIGHBOR(center,  0, +1, 0, +1,  0,  0);
        int toptop        = DOUBLE_NEIGHBOR(center,  0, -1, 0,  0, -1,  0);
        int bottombottom  = DOUBLE_NEIGHBOR(center,  0, +1, 0,  0, +1,  0);
    #endif

    center        = Z_NEIGHBOR(center, k_start);
    left          = Z_NEIGHBOR(left, k_start);
    leftleft      = Z_NEIGHBOR(leftleft, k_start);
    topleft       = Z_NEIGHBOR(topleft, k_start);
    bottomleft    = Z_NEIGHBOR(bottomleft, k_start);
    right         = Z_NEIGHBOR(right, k_start);
    topright      = Z_NEIGHBOR(topright, k_start);
    rightright    = Z_NEIGHBOR(rightright, k_start);
    bottomright   = Z_NEIGHBOR(bottomright, k_start);
    top           = Z_NEIGHBOR(top, k_start);
    toptop        = Z_NEIGHBOR(toptop, k_start);
    bottom        = Z_NEIGHBOR(bottom, k_start);
    bottombottom  = Z_NEIGHBOR(bottombottom, k_start);

    #pragma unroll 4
    for(int k = k_start; k < k_stop; k++) {
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

        // index updates
        center        = Z_NEIGHBOR(center, +1);
        left          = Z_NEIGHBOR(left, +1);
        leftleft      = Z_NEIGHBOR(leftleft, +1);
        topleft       = Z_NEIGHBOR(topleft, +1);
        bottomleft    = Z_NEIGHBOR(bottomleft, +1);
    
        right         = Z_NEIGHBOR(right, +1);
        topright      = Z_NEIGHBOR(topright, +1);
        rightright    = Z_NEIGHBOR(rightright, +1);
        bottomright   = Z_NEIGHBOR(bottomright, +1);
    
        top           = Z_NEIGHBOR(top, +1);
        toptop        = Z_NEIGHBOR(toptop, +1);
    
        bottom        = Z_NEIGHBOR(bottom, +1);
        bottombottom  = Z_NEIGHBOR(bottombottom, +1);
    }
}