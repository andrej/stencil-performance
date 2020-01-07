template<typename value_t>
__global__
void laplap_idxvar_kloop(GRID_ARGS const coord3 halo, const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + halo.y;
    if(i >= max_coord.x || j >= max_coord.y) {
        return;
    }

    int center        = INDEX(i, j, 0);
    int left          = NEIGHBOR_OF_INDEX(center, -1,  0, 0);
    int leftleft      = NEIGHBOR_OF_INDEX(  left, -1,  0, 0);
    int topleft       = NEIGHBOR_OF_INDEX(  left,  0, -1, 0);
    int bottomleft    = NEIGHBOR_OF_INDEX(  left,  0, +1, 0);

    int right         = NEIGHBOR_OF_INDEX(center, +1,  0, 0);
    int topright      = NEIGHBOR_OF_INDEX( right,  0, -1, 0);
    int rightright    = NEIGHBOR_OF_INDEX( right, +1,  0, 0);
    int bottomright   = NEIGHBOR_OF_INDEX( right,  0, +1, 0);

    int top           = NEIGHBOR_OF_INDEX(center,  0, -1, 0);
    int toptop        = NEIGHBOR_OF_INDEX(   top,  0, -1, 0);

    int bottom        = NEIGHBOR_OF_INDEX(center,  0, +1, 0);
    int bottombottom  = NEIGHBOR_OF_INDEX(bottom,  0, +1, 0);

    for(int k = halo.z; k < max_coord.z; k++) {
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
        center        = NEXT_Z_NEIGHBOR_OF_INDEX(center);
        left          = NEXT_Z_NEIGHBOR_OF_INDEX(left);
        leftleft      = NEXT_Z_NEIGHBOR_OF_INDEX(leftleft);
        topleft       = NEXT_Z_NEIGHBOR_OF_INDEX(topleft);
        bottomleft    = NEXT_Z_NEIGHBOR_OF_INDEX(bottomleft);
    
        right         = NEXT_Z_NEIGHBOR_OF_INDEX(right);
        topright      = NEXT_Z_NEIGHBOR_OF_INDEX(topright);
        rightright    = NEXT_Z_NEIGHBOR_OF_INDEX(rightright);
        bottomright   = NEXT_Z_NEIGHBOR_OF_INDEX(bottomright);
    
        top           = NEXT_Z_NEIGHBOR_OF_INDEX(top);
        toptop        = NEXT_Z_NEIGHBOR_OF_INDEX(toptop);
    
        bottom        = NEXT_Z_NEIGHBOR_OF_INDEX(bottom);
        bottombottom  = NEXT_Z_NEIGHBOR_OF_INDEX(bottombottom);
    }
}