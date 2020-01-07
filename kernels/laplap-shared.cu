template<typename value_t>
__global__
void laplap_shared(GRID_ARGS const coord3 halo, const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + halo.y;
    if(i >= max_coord.x || j >= max_coord.y) {
        return;
    }

    extern __shared__ char smem[];
    double *lap = (double *)smem;

    int center        = INDEX(i, j, 0);
    int smem_center   = SMEM_INDEX(i, j, 0);
    int left          = NEIGHBOR_OF_INDEX(center, -1,  0, 0);

    int right         = NEIGHBOR_OF_INDEX(center, +1,  0, 0);
    int top           = NEIGHBOR_OF_INDEX(center,  0, -1, 0);
    int bottom        = NEIGHBOR_OF_INDEX(center,  0, +1, 0);

    for(int k = halo.z; k < max_coord.z; k++) {
        const value_t lap_center= 4 * in[center]
                                    - in[left]
                                    - in[right]
                                    - in[top]
                                    - in[bottom];
        


        const value_t lap_left  = lap[left];
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
        left          = NEXT_Z_NEIGHBOR_OF_INDEX(center);
        leftleft      = NEXT_Z_NEIGHBOR_OF_INDEX(  left);
        topleft       = NEXT_Z_NEIGHBOR_OF_INDEX(  left);
        bottomleft    = NEXT_Z_NEIGHBOR_OF_INDEX(  left);
    
        right         = NEXT_Z_NEIGHBOR_OF_INDEX(center);
        topright      = NEXT_Z_NEIGHBOR_OF_INDEX( right);
        righright     = NEXT_Z_NEIGHBOR_OF_INDEX( right);
        bottomright   = NEXT_Z_NEIGHBOR_OF_INDEX( right);
    
        top           = NEXT_Z_NEIGHBOR_OF_INDEX(center);
        toptop        = NEXT_Z_NEIGHBOR_OF_INDEX(   top);
    
        bottom        = NEXT_Z_NEIGHBOR_OF_INDEX(center);
        bottombottom  = NEXT_Z_NEIGHBOR_OF_INDEX(bottom);
    }
}