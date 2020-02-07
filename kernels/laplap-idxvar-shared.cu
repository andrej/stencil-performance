template<typename value_t>
__global__
void laplap_idxvar_shared(GRID_ARGS const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(!(IS_IN_BOUNDS(i, j, k))) {
        return;
    }

    // Indexing stuff
    extern __shared__ char smem[];
    const int local_idx = (threadIdx.x + threadIdx.y*blockDim.x) * 13;
    int * __restrict__ idxvars = &((int *)smem)[local_idx];
    const bool is_first = threadIdx.z % blockDim.z == 0;
    const int k_step = K_STEP;
    int center, left, leftleft, topleft, bottomleft, right, topright, rightright, bottomright, top, toptop, bottom, bottombottom;
    if(is_first) {
        center        = idxvars[0] = INDEX(i, j, 0);
        left          = idxvars[1] = NEIGHBOR(center, -1,  0, 0);
        right         = idxvars[5] = NEIGHBOR(center, +1,  0, 0);
        top           = idxvars[9] = NEIGHBOR(center,  0, -1, 0);
        bottom        = idxvars[11] = NEIGHBOR(center,  0, +1, 0);
        #ifdef CHASING
        leftleft      = idxvars[2] = NEIGHBOR(left, -1,  0,  0);
        topleft       = idxvars[3] = NEIGHBOR(top,  -1,  0,  0);
        bottomleft    = idxvars[4] = NEIGHBOR(bottom, -1,  0,  0);
        topright      = idxvars[6] = NEIGHBOR(top, +1,  0,  0);
        rightright    = idxvars[7] = NEIGHBOR(right, +1,  0,  0);
        bottomright   = idxvars[8] = NEIGHBOR(bottom, +1,  0,  0);
        toptop        = idxvars[10] = NEIGHBOR(top, 0, -1,  0);
        bottombottom  = idxvars[12] = NEIGHBOR(bottom, 0, +1,  0);
        #else
        leftleft      = idxvars[2] = DOUBLE_NEIGHBOR(center, -1,  0, 0, -1,  0,  0);
        topleft       = idxvars[3] = DOUBLE_NEIGHBOR(center,  0, -1, 0, -1,  0,  0);
        bottomleft    = idxvars[4] = DOUBLE_NEIGHBOR(center,  0, +1, 0, -1,  0,  0);
        topright      = idxvars[6] = DOUBLE_NEIGHBOR(center,  0, -1, 0, +1,  0,  0);
        rightright    = idxvars[7] = DOUBLE_NEIGHBOR(center, +1,  0, 0, +1,  0,  0);
        bottomright   = idxvars[8] = DOUBLE_NEIGHBOR(center,  0, +1, 0, +1,  0,  0);
        toptop        = idxvars[10] = DOUBLE_NEIGHBOR(center,  0, -1, 0,  0, -1,  0);
        bottombottom  = idxvars[12] = DOUBLE_NEIGHBOR(center,  0, +1, 0,  0, +1,  0);
        #endif
    }
    __syncthreads();
    if(!is_first) {
        center        = idxvars[0];
        left          = idxvars[1];     leftleft      = idxvars[2];     topleft       = idxvars[3];     bottomleft    = idxvars[4];
        right         = idxvars[5];     topright      = idxvars[6];     rightright    = idxvars[7];     bottomright   = idxvars[8];
        top           = idxvars[9];     toptop        = idxvars[10];
        bottom        = idxvars[11];    bottombottom  = idxvars[12];
    }
    center        += k_step;
    left          += k_step;    leftleft      += k_step;    topleft       += k_step;    bottomleft    += k_step;
    right         += k_step;    topright      += k_step;    rightright    += k_step;    bottomright   += k_step;
    top           += k_step;    toptop        += k_step;
    bottom        += k_step;    bottombottom  += k_step;

    // Actual laplace calculations
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