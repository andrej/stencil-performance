template<typename value_t>
__global__
void laplap_shared(GRID_ARGS const coord3 halo, const coord3 max_coord, const value_t *in, value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i >= max_coord.x || j >= max_coord.y || k >= max_coord.z) {
        return;
    }
    
    // Index Variables
    const int center        = INDEX(i, j, k);
    const int s_center      = SMEM_INDEX((int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z);
    const int left          = NEIGHBOR(center, -1,  0, 0);
    const int leftleft      = NEIGHBOR(  left, -1,  0, 0);
    const int topleft       = NEIGHBOR(  left,  0, -1, 0);
    const int bottomleft    = NEIGHBOR(  left,  0, +1, 0);
    const int right         = NEIGHBOR(center, +1,  0, 0);
    const int topright      = NEIGHBOR( right,  0, -1, 0);
    const int rightright    = NEIGHBOR( right, +1,  0, 0);
    const int bottomright   = NEIGHBOR( right,  0, +1, 0);
    const int top           = NEIGHBOR(center,  0, -1, 0);
    const int toptop        = NEIGHBOR(   top,  0, -1, 0);
    const int bottom        = NEIGHBOR(center,  0, +1, 0);
    const int bottombottom  = NEIGHBOR(bottom,  0, +1, 0);

    // Shared memory local grid holding first laplace iteration
    extern __shared__ char smem[];
    value_t *local_lap = (value_t *)smem;
    
    // Calculate own laplace
    const value_t lap_center= 4 * in[center] - in[left] - in[right] - in[top] - in[bottom];
    local_lap[s_center] = lap_center;

    // Sync threads to enable access to their laplace calculations
    __syncthreads();

    value_t lap_left;
    if(threadIdx.x == 0) {
        lap_left = 4 * in[left] - in[leftleft] - in[center] - in[topleft] - in[bottomleft];
    } else {
        lap_left = local_lap[SMEM_NEIGHBOR(s_center, -1, 0, 0)];
    }

    value_t lap_right;
    if(threadIdx.x == blockDim.x-1 || i == max_coord.x-1) {
        lap_right = 4 * in[right] - in[center] - in[rightright] - in[topright] - in[bottomright];
    } else {
        lap_right = local_lap[SMEM_NEIGHBOR(s_center, +1, 0, 0)];
    }

    value_t lap_top;
    if(threadIdx.y == 0) {
        lap_top   = 4 * in[top] - in[topleft] - in[topright] - in[toptop] - in[center];
    } else {
        lap_right = local_lap[SMEM_NEIGHBOR(s_center, 0, -1, 0)];
    }
    
    value_t lap_bottom;
    if(threadIdx.y == blockDim.y-1 || j == max_coord.y-1) {
        lap_bottom= 4 * in[bottom] - in[bottomleft] - in[bottomright] - in[center] - in[bottombottom];
    } else {
        lap_right = local_lap[SMEM_NEIGHBOR(s_center, 0, +1, 0)];
    }
    
    out[center] = 4 * lap_center - lap_left - lap_right - lap_top - lap_bottom;
}