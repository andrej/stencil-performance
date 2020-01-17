/** Optimized j-loop implementation. Iterates over Y (j) dimension and re-uses
 * previously done computations in this direction. The reason the iteration is
 * done over the Y-direction is because this way all threads access neighboring
 * cells (in data) in each loop iteration, so memory accesses coalesce. */
template<typename value_t>
__global__
void hdiff_jloop(const HdiffCudaBase::Info info,
                 GRID_ARGS
                 const int j_per_thread,
                 const value_t *in,
                 value_t *out,
                 const value_t *coeff) {
    
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int k = threadIdx.z + blockIdx.z*blockDim.z;
    int j_start = threadIdx.y*j_per_thread + blockIdx.y*blockDim.y*j_per_thread;
    if(!(IS_IN_BOUNDS(i, j_start, k))) {
        return;
    }

    int idx = INDEX(i, j_start, k);

    int j_stop = j_start + j_per_thread;
    if(j_stop > info.max_coord.y) {
        j_stop = info.max_coord.y;
    }
    
    // first calculation outside of loop will be shifted into lap_ijm / fly_ijm on first iteration
    value_t lap_ij = 4 * in[NEIGHBOR(idx, 0, -1, 0)]
                        - in[NEIGHBOR(idx, -1, -1, 0)] - in[NEIGHBOR(idx, +1, -1, 0)]
                        - in[NEIGHBOR(idx, 0, -2, 0)] - in[NEIGHBOR(idx, 0, 0, 0)];
    
    value_t lap_ijp = 4 * in[NEIGHBOR(idx, 0, 0, 0)] 
                        - in[NEIGHBOR(idx, -1, 0, 0)] - in[NEIGHBOR(idx, +1, 0, 0)]
                        - in[NEIGHBOR(idx, 0, -1, 0)] - in[NEIGHBOR(idx, 0, +1, 0)];
    
    value_t fly_ij = lap_ijp - lap_ij;
    fly_ij = fly_ij * (in[INDEX(i, j_start, k)] - in[NEIGHBOR(idx, 0, -1, 0)]) > 0 ? 0 : fly_ij;


    // j-loop, shifts results from previous round for reuse
    #pragma unroll 2
    for(int j = j_start; j < j_stop; j++) {

        idx = INDEX(i, j, k);

        // shift results from previous iteration
        //value_t lap_ijm = lap_ij;
        lap_ij = lap_ijp;
        value_t fly_ijm = fly_ij;

        // x direction dependencies are recalculated for every cell
        value_t lap_imj = 
            4 * in[NEIGHBOR(idx, -1, 0, 0)]
            - in[NEIGHBOR(idx, -2, 0, 0)] - in[NEIGHBOR(idx, 0, 0, 0)]
            - in[NEIGHBOR(idx, -1, -1, 0)] - in[NEIGHBOR(idx, -1, +1, 0)];
        value_t lap_ipj =
            4 * in[NEIGHBOR(idx, +1, 0, 0)]
            - in[NEIGHBOR(idx, 0, 0, 0)] - in[NEIGHBOR(idx, +2, 0, 0)]
            - in[NEIGHBOR(idx, +1, -1, 0)] - in[NEIGHBOR(idx, +1, +1, 0)];

        // will be reused as lap_ij in next iteration
        lap_ijp =
            4 * in[NEIGHBOR(idx, 0, +1, 0)]
            - in[NEIGHBOR(idx, -1, +1, 0)] - in[NEIGHBOR(idx, +1, +1, 0)]
            - in[NEIGHBOR(idx, 0, 0, 0)] - in[NEIGHBOR(idx, 0, +2, 0)];

        // x direction dependencies are recalculated for every cell
        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[NEIGHBOR(idx, +1, 0, 0)] - in[idx]) > 0 ? 0 : flx_ij;
        value_t flx_imj = lap_ij - lap_imj;
        flx_imj = flx_imj * (in[idx] - in[NEIGHBOR(idx, -1, 0, 0)]) > 0 ? 0 : flx_imj;
        
        // will be reused as fly_ijm in next iteration
        fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[NEIGHBOR(idx, 0, +1, 0)] - in[idx]) > 0 ? 0 : fly_ij;

        out[idx] =
            in[idx]
            - coeff[idx] * (flx_ij - flx_imj + fly_ij - fly_ijm);
    }

}