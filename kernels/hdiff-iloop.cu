/** I-loop. This should be really slow and is just to demonstrate the
 * advantages of the coalescing accesses in the j-loop variant. */
template<typename value_t>
__global__
void hdiff_iloop(const HdiffCudaBase::Info info,
                 GRID_ARGS
                 const int i_per_thread,
                 const value_t *in,
                 value_t *out,
                 const value_t *coeff) {
    
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    const int k = threadIdx.z + blockIdx.z*blockDim.z;
    int i_start = threadIdx.x*i_per_thread + blockIdx.x*blockDim.x*i_per_thread;
    if(j >= info.max_coord.y || i_start >= info.max_coord.x || k >= info.max_coord.z) {
        return;
    }

    const int idx = INDEX(i, j, k);

    int i_stop = i_start + i_per_thread;
    if(i_stop > info.max_coord.x) {
        i_stop = info.max_coord.x;
    }
    
    // first calculation outside of loop will be shifted into lap_imj / flx_imj on first iteration
    // therefore lap_ij => lap_imj und lap_ipj => lap_ij
    value_t lap_ij = 4 * in[NEIGHBOR(i_start, j, k, -1, 0, 0)] /* center */
                        - in[NEIGHBOR(i_start, j, k, -2, 0, 0)] /* left */ - in[NEIGHBOR(i_start, j, k, 0, 0, 0)] /* right */
                        - in[NEIGHBOR(i_start, j, k, -1, -1, 0)] /* top */ - in[NEIGHBOR(i_start, j, k, -1, +1, 0)] /* bottom */;
    
    value_t lap_ipj = 4 * in[NEIGHBOR(i_start, j, k, 0, 0, 0)]  /* center */
                        - in[NEIGHBOR(i_start, j, k, -1, 0, 0)] /* left */ - in[NEIGHBOR(i_start, j, k, +1, 0, 0)] /* right */ 
                        - in[NEIGHBOR(i_start, j, k, 0, -1, 0)] /* top */ - in[NEIGHBOR(i_start, j, k, 0, +1, 0)]; /* bottom */
    
    value_t flx_ij = lap_ipj - lap_ij;
    flx_ij = flx_ij * (in[INDEX(i_start, j, k)] - in[NEIGHBOR(i_start, j, k, -1, 0, 0)]) > 0 ? 0 : flx_ij;

    // i-loop, shifts results from previous round for reuse
    for(int i = i_start; i < i_stop; i++) {

        // shift results from previous iteration
        //value_t lap_imj = lap_ij;
        lap_ij = lap_ipj;
        value_t flx_imj = flx_ij;

        // y direction dependencies are recalculated for every cell
        value_t lap_ijm = 
            4 * in[NEIGHBOR(idx, 0, -1, 0)] /* center */
            - in[NEIGHBOR(idx, -1, -1, 0)] /* left */ - in[NEIGHBOR(idx, +1, -1, 0)] /* right */
            - in[NEIGHBOR(idx, 0, -2, 0)] /* top */ - in[NEIGHBOR(idx, 0, 0, 0)] /* bottom */;
        value_t lap_ijp =
            4 * in[NEIGHBOR(idx, 0, +1, 0)] /* center */
            - in[NEIGHBOR(idx, -1, +1, 0)] /* left */ - in[NEIGHBOR(idx, +1, +1, 0)] /* right */
            - in[NEIGHBOR(idx, 0, 0, 0)] /* top */ - in[NEIGHBOR(idx, 0, +2, 0)] /* bottom */;

        // will be reused as lap_ij in next iteration
        lap_ipj =
            4 * in[NEIGHBOR(idx, +1, 0, 0)] /* center */
            - in[NEIGHBOR(idx, 0, 0, 0)] /* left */ - in[NEIGHBOR(idx, +2, 0, 0)] /* right */
            - in[NEIGHBOR(idx, +1, -1, 0)] /* top */ - in[NEIGHBOR(idx, +1, +1, 0)] /* bottom */;

        // y direction dependencies are recalculated for every cell
        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[NEIGHBOR(idx, 0, +1, 0)] - in[idx]) > 0 ? 0 : fly_ij;
        value_t fly_ijm = lap_ij - lap_ijm;
        fly_ijm = fly_ijm * (in[idx] - in[NEIGHBOR(idx, 0, -1, 0)]) > 0 ? 0 : fly_ijm;

        // will be reused as flx_ijm in next iteration
        flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[NEIGHBOR(idx, +1, 0, 0)] - in[idx]) > 0 ? 0 : flx_ij;

        out[idx] =
            in[idx]
            - coeff[idx] * (flx_ij - flx_imj + fly_ij - fly_ijm);
    }

}