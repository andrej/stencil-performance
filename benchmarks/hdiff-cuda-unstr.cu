#ifndef HDIFF_CUDA_UNSTR_H
#define HDIFF_CUDA_UNSTR_H
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstr {

    /** Variants of this benchmark. */
    enum Variant { RegNaive, RegKloop, UnstrNaive, UnstrKloop, UnstrIdxvars, UnstrSharedIdxvar };

    /** Information about this benchmark for use in the kernels. */
    struct Info {
        coord3 halo;
        coord3 max_coord;
    };

    /** Naive implementation of a unstructured grid horizontal diffusion
     * kernel. Runs index calculations in every k-iteration.
     */
     template<typename value_t>
     __global__
     void kernel_naive(Info info,
                         CudaUnstructuredGrid3DInfo<value_t> grid_info,
                         value_t* in,
                         value_t* out,
                         value_t* coeff
                         #ifdef HDIFF_DEBUG
                         , value_t* dbg_lap
                         , value_t* dbg_flx
                         , value_t* dbg_fly
                         #endif
                         ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }
        
        const int n_0_0_0       = CUDA_UNSTR_INDEX(grid_info, i, j, k);
        const int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, 0, -1, 0); /* left */
        const int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_n1_0, 0, -1, 0); /* 2 left */
        const int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, -1, 0, 0); /* top */
        const int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n1_0_0, 0, -1, 0); /* top left */
        const int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n1_0_0, -1, 0, 0); /* 2 top */
        //const int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n2_0_0, 0, -1, 0);
        const int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, 0, +1, 0); /* right */
        const int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_p1_0, 0, +1, 0); /* 2 right */
        const int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, +1, 0, 0); /* bottom */
        const int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p1_0_0, 0, +1, 0); /* bottom right */
        const int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p1_0_0, +1, 0, 0); /* 2 bottom */
        //const int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p2_0_0, 0, +1, 0);     
        const int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n1_0_0, 0, +1, 0); /* top right */
        const int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p1_0_0, 0, -1, 0); /* bottom left */

        const value_t lap_ij = 
            4 * in[n_0_0_0] 
            - in[n_n1_0_0] - in[n_p1_0_0]
            - in[n_0_n1_0] - in[n_0_p1_0];
        const value_t lap_imj = 
            4 * in[n_n1_0_0]
            - in[n_n2_0_0] - in[n_0_0_0]
            - in[n_n1_n1_0] - in[n_n1_p1_0];
        const value_t lap_ipj =
            4 * in[n_p1_0_0]
            - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_p2_0_0]
            - in[n_p1_n1_0] - in[n_p1_p1_0];
        const value_t lap_ijm =
            4 * in[n_0_n1_0]
            - in[n_n1_n1_0] - in[n_p1_n1_0]
            - in[n_0_n2_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)];
        const value_t lap_ijp =
            4 * in[n_0_p1_0]
            - in[n_n1_p1_0] - in[n_p1_p1_0]
            - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_p2_0];

        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[n_p1_0_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : flx_ij;

        value_t flx_imj = lap_ij - lap_imj;
        flx_imj = flx_imj * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_n1_0_0]) > 0 ? 0 : flx_imj;

        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[n_0_p1_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : fly_ij;

        value_t fly_ijm = lap_ij - lap_ijm;
        fly_ijm = fly_ijm * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_n1_0]) > 0 ? 0 : fly_ijm;

        out[CUDA_UNSTR_INDEX(grid_info, i, j, k)] =
            in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]
            - coeff[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        
        // for debugging purposes:
        #ifdef HDIFF_DEBUG
        dbg_lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = lap_ij;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = lap_imj;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = lap_ijm;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] = lap_ipj;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)] = lap_ijp;
        dbg_flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = flx_ij;
        dbg_flx[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = flx_imj;
        dbg_fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = fly_ij;
        dbg_fly[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = fly_ijm;
        #endif
 
    }

    /** Naive implementation of a unstructured grid horizontal diffusion
     * kernel. Runs index calculations in every k-iteration.
     */
     template<typename value_t>
     __global__
     void kernel_naive_kloop(Info info,
                             CudaUnstructuredGrid3DInfo<value_t> grid_info,
                             value_t* in,
                             value_t* out,
                             value_t* coeff
                             #ifdef HDIFF_DEBUG
                             , value_t* dbg_lap
                             , value_t* dbg_flx
                             , value_t* dbg_fly
                             #endif
                             ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }
        
        for(int k = info.halo.z; k < info.max_coord.z; k++) {

            int n_0_0_0       = CUDA_UNSTR_INDEX(grid_info, i, j, k);
            int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, 0, -1, 0);
            int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_n1_0, 0, -1, 0);
            int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, -1, 0, 0);
            int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n1_0_0, 0, -1, 0);
            int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n1_0_0, -1, 0, 0);
            //int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n2_0_0, 0, -1, 0);
            int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, 0, +1, 0);
            int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_p1_0, 0, +1, 0);
            int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_0_0_0, +1, 0, 0);
            int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p1_0_0, 0, +1, 0);
            int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p1_0_0, +1, 0, 0);
            //int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p2_0_0, 0, +1, 0);     
            int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_n1_0_0, 0, +1, 0);
            int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(grid_info, n_p1_0_0, 0, -1, 0);

            value_t lap_ij = 
                4 * in[n_0_0_0] 
                - in[n_n1_0_0] - in[n_p1_0_0]
                - in[n_0_n1_0] - in[n_0_p1_0];
            value_t lap_imj = 
                4 * in[n_n1_0_0]
                - in[n_n2_0_0] - in[n_0_0_0]
                - in[n_n1_n1_0] - in[n_n1_p1_0];
            value_t lap_ipj =
                4 * in[n_p1_0_0]
                - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_p2_0_0]
                - in[n_p1_n1_0] - in[n_p1_p1_0];
            value_t lap_ijm =
                4 * in[n_0_n1_0]
                - in[n_n1_n1_0] - in[n_p1_n1_0]
                - in[n_0_n2_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)];
            value_t lap_ijp =
                4 * in[n_0_p1_0]
                - in[n_n1_p1_0] - in[n_p1_p1_0]
                - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_p2_0];

            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[n_p1_0_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : flx_ij;

            value_t flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_n1_0_0]) > 0 ? 0 : flx_imj;

            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[n_0_p1_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : fly_ij;

            value_t fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_n1_0]) > 0 ? 0 : fly_ijm;

            out[CUDA_UNSTR_INDEX(grid_info, i, j, k)] =
                in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]
                - coeff[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
            
            // for debugging purposes:
            #ifdef HDIFF_DEBUG
            dbg_lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = lap_ij;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = lap_imj;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = lap_ijm;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] = lap_ipj;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)] = lap_ijp;
            dbg_flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = flx_ij;
            dbg_flx[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = flx_imj;
            dbg_fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = fly_ij;
            dbg_fly[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = fly_ijm;
            #endif
        }
 
    }

    /** This kernel makes use of the regularity of the grid in the Z-direction.
     * Instead of naively resolving the neighborship relations at each k-step,
     * The locations of the neighboring cells are calculated at one level and
     * then reused, with the constant (regular) Z-step at each k-iteration.
     */
    template<typename value_t>
    __global__
    void kernel_idxvars(Info info,
                        CudaUnstructuredGrid3DInfo<value_t> grid_info,
                        value_t* in,
                        value_t* out,
                        value_t* coeff
                        #ifdef HDIFF_DEBUG
                        , value_t* dbg_lap
                        , value_t* dbg_flx
                        , value_t* dbg_fly
                        #endif
                        ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }

        /** Store index offsets for the current x and y coordinate, so we do
         * not have to recalculate them in every k-iteration. Instead, with
         * each iteration, the k-stride is simply added once -- thus making
         * use of the regularity of the grid in z-direction. 
         * idx of neighbor X Y Z = n_X_Y_Z with p for positive offset and 
         * n for negative offset. */
        int n_0_0_0       = CUDA_UNSTR_INDEX(grid_info, i, j, 0);
        int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_0_0_0, 0, -1);
        int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_0_n1_0, 0, -1);
        int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_0_0_0, -1, 0);
        int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_n1_0_0, 0, -1);
        int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_n1_0_0, -1, 0);
        //int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_n2_0_0, 0, -1);
        int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_0_0_0, 0, +1);
        int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_0_p1_0, 0, +1);
        int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_0_0_0, +1, 0);
        int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_p1_0_0, 0, +1);
        int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_p1_0_0, +1, 0);
        //int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_p2_0_0, 0, +1);     
        int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_n1_0_0, 0, +1);
        int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, n_p1_0_0, 0, -1);

        for (int k = info.halo.z; k < info.max_coord.z; k++) {

            value_t lap_ij = 
                4 * in[n_0_0_0] 
                - in[n_n1_0_0] - in[n_p1_0_0]
                - in[n_0_n1_0] - in[n_0_p1_0];
            value_t lap_imj = 
                4 * in[n_n1_0_0]
                - in[n_n2_0_0] - in[n_0_0_0]
                - in[n_n1_n1_0] - in[n_n1_p1_0];
            value_t lap_ipj =
                4 * in[n_p1_0_0]
                - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_p2_0_0]
                - in[n_p1_n1_0] - in[n_p1_p1_0];
            value_t lap_ijm =
                4 * in[n_0_n1_0]
                - in[n_n1_n1_0] - in[n_p1_n1_0]
                - in[n_0_n2_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)];
            value_t lap_ijp =
                4 * in[n_0_p1_0]
                - in[n_n1_p1_0] - in[n_p1_p1_0]
                - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_p2_0];

            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[n_p1_0_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : flx_ij;

            value_t flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_n1_0_0]) > 0 ? 0 : flx_imj;

            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[n_0_p1_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : fly_ij;

            value_t fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_n1_0]) > 0 ? 0 : fly_ijm;

            out[CUDA_UNSTR_INDEX(grid_info, i, j, k)] =
                in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]
                - coeff[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
            
            // DEBUG: Output intermediate results as well
            // Disable this for better performance
            #ifdef HDIFF_DEBUG
            dbg_lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = lap_ij;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = lap_imj;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = lap_ijm;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] = lap_ipj;
            dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)] = lap_ijp;
            dbg_flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = flx_ij;
            dbg_flx[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = flx_imj;
            dbg_fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = fly_ij;
            dbg_fly[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = fly_ijm;
            #endif

            // Make use of regularity in Z-direciton: neighbors are exactly the
            // same, just one Z-stride apart.
            n_0_0_0       += grid_info.strides.z;
            n_0_n1_0      += grid_info.strides.z;
            n_0_n2_0      += grid_info.strides.z;
            n_n1_0_0      += grid_info.strides.z;
            n_n1_n1_0     += grid_info.strides.z;
            n_n2_0_0      += grid_info.strides.z;
            //n_n2_n1_0     += grid_info.strides.z;
            n_0_p1_0      += grid_info.strides.z;
            n_0_p2_0      += grid_info.strides.z;
            n_p1_0_0      += grid_info.strides.z;
            n_p1_p1_0     += grid_info.strides.z;
            n_p2_0_0      += grid_info.strides.z;
            //n_p2_p1_0     += grid_info.strides.z;
            n_n1_p1_0     += grid_info.strides.z;
            n_p1_n1_0     += grid_info.strides.z;

        }
    }

    /** A designated kernel invocation (at k=0) loads the neighborship relation
     * for the given x and y coordinates of this kernel. The other kernel
     * invocations at higher levels rely on shared memory to access the
     * neighborship information.
     */
     template<typename value_t>
     __global__
     void kernel_shared_idxvars(Info info,
                                CudaUnstructuredGrid3DInfo<value_t> grid_info,
                                value_t* in,
                                value_t* out,
                                value_t* coeff
                                #ifdef HDIFF_DEBUG
                                , value_t* dbg_lap
                                , value_t* dbg_flx
                                , value_t* dbg_fly
                                #endif
                                ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }
        
        extern __shared__ int smem[]; // stores four neighbors of cell i at smem[i*4]
        const int local_idx = (threadIdx.x + threadIdx.y*blockDim.x) * 12;
        const int global_idx_2d = CUDA_UNSTR_INDEX(grid_info, i, j, 0);

        if(k % blockDim.z == 0) {
            // We are the thread responsible for looking up neighbor info
            /*  0 -1 */ smem[local_idx+0] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, global_idx_2d, 0, -1);
            /*  0 -2 */ smem[local_idx+1] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+0], 0, -1);
            /* -1  0 */ smem[local_idx+2] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, global_idx_2d, -1, 0);
            /* -1 -1 */ smem[local_idx+3] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+2], 0, -1);
            /* -2  0 */ smem[local_idx+4] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+2], -1, 0);
            //n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem_n_n2_0_0, 0, -1);
            /*  0 +1 */ smem[local_idx+5] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, global_idx_2d, 0, +1);
            /*  0 +2 */ smem[local_idx+6] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+5], 0, +1);
            /* +1  0 */ smem[local_idx+7] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, global_idx_2d, +1, 0);
            /* +1 +1 */ smem[local_idx+8] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+7], 0, +1);
            /* +2  0 */ smem[local_idx+9] = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+7], +1, 0);
            //n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem_n_p2_0_0, 0, +1);     
            /* -1 +1 */ smem[local_idx+10]= CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+2], 0, +1);
            /* +1 -1 */ smem[local_idx+11]= CUDA_UNSTR_NEIGHBOR_AT_UNSAFE(grid_info, smem[local_idx+7], 0, -1);
        }
        
        __syncthreads();
        const int k_step = k*grid_info.strides.z;
        const int n_0_0_0       = global_idx_2d + k_step;
        const int n_0_n1_0      = smem[local_idx+0] + k_step;
        const int n_0_n2_0      = smem[local_idx+1] + k_step;
        const int n_n1_0_0      = smem[local_idx+2] + k_step;
        const int n_n1_n1_0     = smem[local_idx+3] + k_step;
        const int n_n2_0_0      = smem[local_idx+4] + k_step;
        //const int n_n2_n1_0     = smem_n_n2_n1_0 + k_step;
        const int n_0_p1_0      = smem[local_idx+5] + k_step;
        const int n_0_p2_0      = smem[local_idx+6] + k_step;
        const int n_p1_0_0      = smem[local_idx+7] + k_step;
        const int n_p1_p1_0     = smem[local_idx+8] + k_step;
        const int n_p2_0_0      = smem[local_idx+9] + k_step;
        //const int n_p2_p1_0     = smem_n_p2_p1_0 + k_step;
        const int n_n1_p1_0     = smem[local_idx+10] + k_step;
        const int n_p1_n1_0     = smem[local_idx+11] + k_step;

        const value_t lap_ij = 
            4 * in[n_0_0_0] 
            - in[n_n1_0_0] - in[n_p1_0_0]
            - in[n_0_n1_0] - in[n_0_p1_0];
        const value_t lap_imj = 
            4 * in[n_n1_0_0]
            - in[n_n2_0_0] - in[n_0_0_0]
            - in[n_n1_n1_0] - in[n_n1_p1_0];
        const value_t lap_ipj =
            4 * in[n_p1_0_0]
            - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_p2_0_0]
            - in[n_p1_n1_0] - in[n_p1_p1_0];
        const value_t lap_ijm =
            4 * in[n_0_n1_0]
            - in[n_n1_n1_0] - in[n_p1_n1_0]
            - in[n_0_n2_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)];
        const value_t lap_ijp =
            4 * in[n_0_p1_0]
            - in[n_n1_p1_0] - in[n_p1_p1_0]
            - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_p2_0];

        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[n_p1_0_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : flx_ij;

        value_t flx_imj = lap_ij - lap_imj;
        flx_imj = flx_imj * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_n1_0_0]) > 0 ? 0 : flx_imj;

        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[n_0_p1_0] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0 ? 0 : fly_ij;

        value_t fly_ijm = lap_ij - lap_ijm;
        fly_ijm = fly_ijm * (in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - in[n_0_n1_0]) > 0 ? 0 : fly_ijm;

        out[CUDA_UNSTR_INDEX(grid_info, i, j, k)] =
            in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]
            - coeff[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        
        // for debugging purposes:
        #ifdef HDIFF_DEBUG
        dbg_lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = lap_ij;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = lap_imj;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = lap_ijm;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] = lap_ipj;
        dbg_lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)] = lap_ijp;
        dbg_flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = flx_ij;
        dbg_flx[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] = flx_imj;
        dbg_fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = fly_ij;
        dbg_fly[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] = fly_ijm;
        #endif
 
    }


};

/** Cuda implementation of different variants of the horizontal diffusion
 * kernel, both for structured and unstructured grid variants.
 *
 * For the available variants, see the HdiffCuda::Variant enum. */
template<typename value_t>
class HdiffCudaUnstrBenchmark : public HdiffBaseBenchmark<value_t> {

    public:

    HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant=HdiffCudaUnstr::UnstrNaive);
    
    HdiffCudaUnstr::Variant variant;

    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    virtual dim3 numblocks();
    virtual dim3 numthreads();

    // Return info struct for kernels
    HdiffCudaUnstr::Info get_info();

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaUnstrBenchmark<value_t>::HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant) :
HdiffBaseBenchmark<value_t>(size) {
    if(variant == HdiffCudaUnstr::UnstrNaive) {
        this->name = "hdiff-unstr-naive";
    } else if(variant == HdiffCudaUnstr::UnstrKloop) {
        this->name = "hdiff-unstr-kloop";
    } else if(variant == HdiffCudaUnstr::UnstrSharedIdxvar) {
        this->name = "hdiff-unstr-shared";
    } else {
        this->name = "hdiff-unstr-idxvar";
    }
    this->error = false;
    this->variant = variant;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::run() {
    auto kernel_fun = &HdiffCudaUnstr::kernel_naive<value_t>;
    int smem = 0;
    if(this->variant == HdiffCudaUnstr::UnstrIdxvars) {
        kernel_fun = &HdiffCudaUnstr::kernel_idxvars<value_t>;
    } else if(this->variant == HdiffCudaUnstr::UnstrKloop) {
        kernel_fun = &HdiffCudaUnstr::kernel_naive_kloop<value_t>;
    } else if(this->variant == HdiffCudaUnstr::UnstrSharedIdxvar) {
        kernel_fun = &HdiffCudaUnstr::kernel_shared_idxvars<value_t>;
        dim3 numthreads = this->numthreads();
        smem = numthreads.x*numthreads.y*12*sizeof(int);
    }
    (*kernel_fun)<<<this->numblocks(), this->numthreads(), smem>>>(
        this->get_info(),
        (dynamic_cast<CudaUnstructuredGrid3D<value_t>*>(this->input))->get_gridinfo(),
        this->input->data,
        this->output->data,
        this->coeff->data
        #ifdef HDIFF_DEBUG
        , this->lap->data
        , this->flx->data
        , this->fly->data
        #endif
    );
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->Benchmark::numblocks();
    // For the vriants that use a k-loop inside the kernel, we only need one block in the k-direction
    if(this->variant == HdiffCudaUnstr::UnstrIdxvars ||
       this->variant == HdiffCudaUnstr::UnstrKloop ) {
        numblocks = dim3(numblocks.x, numblocks.y, 1);
    }
    return numblocks;
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->Benchmark::numthreads();
    // Variants with a k-loop: only one thread in the k-direction
    if(this->variant == HdiffCudaUnstr::UnstrIdxvars ||
        this->variant == HdiffCudaUnstr::UnstrKloop ) {
        numthreads = dim3(numthreads.x, numthreads.y, 1);
    }
    return numthreads;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::setup() {
    this->input = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->output = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->coeff = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->lap = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->flx = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->fly = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    int *neighbor_data = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input)->neighbor_data;
    this->output = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->coeff = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->lap = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->flx = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->fly = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->HdiffBaseBenchmark<value_t>::setup();
    int s1 = cudaMemPrefetchAsync(this->input->data, this->input->size, 0);
    int s2 = cudaMemPrefetchAsync(this->output->data, this->output->size, 0);
    int s3 = cudaMemPrefetchAsync(this->coeff->data, this->coeff->size, 0);
    #ifdef HDIFF_DEBUG
    int s4 = cudaMemPrefetchAsync(this->lap->data, this->lap->size, 0);
    int s5 = cudaMemPrefetchAsync(this->flx->data, this->flx->size, 0);
    int s6 = cudaMemPrefetchAsync(this->fly->data, this->fly->size, 0);
    #endif
    if( s1 != cudaSuccess || s2 != cudaSuccess || s3 != cudaSuccess
        #ifdef HDIFF_DEBUG
            || s4 != cudaSuccess || s5 != cudaSuccess || s6 != cudaSuccess
        #endif
    ) {
        throw std::runtime_error("unable to prefetch memory");
    }
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::teardown() {
    this->input->deallocate();
    this->output->deallocate();
    this->coeff->deallocate();
    this->lap->deallocate();
    this->flx->deallocate();
    this->fly->deallocate();
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffBaseBenchmark<value_t>::post();
}

template<typename value_t>
HdiffCudaUnstr::Info HdiffCudaUnstrBenchmark<value_t>::get_info() {
    return { .halo = this->halo,
             .max_coord = this->input->dimensions - this->halo};
}

#endif