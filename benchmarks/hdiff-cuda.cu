#ifndef HDIFF_CUDA_H
#define HDIFF_CUDA_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaRegular {

    enum Variant { direct, kloop, idxvar, shared, shared_kloop, coop, jloop, iloop };

    /** Naive variant: Every thread computes all of its data dependencies by itself. */
    template<typename value_t>
    __global__
    void kernel_direct(HdiffBase::Info info,
                       CudaRegularGrid3DInfo<value_t> grids_info,
                       value_t *in,
                       value_t *out,
                       value_t *coeff
                       #ifdef HDIFF_DEBUG
                       , value_t *dbg_lap
                       , value_t *dbg_flx
                       , value_t *dbg_fly
                       #endif
                       ) {


        // the loops below replace this condition if gridstride is activated
        #ifdef HDIFF_NO_GRIDSTRIDE
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }
        #endif

        /** Grid stride loop: This triple loop looks scary, but in case the
         * thread-grid is large enough, it is executed only once and it is used
         * to simply check the condition that the coordinates are in bound. In
         * case that the grid is smaller than the data that needs to be handled,
         * each thread has to process multiple data points, which is done in the
         * loop. */
        #ifndef HDIFF_NO_GRIDSTRIDE
        for(int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x; 
            i < info.inner_size.x + info.halo.x; 
            i += info.gridsize.x) {
            for(int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
                j < info.inner_size.y + info.halo.y;
                j += info.gridsize.y) {
                for(int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
                    k < info.inner_size.z + info.halo.z;
                    k += info.gridsize.z) {
        #endif

                    const value_t lap_ij = 
                        4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] 
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)];
                    const value_t lap_imj = 
                        4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -2, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)];
                    const value_t lap_ipj =
                        4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +2, 0, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)];
                    const value_t lap_ijm =
                        4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -2, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)];
                    const value_t lap_ijp =
                        4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +2, 0)];
            
                    value_t flx_ij = lap_ipj - lap_ij;
                    flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : flx_ij;
            
                    value_t flx_imj = lap_ij - lap_imj;
                    flx_imj = flx_imj * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
            
                    value_t fly_ij = lap_ijp - lap_ij;
                    fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : fly_ij;
            
                    value_t fly_ijm = lap_ij - lap_ijm;
                    fly_ijm = fly_ijm * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;
            
                    out[CUDA_REGULAR_INDEX(grids_info, i, j, k)] =
                        in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]
                        - coeff[CUDA_REGULAR_INDEX(grids_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
            
                    // for debugging purposes:
                    #ifdef HDIFF_DEBUG
                    dbg_lap[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = lap_ij;
                    dbg_lap[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)] = lap_imj;
                    dbg_lap[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] = lap_ijm;
                    dbg_lap[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] = lap_ipj;
                    dbg_lap[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] = lap_ijp;
                    dbg_flx[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = flx_ij;
                    dbg_flx[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)] = flx_imj;
                    dbg_fly[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = fly_ij;
                    dbg_fly[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] = fly_ijm;
                    #endif

        #ifndef HDIFF_NO_GRIDSTRIDE
                }
            }
        }
        #endif
    }

    /** Cooperating kernels.
     * Shared intermediate results: In this variant, each kernel first
     * computes its values for lap, flx and fly (at its i, j, k) position.
     * Threads are synchronized at appropriate points so they can access each
     * others results as needed.
     * 
     * Note that at the block boundaries, some threads still need to compute
     * their dependencies, as __synchthreads() only synchronizes within the
     * block, not entire grid.
     * 
     * This can be thought of as a kind of sequential version, but only on the
     * thread-level. */
    template<typename value_t>
    __global__
    void kernel_coop(HdiffBase::Info info,
                    CudaRegularGrid3DInfo<value_t> grids_info,
                    value_t *in,
                    value_t *out,
                    value_t *coeff,
                    value_t *lap,
                    value_t *flx,
                    value_t *fly) {
        
        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        
        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }

        // Calculate own laplace
        value_t lap_ij = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] 
            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)];
        lap[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = lap_ij;

        // Sync threads to enable access to their laplace calculations
        __syncthreads();

        value_t lap_ipj;
        if(threadIdx.x == blockDim.x-1) {
            // rightmost in block, need to compute right dependency ourselves
            lap_ipj = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +2, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)];
        } else {
            lap_ipj = lap[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)];
        }

        value_t lap_ijp;
        if(threadIdx.y == blockDim.y-1) {
            lap_ijp = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +2, 0)];
        } else {
            lap_ijp = lap[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)];
        }

        // Own flx/fly calculation
        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : flx_ij;
        flx[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = flx_ij;

        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : fly_ij;
        fly[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = fly_ij;

        // Make flx/fly available to other threads by synchronizing
        __syncthreads();

        value_t flx_imj;
        if(threadIdx.x == 0) {
            // leftmost in block, need to compute left dependency ourselves
            value_t lap_imj = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -2, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)];
            flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
        } else {
            flx_imj = flx[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)];
        }

        value_t fly_ijm;
        if(threadIdx.y == 0) {
            // need to also calculate lap for j - 1 as we are at boundary
            value_t lap_ijm = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -2, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)];
            fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;
        } else {
            fly_ijm = fly[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)];
        }

        out[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]
                - coeff[CUDA_REGULAR_INDEX(grids_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);

    }

    /** Just like the coop kernel, but uses shared memory fo threads instead
     * of managed memory for the lap, flx, fly results. */
    template<typename value_t>
    __global__
    void kernel_shared(HdiffBase::Info info,
                    CudaRegularGrid3DInfo<value_t> grids_info,
                    CudaRegularGrid3DInfo<value_t> local_grids_info,
                    value_t *in,
                    value_t *out,
                    value_t *coeff) {
        
        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;

        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }

        // Shared memory
        extern __shared__ char smem[];
        const int block_size = local_grids_info.strides.z*blockDim.z;

        // Local grids holding results for laplace, flx and fly calcualted by other threads
        value_t *local_lap = (value_t *)smem;
        value_t *local_flx = &local_lap[block_size];
        value_t *local_fly = &local_flx[block_size];
        
        // Calculate own laplace
        value_t lap_ij = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] 
            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)];
        local_lap[CUDA_REGULAR_INDEX(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z)] = lap_ij;

        // Sync threads to enable access to their laplace calculations
        __syncthreads();

        value_t lap_ipj;
        if(threadIdx.x == blockDim.x-1 || i == info.max_coord.x-1) {
            // rightmost in block, need to compute right dependency ourselves
            lap_ipj = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +2, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)];
        } else {
            lap_ipj = local_lap[CUDA_REGULAR_NEIGHBOR(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z, +1, 0, 0)];
        }

        value_t lap_ijp;
        if(threadIdx.y == blockDim.y-1 || j == info.max_coord.y-1) {
            lap_ijp = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +2, 0)];
        } else {
            lap_ijp = local_lap[CUDA_REGULAR_NEIGHBOR(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z, 0, +1, 0)];
        }

        // Own flx/fly calculation
        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : flx_ij;
        local_flx[CUDA_REGULAR_INDEX(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z)] = flx_ij;

        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : fly_ij;
        local_fly[CUDA_REGULAR_INDEX(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z)] = fly_ij;

        // Make flx/fly available to other threads by synchronizing
        __syncthreads();

        value_t flx_imj;
        if(threadIdx.x == 0) {
            // leftmost in block, need to compute left dependency ourselves
            value_t lap_imj = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -2, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)];
            flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
        } else {
            flx_imj = local_flx[CUDA_REGULAR_NEIGHBOR(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z, -1, 0, 0)];
        }

        value_t fly_ijm;
        if(threadIdx.y == 0) {
            // need to also calculate lap for j - 1 as we are at boundary
            value_t lap_ijm = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -2, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)];
            fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;
        } else {
            fly_ijm = local_fly[CUDA_REGULAR_NEIGHBOR(local_grids_info, threadIdx.x, threadIdx.y, threadIdx.z, 0, -1, 0)];
        }

        out[CUDA_REGULAR_INDEX(grids_info, i, j, k)] = in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]
                - coeff[CUDA_REGULAR_INDEX(grids_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
    }

    /** Like kernel_shared, but uses a k loop and in each loop iteration
     * threads are synchronized. This reduces the amount of memory required,
     * as only the lap/flx/fly for one k-level is stored in shared memory. */
    template<typename value_t>
    __global__
    void kernel_shared_kloop(const HdiffBase::Info info,
                    const CudaRegularGrid3DInfo<value_t> input_grids_info,
                    const int blocksize,
                    const value_t *in,
                    value_t *out,
                    const value_t *coeff) {
        
        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }

        // Shared memory
        extern __shared__ char smem[];

        // Local grids holding results for laplace, flx and fly calcualted by other threads
        value_t *local_lap = (value_t*)smem;
        value_t *local_flx = &local_lap[blocksize];
        value_t *local_fly = &local_flx[blocksize];
        
        // K-loop
        for(int k = info.halo.z; k < info.max_coord.z; k++) {

            // Calculate own laplace
            const value_t lap_ij = 4 * in[CUDA_REGULAR_NEIGHBOR(input_grids_info, i, j, k, 0, 0, 0)] 
                - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, +1, 0)];
            local_lap[CUDA_REGULAR_INDEX_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0)] = lap_ij;

            // Sync threads to enable access to their laplace calculations
            __syncthreads();

            value_t lap_ipj;
            if(threadIdx.x == blockDim.x-1 || i == info.max_coord.x-1) {
                // rightmost in block, need to compute right dependency ourselves
                lap_ipj = 4 * in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, 0, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +2, 0, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, +1, 0)];
            } else {
                lap_ipj = local_lap[CUDA_REGULAR_NEIGHBOR_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0, +1, 0, 0)];
            }

            value_t lap_ijp;
            if(threadIdx.y == blockDim.y-1 || j == info.max_coord.y-1) {
                lap_ijp = 4 * in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, +1, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, +1, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, +1, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, +2, 0)];
            } else {
                lap_ijp = local_lap[CUDA_REGULAR_NEIGHBOR_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0, 0, +1, 0)];
            }

            // Own flx/fly calculation
            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)]) > 0 ? 0 : flx_ij;
            local_flx[CUDA_REGULAR_INDEX_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0)] = flx_ij;

            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)]) > 0 ? 0 : fly_ij;
            local_fly[CUDA_REGULAR_INDEX_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0)] = fly_ij;

            // Make flx/fly available to other threads by synchronizing
            __syncthreads();

            value_t flx_imj;
            if(threadIdx.x == 0) {
                // leftmost in block, need to compute left dependency ourselves
                value_t lap_imj = 4 * in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, 0, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -2, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, 0, 0)]
                    - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, +1, 0)];
                flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (in[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
            } else {
                flx_imj = local_flx[CUDA_REGULAR_NEIGHBOR_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0, -1, 0, 0)];
            }

            value_t fly_ijm;
            if(threadIdx.y == 0) {
                // need to also calculate lap for j - 1 as we are at boundary
                value_t lap_ijm = 4 * in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, -1, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, +1, -1, 0)]
                        - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, -2, 0)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, 0, 0)];
                fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (in[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;
            } else {
                fly_ijm = local_fly[CUDA_REGULAR_NEIGHBOR_(blockDim.x, 0, threadIdx.x, threadIdx.y, 0, 0, -1, 0)];
            }

            out[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)] = in[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)]
                    - coeff[CUDA_REGULAR_INDEX_(input_grids_info.strides.y, input_grids_info.strides.z, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }
    }

    /** This is the same as shared_kloop, but the code has been simplified to
     * not make use uf any structs etc and kept as simple as possible
     * (i.e. un-readable), in order to analyze if this has any impact on the
     * register usage, i.e. if the compiler is able to optimize this. */
     template<typename value_t>
     __global__
     void kernel_shared_kloop_simplecode(const int halo_x, const int halo_y, const int halo_z, const int max_x, const int max_y, const int max_z,
            const int ystride, const int zstride,
            const int blocksize,
            const value_t *in_data, value_t *out_data, const value_t *coeff_data) {

        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + halo_x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + halo_y;
        if(i >= max_x || j >= max_y) {
            return;
        }

        // Shared memory
        extern __shared__ value_t smem[];
        
        value_t *local_lap = smem;
        value_t *local_flx = &smem[blocksize];
        value_t *local_fly = &smem[2*blocksize];

        // K-loop
        for(int k = halo_z; k < max_z; k++) {
            const value_t lap_ij = 4 * in_data[i+j*ystride+k*zstride] 
                - in_data[i+j*ystride+k*zstride + (-1)] - in_data[i+j*ystride+k*zstride + (+1)]
                - in_data[i+j*ystride+k*zstride + (0) + (-1)*ystride] - in_data[i+j*ystride+k*zstride + (0) + (+1)*ystride];
            local_lap[threadIdx.x+threadIdx.y*blockDim.x] = lap_ij;
            __syncthreads();
            value_t lap_ipj;
            if(threadIdx.x == blockDim.x-1 || i == max_x-1) {
                lap_ipj = 4 * in_data[i+j*ystride+k*zstride + (+1)]
                    - in_data[i+j*ystride+k*zstride] - in_data[i+j*ystride+k*zstride + (+2)]
                    - in_data[i+j*ystride+k*zstride + (+1) + (-1)*ystride] - in_data[i+j*ystride+k*zstride + (+1) + (+1)*ystride];
            } else {
                lap_ipj = local_lap[threadIdx.x+threadIdx.y*blockDim.x + (+1)];
            }
            value_t lap_ijp;
            if(threadIdx.y == blockDim.y-1 || j == max_y-1) {
                lap_ijp = 4 * in_data[i+j*ystride+k*zstride + (0) + (+1)*ystride]
                    - in_data[i+j*ystride+k*zstride + (-1) + (+1)*ystride] - in_data[i+j*ystride+k*zstride + (+1) + (+1)*ystride]
                    - in_data[i+j*ystride+k*zstride] - in_data[i+j*ystride+k*zstride + (0) + (+2)*ystride];
            } else {
                lap_ijp = local_lap[threadIdx.x+threadIdx.y*blockDim.x + (0) + (+1)*blockDim.x];
            }
            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in_data[i+j*ystride+k*zstride + (+1)] - in_data[i+j*ystride+k*zstride]) > 0 ? 0 : flx_ij;
            local_flx[threadIdx.x+threadIdx.y*blockDim.x] = flx_ij;
            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in_data[i+j*ystride+k*zstride + (0) + (+1)*ystride] - in_data[i+j*ystride+k*zstride]) > 0 ? 0 : fly_ij;
            local_fly[threadIdx.x+threadIdx.y*blockDim.x] = fly_ij;
            __syncthreads();
            value_t flx_imj;
            if(threadIdx.x == 0) {
                value_t lap_imj = 4 * in_data[i+j*ystride+k*zstride + (-1)]
                    - in_data[i+j*ystride+k*zstride + (-2)] - in_data[i+j*ystride+k*zstride]
                    - in_data[i+j*ystride+k*zstride + (-1) + (-1)*ystride] - in_data[i+j*ystride+k*zstride + (-1) + (+1)*ystride];
                flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (in_data[i+j*ystride+k*zstride] - in_data[i+j*ystride+k*zstride + (-1)]) > 0 ? 0 : flx_imj;
            } else {
                flx_imj = local_flx[threadIdx.x+threadIdx.y*blockDim.x-1];
            }
            value_t fly_ijm;
            if(threadIdx.y == 0) {
                value_t lap_ijm = 4 * in_data[i+j*ystride+k*zstride + (0) + (-1)*ystride]
                        - in_data[i+j*ystride+k*zstride + (-1) + (-1)*ystride] - in_data[i+j*ystride+k*zstride + (+1) + (-1)*ystride]
                        - in_data[i+j*ystride+k*zstride + (0) + (-2)*ystride] - in_data[i+j*ystride+k*zstride];
                fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (in_data[i+j*ystride+k*zstride] - in_data[i+j*ystride+k*zstride + (0) + (-1)*ystride]) > 0 ? 0 : fly_ijm;
            } else {
                fly_ijm = local_fly[threadIdx.x+threadIdx.y*blockDim.x + (0) -1*blockDim.x];
            }
            out_data[i+j*ystride+k*zstride] = in_data[i+j*ystride+k*zstride] - coeff_data[i+j*ystride+k*zstride] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }
     }
    
    template<typename value_t>
    __global__
    void kernel_kloop(HdiffBase::Info info,
                      CudaRegularGrid3DInfo<value_t> grids_info,
                      value_t *in,
                      value_t *out,
                      value_t *coeff) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }
        
        for(int k = info.halo.z; k < info.max_coord.z; k++) {
            value_t lap_ij = 
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] 
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)];
            value_t lap_imj = 
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -2, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)];
            value_t lap_ipj =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +2, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)];
            value_t lap_ijm =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -2, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)];
            value_t lap_ijp =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +2, 0)];

            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : flx_ij;
            value_t flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : fly_ij;
            value_t fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;

            out[CUDA_REGULAR_INDEX(grids_info, i, j, k)] =
                in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]
                - coeff[CUDA_REGULAR_INDEX(grids_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }
    }

    template<typename value_t>
    __global__
    void kernel_idxvar(HdiffBase::Info info,
                       CudaRegularGrid3DInfo<value_t> grids_info,
                       value_t *in,
                       value_t *out,
                       value_t *coeff) {

        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }

        int n_0_0_0       = CUDA_REGULAR_INDEX(grids_info, i, j, 0);
        int n_0_n1_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,   0, -1, 0);
        int n_0_n2_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  0, -2, 0);
        int n_n1_0_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  -1, 0, 0);
        int n_n1_n1_0     = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  -1, -1, 0);
        int n_n2_0_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0, -2, 0, 0);
        //int n_n2_n1_0     = CUDA_REGULAR_NEIGHBOR(in, i, j, 0,  -2, -1, 0);
        int n_0_p1_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,   0, +1, 0);
        int n_0_p2_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  0, +2, 0);
        int n_p1_0_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  +1, 0, 0);
        int n_p1_p1_0     = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  +1, +1, 0);
        int n_p2_0_0      = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0, +2, 0, 0);
        //int n_p2_p1_0     = CUDA_REGULAR_NEIGHBOR(in, i, j, 0,  0, +1, 0);     
        int n_n1_p1_0     = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  -1, +1, 0);
        int n_p1_n1_0     = CUDA_REGULAR_NEIGHBOR(grids_info, i, j, 0,  +1, -1, 0);

        for(int k = info.halo.z; k < info.max_coord.z; k++) {

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
                - in[n_0_0_0] - in[n_p2_0_0]
                - in[n_p1_n1_0] - in[n_p1_p1_0];
            value_t lap_ijm =
                4 * in[n_0_n1_0]
                - in[n_n1_n1_0] - in[n_p1_n1_0]
                - in[n_0_n2_0] - in[n_0_0_0];
            value_t lap_ijp =
                4 * in[n_0_p1_0]
                - in[n_n1_p1_0] - in[n_p1_p1_0]
                - in[n_0_0_0] - in[n_0_p2_0];
    
            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[n_p1_0_0] - in[n_0_0_0]) > 0 ? 0 : flx_ij;
    
            value_t flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[n_0_0_0] - in[n_n1_0_0]) > 0 ? 0 : flx_imj;
    
            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[n_0_p1_0] - in[n_0_0_0]) > 0 ? 0 : fly_ij;
    
            value_t fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[n_0_0_0] - in[n_0_n1_0]) > 0 ? 0 : fly_ijm;
    
            out[n_0_0_0] =
                in[n_0_0_0]
                - coeff[n_0_0_0] * (flx_ij - flx_imj + fly_ij - fly_ijm);

            n_0_0_0       += grids_info.strides.z;
            n_0_n1_0      += grids_info.strides.z;
            n_0_n2_0      += grids_info.strides.z;
            n_n1_0_0      += grids_info.strides.z;
            n_n1_n1_0     += grids_info.strides.z;
            n_n2_0_0      += grids_info.strides.z;
            //n_n2_n1_0     += in.strides.z;
            n_0_p1_0      += grids_info.strides.z;
            n_0_p2_0      += grids_info.strides.z;
            n_p1_0_0      += grids_info.strides.z;
            n_p1_p1_0     += grids_info.strides.z;
            n_p2_0_0      += grids_info.strides.z;
            //n_p2_p1_0     += in.strides.z;
            n_n1_p1_0     += grids_info.strides.z;
            n_p1_n1_0     += grids_info.strides.z;

        }
    }

    template<typename value_t>
    __global__
    void kernel_jloop(HdiffBase::Info info,
                      CudaRegularGrid3DInfo<value_t> grids_info,
                      int j_per_thread,
                      value_t *in,
                      value_t *out,
                      value_t *coeff) {
        
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        int j_start = threadIdx.y*j_per_thread + blockIdx.y*blockDim.y*j_per_thread + info.halo.y;
        if(i >= info.max_coord.x || j_start >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }

        int j_stop = j_start + j_per_thread;
        if(j_stop > info.max_coord.y) {
            j_stop = info.max_coord.y;
        }
        
        // first calculation outside of loop will be shifted into lap_ijm / fly_ijm on first iteration
        value_t lap_ij = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, -1, 0)]
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, +1, -1, 0)]
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, -2, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, 0, 0)];
        
        value_t lap_ijp = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, 0, 0)] 
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, -1, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, +1, 0, 0)]
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, +1, 0)];
        
        value_t fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (in[CUDA_REGULAR_INDEX(grids_info, i, j_start, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j_start, k, 0, -1, 0)]) > 0 ? 0 : fly_ij;


        // j-loop, shifts results from previous round for reuse
        for(int j = j_start; j < j_stop; j++) {

            // shift results from previous iteration
            //value_t lap_ijm = lap_ij;
            lap_ij = lap_ijp;
            value_t fly_ijm = fly_ij;

            // x direction dependencies are recalculated for every cell
            value_t lap_imj = 
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -2, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)];
            value_t lap_ipj =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +2, 0, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)];

            // will be reused as lap_ij in next iteration
            lap_ijp =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)]
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +2, 0)];

            // x direction dependencies are recalculated for every cell
            value_t flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : flx_ij;
            value_t flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, 0, 0)]) > 0 ? 0 : flx_imj;
            
            // will be reused as fly_ijm in next iteration
            fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : fly_ij;

            out[CUDA_REGULAR_INDEX(grids_info, i, j, k)] =
                in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]
                - coeff[CUDA_REGULAR_INDEX(grids_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }

    }

    template<typename value_t>
    __global__
    void kernel_iloop(HdiffBase::Info info,
                      CudaRegularGrid3DInfo<value_t> grids_info,
                      int i_per_thread,
                      value_t *in,
                      value_t *out,
                      value_t *coeff) {
        
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        int i_start = threadIdx.x*i_per_thread + blockIdx.x*blockDim.x*i_per_thread + info.halo.x;
        if(j >= info.max_coord.y || i_start >= info.max_coord.x || k >= info.max_coord.z) {
            return;
        }

        int i_stop = i_start + i_per_thread;
        if(i_stop > info.max_coord.x) {
            i_stop = info.max_coord.x;
        }
        
        // first calculation outside of loop will be shifted into lap_imj / flx_imj on first iteration
        // therefore lap_ij => lap_imj und lap_ipj => lap_ij
        value_t lap_ij = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, -1, 0, 0)] /* center */
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, -2, 0, 0)] /* left */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, 0, 0, 0)] /* right */
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, -1, -1, 0)] /* top */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, -1, +1, 0)] /* bottom */;
        
        value_t lap_ipj = 4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, 0, 0, 0)]  /* center */
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, -1, 0, 0)] /* left */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, +1, 0, 0)] /* right */ 
                            - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, 0, -1, 0)] /* top */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, 0, +1, 0)]; /* bottom */
        
        value_t flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (in[CUDA_REGULAR_INDEX(grids_info, i_start, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i_start, j, k, -1, 0, 0)]) > 0 ? 0 : flx_ij;

        // i-loop, shifts results from previous round for reuse
        for(int i = i_start; i < i_stop; i++) {

            // shift results from previous iteration
            //value_t lap_imj = lap_ij;
            lap_ij = lap_ipj;
            value_t flx_imj = flx_ij;

            // y direction dependencies are recalculated for every cell
            value_t lap_ijm = 
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)] /* center */
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, -1, 0)] /* left */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] /* right */
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -2, 0)] /* top */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] /* bottom */;
            value_t lap_ijp =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] /* center */
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, -1, +1, 0)] /* left */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)] /* right */
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] /* top */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +2, 0)] /* bottom */;

            // will be reused as lap_ij in next iteration
            lap_ipj =
                4 * in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] /* center */
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, 0, 0)] /* left */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +2, 0, 0)] /* right */
                - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, -1, 0)] /* top */ - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, +1, 0)] /* bottom */;

            // y direction dependencies are recalculated for every cell
            value_t fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, +1, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : fly_ij;
            value_t fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in[CUDA_REGULAR_INDEX(grids_info, i, j, k)] - in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, 0, -1, 0)]) > 0 ? 0 : fly_ijm;

            // will be reused as flx_ijm in next iteration
            flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in[CUDA_REGULAR_NEIGHBOR(grids_info, i, j, k, +1, 0, 0)] - in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]) > 0 ? 0 : flx_ij;

            out[CUDA_REGULAR_INDEX(grids_info, i, j, k)] =
                in[CUDA_REGULAR_INDEX(grids_info, i, j, k)]
                - coeff[CUDA_REGULAR_INDEX(grids_info, i, j, k)] * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }

    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
template<typename value_t>
class HdiffCudaBenchmark : public HdiffBaseBenchmark<value_t> {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaBenchmark(coord3 size, HdiffCudaRegular::Variant variant = HdiffCudaRegular::direct);

    HdiffCudaRegular::Variant variant;

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();

    dim3 numthreads();
    dim3 numblocks();
    
    // parameter for the jloop kernel only
    virtual void parse_args();
    int jloop_j_per_thread;
    int iloop_i_per_thread;

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaBenchmark<value_t>::HdiffCudaBenchmark(coord3 size, HdiffCudaRegular::Variant variant) :
HdiffBaseBenchmark<value_t>(size) {
    this->variant = variant;
    if(variant == HdiffCudaRegular::direct) {
        this->name = "hdiff-regular";
    } else if(variant == HdiffCudaRegular::kloop) {
        this->name = "hdiff-regular-kloop";
    } else if(variant == HdiffCudaRegular::shared) {
        this->name = "hdiff-regular-shared";
    } else if(variant == HdiffCudaRegular::shared_kloop) {
        this->name = "hdiff-regular-shared-kloop";
    } else if(variant == HdiffCudaRegular::coop) {
        this->name = "hdiff-regular-coop";
    } else if(variant == HdiffCudaRegular::jloop) {
        this->name = "hdiff-regular-jloop";
    } else if(variant == HdiffCudaRegular::iloop) {
        this->name = "hdiff-regular-iloop";
    } else {
        this->name = "hdiff-regular-idxvar";
    }
}

template<typename value_t>
void HdiffCudaBenchmark<value_t>::run() {
    if(this->variant == HdiffCudaRegular::direct) {
        HdiffCudaRegular::kernel_direct<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            this->input->data,
            this->output->data,
            this->coeff->data
            #ifdef HDIFF_DEBUG
            , this->lap->data
            , this->flx->data
            , this->fly->data
            #endif
        );
    } else if(this->variant == HdiffCudaRegular::kloop) {
        HdiffCudaRegular::kernel_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            this->input->data,
            this->output->data,
            this->coeff->data
        );
    } else if(this->variant == HdiffCudaRegular::coop) {
        HdiffCudaRegular::kernel_coop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            this->input->data,
            this->output->data,
            this->coeff->data,
            this->lap->data,
            this->flx->data,
            this->fly->data
        );
    } else if(this->variant == HdiffCudaRegular::shared) {
        dim3 numthreads = this->numthreads();
        int smem_size = 3*numthreads.x*numthreads.y*numthreads.z*sizeof(value_t);
        HdiffCudaRegular::kernel_shared<value_t><<<this->numblocks(), numthreads, smem_size>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            CudaRegularGrid3DInfo<value_t>{.strides = {.y = (int)numthreads.x, .z = (int)numthreads.x*(int)numthreads.y }}, 
            this->input->data,
            this->output->data,
            this->coeff->data
        );
    } else if(this->variant == HdiffCudaRegular::shared_kloop) {
        dim3 numthreads = this->numthreads();
        dim3 numblocks = this->numblocks();
        int smem_size = 3*numthreads.x*numthreads.y*sizeof(value_t);
        HdiffCudaRegular::kernel_shared_kloop<value_t><<<numblocks, numthreads, smem_size>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            (int)numthreads.x*(int)numthreads.y,
            //CudaRegularGrid3DInfo<value_t>{.strides = { .y = (int)numthreads.x, .z = (int)numthreads.x*(int)numthreads.y } }, 
            this->input->data,
            this->output->data,
            this->coeff->data
        );
        /*HdiffBase::Info info = this->get_info();
        HdiffCudaRegular::kernel_shared_kloop_simplecode<<<this->numblocks(), numthreads, smem_size>>>(
            info.halo.x, info.halo.y, info.halo.z,
            info.max_coord.x, info.max_coord.y, info.max_coord.z,
            this->input->dimensions.x, this->input->dimensions.x*this->input->dimensions.y,
            numthreads.x*numthreads.y,
            this->input->data,
            this->output->data,
            this->coeff->data
        );*/
    } else if(this->variant == HdiffCudaRegular::jloop) {
        HdiffCudaRegular::kernel_jloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            this->jloop_j_per_thread,
            this->input->data,
            this->output->data,
            this->coeff->data
        );
    } else if(this->variant == HdiffCudaRegular::iloop) {
        HdiffCudaRegular::kernel_iloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            this->iloop_i_per_thread,
            this->input->data,
            this->output->data,
            this->coeff->data
        );
    } else {
        HdiffCudaRegular::kernel_idxvar<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_gridinfo(),
            this->input->data,
            this->output->data,
            this->coeff->data
        );
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

template<typename value_t>
void HdiffCudaBenchmark<value_t>::setup() {
    this->input = new CudaRegularGrid3D<value_t>(this->size);
    this->output = new CudaRegularGrid3D<value_t>(this->size);
    this->coeff = new CudaRegularGrid3D<value_t>(this->size);
    this->lap = new CudaRegularGrid3D<value_t>(this->size);
    this->flx = new CudaRegularGrid3D<value_t>(this->size);
    this->fly = new CudaRegularGrid3D<value_t>(this->size);
    this->HdiffBaseBenchmark<value_t>::setup();
    int s1 = cudaMemPrefetchAsync(this->input->data, this->input->size, 0);
    int s2 = cudaMemPrefetchAsync(this->output->data, this->output->size, 0);
    int s3 = cudaMemPrefetchAsync(this->coeff->data, this->coeff->size, 0);
    int s4 = cudaMemPrefetchAsync(this->lap->data, this->lap->size, 0);
    int s5 = cudaMemPrefetchAsync(this->flx->data, this->flx->size, 0);
    int s6 = cudaMemPrefetchAsync(this->fly->data, this->fly->size, 0);
    if( s1 != cudaSuccess || s2 != cudaSuccess || s3 != cudaSuccess ||
        s4 != cudaSuccess || s5 != cudaSuccess || s6 != cudaSuccess) {
        throw std::runtime_error("unable to prefetch memory");
    }
}

template<typename value_t>
void HdiffCudaBenchmark<value_t>::teardown() {
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
void HdiffCudaBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffBaseBenchmark<value_t>::post();
}

template<typename value_t>
dim3 HdiffCudaBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->HdiffBaseBenchmark<value_t>::numthreads();
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar ||
        this->variant == HdiffCudaRegular::shared_kloop) {
        numthreads.z = 1;
    }
    if(this->variant == HdiffCudaRegular::jloop) {
        numthreads.y = 1;
    }
    if(this->variant == HdiffCudaRegular::iloop) {
        numthreads.x = 1;
    }
    return numthreads;
}

template<typename value_t>
dim3 HdiffCudaBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->HdiffBaseBenchmark<value_t>::numblocks();
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar ||
        this->variant == HdiffCudaRegular::shared_kloop) {
        numblocks.z = 1;
    }
    if(this->variant == HdiffCudaRegular::jloop) {
        numblocks.y = (this->size.y + this->jloop_j_per_thread - 1) / this->jloop_j_per_thread;
    }
    if(this->variant == HdiffCudaRegular::iloop) {
        numblocks.x = (this->size.x + this->iloop_i_per_thread - 1) / this->iloop_i_per_thread;
    }
    return numblocks;
}

template<typename value_t>
void HdiffCudaBenchmark<value_t>::parse_args() {
    if(this->argc > 0) {
        // only variant of this that takes an argument is the jloop variant
        if(this->variant == HdiffCudaRegular::jloop) {
            sscanf(this->argv[0], "%d", &this->jloop_j_per_thread);
        }
        if(this->variant == HdiffCudaRegular::iloop) {
            sscanf(this->argv[0], "%d", &this->iloop_i_per_thread);
        }
    } else {
        if(this->variant == HdiffCudaRegular::jloop) {
            this->jloop_j_per_thread = 16; // default value of 16
        }
        if(this->variant == HdiffCudaRegular::iloop) {
            this->iloop_i_per_thread = 16;
        }
    }
}

#endif