#ifndef HDIFF_CUDA_H
#define HDIFF_CUDA_H
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaRegular {

    enum Variant { direct, kloop, idxvar, shared, shared_kloop, coop };

    /** Naive variant: Every thread computes all of its data dependencies by itself. */
    __global__
    void kernel_direct(HdiffBase::Info info,
                       CudaRegularGrid3DInfo<double> in,
                       CudaRegularGrid3DInfo<double> out,
                       CudaRegularGrid3DInfo<double> coeff
                       #ifdef HDIFF_DEBUG
                       , CudaRegularGrid3DInfo<double> dbg_lap
                       , CudaRegularGrid3DInfo<double> dbg_flx
                       , CudaRegularGrid3DInfo<double> dbg_fly
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
                    const coord3 coord = coord3(i, j, k);

                    double lap_ij = 
                        4 * CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) 
                        - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0);
                    double lap_imj = 
                        4 * CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, -2, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0);
                    double lap_ipj =
                        4 * CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +2, 0, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0);
                    double lap_ijm =
                        4 * CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, 0, -2, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0);
                    double lap_ijp =
                        4 * CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0);
            
                    double flx_ij = lap_ipj - lap_ij;
                    flx_ij = flx_ij * (CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : flx_ij;
            
                    double flx_imj = lap_ij - lap_imj;
                    flx_imj = flx_imj * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)) > 0 ? 0 : flx_imj;
            
                    double fly_ij = lap_ijp - lap_ij;
                    fly_ij = fly_ij * (CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : fly_ij;
            
                    double fly_ijm = lap_ij - lap_ijm;
                    fly_ijm = fly_ijm * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)) > 0 ? 0 : fly_ijm;
            
                    CUDA_REGULAR(out, coord) =
                        CUDA_REGULAR(in, coord)
                        - CUDA_REGULAR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);
            
                    // for debugging purposes:
                    #ifdef HDIFF_DEBUG
                    CUDA_REGULAR(dbg_lap, coord) = lap_ij;
                    CUDA_REGULAR_NEIGH(dbg_lap, coord, -1, 0, 0) = lap_imj;
                    CUDA_REGULAR_NEIGH(dbg_lap, coord, 0, -1, 0) = lap_ijm;
                    CUDA_REGULAR_NEIGH(dbg_lap, coord, +1, 0, 0) = lap_ipj;
                    CUDA_REGULAR_NEIGH(dbg_lap, coord, 0, +1, 0) = lap_ijp;
                    CUDA_REGULAR(dbg_flx, coord) = flx_ij;
                    CUDA_REGULAR_NEIGH(dbg_flx, coord, -1, 0, 0) = flx_imj;
                    CUDA_REGULAR(dbg_fly, coord) = fly_ij;
                    CUDA_REGULAR_NEIGH(dbg_fly, coord, 0, -1, 0) = fly_ijm;
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
    __global__
    void kernel_coop(HdiffBase::Info info,
                    CudaRegularGrid3DInfo<double> in,
                    CudaRegularGrid3DInfo<double> out,
                    CudaRegularGrid3DInfo<double> coeff,
                    CudaRegularGrid3DInfo<double> lap,
                    CudaRegularGrid3DInfo<double> flx,
                    CudaRegularGrid3DInfo<double> fly) {
        
        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        const coord3 coord(i, j, k);
        
        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }

        // Calculate own laplace
        double lap_ij = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) 
            - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
            - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0);
        CUDA_REGULAR(lap, coord) = lap_ij;

        // Sync threads to enable access to their laplace calculations
        __syncthreads();

        double lap_ipj;
        if(threadIdx.x == blockDim.x-1) {
            // rightmost in block, need to compute right dependency ourselves
            lap_ipj = 4 * CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +2, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0);
        } else {
            lap_ipj = CUDA_REGULAR_NEIGH(lap, coord, +1, 0, 0);
        }

        double lap_ijp;
        if(threadIdx.y == blockDim.y-1) {
            lap_ijp = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0);
        } else {
            lap_ijp = CUDA_REGULAR_NEIGH(lap, coord, 0, +1, 0);
        }

        // Own flx/fly calculation
        double flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : flx_ij;
        CUDA_REGULAR(flx, coord) = flx_ij;

        double fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : fly_ij;
        CUDA_REGULAR(fly, coord) = fly_ij;

        // Make flx/fly available to other threads by synchronizing
        __syncthreads();

        double flx_imj;
        if(threadIdx.x == 0) {
            // leftmost in block, need to compute left dependency ourselves
            double lap_imj = 4 * CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -2, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0);
            flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)) > 0 ? 0 : flx_imj;
        } else {
            flx_imj = CUDA_REGULAR_NEIGH(flx, coord, -1, 0, 0);
        }

        double fly_ijm;
        if(threadIdx.y == 0) {
            // need to also calculate lap for j - 1 as we are at boundary
            double lap_ijm = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, 0, -2, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0);
            fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)) > 0 ? 0 : fly_ijm;
        } else {
            fly_ijm = CUDA_REGULAR_NEIGH(fly, coord, 0, -1, 0);
        }

        CUDA_REGULAR(out, coord) = CUDA_REGULAR(in, coord)
                - CUDA_REGULAR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);

    }

    /** Just like the coop kernel, but uses shared memory fo threads instead
     * of managed memory for the lap, flx, fly results. */
    __global__
    void kernel_shared(HdiffBase::Info info,
                    int block_size,
                    coord3 local_dimensions,
                    coord3 local_strides,
                    CudaRegularGrid3DInfo<double> in,
                    CudaRegularGrid3DInfo<double> out,
                    CudaRegularGrid3DInfo<double> coeff) {
        
        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;

        if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z) {
            return;
        }

        // Shared memory
        extern __shared__ double smem[];
        //int block_size = blockDim.x*blockDim.y*blockDim.z;

        // Local block grid position (position in this blocks "cache" grid)
        const coord3 local_coord(threadIdx.x, threadIdx.y, threadIdx.z);
        //const coord3 local_dimensions = coord3(blockDim.x, blockDim.y, blockDim.z);
        //const coord3 local_strides = coord3(1, blockDim.x, blockDim.x*blockDim.y);

        // Local grids holding results for laplace, flx and fly calcualted by other threads
        CudaRegularGrid3DInfo<double> local_lap = {
            .data = smem,
            .dimensions = local_dimensions,
            .strides = local_strides
        };
        CudaRegularGrid3DInfo<double> local_flx = {
            .data = &smem[block_size],
            .dimensions = local_dimensions,
            .strides = local_strides
        };
        CudaRegularGrid3DInfo<double> local_fly = {
            .data = &smem[2*block_size],
            .dimensions = local_dimensions,
            .strides = local_strides
        };
        
        const coord3 coord(i, j, k);

        // Calculate own laplace
        double lap_ij = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) 
            - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
            - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0);
        CUDA_REGULAR(local_lap, local_coord) = lap_ij;

        // Sync threads to enable access to their laplace calculations
        __syncthreads();

        double lap_ipj;
        if(threadIdx.x == blockDim.x-1 || i == info.max_coord.x-1) {
            // rightmost in block, need to compute right dependency ourselves
            lap_ipj = 4 * CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +2, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0);
        } else {
            lap_ipj = CUDA_REGULAR_NEIGH(local_lap, local_coord, +1, 0, 0);
        }

        double lap_ijp;
        if(threadIdx.y == blockDim.y-1 || j == info.max_coord.y-1) {
            lap_ijp = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0);
        } else {
            lap_ijp = CUDA_REGULAR_NEIGH(local_lap, local_coord, 0, +1, 0);
        }

        // Own flx/fly calculation
        double flx_ij = lap_ipj - lap_ij;
        flx_ij = flx_ij * (CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : flx_ij;
        CUDA_REGULAR(local_flx, local_coord) = flx_ij;

        double fly_ij = lap_ijp - lap_ij;
        fly_ij = fly_ij * (CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : fly_ij;
        CUDA_REGULAR(local_fly, local_coord) = fly_ij;

        // Make flx/fly available to other threads by synchronizing
        __syncthreads();

        double flx_imj;
        if(threadIdx.x == 0) {
            // leftmost in block, need to compute left dependency ourselves
            double lap_imj = 4 * CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -2, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0);
            flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)) > 0 ? 0 : flx_imj;
        } else {
            flx_imj = CUDA_REGULAR_NEIGH(local_flx, local_coord, -1, 0, 0);
        }

        double fly_ijm;
        if(threadIdx.y == 0) {
            // need to also calculate lap for j - 1 as we are at boundary
            double lap_ijm = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, 0, -2, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0);
            fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)) > 0 ? 0 : fly_ijm;
        } else {
            fly_ijm = CUDA_REGULAR_NEIGH(local_fly, local_coord, 0, -1, 0);
        }

        CUDA_REGULAR(out, coord) = CUDA_REGULAR(in, coord)
                - CUDA_REGULAR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);
    }

    /** Like kernel_shared, but uses a k loop and in each loop iteration
     * threads are synchronized. This reduces the amount of memory required,
     * as only the lap/flx/fly for one k-level is stored in shared memory. */
    __global__
    void kernel_shared_kloop(HdiffBase::Info info,
                    //int block_size,
                    //coord3 local_dimensions,
                    //coord3 local_strides,
                    CudaRegularGrid3DInfo<double> in,
                    CudaRegularGrid3DInfo<double> out,
                    CudaRegularGrid3DInfo<double> coeff) {
        
        // Global grid position
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }

        // Shared memory
        extern __shared__ double smem[];
        int block_size = blockDim.x*blockDim.y;

        // Local block grid position (position in this blocks "cache" grid)
        const coord3 local_coord(threadIdx.x, threadIdx.y, 0);
        const coord3 local_dimensions(blockDim.x, blockDim.y, 1);
        const coord3 local_strides(1, blockDim.x, blockDim.x*blockDim.y);

        // Local grids holding results for laplace, flx and fly calcualted by other threads
        CudaRegularGrid3DInfo<double> local_lap = {
            .data = smem,
            .dimensions = local_dimensions,
            .strides = local_strides
        };
        CudaRegularGrid3DInfo<double> local_flx = {
            .data = &smem[block_size],
            .dimensions = local_dimensions,
            .strides = local_strides
        };
        CudaRegularGrid3DInfo<double> local_fly = {
            .data = &smem[2*block_size],
            .dimensions = local_dimensions,
            .strides = local_strides
        };
        
        // K-loop
        for(int k = info.halo.z; k < info.max_coord.z; k++) {
            const coord3 coord(i, j, k);

            // Calculate own laplace
            double lap_ij = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) 
                - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0);
            CUDA_REGULAR(local_lap, local_coord) = lap_ij;

            // Sync threads to enable access to their laplace calculations
            __syncthreads();

            double lap_ipj;
            if(threadIdx.x == blockDim.x-1 || i == info.max_coord.x-1) {
                // rightmost in block, need to compute right dependency ourselves
                lap_ipj = 4 * CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +2, 0, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0);
            } else {
                lap_ipj = CUDA_REGULAR_NEIGH(local_lap, local_coord, +1, 0, 0);
            }

            double lap_ijp;
            if(threadIdx.y == blockDim.y-1 || j == info.max_coord.y-1) {
                lap_ijp = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0);
            } else {
                lap_ijp = CUDA_REGULAR_NEIGH(local_lap, local_coord, 0, +1, 0);
            }

            // Own flx/fly calculation
            double flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : flx_ij;
            CUDA_REGULAR(local_flx, local_coord) = flx_ij;

            double fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : fly_ij;
            CUDA_REGULAR(local_fly, local_coord) = fly_ij;

            // Make flx/fly available to other threads by synchronizing
            __syncthreads();

            double flx_imj;
            if(threadIdx.x == 0) {
                // leftmost in block, need to compute left dependency ourselves
                double lap_imj = 4 * CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, -2, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0)
                    - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0);
                flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)) > 0 ? 0 : flx_imj;
            } else {
                flx_imj = CUDA_REGULAR_NEIGH(local_flx, local_coord, -1, 0, 0);
            }

            double fly_ijm;
            if(threadIdx.y == 0) {
                // need to also calculate lap for j - 1 as we are at boundary
                double lap_ijm = 4 * CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0)
                        - CUDA_REGULAR_NEIGH(in, coord, 0, -2, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0);
                fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)) > 0 ? 0 : fly_ijm;
            } else {
                fly_ijm = CUDA_REGULAR_NEIGH(local_fly, local_coord, 0, -1, 0);
            }

            CUDA_REGULAR(out, coord) = CUDA_REGULAR(in, coord)
                    - CUDA_REGULAR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }
    }

    __global__
    void kernel_kloop(HdiffBase::Info info,
                       CudaRegularGrid3DInfo<double> in,
                       CudaRegularGrid3DInfo<double> out,
                       CudaRegularGrid3DInfo<double> coeff) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }
        
        for(int k = info.halo.z; k < info.max_coord.z; k++) {
            const coord3 coord = coord3(i, j, k);
            double lap_ij = 
                4 * CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) 
                - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0);
            double lap_imj = 
                4 * CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -2, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0);
            double lap_ipj =
                4 * CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +2, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0);
            double lap_ijm =
                4 * CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, -2, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0);
            double lap_ijp =
                4 * CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0);

            double flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : flx_ij;
            double flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)) > 0 ? 0 : flx_imj;
            double fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : fly_ij;
            double fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)) > 0 ? 0 : fly_ijm;

            CUDA_REGULAR(out, coord) =
                CUDA_REGULAR(in, coord)
                - CUDA_REGULAR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);
        }
    }

    __global__
    void kernel_idxvar(HdiffBase::Info info,
                       CudaRegularGrid3DInfo<double> in,
                       CudaRegularGrid3DInfo<double> out,
                       CudaRegularGrid3DInfo<double> coeff) {

        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }

        coord3 coord = coord3(i, j, info.halo.z);

        int n_0_0_0       = CUDA_REGULAR_INDEX(in, coord);
        int n_0_n1_0      = CUDA_REGULAR_NEIGHBOR(in, coord,   0, -1, 0);
        int n_0_n2_0      = CUDA_REGULAR_NEIGHBOR(in, coord,  0, -2, 0);
        int n_n1_0_0      = CUDA_REGULAR_NEIGHBOR(in, coord,  -1, 0, 0);
        int n_n1_n1_0     = CUDA_REGULAR_NEIGHBOR(in, coord,  -1, -1, 0);
        int n_n2_0_0      = CUDA_REGULAR_NEIGHBOR(in, coord, -2, 0, 0);
        //int n_n2_n1_0     = CUDA_REGULAR_NEIGHBOR(in, coord,  -2, -1, 0);
        int n_0_p1_0      = CUDA_REGULAR_NEIGHBOR(in, coord,   0, +1, 0);
        int n_0_p2_0      = CUDA_REGULAR_NEIGHBOR(in, coord,  0, +2, 0);
        int n_p1_0_0      = CUDA_REGULAR_NEIGHBOR(in, coord,  +1, 0, 0);
        int n_p1_p1_0     = CUDA_REGULAR_NEIGHBOR(in, coord,  +1, +1, 0);
        int n_p2_0_0      = CUDA_REGULAR_NEIGHBOR(in, coord, +2, 0, 0);
        //int n_p2_p1_0     = CUDA_REGULAR_NEIGHBOR(in, coord,  0, +1, 0);     
        int n_n1_p1_0     = CUDA_REGULAR_NEIGHBOR(in, coord,  -1, +1, 0);
        int n_p1_n1_0     = CUDA_REGULAR_NEIGHBOR(in, coord,  +1, -1, 0);

        for(int k = info.halo.z; k < info.max_coord.z; k++) {

            double lap_ij = 
                4 * CUDA_REGULAR_AT(in, n_0_0_0) 
                - CUDA_REGULAR_AT(in, n_n1_0_0) - CUDA_REGULAR_AT(in, n_p1_0_0)
                - CUDA_REGULAR_AT(in, n_0_n1_0) - CUDA_REGULAR_AT(in, n_0_p1_0);
            double lap_imj = 
                4 * CUDA_REGULAR_AT(in, n_n1_0_0)
                - CUDA_REGULAR_AT(in, n_n2_0_0) - CUDA_REGULAR_AT(in, n_0_0_0)
                - CUDA_REGULAR_AT(in, n_n1_n1_0) - CUDA_REGULAR_AT(in, n_n1_p1_0);
            double lap_ipj =
                4 * CUDA_REGULAR_AT(in, n_p1_0_0)
                - CUDA_REGULAR_AT(in, n_0_0_0) - CUDA_REGULAR_AT(in, n_p2_0_0)
                - CUDA_REGULAR_AT(in, n_p1_n1_0) - CUDA_REGULAR_AT(in, n_p1_p1_0);
            double lap_ijm =
                4 * CUDA_REGULAR_AT(in, n_0_n1_0)
                - CUDA_REGULAR_AT(in, n_n1_n1_0) - CUDA_REGULAR_AT(in, n_p1_n1_0)
                - CUDA_REGULAR_AT(in, n_0_n2_0) - CUDA_REGULAR_AT(in, n_0_0_0);
            double lap_ijp =
                4 * CUDA_REGULAR_AT(in, n_0_p1_0)
                - CUDA_REGULAR_AT(in, n_n1_p1_0) - CUDA_REGULAR_AT(in, n_p1_p1_0)
                - CUDA_REGULAR_AT(in, n_0_0_0) - CUDA_REGULAR_AT(in, n_0_p2_0);
    
            double flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (CUDA_REGULAR_AT(in, n_p1_0_0) - CUDA_REGULAR_AT(in, n_0_0_0)) > 0 ? 0 : flx_ij;
    
            double flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (CUDA_REGULAR_AT(in, n_0_0_0) - CUDA_REGULAR_AT(in, n_n1_0_0)) > 0 ? 0 : flx_imj;
    
            double fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (CUDA_REGULAR_AT(in, n_0_p1_0) - CUDA_REGULAR_AT(in, n_0_0_0)) > 0 ? 0 : fly_ij;
    
            double fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (CUDA_REGULAR_AT(in, n_0_0_0) - CUDA_REGULAR_AT(in, n_0_n1_0)) > 0 ? 0 : fly_ijm;
    
            CUDA_REGULAR_AT(out, n_0_0_0) =
                CUDA_REGULAR_AT(in, n_0_0_0)
                - CUDA_REGULAR_AT(coeff, n_0_0_0) * (flx_ij - flx_imj + fly_ij - fly_ijm);

            n_0_0_0       += in.strides.z;
            n_0_n1_0      += in.strides.z;
            n_0_n2_0      += in.strides.z;
            n_n1_0_0      += in.strides.z;
            n_n1_n1_0     += in.strides.z;
            n_n2_0_0      += in.strides.z;
            //n_n2_n1_0     += in.strides.z;
            n_0_p1_0      += in.strides.z;
            n_0_p2_0      += in.strides.z;
            n_p1_0_0      += in.strides.z;
            n_p1_p1_0     += in.strides.z;
            n_p2_0_0      += in.strides.z;
            //n_p2_p1_0     += in.strides.z;
            n_n1_p1_0     += in.strides.z;
            n_p1_n1_0     += in.strides.z;

        }
    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
class HdiffCudaBenchmark : public HdiffBaseBenchmark {

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
    
};

// IMPLEMENTATIONS

HdiffCudaBenchmark::HdiffCudaBenchmark(coord3 size, HdiffCudaRegular::Variant variant) :
HdiffBaseBenchmark(size) {
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
    } else {
        this->name = "hdiff-regular-idxvar";
    }
}

void HdiffCudaBenchmark::run() {
    if(this->variant == HdiffCudaRegular::direct) {
        HdiffCudaRegular::kernel_direct<<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo()
            #ifdef HDIFF_DEBUG
            , (dynamic_cast<CudaRegularGrid3D<double>*>(this->lap))->get_gridinfo()
            , (dynamic_cast<CudaRegularGrid3D<double>*>(this->flx))->get_gridinfo()
            , (dynamic_cast<CudaRegularGrid3D<double>*>(this->fly))->get_gridinfo()
            #endif
        );
    } else if(this->variant == HdiffCudaRegular::kloop) {
        HdiffCudaRegular::kernel_kloop<<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo()
        );
    } else if(this->variant == HdiffCudaRegular::coop) {
        HdiffCudaRegular::kernel_coop<<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->lap))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->flx))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->fly))->get_gridinfo()
        );
    } else if(this->variant == HdiffCudaRegular::shared) {
        dim3 numthreads = this->numthreads();
        int smem_size = 3*numthreads.x*numthreads.y*numthreads.z*sizeof(double);
        HdiffCudaRegular::kernel_shared<<<this->numblocks(), numthreads, smem_size>>>(
            this->get_info(),
            numthreads.x*numthreads.y*numthreads.z,
            coord3(numthreads.x, numthreads.y, numthreads.z),
            coord3(1, numthreads.x, numthreads.x*numthreads.y),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo()
        );
    } else if(this->variant == HdiffCudaRegular::shared_kloop) {
        dim3 numthreads = this->numthreads();
        int smem_size = 3*numthreads.x*numthreads.y*sizeof(double);
        HdiffCudaRegular::kernel_shared_kloop<<<this->numblocks(), numthreads, smem_size>>>(
            this->get_info(),
            //numthreads.x*numthreads.y*numthreads.z,
            //coord3(numthreads.x, numthreads.y, numthreads.z),
            //coord3(1, numthreads.x, numthreads.x*numthreads.y),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo()
        );
    } else {
        HdiffCudaRegular::kernel_idxvar<<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
            (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo()
        );
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

void HdiffCudaBenchmark::setup() {
    this->input = new CudaRegularGrid3D<double>(this->size);
    this->output = new CudaRegularGrid3D<double>(this->size);
    this->coeff = new CudaRegularGrid3D<double>(this->size);
    this->lap = new CudaRegularGrid3D<double>(this->size);
    this->flx = new CudaRegularGrid3D<double>(this->size);
    this->fly = new CudaRegularGrid3D<double>(this->size);
    this->HdiffBaseBenchmark::setup();
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

void HdiffCudaBenchmark::teardown() {
    this->input->deallocate();
    this->output->deallocate();
    this->coeff->deallocate();
    this->lap->deallocate();
    this->flx->deallocate();
    this->fly->deallocate();
    this->HdiffBaseBenchmark::teardown();
}

void HdiffCudaBenchmark::post() {
    this->Benchmark::post();
    this->HdiffBaseBenchmark::post();
}

dim3 HdiffCudaBenchmark::numthreads() {
    dim3 numthreads = this->HdiffBaseBenchmark::numthreads();
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar ||
        this->variant == HdiffCudaRegular::shared_kloop) {
        numthreads.z = 1;
    }
    return numthreads;
}

dim3 HdiffCudaBenchmark::numblocks() {
    dim3 numblocks = this->HdiffBaseBenchmark::numblocks();
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar ||
        this->variant == HdiffCudaRegular::shared_kloop) {
        numblocks.z = 1;
    }
    return numblocks;
}

#endif