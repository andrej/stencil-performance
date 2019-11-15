#ifndef HDIFF_CUDA_H
#define HDIFF_CUDA_H
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaRegular {

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
        if(i-info.halo.x >= info.inner_size.x || j-info.halo.y >= info.inner_size.y || k-info.halo.z > info.inner_size.z) {
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

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
class HdiffCudaBenchmark : public HdiffBaseBenchmark {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();

};

// IMPLEMENTATIONS

HdiffCudaBenchmark::HdiffCudaBenchmark(coord3 size) :
HdiffBaseBenchmark(size) {
    this->name = "hdiff-cuda";
}

void HdiffCudaBenchmark::run() {
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

#endif