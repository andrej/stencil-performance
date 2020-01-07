#ifndef LAPLAP_REGULAR_BENCHMARK_H
#define LAPLAP_REGULAR_BENCHMARK_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/laplap-base.cu"
#include "grids/cuda-regular.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the regular grid macros. */
namespace LapLapRegular {

    enum Variant { unfused, naive, idxvar, idxvar_kloop };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define NEIGHBOR(x, y, z, x_, y_, z_) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, x, y, z, x_, y_, z_)
    #define NEIGHBOR_OF_INDEX(idx, x, y, z) GRID_REGULAR_NEIGHBOR_OF_INDEX(y_stride, z_stride, idx, x, y, z)
    #define NEXT_Z_NEIGHBOR_OF_INDEX(idx) GRID_REGULAR_NEIGHBOR_OF_INDEX(y_stride, z_stride, idx, 0, 0, 1)

    #include "kernels/laplap-unfused.cu"
    #include "kernels/laplap-naive.cu"
    #include "kernels/laplap-idxvar.cu"
    #include "kernels/laplap-idxvar-kloop.cu"

    #define SMEM_GRID_ARGS
    #define SMEM_INDEX(_x, _y, _z) GRID_REGULAR_INDEX(blockDim.x, blockDim.x*blockDim.y, _x, _y, _z)
    #define SMEM_NEIGHBOR(_x, _y, _z, x_, y_, z_) GRID_REGULAR_NEIGHBOR(blockDim.x, blockDim.x*blockDim.y, _x, _y, _z, x_, y_, z_)

    // ...

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
    #undef NEIGHBOR_OF_INDEX
    #undef NEXT_Z_NEIGHBOR_OF_INDEX
    #undef K_STEP
    #undef SMEM_GRID_ARGS
    #undef SMEM_INDEX
    #undef SMEM_NEIGHBOR

};


template<typename value_t>
class LapLapRegularBenchmark : public LapLapBaseBenchmark<value_t> {

    public:

    LapLapRegularBenchmark(coord3 size, LapLapRegular::Variant variant = LapLapRegular::naive);

    LapLapRegular::Variant variant;

    void setup();
    void run();

    dim3 numthreads();
    dim3 numblocks();

};

// IMPLEMENTATIONS

template<typename value_t>
LapLapRegularBenchmark<value_t>::LapLapRegularBenchmark(coord3 size, LapLapRegular::Variant variant) :
LapLapBaseBenchmark<value_t>(size),
variant(variant) {
    this->variant = variant;
    if(variant == LapLapRegular::naive) {
        this->name = "laplap-regular-naive";
    } else if(variant == LapLapRegular::unfused) {
        this->name = "laplap-regular-unfused";
    } else if(variant == LapLapRegular::idxvar) {
        this->name = "laplap-regular-idxvar";
    } else if(variant == LapLapRegular::idxvar_kloop) {
        this->name = "laplap-regular-idxvar-kloop";
    }
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::run() {
    coord3 strides = (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_strides();
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    const coord3 halo1(1, 1, 0);
    const coord3 halo2(2, 2, 0);
    const coord3 max_coord1 = this->size - halo1;
    const coord3 max_coord2 = this->size - halo2;
    if(this->variant == LapLapRegular::naive) {
        LapLapRegular::laplap_naive<value_t><<<this->numblocks(), this->numthreads()>>>(
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapRegular::idxvar) {
        LapLapRegular::laplap_idxvar<value_t><<<this->numblocks(), this->numthreads()>>>(
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapRegular::idxvar_kloop) {
        LapLapRegular::laplap_idxvar_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapRegular::unfused) {
        LapLapRegular::lap<value_t><<<this->numblocks(), this->numthreads()>>>(
            strides.y, strides.z,
            halo1, max_coord1,
            this->input->data,
            this->intermediate->data
        );
        LapLapRegular::lap<value_t><<<this->numblocks(), this->numthreads()>>>(
            strides.y, strides.z,
            halo2, max_coord2,
            this->intermediate->data,
            this->output->data
        );
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::setup() {
    this->input = new CudaRegularGrid3D<value_t>(this->size);
    this->output = new CudaRegularGrid3D<value_t>(this->size);
    if(this->variant == LapLapRegular::unfused) {
        this->intermediate = new CudaRegularGrid3D<value_t>(this->size);
    }
    this->LapLapBaseBenchmark<value_t>::setup();
}

template<typename value_t>
dim3 LapLapRegularBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->LapLapBaseBenchmark<value_t>::numthreads();
    if(this->variant == LapLapRegular::idxvar_kloop) {
        numthreads.z = 1;
    }
    return numthreads;
}

template<typename value_t>
dim3 LapLapRegularBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->LapLapBaseBenchmark<value_t>::numblocks();
    if(this->variant == LapLapRegular::idxvar_kloop) {
        numblocks.z = 1;
    }
    return numblocks;
}

#endif