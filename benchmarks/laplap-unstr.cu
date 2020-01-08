#ifndef LAPLAP_UNSTR_BENCHMARK_H
#define LAPLAP_UNSTR_BENCHMARK_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/laplap-base.cu"
#include "grids/cuda-unstructured.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the unstructured grid macros. */
namespace LapLapUnstr {

    enum Variant { unfused, naive, idxvar, idxvar_kloop, idxvar_shared };

    #define GRID_ARGS const int * __restrict__ neighbor_data, const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_UNSTR_INDEX(y_stride, z_stride, x, y, z)
    #define NEIGHBOR(x, y, z, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighbor_data, y_stride, z_stride, x, y, z, x_, y_, z_)
    #define NEIGHBOR_OF_INDEX(idx, x, y, z) GRID_UNSTR_NEIGHBOR_OF_INDEX(neighbor_data, z_stride, idx, x, y, z)
    #define DOUBLE_NEIGHBOR(x, y, z, x1, y1, z1, x2, y2, z2) NEIGHBOR_OF_INDEX(NEIGHBOR(x, y, z, x2, y2, z2), x1, y1, z1) // x2 y2 z2 can be zero in navie kernel ...
    #define NEXT_Z_NEIGHBOR_OF_INDEX(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/laplap-unfused.cu"
    #include "kernels/laplap-naive.cu"
    #include "kernels/laplap-idxvar.cu"
    #include "kernels/laplap-idxvar-kloop.cu"
    #include "kernels/laplap-idxvar-shared.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
    #undef NEIGHBOR_OF_INDEX
    #undef DOUBLE_NEIGHBOR
    #undef NEXT_Z_NEIGHBOR_OF_INDEX

};


template<typename value_t>
class LapLapUnstrBenchmark : public LapLapBaseBenchmark<value_t> {

    public:

    LapLapUnstrBenchmark(coord3 size, LapLapUnstr::Variant variant = LapLapUnstr::naive);

    LapLapUnstr::Variant variant;

    void setup();
    void run();

    dim3 numthreads();
    dim3 numblocks();

};

// IMPLEMENTATIONS

template<typename value_t>
LapLapUnstrBenchmark<value_t>::LapLapUnstrBenchmark(coord3 size, LapLapUnstr::Variant variant) :
LapLapBaseBenchmark<value_t>(size),
variant(variant) {
    this->variant = variant;
    if(variant == LapLapUnstr::naive) {
        this->name = "laplap-unstr-naive";
    } else if(variant == LapLapUnstr::unfused) {
        this->name = "laplap-unstr-unfused";
    } else if(variant == LapLapUnstr::idxvar) {
        this->name = "laplap-unstr-idxvar";
    } else if(variant == LapLapUnstr::idxvar_kloop) {
        this->name = "laplap-unstr-idxvar-kloop";
    } else if(variant == LapLapUnstr::idxvar_shared) {
        this->name = "laplap-unstr-idxvar-shared";
    }
}

template<typename value_t>
void LapLapUnstrBenchmark<value_t>::run() {
    CudaUnstructuredGrid3D<value_t> *unstr_input = (dynamic_cast<CudaUnstructuredGrid3D<value_t>*>(this->input));
    coord3 strides = coord3(1, unstr_input->dimensions.x, unstr_input->dimensions.x*unstr_input->dimensions.y);
    int *neighbor_data = unstr_input->neighbor_data;
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    const coord3 halo1(1, 1, 0);
    const coord3 halo2(2, 2, 0);
    const coord3 max_coord1 = this->size - halo1;
    const coord3 max_coord2 = this->size - halo2;
    if(this->variant == LapLapUnstr::naive) {
        LapLapUnstr::laplap_naive<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighbor_data,
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapUnstr::idxvar) {
        LapLapUnstr::laplap_idxvar<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighbor_data,
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapUnstr::idxvar_kloop) {
        LapLapUnstr::laplap_idxvar_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighbor_data,
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapUnstr::idxvar_shared) {
        dim3 numthreads = this->numthreads();
        int smem = numthreads.x*numthreads.y*13*sizeof(int);
        LapLapUnstr::laplap_idxvar_shared<value_t><<<this->numblocks(), numthreads, smem>>>(
            neighbor_data,
            strides.y, strides.z,
            halo2, max_coord2,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapUnstr::unfused) {
        LapLapUnstr::lap<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighbor_data,
            strides.y, strides.z,
            halo1, max_coord1,
            this->input->data,
            this->intermediate->data
        );
        LapLapUnstr::lap<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighbor_data,
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
void LapLapUnstrBenchmark<value_t>::setup() {
    CudaUnstructuredGrid3D<value_t> *input = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    this->input = input;
    this->output = new CudaUnstructuredGrid3D<value_t>(this->size, input->neighbor_data);
    if(this->variant == LapLapUnstr::unfused) {
        this->intermediate = new CudaUnstructuredGrid3D<value_t>(this->size, input->neighbor_data);
    }
    if(this->variant == LapLapUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->LapLapBaseBenchmark<value_t>::setup();
}

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->LapLapBaseBenchmark<value_t>::numthreads();
    if(this->variant == LapLapUnstr::idxvar_kloop) {
        numthreads.z = 1;
    }
    return numthreads;
}

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->LapLapBaseBenchmark<value_t>::numblocks();
    if(this->variant == LapLapUnstr::idxvar_kloop) {
        numblocks.z = 1;
    }
    return numblocks;
}

#endif