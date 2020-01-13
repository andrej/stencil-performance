#ifndef LAPLAP_UNSTR_BENCHMARK_H
#define LAPLAP_UNSTR_BENCHMARK_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/laplap-base.cu"
#include "grids/cuda-unstructured.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the unstructured grid macros. */
namespace LapLapUnstr {

    enum Variant { unfused, naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared };

    #define GRID_ARGS const int * __restrict__ neighborships, const int y_stride, const int z_stride, 
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + (z_)*blockDim.x*gridDim.x*blockDim.y*gridDim.y
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
    #define K_STEP k*z_stride

    #include "kernels/laplap-unfused.cu"
    #include "kernels/laplap-naive.cu"
    #include "kernels/laplap-idxvar.cu"
    #include "kernels/laplap-idxvar-kloop.cu"
    #include "kernels/laplap-idxvar-kloop-sliced.cu"
    #include "kernels/laplap-idxvar-shared.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR
    #undef NEXT_Z_NEIGHBOR

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
    
    void parse_args();
    int k_per_thread = 16;

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
    } else if(variant == LapLapUnstr::idxvar_kloop_sliced) {
        this->name = "laplap-unstr-idxvar-kloop-sliced";
    } else if(variant == LapLapUnstr::idxvar_shared) {
        this->name = "laplap-unstr-idxvar-shared";
    }
}

template<typename value_t>
void LapLapUnstrBenchmark<value_t>::run() {
    CudaUnstructuredGrid3D<value_t> *unstr_input = (dynamic_cast<CudaUnstructuredGrid3D<value_t>*>(this->input));
    coord3 strides = coord3(1, unstr_input->dimensions.x, unstr_input->dimensions.x*unstr_input->dimensions.y);
    int *neighborships = unstr_input->neighborships;
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    const coord3 halo1(1, 1, 0);
    const coord3 halo2(2, 2, 0);
    const coord3 max_coord1 = this->size - halo1;
    const coord3 max_coord2 = this->size - halo2;
    if(this->variant == LapLapUnstr::naive) {
        LapLapUnstr::laplap_naive<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            strides.z,
            halo2, max_coord2,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == LapLapUnstr::idxvar) {
        LapLapUnstr::laplap_idxvar<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            strides.z,
            halo2, max_coord2,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == LapLapUnstr::idxvar_kloop) {
        LapLapUnstr::laplap_idxvar_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            strides.z,
            halo2, max_coord2,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        LapLapUnstr::laplap_idxvar_kloop_sliced<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            strides.z,
            this->k_per_thread,
            halo2, max_coord2,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == LapLapUnstr::idxvar_shared) {
        dim3 numthreads = this->numthreads();
        int smem = numthreads.x*numthreads.y*13*sizeof(int);
        LapLapUnstr::laplap_idxvar_shared<value_t><<<this->numblocks(), numthreads, smem>>>(
            neighborships,
            strides.z,
            halo2, max_coord2,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == LapLapUnstr::unfused) {
        LapLapUnstr::lap<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            strides.z,
            halo1, max_coord1,
            this->input->pointer(coord3(0, 0, 0)),
            this->intermediate->pointer(coord3(0, 0, 0))
        );
        LapLapUnstr::lap<value_t><<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            strides.z,
            halo2, max_coord2,
            this->intermediate->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0))
        );
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

template<typename value_t>
void LapLapUnstrBenchmark<value_t>::setup() {
    coord3 halo2(2, 2, 0);
    CudaUnstructuredGrid3D<value_t> *input = CudaUnstructuredGrid3D<value_t>::create_regular(this->size-2*halo, halo);
    this->input = input;
    this->output = new CudaUnstructuredGrid3D<value_t>(this->size-2*halo2, input->neighborships);
    if(this->variant == LapLapUnstr::unfused) {
        this->intermediate = new CudaUnstructuredGrid3D<value_t>(this->size, input->neighborships);
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
}ƒ

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->LapLapBaseBenchmark<value_t>::numblocks();
    dim3 numthreads = this->numthreads();
    if(this->variant == LapLapUnstr::idxvar_kloop) {
        numblocks.z = 1;
    } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        int sz_z = ((this->size.z + this->k_per_thread - 1) / this->k_per_thread);
        numblocks.z = (sz_z + numthreads.z - 1) / numthreads.z;
    }
    return numblocks;
}


template<typename value_t>
void LapLapUnstrBenchmark<value_t>::parse_args() {
    if(this->argc > 0 && this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        sscanf(this->argv[0], "%d", &this->k_per_thread);
    } else {
        this->Benchmark::parse_args();
    }
}

#endif