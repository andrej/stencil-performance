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

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < z_stride && k < max_coord.z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
    #define K_STEP k*z_stride

    namespace NonChasing {
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1)+(x2), (y1)+(y2), (z1)+(z2))
        
        #include "kernels/laplap-unfused.cu"
        #include "kernels/laplap-naive.cu"
        #include "kernels/laplap-idxvar.cu"
        #include "kernels/laplap-idxvar-kloop.cu"
        #include "kernels/laplap-idxvar-kloop-sliced.cu"
        #include "kernels/laplap-idxvar-shared.cu"

        #undef DOUBLE_NEIGHBOR
    };

    namespace Chasing {
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
        
        #include "kernels/laplap-unfused.cu"
        #include "kernels/laplap-naive.cu"
        #include "kernels/laplap-idxvar.cu"
        #include "kernels/laplap-idxvar-kloop.cu"
        #include "kernels/laplap-idxvar-kloop-sliced.cu"
        #include "kernels/laplap-idxvar-shared.cu"

        #undef DOUBLE_NEIGHBOR
    };

    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS
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

    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    
    void parse_args();
    int k_per_thread = 16;
    bool pointer_chasing = false;

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
    int *neighborships = unstr_input->neighborships;
    const int z_stride = unstr_input->z_stride();
    const int offs = this->input->index(coord3(0, 0, 0));
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    const coord3 halo1(1, 1, 0);
    const coord3 halo2(2, 2, 0);
    if(this->variant == LapLapUnstr::unfused
       || this->variant == LapLapUnstr::naive
       || this->variant == LapLapUnstr::idxvar
       || this->variant == LapLapUnstr::idxvar_kloop
       || this->variant == LapLapUnstr::idxvar_shared) {
        int smem = 0;
        auto kernel = &LapLapUnstr::NonChasing::laplap_naive<value_t>;
        if(this->variant == LapLapUnstr::naive && this->pointer_chasing) {
            kernel = &LapLapUnstr::Chasing::laplap_naive<value_t>;
        } else if(this->variant == LapLapUnstr::idxvar) {
            if(this->pointer_chasing) {
                kernel = &LapLapUnstr::Chasing::laplap_idxvar<value_t>;
            } else {
                kernel = &LapLapUnstr::NonChasing::laplap_idxvar<value_t>;
            }
        } else if(this->variant == LapLapUnstr::idxvar_kloop) {
            if(this->pointer_chasing) {
                kernel = &LapLapUnstr::Chasing::laplap_idxvar_kloop<value_t>;
            } else {
                kernel = &LapLapUnstr::NonChasing::laplap_idxvar_kloop<value_t>;
            }
        } else if(this->variant == LapLapUnstr::idxvar_shared) {
            smem = numthreads.x*numthreads.y*13*sizeof(int);
            if(this->pointer_chasing) {
                kernel = &LapLapUnstr::Chasing::laplap_idxvar_shared<value_t>;
            } else {
                kernel = &LapLapUnstr::NonChasing::laplap_idxvar_shared<value_t>;
            }
        }
        (*kernel)<<<numthreads, numblocks, smem>>>(
            neighborships,
            z_stride,
            offs,
            this->inner_size,
            this->input->data,
            this->output->data
        );
    } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        auto kernel = &LapLapUnstr::Chasing::laplap_idxvar_kloop_sliced<value_t>;
        if(!this->pointer_chasing) {
            kernel = &LapLapUnstr::NonChasing::laplap_idxvar_kloop_sliced<value_t>;
        }
        (*kernel)<<<numthreads, numblocks>>>(
            neighborships,
            z_stride,
            offs,
            this->k_per_thread,
            this->inner_size,
            this->input->data,
            this->output->data
        );
    } else {
        auto kernel = &LapLapUnstr::NonChasing::lap<value_t>;
        if(this->pointer_chasing) {
            kernel = &LapLapUnstr::Chasing::lap<value_t>;
        }
        (*kernel)<<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            z_stride,
            this->input->index(coord3(-1, -1, 0)),
            this->size - 2*coord3(1, 1, 0),
            this->input->data,
            this->intermediate->data
        );
        (*kernel)<<<this->numblocks(), this->numthreads()>>>(
            neighborships,
            z_stride,
            offs,
            this->inner_size,
            this->intermediate->data,
            this->output->data
        );
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
void LapLapUnstrBenchmark<value_t>::setup() {
    coord3 halo2(2, 2, 0);
    int neighbor_store_depth = (this->pointer_chasing ? 1 : 2);
    CudaUnstructuredGrid3D<value_t> *input = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, halo2, CudaUnstructuredGrid3D<value_t>::rowmajor, neighbor_store_depth);
    this->input = input;
    this->output = CudaUnstructuredGrid3D<value_t>::clone(*input);
    if(this->variant == LapLapUnstr::unfused) {
        this->intermediate = CudaUnstructuredGrid3D<value_t>::clone(*input);
    }
    if(this->variant == LapLapUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->LapLapBaseBenchmark<value_t>::setup();
}

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapUnstr::unfused) {
        domain = this->size - 2*coord3(1, 1, 0);
    } else if(this->variant == LapLapUnstr::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->LapLapBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapUnstr::unfused) {
        domain = this->size - 2*coord3(1, 1, 0);
    } else if(this->variant == LapLapUnstr::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numblocks = this->LapLapBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}


template<typename value_t>
void LapLapUnstrBenchmark<value_t>::parse_args() {
    for(int i = 0; i < this->argc; i++) {
        std::string arg = std::string(this->argv[i]);
        if(arg == "--chase" || arg == "-c") {
            this->pointer_chasing = true;
        } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
            sscanf(this->argv[0], "%d", &this->k_per_thread);
        }else {
            this->Benchmark::parse_args();
        }
    }
}

#endif