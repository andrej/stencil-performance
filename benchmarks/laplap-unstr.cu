#ifndef LAPLAP_UNSTR_BENCHMARK_H
#define LAPLAP_UNSTR_BENCHMARK_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/laplap-base.cu"
#include "grids/cuda-unstructured.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the unstructured grid macros. */
namespace LapLapUnstr {

    enum Variant { naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared };

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < (z_stride-offs) && k < max_coord.z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
    #define K_STEP k*z_stride

    namespace NonChasing {
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1)+(x2), (y1)+(y2), (z1)+(z2))
        
        #include "kernels/laplap-naive.cu"
        #include "kernels/laplap-idxvar.cu"
        #include "kernels/laplap-idxvar-kloop.cu"
        #include "kernels/laplap-idxvar-kloop-sliced.cu"
        #include "kernels/laplap-idxvar-shared.cu"

        #undef DOUBLE_NEIGHBOR
    };

    namespace Chasing {
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
        #define CHASING
        
        #include "kernels/laplap-naive.cu"
        #include "kernels/laplap-idxvar.cu"
        #include "kernels/laplap-idxvar-kloop.cu"
        #include "kernels/laplap-idxvar-kloop-sliced.cu"
        #include "kernels/laplap-idxvar-shared.cu"

        #undef DOUBLE_NEIGHBOR
        #undef CHASING
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
    bool pointer_chasing = true;
    
    int z_stride;
    int offs;
    int *neighborships;

    typename UnstructuredGrid3D<value_t>::layout_t layout = UnstructuredGrid3D<value_t>::rowmajor;
    int z_curve_width = 4;

    int smem = 0;
    dim3 blocks;
    dim3 threads;
    value_t *input_ptr;
    value_t *output_ptr;
    void (*kernel_kloop_sliced)(const int *, const int, const int, const int, const coord3, const value_t *, value_t *);
    void (*kernel)(const int *, const int, const int, const coord3, const value_t *, value_t *);

};

// IMPLEMENTATIONS

template<typename value_t>
LapLapUnstrBenchmark<value_t>::LapLapUnstrBenchmark(coord3 size, LapLapUnstr::Variant variant) :
LapLapBaseBenchmark<value_t>(size),
variant(variant) {
    this->variant = variant;
    if(variant == LapLapUnstr::naive) {
        this->name = "laplap-unstr-naive";
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
void LapLapUnstrBenchmark<value_t>::setup() {
    coord3 halo2(2, 2, 0);
    int neighbor_store_depth = (this->pointer_chasing ? 1 : 2);
    CudaUnstructuredGrid3D<value_t> *input = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, halo2, this->layout, neighbor_store_depth, this->z_curve_width);
    this->input = input;
    this->output = CudaUnstructuredGrid3D<value_t>::clone(*input);
    if(this->variant == LapLapUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->neighborships = input->neighborships;
    this->z_stride = input->z_stride();
    this->offs = this->input->index(coord3(0, 0, 0));
    this->LapLapBaseBenchmark<value_t>::setup();

    this->input_ptr = this->input->data;
    this->output_ptr = this->output->data;
    this->threads = this->numthreads();
    this->blocks = this->numblocks();

    smem = 0;
    kernel = &LapLapUnstr::NonChasing::laplap_naive<value_t>;
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
        smem = threads.x*threads.y*13*sizeof(int);
        if(this->pointer_chasing) {
            kernel = &LapLapUnstr::Chasing::laplap_idxvar_shared<value_t>;
        } else {
            kernel = &LapLapUnstr::NonChasing::laplap_idxvar_shared<value_t>;
        }
    }
    kernel_kloop_sliced = &LapLapUnstr::Chasing::laplap_idxvar_kloop_sliced<value_t>;
    if(!this->pointer_chasing) {
        kernel_kloop_sliced = &LapLapUnstr::NonChasing::laplap_idxvar_kloop_sliced<value_t>;
    }
}

template<typename value_t>
void LapLapUnstrBenchmark<value_t>::run() {
    if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        (*kernel_kloop_sliced)<<<blocks, threads>>>(
            neighborships,
            this->z_stride,
            this->offs,
            this->k_per_thread,
            this->inner_size,
            this->input->data,
            this->output->data
        );
    } else {
        (*kernel)<<<blocks, threads, smem>>>(
            neighborships,
            this->z_stride,
            this->offs,
            this->inner_size,
            this->input->data,
            this->output->data
        );
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapUnstr::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->LapLapBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 LapLapUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
if(this->variant == LapLapUnstr::idxvar_kloop) {
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
        if(arg == "--z-curves" || arg == "-z") {
            this->layout = CudaUnstructuredGrid3D<value_t>::zcurve;
        } else if(arg == "--random" || arg == "-r") {
            this->layout = CudaUnstructuredGrid3D<value_t>::random;
        } else if(arg == "--no-chase" || arg == "-c") {
            this->pointer_chasing = false;
        } else if(arg == "--z-curve-width" && this->argc > i+1) {
            sscanf(this->argv[i+1], "%d", &this->z_curve_width);
            ++i;
        } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
            sscanf(this->argv[i], "%d", &this->k_per_thread);
        } else {
            this->Benchmark::parse_args();
        }
    }
    if(!this->pointer_chasing) {
        this->name.append("-no-chase");
    }
    if(this->layout == CudaUnstructuredGrid3D<value_t>::zcurve) {
        this->name.append("-z-curves");
    }
    if(this->layout == CudaUnstructuredGrid3D<value_t>::random) {
        this->name.append("-random");
    }
}

#endif