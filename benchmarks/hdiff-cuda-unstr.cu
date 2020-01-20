#ifndef HDIFF_CUDA_UNSTR_H
#define HDIFF_CUDA_UNSTR_H
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-cuda-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstr {

    /** Variants of this benchmark. */
    enum Variant { naive, idxvar, idxvar_kloop, idxvar_shared };

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < (z_stride-offs) && k < info.max_coord.z)

    namespace NonChasing {
        #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
        #include "kernels/hdiff-naive.cu"
        #undef NEIGHBOR
        #undef DOUBLE_NEIGHBOR

        #define NEIGHBOR(idx, x, y, z) GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, idx, x, y)
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
        #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
        #define K_STEP k*z_stride
        #include "kernels/hdiff-idxvar.cu"
        #include "kernels/hdiff-idxvar-kloop.cu"
        #include "kernels/hdiff-idxvar-shared.cu"
        #undef NEIGHBOR
        #undef DOUBLE_NEIGHBOR
        #undef NEXT_Z_NEIGHBOR
        #undef K_STEP
    }

    namespace Chasing {
        #define CHASING
        
        #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
        #include "kernels/hdiff-naive.cu"
        #undef NEIGHBOR
        #undef DOUBLE_NEIGHBOR

        #define NEIGHBOR(idx, x, y, z) GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, idx, x, y)
        #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
        #define K_STEP k*z_stride
        #include "kernels/hdiff-idxvar.cu"
        #include "kernels/hdiff-idxvar-kloop.cu"
        #include "kernels/hdiff-idxvar-shared.cu"

        #undef NEIGHBOR
        #undef CHASING
        #undef NEXT_Z_NEIGHBOR
        #undef K_STEP
    }

    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS

};

/** Cuda implementation of different variants of the horizontal diffusion
 * kernel, both for structured and unstructured grid variants.
 *
 * For the available variants, see the HdiffCuda::Variant enum. */
template<typename value_t>
class HdiffCudaUnstrBenchmark : public HdiffCudaBaseBenchmark<value_t> {

    public:

    HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant=HdiffCudaUnstr::idxvar);
    
    HdiffCudaUnstr::Variant variant;

    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    virtual dim3 numblocks(coord3 domain=coord3());
    virtual dim3 numthreads(coord3 domain=coord3());
    virtual void parse_args();

    bool pointer_chasing = true;
    int *neighborships;
    int z_stride;
    int offs;

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaUnstrBenchmark<value_t>::HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant) :
HdiffCudaBaseBenchmark<value_t>(size) {
    if(variant == HdiffCudaUnstr::naive) {
        this->name = "hdiff-unstr-naive";
    } else if(variant == HdiffCudaUnstr::idxvar) {
        this->name = "hdiff-unstr-idxvar";
    } else if(variant == HdiffCudaUnstr::idxvar_kloop) {
        this->name = "hdiff-unstr-idxvar-kloop";
    } else if(variant == HdiffCudaUnstr::idxvar_shared) {
        this->name = "hdiff-unstr-idxvar-shared";
    }
    this->error = false;
    this->variant = variant;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::run() {
    auto kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar<value_t>;
    if(this->pointer_chasing) {
        kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar<value_t>;
    }
    int smem = 0;
    if(this->variant == HdiffCudaUnstr::naive) {
        if(this->pointer_chasing) {
            kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_naive<value_t>;
        } else {
            kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_naive<value_t>;
        }
    } else if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        if(this->pointer_chasing) {
            kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar_kloop<value_t>;
        } else {
            kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar_kloop<value_t>;
        }
    } else if(this->variant == HdiffCudaUnstr::idxvar_shared) {
        if(this->pointer_chasing) {
            kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar_shared<value_t>;
        } else {
            kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar_shared<value_t>;
        }
        dim3 numthreads = this->numthreads();
        smem = numthreads.x*numthreads.y*12*sizeof(int);
    }
    (*kernel_fun)<<<this->numblocks(), this->numthreads(), smem>>>(
        this->get_info(),
        this->neighborships, this->z_stride, this->offs,
        this->input->data,
        this->output->data,
        this->coeff->data
    );
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    // For the vriants that use a k-loop inside the kernel, we only need one block in the k-direction
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numblocks = this->Benchmark::numblocks(domain);
    return numblocks;
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    // Variants with a k-loop: only one thread in the k-direction
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->Benchmark::numthreads(domain);
    return numthreads;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::setup() {
    int neighbor_store_depth = (this->pointer_chasing ? 1 : 2);
    CudaUnstructuredGrid3D<value_t> *input = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo, UnstructuredGrid3D<value_t>::rowmajor, neighbor_store_depth);
    this->input = input;
    this->output = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->coeff = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->lap = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->flx = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->fly = CudaUnstructuredGrid3D<value_t>::clone(*input);
    if(this->variant == HdiffCudaUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->neighborships = input->neighborships;
    this->z_stride = input->z_stride();
    this->offs = input->index(coord3(0, 0, 0));
    this->HdiffCudaBaseBenchmark<value_t>::setup();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::teardown() {
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffCudaBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffCudaBaseBenchmark<value_t>::post();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::parse_args() {
    for(int i = 0; i < this->argc; i++) {
        std::string arg = std::string(this->argv[i]);
        if(arg == "--no-chase" || arg == "-c") {
            this->pointer_chasing = false;
        } else {
            this->Benchmark::parse_args();
        }
    }
    if(!this->pointer_chasing) {
        this->name.append("-no-chase");
    }
}

#endif