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
    enum Variant { idxvar, idxvar_kloop, idxvar_shared };

    #define GRID_ARGS const int *neighbor_data, const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_UNSTR_INDEX(y_stride, z_stride, x, y, z)
    #define NEIGHBOR_OF_INDEX(idx, x, y, z) GRID_UNSTR_2D_NEIGHBOR_OF_INDEX(neighbor_data, z_stride, idx, x, y)
    #define NEXT_Z_NEIGHBOR_OF_INDEX(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/hdiff-idxvar.cu"
    #include "kernels/hdiff-idxvar-kloop.cu"
    #include "kernels/hdiff-idxvar-shared.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
    #undef NEIGHBOR_OF_INDEX
    #undef NEXT_Z_NEIGHBOR_OF_INDEX
    #undef K_STEP

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
    virtual dim3 numblocks();
    virtual dim3 numthreads();

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaUnstrBenchmark<value_t>::HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant) :
HdiffCudaBaseBenchmark<value_t>(size) {
    if(variant == HdiffCudaUnstr::idxvar) {
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
    auto kernel_fun = &HdiffCudaUnstr::hdiff_idxvar<value_t>;
    int smem = 0;
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        kernel_fun = &HdiffCudaUnstr::hdiff_idxvar_kloop<value_t>;
    } else if(this->variant == HdiffCudaUnstr::idxvar_shared) {
        kernel_fun = &HdiffCudaUnstr::hdiff_idxvar_shared<value_t>;
        dim3 numthreads = this->numthreads();
        smem = numthreads.x*numthreads.y*12*sizeof(int);
    }
    CudaUnstructuredGrid3D<value_t> *unstr_input = dynamic_cast<CudaUnstructuredGrid3D<value_t>*>(this->input);
    (*kernel_fun)<<<this->numblocks(), this->numthreads(), smem>>>(
        this->get_info(),
        unstr_input->neighbor_data,
        unstr_input->dimensions.x,
        unstr_input->dimensions.x*unstr_input->dimensions.y,
        this->input->data,
        this->output->data,
        this->coeff->data
    );
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->Benchmark::numblocks();
    // For the vriants that use a k-loop inside the kernel, we only need one block in the k-direction
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        numblocks = dim3(numblocks.x, numblocks.y, 1);
    }
    return numblocks;
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->Benchmark::numthreads();
    // Variants with a k-loop: only one thread in the k-direction
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        numthreads = dim3(numthreads.x, numthreads.y, 1);
    }
    return numthreads;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::setup() {
    this->input = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    int *neighbor_data = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input)->neighbor_data;
    this->output = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->coeff = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->lap = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->flx = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    this->fly = new CudaUnstructuredGrid3D<value_t>(this->size, neighbor_data);
    if(this->variant == HdiffCudaUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
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

#endif