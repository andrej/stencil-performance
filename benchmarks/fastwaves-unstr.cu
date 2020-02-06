#ifndef FASTWAVES_UNSTR_H
#define FASTWAVES_UNSTR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-unstructured.cu"

namespace FastWavesUnstrBenchmarkNamespace {

    enum Variant { unfused, naive, idxvar, idxvar_kloop, kloop };

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < (z_stride-offs) && k < info.max_coord.z-1)
    #define IS_IN_BOUNDS_P1(i, j, k) (i + j*blockDim.x*gridDim.x < (z_stride-offs) && k >= info.max_coord.z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/fastwaves-unfused.cu"
    #include "kernels/fastwaves-naive.cu"
    #include "kernels/fastwaves-idxvar.cu"
    #include "kernels/fastwaves-idxvar-kloop.cu"
    #include "kernels/fastwaves-kloop.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef IS_IN_BOUNDS_P1
    #undef NEIGHBOR
    #undef NEXT_Z_NEIGHBOR
    #undef K_STEP

}

template<typename value_t>
class FastWavesUnstrBenchmark : public FastWavesBaseBenchmark<value_t> {

    public:
    
    FastWavesUnstrBenchmark(coord3 size, FastWavesUnstrBenchmarkNamespace::Variant variant);
    
    FastWavesUnstrBenchmarkNamespace::Variant variant;

    void setup();
    void run();
    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());

    int *neighborships;
    int z_stride;
    int offs;

};

template<typename value_t>
FastWavesUnstrBenchmark<value_t>::FastWavesUnstrBenchmark(coord3 size, FastWavesUnstrBenchmarkNamespace::Variant variant) :
FastWavesBaseBenchmark<value_t>(size),
variant(variant) {
    if(this->variant == FastWavesUnstrBenchmarkNamespace::unfused) {
        this->name = "fastwaves-unstr-unfused";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::naive) {
        this->name = "fastwaves-unstr-naive";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
        this->name = "fastwaves-unstr-idxvar";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        this->name = "fastwaves-unstr-idxvar-kloop";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        this->name = "fastwaves-unstr-kloop";
    }
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::setup() {
    CudaUnstructuredGrid3D<value_t> *u_in = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo);
    this->u_in = u_in;
    this->v_in = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->u_tens = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->v_tens = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->rho = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->ppuv = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->fx = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->wgtfac = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->hhl = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->u_out = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    this->v_out = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    if(this->variant == FastWavesUnstrBenchmarkNamespace::unfused) {
        this->ppgk = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->ppgc = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
    }
    this->neighborships = u_in->neighborships;
    this->z_stride = u_in->z_stride();
    this->offs =  u_in->index(coord3(0, 0, 0)),
    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::run() {
    dim3 blocks = this->numblocks();
    dim3 threads = this->numthreads();
    // Unfused: Call kernels one by one
    if(this->variant == FastWavesUnstrBenchmarkNamespace::unfused) {
        FastWavesUnstrBenchmarkNamespace::fastwaves_ppgk<<<blocks, threads>>>(
            this->get_info(),
            this->c_flat_limit,
            this->neighborships, this->z_stride, this->offs,
            this->ppuv->data,
            this->wgtfac->data,
            this->ppgk->data);
        FastWavesUnstrBenchmarkNamespace::fastwaves_ppgc<<<blocks, threads>>>(
            this->get_info(),
            this->neighborships, this->z_stride, this->offs,
            this->c_flat_limit,
            this->ppgk->data,
            this->ppgc->data);
        FastWavesUnstrBenchmarkNamespace::fastwaves_ppgrad_uv<<<blocks, threads>>>(
            this->get_info(),
            this->neighborships, this->z_stride, this->offs,
            this->ppuv->data,
            this->ppgc->data,
            this->hhl->data,
            this->v_in->data,
            this->u_in->data,
            this->v_tens->data,
            this->u_tens->data,
            this->rho->data,
            this->fx->data,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->u_out->data,
            this->v_out->data);
    } else {
        auto kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_naive<value_t>;
        if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
            kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar<value_t>;
        } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
            kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_kloop<value_t>;
        } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
            kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_kloop<value_t>;
        }
        (*kernel)<<<blocks, threads>>>(
            this->get_info(),
            this->neighborships, this->z_stride, this->offs,
            this->ppuv->data,
            this->wgtfac->data,
            this->hhl->data,
            this->v_in->data,
            this->u_in->data,
            this->v_tens->data,
            this->u_tens->data,
            this->rho->data,
            this->fx->data,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->u_out->data,
            this->v_out->data);
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant != FastWavesUnstrBenchmarkNamespace::unfused) {
        domain.z -= 1;
    }
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop
        || this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->FastWavesBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    if(this->variant != FastWavesUnstrBenchmarkNamespace::unfused) {
        domain = this->inner_size;
        domain.z -= 1;
    }
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop
       || this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numblocks = this->FastWavesBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

#endif