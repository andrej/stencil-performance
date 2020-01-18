#ifndef FASTWAVES_REGULAR_H
#define FASTWAVES_REGULAR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-regular.cu"

namespace FastWavesRegularBenchmarkNamespace {

    enum Variant { unfused, naive, idxvar, idxvar_kloop, kloop };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define IS_IN_BOUNDS(i, j, k) (i < info.max_coord.x && j < info.max_coord.y && k < info.max_coord.z-1)
    #define IS_IN_BOUNDS_P1(i, j, k) (i < info.max_coord.x + 1 && j < info.max_coord.y + 1 && k >= info.max_coord.z)
    #define NEIGHBOR(idx, x, y, z) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, idx, x, y, z)
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
class FastWavesRegularBenchmark : public FastWavesBaseBenchmark<value_t> {

    public:
    
    FastWavesRegularBenchmark(coord3 size, FastWavesRegularBenchmarkNamespace::Variant variant);
    
    FastWavesRegularBenchmarkNamespace::Variant variant;

    void setup();
    void run();
    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    
    coord3 strides;

};

template<typename value_t>
FastWavesRegularBenchmark<value_t>::FastWavesRegularBenchmark(coord3 size, FastWavesRegularBenchmarkNamespace::Variant variant) :
FastWavesBaseBenchmark<value_t>(size),
variant(variant) {
    if(this->variant == FastWavesRegularBenchmarkNamespace::unfused) {
        this->name = "fastwaves-regular-unfused";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::naive) {
        this->name = "fastwaves-regular-naive";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar) {
        this->name = "fastwaves-regular-idxvar";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        this->name = "fastwaves-regular-idxvar-kloop";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::kloop) {
        this->name = "fastwaves-regular-kloop";
    }
}

template<typename value_t>
void FastWavesRegularBenchmark<value_t>::setup() {
    this->u_in = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->v_in = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->u_tens = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->v_tens = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->rho = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->ppuv = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->fx = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->wgtfac = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->hhl = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->u_out = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->v_out = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    if(this->variant == FastWavesRegularBenchmarkNamespace::unfused) {
        this->ppgk = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
        this->ppgc = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    }
    this->strides = (dynamic_cast<RegularGrid3D<value_t> *>(this->u_in))->get_strides();
    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values
}

template<typename value_t>
void FastWavesRegularBenchmark<value_t>::run() {
    dim3 blocks = this->numblocks();
    dim3 threads = this->numthreads();
    // Unfused: Call kernels one by one
    if(this->variant == FastWavesRegularBenchmarkNamespace::unfused) {
        FastWavesRegularBenchmarkNamespace::fastwaves_ppgk<<<blocks, threads>>>(
            this->get_info(),
            this->c_flat_limit,
            this->strides.y, this->strides.z,
            this->ppuv->pointer(coord3(0, 0, 0)),
            this->wgtfac->pointer(coord3(0, 0, 0)),
            this->ppgk->pointer(coord3(0, 0, 0)));
        FastWavesRegularBenchmarkNamespace::fastwaves_ppgc<<<blocks, threads>>>(
            this->get_info(),
            this->strides.y, this->strides.z,
            this->c_flat_limit,
            this->ppgk->pointer(coord3(0, 0, 0)),
            this->ppgc->pointer(coord3(0, 0, 0)));
        FastWavesRegularBenchmarkNamespace::fastwaves_ppgrad_uv<<<blocks, threads>>>(
            this->get_info(),
            this->strides.y, this->strides.z,
            this->ppuv->pointer(coord3(0, 0, 0)),
            this->ppgc->pointer(coord3(0, 0, 0)),
            this->hhl->pointer(coord3(0, 0, 0)),
            this->v_in->pointer(coord3(0, 0, 0)),
            this->u_in->pointer(coord3(0, 0, 0)),
            this->v_tens->pointer(coord3(0, 0, 0)),
            this->u_tens->pointer(coord3(0, 0, 0)),
            this->rho->pointer(coord3(0, 0, 0)),
            this->fx->pointer(coord3(0, 0, 0)),
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->u_out->pointer(coord3(0, 0, 0)),
            this->v_out->pointer(coord3(0, 0, 0)));
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::naive
              || this->variant == FastWavesRegularBenchmarkNamespace::idxvar
              || this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop
              || this->variant == FastWavesRegularBenchmarkNamespace::kloop) {
        auto kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_naive<value_t>;
        if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar) {
            kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_idxvar<value_t>;
        } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
            kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_idxvar_kloop<value_t>;
        } else if(this->variant == FastWavesRegularBenchmarkNamespace::kloop) {
            kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_kloop<value_t>;
        }
        (*kernel)<<<blocks, threads>>>(
            this->get_info(),
            this->strides.y, this->strides.z,
            this->ppuv->pointer(coord3(0, 0, 0)),
            this->wgtfac->pointer(coord3(0, 0, 0)),
            this->hhl->pointer(coord3(0, 0, 0)),
            this->v_in->pointer(coord3(0, 0, 0)),
            this->u_in->pointer(coord3(0, 0, 0)),
            this->v_tens->pointer(coord3(0, 0, 0)),
            this->u_tens->pointer(coord3(0, 0, 0)),
            this->rho->pointer(coord3(0, 0, 0)),
            this->fx->pointer(coord3(0, 0, 0)),
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->u_out->pointer(coord3(0, 0, 0)),
            this->v_out->pointer(coord3(0, 0, 0)));
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 FastWavesRegularBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant != FastWavesRegularBenchmarkNamespace::unfused) {
        domain.z -= 1;
    }
    if(this->variant == FastWavesRegularBenchmarkNamespace::kloop ||
       this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->FastWavesBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 FastWavesRegularBenchmark<value_t>::numblocks(coord3 domain) {
    if(this->variant != FastWavesRegularBenchmarkNamespace::unfused) {
        domain = this->inner_size;
        domain.z -= 1;
    }
    if(this->variant == FastWavesRegularBenchmarkNamespace::kloop ||
       this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numblocks = this->FastWavesBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

#endif