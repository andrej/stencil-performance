#ifndef FASTWAVES_REGULAR_H
#define FASTWAVES_REGULAR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-regular.cu"

namespace FastWavesRegularBenchmarkNamespace {

    enum Variant { naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared, kloop };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define IS_IN_BOUNDS(i, j, k) (i < max_coord.x && j < max_coord.y && k < max_coord.z-1)
    #define NOT_IN_BOUNDS(i, j, k) (i >= max_coord.x || j >= max_coord.y || k >= max_coord.z-1)
    #define NEIGHBOR(idx, x, y, z) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, idx, x, y, z)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/fastwaves-naive.cu"
    #include "kernels/fastwaves-idxvar.cu"
    #include "kernels/fastwaves-idxvar-kloop.cu"
    #include "kernels/fastwaves-idxvar-kloop-sliced.cu"
    #include "kernels/fastwaves-idxvar-shared.cu"
    #include "kernels/fastwaves-kloop.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef NOT_IN_BOUNDS
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
    void parse_args();
    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    
    coord3 strides;

    int k_per_thread = 8;
    int smem = 0;

    void (*kernel)(const coord3, const int, const int, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const double, const double, const int, value_t *, value_t *);

};

template<typename value_t>
FastWavesRegularBenchmark<value_t>::FastWavesRegularBenchmark(coord3 size, FastWavesRegularBenchmarkNamespace::Variant variant) :
FastWavesBaseBenchmark<value_t>(size),
variant(variant) {
    if(this->variant == FastWavesRegularBenchmarkNamespace::naive) {
        this->name = "fastwaves-regular-naive";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar) {
        this->name = "fastwaves-regular-idxvar";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        this->name = "fastwaves-regular-idxvar-kloop";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop_sliced) {
        this->name = "fastwaves-regular-idxvar-kloop-sliced";
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_shared) {
        this->name = "fastwaves-regular-idxvar-shared";
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
    this->strides = (dynamic_cast<RegularGrid3D<value_t> *>(this->u_in))->get_strides();
    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values

    this->ptr_ppuv = this->ppuv->pointer(coord3(0, 0, 0));
    this->ptr_wgtfac = this->wgtfac->pointer(coord3(0, 0, 0));
    this->ptr_hhl = this->hhl->pointer(coord3(0, 0, 0));
    this->ptr_v_in = this->v_in->pointer(coord3(0, 0, 0));
    this->ptr_u_in = this->u_in->pointer(coord3(0, 0, 0));
    this->ptr_v_tens = this->v_tens->pointer(coord3(0, 0, 0));
    this->ptr_u_tens = this->u_tens->pointer(coord3(0, 0, 0));
    this->ptr_rho = this->rho->pointer(coord3(0, 0, 0));
    this->ptr_fx = this->fx->pointer(coord3(0, 0, 0));
    this->ptr_u_out = this->u_out->pointer(coord3(0, 0, 0));
    this->ptr_v_out = this->v_out->pointer(coord3(0, 0, 0));

    this->blocks = this->numblocks();
    this->threads = this->numthreads();
    this->kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_naive<value_t>;
    if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar) {
        this->kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_idxvar<value_t>;
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        this->kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_idxvar_kloop<value_t>;
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_shared) {
        this->kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_idxvar_shared<value_t>;
        dim3 threads = this->threads;
        this->smem = threads.x*threads.y*FASTWAVES_IDXVAR_SHARED_SMEM_SZ_PER_THREAD*sizeof(int);
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::kloop) {
        this->kernel = &FastWavesRegularBenchmarkNamespace::fastwaves_kloop<value_t>;
    }
}

template<typename value_t>
void FastWavesRegularBenchmark<value_t>::run() {
    if(this->variant != FastWavesRegularBenchmarkNamespace::idxvar_kloop_sliced) {
        (*this->kernel)<<<this->blocks, this->threads, this->smem>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->ptr_ppuv,
            this->ptr_wgtfac,
            this->ptr_hhl,
            this->ptr_v_in,
            this->ptr_u_in,
            this->ptr_v_tens,
            this->ptr_u_tens,
            this->ptr_rho,
            this->ptr_fx,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->ptr_u_out,
            this->ptr_v_out
        );
    } else {
        FastWavesRegularBenchmarkNamespace::fastwaves_idxvar_kloop_sliced<<<this->blocks, this->threads>>>(
            this->k_per_thread,
            this->inner_size,
            this->strides.y, this->strides.z,
            this->ptr_ppuv,
            this->ptr_wgtfac,
            this->ptr_hhl,
            this->ptr_v_in,
            this->ptr_u_in,
            this->ptr_v_tens,
            this->ptr_u_tens,
            this->ptr_rho,
            this->ptr_fx,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->ptr_u_out,
            this->ptr_v_out);
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 FastWavesRegularBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    domain.z -= 1;
    if(this->variant == FastWavesRegularBenchmarkNamespace::kloop ||
       this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numthreads = this->FastWavesBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 FastWavesRegularBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    domain.z -= 1;
    if(this->variant == FastWavesRegularBenchmarkNamespace::kloop ||
       this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numblocks = this->FastWavesBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

template<typename value_t>
void FastWavesRegularBenchmark<value_t>::parse_args() {
    if(this->argc > 0 && this->variant == FastWavesRegularBenchmarkNamespace::idxvar_kloop_sliced) {
        sscanf(this->argv[0], "%d", &this->k_per_thread);
    } else {
        this->Benchmark::parse_args();
    }
}

#endif