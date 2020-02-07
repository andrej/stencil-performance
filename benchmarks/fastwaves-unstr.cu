#ifndef FASTWAVES_UNSTR_H
#define FASTWAVES_UNSTR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-unstructured.cu"

namespace FastWavesUnstrBenchmarkNamespace {

    enum Variant { naive, idxvar, idxvar_kloop, kloop };

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs, const int xysize,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < xysize && k < max_coord.z-1)
    #define NOT_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x >= xysize || k >= max_coord.z-1)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/fastwaves-naive.cu"
    #include "kernels/fastwaves-idxvar.cu"
    #include "kernels/fastwaves-idxvar-kloop.cu"
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
class FastWavesUnstrBenchmark : public FastWavesBaseBenchmark<value_t> {

    public:
    
    FastWavesUnstrBenchmark(coord3 size, FastWavesUnstrBenchmarkNamespace::Variant variant);
    
    FastWavesUnstrBenchmarkNamespace::Variant variant;

    void setup();
    void run();
    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    void parse_args();

    int *neighborships;
    int z_stride;
    int offs;
    int xysize;

    typename UnstructuredGrid3D<value_t>::layout_t layout = UnstructuredGrid3D<value_t>::rowmajor;

    void (*kernel)(const coord3, const int *, const int, const int, const int, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const double, const double, const int, value_t *, value_t *);

};

template<typename value_t>
FastWavesUnstrBenchmark<value_t>::FastWavesUnstrBenchmark(coord3 size, FastWavesUnstrBenchmarkNamespace::Variant variant) :
FastWavesBaseBenchmark<value_t>(size),
variant(variant) {
    if(this->variant == FastWavesUnstrBenchmarkNamespace::naive) {
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
    CudaUnstructuredGrid3D<value_t> *u_in = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo, this->layout);
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

    this->neighborships = u_in->neighborships;
    this->z_stride = u_in->z_stride();
    this->offs = u_in->index(coord3(0, 0, 0)),
    this->xysize = u_in->dimensions.x*u_in->dimensions.y;
    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values

    this->ptr_ppuv = this->ppuv->data;
    this->ptr_wgtfac = this->wgtfac->data;
    this->ptr_hhl = this->hhl->data;
    this->ptr_v_in = this->v_in->data;
    this->ptr_u_in = this->u_in->data;
    this->ptr_v_tens = this->v_tens->data;
    this->ptr_u_tens = this->u_tens->data;
    this->ptr_rho = this->rho->data;
    this->ptr_fx = this->fx->data;
    this->ptr_u_out = this->u_out->data;
    this->ptr_v_out = this->v_out->data;

    this->blocks = this->numblocks();
    this->threads = this->numthreads();

    this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_naive<value_t>;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar<value_t>;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_kloop<value_t>;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_kloop<value_t>;
    }
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::run() {
    (*this->kernel)<<<this->blocks, this->threads>>>(
        this->inner_size,
        this->neighborships, this->z_stride, this->offs, this->xysize,
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
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    domain.z -= 1;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop
        || this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->FastWavesBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    domain.z -= 1;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop
       || this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numblocks = this->FastWavesBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::parse_args() {
    for(int i = 0; i < this->argc; i++) {
        std::string arg = std::string(this->argv[i]);
        if(arg == "--z-curves" || arg == "-z") {
            this->layout = CudaUnstructuredGrid3D<value_t>::zcurve;
        } else if(arg == "--random" || arg == "-r") {
            this->layout = CudaUnstructuredGrid3D<value_t>::random;
        }
    }
    if(this->layout == CudaUnstructuredGrid3D<value_t>::zcurve) {
        this->name.append("-z-curves");
    }
    if(this->layout == CudaUnstructuredGrid3D<value_t>::random) {
        this->name.append("-random");
    }
}

#endif