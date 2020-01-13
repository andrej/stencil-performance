#ifndef FASTWAVES_UNSTR_H
#define FASTWAVES_UNSTR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-unstructured.cu"

namespace FastWavesUnstrBenchmarkNamespace {

    enum Variant { unfused, naive, idxvar, kloop };

    #define GRID_ARGS const int * __restrict__ neighbor_data, const int y_stride, const int z_stride, 
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + (z_)*blockDim.x*gridDim.x*blockDim.y*gridDim.y
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/fastwaves-unfused.cu"
    #include "kernels/fastwaves-naive.cu"
    #include "kernels/fastwaves-idxvar.cu"
    #include "kernels/fastwaves-kloop.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
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
    dim3 numthreads();
    dim3 numblocks();

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
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        this->name = "fastwaves-unstr-kloop";
    }
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::setup() {
    this->u_in = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo);
    int *neighbor_data = (dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->u_in))->neighbor_data;
    this->v_in = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->u_tens = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->v_tens = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->rho = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->ppuv = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->fx = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->wgtfac = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->hhl = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->u_out = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    this->v_out = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    if(this->variant == FastWavesUnstrBenchmarkNamespace::unfused) {
        this->ppgk = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
        this->ppgc = new CudaUnstructuredGrid3D<value_t>(this->inner_size, neighbor_data);
    }
    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::run() {
    dim3 blocks = this->numblocks();
    dim3 threads = this->numthreads();
    CudaUnstructuredGrid3D<value_t> *unstr_u_in = (dynamic_cast<CudaUnstructuredGrid3D<value_t>*>(this->u_in));
    coord3 strides = coord3(1, unstr_u_in->dimensions.x, unstr_u_in->dimensions.x*unstr_u_in->dimensions.y);
    int *neighbor_data = unstr_u_in->neighbor_data;
    // Unfused: Call kernels one by one
    if(this->variant == FastWavesUnstrBenchmarkNamespace::unfused) {
        FastWavesUnstrBenchmarkNamespace::fastwaves_ppgk<<<blocks, threads>>>(
            this->get_info(),
            this->c_flat_limit,
            neighbor_data, strides.z,
            this->ppuv->pointer(coord(0, 0, 0)),
            this->wgtfac->pointer(coord(0, 0, 0)),
            this->ppgk->pointer(coord(0, 0, 0)));
        FastWavesUnstrBenchmarkNamespace::fastwaves_ppgc<<<blocks, threads>>>(
            this->get_info(),
            neighbor_data, strides.z,
            this->c_flat_limit,
            this->ppgk->pointer(coord(0, 0, 0)),
            this->ppgc->pointer(coord(0, 0, 0)));
        FastWavesUnstrBenchmarkNamespace::fastwaves_ppgrad_uv<<<blocks, threads>>>(
            this->get_info(),
            neighbor_data, strides.z,
            this->ppuv->pointer(coord(0, 0, 0)),
            this->ppgc->pointer(coord(0, 0, 0)),
            this->hhl->pointer(coord(0, 0, 0)),
            this->v_in->pointer(coord(0, 0, 0)),
            this->u_in->pointer(coord(0, 0, 0)),
            this->v_tens->pointer(coord(0, 0, 0)),
            this->u_tens->pointer(coord(0, 0, 0)),
            this->rho->pointer(coord(0, 0, 0)),
            this->fx->pointer(coord(0, 0, 0)),
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->u_out->pointer(coord(0, 0, 0)),
            this->v_out->pointer(coord(0, 0, 0)));
    } else {
        auto kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_naive<value_t>;
        if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
            kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar<value_t>;
        } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
            kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_kloop<value_t>;
        }
        (*kernel)<<<blocks, threads>>>(
            this->get_info(),
            neighbor_data, strides.z,
            this->ppuv->pointer(coord(0, 0, 0)),
            this->wgtfac->pointer(coord(0, 0, 0)),
            this->hhl->pointer(coord(0, 0, 0)),
            this->v_in->pointer(coord(0, 0, 0)),
            this->u_in->pointer(coord(0, 0, 0)),
            this->v_tens->pointer(coord(0, 0, 0)),
            this->u_tens->pointer(coord(0, 0, 0)),
            this->rho->pointer(coord(0, 0, 0)),
            this->fx->pointer(coord(0, 0, 0)),
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->u_out->pointer(coord(0, 0, 0)),
            this->v_out->pointer(coord(0, 0, 0)));
    }
    cudaDeviceSynchronize();
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->FastWavesBaseBenchmark<value_t>::numthreads();
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        numthreads.z = 1;
    }
    return numthreads;
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->FastWavesBaseBenchmark<value_t>::numblocks();
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        numblocks.z = 1;
    }
    return numblocks;
}

#endif