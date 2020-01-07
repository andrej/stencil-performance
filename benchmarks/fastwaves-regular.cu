#ifndef FASTWAVES_REGULAR_H
#define FASTWAVES_REGULAR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-regular.cu"

namespace FastWavesRegularBenchmarkNamespace {

    enum Variant { unfused, naive, idxvar, idxvar_kloop };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define NEIGHBOR(x, y, z, x_, y_, z_) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, x, y, z, x_, y_, z_)
    #define NEIGHBOR_OF_INDEX(idx, x, y, z) GRID_REGULAR_NEIGHBOR_OF_INDEX(y_stride, z_stride, idx, x, y, z)
    #define NEXT_Z_NEIGHBOR_OF_INDEX(idx) GRID_REGULAR_NEIGHBOR_OF_INDEX(y_stride, z_stride, idx, 0, 0, 1)
    #define K_STEP k*z_stride

    #include "kernels/fastwaves-unfused.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
    #undef NEIGHBOR_OF_INDEX
    #undef NEXT_Z_NEIGHBOR_OF_INDEX
    #undef K_STEP

}

template<typename value_t>
class FastWavesRegularBenchmark : public FastWavesBaseBenchmark<value_t> {

    public:
    
    FastWavesRegularBenchmark(coord3 size, FastWavesRegularBenchmarkNamespace::Variant variant);
    
    FastWavesRegularBenchmarkNamespace::Variant variant;

    void setup();
    void run();
    //dim3 numthreads();
    //dim3 numblocks();

};

template<typename value_t>
FastWavesRegularBenchmark<value_t>::FastWavesRegularBenchmark(coord3 size, FastWavesRegularBenchmarkNamespace::Variant variant) :
FastWavesBaseBenchmark<value_t>(size),
variant(variant) {
    if(this->variant == FastWavesRegularBenchmarkNamespace::unfused) {
        this->name = "fastwaves-regular-unfused";
    }
}

template<typename value_t>
void FastWavesRegularBenchmark<value_t>::setup() {
    this->u_in = new CudaRegularGrid3D<value_t>(this->size);
    this->v_in = new CudaRegularGrid3D<value_t>(this->size);
    this->u_tens = new CudaRegularGrid3D<value_t>(this->size);
    this->v_tens = new CudaRegularGrid3D<value_t>(this->size);
    this->rho = new CudaRegularGrid3D<value_t>(this->size);
    this->ppuv = new CudaRegularGrid3D<value_t>(this->size);
    this->fx = new CudaRegularGrid3D<value_t>(this->size);
    this->wgtfac = new CudaRegularGrid3D<value_t>(this->size);
    this->hhl = new CudaRegularGrid3D<value_t>(this->size);
    this->u_out = new CudaRegularGrid3D<value_t>(this->size);
    this->v_out = new CudaRegularGrid3D<value_t>(this->size);
    if(this->variant == FastWavesRegularBenchmarkNamespace::unfused) {
        this->ppgk = new CudaRegularGrid3D<value_t>(this->size);
        this->ppgc = new CudaRegularGrid3D<value_t>(this->size);
    }
    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values
}

template<typename value_t>
void FastWavesRegularBenchmark<value_t>::run() {
    dim3 blocks = this->numblocks();
    dim3 threads = this->numthreads();
    coord3 strides = (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->u_in))->get_strides();
    // Unfused: Call kernels one by one
    if(this->variant == FastWavesRegularBenchmarkNamespace::unfused) {
        FastWavesRegularBenchmarkNamespace::fastwaves_ppgk<<<blocks, threads>>>(
            this->get_info(),
            this->c_flat_limit,
            strides.y, strides.z,
            this->ppuv->data,
            this->wgtfac->data,
            this->ppgk->data);
        FastWavesRegularBenchmarkNamespace::fastwaves_ppgc<<<blocks, threads>>>(
            this->get_info(),
            strides.y, strides.z,
            this->c_flat_limit,
            this->ppgk->data,
            this->ppgc->data);
        FastWavesRegularBenchmarkNamespace::fastwaves_ppgrad_uv<<<blocks, threads>>>(
            this->get_info(),
            strides.y, strides.z,
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
        cudaDeviceSynchronize();
    }
}

#endif