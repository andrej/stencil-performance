#ifndef LAPLAP_REGULAR_BENCHMARK_H
#define LAPLAP_REGULAR_BENCHMARK_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/laplap-base.cu"
#include "grids/cuda-regular.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the regular grid macros. */
namespace LapLapRegular {

    enum Variant { naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared, shared };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define PROTO(x)
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define IS_IN_BOUNDS(i, j, k) (i < max_coord.x && j < max_coord.y && k < max_coord.z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
    #define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
    #define K_STEP k*z_stride

    #include "kernels/laplap-naive.cu"
    #include "kernels/laplap-idxvar.cu"
    #include "kernels/laplap-idxvar-kloop.cu"
    #include "kernels/laplap-idxvar-kloop-sliced.cu"
    #include "kernels/laplap-idxvar-shared.cu"

    #define SMEM_GRID_ARGS
    #define SMEM_INDEX(_x, _y, _z) GRID_REGULAR_INDEX((int)blockDim.x, (int)blockDim.x*blockDim.y, _x, _y, _z)
    #define SMEM_NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR((int)blockDim.x, (int)blockDim.x*blockDim.y, idx, x_, y_, z_)
    #include "kernels/laplap-shared.cu"

    #undef GRID_ARGS
    #undef PROTO
    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR
    #undef K_STEP
    #undef SMEM_GRID_ARGS
    #undef SMEM_INDEX
    #undef SMEM_NEIGHBOR

};


template<typename value_t>
class LapLapRegularBenchmark : public LapLapBaseBenchmark<value_t> {

    public:

    LapLapRegularBenchmark(coord3 size, LapLapRegular::Variant variant = LapLapRegular::naive);

    LapLapRegular::Variant variant;

    void setup();
    void run();

    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());

    void setup_from_archive(Benchmark::cache_iarchive &ia);
    void store_to_archive(Benchmark::cache_oarchive &oa);
    std::string cache_file_name();

    void parse_args();
    int k_per_thread = 16;

    coord3 strides;

    dim3 threads;
    dim3 blocks;
    value_t *input_ptr;
    value_t *output_ptr;

};

// IMPLEMENTATIONS

template<typename value_t>
LapLapRegularBenchmark<value_t>::LapLapRegularBenchmark(coord3 size, LapLapRegular::Variant variant) :
LapLapBaseBenchmark<value_t>(size),
variant(variant) {
    this->variant = variant;
    if(variant == LapLapRegular::naive) {
        this->name = "laplap-regular-naive";
    } else if(variant == LapLapRegular::idxvar) {
        this->name = "laplap-regular-idxvar";
    } else if(variant == LapLapRegular::idxvar_kloop) {
        this->name = "laplap-regular-idxvar-kloop";
    } else if(variant == LapLapRegular::idxvar_kloop_sliced) {
        this->name = "laplap-regular-idxvar-kloop-sliced";
    } else if(variant == LapLapRegular::idxvar_shared) {
        this->name = "laplap-regular-idxvar-shared";
    } else if(variant == LapLapRegular::shared) {
        this->name = "laplap-regular-shared";
    }
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::setup() {
    coord3 halo2(2, 2, 0);
    if(!this->setup_from_cache()) {
        this->input = CudaRegularGrid3D<value_t>::create(this->inner_size, halo2);
        this->input->fill_random();
        this->output = CudaRegularGrid3D<value_t>::create(this->inner_size, halo2);
        this->store_to_cache();
    }
    if(this->variant == LapLapRegular::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->strides = (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_strides();
    this->LapLapBaseBenchmark<value_t>::setup();
    this->blocks = this->numblocks();
    this->threads = this->numthreads();
    this->input_ptr = this->input->pointer(coord3(0, 0, 0));
    this->output_ptr = this->output->pointer(coord3(0, 0, 0));
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::setup_from_archive(Benchmark::cache_iarchive &ar) {
    CudaRegularGrid3D<value_t> *input = new CudaRegularGrid3D<value_t>();
    CudaRegularGrid3D<value_t> *output = new CudaRegularGrid3D<value_t>();
    ar >> *input;
    ar >> *output;
    this->input = input;
    this->output = output;
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::store_to_archive(Benchmark::cache_oarchive &ar) {
    CudaRegularGrid3D<value_t> *input = dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input);
    CudaRegularGrid3D<value_t> *output = dynamic_cast<CudaRegularGrid3D<value_t> *>(this->output);
    ar << *input;
    ar << *output;
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::run() {
    if(this->variant == LapLapRegular::naive) {
        LapLapRegular::laplap_naive<value_t><<<this->blocks, this->threads>>>(
            this->strides.y, this->strides.z,
            this->inner_size,
            this->input_ptr,
            this->output_ptr
        );
    } else if(this->variant == LapLapRegular::idxvar) {
        LapLapRegular::laplap_idxvar<value_t><<<this->blocks, this->threads>>>(
            this->strides.y, this->strides.z,
            this->inner_size,
            this->input_ptr,
            this->output_ptr
        );
    } else if(this->variant == LapLapRegular::idxvar_kloop) {
        LapLapRegular::laplap_idxvar_kloop<value_t><<<this->blocks, this->threads>>>(
            this->strides.y, this->strides.z,
            this->inner_size,
            this->input_ptr,
            this->output_ptr
        );
    } else if(this->variant == LapLapRegular::idxvar_kloop_sliced) {
        LapLapRegular::laplap_idxvar_kloop_sliced<value_t><<<this->blocks, this->threads>>>(
            this->strides.y, this->strides.z,
            this->k_per_thread,
            this->inner_size,
            this->input_ptr,
            this->output_ptr
        );
    } else if(this->variant == LapLapRegular::idxvar_shared) {
        int smem = this->threads.x*this->threads.y*13*sizeof(int);
        LapLapRegular::laplap_idxvar_shared<value_t><<<this->blocks, this->threads, smem>>>(
            this->strides.y, this->strides.z,
            this->inner_size,
            this->input_ptr,
            this->output_ptr
        );
    } else if(this->variant == LapLapRegular::shared){
        int smem = sizeof(value_t) * this->threads.x * this->threads.y * this->threads.z;
        LapLapRegular::laplap_shared<value_t><<<this->blocks, this->threads, smem>>>(
            this->strides.y, this->strides.z,
            this->inner_size,
            this->input_ptr,
            this->output_ptr
        );
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 LapLapRegularBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapRegular::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->LapLapBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 LapLapRegularBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapRegular::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == LapLapRegular::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numblocks = this->LapLapBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

template<typename value_t>
void LapLapRegularBenchmark<value_t>::parse_args() {
    if(this->argc > 0 && this->variant == LapLapRegular::idxvar_kloop_sliced) {
        sscanf(this->argv[0], "%d", &this->k_per_thread);
    } else {
        this->Benchmark::parse_args();
    }
}

template<typename value_t>
std::string LapLapRegularBenchmark<value_t>::cache_file_name() {
    return "laplap-regular";
}

#endif