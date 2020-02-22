#ifndef HDIFF_CUDA_REGULAR_H
#define HDIFF_CUDA_REGULAR_H
#include <stdio.h>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-cuda-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the regular grid macros. */
namespace HdiffCudaRegular {

    enum Variant { naive, iloop, jloop, kloop, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared, shared };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define IS_IN_BOUNDS(i, j, k) (i < max_coord.x && j < max_coord.y && k < max_coord.z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
    #define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
    #define K_STEP k*z_stride
    #define PROTO(x)

    #include "kernels/hdiff-naive.cu"
    #include "kernels/hdiff-iloop.cu"
    #include "kernels/hdiff-jloop.cu"
    #include "kernels/hdiff-kloop.cu"
    #include "kernels/hdiff-idxvar.cu"
    #include "kernels/hdiff-idxvar-kloop.cu"
    #include "kernels/hdiff-idxvar-kloop-sliced.cu"
    #include "kernels/hdiff-idxvar-shared.cu"

    #define SMEM_GRID_ARGS
    #define SMEM_INDEX(_x, _y, _z) GRID_REGULAR_INDEX((int)blockDim.x, (int)blockDim.x*(int)blockDim.y, _x, _y, _z)
    #define SMEM_NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR((int)blockDim.x, (int)blockDim.x*(int)blockDim.y, idx, x_, y_, z_)

    #include "kernels/hdiff-shared.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR
    #undef Z_NEIGHBOR
    #undef K_STEP
    #undef SMEM_GRID_ARGS
    #undef SMEM_INDEX
    #undef SMEM_NEIGHBOR
    #undef PROTO

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
template<typename value_t>
class HdiffCudaRegularBenchmark : public HdiffCudaBaseBenchmark<value_t> {

    public:

    HdiffCudaRegularBenchmark(coord3 size, HdiffCudaRegular::Variant variant = HdiffCudaRegular::naive);

    HdiffCudaRegular::Variant variant;

    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    virtual bool setup_from_archive(Benchmark::cache_iarchive &ar);
    virtual void store_to_archive(Benchmark::cache_oarchive &ar);
    virtual std::string cache_file_name();

    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    
    // parameter for the jloop/iloop kernel only
    virtual void parse_args();
    int jloop_j_per_thread;
    int iloop_i_per_thread;
    int k_per_thread = 8;

    coord3 strides;

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaRegularBenchmark<value_t>::HdiffCudaRegularBenchmark(coord3 size, HdiffCudaRegular::Variant variant) :
HdiffCudaBaseBenchmark<value_t>(size) {
    this->variant = variant;
    if(variant == HdiffCudaRegular::naive) {
        this->name = "hdiff-regular-naive";
    } else if(variant == HdiffCudaRegular::kloop) {
        this->name = "hdiff-regular-kloop";
    } else if(variant == HdiffCudaRegular::shared) {
        this->name = "hdiff-regular-shared";
    } else if(variant == HdiffCudaRegular::jloop) {
        this->name = "hdiff-regular-jloop";
    } else if(variant == HdiffCudaRegular::iloop) {
        this->name = "hdiff-regular-iloop";
    } else if(variant == HdiffCudaRegular::idxvar) {
        this->name = "hdiff-regular-idxvar";
    } else if(variant == HdiffCudaRegular::idxvar_kloop) {
        this->name = "hdiff-regular-idxvar-kloop";
    } else if(variant == HdiffCudaRegular::idxvar_kloop_sliced) {
        this->name = "hdiff-regular-idxvar-kloop-sliced";
    } else if(variant == HdiffCudaRegular::idxvar_shared){
        this->name = "hdiff-regular-idxvar-shared";
    }
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::run() {
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    if(this->variant == HdiffCudaRegular::naive) {
        HdiffCudaRegular::hdiff_naive<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::kloop) {
        HdiffCudaRegular::hdiff_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::shared) {
        int smem_size = 3*numthreads.x*numthreads.y*numthreads.z*sizeof(value_t);
        HdiffCudaRegular::hdiff_shared<value_t><<<this->numblocks(), numthreads, smem_size>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::jloop) {
        HdiffCudaRegular::hdiff_jloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->jloop_j_per_thread,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::iloop) {
        HdiffCudaRegular::hdiff_iloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->iloop_i_per_thread,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::idxvar) {
        HdiffCudaRegular::hdiff_idxvar<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::idxvar_shared) {
        int smem_size = HDIFF_IDXVAR_SHARED_SMEM_SZ_PER_THREAD*numthreads.x*numthreads.y*sizeof(int);
        HdiffCudaRegular::hdiff_idxvar_shared<value_t><<<this->numblocks(), this->numthreads(), smem_size>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::idxvar_kloop) {
        HdiffCudaRegular::hdiff_idxvar_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::idxvar_kloop_sliced) {
        HdiffCudaRegular::hdiff_idxvar_kloop_sliced<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->k_per_thread,
            this->inner_size,
            this->strides.y, this->strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::setup() {
    this->HdiffCudaBaseBenchmark<value_t>::setup();
    if(!this->setup_from_cache()) {
        this->input = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
        this->coeff = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
        this->populate_grids();
        this->store_to_cache();
    }
    this->output = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->lap = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->flx = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->fly = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    if(this->variant == HdiffCudaRegular::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->strides = (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_strides();
}

template<typename value_t>
bool HdiffCudaRegularBenchmark<value_t>::setup_from_archive(Benchmark::cache_iarchive &ar) {
    auto input = new CudaRegularGrid3D<value_t>(); //dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input);
    auto coeff = new CudaRegularGrid3D<value_t>(); //dynamic_cast<CudaRegularGrid3D<value_t> *>(this->coeff);
    ar >> *input;
    ar >> *coeff;
    this->input = input;
    this->coeff = coeff;
    return true;
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::store_to_archive(Benchmark::cache_oarchive &ar) {
    auto input = dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input);
    auto coeff = dynamic_cast<CudaRegularGrid3D<value_t> *>(this->coeff);
    ar << *input;
    ar << *coeff;
}

template<typename value_t>
std::string HdiffCudaRegularBenchmark<value_t>::cache_file_name() {
    return "hdiff-regular";
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::teardown() {
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffCudaBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffCudaBaseBenchmark<value_t>::post();
}

template<typename value_t>
dim3 HdiffCudaRegularBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == HdiffCudaRegular::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    } else if(this->variant == HdiffCudaRegular::jloop) {
        domain.y = 1;
    } else if(this->variant == HdiffCudaRegular::iloop) {
        domain.x = 1;
    }
    dim3 numthreads = this->HdiffCudaBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 HdiffCudaRegularBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->size;
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == HdiffCudaRegular::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    } else if(this->variant == HdiffCudaRegular::jloop) {
        domain.y = (this->size.y + this->jloop_j_per_thread - 1) / this->jloop_j_per_thread;
    } else if(this->variant == HdiffCudaRegular::iloop) {
        domain.x = (this->size.x + this->iloop_i_per_thread - 1) / this->iloop_i_per_thread;
    }
    dim3 numblocks = this->HdiffCudaBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::parse_args() {
    if(this->argc > 0) {
        // only variant of this that takes an argument is the jloop variant
        if(this->variant == HdiffCudaRegular::jloop) {
            sscanf(this->argv[0], "%d", &this->jloop_j_per_thread);
        }
        if(this->variant == HdiffCudaRegular::iloop) {
            sscanf(this->argv[0], "%d", &this->iloop_i_per_thread);
        }
        if(this->variant == HdiffCudaRegular::idxvar_kloop_sliced) {
            sscanf(this->argv[0], "%d", &this->k_per_thread);
        }
    } else {
        if(this->variant == HdiffCudaRegular::jloop) {
            this->jloop_j_per_thread = 16; // default value of 16
        }
        if(this->variant == HdiffCudaRegular::iloop) {
            this->iloop_i_per_thread = 16;
        }
    }
}

#endif