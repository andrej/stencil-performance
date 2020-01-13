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

    enum Variant { naive, iloop, jloop, kloop, coop, idxvar, idxvar_kloop, idxvar_shared, shared, shared_kloop };

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/hdiff-naive.cu"
    #include "kernels/hdiff-iloop.cu"
    #include "kernels/hdiff-jloop.cu"
    #include "kernels/hdiff-kloop.cu"
    #include "kernels/hdiff-coop.cu"
    #include "kernels/hdiff-idxvar.cu"
    #include "kernels/hdiff-idxvar-kloop.cu"
    #include "kernels/hdiff-idxvar-shared.cu"

    #define SMEM_GRID_ARGS
    #define SMEM_INDEX(_x, _y, _z) GRID_REGULAR_INDEX(blockDim.x, blockDim.x*blockDim.y, _x, _y, _z)
    #define SMEM_NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR(blockDim.x, blockDim.x*blockDim.y, idx, x_, y_, z_)

    #include "kernels/hdiff-shared.cu"
    #include "kernels/hdiff-shared-kloop.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR
    #undef NEXT_Z_NEIGHBOR
    #undef K_STEP
    #undef SMEM_GRID_ARGS
    #undef SMEM_INDEX
    #undef SMEM_NEIGHBOR

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

    dim3 numthreads();
    dim3 numblocks();
    
    // parameter for the jloop/iloop kernel only
    virtual void parse_args();
    int jloop_j_per_thread;
    int iloop_i_per_thread;

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
    } else if(variant == HdiffCudaRegular::shared_kloop) {
        this->name = "hdiff-regular-shared-kloop";
    } else if(variant == HdiffCudaRegular::coop) {
        this->name = "hdiff-regular-coop";
    } else if(variant == HdiffCudaRegular::jloop) {
        this->name = "hdiff-regular-jloop";
    } else if(variant == HdiffCudaRegular::iloop) {
        this->name = "hdiff-regular-iloop";
    } else if(variant == HdiffCudaRegular::idxvar) {
        this->name = "hdiff-regular-idxvar";
    } else if(variant == HdiffCudaRegular::idxvar_kloop) {
        this->name = "hdiff-regular-idxvar-kloop";
    } else {
        this->name = "hdiff-regular-idxvar-shared";
    }
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::run() {
    coord3 strides = (dynamic_cast<CudaRegularGrid3D<value_t>*>(this->input))->get_strides();
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    if(this->variant == HdiffCudaRegular::naive) {
        HdiffCudaRegular::hdiff_naive<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::kloop) {
        HdiffCudaRegular::hdiff_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::coop) {
        HdiffCudaRegular::hdiff_coop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0)),
            this->lap->pointer(coord3(0, 0, 0)),
            this->flx->pointer(coord3(0, 0, 0)),
            this->fly->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::shared) {
        int smem_size = 3*numthreads.x*numthreads.y*numthreads.z*sizeof(value_t);
        HdiffCudaRegular::hdiff_shared<value_t><<<this->numblocks(), numthreads, smem_size>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::shared_kloop) {
        int smem_size = 3*numthreads.x*numthreads.y*sizeof(value_t);
        HdiffCudaRegular::hdiff_shared_kloop<value_t><<<numblocks, numthreads, smem_size>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::jloop) {
        HdiffCudaRegular::hdiff_jloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->jloop_j_per_thread,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::iloop) {
        HdiffCudaRegular::hdiff_iloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->iloop_i_per_thread,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::idxvar) {
        HdiffCudaRegular::hdiff_idxvar<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else if(this->variant == HdiffCudaRegular::idxvar_shared) {
        int smem_size = numthreads.x*numthreads.y*12*sizeof(int);
        HdiffCudaRegular::hdiff_idxvar_shared<value_t><<<this->numblocks(), this->numthreads(), smem_size>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    } else {
        HdiffCudaRegular::hdiff_idxvar_kloop<value_t><<<this->numblocks(), this->numthreads()>>>(
            this->get_info(),
            strides.y, strides.z,
            this->input->pointer(coord3(0, 0, 0)),
            this->output->pointer(coord3(0, 0, 0)),
            this->coeff->pointer(coord3(0, 0, 0))
        );
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

template<typename value_t>
void HdiffCudaRegularBenchmark<value_t>::setup() {
    this->input = new CudaRegularGrid3D<value_t>(this->inner_size, this->halo);
    this->output = new CudaRegularGrid3D<value_t>(this->inner_size, this->halo);
    this->coeff = new CudaRegularGrid3D<value_t>(this->inner_size, this->halo);
    this->lap = new CudaRegularGrid3D<value_t>(this->inner_size, this->halo);
    this->flx = new CudaRegularGrid3D<value_t>(this->inner_size, this->halo);
    this->fly = new CudaRegularGrid3D<value_t>(this->inner_size, this->halo);
    if(this->variant == HdiffCudaRegular::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->HdiffCudaBaseBenchmark<value_t>::setup();
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
dim3 HdiffCudaRegularBenchmark<value_t>::numthreads() {
    dim3 numthreads = this->HdiffCudaBaseBenchmark<value_t>::numthreads();
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar_kloop ||
        this->variant == HdiffCudaRegular::shared_kloop) {
        numthreads.z = 1;
    }
    if(this->variant == HdiffCudaRegular::jloop) {
        numthreads.y = 1;
    }
    if(this->variant == HdiffCudaRegular::iloop) {
        numthreads.x = 1;
    }
    return numthreads;
}

template<typename value_t>
dim3 HdiffCudaRegularBenchmark<value_t>::numblocks() {
    dim3 numblocks = this->HdiffCudaBaseBenchmark<value_t>::numblocks();
    if(this->variant == HdiffCudaRegular::kloop ||
        this->variant == HdiffCudaRegular::idxvar_kloop ||
        this->variant == HdiffCudaRegular::shared_kloop) {
        numblocks.z = 1;
    }
    if(this->variant == HdiffCudaRegular::jloop) {
        numblocks.y = (this->size.y + this->jloop_j_per_thread - 1) / this->jloop_j_per_thread;
    }
    if(this->variant == HdiffCudaRegular::iloop) {
        numblocks.x = (this->size.x + this->iloop_i_per_thread - 1) / this->iloop_i_per_thread;
    }
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