#ifndef HDIFF_CUDA_SEQ_H
#define HDIFF_CUDA_SEQ_H
#include <sstream>
#include <string>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"

namespace HdiffCudaSequential {

    /** Information about this benchmark for use in the kernels. */
    struct Info {
        coord3 halo;
        coord3 max_coord;
    };

    // Laplace Kernel
    template<typename value_t>
    __global__
    void kernel_lap(Info info, CudaRegularGrid3DInfo<value_t> grid_info, value_t *in, value_t *lap) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1; // ref implementation starts at i = -1
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i >= info.max_coord.x-1 ||
           j >= info.max_coord.y-1 ||
           k >= info.max_coord.z) {
            return;
        }
        lap[CUDA_REGULAR_INDEX(grid_info, i, j, k)] = 
            4 * in[CUDA_REGULAR_INDEX(grid_info, i, j, k)] 
            - (in[CUDA_REGULAR_INDEX(grid_info, i - 1, j, k)] 
                + in[CUDA_REGULAR_INDEX(grid_info, i + 1, j, k)] 
                + in[CUDA_REGULAR_INDEX(grid_info, i, j - 1, k)] 
                + in[CUDA_REGULAR_INDEX(grid_info, i, j + 1, k)]);
    }

    // Flx Kernel
    template<typename value_t>
    __global__
    void kernel_flx(Info info, CudaRegularGrid3DInfo<value_t> grid_info, value_t *in, value_t *lap, value_t *flx) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i >= info.max_coord.x-1 ||
            j >= info.max_coord.y ||
            k >= info.max_coord.z) {
             return;
         }
        flx[CUDA_REGULAR_INDEX(grid_info, i, j, k)] = lap[CUDA_REGULAR_INDEX(grid_info, i+1, j, k)] - lap[CUDA_REGULAR_INDEX(grid_info, i, j, k)];
        if (flx[CUDA_REGULAR_INDEX(grid_info, i, j, k)] * (in[CUDA_REGULAR_INDEX(grid_info, i+1, j, k)] - in[CUDA_REGULAR_INDEX(grid_info, i, j, k)]) > 0) {
            flx[CUDA_REGULAR_INDEX(grid_info, i, j, k)] = 0.;
        }
    }

    // Fly Kernel
    template<typename value_t>
    __global__
    void kernel_fly(Info info, CudaRegularGrid3DInfo<value_t> grid_info, value_t *in, value_t *lap, value_t *fly) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i >= info.max_coord.x ||
            j >= info.max_coord.y-1 ||
            k >= info.max_coord.z) {
             return;
         }
        fly[CUDA_REGULAR_INDEX(grid_info, i, j, k)] = lap[CUDA_REGULAR_INDEX(grid_info, i, j+1, k)] - lap[CUDA_REGULAR_INDEX(grid_info, i, j, k)];
        if (fly[CUDA_REGULAR_INDEX(grid_info, i, j, k)] * (in[CUDA_REGULAR_INDEX(grid_info, i, j+1, k)] - in[CUDA_REGULAR_INDEX(grid_info, i, j, k)]) > 0) {
            fly[CUDA_REGULAR_INDEX(grid_info, i, j, k)] = 0.;
        }
    }

    // Output kernel
    template<typename value_t>
    __global__
    void kernel_out(Info info, CudaRegularGrid3DInfo<value_t> grid_info, value_t *in, value_t *coeff, value_t *flx, value_t *fly, value_t *out) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i >= info.max_coord.x ||
            j >= info.max_coord.y ||
            k >= info.max_coord.z) {
             return;
         }
        out[CUDA_REGULAR_INDEX(grid_info, i, j, k)] =
            in[CUDA_REGULAR_INDEX(grid_info, i, j, k)] -
            coeff[CUDA_REGULAR_INDEX(grid_info, i, j, k)] * (flx[CUDA_REGULAR_INDEX(grid_info, i, j, k)] - flx[CUDA_REGULAR_INDEX(grid_info, i - 1, j, k)] +
                                            fly[CUDA_REGULAR_INDEX(grid_info, i, j, k)] - fly[CUDA_REGULAR_INDEX(grid_info, i, j - 1, k)]);

    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
template<typename value_t>
class HdiffCudaSequentialBenchmark : public HdiffBaseBenchmark<value_t> {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaSequentialBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    
    dim3 gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop);

    // Return info struct for kernels
    HdiffCudaSequential::Info get_info();

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaSequentialBenchmark<value_t>::HdiffCudaSequentialBenchmark(coord3 size) :
HdiffBaseBenchmark<value_t>(size) {
    this->name = "hdiff-regular-seq";
    this->error = false;
}

template<typename value_t>
void HdiffCudaSequentialBenchmark<value_t>::run() {
    #define CALL_KERNEL(knl_func, nblocks, nthreads, ...) \
    knl_func<<<nblocks, nthreads>>>(__VA_ARGS__); \
    if (cudaGetLastError() != cudaSuccess) { \
        std::ostringstream msg; \
        msg << "Error trying to run kernel '" #knl_func "' with (" << nblocks.x << ", " << nblocks.y << ", " << nblocks.z << ") blocks and (" << nthreads.x << ", " << nthreads.y << ", " << nthreads.z << ") threads."; \
        throw std::runtime_error(msg.str()); \
    }
    dim3 nthreads = this->numthreads();
    coord3 _nthreads = coord3(nthreads.x, nthreads.y, nthreads.z);
    dim3 nblocks_lap = this->gridsize(_nthreads, coord3(-1, -1, 0), coord3(1, 1, 1), this->inner_size + coord3(+1, +1, 0));
    CALL_KERNEL(HdiffCudaSequential::kernel_lap, \
                nblocks_lap, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->lap->data);
    /*if (cudaDeviceSynchronize() != cudaSuccess) {
        // need to synchronize because Flx kernel requires Lap
        assert(false);
    }*/
    dim3 nblocks_flx = this->gridsize(_nthreads, coord3(-1, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaSequential::kernel_flx, \
                nblocks_flx, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->lap->data,
                this->flx->data);
    /*if (cudaDeviceSynchronize() != cudaSuccess) {
        assert(false);
    }*/
    // Fly does not depend on Flx, so no need to synchronize here.
    dim3 nblocks_fly = this->gridsize(_nthreads, coord3(0, -1, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaSequential::kernel_fly, \
                nblocks_fly, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->lap->data,
                this->fly->data);
    /*if (cudaDeviceSynchronize() != cudaSuccess) {
        assert(false);
    }*/
    dim3 nblocks_out = this->gridsize(_nthreads, coord3(0, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaSequential::kernel_out, \
                nblocks_out, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->coeff->data,
                this->flx->data,
                this->fly->data,
                this->output->data);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

// Returns the correct amount of blocks (each of given blocksize) to implement
// a loop that runs from start, in increments of step, up to (not including) stop
// assumes step is positive!
template<typename value_t>
dim3 HdiffCudaSequentialBenchmark<value_t>::gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop) {
    dim3 n_iterations = dim3(
        (unsigned int) (stop.x-start.x)/(step.x),
        (unsigned int) (stop.y-start.y)/(step.y),
        (unsigned int) (stop.z-start.z)/(step.z)
    );
    return dim3(
        ((n_iterations.x + blocksize.x - 1) / blocksize.x),
        ((n_iterations.y + blocksize.y - 1) / blocksize.y),
        ((n_iterations.z + blocksize.z - 1) / blocksize.z)
    );
}

template<typename value_t>
void HdiffCudaSequentialBenchmark<value_t>::setup() {
    this->input = new CudaRegularGrid3D<value_t>(this->size);
    this->output = new CudaRegularGrid3D<value_t>(this->size);
    this->coeff = new CudaRegularGrid3D<value_t>(this->size);
    this->lap = new CudaRegularGrid3D<value_t>(this->size);
    this->flx = new CudaRegularGrid3D<value_t>(this->size);
    this->fly = new CudaRegularGrid3D<value_t>(this->size);
    this->HdiffBaseBenchmark<value_t>::setup();
}

template<typename value_t>
void HdiffCudaSequentialBenchmark<value_t>::teardown() {
    this->input->deallocate();
    this->output->deallocate();
    this->coeff->deallocate();
    this->lap->deallocate();
    this->flx->deallocate();
    this->fly->deallocate();
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
HdiffCudaSequential::Info HdiffCudaSequentialBenchmark<value_t>::get_info() {
    return { .halo = this->halo,
             .max_coord = this->input->dimensions - this->halo};
}

template<typename value_t>
void HdiffCudaSequentialBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffBaseBenchmark<value_t>::post();
}

#endif