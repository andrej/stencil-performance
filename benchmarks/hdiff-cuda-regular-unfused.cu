#ifndef HDIFF_CUDA_UNFUSED_H
#define HDIFF_CUDA_UNFUSED_H
#include <sstream>
#include <string>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-cuda-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"

namespace HdiffCudaRegularUnfused {

    #define GRID_ARGS const int y_stride, const int z_stride, 
    #define INDEX(x, y, z) GRID_REGULAR_INDEX(y_stride, z_stride, x, y, z)
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < z_stride && k < info.max_coord.z)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_REGULAR_NEIGHBOR(y_stride, z_stride, idx, x_, y_, z_)
    #include "kernels/hdiff-unfused.cu"
    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef NEIGHBOR

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
template<typename value_t>
class HdiffCudaRegularUnfusedBenchmark : public HdiffCudaBaseBenchmark<value_t> {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaRegularUnfusedBenchmark(coord3 size);

    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    
    dim3 gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop);

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaRegularUnfusedBenchmark<value_t>::HdiffCudaRegularUnfusedBenchmark(coord3 size) :
HdiffCudaBaseBenchmark<value_t>(size) {
    this->name = "hdiff-regular-unfused";
    this->error = false;
}

template<typename value_t>
void HdiffCudaRegularUnfusedBenchmark<value_t>::run() {
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
    coord3 strides = (dynamic_cast<CudaRegularGrid3D<value_t> *>(this->input))->get_strides();
    CALL_KERNEL(HdiffCudaRegularUnfused::hdiff_unfused_lap, \
                nblocks_lap, nthreads, \
                this->get_info(),
                strides.y, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->lap->pointer(coord3(0, 0, 0)));
    dim3 nblocks_flx = this->gridsize(_nthreads, coord3(-1, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaRegularUnfused::hdiff_unfused_flx, \
                nblocks_flx, nthreads, \
                this->get_info(),
                strides.y, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->lap->pointer(coord3(0, 0, 0)),
                this->flx->pointer(coord3(0, 0, 0)));
    // Fly does not depend on Flx, so no need to synchronize here.
    dim3 nblocks_fly = this->gridsize(_nthreads, coord3(0, -1, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaRegularUnfused::hdiff_unfused_fly, \
                nblocks_fly, nthreads, \
                this->get_info(),
                strides.y, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->lap->pointer(coord3(0, 0, 0)),
                this->fly->pointer(coord3(0, 0, 0)));

    dim3 nblocks_out = this->gridsize(_nthreads, coord3(0, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaRegularUnfused::hdiff_unfused_out, \
                nblocks_out, nthreads, \
                this->get_info(),
                strides.y, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->coeff->pointer(coord3(0, 0, 0)),
                this->flx->pointer(coord3(0, 0, 0)),
                this->fly->pointer(coord3(0, 0, 0)),
                this->output->pointer(coord3(0, 0, 0)));
    if (cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

// Returns the correct amount of blocks (each of given blocksize) to implement
// a loop that runs from start, in increments of step, up to (not including) stop
// assumes step is positive!
template<typename value_t>
dim3 HdiffCudaRegularUnfusedBenchmark<value_t>::gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop) {
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
void HdiffCudaRegularUnfusedBenchmark<value_t>::setup() {
    this->input = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->output = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->coeff = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->lap = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->flx = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->fly = CudaRegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->HdiffCudaBaseBenchmark<value_t>::setup();
}

template<typename value_t>
void HdiffCudaRegularUnfusedBenchmark<value_t>::teardown() {
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffCudaBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
void HdiffCudaRegularUnfusedBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffCudaBaseBenchmark<value_t>::post();
}

#endif