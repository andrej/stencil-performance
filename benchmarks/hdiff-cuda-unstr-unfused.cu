#ifndef HDIFF_CUDA_UNSTR_UNFUSED_H
#define HDIFF_CUDA_UNSTR_UNFUSED_H
#include <sstream>
#include <string>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstructuredUnfused {

    #define GRID_ARGS const int* neighborships, const int z_stride, 
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + (z_)*blockDim.x*gridDim.x*blockDim.y*gridDim.y
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #include "kernels/hdiff-unfused.cu"
    #undef GRID_ARGS
    #undef INDEX
    #undef NEIGHBOR

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
template<typename value_t>
class HdiffCudaUnstructuredUnfusedBenchmark : public HdiffCudaBaseBenchmark<value_t> {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaUnstructuredUnfusedBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    
    dim3 gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop);

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaUnstructuredUnfusedBenchmark<value_t>::HdiffCudaUnstructuredUnfusedBenchmark(coord3 size) :
HdiffCudaBaseBenchmark<value_t>(size) {
    this->name = "hdiff-unstr-unfused";
    this->error = false;
}

template<typename value_t>
void HdiffCudaUnstructuredUnfusedBenchmark<value_t>::run() {
    /*#define CALL_KERNEL(knl_func, nblocks, nthreads, ...) \
    knl_func<<<nblocks, nthreads>>>(__VA_ARGS__); \
    if (cudaGetLastError() != cudaSuccess) { \
        std::ostringstream msg; \
        msg << "Unable to run kernel '" #knl_func "' with (" << nblocks.x << ", " << nblocks.y << ", " << nblocks.z << ") blocks and (" << nthreads.x << ", " << nthreads.y << ", " << nthreads.z << ") threads."; \
        throw std::runtime_error(msg.str()); \
    }*/
    dim3 nthreads = this->numthreads();
    coord3 _nthreads = coord3(nthreads.x, nthreads.y, nthreads.z);
    dim3 nblocks_lap = this->gridsize(_nthreads, coord3(-1, -1, 0), coord3(1, 1, 1), this->inner_size + coord3(+1, +1, 0));
    CudaUnstructuredGrid3D<value_t> *unstr_input = (dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input));
    coord3 strides(1, unstr_input->dimensions.x, unstr_input->dimensions.x*unstr_input->dimensions.y);
    int *neighborships = unstr_input->neighborships;
    CALL_KERNEL(HdiffCudaUnstructuredUnfused::hdiff_unfused_lap, \
                nblocks_lap, nthreads, \
                this->get_info(),
                neighborships, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->lap->pointer(coord3(0, 0, 0)));
    dim3 nblocks_flx = this->gridsize(_nthreads, coord3(-1, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaUnstructuredUnfused::hdiff_unfused_flx, \
                nblocks_flx, nthreads, \
                this->get_info(),
                neighborships, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->lap->pointer(coord3(0, 0, 0)),
                this->flx->pointer(coord3(0, 0, 0)));
    // Fly does not depend on Flx, so no need to synchronize here.
    dim3 nblocks_fly = this->gridsize(_nthreads, coord3(0, -1, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaUnstructuredUnfused::hdiff_unfused_fly, \
                nblocks_fly, nthreads, \
                this->get_info(),
                neighborships, strides.z,
                this->input->pointer(coord3(0, 0, 0)),
                this->lap->pointer(coord3(0, 0, 0)),
                this->fly->pointer(coord3(0, 0, 0)));
    dim3 nblocks_out = this->gridsize(_nthreads, coord3(0, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaUnstructuredUnfused::hdiff_unfused_out, \
                nblocks_out, nthreads, \
                this->get_info(),
                neighborships, strides.z,
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
dim3 HdiffCudaUnstructuredUnfusedBenchmark<value_t>::gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop) {
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
void HdiffCudaUnstructuredUnfusedBenchmark<value_t>::setup() {
    this->input = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    int *neighborships = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input)->neighborships;
    //this->output = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->coeff = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->lap = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->flx = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    //this->fly = CudaUnstructuredGrid3D<value_t>::create_regular(this->size);
    this->output = new CudaUnstructuredGrid3D<value_t>(this->size, neighborships);
    this->coeff = new CudaUnstructuredGrid3D<value_t>(this->size, neighborships);
    this->lap = new CudaUnstructuredGrid3D<value_t>(this->size, neighborships);
    this->flx = new CudaUnstructuredGrid3D<value_t>(this->size, neighborships);
    this->fly = new CudaUnstructuredGrid3D<value_t>(this->size, neighborships);
    this->HdiffCudaBaseBenchmark<value_t>::setup();
}

template<typename value_t>
void HdiffCudaUnstructuredUnfusedBenchmark<value_t>::teardown() {
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffCudaBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
void HdiffCudaUnstructuredUnfusedBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffCudaBaseBenchmark<value_t>::post();
}

#endif