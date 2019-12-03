#ifndef HDIFF_CUDA_UNSTR_SEQ_H
#define HDIFF_CUDA_UNSTR_SEQ_H
#include <sstream>
#include <string>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstructuredSequential {

    /** Information about this benchmark for use in the kernels. */
    struct Info {
        coord3 halo;
        coord3 max_coord;
    };

    // Laplace Kernel
    __global__
    void kernel_lap(Info info, CudaUnstructuredGrid3DInfo<double> grid_info, double *in, double *lap) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i > info.max_coord.x-1 ||
           j > info.max_coord.y-1 ||
           k > info.max_coord.z) {
            return;
        }
        lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = 
            4 * in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] 
            - (in[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] 
                + in[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] 
                + in[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)] 
                + in[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)]);
    }

    // Flx Kernel
    __global__
    void kernel_flx(Info info, CudaUnstructuredGrid3DInfo<double> grid_info, double *in, double *lap, double *flx) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i > info.max_coord.x-1 ||
            j > info.max_coord.y ||
            k > info.max_coord.z) {
             return;
        }
        flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] - lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)];
        if (flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (in[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, +1, 0, 0)] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0) {
            flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = 0.;
        }
    }

    // Fly Kernel
    __global__
    void kernel_fly(Info info, CudaUnstructuredGrid3DInfo<double> grid_info, double *in, double *lap, double *fly) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i > info.max_coord.x ||
            j > info.max_coord.y-1 ||
            k > info.max_coord.z) {
             return;
        }
        fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = lap[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)] - lap[CUDA_UNSTR_INDEX(grid_info, i, j, k)];
        if (fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (in[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, +1, 0)] - in[CUDA_UNSTR_INDEX(grid_info, i, j, k)]) > 0) {
            fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] = 0.;
        }
    }

    // Output kernel
    __global__
    void kernel_out(Info info, CudaUnstructuredGrid3DInfo<double> grid_info, double *in, double *coeff, double *flx, double *fly, double *out) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
        const int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        if(i > info.max_coord.x ||
           j > info.max_coord.y ||
           k > info.max_coord.z) {
            return;
        }
        out[CUDA_UNSTR_INDEX(grid_info, i, j, k)] =
            in[CUDA_UNSTR_INDEX(grid_info, i, j, k)] -
            coeff[CUDA_UNSTR_INDEX(grid_info, i, j, k)] * (flx[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - flx[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, -1, 0, 0)] +
                                            fly[CUDA_UNSTR_INDEX(grid_info, i, j, k)] - fly[CUDA_UNSTR_NEIGHBOR(grid_info, i, j, k, 0, -1, 0)]);
    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
class HdiffCudaUnstructuredSequentialBenchmark : public HdiffBaseBenchmark {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaUnstructuredSequentialBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    
    dim3 gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop);

    // Return info struct for kernels
    HdiffCudaUnstructuredSequential::Info get_info();

};

// IMPLEMENTATIONS

HdiffCudaUnstructuredSequentialBenchmark::HdiffCudaUnstructuredSequentialBenchmark(coord3 size) :
HdiffBaseBenchmark(size) {
    this->name = "hdiff-unstr-seq";
    this->error = false;
}

void HdiffCudaUnstructuredSequentialBenchmark::run() {
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
    CALL_KERNEL(HdiffCudaUnstructuredSequential::kernel_lap, \
                nblocks_lap, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaUnstructuredGrid3D<double> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->lap->data);
    dim3 nblocks_flx = this->gridsize(_nthreads, coord3(-1, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaUnstructuredSequential::kernel_flx, \
                nblocks_flx, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaUnstructuredGrid3D<double> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->lap->data,
                this->flx->data);
    // Fly does not depend on Flx, so no need to synchronize here.
    dim3 nblocks_fly = this->gridsize(_nthreads, coord3(0, -1, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaUnstructuredSequential::kernel_fly, \
                nblocks_fly, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaUnstructuredGrid3D<double> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->lap->data,
                this->fly->data);
    dim3 nblocks_out = this->gridsize(_nthreads, coord3(0, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaUnstructuredSequential::kernel_out, \
                nblocks_out, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaUnstructuredGrid3D<double> *>(this->input))->get_gridinfo(),
                this->input->data,
                this->coeff->data,
                this->flx->data,
                this->fly->data,
                this->output->data);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        assert(false);
    }
}

// Returns the correct amount of blocks (each of given blocksize) to implement
// a loop that runs from start, in increments of step, up to (not including) stop
// assumes step is positive!
dim3 HdiffCudaUnstructuredSequentialBenchmark::gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop) {
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

void HdiffCudaUnstructuredSequentialBenchmark::setup() {
    this->input = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    int *neighbor_data = dynamic_cast<CudaUnstructuredGrid3D<double> *>(this->input)->neighbor_data;
    //this->output = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    //this->coeff = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    //this->lap = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    //this->flx = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    //this->fly = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->output = new CudaUnstructuredGrid3D<double>(this->size, neighbor_data);
    this->coeff = new CudaUnstructuredGrid3D<double>(this->size, neighbor_data);
    this->lap = new CudaUnstructuredGrid3D<double>(this->size, neighbor_data);
    this->flx = new CudaUnstructuredGrid3D<double>(this->size, neighbor_data);
    this->fly = new CudaUnstructuredGrid3D<double>(this->size, neighbor_data);
    this->HdiffBaseBenchmark::setup();
}

void HdiffCudaUnstructuredSequentialBenchmark::teardown() {
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
    this->HdiffBaseBenchmark::teardown();
}

HdiffCudaUnstructuredSequential::Info HdiffCudaUnstructuredSequentialBenchmark::get_info() {
    return { .halo = this->halo,
             .max_coord = this->input->dimensions-this->halo};
}

void HdiffCudaUnstructuredSequentialBenchmark::post() {
    this->Benchmark::post();
    this->HdiffBaseBenchmark::post();
}

#endif