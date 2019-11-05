#ifndef HDIFF_CUDA_SEQ_H
#define HDIFF_CUDA_SEQ_H
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"

namespace HdiffCudaSequential {

    /** Information about this benchmark for use in the kernels. */
    __device__ __host__
    struct Info {
        coord3 halo;
        coord3 inner_size;
    };

    // Laplace Kernel
    __global__
    void kernel_lap(Info info, CudaGridInfo<double> in, CudaGridInfo<double> lap) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1; // ref implementation starts at i = -1
        int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
        int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        // We have N threads per block and M blocks, so N*M threads total
        // Let our data set be of size L
        // Therefore, each thread needs to handle L/(N*M) data points to cover the entire data set
        // We do this in a "grid-stride loop": If the grid is large enough, each thread simply
        // calculates one data point at its thread idx (+ "base" starting offset of current block).
        // If the data set is larger, each thread addresses data points with a stride of the entire grid.
        // Rather than assume that the thread grid is large enough to cover the entire data array, this 
        // kernel loops over the data array one grid-size at a time.
        // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
        int data_size = (info.inner_size.x + 1) * (info.inner_size.y + 1) * info.inner_size.z; // size of data set in # total data points
        // the plus ones are because the reference implementation runs to isize + 1 ..
        int grid_size = (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y) * (gridDim.z*blockDim.z); // size of grid in # total threads (like this one)
        for(int l = 0; l < data_size; l += grid_size) {
            CUDA_REGULAR(lap, coord3(i, j, k)) = 
                4 * CUDA_REGULAR(in, coord3(i, j, k)) 
                - (CUDA_REGULAR(in, coord3(i-1, j, k)) 
                    + CUDA_REGULAR(in, coord3(i + 1, j, k)) 
                    + CUDA_REGULAR(in, coord3(i, j - 1, k)) 
                    + CUDA_REGULAR(in, coord3(i, j + 1, k)));
        }
    }

    // Flx Kernel
    __global__
    void kernel_flx(Info info, CudaGridInfo<double> in, CudaGridInfo<double> lap, CudaGridInfo<double> flx) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x - 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        int data_size = info.inner_size.x * info.inner_size.y * info.inner_size.z; 
        int grid_size = (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y) * (gridDim.z*blockDim.z); // size of grid in # total threads (like this one)
        for (int l = 0; l < data_size; l += grid_size) {
            CUDA_REGULAR(flx, coord3(i, j, k)) = CUDA_REGULAR(lap, coord3(i+1, j, k)) - CUDA_REGULAR(lap, coord3(i, j, k));
            if (CUDA_REGULAR(flx, coord3(i, j, k)) * (CUDA_REGULAR(in, coord3(i+1, j, k)) - CUDA_REGULAR(in, coord3(i, j, k))) > 0) {
                CUDA_REGULAR(flx, coord3(i, j, k)) = 0.;
            }
        }
    }

    // Fly Kernel
    __global__
    void kernel_fly(Info info, CudaGridInfo<double> in, CudaGridInfo<double> lap, CudaGridInfo<double> fly) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y - 1;
        int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        int data_size = info.inner_size.x * info.inner_size.y * info.inner_size.z; 
        int grid_size = (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y) * (gridDim.z*blockDim.z);
        for (int l = 0; l < data_size; l += grid_size) {
            CUDA_REGULAR(fly, coord3(i, j, k)) = CUDA_REGULAR(lap, coord3(i, j+1, k)) - CUDA_REGULAR(lap, coord3(i, j, k));
            if (CUDA_REGULAR(fly, coord3(i, j, k)) * (CUDA_REGULAR(in, coord3(i, j+1, k)) - CUDA_REGULAR(in, coord3(i, j, k))) > 0) {
                CUDA_REGULAR(fly, coord3(i, j, k)) = 0.;
            }
        }
    }

    // Output kernel
    __global__
    void kernel_out(Info info, CudaGridInfo<double> in, CudaGridInfo<double> coeff, CudaGridInfo<double> flx, CudaGridInfo<double> fly, CudaGridInfo<double> out) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + info.halo.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y + info.halo.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z + info.halo.z;
        int data_size = info.inner_size.x * info.inner_size.y * info.inner_size.z; 
        int grid_size = (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y) * (gridDim.z*blockDim.z);
        for (int l = 0; l < data_size; l += grid_size) {
            CUDA_REGULAR(out, coord3(i, j, k)) =
                CUDA_REGULAR(in, coord3(i, j, k)) -
                CUDA_REGULAR(coeff, coord3(i, j, k)) * (CUDA_REGULAR(flx, coord3(i, j, k)) - CUDA_REGULAR(flx, coord3(i - 1, j, k)) +
                                                CUDA_REGULAR(fly, coord3(i, j, k)) - CUDA_REGULAR(fly, coord3(i, j - 1, k)));

        }
    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
class HdiffCudaSequentialBenchmark : public HdiffReferenceBenchmark {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaSequentialBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void post();
    
    dim3 gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop);

    //CudaRegularGrid3D<double> *input;
    //CudaRegularGrid3D<double> *output;
    //CudaRegularGrid3D<double> *coeff;
    //CudaRegularGrid3D<double> *lap;
    //CudaRegularGrid3D<double> *flx;
    //CudaRegularGrid3D<double> *fly;

    // Return info struct for kernels
    HdiffCudaSequential::Info get_info();

};

// IMPLEMENTATIONS

HdiffCudaSequentialBenchmark::HdiffCudaSequentialBenchmark(coord3 size)
: HdiffReferenceBenchmark(size, RegularGrid) {
    this->name = "hdiff-cuda-seq";
}

void HdiffCudaSequentialBenchmark::run() {
    #define CALL_KERNEL(knl_func, nblocks, nthreads, ...) \
    knl_func<<<nblocks, nthreads>>>(__VA_ARGS__); \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        assert(false); \
    }
    dim3 nthreads = this->numthreads();
    coord3 _nthreads = coord3(nthreads.x, nthreads.y, nthreads.z);
    dim3 nblocks_lap = this->gridsize(_nthreads, coord3(-1, -1, 0), coord3(1, 1, 1), this->inner_size + coord3(+1, +1, 0));
    CALL_KERNEL(HdiffCudaSequential::kernel_lap, \
                nblocks_lap, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->input))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->lap))->get_gridinfo());
    if (cudaDeviceSynchronize() != cudaSuccess) {
        // need to synchronize because Flx kernel requires Lap
        assert(false);
    }
    dim3 nblocks_flx = this->gridsize(_nthreads, coord3(-1, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaSequential::kernel_flx, \
                nblocks_flx, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->input))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->lap))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->flx))->get_gridinfo());
    // Fly does not depend on Flx, so no need to synchronize here.
    dim3 nblocks_fly = this->gridsize(_nthreads, coord3(0, -1, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaSequential::kernel_fly, \
                nblocks_fly, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->input))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->lap))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->fly))->get_gridinfo());
    if (cudaDeviceSynchronize() != cudaSuccess) {
        assert(false);
    }
    dim3 nblocks_out = this->gridsize(_nthreads, coord3(0, 0, 0), coord3(1, 1, 1), this->inner_size);
    CALL_KERNEL(HdiffCudaSequential::kernel_out, \
                nblocks_out, nthreads, \
                this->get_info(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->input))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->coeff))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->flx))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->fly))->get_gridinfo(),
                (dynamic_cast<CudaRegularGrid3D<double> *>(this->output))->get_gridinfo());
    if (cudaDeviceSynchronize() != cudaSuccess) {
        assert(false);
    }
}

// Returns the correct amount of blocks (each of given blocksize) to implement
// a loop that runs from start, in increments of step, up to (not including) stop
// assumes step is positive!
dim3 HdiffCudaSequentialBenchmark::gridsize(coord3 blocksize, coord3 start, coord3 step, coord3 stop) {
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

void HdiffCudaSequentialBenchmark::setup() {
    this->input = new CudaRegularGrid3D<double>(this->size);
    this->output = new CudaRegularGrid3D<double>(this->size);
    this->coeff = new CudaRegularGrid3D<double>(this->size);
    this->lap = new CudaRegularGrid3D<double>(this->size);
    this->flx = new CudaRegularGrid3D<double>(this->size);
    this->fly = new CudaRegularGrid3D<double>(this->size);
    this->inner_size = this->size - 2*this->halo;
    this->HdiffReferenceBenchmark::populate_grids();
}

HdiffCudaSequential::Info HdiffCudaSequentialBenchmark::get_info() {
    return { .halo = this->halo,
             .inner_size = this->input->dimensions-2*this->halo};
}

void HdiffCudaSequentialBenchmark::post() {
    this->Benchmark::post();
}

#endif