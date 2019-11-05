#ifndef HDIFF_CUDA_UNSTR_H
#define HDIFF_CUDA_UNSTR_H
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstr {

    /** Information about this benchmark for use in the kernels. */
    __device__ __host__
    struct Info {
        coord3 halo;
        coord3 inner_size;
    };

    /** This kernel makes use of the regularity of the grid in the Z-direction.
     * Instead of naively resolving the neighborship relations at each k-step,
     * The locations of the neighboring cells are calculated at one level and
     * then reused, with the constant (regular) Z-step at each k-iteration.
     */
    __global__
    void kernel_idxvars(Info info,
                        CudaGridInfo<double> in,
                        CudaGridInfo<double> out,
                        CudaGridInfo<double> coeff) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i < info.halo.x || j < info.halo.y || i-info.halo.x > info.inner_size.x || j-info.halo.y > info.inner_size.y) {
            return;
        }

        /** Store index offsets for the current x and y coordinate, so we do
         * not have to recalculate them in every k-iteration. Instead, with
         * each iteration, the k-stride is simply added once -- thus making
         * use of the regularity of the grid in z-direction. 
         * idx of neighbor X Y Z = n_X_Y_Z with p for positive offset and 
         * n for negative offset. */
        coord3 coord(i, j, 0);
        int n_0_0_0       = CUDA_UNSTR_INDEX(in, coord);
        int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, -1, 0);
        int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_n1_0,  0, -1, 0);
        int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  -1, 0, 0);
        int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, -1, 0);
        int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0, -1, 0, 0);
        int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n2_0_0,  0, -1, 0);
        int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, +1, 0);
        int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_p1_0,  0, +1, 0);
        int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  +1, 0, 0);
        int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, +1, 0);
        int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0, +1, 0, 0);
        int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p2_0_0,  0, +1, 0);     
        int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, +1, 0);
        int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, -1, 0);

        for (int k = info.halo.z; k < info.inner_size.z + info.halo.z; k++) {
            const coord3 coord(i, j, k);

            double lap_ij = 
                4 * CUDA_UNSTR_AT(in, n_0_0_0) 
                - CUDA_UNSTR_AT(in, n_n1_0_0) - CUDA_UNSTR_AT(in, n_p1_0_0)
                - CUDA_UNSTR_AT(in, n_0_n1_0) - CUDA_UNSTR_AT(in, n_0_p1_0);
            double lap_imj = 
                4 * CUDA_UNSTR_AT(in, n_n1_0_0)
                - CUDA_UNSTR_AT(in, n_n2_0_0) - CUDA_UNSTR_AT(in, n_0_0_0)
                - CUDA_UNSTR_AT(in, n_n1_n1_0) - CUDA_UNSTR_AT(in, n_n1_p1_0);
            double lap_ipj =
                4 * CUDA_UNSTR_AT(in, n_p1_0_0)
                - CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_0_p2_0)
                - CUDA_UNSTR_AT(in, n_p1_n1_0) - CUDA_UNSTR_AT(in, n_p1_p1_0);
            double lap_ijm =
                4 * CUDA_UNSTR_AT(in, n_0_n1_0)
                - CUDA_UNSTR_AT(in, n_n1_n1_0) - CUDA_UNSTR_AT(in, n_p1_n1_0)
                - CUDA_UNSTR_AT(in, n_0_n2_0) - CUDA_UNSTR(in, coord);
            double lap_ijp =
                4 * CUDA_UNSTR_AT(in, n_0_p1_0)
                - CUDA_UNSTR_AT(in, n_n1_p1_0) - CUDA_UNSTR_AT(in, n_p1_p1_0)
                - CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_0_p2_0);

            double flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (CUDA_UNSTR_AT(in, n_p1_0_0) - CUDA_UNSTR(in, coord)) > 0 ? 0 : flx_ij;

            double flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_n1_0_0)) > 0 ? 0 : flx_imj;

            double fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (CUDA_UNSTR_AT(in, n_0_p1_0) - CUDA_UNSTR(in, coord)) > 0 ? 0 : fly_ij;

            double fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_0_n1_0)) > 0 ? 0 : fly_ijm;

            CUDA_UNSTR(out, coord) =
                CUDA_UNSTR(in, coord)
                - CUDA_UNSTR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);
            
            n_0_0_0       += in.strides.z;
            n_0_n1_0      += in.strides.z;
            n_0_n2_0      += in.strides.z;
            n_n1_0_0      += in.strides.z;
            n_n1_n1_0     += in.strides.z;
            n_n2_0_0      += in.strides.z;
            n_n2_n1_0     += in.strides.z;
            n_0_p1_0      += in.strides.z;
            n_0_p2_0      += in.strides.z;
            n_p1_0_0      += in.strides.z;
            n_p1_p1_0     += in.strides.z;
            n_p2_0_0      += in.strides.z;
            n_p2_p1_0     += in.strides.z;

        }
    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
class HdiffCudaUnstrBenchmark : public HdiffReferenceBenchmark {

    public:

    HdiffCudaUnstrBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void post();

    // Return info struct for kernels
    HdiffCudaUnstr::Info get_info();

};

// IMPLEMENTATIONS

HdiffCudaUnstrBenchmark::HdiffCudaUnstrBenchmark(coord3 size)
: HdiffReferenceBenchmark(size, UnstructuredGrid) {
    this->name = "hdiff-cuda-unstr";
}

void HdiffCudaUnstrBenchmark::run() {
    HdiffCudaUnstr::kernel_idxvars<<<this->numblocks(), this->numthreads()>>>(
        this->get_info(),
        (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->input))->get_gridinfo(),
        (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->output))->get_gridinfo(),
        (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->coeff))->get_gridinfo()
    );
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

void HdiffCudaUnstrBenchmark::setup() {
    this->input = new CudaUnstructuredGrid3D<double>(this->size);
    this->output = new CudaUnstructuredGrid3D<double>(this->size);
    this->coeff = new CudaUnstructuredGrid3D<double>(this->size);
    this->lap = new CudaUnstructuredGrid3D<double>(this->size);
    this->flx = new CudaUnstructuredGrid3D<double>(this->size);
    this->fly = new CudaUnstructuredGrid3D<double>(this->size);
    this->inner_size = this->size - 2*this->halo;
    this->HdiffReferenceBenchmark::populate_grids();
}

HdiffCudaUnstr::Info HdiffCudaUnstrBenchmark::get_info() {
    return { .halo = this->halo,
             .inner_size = this->input->dimensions-2*this->halo};
}

void HdiffCudaUnstrBenchmark::post() {
    this->Benchmark::post();
    this->HdiffReferenceBenchmark::post();
}

#endif