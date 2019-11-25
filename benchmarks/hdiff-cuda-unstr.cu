#ifndef HDIFF_CUDA_UNSTR_H
#define HDIFF_CUDA_UNSTR_H
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstr {

    /** Variants of this benchmark. */
    enum Variant { RegNaive, RegKloop, UnstrNaive, UnstrKloop, UnstrIdxvars };

    /** Information about this benchmark for use in the kernels. */
    struct Info {
        coord3 halo;
        coord3 max_coord;
    };

    /** Naive implementation of a unstructured grid horizontal diffusion
     * kernel. Runs index calculations in every k-iteration.
     */
     __global__
     void kernel_naive(Info info,
                         CudaUnstructuredGrid3DInfo<double> in,
                         CudaUnstructuredGrid3DInfo<double> out,
                         CudaUnstructuredGrid3DInfo<double> coeff
                         #ifdef HDIFF_DEBUG
                         , CudaUnstructuredGrid3DInfo<double> dbg_lap
                         , CudaUnstructuredGrid3DInfo<double> dbg_flx
                         , CudaUnstructuredGrid3DInfo<double> dbg_fly
                         #endif
                         ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        const int k = threadIdx.z + blockIdx.z*blockDim.z + info.halo.z;
        if(i >= info.max_coord.x || j >= info.max_coord.y || info.halo.z >= info.max_coord.z) {
            return;
        }
        coord3 coord(i, j, k);
        
        int n_0_0_0       = CUDA_UNSTR_INDEX(in, coord);
        int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, -1, 0);
        int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_n1_0,  0, -1, 0);
        int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  -1, 0, 0);
        int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, -1, 0);
        int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0, -1, 0, 0);
        //int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n2_0_0,  0, -1, 0);
        int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, +1, 0);
        int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_p1_0,  0, +1, 0);
        int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  +1, 0, 0);
        int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, +1, 0);
        int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0, +1, 0, 0);
        //int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p2_0_0,  0, +1, 0);     
        int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, +1, 0);
        int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, -1, 0);

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
            - CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_p2_0_0)
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
        
        // for debugging purposes:
        #ifdef HDIFF_DEBUG
        CUDA_UNSTR(dbg_lap, coord) = lap_ij;
        CUDA_UNSTR_NEIGH(dbg_lap, coord, -1, 0, 0) = lap_imj;
        CUDA_UNSTR_NEIGH(dbg_lap, coord, 0, -1, 0) = lap_ijm;
        CUDA_UNSTR_NEIGH(dbg_lap, coord, +1, 0, 0) = lap_ipj;
        CUDA_UNSTR_NEIGH(dbg_lap, coord, 0, +1, 0) = lap_ijp;
        CUDA_UNSTR(dbg_flx, coord) = flx_ij;
        CUDA_UNSTR_NEIGH(dbg_flx, coord, -1, 0, 0) = flx_imj;
        CUDA_UNSTR(dbg_fly, coord) = fly_ij;
        CUDA_UNSTR_NEIGH(dbg_fly, coord, 0, -1, 0) = fly_ijm;
        #endif
 
    }

        /** Naive implementation of a unstructured grid horizontal diffusion
     * kernel. Runs index calculations in every k-iteration.
     */
     __global__
     void kernel_naive_kloop(Info info,
                             CudaUnstructuredGrid3DInfo<double> in,
                             CudaUnstructuredGrid3DInfo<double> out,
                             CudaUnstructuredGrid3DInfo<double> coeff
                             #ifdef HDIFF_DEBUG
                             , CudaUnstructuredGrid3DInfo<double> dbg_lap
                             , CudaUnstructuredGrid3DInfo<double> dbg_flx
                             , CudaUnstructuredGrid3DInfo<double> dbg_fly
                             #endif
                             ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }
        
        for(int k = info.halo.z; k < info.max_coord.z; k++) {
            coord3 coord(i, j, k);

            int n_0_0_0       = CUDA_UNSTR_INDEX(in, coord);
            int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, -1, 0);
            int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_n1_0,  0, -1, 0);
            int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  -1, 0, 0);
            int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, -1, 0);
            int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0, -1, 0, 0);
            //int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n2_0_0,  0, -1, 0);
            int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, +1, 0);
            int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_p1_0,  0, +1, 0);
            int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  +1, 0, 0);
            int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, +1, 0);
            int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0, +1, 0, 0);
            //int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p2_0_0,  0, +1, 0);     
            int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, +1, 0);
            int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, -1, 0);

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
                - CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_p2_0_0)
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
            
            // for debugging purposes:
            #ifdef HDIFF_DEBUG
            CUDA_UNSTR(dbg_lap, coord) = lap_ij;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, -1, 0, 0) = lap_imj;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, 0, -1, 0) = lap_ijm;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, +1, 0, 0) = lap_ipj;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, 0, +1, 0) = lap_ijp;
            CUDA_UNSTR(dbg_flx, coord) = flx_ij;
            CUDA_UNSTR_NEIGH(dbg_flx, coord, -1, 0, 0) = flx_imj;
            CUDA_UNSTR(dbg_fly, coord) = fly_ij;
            CUDA_UNSTR_NEIGH(dbg_fly, coord, 0, -1, 0) = fly_ijm;
            #endif
        }
 
    }

    /** This kernel makes use of the regularity of the grid in the Z-direction.
     * Instead of naively resolving the neighborship relations at each k-step,
     * The locations of the neighboring cells are calculated at one level and
     * then reused, with the constant (regular) Z-step at each k-iteration.
     */
    __global__
    void kernel_idxvars(Info info,
                        CudaUnstructuredGrid3DInfo<double> in,
                        CudaUnstructuredGrid3DInfo<double> out,
                        CudaUnstructuredGrid3DInfo<double> coeff
                        #ifdef HDIFF_DEBUG
                        , CudaUnstructuredGrid3DInfo<double> dbg_lap
                        , CudaUnstructuredGrid3DInfo<double> dbg_flx
                        , CudaUnstructuredGrid3DInfo<double> dbg_fly
                        #endif
                        ) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        if(i >= info.max_coord.x || j >= info.max_coord.y) {
            return;
        }

        /** Store index offsets for the current x and y coordinate, so we do
         * not have to recalculate them in every k-iteration. Instead, with
         * each iteration, the k-stride is simply added once -- thus making
         * use of the regularity of the grid in z-direction. 
         * idx of neighbor X Y Z = n_X_Y_Z with p for positive offset and 
         * n for negative offset. */
        coord3 coord(i, j, info.halo.z);
        int n_0_0_0       = CUDA_UNSTR_INDEX(in, coord);
        int n_0_n1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, -1, 0);
        int n_0_n2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_n1_0,  0, -1, 0);
        int n_n1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  -1, 0, 0);
        int n_n1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, -1, 0);
        int n_n2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0, -1, 0, 0);
        //int n_n2_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n2_0_0,  0, -1, 0);
        int n_0_p1_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,   0, +1, 0);
        int n_0_p2_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_p1_0,  0, +1, 0);
        int n_p1_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_0_0_0,  +1, 0, 0);
        int n_p1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, +1, 0);
        int n_p2_0_0      = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0, +1, 0, 0);
        //int n_p2_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p2_0_0,  0, +1, 0);     
        int n_n1_p1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_n1_0_0,  0, +1, 0);
        int n_p1_n1_0     = CUDA_UNSTR_NEIGHBOR_AT(in, n_p1_0_0,  0, -1, 0);

        for (int k = info.halo.z; k < info.max_coord.z; k++) {
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
                - CUDA_UNSTR(in, coord) - CUDA_UNSTR_AT(in, n_p2_0_0)
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
            
            // DEBUG: Output intermediate results as well
            // Disable this for better performance
            #ifdef HDIFF_DEBUG
            CUDA_UNSTR(dbg_lap, coord) = lap_ij;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, -1, 0, 0) = lap_imj;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, 0, -1, 0) = lap_ijm;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, +1, 0, 0) = lap_ipj;
            CUDA_UNSTR_NEIGH(dbg_lap, coord, 0, +1, 0) = lap_ijp;
            CUDA_UNSTR(dbg_flx, coord) = flx_ij;
            CUDA_UNSTR_NEIGH(dbg_flx, coord, -1, 0, 0) = flx_imj;
            CUDA_UNSTR(dbg_fly, coord) = fly_ij;
            CUDA_UNSTR_NEIGH(dbg_fly, coord, 0, -1, 0) = fly_ijm;
            #endif

            // Make use of regularity in Z-direciton: neighbors are exactly the
            // same, just one Z-stride apart.
            n_0_0_0       += in.strides.z;
            n_0_n1_0      += in.strides.z;
            n_0_n2_0      += in.strides.z;
            n_n1_0_0      += in.strides.z;
            n_n1_n1_0     += in.strides.z;
            n_n2_0_0      += in.strides.z;
            //n_n2_n1_0     += in.strides.z;
            n_0_p1_0      += in.strides.z;
            n_0_p2_0      += in.strides.z;
            n_p1_0_0      += in.strides.z;
            n_p1_p1_0     += in.strides.z;
            n_p2_0_0      += in.strides.z;
            //n_p2_p1_0     += in.strides.z;
            n_n1_p1_0     += in.strides.z;
            n_p1_n1_0     += in.strides.z;

        }
    }

};

/** Cuda implementation of different variants of the horizontal diffusion
 * kernel, both for structured and unstructured grid variants.
 *
 * For the available variants, see the HdiffCuda::Variant enum. */
class HdiffCudaUnstrBenchmark : public HdiffBaseBenchmark {

    public:

    HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant=HdiffCudaUnstr::UnstrNaive);
    
    HdiffCudaUnstr::Variant variant;

    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    virtual dim3 numblocks();
    virtual dim3 numthreads();

    // Return info struct for kernels
    HdiffCudaUnstr::Info get_info();

};

// IMPLEMENTATIONS

HdiffCudaUnstrBenchmark::HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant) :
HdiffBaseBenchmark(size) {
    if(variant == HdiffCudaUnstr::UnstrNaive) {
        this->name = "hdiff-unstr-naive";
    } else if(variant == HdiffCudaUnstr::UnstrKloop) {
        this->name = "hdiff-unstr-kloop";
    } else {
        this->name = "hdiff-unstr-idxvar";
    }
    this->error = false;
    this->variant = variant;
}

void HdiffCudaUnstrBenchmark::run() {
    auto kernel_fun = &HdiffCudaUnstr::kernel_naive;
    if(this->variant == HdiffCudaUnstr::UnstrIdxvars) {
        kernel_fun = &HdiffCudaUnstr::kernel_idxvars;
    } else if(this->variant == HdiffCudaUnstr::UnstrKloop) {
        kernel_fun = &HdiffCudaUnstr::kernel_naive_kloop;
    }
    (*kernel_fun)<<<this->numblocks(), this->numthreads()>>>(
        this->get_info(),
        (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->input))->get_gridinfo(),
        (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->output))->get_gridinfo(),
        (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->coeff))->get_gridinfo()
        #ifdef HDIFF_DEBUG
        , (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->lap))->get_gridinfo()
        , (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->flx))->get_gridinfo()
        , (dynamic_cast<CudaUnstructuredGrid3D<double>*>(this->fly))->get_gridinfo()
        #endif
    );
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

dim3 HdiffCudaUnstrBenchmark::numblocks() {
    dim3 numblocks = this->Benchmark::numblocks();
    // For the vriants that use a k-loop inside the kernel, we only need one block in the k-direction
    if(this->variant == HdiffCudaUnstr::UnstrIdxvars ||
       this->variant == HdiffCudaUnstr::UnstrKloop ) {
        numblocks = dim3(numblocks.x, numblocks.y, 1);
    }
    return numblocks;
}

dim3 HdiffCudaUnstrBenchmark::numthreads() {
    dim3 numthreads = this->Benchmark::numthreads();
    // Variants with a k-loop: only one thread in the k-direction
    if(this->variant == HdiffCudaUnstr::UnstrIdxvars ||
        this->variant == HdiffCudaUnstr::UnstrKloop ) {
        numthreads = dim3(numthreads.x, numthreads.y, 1);
    }
    return numthreads;
}

void HdiffCudaUnstrBenchmark::setup() {
    this->input = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->output = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->coeff = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->lap = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->flx = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->fly = CudaUnstructuredGrid3D<double>::create_regular(this->size);
    this->HdiffBaseBenchmark::setup();
    int s1 = cudaMemPrefetchAsync(this->input->data, this->input->size, 0);
    int s2 = cudaMemPrefetchAsync(this->output->data, this->output->size, 0);
    int s3 = cudaMemPrefetchAsync(this->coeff->data, this->coeff->size, 0);
    #ifdef HDIFF_DEBUG
    int s4 = cudaMemPrefetchAsync(this->lap->data, this->lap->size, 0);
    int s5 = cudaMemPrefetchAsync(this->flx->data, this->flx->size, 0);
    int s6 = cudaMemPrefetchAsync(this->fly->data, this->fly->size, 0);
    #endif
    if( s1 != cudaSuccess || s2 != cudaSuccess || s3 != cudaSuccess
        #ifdef HDIFF_DEBUG
            || s4 != cudaSuccess || s5 != cudaSuccess || s6 != cudaSuccess
        #endif
    ) {
        throw std::runtime_error("unable to prefetch memory");
    }
}

void HdiffCudaUnstrBenchmark::teardown() {
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

void HdiffCudaUnstrBenchmark::post() {
    this->Benchmark::post();
    this->HdiffBaseBenchmark::post();
}

HdiffCudaUnstr::Info HdiffCudaUnstrBenchmark::get_info() {
    return { .halo = this->halo,
             .max_coord = this->input->dimensions - this->halo};
}

#endif