#ifndef HDIFF_CUDA_H
#define HDIFF_CUDA_H
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaRegular {

    /** Information about this benchmark for use in the kernels. */
    __device__ __host__
    struct Info {
        coord3 halo;
        coord3 inner_size;
    };

    __global__
    void kernel_direct(Info info,
                       CudaGridInfo<double> in,
                       CudaGridInfo<double> out,
                       CudaGridInfo<double> coeff) {
        const int i = threadIdx.x + blockIdx.x*blockDim.x + info.halo.x;
        const int j = threadIdx.y + blockIdx.y*blockDim.y + info.halo.y;
        //output.data[CUDA_REGULAR_INDEX(output, coord3(x, y, 0))] = 2*CUDA_REGULAR((input, coord3(x, y, 0));
        //return;
        if(i < info.halo.x || j < info.halo.y || i-info.halo.x > info.inner_size.x || j-info.halo.y > info.inner_size.y) {
            return;
        }
        for (int k = info.halo.z; k < info.inner_size.z + info.halo.z; k++) {
            const coord3 coord = coord3(i, j, k);

            double lap_ij = 
                4 * CUDA_REGULAR(in, coord) 
                - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0);
            double lap_imj = 
                4 * CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -2, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0);
            double lap_ipj =
                4 * CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, 0, 0) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0)
                - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0);
            double lap_ijm =
                4 * CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, -1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, -1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, 0, -2, 0) - CUDA_REGULAR(in, coord);
            double lap_ijp =
                4 * CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0)
                - CUDA_REGULAR_NEIGH(in, coord, -1, +1, 0) - CUDA_REGULAR_NEIGH(in, coord, +1, +1, 0)
                - CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, +2, 0);

            double flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (CUDA_REGULAR_NEIGH(in, coord, +1, 0, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : flx_ij;

            double flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, -1, 0, 0)) > 0 ? 0 : flx_imj;

            double fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (CUDA_REGULAR_NEIGH(in, coord, 0, +1, 0) - CUDA_REGULAR(in, coord)) > 0 ? 0 : fly_ij;

            double fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (CUDA_REGULAR(in, coord) - CUDA_REGULAR_NEIGH(in, coord, 0, -1, 0)) > 0 ? 0 : fly_ijm;

            CUDA_REGULAR(out, coord) =
                CUDA_REGULAR(in, coord)
                - CUDA_REGULAR(coeff, coord) * (flx_ij - flx_imj + fly_ij - fly_ijm);

            // for debugging purposes:
            //CUDA_REGULAR(out, coord) = flx_ij;
            //CUDA_REGULAR_NEIGH(out, coord, -1, 0, 0) = flx_imj;
            //CUDA_REGULAR_NEIGH(out, coord, 0, -1, 0) = lap_ijm;
            /*for (int i_offs = -1; i_offs <= 0; i_offs++) {
                for (int j_offs = -1; j_offs <= 0; j_offs++) {
                    const int i = x + i_offs;
                    const int j = y + j_offs;
                    lap[i_offs+1][j_offs+1] = 
                        4 * CUDA_REGULAR(input, coord3(i, j, k))
                            - (CUDA_REGULAR_NEIGH(input, coord3(i, j, k), -1, 0, 0)
                                + CUDA_REGULAR_NEIGH(input, coord3(i, j, k), +1, 0, 0)
                                + CUDA_REGULAR_NEIGH(input, coord3(i, j, k), 0, -1, 0)
                                + CUDA_REGULAR_NEIGH(input, coord3(i, j, k), 0, +1, 0));
                }
            }
            for (int i_offs = -1; i_offs <= 0; i_offs++) {
                int i = x + i_offs;
                int j = y;
                flx[i_offs+1][0] = lap[1][1] - lap[0][1]; // lap center - lap left
                if (flx[i_offs+1][0] * (CUDA_REGULAR_NEIGH(input, coord3(i, j, k), +1, 0, 0)
                                        - CUDA_REGULAR(input, coord3(i, j, k))) > 0) {
                    flx[i_offs+1][0] = 0.;
                }
            }
            for (int j_offs = -1; j_offs <= 0; j_offs++) {
                int i = x;
                int j = y + j_offs;
                fly[0][j_offs+1] = lap[1][1] - lap[1][0]; // lap center - lap top
                if (fly[0][j_offs+1] * (CUDA_REGULAR_NEIGH(input, coord3(i, j, k), 0, +1, 0)
                                        - CUDA_REGULAR(input, coord3(i, j, k))) > 0) {
                    flx[0][j_offs+1] = 0.;
                }
            }
            out =
                CUDA_REGULAR(input, coord3(x, y, k))
                - CUDA_REGULAR(coeff, coord3(x, y, k))
                * ( flx[1][0]-flx[0][0] + fly[0][1] - fly[0][0] );
            output.data[CUDA_REGULAR_INDEX(output, coord3(x, y, k))] = out;*/
        }
    }

};

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
class HdiffCudaBenchmark : public HdiffReferenceBenchmark {

    public:

    // The padding option currently only applies to regular grids
    HdiffCudaBenchmark(coord3 size);

    // CPU implementation
    // As in hdiff_stencil_variant.h
    virtual void run();
    virtual void setup();
    virtual void post();
    
    //CudaRegularGrid3D<double> *input;
    //CudaRegularGrid3D<double> *output;
    //CudaRegularGrid3D<double> *coeff;
    //CudaRegularGrid3D<double> *lap;
    //CudaRegularGrid3D<double> *flx;
    //CudaRegularGrid3D<double> *fly;

    // Return info struct for kernels
    HdiffCudaRegular::Info get_info();

};

// IMPLEMENTATIONS

HdiffCudaBenchmark::HdiffCudaBenchmark(coord3 size)
: HdiffReferenceBenchmark(size, RegularGrid) {
    this->name = "hdiff-cuda";
}

void HdiffCudaBenchmark::run() {
    HdiffCudaRegular::kernel_direct<<<this->numblocks(), this->numthreads()>>>(
        this->get_info(),
        (dynamic_cast<CudaRegularGrid3D<double>*>(this->input))->get_gridinfo(),
        (dynamic_cast<CudaRegularGrid3D<double>*>(this->output))->get_gridinfo(),
        (dynamic_cast<CudaRegularGrid3D<double>*>(this->coeff))->get_gridinfo()
    );
    if(cudaDeviceSynchronize() != cudaSuccess) {
        this->error = true;
    }
}

void HdiffCudaBenchmark::setup() {
    this->input = new CudaRegularGrid3D<double>(this->size);
    this->output = new CudaRegularGrid3D<double>(this->size);
    this->coeff = new CudaRegularGrid3D<double>(this->size);
    this->lap = new CudaRegularGrid3D<double>(this->size);
    this->flx = new CudaRegularGrid3D<double>(this->size);
    this->fly = new CudaRegularGrid3D<double>(this->size);
    this->inner_size = this->size - 2*this->halo;
    this->HdiffReferenceBenchmark::populate_grids();
    /*int i = 0;
    for(int x=0;x<this->size.x; x++) {
        for(int y=0;y<this->size.y; y++) {
            for(int z=0;z<this->size.z; z++) {
                this->input->set(coord3(x, y, z), (double)(i*0.5));
                i++;
            }
        }
    }
    this->input->print();*/
}

HdiffCudaRegular::Info HdiffCudaBenchmark::get_info() {
    return { .halo = this->halo,
             .inner_size = this->input->dimensions-2*this->halo};
}

void HdiffCudaBenchmark::post() {
    this->Benchmark::post();
}

#endif