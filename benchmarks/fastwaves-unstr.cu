#ifndef FASTWAVES_UNSTR_H
#define FASTWAVES_UNSTR_H
#include "benchmarks/fastwaves-base.cu"
#include "grids/cuda-unstructured.cu"

namespace FastWavesUnstrBenchmarkNamespace {

    enum Variant { naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared, kloop, aos_idxvar };

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs, const int xysize,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < xysize && k < max_coord.z-1)
    #define NOT_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x >= xysize || k >= max_coord.z-1)
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define NEXT_Z_NEIGHBOR(idx) (idx+z_stride)
    #define K_STEP k*z_stride

    #include "kernels/fastwaves-naive.cu"
    #include "kernels/fastwaves-idxvar.cu"
    #include "kernels/fastwaves-idxvar-kloop.cu"
    #include "kernels/fastwaves-idxvar-kloop-sliced.cu"
    #include "kernels/fastwaves-idxvar-shared.cu"
    #include "kernels/fastwaves-kloop.cu"
    #include "kernels/fastwaves-aos-idxvar.cu"

    #undef GRID_ARGS
    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef NOT_IN_BOUNDS
    #undef NEIGHBOR
    #undef NEXT_Z_NEIGHBOR
    #undef K_STEP

}

template<typename value_t>
class FastWavesUnstrBenchmark : public FastWavesBaseBenchmark<value_t> {

    public:
    
    FastWavesUnstrBenchmark(coord3 size, FastWavesUnstrBenchmarkNamespace::Variant variant);
    
    FastWavesUnstrBenchmarkNamespace::Variant variant;

    void setup();
    void run();
    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    void parse_args();

    int *neighborships;
    int z_stride;
    int offs;
    int xysize;

    int k_per_thread = 8;
    int smem = 0;

    layout_t layout = rowmajor;

    void (*kernel)(const coord3, const int *, const int, const int, const int, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const double, const double, const int, value_t *, value_t *);

};

template<typename value_t>
FastWavesUnstrBenchmark<value_t>::FastWavesUnstrBenchmark(coord3 size, FastWavesUnstrBenchmarkNamespace::Variant variant) :
FastWavesBaseBenchmark<value_t>(size),
variant(variant) {
    if(this->variant == FastWavesUnstrBenchmarkNamespace::naive) {
        this->name = "fastwaves-unstr-naive";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
        this->name = "fastwaves-unstr-idxvar";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        this->name = "fastwaves-unstr-idxvar-kloop";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
        this->name = "fastwaves-unstr-idxvar-kloop-sliced";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_shared) {
        this->name = "fastwaves-unstr-idxvar-shared";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        this->name = "fastwaves-unstr-kloop";
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        this->name = "fastwaves-unstr-aos-idxvar";
    }
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::setup() {
    if(this->variant != FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        CudaUnstructuredGrid3D<value_t> *u_in = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo, this->layout);
        this->u_in = u_in;
        this->v_in = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->u_tens = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->v_tens = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->rho = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->ppuv = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->fx = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->wgtfac = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->hhl = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->u_out = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->v_out = CudaUnstructuredGrid3D<value_t>::clone(*u_in);

        this->neighborships = u_in->neighborships;
        this->z_stride = u_in->z_stride();
        this->offs = u_in->index(coord3(0, 0, 0)),
        this->xysize = u_in->dimensions.x*u_in->dimensions.y;

        this->ptr_ppuv = this->ppuv->data;
        this->ptr_wgtfac = this->wgtfac->data;
        this->ptr_hhl = this->hhl->data;
        this->ptr_v_in = this->v_in->data;
        this->ptr_u_in = this->u_in->data;
        this->ptr_v_tens = this->v_tens->data;
        this->ptr_u_tens = this->u_tens->data;
        this->ptr_rho = this->rho->data;
        this->ptr_fx = this->fx->data;
        this->ptr_u_out = this->u_out->data;
        this->ptr_v_out = this->v_out->data;
    } else {
        this->aos = true;
        CudaUnstructuredGrid3D<fastwaves_aos_val<value_t>> *ins = CudaUnstructuredGrid3D<fastwaves_aos_val<value_t>>::create_regular(this->inner_size, this->halo, this->layout);
        this->inputs = ins;
        CudaUnstructuredGrid3D<value_t> *u_out = CudaUnstructuredGrid3D<value_t>::create(this->inner_size, this->halo, 1, ins->neighborships);
        u_out->indices = ins->indices;
        u_out->coordinates = ins->coordinates;
        this->u_out = u_out;
        this->v_out = CudaUnstructuredGrid3D<value_t>::clone(*u_out);
        this->ptr_u_out = this->u_out->data;
        this->ptr_v_out = this->v_out->data;
        this->neighborships = ins->neighborships;
        this->z_stride = ins->z_stride();
        this->offs = ins->index(coord3(0, 0, 0)),
        this->xysize = ins->dimensions.x*ins->dimensions.y;
    }

    this->FastWavesBaseBenchmark<value_t>::setup(); // set initial values

    this->blocks = this->numblocks();
    this->threads = this->numthreads();

    this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_naive<value_t>;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar<value_t>;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_kloop<value_t>;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_shared) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_shared<value_t>;
        dim3 threads = this->threads;
        this->smem = threads.x*threads.y*FASTWAVES_IDXVAR_SHARED_SMEM_SZ_PER_THREAD*sizeof(int);
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_kloop<value_t>;
    }
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::run() {
    if(this->variant != FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced &&
       this->variant != FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        (*this->kernel)<<<this->blocks, this->threads, this->smem>>>(
            this->inner_size,
            this->neighborships, this->z_stride, this->offs, this->xysize,
            this->ptr_ppuv,
            this->ptr_wgtfac,
            this->ptr_hhl,
            this->ptr_v_in,
            this->ptr_u_in,
            this->ptr_v_tens,
            this->ptr_u_tens,
            this->ptr_rho,
            this->ptr_fx,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->ptr_u_out,
            this->ptr_v_out);
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
        FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_kloop_sliced<<<this->blocks, this->threads>>>(
            this->k_per_thread,
            this->inner_size,
            this->neighborships, this->z_stride, this->offs, this->xysize,
            this->ptr_ppuv,
            this->ptr_wgtfac,
            this->ptr_hhl,
            this->ptr_v_in,
            this->ptr_u_in,
            this->ptr_v_tens,
            this->ptr_u_tens,
            this->ptr_rho,
            this->ptr_fx,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->ptr_u_out,
            this->ptr_v_out);
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        FastWavesUnstrBenchmarkNamespace::fastwaves_aos_idxvar<<<this->blocks, this->threads>>>(
            this->inner_size,
            this->neighborships, this->z_stride, this->offs, this->xysize,
            this->inputs->data,
            this->edadlat,
            this->dt_small,
            this->c_flat_limit,
            this->ptr_u_out,
            this->ptr_v_out);
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    domain.z -= 1;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop
        || this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numthreads = this->FastWavesBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t>
dim3 FastWavesUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    domain.z -= 1;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop
       || this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        domain.z = 1;
    }  else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numblocks = this->FastWavesBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::parse_args() {
    for(int i = 0; i < this->argc; i++) {
        std::string arg = std::string(this->argv[i]);
        if(arg == "--z-curves" || arg == "-z") {
            this->layout = zcurve;
        } else if(arg == "--random" || arg == "-r") {
            this->layout = random_layout;
        } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
            sscanf(this->argv[i], "%d", &this->k_per_thread);
        }
    }
    if(this->layout == zcurve) {
        this->name.append("-z-curves");
    }
    if(this->layout == random_layout) {
        this->name.append("-random");
    }
}

#endif