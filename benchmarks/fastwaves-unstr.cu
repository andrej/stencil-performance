#ifndef FASTWAVES_UNSTR_H
#define FASTWAVES_UNSTR_H
#include <ostream>
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
    #define PROTO(x)

    #include "kernels/fastwaves-naive.cu"
    #include "kernels/fastwaves-idxvar.cu"
    #include "kernels/fastwaves-idxvar-kloop.cu"
    #include "kernels/fastwaves-idxvar-kloop-sliced.cu"
    #include "kernels/fastwaves-idxvar-shared.cu"
    #include "kernels/fastwaves-kloop.cu"
    #include "kernels/fastwaves-aos-idxvar.cu"

    #undef GRID_ARGS
    #undef NEIGHBOR
    #undef PROTO

    namespace Compressed {
        #define GRID_ARGS const int * __restrict__ prototypes, const int * __restrict__ neighborships, const int z_stride, const int neigh_stride, const int offs, const int xysize,
        #define PROTO(idx) //int idx ## _proto = prototypes[idx]
        #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_PROTO_NEIGHBOR(prototypes, neighborships, z_stride, neigh_stride, idx, x_, y_, z_)

        #include "kernels/fastwaves-naive.cu"
        #include "kernels/fastwaves-idxvar.cu"
        #include "kernels/fastwaves-idxvar-kloop.cu"
        #include "kernels/fastwaves-idxvar-kloop-sliced.cu"
        #include "kernels/fastwaves-idxvar-shared.cu"
        #include "kernels/fastwaves-kloop.cu"
        #include "kernels/fastwaves-aos-idxvar.cu"

        #undef GRID_ARGS
        #undef NEIGHBOR
        #undef PROTO
    }

    #undef INDEX
    #undef IS_IN_BOUNDS
    #undef NOT_IN_BOUNDS
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
    std::string cache_file_name();
    bool setup_from_archive(Benchmark::cache_iarchive &ar);
    void store_to_archive(Benchmark::cache_oarchive &ar);

    int *neighborships;
    int *prototypes;
    int z_stride;
    int neigh_stride;
    int offs;
    int xysize;

    int k_per_thread = 8;
    int smem = 0;

    layout_t layout = rowmajor;
    bool use_compression = false;

    void (*kernel)(const coord3, const int *, const int, const int, const int, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const double, const double, const int, value_t *, value_t *);
    void (*kernel_comp)(const coord3, const int *, const int *, const int, const int, const int, const int, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const value_t *, const double, const double, const int, value_t *, value_t *);


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
    this->FastWavesBaseBenchmark<value_t>::setup(); // set up reference
    if(this->variant != FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        CudaUnstructuredGrid3D<value_t> *u_in;
        u_in = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo, this->layout, 1, DEFAULT_Z_CURVE_WIDTH, this->use_compression);
        if(!this->setup_from_cache()) {
            this->u_in = u_in;
            if(this->use_compression) {
                u_in->compress();
            }
            this->v_in = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->u_tens = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->v_tens = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->rho = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->ppuv = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->fx = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->wgtfac = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->hhl = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
            this->populate_grids(); // set initial values
            this->store_to_cache();
        }

        this->u_out = CudaUnstructuredGrid3D<value_t>::clone(*u_in);
        this->v_out = CudaUnstructuredGrid3D<value_t>::clone(*u_in);

        this->neighborships = u_in->neighborships;
        this->prototypes = u_in->prototypes;
        this->z_stride = u_in->z_stride();
        this->neigh_stride = u_in->neigh_stride();
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

        if(!this->quiet && this->use_compression) {
            u_in->print_prototypes();
        }
    } else {
        this->aos = true;
        CudaUnstructuredGrid3D<fastwaves_aos_val<value_t>> *ins;
        ins = CudaUnstructuredGrid3D<fastwaves_aos_val<value_t>>::create_regular(this->inner_size, this->halo, this->layout);
        if(this->use_compression) {
            ins->compress();
        }
        this->populate_grids();
        this->inputs = ins;
        CudaUnstructuredGrid3D<value_t> *u_out = CudaUnstructuredGrid3D<value_t>::clone(*ins);
        this->u_out = u_out;
        this->v_out = CudaUnstructuredGrid3D<value_t>::clone(*u_out);
        this->ptr_u_out = this->u_out->data;
        this->ptr_v_out = this->v_out->data;
        this->neighborships = ins->neighborships;
        this->prototypes = ins->prototypes;
        this->z_stride = ins->z_stride();
        this->neigh_stride = ins->neigh_stride();
        this->offs = ins->index(coord3(0, 0, 0)),
        this->xysize = ins->dimensions.x*ins->dimensions.y;
    }

    this->blocks = this->numblocks();
    this->threads = this->numthreads();

    this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_naive<value_t>;
    this->kernel_comp = &FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_naive<value_t>;
    if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar<value_t>;
        this->kernel_comp = &FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_idxvar<value_t>;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_kloop<value_t>;
        this->kernel_comp = &FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_idxvar_kloop<value_t>;
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_shared) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_shared<value_t>;
        this->kernel_comp = &FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_idxvar_shared<value_t>;
        dim3 threads = this->threads;
        this->smem = threads.x*threads.y*FASTWAVES_IDXVAR_SHARED_SMEM_SZ_PER_THREAD*sizeof(int);
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::kloop) {
        this->kernel = &FastWavesUnstrBenchmarkNamespace::fastwaves_kloop<value_t>;
        this->kernel_comp = &FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_kloop<value_t>;
    }
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::run() {
    if(this->variant != FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced &&
       this->variant != FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        if(!this->use_compression) {
            (*this->kernel)<<<this->blocks, this->threads, this->smem>>>(
                this->inner_size,
                this->neighborships, this->z_stride, this->offs, this->xysize,
                this->ptr_ppuv, this->ptr_wgtfac, this->ptr_hhl, this->ptr_v_in, this->ptr_u_in, this->ptr_v_tens, this->ptr_u_tens, this->ptr_rho, this->ptr_fx,
                this->edadlat, this->dt_small, this->c_flat_limit,
                this->ptr_u_out, this->ptr_v_out);
        } else {
            (*this->kernel_comp)<<<this->blocks, this->threads, this->smem>>>(
                this->inner_size,
                this->prototypes, this->neighborships, this->z_stride, this->neigh_stride, this->offs, this->xysize,
                this->ptr_ppuv, this->ptr_wgtfac, this->ptr_hhl, this->ptr_v_in, this->ptr_u_in, this->ptr_v_tens, this->ptr_u_tens, this->ptr_rho, this->ptr_fx,
                this->edadlat, this->dt_small, this->c_flat_limit,
                this->ptr_u_out, this->ptr_v_out);
        }
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
        if(!this->use_compression) {
            FastWavesUnstrBenchmarkNamespace::fastwaves_idxvar_kloop_sliced<<<this->blocks, this->threads>>>(
                this->k_per_thread,
                this->inner_size,
                this->neighborships, this->z_stride, this->offs, this->xysize,
                this->ptr_ppuv, this->ptr_wgtfac, this->ptr_hhl, this->ptr_v_in, this->ptr_u_in, this->ptr_v_tens, this->ptr_u_tens, this->ptr_rho, this->ptr_fx,
                this->edadlat, this->dt_small, this->c_flat_limit,
                this->ptr_u_out, this->ptr_v_out);
        } else {
            FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_idxvar_kloop_sliced<<<this->blocks, this->threads>>>(
                this->k_per_thread,
                this->inner_size,
                this->prototypes, this->neighborships, this->z_stride, this->neigh_stride, this->offs, this->xysize,
                this->ptr_ppuv, this->ptr_wgtfac, this->ptr_hhl, this->ptr_v_in, this->ptr_u_in, this->ptr_v_tens, this->ptr_u_tens, this->ptr_rho, this->ptr_fx,
                this->edadlat, this->dt_small, this->c_flat_limit,
                this->ptr_u_out, this->ptr_v_out);
        }
    } else if(this->variant == FastWavesUnstrBenchmarkNamespace::aos_idxvar) {
        if(!this->use_compression) {
            FastWavesUnstrBenchmarkNamespace::fastwaves_aos_idxvar<<<this->blocks, this->threads>>>(
                this->inner_size,
                this->neighborships, this->z_stride, this->offs, this->xysize,
                this->inputs->data,
                this->edadlat, this->dt_small, this->c_flat_limit,
                this->ptr_u_out, this->ptr_v_out);
        } else {
            FastWavesUnstrBenchmarkNamespace::Compressed::fastwaves_aos_idxvar<<<this->blocks, this->threads>>>(
                this->inner_size,
                this->prototypes, this->neighborships, this->z_stride, this->neigh_stride, this->offs, this->xysize,
                this->inputs->data,
                this->edadlat, this->dt_small, this->c_flat_limit,
                this->ptr_u_out, this->ptr_v_out);
        }
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
        } else if(arg == "--compress" || arg == "-c") {
            this->use_compression = true;
        } else if(this->variant == FastWavesUnstrBenchmarkNamespace::idxvar_kloop_sliced) {
            sscanf(this->argv[i], "%d", &this->k_per_thread);
        } else {
            this->Benchmark::parse_args(); // throws error on unrecognized
        }
    }
    if(this->use_compression) {
        this->name.append("-comp");
    }
    if(this->layout == zcurve) {
        this->name.append("-z-curves");
    }
    if(this->layout == random_layout) {
        this->name.append("-random");
    }
}

template<typename value_t>
std::string FastWavesUnstrBenchmark<value_t>::cache_file_name() {
    std::ostringstream s;
    s << "fastwaves-unstr";
    if(this->use_compression) {
        s << "-comp";
    }
    if(this->layout == zcurve) {
        s << "-z-curves";
    }
    if(this->layout == random_layout) {
        s << "-random";
    }
    return s.str();
}

template<typename value_t>
bool FastWavesUnstrBenchmark<value_t>::setup_from_archive(Benchmark::cache_iarchive &ar) {
    if(this->aos) {
        return false;
    }
    auto u_in = new CudaUnstructuredGrid3D<value_t>();
    auto v_in = new CudaUnstructuredGrid3D<value_t>();
    auto u_tens = new CudaUnstructuredGrid3D<value_t>();
    auto v_tens = new CudaUnstructuredGrid3D<value_t>();
    auto rho = new CudaUnstructuredGrid3D<value_t>();
    auto ppuv = new CudaUnstructuredGrid3D<value_t>();
    auto fx = new CudaUnstructuredGrid3D<value_t>();
    auto wgtfac = new CudaUnstructuredGrid3D<value_t>();
    auto hhl = new CudaUnstructuredGrid3D<value_t>();
    ar >> *u_in;
    ar >> *v_in;
    ar >> *u_tens;
    ar >> *v_tens;
    ar >> *rho;
    ar >> *ppuv;
    ar >> *fx;
    ar >> *wgtfac;
    ar >> *hhl;
    this->u_in = u_in;
    this->v_in = v_in;
    this->u_tens = u_tens;
    this->v_tens = v_tens;
    this->rho = rho;
    this->ppuv = ppuv;
    this->fx = fx;
    this->wgtfac = wgtfac;
    this->hhl = hhl;
    return true;
}

template<typename value_t>
void FastWavesUnstrBenchmark<value_t>::store_to_archive(Benchmark::cache_oarchive &ar) {
    if(this->aos) {
        return;
    }
    auto u_in = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->u_in);
    auto v_in = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->v_in);
    auto u_tens = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->u_tens);
    auto v_tens = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->v_tens);
    auto rho = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->rho);
    auto ppuv = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->ppuv);
    auto fx = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->fx);
    auto wgtfac = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->wgtfac);
    auto hhl = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->hhl);
    ar << *u_in;
    ar << *v_in;
    ar << *u_tens;
    ar << *v_tens;
    ar << *rho;
    ar << *ppuv;
    ar << *fx;
    ar << *wgtfac;
    ar << *hhl;
}

#endif