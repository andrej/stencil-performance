#ifndef HDIFF_CUDA_UNSTR_H
#define HDIFF_CUDA_UNSTR_H
#include <ostream>
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-cuda-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/cuda-regular.cu"
#include "grids/cuda-unstructured.cu"

namespace HdiffCudaUnstr {

    /** Variants of this benchmark. */
    enum Variant { naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared };

    #define GRID_ARGS const int * __restrict__ neighborships, const int z_stride, const int offs,
    #define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
    #define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < (z_stride-offs) && k < max_coord.z)
    #define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
    #define K_STEP k*z_stride
    #define PROTO(x)

    namespace NonChasing {
        #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
        #include "kernels/hdiff-naive.cu"
        #undef NEIGHBOR
        #undef DOUBLE_NEIGHBOR

        #define NEIGHBOR(idx, x, y, z) GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, idx, x, y)
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1+x2), (y1+y2), (z1+z2))
        #include "kernels/hdiff-idxvar.cu"
        #include "kernels/hdiff-idxvar-kloop.cu"
        #include "kernels/hdiff-idxvar-kloop-sliced.cu"
        #include "kernels/hdiff-idxvar-shared.cu"
        #undef NEIGHBOR
        #undef DOUBLE_NEIGHBOR
    }

    namespace Chasing {
        #define CHASING
        
        #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
        #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
        #include "kernels/hdiff-naive.cu"
        #undef NEIGHBOR
        #undef DOUBLE_NEIGHBOR

        #define NEIGHBOR(idx, x, y, z) GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, idx, x, y)
        #include "kernels/hdiff-idxvar.cu"
        #include "kernels/hdiff-idxvar-kloop.cu"
        #include "kernels/hdiff-idxvar-kloop-sliced.cu"
        #include "kernels/hdiff-idxvar-shared.cu"

        #undef NEIGHBOR
        #undef CHASING
    }

    #undef PROTO
    #undef GRID_ARGS

    #define GRID_ARGS int* prototypes, int* neighborships, const int z_stride, const int neigh_stride, const int offs,
    #define PROTO(idx) int idx ## _proto = prototypes[idx]

    namespace Compressed {

        namespace NonChasing {
            #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_PROTO_NEIGHBOR(prototypes, neighborships, z_stride, neigh_stride, idx, x_, y_, z_)
            #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1)+(x2), (y1)+(y2), (z1)+(z2))
            #include "kernels/hdiff-naive.cu"
            #undef NEIGHBOR
            #undef DOUBLE_NEIGHBOR

            #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_2D_NEIGHBOR_(neighborships, neigh_stride, idx, idx ## _proto, x_, y_)
            #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1)+(x2), (y1)+(y2), (z1)+(z2))
            #include "kernels/hdiff-idxvar.cu"
            #include "kernels/hdiff-idxvar-kloop.cu"
            #include "kernels/hdiff-idxvar-kloop-sliced.cu"
            #include "kernels/hdiff-idxvar-shared.cu"
            #undef NEIGHBOR
            #undef DOUBLE_NEIGHBOR
        }

        namespace Chasing {
            #define CHASING
            
            #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_PROTO_NEIGHBOR(prototypes, neighborships, z_stride, neigh_stride, idx, x_, y_, z_)
            #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
            #include "kernels/hdiff-naive.cu"
            #undef NEIGHBOR
            #undef DOUBLE_NEIGHBOR

            #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_2D_NEIGHBOR_(neighborships, neigh_stride, idx, idx ## _proto, x_, y_)
            #include "kernels/hdiff-idxvar.cu"
            #include "kernels/hdiff-idxvar-kloop.cu"
            #include "kernels/hdiff-idxvar-kloop-sliced.cu"
            #include "kernels/hdiff-idxvar-shared.cu"
            #undef NEIGHBOR
            #undef CHASING
        }

        #undef GRID_ARGS
        #undef INDEX
        #undef IS_IN_BOUNDS
        #undef Z_NEIGHBOR
        #undef K_STEP
        #undef PROTO
    }

};

/** Cuda implementation of different variants of the horizontal diffusion
 * kernel, both for structured and unstructured grid variants.
 *
 * For the available variants, see the HdiffCuda::Variant enum. */
template<typename value_t>
class HdiffCudaUnstrBenchmark : public HdiffCudaBaseBenchmark<value_t> {

    public:

    HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant=HdiffCudaUnstr::idxvar);
    
    HdiffCudaUnstr::Variant variant;

    virtual void run();
    virtual void setup();
    virtual void teardown();
    virtual void post();
    virtual dim3 numblocks(coord3 domain=coord3());
    virtual dim3 numthreads(coord3 domain=coord3());
    virtual void parse_args();

    virtual void setup_from_archive(Benchmark::cache_iarchive &ar);
    virtual void store_to_archive(Benchmark::cache_oarchive &ar);
    virtual std::string cache_file_name();

    bool pointer_chasing = true;
    bool use_compression = false;
    int *neighborships;
    int *prototypes;
    int z_stride;
    int neigh_stride;
    int offs;

    int k_per_thread = 8;

    layout_t layout = rowmajor;

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaUnstrBenchmark<value_t>::HdiffCudaUnstrBenchmark(coord3 size, HdiffCudaUnstr::Variant variant) :
HdiffCudaBaseBenchmark<value_t>(size) {
    if(variant == HdiffCudaUnstr::naive) {
        this->name = "hdiff-unstr-naive";
    } else if(variant == HdiffCudaUnstr::idxvar) {
        this->name = "hdiff-unstr-idxvar";
    } else if(variant == HdiffCudaUnstr::idxvar_kloop) {
        this->name = "hdiff-unstr-idxvar-kloop";
    }  else if(variant == HdiffCudaUnstr::idxvar_kloop_sliced) {
        this->name = "hdiff-unstr-idxvar-kloop-sliced";
    } else if(variant == HdiffCudaUnstr::idxvar_shared) {
        this->name = "hdiff-unstr-idxvar-shared";
    }
    this->error = false;
    this->variant = variant;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::run() {
    if(this->variant != HdiffCudaUnstr::idxvar_kloop_sliced) {
        auto kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar<value_t>;
        auto kernel_fun_comp = &HdiffCudaUnstr::Compressed::NonChasing::hdiff_idxvar<value_t>;
        if(this->pointer_chasing) {
            kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar<value_t>;
            kernel_fun_comp = &HdiffCudaUnstr::Compressed::Chasing::hdiff_idxvar<value_t>;
        }
        int smem = 0;
        if(this->variant == HdiffCudaUnstr::naive) {
            if(this->pointer_chasing) {
                kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_naive<value_t>;
                kernel_fun_comp = &HdiffCudaUnstr::Compressed::Chasing::hdiff_naive<value_t>;
            } else {
                kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_naive<value_t>;
                kernel_fun_comp = &HdiffCudaUnstr::Compressed::NonChasing::hdiff_naive<value_t>;
            }
        } else if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
            if(this->pointer_chasing) {
                kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar_kloop<value_t>;
                kernel_fun_comp = &HdiffCudaUnstr::Compressed::Chasing::hdiff_idxvar_kloop<value_t>;
            } else {
                kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar_kloop<value_t>;
                kernel_fun_comp = &HdiffCudaUnstr::Compressed::NonChasing::hdiff_idxvar_kloop<value_t>;
            }
        } else if(this->variant == HdiffCudaUnstr::idxvar_shared) {
            if(this->pointer_chasing) {
                kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar_shared<value_t>;
                kernel_fun_comp = &HdiffCudaUnstr::Compressed::Chasing::hdiff_idxvar_shared<value_t>;
            } else {
                kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar_shared<value_t>;
                kernel_fun_comp = &HdiffCudaUnstr::Compressed::NonChasing::hdiff_idxvar_shared<value_t>;
            }
            dim3 numthreads = this->numthreads();
            smem = HDIFF_IDXVAR_SHARED_SMEM_SZ_PER_THREAD*numthreads.x*numthreads.y*sizeof(int);
        }
        if(!this->use_compression) {
            (*kernel_fun)<<<this->numblocks(), this->numthreads(), smem>>>(
                this->inner_size,
                this->neighborships, this->z_stride, this->offs,
                this->input->data,
                this->output->data,
                this->coeff->data
            );
        } else {
            (*kernel_fun_comp)<<<this->numblocks(), this->numthreads(), smem>>>(
                this->inner_size,
                this->prototypes, this->neighborships, this->z_stride, this->neigh_stride, this->offs,
                this->input->data,
                this->output->data,
                this->coeff->data
            );
        }
    } else {
        auto kernel_fun = &HdiffCudaUnstr::NonChasing::hdiff_idxvar_kloop_sliced<value_t>;
        auto kernel_fun_comp = &HdiffCudaUnstr::Compressed::NonChasing::hdiff_idxvar_kloop_sliced<value_t>;
        if(this->pointer_chasing) {
            kernel_fun = &HdiffCudaUnstr::Chasing::hdiff_idxvar_kloop_sliced<value_t>;
            kernel_fun_comp = &HdiffCudaUnstr::Compressed::Chasing::hdiff_idxvar_kloop_sliced<value_t>;
        }
        if(!this->use_compression) {
            (*kernel_fun)<<<this->numblocks(), this->numthreads()>>>(
                this->k_per_thread,
                this->inner_size,
                this->neighborships, this->z_stride, this->offs,
                this->input->data,
                this->output->data,
                this->coeff->data
            );
        } else {
            (*kernel_fun_comp)<<<this->numblocks(), this->numthreads()>>>(
                this->k_per_thread,
                this->inner_size,
                this->prototypes, this->neighborships, this->z_stride, this->neigh_stride, this->offs,
                this->input->data,
                this->output->data,
                this->coeff->data
            );
        }
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    // For the vriants that use a k-loop inside the kernel, we only need one block in the k-direction
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == HdiffCudaUnstr::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numblocks = this->Benchmark::numblocks(domain);
    return numblocks;
}

template<typename value_t>
dim3 HdiffCudaUnstrBenchmark<value_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    // Variants with a k-loop: only one thread in the k-direction
    if(this->variant == HdiffCudaUnstr::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == HdiffCudaUnstr::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numthreads = this->Benchmark::numthreads(domain);
    return numthreads;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::setup() {
    this->HdiffCudaBaseBenchmark<value_t>::setup(); // set up reference benchmark
    if(this->variant == HdiffCudaUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    CudaUnstructuredGrid3D<value_t> *input;
    CudaUnstructuredGrid3D<value_t> *coeff;
    if(!this->setup_from_cache()) {
        int neighbor_store_depth = (this->pointer_chasing ? 1 : 2);
        input = CudaUnstructuredGrid3D<value_t>::create_regular(this->inner_size, this->halo, this->layout, neighbor_store_depth, DEFAULT_Z_CURVE_WIDTH, this->use_compression);
        if(this->use_compression) {
            input->compress();
            if(!this->quiet) {
                input->print_prototypes();
            }
        }
        coeff = CudaUnstructuredGrid3D<value_t>::clone(*input);
        this->input = input;
        this->coeff = coeff;
        this->populate_grids();
        this->store_to_cache();
    } else {
        input = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input);
        coeff = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input);
    }
    // the following member assignments have to happen after base class setup
    // grids might come from serialized file read in base class setup()
    coeff->link(input);
    this->output = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->lap = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->flx = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->fly = CudaUnstructuredGrid3D<value_t>::clone(*input);
    this->neighborships = input->neighborships;
    this->prototypes = input->prototypes;
    this->z_stride = input->z_stride();
    this->neigh_stride = input->neigh_stride();
    this->offs = input->index(coord3(0, 0, 0));
    
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::setup_from_archive(Benchmark::cache_iarchive &ar) {
    auto input = new CudaUnstructuredGrid3D<value_t>(); //dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input);
    auto coeff = new CudaUnstructuredGrid3D<value_t>(); //dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->coeff);
    ar >> *input;
    ar >> *coeff;
    this->input = input;
    this->coeff = coeff;
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::store_to_archive(Benchmark::cache_oarchive &ar) {
    auto input = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->input);
    auto coeff = dynamic_cast<CudaUnstructuredGrid3D<value_t> *>(this->coeff);
    ar << *input;
    ar << *coeff;
}

template<typename value_t>
std::string HdiffCudaUnstrBenchmark<value_t>::cache_file_name() {
    std::ostringstream s;
    s << "hdiff-unstr";
    if(!this->pointer_chasing) {
        s << "-no-chase";
    }
    if(this->layout == zcurve) {
        s << "-z-curves";
    }
    if(this->layout == random_layout) {
        s << "-random";
    }
    if(this->use_compression) {
        s << "-comp";
    }
    return s.str();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::teardown() {
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffCudaBaseBenchmark<value_t>::teardown();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::post() {
    this->Benchmark::post();
    this->HdiffCudaBaseBenchmark<value_t>::post();
}

template<typename value_t>
void HdiffCudaUnstrBenchmark<value_t>::parse_args() {
    for(int i = 0; i < this->argc; i++) {
        std::string arg = std::string(this->argv[i]);
        if(arg == "--z-curves" || arg == "-z") {
            this->layout = zcurve;
        } else if(arg == "--random" || arg == "-r") {
            this->layout = random_layout;
        } else if(arg == "--no-chase" || arg == "-n") {
            this->pointer_chasing = false;
        } else if(arg == "--compress" || arg == "-c") {
            this->use_compression = true;
        } else if(this->variant == HdiffCudaUnstr::idxvar_kloop_sliced) {
            sscanf(this->argv[i], "%d", &this->k_per_thread);
        } else {
            this->Benchmark::parse_args();
        }
    }
    if(this->use_compression) {
        this->name.append("-comp");
    }
    if(!this->pointer_chasing) {
        this->name.append("-no-chase");
    }
    if(this->layout == zcurve) {
        this->name.append("-z-curves");
    }
    if(this->layout == random_layout) {
        this->name.append("-random");
    }
}

#endif