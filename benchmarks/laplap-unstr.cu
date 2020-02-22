#ifndef LAPLAP_UNSTR_BENCHMARK_H
#define LAPLAP_UNSTR_BENCHMARK_H
#include <stdio.h>
#include <stdexcept>
#include <type_traits>
#include <cstdint>
#include <sstream>
#include <functional>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "benchmarks/laplap-base.cu"
#include "grids/cuda-unstructured.cu"
 
/** Kernels
 * Namespace containing the kernel variants that use the unstructured grid macros. */
namespace LapLapUnstr {

    enum Variant { naive, idxvar, idxvar_kloop, idxvar_kloop_sliced, idxvar_shared };

    #define neigh_ptr_t int
    #include "laplap-unstr-kernels.cu"
    #undef neigh_ptr_t

};


template<typename value_t, typename neigh_ptr_t = int>
class LapLapUnstrBenchmark : public LapLapBaseBenchmark<value_t> {

    public:
    
    LapLapUnstrBenchmark(coord3 size = coord3(1, 1, 1), LapLapUnstr::Variant variant = LapLapUnstr::naive);

    LapLapUnstr::Variant variant;

    void setup();
    void run();
    dim3 numthreads(coord3 domain=coord3());
    dim3 numblocks(coord3 domain=coord3());
    
    void parse_args();
    int k_per_thread = 16;
    bool pointer_chasing = true;

    layout_t layout = rowmajor;
    int z_curve_width = 4;
    bool use_compression = false;
    bool print_comp_info = false;

    virtual void setup_from_archive(Benchmark::cache_iarchive &ia);
    virtual void store_to_archive(Benchmark::cache_oarchive &oa);
    virtual std::string cache_file_name();

    int smem = 0;
    neigh_ptr_t *neighborships;
    neigh_ptr_t *prototypes;
    int offs;
    int z_stride;
    int neigh_stride;

    dim3 blocks;
    dim3 threads;
    value_t *input_ptr;
    value_t *output_ptr;
    void set_kernel_fun();
    void (*kernel)(const neigh_ptr_t *, const int, const int, const coord3, const value_t *, value_t *);
    void (*kernel_kloop_sliced)(const neigh_ptr_t *, const int, const int, const int, const coord3, const value_t *, value_t *);
    void (*kernel_comp)(neigh_ptr_t *, neigh_ptr_t *, const int, const int, const int, const coord3, const value_t *, value_t *);
    void (*kernel_kloop_sliced_comp)(neigh_ptr_t *, neigh_ptr_t *, const int, const int, const int, const int, const coord3, const value_t *, value_t *);
};

// Class Ids for serialization
//BOOST_CLASS_EXPORT(LapLapUnstrBenchmark<double>)

// IMPLEMENTATIONS

template<typename value_t, typename neigh_ptr_t>
LapLapUnstrBenchmark<value_t, neigh_ptr_t>::LapLapUnstrBenchmark(coord3 size, LapLapUnstr::Variant variant) :
LapLapBaseBenchmark<value_t>(size),
variant(variant) {
    this->variant = variant;
    if(variant == LapLapUnstr::naive) {
        this->name = "laplap-unstr-naive";
    } else if(variant == LapLapUnstr::idxvar) {
        this->name = "laplap-unstr-idxvar";
    } else if(variant == LapLapUnstr::idxvar_kloop) {
        this->name = "laplap-unstr-idxvar-kloop";
    } else if(variant == LapLapUnstr::idxvar_kloop_sliced) {
        this->name = "laplap-unstr-idxvar-kloop-sliced";
    } else if(variant == LapLapUnstr::idxvar_shared) {
        this->name = "laplap-unstr-idxvar-shared";
    }
}

template<typename value_t, typename neigh_ptr_t>
void LapLapUnstrBenchmark<value_t, neigh_ptr_t>::setup() {
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *input = NULL;
    if(!this->setup_from_cache()) {
        coord3 halo2(2, 2, 0);
        int neighbor_store_depth = (this->pointer_chasing ? 1 : 2);
        input = CudaUnstructuredGrid3D<value_t, neigh_ptr_t>::create_regular(this->inner_size, halo2, this->layout, neighbor_store_depth, this->z_curve_width, this->use_compression);
        this->input = input;
        input->fill_random();
        if(this->use_compression) {
            input->compress();
            if(this->print_comp_info) {
                fprintf(stderr, "XY-cells: %d. Prototypes: %d. Ratio: %4.2f%%.\n", input->z_stride(), input->neigh_stride(), input->neigh_stride()/(double)input->z_stride()*100);
                input->print_prototypes();
            }
        }
        this->store_to_cache();
    } else {
        input = dynamic_cast<CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *>(this->input);
    }
    //this->input = input;
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *output = CudaUnstructuredGrid3D<value_t, neigh_ptr_t>::clone(*input);
    this->output = output;
    if(this->variant == LapLapUnstr::idxvar_shared) {
        this->input->setSmemBankSize(sizeof(int));
    }
    this->z_stride = input->z_stride();
    this->neigh_stride = input->neigh_stride();
    this->neighborships = input->neighborships;
    this->prototypes = input->prototypes;
    this->offs = this->input->index(coord3(0, 0, 0));
    this->input_ptr = this->input->data;
    this->output_ptr = this->output->data;
    this->threads = this->numthreads();
    this->blocks = this->numblocks();
    smem = 0;
    this->set_kernel_fun();
}

template<typename value_t, typename neigh_ptr_t>
void LapLapUnstrBenchmark<value_t, neigh_ptr_t>::setup_from_archive(Benchmark::cache_iarchive &ia) {
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *input = new CudaUnstructuredGrid3D<value_t, neigh_ptr_t>();
    ia >> *input;
    this->input = input;
}

template<typename value_t, typename neigh_ptr_t>
void LapLapUnstrBenchmark<value_t, neigh_ptr_t>::store_to_archive(Benchmark::cache_oarchive &oa) {
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *input = dynamic_cast<CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *>(this->input);
    oa << *input;
}

template<typename value_t, typename neigh_ptr_t>
void LapLapUnstrBenchmark<value_t, neigh_ptr_t>::set_kernel_fun() {
    kernel = &LapLapUnstr::NonChasing::laplap_naive<value_t>;
    kernel_comp = &LapLapUnstr::Compressed::NonChasing::laplap_naive<value_t>;
    if(this->variant == LapLapUnstr::naive && this->pointer_chasing) {
        kernel = &LapLapUnstr::Chasing::laplap_naive<value_t>;
        kernel_comp = &LapLapUnstr::Compressed::Chasing::laplap_naive<value_t>;
    } else if(this->variant == LapLapUnstr::idxvar) {
        if(this->pointer_chasing) {
            kernel = &LapLapUnstr::Chasing::laplap_idxvar<value_t>;
            kernel_comp = &LapLapUnstr::Compressed::Chasing::laplap_idxvar<value_t>;
        } else {
            kernel = &LapLapUnstr::NonChasing::laplap_idxvar<value_t>;
            kernel_comp = &LapLapUnstr::Compressed::NonChasing::laplap_idxvar<value_t>;
        }
    } else if(this->variant == LapLapUnstr::idxvar_kloop) {
        if(this->pointer_chasing) {
            kernel = &LapLapUnstr::Chasing::laplap_idxvar_kloop<value_t>;
            kernel_comp = &LapLapUnstr::Compressed::Chasing::laplap_idxvar_kloop<value_t>;
        } else {
            kernel = &LapLapUnstr::NonChasing::laplap_idxvar_kloop<value_t>;
            kernel_comp = &LapLapUnstr::Compressed::NonChasing::laplap_idxvar_kloop<value_t>;
        }
    } else if(this->variant == LapLapUnstr::idxvar_shared) {
        smem = threads.x*threads.y*13*sizeof(int);
        if(this->pointer_chasing) {
            kernel = &LapLapUnstr::Chasing::laplap_idxvar_shared<value_t>;
            kernel_comp = &LapLapUnstr::Compressed::Chasing::laplap_idxvar_shared<value_t>;
        } else {
            kernel = &LapLapUnstr::NonChasing::laplap_idxvar_shared<value_t>;
            kernel_comp = &LapLapUnstr::Compressed::NonChasing::laplap_idxvar_shared<value_t>;
        }
    }
    kernel_kloop_sliced = &LapLapUnstr::Chasing::laplap_idxvar_kloop_sliced<value_t>;
    kernel_kloop_sliced_comp = &LapLapUnstr::Compressed::Chasing::laplap_idxvar_kloop_sliced<value_t>;
    if(!this->pointer_chasing) {
        kernel_kloop_sliced = &LapLapUnstr::NonChasing::laplap_idxvar_kloop_sliced<value_t>;
        kernel_kloop_sliced_comp = &LapLapUnstr::Compressed::NonChasing::laplap_idxvar_kloop_sliced<value_t>;
    }
}

template<typename value_t, typename neigh_ptr_t>
void LapLapUnstrBenchmark<value_t, neigh_ptr_t>::run() {
    if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        if(this->use_compression) {
            (*kernel_kloop_sliced_comp)<<<blocks, threads, smem>>>(
                this->prototypes,
                this->neighborships,
                this->z_stride,
                this->neigh_stride,
                this->offs,
                this->k_per_thread,
                this->inner_size,
                this->input->data,
                this->output->data
            );
        } else {
            (*kernel_kloop_sliced)<<<blocks, threads>>>(
                this->neighborships,
                this->z_stride,
                this->offs,
                this->k_per_thread,
                this->inner_size,
                this->input->data,
                this->output->data
            );
        } 
    } else {
        if(this->use_compression) {
            (*kernel_comp)<<<blocks, threads, smem>>>(
                this->prototypes,
                this->neighborships,
                this->z_stride,
                this->neigh_stride,
                this->offs,
                this->inner_size,
                this->input->data,
                this->output->data
            );
        } else {
            (*kernel)<<<blocks, threads, smem>>>(
                this->neighborships,
                this->z_stride,
                this->offs,
                this->inner_size,
                this->input->data,
                this->output->data
            );
        }
    }
    CUDA_THROW_LAST();
    CUDA_THROW( cudaDeviceSynchronize() );
}

template<typename value_t, typename neigh_ptr_t>
dim3 LapLapUnstrBenchmark<value_t, neigh_ptr_t>::numthreads(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapUnstr::idxvar_kloop) {
        domain.z = 1;
    }
    dim3 numthreads = this->LapLapBaseBenchmark<value_t>::numthreads(domain);
    return numthreads;
}

template<typename value_t, typename neigh_ptr_t>
dim3 LapLapUnstrBenchmark<value_t, neigh_ptr_t>::numblocks(coord3 domain) {
    domain = this->inner_size;
    if(this->variant == LapLapUnstr::idxvar_kloop) {
        domain.z = 1;
    } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
        domain.z = ((this->inner_size.z + this->k_per_thread - 1) / this->k_per_thread);
    }
    dim3 numblocks = this->LapLapBaseBenchmark<value_t>::numblocks(domain);
    return numblocks;
}


template<typename value_t, typename neigh_ptr_t>
void LapLapUnstrBenchmark<value_t, neigh_ptr_t>::parse_args() {
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
        } else if(arg == "--print-comp-info") {
            this->print_comp_info = true;
        } else if(arg == "--z-curve-width" && this->argc > i+1) {
            sscanf(this->argv[i+1], "%d", &this->z_curve_width);
            ++i;
        } else if(this->variant == LapLapUnstr::idxvar_kloop_sliced) {
            sscanf(this->argv[i], "%d", &this->k_per_thread);
        } else {
            this->Benchmark::parse_args();
        }
    }
    std::ostringstream s;
    if(this->use_compression) {
        s << "-comp";
    }
    if(!this->pointer_chasing) {
        s << "-no-chase";
    }
    if(this->layout == zcurve) {
        s << "-z-curves-";
        s << this->z_curve_width;
    }
    if(this->layout == random_layout) {
        s << "-random";
    }
    this->name.append(s.str());
}

template<typename value_t, typename neigh_ptr_t>
std::string LapLapUnstrBenchmark<value_t, neigh_ptr_t>::cache_file_name() {
    std::ostringstream s;
    s << "laplap-unstr";
    // don't include kernel variant in cache name so cache can be reused across variants that use same grid
    if(this->use_compression) {
        s << "-comp";
    }
    if(!this->pointer_chasing) {
        s << "-no-chase";
    }
    if(this->layout == zcurve) {
        s << "-z-curves-";
        s << this->z_curve_width;
    }
    if(this->layout == random_layout) {
        s << "-random";
    }
    return s.str();
}

#endif