#ifndef HDIFF_CUDA_BASE_H
#define HDIFF_CUDA_BASE_H
#include <map>
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-ref.cu"
#include "coord3.cu"
#include "grids/cuda-base.cu"
#include "grids/grid.cu"
#include "grids/regular.cu"
#include "grids/unstructured.cu"

namespace HdiffCudaBase {
    /** Information about this benchmark for use in the kernels. */
    struct Info {
        coord3 halo;
        /** Maximum coordinates, i.e. inner_size.x+halo.x, etc. */
        coord3 max_coord;

        #ifndef HDIFF_NO_GRIDSTRIDE
        /** Size without the halo. */
        coord3 inner_size;
        /** Size of the entire grid in each dimension, i.e. number of blocks
            * times number of threads. If the thread-grid is smaller than the
            * data-grid, each kernel execution needs to handle multiple cells to
            * cover the entire data set! */
        coord3 gridsize;
        #endif
    };
}

/** Base class for horizontal diffusion benchmarks. Provides verification
 * against reference benchmark and "halo" functionality, i.e. padding the
 * coordinate space on its sides to prevent out of bounds accesses. */
template<typename value_t>
class HdiffCudaBaseBenchmark :  public Benchmark {

    public:

    HdiffCudaBaseBenchmark(coord3 size);

    CudaBaseGrid<value_t, coord3> *input = NULL;
    CudaBaseGrid<value_t, coord3> *output = NULL;
    CudaBaseGrid<value_t, coord3> *coeff = NULL;
    CudaBaseGrid<value_t, coord3> *lap = NULL;
    CudaBaseGrid<value_t, coord3> *flx = NULL;
    CudaBaseGrid<value_t, coord3> *fly = NULL;

    // reference grids used for verification
    // in debug mode, we keep a grid for each intermediate step to aid
    // debugging (at which step do we go wrong)
    HdiffReferenceBenchmark<value_t> *reference_bench = NULL;
    bool reference_calculated = false;
    
    // Cache for reference benchmarks
    // maps size to reference benchmark, so that if a reference has been
    // calculated for a given size before, it will be reused instead of
    // reacalculated (so the benchmarks run faster)
    static std::map<coord3, HdiffReferenceBenchmark<value_t> *> *reference_benchs;

    // Setup Input values
    // As in hdiff_stencil_variant.h
    virtual void setup();
    virtual void teardown();
    virtual void pre();
    virtual void post();

    // CPU implementation
    // As in hdiff_stencil_variant.h
    void calc_ref();

    // halo around the input data, padding that is not touched
    coord3 halo;
    coord3 inner_size; // size w.o. 2* halo
    coord3 inner_coord(coord3 inner_coord);

    // return information for the use inside the kernels
    HdiffCudaBase::Info get_info();


};

template<typename value_t>
std::map<coord3, HdiffReferenceBenchmark<value_t> *> *HdiffCudaBaseBenchmark<value_t>::reference_benchs = NULL;//new std::map<coord3, HdiffReferenceBenchmark<value_t> *>();

// IMPLEMENTATIONS

template<typename value_t>
HdiffCudaBaseBenchmark<value_t>::HdiffCudaBaseBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2,2,0)) {
    this->name = "hdiff";
    if(!HdiffCudaBaseBenchmark<value_t>::reference_benchs) {
        HdiffCudaBaseBenchmark<value_t>::reference_benchs = new std::map<coord3, HdiffReferenceBenchmark<value_t> *>();
    }
}

template<typename value_t>
void HdiffCudaBaseBenchmark<value_t>::setup(){
    if(!this->reference_bench) {
        if(HdiffCudaBaseBenchmark<value_t>::reference_benchs->count(this->size) > 0) {
            // already calculated in cache
            this->reference_bench = (*HdiffCudaBaseBenchmark<value_t>::reference_benchs)[this->size];
            this->reference_calculated = true;
        } else {
            this->reference_bench = new HdiffReferenceBenchmark<value_t>(this->size);
            this->reference_bench->setup();
        }
    }
    /* Import values from reference grids to ensure same conditions. */
    this->input->import(this->reference_bench->input);
    this->coeff->import(this->reference_bench->coeff);
    this->output->fill(0.0);
    this->inner_size = this->size - 2*this->halo;
}

template<typename value_t>
void HdiffCudaBaseBenchmark<value_t>::teardown() {
    if(this->reference_bench && !this->do_verify) {
        this->reference_bench->teardown();
        delete this->reference_bench;
        this->reference_bench = NULL;
        (*HdiffCudaBaseBenchmark<value_t>::reference_benchs).erase(this->size);
    }
    // Don't free, because this reference benchmark will be reused.
    // This is ugly but not important enough to fix right now. If the memory 
    // leak becomes an issue, simply run gridbenchmark with --no-verify option.
}

template<typename value_t>
coord3 HdiffCudaBaseBenchmark<value_t>::inner_coord(coord3 coord){
    return coord + this->halo;
}

template<typename value_t>
HdiffCudaBase::Info HdiffCudaBaseBenchmark<value_t>::get_info() {
    coord3 inner_size = this->inner_size;
    dim3 numthreads = this->numthreads();
    dim3 numblocks = this->numblocks();
    return { .halo = this->halo,
             .max_coord = inner_size + this->halo
             #ifndef HDIFF_NO_GRIDSTRIDE
             , .inner_size = inner_size
             , .gridsize = coord3(numblocks.x*numthreads.x, numblocks.y*numthreads.y, numblocks.z*numthreads.z) 
             #endif
            };
}

template<typename value_t>
void HdiffCudaBaseBenchmark<value_t>::calc_ref() {
    this->reference_bench->run();
    if(this->reference_bench->error) {
        return;
    }
    this->reference_calculated = true;
    (*HdiffCudaBaseBenchmark<value_t>::reference_benchs)[this->size] = this->reference_bench;
}

template<typename value_t>
void HdiffCudaBaseBenchmark<value_t>::pre() {
    this->input->prefetchToDevice();
    this->coeff->prefetchToDevice();
    // to prevent page faults, even memory addresses that are only written to need to be prefetched apparently (see tests/unified-test.cu)
    this->output->prefetchToDevice();
    #ifdef HDIFF_DEBUG
    this->lap->prefetchToDevice();
    this->flx->prefetchToDevice();
    this->fly->prefetchToDevice();
    #endif
}

template<typename value_t>
void HdiffCudaBaseBenchmark<value_t>::post() {
    if(!this->do_verify) {
        return;
    }
    // Verify output
    this->output->prefetchToHost();
    if(!this->reference_calculated) {
        this->calc_ref();
    }
    if(!this->verify(this->reference_bench->output, this->output)) {
        this->error = true;
    }
    // Verify intermediate results
    #ifdef HDIFF_DEBUG
    this->lap->prefetchToHost();
    this->flx->prefetchToHost();
    this->fly->prefetchToHost();
    if(this->error && !this->quiet) {
        fprintf(stderr, "\n==============\n%s\n", this->name.c_str());
        if(!this->verify(this->reference_bench->lap, this->lap)) {
            fprintf(stderr, "\n--- lap wrong ---\n");
            this->lap->print();
            fprintf(stderr, "\n--- lap reference ---\n");
            this->reference_bench->lap->print();
        }
        if(!this->verify(this->reference_bench->flx, this->flx)) {
            fprintf(stderr, "\n--- flx wrong ---\n");
            this->flx->print();
            fprintf(stderr, "\n--- flx reference ---\n");
            this->reference_bench->flx->print();
        }
        if(!this->verify(this->reference_bench->fly, this->fly)) {
            fprintf(stderr, "\n--- fly wrong ---\n");
            this->fly->print();
            fprintf(stderr, "\n--- fly reference ---\n");
            this->reference_bench->fly->print();
        }
        if(!this->verify(this->reference_bench->output, this->output)) {
            fprintf(stderr, "\n--- out wrong ---\n");
            this->output->print();
            fprintf(stderr, "\n--- out reference ---\n");
            this->reference_bench->output->print();
        }
    }
    #endif
}

#endif