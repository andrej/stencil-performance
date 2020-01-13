#ifndef FASTWAVES_BASE_H
#define FASTWAVES_BASE_H
#include "grids/cuda-base.cu"
#include "benchmarks/benchmark.cu"
#include "benchmarks/fastwaves-ref.cu"

namespace FastWavesBenchmark {
    struct Info {
        coord3 max_coord;
    };
}

/** Base Class for Fastwaves benchmarks */
template<typename value_t>
class FastWavesBaseBenchmark : public Benchmark {

    public:

    FastWavesBaseBenchmark(coord3 size);
    
    const coord3 halo;
    coord3 inner_size;

    const int c_flat_limit;
    const int dt_small;
    const int edadlat;

    // Inputs / Constants
    CudaBaseGrid<value_t, coord3> *u_in = NULL;
    CudaBaseGrid<value_t, coord3> *v_in = NULL;
    CudaBaseGrid<value_t, coord3> *u_tens = NULL;
    CudaBaseGrid<value_t, coord3> *v_tens = NULL;
    CudaBaseGrid<value_t, coord3> *rho = NULL;
    CudaBaseGrid<value_t, coord3> *ppuv = NULL;
    CudaBaseGrid<value_t, coord3> *fx = NULL;
    CudaBaseGrid<value_t, coord3> *wgtfac = NULL;
    CudaBaseGrid<value_t, coord3> *hhl = NULL;

    // Intermediate Results
    CudaBaseGrid<value_t, coord3> *ppgk = NULL;
    CudaBaseGrid<value_t, coord3> *ppgc = NULL;
    CudaBaseGrid<value_t, coord3> *ppgu = NULL;
    CudaBaseGrid<value_t, coord3> *ppgv = NULL;

    // Outputs
    CudaBaseGrid<value_t, coord3> *u_out = NULL;
    CudaBaseGrid<value_t, coord3> *v_out = NULL;

    FastWavesRefBenchmark<value_t> *reference_benchmark = NULL;
    bool reference_calculated = false;
    
    virtual void setup();
    virtual void teardown();
    virtual void pre();
    virtual void post();
    bool verify(double tol=1e-4);

    FastWavesBenchmark::Info get_info();

};

// IMPLEMENTATIONS

template<typename value_t>
FastWavesBaseBenchmark<value_t>::FastWavesBaseBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2, 2, 2)),
c_flat_limit(0), // FIXME 10
dt_small(10),
edadlat(1) {}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::setup() {
    this->inner_size = this->size - 2*this->halo;
    if(this->inner_size.x < 0 || this->inner_size.y < 0 || this->inner_size.z < 0) {
        throw std::runtime_error("Grid too small for this kernel.");
    }

    if(!this->reference_benchmark) {
        this->reference_benchmark = new FastWavesRefBenchmark<value_t>(this->size);
        this->reference_benchmark->c_flat_limit = this->c_flat_limit;
        this->reference_benchmark->dt_small = this->dt_small;
        this->reference_benchmark->edadlat = this->edadlat;
        this->reference_benchmark->setup();
    }

    // Import values
    // Inputs / Constants
    this->u_in->import(this->reference_benchmark->u_pos);
    this->v_in->import(this->reference_benchmark->v_pos);
    this->u_tens->import(this->reference_benchmark->u_tens);
    this->v_tens->import(this->reference_benchmark->v_tens);
    this->rho->import(this->reference_benchmark->rho);
    this->ppuv->import(this->reference_benchmark->ppuv);
    this->fx->import(this->reference_benchmark->fx);
    this->wgtfac->import(this->reference_benchmark->wgtfac);
    this->hhl->import(this->reference_benchmark->hhl);

    // Intermediate Results (not all subclasses may need these)
    if(this->ppgk) {
        this->ppgk->fill(0.0);
    }
    if(this->ppgc) {
        this->ppgc->fill(0.0);
    }
    if(this->ppgu) {
        this->ppgu->fill(0.0);
    }
    if(this->ppgv) {
        this->ppgv->fill(0.0);
    }

    // Outputs
    this->u_out->fill(0.0);
    this->v_out->fill(0.0);

    this->Benchmark::setup();
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::teardown() {
    this->reference_benchmark->teardown();
    delete this->u_in;
    delete this->v_in;
    delete this->u_tens;
    delete this->v_tens;
    delete this->rho;
    delete this->ppuv;
    delete this->fx;
    delete this->wgtfac;
    delete this->hhl;
    delete this->ppgk;
    delete this->ppgc;
    delete this->ppgu;
    delete this->ppgv;
    delete this->u_out;
    delete this->v_out;
    this->Benchmark::teardown();
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::pre() {
    // Inputs / Constants
    this->u_in->prefetchToDevice();
    this->v_in->prefetchToDevice();
    this->u_tens->prefetchToDevice();
    this->v_tens->prefetchToDevice();
    this->rho->prefetchToDevice();
    this->ppuv->prefetchToDevice();
    this->fx->prefetchToDevice();
    this->wgtfac->prefetchToDevice();
    this->hhl->prefetchToDevice();

    // Intermediate Results
    if(this->ppgk) {
        this->ppgk->prefetchToDevice();
    }
    if(this->ppgc) {
        this->ppgc->prefetchToDevice();
    }
    if(this->ppgu) {
         this->ppgu->prefetchToDevice();
    }
    if(this->ppgv) {
        this->ppgv->prefetchToDevice();
    }

    // Outputs
    this->u_out->prefetchToDevice();
    this->v_out->prefetchToDevice();
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::post() {    
    // Outputs
    this->u_out->prefetchToHost();
    this->v_out->prefetchToHost();
}

template<typename value_t>
bool FastWavesBaseBenchmark<value_t>::verify(double tol) {
    if(!this->reference_calculated) {
        this->reference_benchmark->run();
        this->reference_calculated = true;
    }
    bool ret = this->u_out->compare(this->reference_benchmark->u_ref, tol) &&
               this->v_out->compare(this->reference_benchmark->v_ref, tol);
    if(!ret && !this->quiet) {
        fprintf(stderr, "Reference -------------------------------------------------\n");
        this->reference_benchmark->u_ref->print();
        fprintf(stderr, "Output ----------------------------------------------------\n");
        this->u_out->print();
    }
    return ret;
}

template<typename value_t>
FastWavesBenchmark::Info FastWavesBaseBenchmark<value_t>::get_info() {
    return { .max_coord = this->inner_size + this->halo };
}

#endif