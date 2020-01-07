#ifndef FASTWAVES_BASE_H
#define FASTWAVES_BASE_H
#include "grids/cuda-base.cu"
#include "benchmarks/benchmark.cu"
#include "benchmarks/fastwaves-ref.cu"

namespace FastWavesBenchmark {
    struct Info {
        coord3 halo;
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
    CudaBaseGrid<value_t, coord3> *u_in;
    CudaBaseGrid<value_t, coord3> *v_in;
    CudaBaseGrid<value_t, coord3> *u_tens;
    CudaBaseGrid<value_t, coord3> *v_tens;
    CudaBaseGrid<value_t, coord3> *rho;
    CudaBaseGrid<value_t, coord3> *ppuv;
    CudaBaseGrid<value_t, coord3> *fx;
    CudaBaseGrid<value_t, coord3> *wgtfac;
    CudaBaseGrid<value_t, coord3> *hhl;

    // Intermediate Results
    CudaBaseGrid<value_t, coord3> *ppgk;
    CudaBaseGrid<value_t, coord3> *ppgc;
    CudaBaseGrid<value_t, coord3> *ppgu;
    CudaBaseGrid<value_t, coord3> *ppgv;

    // Outputs
    CudaBaseGrid<value_t, coord3> *u_out;
    CudaBaseGrid<value_t, coord3> *v_out;

    FastWavesRefBenchmark<value_t> *reference_benchmark = NULL;
    bool reference_calculated = false;
    
    virtual void setup();
    virtual void teardown();
    virtual void pre();
    virtual void post();
    bool verify(double tol=1e-5);

    FastWavesBenchmark::Info get_info();

};

// IMPLEMENTATIONS

template<typename value_t>
FastWavesBaseBenchmark<value_t>::FastWavesBaseBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2, 2, 2)),
c_flat_limit(0),//c_flat_limit(10),
dt_small(10),
edadlat(1) {}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::setup() {
    this->inner_size = this->size - 2*this->halo;

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
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::teardown() {
    this->reference_benchmark->teardown();
    delete u_in;
    delete v_in;
    delete u_tens;
    delete v_tens;
    delete rho;
    delete ppuv;
    delete fx;
    delete wgtfac;
    delete hhl;
    delete ppgk;
    delete ppgc;
    delete ppgu;
    delete ppgv;
    delete u_out;
    delete v_out;
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
        this->reference_benchmark->ppgradcor->print();
        fprintf(stderr, "Output ----------------------------------------------------\n");
        this->ppgc->print();
    }
    return ret;
}

template<typename value_t>
FastWavesBenchmark::Info FastWavesBaseBenchmark<value_t>::get_info() {
    return { .halo = this->halo,
             .max_coord = this->inner_size + this->halo };
}

#endif