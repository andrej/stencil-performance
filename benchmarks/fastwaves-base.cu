#ifndef FASTWAVES_BASE_H
#define FASTWAVES_BASE_H
#include "grids/cuda-base.cu"
#include "benchmarks/benchmark.cu"
#include "benchmarks/fastwaves-ref.cu"

/** Values stored in *one* grid, used for array of structs implementations. */
template<typename value_t>
struct fastwaves_aos_val {
    value_t u_in = 0;
    value_t v_in = 0;
    value_t u_tens = 0;
    value_t v_tens = 0;
    value_t rho = 0;
    value_t ppuv = 0;
    value_t fx = 0;
    value_t wgtfac = 0;
    value_t hhl = 0;
};

/** Values stored in *one* grid, used for array of struct of arrays implementations.
 * Combines cache locality with AoS + coalescing of 32bit instructions (8 byte double * 4 = 32 bits). */
template<typename value_t>
struct fastwaves_aosoa_val {
    value_t u_in[4];
    value_t v_in[4];
    value_t u_tens[4];
    value_t v_tens[4];
    value_t rho[4];
    value_t ppuv[4];
    value_t fx[4];
    value_t wgtfac[4];
    value_t hhl[4];
};

/** Base Class for Fastwaves benchmarks */
template<typename value_t>
class FastWavesBaseBenchmark : public Benchmark {

    public:

    FastWavesBaseBenchmark(coord3 size);
    
    coord3 halo;
    coord3 inner_size;

    bool aos = false; // Use array of structs instead of struct of arrays (better cache locality)
    bool aosoa = false;

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

    // Alternatively to the above, for AoS
    CudaBaseGrid<fastwaves_aos_val<value_t>, coord3> *inputs = NULL;


    // Outputs
    CudaBaseGrid<value_t, coord3> *u_out = NULL;
    CudaBaseGrid<value_t, coord3> *v_out = NULL;

    // Entry pointers to the (0, 0, 0) value of above grids for kernels
    value_t *ptr_ppuv;
    value_t *ptr_wgtfac;
    value_t *ptr_hhl;
    value_t *ptr_v_in;
    value_t *ptr_u_in;
    value_t *ptr_v_tens;
    value_t *ptr_u_tens;
    value_t *ptr_rho;
    value_t *ptr_fx;
    value_t *ptr_u_out;
    value_t *ptr_v_out;

    FastWavesRefBenchmark<value_t> *reference_benchmark = NULL;
    bool reference_calculated = false;
    
    virtual void setup();
    virtual void populate_grids();
    virtual void teardown();
    virtual void pre();
    virtual void post();
    bool verify(double tol=1e-4);
    
    dim3 threads;
    dim3 blocks;

};

// IMPLEMENTATIONS

template<typename value_t>
FastWavesBaseBenchmark<value_t>::FastWavesBaseBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2, 2, 2)),
c_flat_limit(0), // FIXME 10
dt_small(10),
edadlat(1) {
    this->size = size;
    this->inner_size = this->size - 2*this->halo;
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::setup() {
    if(!this->reference_benchmark && this->do_verify || this->aos || this->aosoa) {
        this->reference_benchmark = new FastWavesRefBenchmark<value_t>(this->size);
        this->reference_benchmark->c_flat_limit = this->c_flat_limit;
        this->reference_benchmark->dt_small = this->dt_small;
        this->reference_benchmark->edadlat = this->edadlat;
        this->reference_benchmark->setup();
    }
    this->Benchmark::setup();
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::populate_grids() {
        // Reset/ Import values
    if(this->aos) {
        for(int z = -this->halo.z; z < this->inner_size.z + this->halo.z; z++) {
            for(int y = -this->halo.y; y < this->inner_size.y + this->halo.y; y++) {
                for(int x = -this->halo.x; x < this->inner_size.x + this->halo.x; x++) {
                    coord3 pos(x, y, z);
                    struct fastwaves_aos_val<value_t> v = {
                        .u_in = (*this->reference_benchmark->u_pos)[pos],
                        .v_in = (*this->reference_benchmark->v_pos)[pos],
                        .u_tens = (*this->reference_benchmark->u_tens)[pos],
                        .v_tens = (*this->reference_benchmark->v_tens)[pos],
                        .rho = (*this->reference_benchmark->rho)[pos],
                        .ppuv = (*this->reference_benchmark->ppuv)[pos],
                        .fx = (*this->reference_benchmark->fx)[pos],
                        .wgtfac = (*this->reference_benchmark->wgtfac)[pos],
                        .hhl = (*this->reference_benchmark->hhl)[pos]
                    };
                    this->inputs->set(pos, v);
                }
            }
        }
    } else {
        // Inputs / Constants
        if(this->do_verify) {
            this->u_in->import(this->reference_benchmark->u_pos);
            this->v_in->import(this->reference_benchmark->v_pos);
            this->u_tens->import(this->reference_benchmark->u_tens);
            this->v_tens->import(this->reference_benchmark->v_tens);
            this->rho->import(this->reference_benchmark->rho);
            this->ppuv->import(this->reference_benchmark->ppuv);
            this->fx->import(this->reference_benchmark->fx);
            this->wgtfac->import(this->reference_benchmark->wgtfac);
            this->hhl->import(this->reference_benchmark->hhl);
        } else {
            // Importing is expensive. If we do not verify, simply create some random values
            // they need not be the same as some reference grid, as there is no reference
            this->u_in->fill_random();
            this->v_in->fill_random();
            this->u_tens->fill_random();
            this->v_tens->fill_random();
            this->rho->fill_random();
            this->ppuv->fill_random();
            this->fx->fill_random();
            this->wgtfac->fill_random();
            this->hhl->fill_random();
        }
    }

    // Reset Outputs
    //this->u_out->fill(0.0);
    //this->v_out->fill(0.0);
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::teardown() {
    if(this->reference_benchmark) {
        this->reference_benchmark->teardown();
    }
    delete this->u_in;
    delete this->v_in;
    delete this->u_tens;
    delete this->v_tens;
    delete this->rho;
    delete this->ppuv;
    delete this->fx;
    delete this->wgtfac;
    delete this->hhl;
    delete this->u_out;
    delete this->v_out;
    this->Benchmark::teardown();
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::pre() {
    // Inputs / Constants
    if(!this->aos) {
        this->u_in->prefetchToDevice();
        this->v_in->prefetchToDevice();
        this->u_tens->prefetchToDevice();
        this->v_tens->prefetchToDevice();
        this->rho->prefetchToDevice();
        this->ppuv->prefetchToDevice();
        this->fx->prefetchToDevice();
        this->wgtfac->prefetchToDevice();
        this->hhl->prefetchToDevice();
    } else {
        this->inputs->prefetchToDevice();
    }

    // Outputs
    this->u_out->prefetchToDevice();
    this->v_out->prefetchToDevice();

    this->Benchmark::pre();
}

template<typename value_t>
void FastWavesBaseBenchmark<value_t>::post() {    
    // Outputs
    this->u_out->prefetchToHost();
    this->v_out->prefetchToHost();

    this->Benchmark::post();
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

#endif