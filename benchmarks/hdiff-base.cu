#ifndef HDIFF_BASE_H
#define HDIFF_BASE_H
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-ref.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/regular.cu"
#include "grids/unstructured.cu"


/** Base class for horizontal diffusion benchmarks. Provides verification
 * against reference benchmark and "halo" functionality, i.e. padding the
 * coordinate space on its sides to prevent out of bounds accesses. */
class HdiffBaseBenchmark :  public Benchmark<double> {

    public:

    HdiffBaseBenchmark(coord3 size);

    Grid<double, coord3> *coeff = NULL;
    Grid<double, coord3> *lap = NULL;
    Grid<double, coord3> *flx = NULL;
    Grid<double, coord3> *fly = NULL;

    // reference grids used for verification
    // in debug mode, we keep a grid for each intermediate step to aid
    // debugging (at which step do we go wrong)
    HdiffReferenceBenchmark *reference_bench = NULL;
    bool reference_calculated = false;
 
    // Setup Input values
    // As in hdiff_stencil_variant.h
    virtual void setup();
    virtual void teardown();
    virtual void post();

    // CPU implementation
    // As in hdiff_stencil_variant.h
    void calc_ref();

    // halo around the input data, padding that is not touched
    coord3 halo;
    coord3 inner_size; // size w.o. 2* halo
    coord3 inner_coord(coord3 inner_coord);

};

// IMPLEMENTATIONS

HdiffBaseBenchmark::HdiffBaseBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2,2,0)) {
    this->name = "hdiff";
}

void HdiffBaseBenchmark::setup(){
    if(!this->reference_bench) {
        this->reference_bench = new HdiffReferenceBenchmark(this->size);
        this->reference_bench->setup();
    }
    /* Import values from reference grids to ensure same conditions. */
    this->input->import(this->reference_bench->input);
    this->coeff->import(this->reference_bench->coeff);
    this->output->import(this->reference_bench->output);
    this->inner_size = this->size - 2*this->halo;
}

void HdiffBaseBenchmark::teardown() {
    this->reference_bench->teardown();
}

coord3 HdiffBaseBenchmark::inner_coord(coord3 coord){
    return coord + this->halo;
}

void HdiffBaseBenchmark::calc_ref() {
    if(!this->reference_calculated) {
        this->reference_bench->run();
        this->reference_calculated = true;
    }
}

void HdiffBaseBenchmark::post() {
    if(!this->reference_calculated) {
        this->calc_ref();
    }
    if(!this->verify(this->reference_bench->output)) {
        this->error = true;
    }
    #ifdef HDIFF_DEBUG
    if(this->error) {
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