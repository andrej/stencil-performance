#ifndef HDIFF_CPU_UNSTR_H
#define HDIFF_CPU_UNSTR_H
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-base.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/regular.cu"
#include "grids/unstructured.cu"


/** This is a CPU-version of the hdiff reference implementation that runs on
* top of a unstructured grid and respects its neighborship relations. */
class HdiffCPUUnstrBenchmark :  public HdiffBaseBenchmark {

    public:

    HdiffCPUUnstrBenchmark(coord3 size);
 
    // Setup Input values
    // As in hdiff_stencil_variant.h
    virtual void setup();
    virtual void run();
    virtual void teardown();
    
    // This is CPU so just print 1 for those values
    virtual dim3 numblocks();
    virtual dim3 numthreads();

    // CPU implementation
    // As in hdiff_stencil_variant.h
    void calc_ref();

};

// IMPLEMENTATIONS

HdiffCPUUnstrBenchmark::HdiffCPUUnstrBenchmark(coord3 size) :
HdiffBaseBenchmark(size) {
    this->name = "hdiff-unstr-cpu";
}

void HdiffCPUUnstrBenchmark::setup(){
    this->input = UnstructuredGrid3D<double>::create_regular(this->size);
    this->coeff = UnstructuredGrid3D<double>::create_regular(this->size);
    this->output = UnstructuredGrid3D<double>::create_regular(this->size);
    this->lap = UnstructuredGrid3D<double>::create_regular(this->size);
    this->flx = UnstructuredGrid3D<double>::create_regular(this->size);
    this->fly = UnstructuredGrid3D<double>::create_regular(this->size);
    this->HdiffBaseBenchmark::setup(); /**< Fills input, output and coeff and also sets up reference benchmark. */
}

void HdiffCPUUnstrBenchmark::teardown() {
    this->input->deallocate();
    this->coeff->deallocate();
    this->output->deallocate();
    this->lap->deallocate();
    this->flx->deallocate();
    this->fly->deallocate();
    delete this->input;
    delete this->coeff;
    delete this->output;
    delete this->lap;
    delete this->flx;
    delete this->fly;
    this->HdiffBaseBenchmark::teardown();
}

// Same as calc ref, but uses neighbors relations instead of directly indexing
void HdiffCPUUnstrBenchmark::run() {
    const int isize = this->inner_size.x;
    const int jsize = this->inner_size.y;
    const int ksize = this->inner_size.z;
    double *in = this->input->data;
    double *out_ref = this->output->data;
    double *coeff = this->coeff->data;
    double *lap_ref = this->lap->data;
    double *flx_ref = this->flx->data;
    double *fly_ref = this->fly->data;
    // begin copied code
    for (int k = 0; k < ksize; ++k) {
        for (int j = -1; j < jsize + 1; ++j) {
            for (int i = -1; i < isize + 1; ++i) {
                coord3 cell = this->inner_coord(coord3(i, j, k));
                lap_ref[this->lap->index(cell)] =
                    4 * in[this->input->index(cell)] 
                    - (   in[this->input->neighbor(cell, coord3(-1, 0, 0))] 
                        + in[this->input->neighbor(cell, coord3(+1, 0, 0))]
                        + in[this->input->neighbor(cell, coord3(0, -1, 0))]
                        + in[this->input->neighbor(cell, coord3(0, +1, 0))]);
            }
        }
        for (int j = 0; j < jsize; ++j) {
            for (int i = -1; i < isize; ++i) {
                coord3 cell = this->inner_coord(coord3(i, j, k));
                flx_ref[this->flx->index(cell)] = 
                      lap_ref[this->lap->neighbor(cell, coord3(+1, 0, 0))]
                    - lap_ref[this->lap->index(cell)];
                if (flx_ref[this->lap->index(cell)]
                    * (  in[this->input->neighbor(cell, coord3(+1, 0, 0))] 
                       - in[this->input->index(cell)]) > 0)
                    flx_ref[this->flx->index(cell)] = 0.;
            }
        }
        for (int j = -1; j < jsize; ++j) {
            for (int i = 0; i < isize; ++i) {
                coord3 cell = this->inner_coord(coord3(i, j, k));
                fly_ref[this->fly->index(cell)] = 
                      lap_ref[this->lap->neighbor(cell, coord3(0, +1, 0))] 
                    - lap_ref[this->lap->index(cell)];
                if (fly_ref[this->fly->index(cell)] 
                    * (  in[this->input->neighbor(cell, coord3(0, +1, 0))] 
                       - in[this->input->index(cell)]) > 0)
                    fly_ref[this->fly->index(cell)] = 0.;
            }
        }
        for (int i = 0; i < isize; ++i) {
            for (int j = 0; j < jsize; ++j) {
                coord3 cell = this->inner_coord(coord3(i, j, k));
                out_ref[this->output->index(cell)] =
                      in[this->input->index(cell)]
                    - coeff[this->coeff->index(cell)]
                    * (  flx_ref[this->flx->index(cell)]
                       - flx_ref[this->flx->neighbor(cell, coord3(-1, 0, 0))]
                       + fly_ref[this->fly->index(cell)]
                       - fly_ref[this->fly->neighbor(cell, coord3(0, -1, 0))]);
            }
        }
    }
}

dim3 HdiffCPUUnstrBenchmark::numblocks() {
    return dim3(1, 1, 1);
}

dim3 HdiffCPUUnstrBenchmark::numthreads() {
    return dim3(1, 1, 1);
}

#endif