#ifndef HDIFF_REF_H
#define HDIFF_REF_H
#include "benchmarks/benchmark.cu"
#include "coord3.cu"
#include "grids/grid.cu"
#include "grids/regular.cu"
#include "grids/unstructured.cu"

/** This is the reference implementation for the horizontal diffusion kernel, 
 * which is executed on the CPU and used to verify other implementations. */
template<typename value_t>
class HdiffReferenceBenchmark : public Benchmark {

    public:

    // The padding option currently only applies to regular grids
    HdiffReferenceBenchmark(coord3 size, size_t padding=0);
 
    RegularGrid3D<value_t> *input = NULL;
    RegularGrid3D<value_t> *output = NULL;
    RegularGrid3D<value_t> *coeff = NULL;
    RegularGrid3D<value_t> *lap = NULL;
    RegularGrid3D<value_t> *flx = NULL;
    RegularGrid3D<value_t> *fly = NULL;

    // Setup Input values
    // As in hdiff_stencil_variant.h
    virtual void setup();
    virtual void populate_grids();
    virtual void teardown();

    // CPU implementation
    // As in hdiff_stencil_variant.h
    void run();

    // halo around the input data, padding that is not touched
    size_t padding;
    coord3 halo;
    coord3 inner_size; // size w.o. 2* halo
    coord3 inner_coord(coord3 inner_coord);

    // Print (1, 1, 1) for numblocks/numthreads as this is on CPU
    dim3 numblocks();
    dim3 numthreads();

};

// IMPLEMENTATIONS

template<typename value_t>
HdiffReferenceBenchmark<value_t>::HdiffReferenceBenchmark(coord3 size, size_t padding) :
Benchmark(size),
padding(padding),
halo(coord3(2,2,0)){
    if(this->padding > 0) {
        this->name = "hdiff-ref-pad";
    } else {
        this->name = "hdiff-ref";
    }
}

template<typename value_t>
void HdiffReferenceBenchmark<value_t>::setup(){
    // Set up grids
    this->input = new RegularGrid3D<value_t>(this->size, this->padding);
    this->output = new RegularGrid3D<value_t>(this->size, this->padding);
    this->coeff = new RegularGrid3D<value_t>(this->size, this->padding);
    this->lap = new RegularGrid3D<value_t>(this->size, this->padding);
    this->flx = new RegularGrid3D<value_t>(this->size, this->padding);
    this->fly = new RegularGrid3D<value_t>(this->size, this->padding);
    // Algorithm requires a halo: padding that is not touched
    this->inner_size = this->size - 2*this->halo;
    // Populate with data
    this->populate_grids();
    // do NOT call base setup here, this would lead to infinite recursion as
    // in base setup we create a reference benchmark such as this one
}

template<typename value_t>
void HdiffReferenceBenchmark<value_t>::populate_grids() {
    // Populate memory with values as in reference implementation (copied 1:1)
    value_t *m_in = this->input->data;
    value_t *m_out = this->output->data;
    value_t *m_coeff = this->coeff->data;
    value_t *m_lap = this->lap->data;
    value_t *m_flx = this->flx->data;
    value_t *m_fly = this->fly->data;
    const int isize = this->inner_size.x;
    const int jsize = this->inner_size.y;
    const int ksize = this->inner_size.z;
    // original code starts here
    value_t dx = 1. / (value_t)(isize);
    value_t dy = 1. / (value_t)(jsize);
    value_t dz = 1. / (value_t)(ksize);
    for (int j = 0; j < isize; j++) {
        for (int i = 0; i < jsize; i++) {
            value_t x = dx * (value_t)(i);
            value_t y = dy * (value_t)(j);
            for (int k = 0; k < ksize; k++) {
                int cnt = this->input->index(this->inner_coord(coord3(j, i, k))); // MODIFIED
                value_t z = dz * (value_t)(k);
                // u values between 5 and 9
                m_in[cnt] = 3.0 +
                            1.25 * (2.5 + cos(M_PI * (18.4 * x + 20.3 * y)) +
                                        0.78 * sin(2 * M_PI * (18.4 * x + 20.3 * y) * z)) /
                                4.;
                m_coeff[cnt] = 1.4 +
                                0.87 * (0.3 + cos(M_PI * (1.4 * x + 2.3 * y)) +
                                            1.11 * sin(2 * M_PI * (1.4 * x + 2.3 * y) * z)) /
                                    4.;
                m_out[cnt] = 5.4;
                m_flx[cnt] = 0.0;
                m_fly[cnt] = 0.0;
                m_lap[cnt] = 0.0;
            }
        }
    }
}

template<typename value_t>
void HdiffReferenceBenchmark<value_t>::run() {
    // Grids
    value_t *in = this->input->data;
    value_t *coeff = this->coeff->data;
    value_t *out_ref = this->output->data;
    value_t *lap_ref = this->lap->data;
    value_t *flx_ref = this->flx->data;
    value_t *fly_ref = this->fly->data;
    // convenience variables
    const int isize = this->inner_size.x;
    const int jsize = this->inner_size.y;
    const int ksize = this->inner_size.z;
    auto index = [this](int x, int y, int z) { return this->input->index(this->inner_coord(coord3(x, y, z))); };
    // begin copied code
    for (int k = 0; k < ksize; ++k) {
        for (int j = -1; j < jsize + 1; ++j) {
            for (int i = -1; i < isize + 1; ++i) {
                lap_ref[index(i, j, k)] =
                    4 * in[index(i, j, k)] - (in[index(i - 1, j, k)] + in[index(i + 1, j, k)] +
                                                   in[index(i, j - 1, k)] + in[index(i, j + 1, k)]);
            }
        }
        for (int j = 0; j < jsize; ++j) {
            for (int i = -1; i < isize; ++i) {
                flx_ref[index(i, j, k)] = lap_ref[index(i + 1, j, k)] - lap_ref[index(i, j, k)];
                if (flx_ref[index(i, j, k)] * (in[index(i + 1, j, k)] - in[index(i, j, k)]) > 0)
                    flx_ref[index(i, j, k)] = 0.;
            }
        }
        for (int j = -1; j < jsize; ++j) {
            for (int i = 0; i < isize; ++i) {
                fly_ref[index(i, j, k)] = lap_ref[index(i, j + 1, k)] - lap_ref[index(i, j, k)];
                if (fly_ref[index(i, j, k)] * (in[index(i, j + 1, k)] - in[index(i, j, k)]) > 0)
                    fly_ref[index(i, j, k)] = 0.;
            }
        }
        for (int i = 0; i < isize; ++i) {
            for (int j = 0; j < jsize; ++j) {
                out_ref[index(i, j, k)] =
                    in[index(i, j, k)] -
                    coeff[index(i, j, k)] * (flx_ref[index(i, j, k)] - flx_ref[index(i - 1, j, k)] +
                                                  fly_ref[index(i, j, k)] - fly_ref[index(i, j - 1, k)]);
            }
        }
    }
}

template<typename value_t>
void HdiffReferenceBenchmark<value_t>::teardown() {
    this->input->deallocate();
    this->output->deallocate();
    this->coeff->deallocate();
    this->lap->deallocate();
    this->flx->deallocate();
    this->fly->deallocate();
    delete this->input;
    delete this->output;
    delete this->coeff;
    delete this->lap;
    delete this->flx;
    delete this->fly;
}

template<typename value_t>
coord3 HdiffReferenceBenchmark<value_t>::inner_coord(coord3 coord){
    return coord + this->halo;
}

template<typename value_t>
dim3 HdiffReferenceBenchmark<value_t>::numblocks() {
    return dim3(1, 1, 1);
}

template<typename value_t>
dim3 HdiffReferenceBenchmark<value_t>::numthreads() {
    return dim3(1, 1, 1);
}

#endif