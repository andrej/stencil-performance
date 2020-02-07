#ifndef FAST_WAVES_REF_H
#define FAST_WAVES_REF_H

#include <vector>
#include <random>
#include <stdexcept>
#include "benchmarks/benchmark.cu"
#include "grids/regular.cu"

/** Reference implementation of fast waves kernel. Runs on CPU. Ported from
 * given source with fewest possible modifications. Simplified to exclude XHRS stages. */
template<typename value_t>
class FastWavesRefBenchmark : public Benchmark {

    public:

    FastWavesRefBenchmark(coord3 size);

    Grid<value_t, coord3> *u_ref;
    Grid<value_t, coord3> *v_ref;
    Grid<value_t, coord3> *u_pos;
    Grid<value_t, coord3> *v_pos;
    Grid<value_t, coord3> *u_tens;
    Grid<value_t, coord3> *v_tens;
    Grid<value_t, coord3> *rho;
    Grid<value_t, coord3> *ppuv;
    Grid<value_t, coord3> *fx;
    Grid<value_t, coord3> *wgtfac;
    Grid<value_t, coord3> *hhl;
    Grid<value_t, coord3> *ppgradcor;
    Grid<value_t, coord3> *ppgradu;
    Grid<value_t, coord3> *ppgradv;

    coord3 halo;
    coord3 inner_size; /**< size without 2*halo */
    int dt_small;
    int edadlat;
    int c_flat_limit;

    virtual void setup();
    virtual void run();

    virtual dim3 numthreads(coord3 domain=coord3());
    virtual dim3 numblocks(coord3 domain=coord3());

    /** pre-populate the grids with some values (randomly), exactly as in the
     * reference implementation. */
    void populate_grids();

};

// IMPLEMENTATIONS

template<typename value_t>
FastWavesRefBenchmark<value_t>::FastWavesRefBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2, 2, 2)) {
    this->inner_size = size - 2*this->halo;
    this->name = "fastwaves-ref";
}

template<typename value_t>
dim3 FastWavesRefBenchmark<value_t>::numthreads(coord3 domain) {
    return dim3(1, 1, 1);
}

template<typename value_t>
dim3 FastWavesRefBenchmark<value_t>::numblocks(coord3 domain) {
    return dim3(1, 1, 1);
}

template<typename value_t>
void FastWavesRefBenchmark<value_t>::setup() {

    // allocate these grids ...
    this->u_ref = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->v_ref = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->u_pos = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->v_pos = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->u_tens = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->v_tens = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->rho = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->ppuv = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->fx = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->wgtfac = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->hhl = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->ppgradcor = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->ppgradu = RegularGrid3D<value_t>::create(this->inner_size, this->halo);
    this->ppgradv = RegularGrid3D<value_t>::create(this->inner_size, this->halo);

    this->populate_grids();

}

template<typename value_t>
void FastWavesRefBenchmark<value_t>::populate_grids() {

    // Fill all grid elements with random values
    int total_size = this->u_ref->size / sizeof(value_t);
    std::minstd_rand eng;
    std::uniform_real_distribution<value_t> dist(-1, 1);
    for (int i = 0; i < total_size; ++i) {
        this->u_pos->data[i] = dist(eng);
        this->v_pos->data[i] = dist(eng);
        this->u_tens->data[i] = dist(eng);
        this->v_tens->data[i] = dist(eng);
        this->rho->data[i] = dist(eng);
        this->ppuv->data[i] = dist(eng);
        this->fx->data[i] = dist(eng);
        this->wgtfac->data[i] = dist(eng);
        this->hhl->data[i] = dist(eng);
    }

    // The reference implementation populates the grids to the values below
    // I do not see how it makes any difference for the computation if we just run it on random data
    // Therefore, commented out to allow us to run the benchmarks a little more quickly
    /*
    // Now, to fill some of the Grids with more regular data in a certain range
    // Using this helper function
    auto fill_field = [&](Grid<value_t, coord3> *ptr, value_t offset1, value_t offset2, value_t base1, value_t base2, value_t spreadx, value_t spready) {
        value_t dx = 1. / this->size.x;
        value_t dy = 1. / this->size.y;
        value_t dz = 1. / this->size.z;
        // NOTE: in ref implementation, indices ran from 0 to size+2*halo, but the references passed in
        // were + zero_offset(), i.e. the accessed indices ranged from halo to size+3*halo, which makes no sense
        // it appears the intent was to simply fill the entire grid, including padding/halo, so we just iterate over
        // all indices from 0 to size. either way, this just initializes the grid to some values so it should not be
        // that a huge difference, and this way we have no out-of-bounds indices
        for (int j = 0; j < this->size.y; j++) {
            for (int i = 0; i < this->size.x; i++) {
                value_t x = dx * (value_t)(i);
                value_t y = dy * (value_t)(j);
                for (int k = 0; k < this->size.z; k++) {
                    value_t z = dz * (value_t)(k);
                    // u values between 5 and 9
                    ptr->set(coord3(i-this->halo.x, j-this->halo.y, k-this->halo.z), 
                             offset1
                             + base1 * (offset2 + cos(M_PI * (spreadx * x + spready * y))
                             + base2 * sin(2 * M_PI * (spreadx * x + spready * y) * z))
                            / 4. );
                }
            }
        }
    };

    fill_field(this->u_pos, 3.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->v_pos, 2.4, 1.3, 0.77, 1.11, 1.4, 2.3);
    fill_field(this->u_tens, 4.3, 0.3, 0.97, 1.11, 1.4, 2.3);
    fill_field(this->v_tens, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->ppuv, 1.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->rho, 1.4, 4.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->hhl, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->wgtfac, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->fx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    */
}

template<typename value_t>
void FastWavesRefBenchmark<value_t>::run() {

    if(this->inner_size.x < 0 || this->inner_size.y < 0 || this->inner_size.z < 0) {
        throw std::runtime_error("Grid too small for this kernel.");
    }

    const int dt_small = this->dt_small; //10;
    const int edadlat = this->edadlat; //1;
    const int cFlatLimit = this->c_flat_limit; //10;
    const coord3 max_coord = this->inner_size + this->halo;

    auto computePPGradCor = [&](int i, int j, int k) {
        this->ppgradcor->set(coord3(i, j, k), this->wgtfac->get(coord3(i, j, k)) * this->ppuv->get(coord3(i, j, k))
                                              + (1.0 - this->wgtfac->get(coord3(i, j, k))) * this->ppuv->get(coord3(i, j, k - 1)));
    };

    // PPGradCorStage
    int k = cFlatLimit + 0;
    for (int i = 0; i < this->inner_size.x + 1; ++i) {
        for (int j = 0; j < this->inner_size.y + 1; ++j) {
            computePPGradCor(i, j, k);
        }
    }

    for (k = cFlatLimit + 1 + 0; k < this->inner_size.z; ++k) {
        for (int i = 0; i < this->inner_size.x + 1; ++i) {
            for (int j = 0; j < this->inner_size.y + 1; ++j) {
                computePPGradCor(i, j, k);
                this->ppgradcor->set(coord3(i, j, k - 1), (this->ppgradcor->get(coord3(i, j, k)) - this->ppgradcor->get(coord3(i, j, k - 1))));
            }
        }
    }

    // PPGradStage
    for (k = 0; k < this->inner_size.z - 1; ++k) {
        for (int i = 0; i < this->inner_size.x; ++i) {
            for (int j = 0; j < this->inner_size.y; ++j) {
                if (k < cFlatLimit + 0) {
                    this->ppgradu->set(coord3(i, j, k), (this->ppuv->get(coord3(i + 1, j, k)) - this->ppuv->get(coord3(i, j, k))));
                    this->ppgradv->set(coord3(i, j, k), (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))));
                } else {
                    this->ppgradu->set(coord3(i, j, k), (this->ppuv->get(coord3(i + 1, j, k)) - this->ppuv->get(coord3(i, j, k))) +
                                               (this->ppgradcor->get(coord3(i + 1, j, k)) + this->ppgradcor->get(coord3(i, j, k))) *
                                                   (value_t)0.5 *
                                                   ((this->hhl->get(coord3(i, j, k + 1)) + this->hhl->get(coord3(i, j, k))) -
                                                       (this->hhl->get(coord3(i + 1, j, k + 1)) + this->hhl->get(coord3(i + 1, j, k)))) /
                                                   ((this->hhl->get(coord3(i, j, k + 1)) - this->hhl->get(coord3(i, j, k))) +
                                                       (this->hhl->get(coord3(i + 1, j, k + 1)) - this->hhl->get(coord3(i + 1, j, k)))));
                    this->ppgradv->set(coord3(i, j, k), (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))) +
                                               (this->ppgradcor->get(coord3(i, j + 1, k)) + this->ppgradcor->get(coord3(i, j, k))) *
                                                   (value_t)0.5 *
                                                   ((this->hhl->get(coord3(i, j, k + 1)) + this->hhl->get(coord3(i, j, k))) -
                                                       (this->hhl->get(coord3(i, j + 1, k + 1)) + this->hhl->get(coord3(i, j + 1, k)))) /
                                                   ((this->hhl->get(coord3(i, j, k + 1)) - this->hhl->get(coord3(i, j, k))) +
                                                       (this->hhl->get(coord3(i, j + 1, k + 1)) - this->hhl->get(coord3(i, j + 1, k)))));
                }
            }
        }
    }

    // UVStage
    // FullDomain
    for (k = 0; k < this->inner_size.z - 1; ++k) {
        for (int i = 0; i < this->inner_size.x; ++i) {
            for (int j = 0; j < this->inner_size.y; ++j) {
                value_t rhou =
                    this->fx->get(coord3(i, j, k)) / ((value_t)0.5 * (this->rho->get(coord3(i + 1, j, k)) + this->rho->get(coord3(i, j, k))));
                value_t rhov = edadlat / ((value_t)0.5 * (this->rho->get(coord3(i, j + 1, k)) + this->rho->get(coord3(i, j, k))));

                this->u_ref->set(coord3(i, j, k), 
                    this->u_pos->get(coord3(i, j, k)) +
                    (this->u_tens->get(coord3(i, j, k)) - this->ppgradu->get(coord3(i, j, k)) * rhou) * dt_small);
                this->v_ref->set(coord3(i, j, k), 
                    this->v_pos->get(coord3(i, j, k)) +
                    (this->v_tens->get(coord3(i, j, k)) - this->ppgradv->get(coord3(i, j, k)) * rhov) * dt_small);
            }
        }
    }

}

#endif