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
    //Grid<value_t, coord3> *u_out;
    //Grid<value_t, coord3> *v_out;
    Grid<value_t, coord3> *u_pos;
    Grid<value_t, coord3> *v_pos;
    Grid<value_t, coord3> *u_tens;
    Grid<value_t, coord3> *v_tens;
    Grid<value_t, coord3> *rho;
    Grid<value_t, coord3> *ppuv;
    Grid<value_t, coord3> *fx;
    /*Grid<value_t, coord3> *rho0;
    Grid<value_t, coord3> *cwp;
    Grid<value_t, coord3> *p0;
    Grid<value_t, coord3> *wbbctens_stage;*/
    Grid<value_t, coord3> *wgtfac;
    Grid<value_t, coord3> *hhl;
    /*Grid<value_t, coord3> *xlhsx;
    Grid<value_t, coord3> *xlhsy;
    Grid<value_t, coord3> *xdzdx;
    Grid<value_t, coord3> *xdzdy;
    Grid<value_t, coord3> *xrhsx_ref;
    Grid<value_t, coord3> *xrhsy_ref;
    Grid<value_t, coord3> *xrhsz_ref;*/
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
void FastWavesRefBenchmark<value_t>::setup() {

    // allocate these grids ...
    this->u_ref = new RegularGrid3D<value_t>(this->size);
    this->v_ref = new RegularGrid3D<value_t>(this->size);
    //this->u_out = new RegularGrid3D<value_t>(this->size);
    //this->v_out = new RegularGrid3D<value_t>(this->size);
    this->u_pos = new RegularGrid3D<value_t>(this->size);
    this->v_pos = new RegularGrid3D<value_t>(this->size);
    this->u_tens = new RegularGrid3D<value_t>(this->size);
    this->v_tens = new RegularGrid3D<value_t>(this->size);
    this->rho = new RegularGrid3D<value_t>(this->size);
    this->ppuv = new RegularGrid3D<value_t>(this->size);
    this->fx = new RegularGrid3D<value_t>(this->size);
    /*this->rho0 = new RegularGrid3D<value_t>(this->size);
    this->cwp = new RegularGrid3D<value_t>(this->size);
    this->p0 = new RegularGrid3D<value_t>(this->size);
    this->wbbctens_stage = new RegularGrid3D<value_t>(this->size);*/
    this->wgtfac = new RegularGrid3D<value_t>(this->size);
    this->hhl = new RegularGrid3D<value_t>(this->size);
    /*this->xlhsx = new RegularGrid3D<value_t>(this->size);
    this->xlhsy = new RegularGrid3D<value_t>(this->size);
    this->xdzdx = new RegularGrid3D<value_t>(this->size);
    this->xdzdy = new RegularGrid3D<value_t>(this->size);
    this->xrhsx_ref = new RegularGrid3D<value_t>(this->size);
    this->xrhsy_ref = new RegularGrid3D<value_t>(this->size);
    this->xrhsz_ref = new RegularGrid3D<value_t>(this->size);*/
    this->ppgradcor = new RegularGrid3D<value_t>(this->size);
    this->ppgradu = new RegularGrid3D<value_t>(this->size);
    this->ppgradv = new RegularGrid3D<value_t>(this->size);

    this->populate_grids();

}

template<typename value_t>
void FastWavesRefBenchmark<value_t>::populate_grids() {

    // Fill all grid elements with random values
    int total_size = this->u_ref->size / sizeof(value_t);
    std::minstd_rand eng;
    std::uniform_real_distribution<value_t> dist(-1, 1);
    for (int i = 0; i < total_size; ++i) {
        //this->u_ref->data[i] = dist(eng);
        //this->v_ref->data[i] = dist(eng);
        //this->u_out->data[i] = dist(eng);
        //this->v_out->data[i] = dist(eng);
        this->u_pos->data[i] = dist(eng);
        this->v_pos->data[i] = dist(eng);
        this->u_tens->data[i] = dist(eng);
        this->v_tens->data[i] = dist(eng);
        this->rho->data[i] = dist(eng);
        this->ppuv->data[i] = dist(eng);
        this->fx->data[i] = dist(eng);
        /*this->rho0->data[i] = dist(eng);
        this->cwp->data[i] = dist(eng);
        this->p0->data[i] = dist(eng);
        this->wbbctens_stage->data[i] = dist(eng);*/
        this->wgtfac->data[i] = dist(eng);
        this->hhl->data[i] = dist(eng);
        /*this->xlhsx->data[i] = dist(eng);
        this->xlhsy->data[i] = dist(eng);
        this->xdzdx->data[i] = dist(eng);
        this->xdzdy->data[i] = dist(eng);
        this->xrhsy_ref->data[i] = dist(eng);
        this->xrhsx_ref->data[i] = dist(eng);
        this->xrhsz_ref->data[i] = dist(eng);*/
        //this->ppgradcor->data[i] = dist(eng);
        //this->ppgradu->data[i] = dist(eng);
        //this->ppgradv->data[i] = dist(eng);
    }

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
                    ptr->set(coord3(i, j, k), offset1
                                              + base1 * (offset2 + cos(M_PI * (spreadx * x + spready * y)) +
                                                       base2 * sin(2 * M_PI * (spreadx * x + spready * y) * z))
                                              / 4. );
                }
            }
        }
    };

    //fill_field(this->u_ref, 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
    //fill_field(this->v_ref, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    //fill_field(this->u_out, 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
    //fill_field(this->v_out, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->u_pos, 3.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->v_pos, 2.4, 1.3, 0.77, 1.11, 1.4, 2.3);
    fill_field(this->u_tens, 4.3, 0.3, 0.97, 1.11, 1.4, 2.3);
    fill_field(this->v_tens, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->ppuv, 1.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->rho, 1.4, 4.3, 0.87, 1.11, 1.4, 2.3);
    /*fill_field(this->rho0, 3.4, 1.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->p0, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);*/
    fill_field(this->hhl, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->wgtfac, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->fx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    /*fill_field(this->cwp, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xdzdx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xdzdy, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xlhsx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xlhsy, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->wbbctens_stage, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);*/
}

template<typename value_t>
void FastWavesRefBenchmark<value_t>::run() {

    if(this->inner_size.x < 0 || this->inner_size.y < 0 || this->inner_size.z < 0) {
        throw std::runtime_error("Grid too small for this kernel.");
    }

    const int dt_small = this->dt_small; //10;
    const int edadlat = this->edadlat; //1;
    const int cFlatLimit = this->c_flat_limit; //10;
    coord3 halo = this->halo;
    //coord3 size = this->inner_size;
    const coord3 max_coord = this->inner_size + this->halo;

    auto computePPGradCor = [&](int i, int j, int k) {
        this->ppgradcor->set(coord3(i, j, k), this->wgtfac->get(coord3(i, j, k)) * this->ppuv->get(coord3(i, j, k))
                                              + (1.0 - this->wgtfac->get(coord3(i, j, k))) * this->ppuv->get(coord3(i, j, k - 1)));
    };

    // PPGradCorStage
    int k = cFlatLimit + halo.z;
    for (int i = halo.x; i < max_coord.x + 1; ++i) {
        for (int j = halo.y; j < max_coord.y + 1; ++j) {
            computePPGradCor(i, j, k);
        }
    }

    for (k = cFlatLimit + 1 + halo.z; k < max_coord.z; ++k) {
        for (int i = halo.x; i < max_coord.x + 1; ++i) {
            for (int j = halo.y; j < max_coord.y + 1; ++j) {
                computePPGradCor(i, j, k);
                this->ppgradcor->set(coord3(i, j, k - 1), (this->ppgradcor->get(coord3(i, j, k)) - this->ppgradcor->get(coord3(i, j, k - 1))));
            }
        }
    }

    // XRHSXStage
    // FullDomain
    /*k = max_coord.z - 1;
    for (int i = halo.x - 1; i < max_coord.x; ++i) {
        for (int j = halo.y; j < max_coord.y + 1; ++j) {
            this->xrhsx_ref->set(coord3(i, j, k), -this->fx->get(coord3(i, j, k)) /
                                         ((value_t)0.5 * (this->rho->get(coord3(i, j, k)) + this->rho->get(coord3(i + 1, j, k)))) *
                                         (this->ppuv->get(coord3(i + 1, j, k)) - this->ppuv->get(coord3(i, j, k))) +
                                     this->u_tens->get(coord3(i, j, k)));
            this->xrhsy_ref->set(coord3(i, j, k), -edadlat /
                                         ((value_t)0.5 * (this->rho->get(coord3(i, j + 1, k)) + this->rho->get(coord3(i, j, k)))) *
                                         (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))) +
                                     this->v_tens->get(coord3(i, j, k)));
        }
    }
    for (int i = halo.x; i < max_coord.x + 1; ++i) {
        for (int j = halo.y-1; j < max_coord.y; ++j) {
            this->xrhsy_ref->set(coord3(i, j, k), -edadlat /
                                         ((value_t)0.5 * (this->rho->get(coord3(i, j + 1, k)) + this->rho->get(coord3(i, j, k)))) *
                                         (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))) +
                                     this->v_tens->get(coord3(i, j, k)));
        }
    }
    for (int i = halo.x; i < max_coord.x + 1; ++i) {
        for (int j = halo.y; j < max_coord.y + 1; ++j) {
            this->xrhsz_ref->set(coord3(i, j, k), 
                this->rho0->get(coord3(i, j, k)) / this->rho->get(coord3(i, j, k)) * 9.8 *
                    ((value_t)1.0 - this->cwp->get(coord3(i, j, k)) * (this->p0->get(coord3(i, j, k)) + this->ppuv->get(coord3(i, j, k)))) +
                this->wbbctens_stage->get(coord3(i, j, k + 1)));
        }
    }*/

    // PPGradStage
    for (k = halo.z; k < max_coord.z - 1; ++k) {
        for (int i = halo.x; i < max_coord.x; ++i) {
            for (int j = halo.y; j < max_coord.y; ++j) {
                if (k < cFlatLimit + halo.z) {
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
    for (k = halo.z; k < max_coord.z - 1; ++k) {
        for (int i = halo.x; i < max_coord.x; ++i) {
            for (int j = halo.y; j < max_coord.y; ++j) {
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

    /*k = max_coord.z - 1;
    for (int i = halo.x; i < max_coord.x; ++i) {
        for (int j = halo.y; j < max_coord.y; ++j) {
            value_t bottU =
                this->xlhsx->get(coord3(i, j, k)) * this->xdzdx->get(coord3(i, j, k)) *
                    ((value_t)0.5 * (this->xrhsz_ref->get(coord3(i + 1, j, k)) + this->xrhsz_ref->get(coord3(i, j, k))) -
                        this->xdzdx->get(coord3(i, j, k)) * this->xrhsx_ref->get(coord3(i, j, k)) -
                        (value_t)0.5 *
                            ((value_t)0.5 * (this->xdzdy->get(coord3(i + 1, j - 1, k)) + this->xdzdy->get(coord3(i + 1, j, k))) +
                                (value_t)0.5 * (this->xdzdy->get(coord3(i, j - 1, k)) + this->xdzdy->get(coord3(i, j, k)))) *
                            (value_t)0.5 *
                            ((value_t)0.5 * (this->xrhsy_ref->get(coord3(i + 1, j - 1, k)) + this->xrhsy_ref->get(coord3(i + 1, j, k))) +
                                (value_t)0.5 * (this->xrhsy_ref->get(coord3(i, j - 1, k)) + this->xrhsy_ref->get(coord3(i, j, k))))) +
                this->xrhsx_ref->get(coord3(i, j, k));
            this->u_ref->set(coord3(i, j, k), this->u_pos->get(coord3(i, j, k)) + bottU * dt_small);
            value_t bottV =
                this->xlhsy->get(coord3(i, j, k)) * this->xdzdy->get(coord3(i, j, k)) *
                    ((value_t)0.5 * (this->xrhsz_ref->get(coord3(i, j + 1, k)) + this->xrhsz_ref->get(coord3(i, j, k))) -
                        this->xdzdy->get(coord3(i, j, k)) * this->xrhsy_ref->get(coord3(i, j, k)) -
                        (value_t)0.5 *
                            ((value_t)0.5 * (this->xdzdx->get(coord3(i - 1, j + 1, k)) + this->xdzdx->get(coord3(i, j + 1, k))) +
                                (value_t)0.5 * (this->xdzdx->get(coord3(i - 1, j, k)) + this->xdzdx->get(coord3(i, j, k)))) *
                            (value_t)0.5 *
                            ((value_t)0.5 * (this->xrhsx_ref->get(coord3(i - 1, j + 1, k)) + this->xrhsx_ref->get(coord3(i, j + 1, k))) +
                                (value_t)0.5 * (this->xrhsx_ref->get(coord3(i - 1, j, k)) + this->xrhsx_ref->get(coord3(i, j, k))))) +
                this->xrhsy_ref->get(coord3(i, j, k));
            this->v_ref->set(coord3(i, j, k), this->v_pos->get(coord3(i, j, k)) + bottV * dt_small);
        }
    }*/

}

#endif