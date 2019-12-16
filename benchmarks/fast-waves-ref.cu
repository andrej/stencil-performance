#ifndef FAST_WAVES_REF_H
#define FAST_WAVES_REF_H

#include <vector>
#include <random>
#include "benchmarks/benchmark.cu"
#include "grids/regular.cu"

/** Reference implementation of fast waves kernel. Runs on CPU. Ported from
 * given source with fewest possible modifications. */
class FastWavesRefBenchmark : public Benchmark {

    public:

    FastWavesRefBenchmark(coord3 size);

    Grid<double, coord3> *u_ref;
    Grid<double, coord3> *v_ref;
    Grid<double, coord3> *u_out;
    Grid<double, coord3> *v_out;
    Grid<double, coord3> *u_pos;
    Grid<double, coord3> *v_pos;
    Grid<double, coord3> *u_tens;
    Grid<double, coord3> *v_tens;
    Grid<double, coord3> *rho;
    Grid<double, coord3> *ppuv;
    Grid<double, coord3> *fx;
    Grid<double, coord3> *rho0;
    Grid<double, coord3> *cwp;
    Grid<double, coord3> *p0;
    Grid<double, coord3> *wbbctens_stage;
    Grid<double, coord3> *wgtfac;
    Grid<double, coord3> *hhl;
    Grid<double, coord3> *xlhsx;
    Grid<double, coord3> *xlhsy;
    Grid<double, coord3> *xdzdx;
    Grid<double, coord3> *xdzdy;
    Grid<double, coord3> *xrhsx_ref;
    Grid<double, coord3> *xrhsy_ref;
    Grid<double, coord3> *xrhsz_ref;
    Grid<double, coord3> *ppgradcor;
    Grid<double, coord3> *ppgradu;
    Grid<double, coord3> *ppgradv;

    /** The halo prevents out-of-bounds indexing by restricting corodinates to
     * some inner coordinates, i.e. the size of the grids is shrunk from all
     * sides by a certain padding as specified in halo. */
    coord3 halo;
    coord3 inner_size; /**< size without 2*halo */
    coord3 inner_coord(coord3 inner_coord); /**< give "global" coords from inner coord, i.e. inner_coord + halo. */

    virtual void setup();
    virtual void run();

    /** pre-populate the grids with some values (randomly), exactly as in the
     * reference implementation. */
    void populate_grids();

};

// IMPLEMENTATIONS

FastWavesRefBenchmark::FastWavesRefBenchmark(coord3 size) :
Benchmark(size),
halo(coord3(2, 2, 2)) {
    this->inner_size = size - 2*this->halo;
    this->name = "fastwaves-ref";
}

void FastWavesRefBenchmark::setup() {

    // allocate these grids ...
    this->u_ref = new RegularGrid3D<double>(this->size);
    this->v_ref = new RegularGrid3D<double>(this->size);
    this->u_out = new RegularGrid3D<double>(this->size);
    this->v_out = new RegularGrid3D<double>(this->size);
    this->u_pos = new RegularGrid3D<double>(this->size);
    this->v_pos = new RegularGrid3D<double>(this->size);
    this->u_tens = new RegularGrid3D<double>(this->size);
    this->v_tens = new RegularGrid3D<double>(this->size);
    this->rho = new RegularGrid3D<double>(this->size);
    this->ppuv = new RegularGrid3D<double>(this->size);
    this->fx = new RegularGrid3D<double>(this->size);
    this->rho0 = new RegularGrid3D<double>(this->size);
    this->cwp = new RegularGrid3D<double>(this->size);
    this->p0 = new RegularGrid3D<double>(this->size);
    this->wbbctens_stage = new RegularGrid3D<double>(this->size);
    this->wgtfac = new RegularGrid3D<double>(this->size);
    this->hhl = new RegularGrid3D<double>(this->size);
    this->xlhsx = new RegularGrid3D<double>(this->size);
    this->xlhsy = new RegularGrid3D<double>(this->size);
    this->xdzdx = new RegularGrid3D<double>(this->size);
    this->xdzdy = new RegularGrid3D<double>(this->size);
    this->xrhsx_ref = new RegularGrid3D<double>(this->size);
    this->xrhsy_ref = new RegularGrid3D<double>(this->size);
    this->xrhsz_ref = new RegularGrid3D<double>(this->size);
    this->ppgradcor = new RegularGrid3D<double>(this->size);
    this->ppgradu = new RegularGrid3D<double>(this->size);
    this->ppgradv = new RegularGrid3D<double>(this->size);

}

void FastWavesRefBenchmark::populate_grids() {

    // Fill all grid elements with random values
    int total_size = this->u_ref->size / sizeof(double);
    std::minstd_rand eng;
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int i = 0; i < total_size; ++i) {
        this->u_ref->data[i] = dist(eng);
        this->v_ref->data[i] = dist(eng);
        this->u_out->data[i] = dist(eng);
        this->v_out->data[i] = dist(eng);
        this->u_pos->data[i] = dist(eng);
        this->v_pos->data[i] = dist(eng);
        this->u_tens->data[i] = dist(eng);
        this->v_tens->data[i] = dist(eng);
        this->rho->data[i] = dist(eng);
        this->ppuv->data[i] = dist(eng);
        this->fx->data[i] = dist(eng);
        this->rho0->data[i] = dist(eng);
        this->cwp->data[i] = dist(eng);
        this->p0->data[i] = dist(eng);
        this->wbbctens_stage->data[i] = dist(eng);
        this->wgtfac->data[i] = dist(eng);
        this->hhl->data[i] = dist(eng);
        this->xlhsx->data[i] = dist(eng);
        this->xlhsy->data[i] = dist(eng);
        this->xdzdx->data[i] = dist(eng);
        this->xdzdy->data[i] = dist(eng);
        this->xrhsy_ref->data[i] = dist(eng);
        this->xrhsx_ref->data[i] = dist(eng);
        this->xrhsz_ref->data[i] = dist(eng);
        this->ppgradcor->data[i] = dist(eng);
        this->ppgradu->data[i] = dist(eng);
        this->ppgradv->data[i] = dist(eng);
    }

    // Now, to fill some of the Grids with more regular data in a certain range
    // Using this helper function
    auto fill_field = [&](Grid<double, coord3> *ptr, double offset1, double offset2, double base1, double base2, double spreadx, double spready) {
        double dx = 1. / this->size.x;
        double dy = 1. / this->size.y;
        double dz = 1. / this->size.z;
        // NOTE: in ref implementation, indices ran from 0 to size+2*halo, but the references passed in
        // were + zero_offset(), i.e. the accessed indices ranged from halo to size+3*halo, which makes no sense
        // it appears the intent was to simply fill the entire grid, including padding/halo, so we just iterate over
        // all indices from 0 to size. either way, this just initializes the grid to some values so it should not be
        // that a huge difference, and this way we have no out-of-bounds indices
        for (int j = 0; j < this->size.y; j++) {
            for (int i = 0; i < this->size.x; i++) {
                double x = dx * (double)(i);
                double y = dy * (double)(j);
                for (int k = 0; k < this->size.z; k++) {
                    double z = dz * (double)(k);
                    // u values between 5 and 9
                    ptr->set(coord3(i, j, k), offset1
                                              + base1 * (offset2 + cos(M_PI * (spreadx * x + spready * y)) +
                                                       base2 * sin(2 * M_PI * (spreadx * x + spready * y) * z))
                                              / 4. );
                }
            }
        }
    };

    fill_field(this->u_ref, 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
    fill_field(this->v_ref, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->u_out, 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
    fill_field(this->v_out, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->u_pos, 3.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->v_pos, 2.4, 1.3, 0.77, 1.11, 1.4, 2.3);
    fill_field(this->u_tens, 4.3, 0.3, 0.97, 1.11, 1.4, 2.3);
    fill_field(this->v_tens, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->ppuv, 1.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->rho, 1.4, 4.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->rho0, 3.4, 1.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->p0, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->hhl, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->wgtfac, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->fx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->cwp, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xdzdx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xdzdy, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xlhsx, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->xlhsy, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    fill_field(this->wbbctens_stage, 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
}

void FastWavesRefBenchmark::run() {

    const int cFlatLimit = 10;
    const int dt_small = 10;
    const int edadlat = 1;

    auto computePPGradCor = [&](int i, int j, int k) {
        this->ppgradcor->set(coord3(i, j, k), this->wgtfac->get(coord3(i, j, k)) * this->ppuv->get(coord3(i, j, k))
                                              + (1.0 - this->wgtfac->get(coord3(i, j, k))) * this->ppuv->get(coord3(i, j, k - 1)));
    };

    // PPGradCorStage
    int k = cFlatLimit;
    for (int i = 0; i < this->inner_size.x + 1; ++i) {
        for (int j = 0; j < this->inner_size.y + 1; ++j) {
            computePPGradCor(i, j, k);
        }
    }

    for (k = cFlatLimit + 1; k < this->inner_size.z; ++k) {
        for (int i = 0; i < this->inner_size.x + 1; ++i) {
            for (int j = 0; j < this->inner_size.y + 1; ++j) {
                computePPGradCor(i, j, k);
                this->ppgradcor->set(coord3(i, j, k - 1), (this->ppgradcor->get(coord3(i, j, k)) - this->ppgradcor->get(coord3(i, j, k - 1))));
            }
        }
    }

    // XRHSXStage
    // FullDomain
    k = this->inner_size.z - 1;
    for (int i = -1; i < this->inner_size.x; ++i) {
        for (int j = 0; j < this->inner_size.y + 1; ++j) {
            this->xrhsx_ref->set(coord3(i, j, k), -this->fx->get(coord3(i, j, k)) /
                                         ((double)0.5 * (this->rho->get(coord3(i, j, k)) + this->rho->get(coord3(i + 1, j, k)))) *
                                         (this->ppuv->get(coord3(i + 1, j, k)) - this->ppuv->get(coord3(i, j, k))) +
                                     this->u_tens->get(coord3(i, j, k)));
            this->xrhsy_ref->set(coord3(i, j, k), -edadlat /
                                         ((double)0.5 * (this->rho->get(coord3(i, j + 1, k)) + this->rho->get(coord3(i, j, k)))) *
                                         (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))) +
                                     this->v_tens->get(coord3(i, j, k)));
        }
    }
    for (int i = 0; i < this->inner_size.x + 1; ++i) {
        for (int j = -1; j < this->inner_size.y; ++j) {
            this->xrhsy_ref->set(coord3(i, j, k), -edadlat /
                                         ((double)0.5 * (this->rho->get(coord3(i, j + 1, k)) + this->rho->get(coord3(i, j, k)))) *
                                         (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))) +
                                     this->v_tens->get(coord3(i, j, k)));
        }
    }
    for (int i = 0; i < this->inner_size.x + 1; ++i) {
        for (int j = 0; j < this->inner_size.y + 1; ++j) {
            this->xrhsz_ref->set(coord3(i, j, k), 
                this->rho0->get(coord3(i, j, k)) / this->rho->get(coord3(i, j, k)) * 9.8 *
                    ((double)1.0 - this->cwp->get(coord3(i, j, k)) * (this->p0->get(coord3(i, j, k)) + this->ppuv->get(coord3(i, j, k)))) +
                this->wbbctens_stage->get(coord3(i, j, k + 1)));
        }
    }

    // PPGradStage
    for (k = 0; k < this->inner_size.z - 1; ++k) {
        for (int i = 0; i < this->inner_size.x; ++i) {
            for (int j = 0; j < this->inner_size.y; ++j) {
                if (k < cFlatLimit) {
                    this->ppgradu->set(coord3(i, j, k), (this->ppuv->get(coord3(i + 1, j, k)) - this->ppuv->get(coord3(i, j, k))));
                    this->ppgradv->set(coord3(i, j, k), (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))));
                } else {
                    this->ppgradu->set(coord3(i, j, k), (this->ppuv->get(coord3(i + 1, j, k)) - this->ppuv->get(coord3(i, j, k))) +
                                               (this->ppgradcor->get(coord3(i + 1, j, k)) + this->ppgradcor->get(coord3(i, j, k))) *
                                                   (double)0.5 *
                                                   ((this->hhl->get(coord3(i, j, k + 1)) + this->hhl->get(coord3(i, j, k))) -
                                                       (this->hhl->get(coord3(i + 1, j, k + 1)) + this->hhl->get(coord3(i + 1, j, k)))) /
                                                   ((this->hhl->get(coord3(i, j, k + 1)) - this->hhl->get(coord3(i, j, k))) +
                                                       (this->hhl->get(coord3(i + 1, j, k + 1)) - this->hhl->get(coord3(i + 1, j, k)))));
                    this->ppgradv->set(coord3(i, j, k), (this->ppuv->get(coord3(i, j + 1, k)) - this->ppuv->get(coord3(i, j, k))) +
                                               (this->ppgradcor->get(coord3(i, j + 1, k)) + this->ppgradcor->get(coord3(i, j, k))) *
                                                   (double)0.5 *
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
                double rhou =
                    this->fx->get(coord3(i, j, k)) / ((double)0.5 * (this->rho->get(coord3(i + 1, j, k)) + this->rho->get(coord3(i, j, k))));
                double rhov = edadlat / ((double)0.5 * (this->rho->get(coord3(i, j + 1, k)) + this->rho->get(coord3(i, j, k))));

                this->u_ref->set(coord3(i, j, k), 
                    this->u_pos->get(coord3(i, j, k)) +
                    (this->u_tens->get(coord3(i, j, k)) - this->ppgradu->get(coord3(i, j, k)) * rhou) * dt_small);
                this->v_ref->set(coord3(i, j, k), 
                    this->v_pos->get(coord3(i, j, k)) +
                    (this->v_tens->get(coord3(i, j, k)) - this->ppgradv->get(coord3(i, j, k)) * rhov) * dt_small);
            }
        }
    }

    k = this->inner_size.z - 1;

    for (int i = 0; i < this->inner_size.x; ++i) {
        for (int j = 0; j < this->inner_size.y; ++j) {
            double bottU =
                this->xlhsx->get(coord3(i, j, k)) * this->xdzdx->get(coord3(i, j, k)) *
                    ((double)0.5 * (this->xrhsz_ref->get(coord3(i + 1, j, k)) + this->xrhsz_ref->get(coord3(i, j, k))) -
                        this->xdzdx->get(coord3(i, j, k)) * this->xrhsx_ref->get(coord3(i, j, k)) -
                        (double)0.5 *
                            ((double)0.5 * (this->xdzdy->get(coord3(i + 1, j - 1, k)) + this->xdzdy->get(coord3(i + 1, j, k))) +
                                (double)0.5 * (this->xdzdy->get(coord3(i, j - 1, k)) + this->xdzdy->get(coord3(i, j, k)))) *
                            (double)0.5 *
                            ((double)0.5 * (this->xrhsy_ref->get(coord3(i + 1, j - 1, k)) + this->xrhsy_ref->get(coord3(i + 1, j, k))) +
                                (double)0.5 * (this->xrhsy_ref->get(coord3(i, j - 1, k)) + this->xrhsy_ref->get(coord3(i, j, k))))) +
                this->xrhsx_ref->get(coord3(i, j, k));
            this->u_ref->set(coord3(i, j, k), this->u_pos->get(coord3(i, j, k)) + bottU * dt_small);
            double bottV =
                this->xlhsy->get(coord3(i, j, k)) * this->xdzdy->get(coord3(i, j, k)) *
                    ((double)0.5 * (this->xrhsz_ref->get(coord3(i, j + 1, k)) + this->xrhsz_ref->get(coord3(i, j, k))) -
                        this->xdzdy->get(coord3(i, j, k)) * this->xrhsy_ref->get(coord3(i, j, k)) -
                        (double)0.5 *
                            ((double)0.5 * (this->xdzdx->get(coord3(i - 1, j + 1, k)) + this->xdzdx->get(coord3(i, j + 1, k))) +
                                (double)0.5 * (this->xdzdx->get(coord3(i - 1, j, k)) + this->xdzdx->get(coord3(i, j, k)))) *
                            (double)0.5 *
                            ((double)0.5 * (this->xrhsx_ref->get(coord3(i - 1, j + 1, k)) + this->xrhsx_ref->get(coord3(i, j + 1, k))) +
                                (double)0.5 * (this->xrhsx_ref->get(coord3(i - 1, j, k)) + this->xrhsx_ref->get(coord3(i, j, k))))) +
                this->xrhsy_ref->get(coord3(i, j, k));
            this->v_ref->set(coord3(i, j, k), this->v_pos->get(coord3(i, j, k)) + bottV * dt_small);
        }
    }

}

coord3 FastWavesRefBenchmark::inner_coord(coord3 coord){
    return coord + this->halo;
}

#endif