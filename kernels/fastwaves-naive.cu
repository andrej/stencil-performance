template<typename value_t>
__global__
void fastwaves_naive(const FastWavesBenchmark::Info info,
                     GRID_ARGS
                     const value_t *ppuv,
                     const value_t *wgtfac,
                     const value_t *hhl,
                     const value_t *vin,
                     const value_t *uin,
                     const value_t *vtens,
                     const value_t *utens, 
                     const value_t *rho,
                     const value_t *fx,
                     const double edadlat,
                     const double dt_small,
                     const int c_flat_limit,
                     value_t *uout,
                     value_t *vout) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + info.halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + info.halo.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z + info.halo.z;
    if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z - 1) {
        return;
    }

    value_t ppgu, ppgv;
    if(k < c_flat_limit + info.halo.z) {
        ppgu = ppuv[NEIGHBOR(i, j, k, +1, 0, 0)] - ppuv[INDEX(i, j, k)];
        ppgv = ppuv[NEIGHBOR(i, j, k, 0, +1, 0)] - ppuv[INDEX(i, j, k)];
    } else {
        value_t ppgk_0_0_0, ppgk_p1_0_0, ppgk_0_p1_0, ppgk_0_0_p1, ppgk_p1_0_p1, ppgk_0_p1_p1;
        value_t ppgc_0_0_0, ppgc_p1_0_0, ppgc_0_p1_0;
        ppgk_0_0_0   = wgtfac[INDEX(i, j, k)] * ppuv[INDEX(i, j, k)] +
                       (1.0 - wgtfac[INDEX(i, j, k)]) * ppuv[NEIGHBOR(i, j, k, 0, 0, -1)];
        ppgk_p1_0_0  = wgtfac[NEIGHBOR(i, j, k, +1, 0, 0)] * ppuv[NEIGHBOR(i, j, k, +1, 0, 0)] +
                       (1.0 - wgtfac[NEIGHBOR(i, j, k, +1, 0, 0)]) * ppuv[NEIGHBOR(i, j, k, +1, 0, -1)];
        ppgk_0_p1_0  = wgtfac[NEIGHBOR(i, j, k, 0, +1, 0)] * ppuv[NEIGHBOR(i, j, k, 0, +1, 0)] +
                       (1.0 - wgtfac[NEIGHBOR(i, j, k, 0, +1, 0)]) * ppuv[NEIGHBOR(i, j, k, 0, +1, -1)];
        ppgk_0_0_p1  = wgtfac[NEIGHBOR(i, j, k, 0, 0, +1)] * ppuv[NEIGHBOR(i, j, k, 0, 0, +1)] +
                       (1.0 - wgtfac[NEIGHBOR(i, j, k, 0, 0, +1)]) * ppuv[INDEX(i, j, k)];
        ppgk_p1_0_p1 = wgtfac[NEIGHBOR(i, j, k, +1, 0, +1)] * ppuv[NEIGHBOR(i, j, k, +1, 0, +1)] +
                       (1.0 - wgtfac[NEIGHBOR(i, j, k, +1, 0, +1)]) * ppuv[NEIGHBOR(i, j, k, +1, 0, 0)];
        ppgk_0_p1_p1 = wgtfac[NEIGHBOR(i, j, k, 0, +1, +1)] * ppuv[NEIGHBOR(i, j, k, 0, +1, +1)] +
                       (1.0 - wgtfac[NEIGHBOR(i, j, k, 0, +1, +1)]) * ppuv[NEIGHBOR(i, j, k, 0, +1, 0)];
        ppgc_0_0_0  = ppgk_0_0_p1  - ppgk_0_0_0;
        ppgc_p1_0_0 = ppgk_p1_0_p1 - ppgk_p1_0_0;
        ppgc_0_p1_0 = ppgk_0_p1_p1 - ppgk_0_p1_0;
        ppgu =
            (ppuv[NEIGHBOR(i, j, k, +1, 0, 0)] - ppuv[INDEX(i, j, k)]) + (ppgc_p1_0_0 + ppgc_0_0_0) * 0.5 * 
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] + hhl[INDEX(i, j, k)]) - (hhl[NEIGHBOR(i, j, k, +1, 0, +1)] + hhl[NEIGHBOR(i, j, k, +1, 0, 0)])) / 
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] - hhl[INDEX(i, j, k)]) + (hhl[NEIGHBOR(i, j, k, +1, 0, +1)] - hhl[NEIGHBOR(i, j, k, +1, 0, 0)]));
        ppgv =
            (ppuv[NEIGHBOR(i, j, k, 0, +1, 0)] - ppuv[INDEX(i, j, k)]) + (ppgc_0_p1_0 + ppgc_0_0_0) * 0.5 *
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] + hhl[INDEX(i, j, k)]) - (hhl[NEIGHBOR(i, j, k, 0, +1, +1)] + hhl[NEIGHBOR(i, j, k, 0, +1, 0)])) /
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] - hhl[INDEX(i, j, k)]) + (hhl[NEIGHBOR(i, j, k, 0, +1, +1)] - hhl[NEIGHBOR(i, j, k, 0, +1, 0)]));
    }

    // out
    uout[INDEX(i, j, k)] =
        uin[INDEX(i, j, k)] + dt_small * (utens[INDEX(i, j, k)] - ppgu * 
        (fx[INDEX(i, j, k)] / (0.5 * (rho[NEIGHBOR(i, j, k, +1, 0, 0)] + rho[INDEX(i, j, k)]))));
    vout[INDEX(i, j, k)] =
        vin[INDEX(i, j, k)] + dt_small * (vtens[INDEX(i, j, k)] - ppgv * 
        (edadlat / (0.5 * ((rho[NEIGHBOR(i, j, k, 0, +1, 0)] + rho[INDEX(i, j, k)])))));
    }