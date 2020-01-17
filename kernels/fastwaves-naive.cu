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
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(!(IS_IN_BOUNDS(i, j, k))) {
        return;
    }

    const int idx = INDEX(i, j, k);
    value_t ppgu, ppgv;
    if(k < c_flat_limit) {
        ppgu = ppuv[NEIGHBOR(idx, +1, 0, 0)] - ppuv[idx];
        ppgv = ppuv[NEIGHBOR(idx, 0, +1, 0)] - ppuv[idx];
    } else {
        value_t ppgk_0_0_0, ppgk_p1_0_0, ppgk_0_p1_0, ppgk_0_0_p1, ppgk_p1_0_p1, ppgk_0_p1_p1;
        value_t ppgc_0_0_0, ppgc_p1_0_0, ppgc_0_p1_0;
        ppgk_0_0_0   = wgtfac[idx] * ppuv[idx] +
                       (1.0 - wgtfac[idx]) * ppuv[NEIGHBOR(idx, 0, 0, -1)];
        ppgk_p1_0_0  = wgtfac[NEIGHBOR(idx, +1, 0, 0)] * ppuv[NEIGHBOR(idx, +1, 0, 0)] +
                       (1.0 - wgtfac[NEIGHBOR(idx, +1, 0, 0)]) * ppuv[NEIGHBOR(idx, +1, 0, -1)];
        ppgk_0_p1_0  = wgtfac[NEIGHBOR(idx, 0, +1, 0)] * ppuv[NEIGHBOR(idx, 0, +1, 0)] +
                       (1.0 - wgtfac[NEIGHBOR(idx, 0, +1, 0)]) * ppuv[NEIGHBOR(idx, 0, +1, -1)];
        ppgk_0_0_p1  = wgtfac[NEIGHBOR(idx, 0, 0, +1)] * ppuv[NEIGHBOR(idx, 0, 0, +1)] +
                       (1.0 - wgtfac[NEIGHBOR(idx, 0, 0, +1)]) * ppuv[idx];
        ppgk_p1_0_p1 = wgtfac[NEIGHBOR(idx, +1, 0, +1)] * ppuv[NEIGHBOR(idx, +1, 0, +1)] +
                       (1.0 - wgtfac[NEIGHBOR(idx, +1, 0, +1)]) * ppuv[NEIGHBOR(idx, +1, 0, 0)];
        ppgk_0_p1_p1 = wgtfac[NEIGHBOR(idx, 0, +1, +1)] * ppuv[NEIGHBOR(idx, 0, +1, +1)] +
                       (1.0 - wgtfac[NEIGHBOR(idx, 0, +1, +1)]) * ppuv[NEIGHBOR(idx, 0, +1, 0)];
        ppgc_0_0_0  = ppgk_0_0_p1  - ppgk_0_0_0;
        ppgc_p1_0_0 = ppgk_p1_0_p1 - ppgk_p1_0_0;
        ppgc_0_p1_0 = ppgk_0_p1_p1 - ppgk_0_p1_0;
        ppgu =
            (ppuv[NEIGHBOR(idx, +1, 0, 0)] - ppuv[idx]) + (ppgc_p1_0_0 + ppgc_0_0_0) * 0.5 * 
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] + hhl[idx]) - (hhl[NEIGHBOR(idx, +1, 0, +1)] + hhl[NEIGHBOR(idx, +1, 0, 0)])) / 
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] - hhl[idx]) + (hhl[NEIGHBOR(idx, +1, 0, +1)] - hhl[NEIGHBOR(idx, +1, 0, 0)]));
        ppgv =
            (ppuv[NEIGHBOR(idx, 0, +1, 0)] - ppuv[idx]) + (ppgc_0_p1_0 + ppgc_0_0_0) * 0.5 *
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] + hhl[idx]) - (hhl[NEIGHBOR(idx, 0, +1, +1)] + hhl[NEIGHBOR(idx, 0, +1, 0)])) /
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] - hhl[idx]) + (hhl[NEIGHBOR(idx, 0, +1, +1)] - hhl[NEIGHBOR(idx, 0, +1, 0)]));
    }

    // out
    uout[idx] =
        uin[idx] + dt_small * (utens[idx] - ppgu * 
        (fx[idx] / (0.5 * (rho[NEIGHBOR(idx, +1, 0, 0)] + rho[idx]))));
    vout[idx] =
        vin[idx] + dt_small * (vtens[idx] - ppgv * 
        (edadlat / (0.5 * ((rho[NEIGHBOR(idx, 0, +1, 0)] + rho[idx])))));
    }