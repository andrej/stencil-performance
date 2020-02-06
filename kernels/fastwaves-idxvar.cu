template<typename value_t>
__global__
void fastwaves_idxvar(const coord3 max_coord,
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
    if(NOT_IN_BOUNDS(i, j, k)) {
        return;
    }

    const int idx = INDEX(i, j, k);

    const int idx_0_0_n1  = NEIGHBOR(idx, 0, 0, -1);
    const int idx_0_0_0   = NEXT_Z_NEIGHBOR(idx_0_0_n1);
    const int idx_0_0_p1  = NEXT_Z_NEIGHBOR(idx_0_0_0);
    const int idx_p1_0_n1 = NEIGHBOR(idx_0_0_n1, +1, 0, 0);
    const int idx_p1_0_0  = NEXT_Z_NEIGHBOR(idx_p1_0_n1);
    const int idx_p1_0_p1 = NEXT_Z_NEIGHBOR(idx_p1_0_0);
    const int idx_0_p1_n1 = NEIGHBOR(idx_0_0_n1, 0, +1, 0);
    const int idx_0_p1_0  = NEXT_Z_NEIGHBOR(idx_0_p1_n1);
    const int idx_0_p1_p1 = NEXT_Z_NEIGHBOR(idx_0_p1_0);

    // ppgu, ppgv
    value_t ppgu, ppgv;
    if(k < c_flat_limit) {
        ppgu = ppuv[idx_p1_0_0] - ppuv[idx_0_0_0];
        ppgv = ppuv[idx_0_p1_0] - ppuv[idx_0_0_0];
    } else {
        value_t ppgk_0_0_0, ppgk_p1_0_0, ppgk_0_p1_0, ppgk_0_0_p1, ppgk_p1_0_p1, ppgk_0_p1_p1;
        value_t ppgc_0_0_0, ppgc_p1_0_0, ppgc_0_p1_0;
        ppgk_0_0_0   = wgtfac[idx_0_0_0] * ppuv[idx_0_0_0] +
                        (1.0 - wgtfac[idx_0_0_0]) * ppuv[idx_0_0_n1];
        ppgk_p1_0_0  = wgtfac[idx_p1_0_0] * ppuv[idx_p1_0_0] +
                        (1.0 - wgtfac[idx_p1_0_0]) * ppuv[idx_p1_0_n1];
        ppgk_0_p1_0  = wgtfac[idx_0_p1_0] * ppuv[idx_0_p1_0] +
                        (1.0 - wgtfac[idx_0_p1_0]) * ppuv[idx_0_p1_n1];
        ppgk_0_0_p1  = wgtfac[idx_0_0_p1] * ppuv[idx_0_0_p1] +
                        (1.0 - wgtfac[idx_0_0_p1]) * ppuv[idx_0_0_0];
        ppgk_p1_0_p1 = wgtfac[idx_p1_0_p1] * ppuv[idx_p1_0_p1] +
                        (1.0 - wgtfac[idx_p1_0_p1]) * ppuv[idx_p1_0_0];
        ppgk_0_p1_p1 = wgtfac[idx_0_p1_p1] * ppuv[idx_0_p1_p1] +
                        (1.0 - wgtfac[idx_0_p1_p1]) * ppuv[idx_0_p1_0];
        ppgc_0_0_0  = ppgk_0_0_p1  - ppgk_0_0_0;
        ppgc_p1_0_0 = ppgk_p1_0_p1 - ppgk_p1_0_0;
        ppgc_0_p1_0 = ppgk_0_p1_p1 - ppgk_0_p1_0;
        ppgu =
            (ppuv[idx_p1_0_0] - ppuv[idx_0_0_0]) + (ppgc_p1_0_0 + ppgc_0_0_0) * 0.5 * 
            ((hhl[idx_0_0_p1] + hhl[idx_0_0_0]) - (hhl[idx_p1_0_p1] + hhl[idx_p1_0_0])) / 
            ((hhl[idx_0_0_p1] - hhl[idx_0_0_0]) + (hhl[idx_p1_0_p1] - hhl[idx_p1_0_0]));
        ppgv =
            (ppuv[idx_0_p1_0] - ppuv[idx_0_0_0]) + (ppgc_0_p1_0 + ppgc_0_0_0) * 0.5 *
            ((hhl[idx_0_0_p1] + hhl[idx_0_0_0]) - (hhl[idx_0_p1_p1] + hhl[idx_0_p1_0])) /
            ((hhl[idx_0_0_p1] - hhl[idx_0_0_0]) + (hhl[idx_0_p1_p1] - hhl[idx_0_p1_0]));
    }

    // out
    uout[idx_0_0_0] =
        uin[idx_0_0_0] + dt_small * (utens[idx_0_0_0] - ppgu * 
        (fx[idx_0_0_0] / (0.5 * (rho[idx_p1_0_0] + rho[idx_0_0_0]))));
    vout[idx_0_0_0] =
        vin[idx_0_0_0] + dt_small * (vtens[idx_0_0_0] - ppgv * 
        (edadlat / (0.5 * ((rho[idx_0_p1_0] + rho[idx_0_0_0])))));
}