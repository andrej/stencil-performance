template<typename value_t>
__global__
void fastwaves_kloop(const FastWavesBenchmark::Info info,
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
    const int k_start = info.halo.z;
    if(i >= info.max_coord.x || j >= info.max_coord.y || k_start >= info.max_coord.z - 1) {
        return;
    }

    // first iteration indices will be passed into -> 0
    int idx_0_0_0, idx_0_0_n1, idx_0_0_p1, idx_p1_0_n1, idx_p1_0_0, idx_p1_0_p1, idx_0_p1_n1, idx_0_p1_0, idx_0_p1_p1;
    idx_0_0_n1    = NEIGHBOR(i, j, k_start, 0, 0, -1);
    idx_0_0_0     = NEXT_Z_NEIGHBOR_OF_INDEX(idx_0_0_n1);
    idx_0_0_p1    = NEXT_Z_NEIGHBOR_OF_INDEX(idx_0_0_0);
    idx_p1_0_n1   = NEIGHBOR_OF_INDEX(idx_0_0_n1, +1, 0, 0);
    idx_p1_0_0    = NEXT_Z_NEIGHBOR_OF_INDEX(idx_p1_0_n1);
    idx_p1_0_p1   = NEXT_Z_NEIGHBOR_OF_INDEX(idx_p1_0_0);
    idx_0_p1_n1   = NEIGHBOR_OF_INDEX(idx_0_0_n1, 0, +1, 0);
    idx_0_p1_0    = NEXT_Z_NEIGHBOR_OF_INDEX(idx_0_p1_n1);
    idx_0_p1_p1   = NEXT_Z_NEIGHBOR_OF_INDEX(idx_0_p1_0);

    value_t ppgk_0_0_0, ppgk_p1_0_0, ppgk_0_p1_0, ppgk_0_0_p1, ppgk_p1_0_p1, ppgk_0_p1_p1;
    value_t ppgc_0_0_0, ppgc_p1_0_0, ppgc_0_p1_0;

    for(int k = k_start; k < info.max_coord.z - 1; k++) {
        // ppgu, ppgv
        value_t ppgu, ppgv;
        if(k < c_flat_limit + info.halo.z) {
            ppgu = ppuv[idx_p1_0_0] - ppuv[idx_0_0_0];
            ppgv = ppuv[idx_0_p1_0] - ppuv[idx_0_0_0];
        } else {
            if(k == c_flat_limit + info.halo.z) {
                // first iteration, need to compute all dependencies
                ppgk_0_0_0   = wgtfac[idx_0_0_0] * ppuv[idx_0_0_0] +
                               (1.0 - wgtfac[idx_0_0_0]) * ppuv[idx_0_0_n1];
                ppgk_p1_0_0  = wgtfac[idx_p1_0_0] * ppuv[idx_p1_0_0] +
                               (1.0 - wgtfac[idx_p1_0_0]) * ppuv[idx_p1_0_n1];
                ppgk_0_p1_0  = wgtfac[idx_0_p1_0] * ppuv[idx_0_p1_0] +
                               (1.0 - wgtfac[idx_0_p1_0]) * ppuv[idx_0_p1_n1];
            } else {
                // pass-through results from previous iteration
                ppgk_0_0_0   = ppgk_0_0_p1;
                ppgk_p1_0_0  = ppgk_p1_0_p1;
                ppgk_0_p1_0  = ppgk_0_p1_p1;
            }
            ppgk_0_0_p1  = wgtfac[idx_0_0_p1] * ppuv[idx_0_0_p1] +
                            (1.0 - wgtfac[idx_0_0_p1]) * ppuv[idx_0_0_0];
            ppgk_p1_0_p1 = wgtfac[idx_p1_0_p1] * ppuv[idx_p1_0_p1] +
                            (1.0 - wgtfac[idx_p1_0_p1]) * ppuv[idx_p1_0_0];
            ppgk_0_p1_p1 = wgtfac[idx_0_p1_p1] * ppuv[idx_0_p1_p1] +
                            (1.0 - wgtfac[idx_0_p1_p1]) * ppuv[idx_0_p1_0];
            ppgc_0_0_0   = ppgk_0_0_p1  - ppgk_0_0_0;
            ppgc_p1_0_0  = ppgk_p1_0_p1 - ppgk_p1_0_0;
            ppgc_0_p1_0  = ppgk_0_p1_p1 - ppgk_0_p1_0;
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
        
        // pass through indices from previous iteration
        idx_0_0_n1  = idx_0_0_0;
        idx_0_0_0   = idx_0_0_p1;
        idx_p1_0_n1 = idx_p1_0_0;
        idx_p1_0_0  = idx_p1_0_p1;
        idx_0_p1_n1 = idx_0_p1_0;
        idx_0_p1_0  = idx_0_p1_p1;
        idx_0_0_p1  = NEXT_Z_NEIGHBOR_OF_INDEX(idx_0_0_0);
        idx_p1_0_p1 = NEXT_Z_NEIGHBOR_OF_INDEX(idx_p1_0_0);
        idx_0_p1_p1 = NEXT_Z_NEIGHBOR_OF_INDEX(idx_0_p1_0);
    }
}