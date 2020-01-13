// ppgk: First part of the PPGradCorStage (not including difference on line 214)
template<typename value_t>
__global__
void fastwaves_ppgk(const FastWavesBenchmark::Info info,
                    const int c_flat_limit,
                    GRID_ARGS
                    const value_t *ppuv,
                    const value_t *wgtfac,
                    value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z + c_flat_limit;
    if(i >= info.max_coord.x + 1 || j >= info.max_coord.y + 1 || k >= info.max_coord.z) {
        return;
    }

    const int idx = INDEX(i, j, k);

    out[idx] =
        wgtfac[idx] * ppuv[idx] +
        (1.0 - wgtfac[idx]) * ppuv[NEIGHBOR(idx, 0, 0, -1)];
}

// ppgc: PPGradCorStage taking difference of previous results
template<typename value_t>
__global__
void fastwaves_ppgc(const FastWavesBenchmark::Info info,
                    GRID_ARGS
                    int c_flat_limit,
                    const value_t *ppgk,
                    value_t *ppgc) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z + c_flat_limit;
    if(i >= info.max_coord.x + 1 || j >= info.max_coord.y + 1 || k >= info.max_coord.z) {
        return;
    }

    const int idx = INDEX(i, j, k);

    // FIXME race condition!

    if(k == info.max_coord.z - 1) {
        ppgc[idx] = ppgk[idx];
    } else {
        ppgc[idx] =
            ppgk[NEIGHBOR(idx, 0, 0, +1)] - ppgk[idx];
    }
    
}

// PPGradStage + UV fused (there are no complicated dependencies)
template<typename value_t>
__global__
void fastwaves_ppgrad_uv(const FastWavesBenchmark::Info info,
                         GRID_ARGS
                         const value_t *ppuv,
                         const value_t *ppgc,
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
    if(i >= info.max_coord.x || j >= info.max_coord.y || k >= info.max_coord.z - 1) {
        return;
    }

    /* "ppgu" :
        "auto res = "
        "(ppuv(i+1,j,k) - ppuv(i,j,k)) + (ppgc(i+1,j,k) + ppgc(i,j,k)) * 0.5 * "
        "((hhl(i,j,k+1) + hhl(i,j,k)) - (hhl(i+1,j,k+1) + hhl(i+1,j,k))) / "
        "((hhl(i,j,k+1) - hhl(i,j,k)) + (hhl(i+1,j,k+1) - hhl(i+1,j,k)));",
    "ppgv" :
        "auto res = "
        "(ppuv(i,j+1,k) - ppuv(i,j,k)) + (ppgc(i,j+1,k) + ppgc(i,j,k)) * 0.5 * "
        "((hhl(i,j,k+1) + hhl(i,j,k)) - (hhl(i,j+1,k+1) + hhl(i,j+1,k))) / "
        "((hhl(i,j,k+1) - hhl(i,j,k)) + (hhl(i,j+1,k+1) - hhl(i,j+1,k)));",
    "uout" :
        "auto res = uin(i,j,k) + 0.01 * (utens(i,j,k) - ppgu(i,j,k) * "
        "(2.0 / (rho(i+1,j,k) + rho(i,j,k))));",
    "vout" :
        "auto res = vin(i,j,k) + 0.01 * (vtens(i,j,k) - ppgv(i,j,k) * "
        "(2.0 / (rho(i,j+1,k) + rho(i,j,k))));" */

    value_t ppgu, ppgv;
    if(k < c_flat_limit) {
        ppgu = ppuv[NEIGHBOR(idx, +1, 0, 0)] - ppuv[idx];
        ppgv = ppuv[NEIGHBOR(idx, 0, +1, 0)] - ppuv[idx];
    } else {
        ppgu =
            (ppuv[NEIGHBOR(idx, +1, 0, 0)] - ppuv[idx]) + (ppgc[NEIGHBOR(idx, +1, 0, 0)] + ppgc[idx]) * 0.5 * 
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] + hhl[idx]) - (hhl[NEIGHBOR(idx, +1, 0, +1)] + hhl[NEIGHBOR(idx, +1, 0, 0)])) / 
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] - hhl[idx]) + (hhl[NEIGHBOR(idx, +1, 0, +1)] - hhl[NEIGHBOR(idx, +1, 0, 0)]));
        ppgv =
            (ppuv[NEIGHBOR(idx, 0, +1, 0)] - ppuv[idx]) + (ppgc[NEIGHBOR(idx, 0, +1, 0)] + ppgc[idx]) * 0.5 *
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] + hhl[idx]) - (hhl[NEIGHBOR(idx, 0, +1, +1)] + hhl[NEIGHBOR(idx, 0, +1, 0)])) /
            ((hhl[NEIGHBOR(idx, 0, 0, +1)] - hhl[idx]) + (hhl[NEIGHBOR(idx, 0, +1, +1)] - hhl[NEIGHBOR(idx, 0, +1, 0)]));
    }

    uout[idx] =
        uin[idx] + dt_small * (utens[idx] - ppgu * 
        (fx[idx] / (0.5 * (rho[NEIGHBOR(idx, +1, 0, 0)] + rho[idx]))));
    vout[idx] =
        vin[idx] + dt_small * (vtens[idx] - ppgv * 
        (edadlat / (0.5 * ((rho[NEIGHBOR(idx, 0, +1, 0)] + rho[idx])))));

}