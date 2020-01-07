// ppgk: First part of the PPGradCorStage (not including difference on line 214)
template<typename value_t>
__global__
void fastwaves_ppgk(const FastWavesBenchmark::Info info,
                    const int c_flat_limit,
                    GRID_ARGS
                    const value_t *ppuv,
                    const value_t *wgtfac,
                    value_t *out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + info.halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + info.halo.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z + c_flat_limit + info.halo.z;
    if(i >= info.max_coord.x + 1 || j >= info.max_coord.y + 1 || k >= info.max_coord.z) {
        return;
    }

    out[INDEX(i, j, k)] =
        wgtfac[INDEX(i, j, k)] * ppuv[INDEX(i, j, k)] +
        (1.0 - wgtfac[INDEX(i, j, k)]) * ppuv[NEIGHBOR(i, j, k, 0, 0, -1)];
}

// ppgc: PPGradCorStage taking difference of previous results
template<typename value_t>
__global__
void fastwaves_ppgc(const FastWavesBenchmark::Info info,
                    GRID_ARGS
                    int c_flat_limit,
                    const value_t *ppgk,
                    value_t *ppgc) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + info.halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + info.halo.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z + c_flat_limit + info.halo.z;
    if(i >= info.max_coord.x + 1 || j >= info.max_coord.y + 1 || k >= info.max_coord.z) {
        return;
    }

    // FIXME race condition!

    if(k >= c_flat_limit + info.halo.z) {
        ppgc[INDEX(i, j, k)] =
            ppgk[NEIGHBOR(i, j, k, 0, 0, +1)] - ppgk[INDEX(i, j, k)];
    } else {
        ppgc[INDEX(i, j, k)] = ppgk[INDEX(i, j, k)];
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
        
    const int i = blockIdx.x*blockDim.x + threadIdx.x + info.halo.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + info.halo.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z + info.halo.z;
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
    if(k < c_flat_limit + info.halo.z) {
        ppgu = ppuv[NEIGHBOR(i, j, k, +1, 0, 0)] - ppuv[INDEX(i, j, k)];
        ppgv = ppuv[NEIGHBOR(i, j, k, 0, +1, 0)] - ppuv[INDEX(i, j, k)];
    } else {
        ppgu =
            (ppuv[NEIGHBOR(i, j, k, +1, 0, 0)] - ppuv[INDEX(i, j, k)]) + (ppgc[NEIGHBOR(i, j, k, +1, 0, 0)] + ppgc[INDEX(i, j, k)]) * 0.5 * 
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] + hhl[INDEX(i, j, k)]) - (hhl[NEIGHBOR(i, j, k, +1, 0, +1)] + hhl[NEIGHBOR(i, j, k, +1, 0, 0)])) / 
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] - hhl[INDEX(i, j, k)]) + (hhl[NEIGHBOR(i, j, k, +1, 0, +1)] - hhl[NEIGHBOR(i, j, k, +1, 0, 0)]));
        ppgv =
            (ppuv[NEIGHBOR(i, j, k, 0, +1, 0)] - ppuv[INDEX(i, j, k)]) + (ppgc[NEIGHBOR(i, j, k, 0, +1, 0)] + ppgc[INDEX(i, j, k)]) * 0.5 *
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] + hhl[INDEX(i, j, k)]) - (hhl[NEIGHBOR(i, j, k, 0, +1, +1)] + hhl[NEIGHBOR(i, j, k, 0, +1, 0)])) /
            ((hhl[NEIGHBOR(i, j, k, 0, 0, +1)] - hhl[INDEX(i, j, k)]) + (hhl[NEIGHBOR(i, j, k, 0, +1, +1)] - hhl[NEIGHBOR(i, j, k, 0, +1, 0)]));
    }

    uout[INDEX(i, j, k)] =
        uin[INDEX(i, j, k)] + dt_small * (utens[INDEX(i, j, k)] - ppgu * 
        (fx[INDEX(i, j, k)] / (0.5 * (rho[NEIGHBOR(i, j, k, +1, 0, 0)] + rho[INDEX(i, j, k)]))));
    vout[INDEX(i, j, k)] =
        vin[INDEX(i, j, k)] + dt_small * (vtens[INDEX(i, j, k)] - ppgv * 
        (edadlat / (0.5 * ((rho[NEIGHBOR(i, j, k, 0, +1, 0)] + rho[INDEX(i, j, k)])))));

}