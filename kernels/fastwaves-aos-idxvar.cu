template<typename value_t>
__global__
void fastwaves_aos_idxvar(const coord3 max_coord,
                          GRID_ARGS
                          const fastwaves_aos_val<value_t> *inputs,
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

    const int idx_0_0_n1  = INDEX(i, j, k-1);
    PROTO(idx_0_0_n1);
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
        ppgu = inputs[idx_p1_0_0].ppuv - inputs[idx_0_0_0].ppuv;
        ppgv = inputs[idx_0_p1_0].ppuv - inputs[idx_0_0_0].ppuv;
    } else {
        value_t ppgk_0_0_0, ppgk_p1_0_0, ppgk_0_p1_0, ppgk_0_0_p1, ppgk_p1_0_p1, ppgk_0_p1_p1;
        value_t ppgc_0_0_0, ppgc_p1_0_0, ppgc_0_p1_0;
        ppgk_0_0_0   = inputs[idx_0_0_0].wgtfac * inputs[idx_0_0_0].ppuv +
                        (1.0 - inputs[idx_0_0_0].wgtfac) * inputs[idx_0_0_n1].ppuv;
        ppgk_p1_0_0  = inputs[idx_p1_0_0].wgtfac * inputs[idx_p1_0_0].ppuv +
                        (1.0 - inputs[idx_p1_0_0].wgtfac) * inputs[idx_p1_0_n1].ppuv;
        ppgk_0_p1_0  = inputs[idx_0_p1_0].wgtfac * inputs[idx_0_p1_0].ppuv +
                        (1.0 - inputs[idx_0_p1_0].wgtfac) * inputs[idx_0_p1_n1].ppuv;
        ppgk_0_0_p1  = inputs[idx_0_0_p1].wgtfac * inputs[idx_0_0_p1].ppuv +
                        (1.0 - inputs[idx_0_0_p1].wgtfac) * inputs[idx_0_0_0].ppuv;
        ppgk_p1_0_p1 = inputs[idx_p1_0_p1].wgtfac * inputs[idx_p1_0_p1].ppuv +
                        (1.0 - inputs[idx_p1_0_p1].wgtfac) * inputs[idx_p1_0_0].ppuv;
        ppgk_0_p1_p1 = inputs[idx_0_p1_p1].wgtfac * inputs[idx_0_p1_p1].ppuv +
                        (1.0 - inputs[idx_0_p1_p1].wgtfac) * inputs[idx_0_p1_0].ppuv;
        ppgc_0_0_0  = ppgk_0_0_p1  - ppgk_0_0_0;
        ppgc_p1_0_0 = ppgk_p1_0_p1 - ppgk_p1_0_0;
        ppgc_0_p1_0 = ppgk_0_p1_p1 - ppgk_0_p1_0;
        ppgu =
            (inputs[idx_p1_0_0].ppuv - inputs[idx_0_0_0].ppuv) + (ppgc_p1_0_0 + ppgc_0_0_0) * 0.5 * 
            ((inputs[idx_0_0_p1].hhl + inputs[idx_0_0_0].hhl) - (inputs[idx_p1_0_p1].hhl + inputs[idx_p1_0_0].hhl)) / 
            ((inputs[idx_0_0_p1].hhl - inputs[idx_0_0_0].hhl) + (inputs[idx_p1_0_p1].hhl - inputs[idx_p1_0_0].hhl));
        ppgv =
            (inputs[idx_0_p1_0].ppuv - inputs[idx_0_0_0].ppuv) + (ppgc_0_p1_0 + ppgc_0_0_0) * 0.5 *
            ((inputs[idx_0_0_p1].hhl + inputs[idx_0_0_0].hhl) - (inputs[idx_0_p1_p1].hhl + inputs[idx_0_p1_0].hhl)) /
            ((inputs[idx_0_0_p1].hhl - inputs[idx_0_0_0].hhl) + (inputs[idx_0_p1_p1].hhl - inputs[idx_0_p1_0].hhl));
    }

    // out
    uout[idx_0_0_0] =
        inputs[idx_0_0_0].u_in + dt_small * (inputs[idx_0_0_0].u_tens - ppgu * 
        (inputs[idx_0_0_0].fx / (0.5 * (inputs[idx_p1_0_0].rho + inputs[idx_0_0_0].rho))));
    vout[idx_0_0_0] =
        inputs[idx_0_0_0].v_in + dt_small * (inputs[idx_0_0_0].v_tens - ppgv * 
        (edadlat / (0.5 * ((inputs[idx_0_p1_0].rho + inputs[idx_0_0_0].rho)))));
}