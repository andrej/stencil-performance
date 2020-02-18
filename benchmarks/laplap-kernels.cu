#define GRID_ARGS const neigh_ptr_t * __restrict__ neighborships, const int z_stride, const int offs,
#define PROTO(x)
#define INDEX(x_, y_, z_) (x_) + (y_)*blockDim.x*gridDim.x + offs + (z_)*z_stride
#define IS_IN_BOUNDS(i, j, k) (i + j*blockDim.x*gridDim.x < (z_stride-offs) && k < max_coord.z)
#define Z_NEIGHBOR(idx, z) (idx+z*z_stride)
#define K_STEP k*z_stride

namespace NonChasing {
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1)+(x2), (y1)+(y2), (z1)+(z2))
    #include "kernels/laplap-naive.cu"
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR

    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, idx, x_, y_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(idx, (x1)+(x2), (y1)+(y2), (z1)+(z2))
    #include "kernels/laplap-idxvar.cu"
    #include "kernels/laplap-idxvar-kloop.cu"
    #include "kernels/laplap-idxvar-kloop-sliced.cu"
    #include "kernels/laplap-idxvar-shared.cu"
    #undef DOUBLE_NEIGHBOR
    #undef NEIGHBOR
};

namespace Chasing {
    #define CHASING
    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_NEIGHBOR(neighborships, z_stride, idx, x_, y_, z_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
    #include "kernels/laplap-naive.cu"
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR

    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, idx, x_, y_)
    #define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
    #include "kernels/laplap-idxvar.cu"
    #include "kernels/laplap-idxvar-kloop.cu"
    #include "kernels/laplap-idxvar-kloop-sliced.cu"
    #include "kernels/laplap-idxvar-shared.cu"
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR
    #undef CHASING
};

#undef GRID_ARGS
#undef PROTO

#define GRID_ARGS neigh_ptr_t * prototypes, neigh_ptr_t * neighborships, const int z_stride, const int neigh_stride, const int offs,
#define PROTO(idx) int idx ## _proto = prototypes[idx]

namespace Compressed {
    #define CHASING
    #define CHASING
    //#define NEIGHBOR(idx, x_, y_, _z) GRID_UNSTR_PROTO_NEIGHBOR(prototypes, neighborships, z_stride, neigh_stride, idx, x_, y_, z_)
    //#define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
    //#include "kernels/laplap-naive.cu"
    #undef NEIGHBOR
    //#undef DOUBLE_NEIGHBOR

    #define NEIGHBOR(idx, x_, y_, z_) GRID_UNSTR_2D_NEIGHBOR_(neighborships, neigh_stride, idx, idx ## _proto, x_, y_)
    //#define DOUBLE_NEIGHBOR(idx, x1, y1, z1, x2, y2, z2) NEIGHBOR(NEIGHBOR(idx, x1, y1, z1), x2, y2, z2)
    #include "kernels/laplap-idxvar.cu"
    //#include "kernels/laplap-idxvar-kloop.cu"
    //#include "kernels/laplap-idxvar-kloop-sliced.cu"
    //#include "kernels/laplap-idxvar-shared.cu"
    #undef NEIGHBOR
    #undef DOUBLE_NEIGHBOR
    #undef CHASING
};

#undef GRID_ARGS
#undef INDEX
#undef PROTO
#undef IS_IN_BOUNDS