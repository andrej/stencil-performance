#ifndef CUDA_UNSTRUCTURED_GRID_H
#define CUDA_UNSTRUCTURED_GRID_H
#include "unstructured.cu"

/** This struct is used to pass grid information to the kernel (because we
 * cannot use the C++ classes inside the kernel). Pass this struct
 * to the appropriate kernel macros.
 */
template<typename value_t>
struct CudaUnstructuredGrid3DInfo {
    int *neighbor_data;
    struct {
        int y;
        int z;
    } strides;
};

/** Cuda version of the unstructured grid
 *
 * Lays out the grid in Cuda unified memory and provides compiler macros for
 * indexing/setting/getting values in the grid.
 */
 template<typename value_t>
 class CudaUnstructuredGrid3D : 
 virtual public UnstructuredGrid3D<value_t>,
 virtual public CudaBaseGrid<value_t, coord3>
 {

    public:

    CudaUnstructuredGrid3D(coord3 dimensions);
    CudaUnstructuredGrid3D(coord3 dimensions, int *neighbor_data);

    CudaUnstructuredGrid3DInfo<value_t> get_gridinfo();

    static CudaUnstructuredGrid3D<value_t> *create_regular(coord3 dimensions);

};

/** Use this macro instead of the CudaUnstructuredGrid3D::index() function for
 * getting the index of some coordinates inside the kernel. */
#define CUDA_UNSTR_INDEX(grid_info, _x, _y, _z) \
    ((int)  (  (int)(_x) \
             + (int)(_y) * grid_info.strides.y \
             + (int)(_z) * grid_info.strides.z ))

/** Use this macro instead of the CudaUnstructuredGrid3D::neighbor() function for
 * getting the index of some coordinate offset from within the kernel. 
 *
 * Note only ONE of X, Y, Z may be -1 or +1 for this to work! Because
 * neighborship relations are stored only for direct neighbors! */
#define CUDA_UNSTR_NEIGHBOR_PTR_AT(grid_info, index, x, y, _z) \
        ((index % grid_info.strides.z) * 4 \
                + 1 * (y == -1) \
                + 2 * (x == +1) \
                + 3 * (y == +1))

#define CUDA_UNSTR_NEIGHBOR_PTR(grid_info, _x, _y, _z, neigh_x, neigh_y, neigh_z) \
        (CUDA_UNSTR_NEIGHBOR_PTR_AT(grid_info, CUDA_UNSTR_INDEX(grid_info, _x, _y, _z), neigh_x, neigh_y, neigh_z))

/** Gives the index of the neighbor. */
#define CUDA_UNSTR_NEIGHBOR_AT(grid_info, index, x, y, _z) \
        ((int)(grid_info.neighbor_data[CUDA_UNSTR_NEIGHBOR_PTR_AT(grid_info, index, x, y, _z)]\
               + (index / grid_info.strides.z) * grid_info.strides.z \
               + _z*grid_info.strides.z))

#define CUDA_UNSTR_NEIGHBOR(grid_info, _x, _y, _z, neigh_x, neigh_y, neigh_z) \
        ((int)(grid_info.neighbor_data[CUDA_UNSTR_NEIGHBOR_PTR(grid_info, _x, _y, _z, neigh_x, neigh_y, neigh_z)] \
               + _z*grid_info.strides.z \
               + neigh_z*grid_info.strides.z))


template<typename value_t>
CudaUnstructuredGrid3D<value_t>::CudaUnstructuredGrid3D(coord3 dimensions) :
Grid<value_t, coord3>(dimensions,
                UnstructuredGrid3D<value_t>::space_req(dimensions)) {
    this->neighbor_data = NULL;
    this->init();
}

template<typename value_t>
CudaUnstructuredGrid3D<value_t>::CudaUnstructuredGrid3D(coord3 dimensions, int *neighbor_data) :
Grid<value_t, coord3>(dimensions,
                UnstructuredGrid3D<value_t>::space_req(dimensions, true)) {
    this->neighbor_data = neighbor_data;
    this->init();
}

template<typename value_t>
CudaUnstructuredGrid3DInfo<value_t> CudaUnstructuredGrid3D<value_t>::get_gridinfo() {
    coord3 dimensions = this->dimensions;
    return {.neighbor_data = this->neighbor_data,
            .strides = { .y = (int)dimensions.x,
                         .z = (int)dimensions.x*(int)dimensions.y } };
}

template<typename value_t>
CudaUnstructuredGrid3D<value_t> *CudaUnstructuredGrid3D<value_t>::create_regular(coord3 dimensions) {
    CudaUnstructuredGrid3D<value_t> *grid = new CudaUnstructuredGrid3D<value_t>(dimensions);
    grid->add_regular_neighbors();
    return grid;
}

#endif