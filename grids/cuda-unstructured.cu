#ifndef CUDA_UNSTRUCTURED_GRID_H
#define CUDA_UNSTRUCTURED_GRID_H
#include "unstructured.cu"

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

    CudaGridInfo<value_t> get_gridinfo();

    static CudaUnstructuredGrid3D<value_t> *create_regular(coord3 dimensions);

};

template<typename value_t>
CudaUnstructuredGrid3D<value_t>::CudaUnstructuredGrid3D(coord3 dimensions) :
Grid<value_t, coord3>(),
UnstructuredGrid3D<value_t>(),
CudaBaseGrid<value_t, coord3>(dimensions, dimensions.x*dimensions.y*dimensions.z*5) {
    this->allocate();
}


#define CUDA_UNSTR(grid_info, coords) \
    (grid_info.data[CUDA_UNSTR_INDEX(grid_info, coords)])

#define CUDA_UNSTR_AT(grid_info, index) \
    (grid_info.data[index])

/** Use this to get the value of a neighbor. */
#define CUDA_UNSTR_NEIGH(grid_info, coords, x, y, z) \
    (grid_info.data[CUDA_UNSTR_NEIGHBOR(grid_info, coords, x, y, z)])

/** Use this macro instead of the CudaUnstructuredGrid3D::index() function for
 * getting the index of some coordinates inside the kernel. */
#define CUDA_UNSTR_INDEX(grid_info, coords) \
    ((int)  (  coords.x * grid_info.strides.x \
             + coords.y * grid_info.strides.y \
             + coords.z * grid_info.strides.z ))

/** Use this macro instead of the CudaUnstructuredGrid3D::neighbor() function for
 * getting the index of some coordinate offset from within the kernel. 
 *
 * Note only ONE of X, Y, Z may be -1 or +1 for this to work! Because
 * neighborship relations are stored only for direct neighbors! */
#define CUDA_UNSTR_NEIGHBOR_PTR_AT(grid_info, index, x, y, _z) \
        (index  + _z * grid_info.strides.z  \
                +     (x == -1) \
                + 2 * (y == -1) \
                + 3 * (x == +1) \
                + 4 * (y == +1))
#define CUDA_UNSTR_NEIGHBOR_PTR(grid_info, coords, x, y, z) \
        (CUDA_UNSTR_NEIGHBOR_PTR_AT(grid_info, CUDA_UNSTR_INDEX(grid_info, coords), x, y, z))

/** Gives the index of the neighbor. */
#define CUDA_UNSTR_NEIGHBOR_AT(grid_info, index, x, y, z) \
        ((int)(grid_info.data[CUDA_UNSTR_NEIGHBOR_PTR_AT(grid_info, index, x, y, z)]))
#define CUDA_UNSTR_NEIGHBOR(grid_info, coords, x, y, z) \
        ((int)(grid_info.data[CUDA_UNSTR_NEIGHBOR_PTR(grid_info, coords, x, y, z)]))

template<typename value_t>
CudaGridInfo<value_t> CudaUnstructuredGrid3D<value_t>::get_gridinfo() {
    coord3 dimensions = this->dimensions;
    return { .data = this->data,
                .dimensions = dimensions,
                .strides = coord3(5, 
                                  dimensions.x*5,
                                  dimensions.y*dimensions.x*5) };
}

template<typename value_t>
CudaUnstructuredGrid3D<value_t> *CudaUnstructuredGrid3D<value_t>::create_regular(coord3 dimensions) {
    CudaUnstructuredGrid3D<value_t> *grid = new CudaUnstructuredGrid3D<value_t>(dimensions);
    grid->add_regular_neighbors();
    return grid;
}

#endif