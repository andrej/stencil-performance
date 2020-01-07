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
    CudaUnstructuredGrid3D(coord3 dimensions, int *neighbor_data);

    static CudaUnstructuredGrid3D<value_t> *create_regular(coord3 dimensions);

};

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
CudaUnstructuredGrid3D<value_t> *CudaUnstructuredGrid3D<value_t>::create_regular(coord3 dimensions) {
    CudaUnstructuredGrid3D<value_t> *grid = new CudaUnstructuredGrid3D<value_t>(dimensions);
    grid->add_regular_neighbors();
    return grid;
}

#endif