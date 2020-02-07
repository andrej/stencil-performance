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

    protected:
    using UnstructuredGrid3D<value_t>::UnstructuredGrid3D;

    public:    
    static CudaUnstructuredGrid3D<value_t> *create(coord3 dimensions, coord3 halo=coord3(0, 0, 0), int neighbor_store_depth=1, int *neighborships=coord3(0, 0, 0));
    static CudaUnstructuredGrid3D<value_t> *create_regular(coord3 dimensions, coord3 halo=coord3(0, 0, 0), typename UnstructuredGrid3D<value_t>::layout_t layout=UnstructuredGrid3D<value_t>::rowmajor, int neighbor_store_depth=1, unsigned char z_curve_width=DEFAULT_Z_CURVE_WIDTH);
    static CudaUnstructuredGrid3D<value_t> *clone(const CudaUnstructuredGrid3D<value_t> &other);

};


template<typename value_t>
CudaUnstructuredGrid3D<value_t> *CudaUnstructuredGrid3D<value_t>::create(coord3 dimensions, coord3 halo, int neighbor_store_depth, int *neighborships) {
    CudaUnstructuredGrid3D<value_t> *obj = new CudaUnstructuredGrid3D<value_t>(dimensions, halo, neighbor_store_depth, neighborships);
    obj->init();
    return obj;
}

template<typename value_t>
CudaUnstructuredGrid3D<value_t> *CudaUnstructuredGrid3D<value_t>::create_regular(coord3 dimensions, coord3 halo, typename UnstructuredGrid3D<value_t>::layout_t layout, int neighbor_store_depth, unsigned char z_curve_width) {
    CudaUnstructuredGrid3D<value_t> *obj = new CudaUnstructuredGrid3D<value_t>(dimensions, halo, neighbor_store_depth);
    obj->init();
    obj->add_regular_nodes(layout, z_curve_width);
    obj->add_regular_neighbors();
    return obj;
}

template<typename value_t>
CudaUnstructuredGrid3D<value_t> *CudaUnstructuredGrid3D<value_t>::clone(const CudaUnstructuredGrid3D<value_t> &other) {
    //CudaUnstructuredGrid3D<value_t> *obj = new CudaUnstructuredGrid3D<value_t>(other);
    CudaUnstructuredGrid3D<value_t> *obj = new CudaUnstructuredGrid3D<value_t>(other.dimensions, other.halo, other.neighbor_store_depth, other.neighborships);
    obj->init();
    obj->indices = other.indices;
    obj->coordinates = other.coordinates;
    return obj;
}

#endif