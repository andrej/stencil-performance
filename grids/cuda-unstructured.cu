#ifndef CUDA_UNSTRUCTURED_GRID_H
#define CUDA_UNSTRUCTURED_GRID_H
#include "unstructured.cu"

/** Cuda version of the unstructured grid
 *
 * Lays out the grid in Cuda unified memory and provides compiler macros for
 * indexing/setting/getting values in the grid.
 */
 template<typename value_t, typename neigh_ptr_t = int>
 class CudaUnstructuredGrid3D : 
 virtual public UnstructuredGrid3D<value_t, neigh_ptr_t>,
 virtual public CudaBaseGrid<value_t, coord3>
 {

    protected:
    using UnstructuredGrid3D<value_t, neigh_ptr_t>::UnstructuredGrid3D;

    public:    
    static CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *create(coord3 dimensions, coord3 halo=coord3(0, 0, 0), int neighbor_store_depth=1, neigh_ptr_t *neighborships=NULL, bool use_prototypes = false);
    static CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *create_regular(coord3 dimensions, coord3 halo=coord3(0, 0, 0), layout_t layout=rowmajor, int neighbor_store_depth=1, unsigned char z_curve_width=DEFAULT_Z_CURVE_WIDTH, bool use_prototypes=false);
    static CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *clone(const CudaUnstructuredGrid3D<value_t, neigh_ptr_t> &other);

};


template<typename value_t, typename neigh_ptr_t>
CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *CudaUnstructuredGrid3D<value_t, neigh_ptr_t>::create(coord3 dimensions, coord3 halo, int neighbor_store_depth, neigh_ptr_t *neighborships, bool use_prototypes) {
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *obj = new CudaUnstructuredGrid3D<value_t, neigh_ptr_t>(dimensions, halo, neighbor_store_depth, neighborships, use_prototypes);
    obj->init();
    return obj;
}

template<typename value_t, typename neigh_ptr_t>
CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *CudaUnstructuredGrid3D<value_t, neigh_ptr_t>::create_regular(coord3 dimensions, coord3 halo, layout_t layout, int neighbor_store_depth, unsigned char z_curve_width, bool use_prototypes) {
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *obj = new CudaUnstructuredGrid3D<value_t, neigh_ptr_t>(dimensions, halo, neighbor_store_depth, NULL, use_prototypes);
    obj->init();
    obj->add_regular_nodes(layout, z_curve_width);
    obj->add_regular_neighbors();
    return obj;
}

template<typename value_t, typename neigh_ptr_t>
CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *CudaUnstructuredGrid3D<value_t, neigh_ptr_t>::clone(const CudaUnstructuredGrid3D<value_t, neigh_ptr_t> &other) {
    //CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *obj = new CudaUnstructuredGrid3D<value_t, neigh_ptr_t>(other);
    CudaUnstructuredGrid3D<value_t, neigh_ptr_t> *obj = new CudaUnstructuredGrid3D<value_t, neigh_ptr_t>(other.dimensions, other.halo, other.neighbor_store_depth, other.neighborships, other.use_prototypes);
    obj->init();
    obj->indices = other.indices;
    obj->coordinates = other.coordinates;
    obj->prototypes = other.prototypes;
    return obj;
}

#endif