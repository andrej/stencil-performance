#ifndef CUDA_REGULAR_GRID_H
#define CUDA_REGULAR_GRID_H
#include "grid.cu"
#include "coord3.cu"
#include "cuda-base.cu"
#include "coord3-base.cu"
#include "regular.cu"

/** Regular grid which stores its data in Cuda unified memory and provides a
 * compiler macro for indexing so that indexing within a GPU kernel does not 
 * require a function call. 
 *
 * This is really just a Regular3D grid that uses the Cuda allocator.*/
template<typename value_t>
class CudaRegularGrid3D : 
virtual public RegularGrid3D<value_t>,
virtual public CudaBaseGrid<value_t, coord3>
{
    public:
    using RegularGrid3D<value_t>::RegularGrid3D;
    static CudaRegularGrid3D<value_t> *create(coord3 dimensions, coord3 halo=coord3(0, 0, 0));

};

template<typename value_t>
CudaRegularGrid3D<value_t> *CudaRegularGrid3D<value_t>::create(coord3 dimensions, coord3 halo) {
    CudaRegularGrid3D<value_t> *obj = new CudaRegularGrid3D<value_t>(dimensions, halo);
    obj->init();
    return obj;
}

#endif