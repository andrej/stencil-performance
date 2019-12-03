#ifndef CUDA_REGULAR_GRID_H
#define CUDA_REGULAR_GRID_H
#include "grid.cu"
#include "cuda-base.cu"
#include "coord3-base.cu"
#include "regular.cu"

/** This struct is used to pass grid information to the kernel (because we
 * cannot use the C++ classes inside the kernel). Pass this struct
 * to the appropriate kernel macros.
 */
 template<typename value_t>
 struct CudaRegularGrid3DInfo {
    struct {
        // stride x is always 1 for regular grids, don't waste register storing that information
        const int y;
        const int z;
    } strides;
 };

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
    
    CudaRegularGrid3D(coord3 dimensions);
    
    int index(coord3 coord); /**< Reimplemented to use the kernel macro. */
    
    using RegularGrid3D<value_t>::neighbor;
    
    int neighbor(coord3 coord, coord3 offset); /**< Reimplemented to use the kernel macro. */
    
    virtual CudaRegularGrid3DInfo<value_t> get_gridinfo(); /**< Return grid info struct required by kernel macros. */
    
};

/** Use this macro instead of the CudaRegularGrid3D::index() function for
 * getting the index of some coordinates inside the kernel. */
#define CUDA_REGULAR_INDEX(grid_info, _x, _y, _z) \
        CUDA_REGULAR_INDEX_(grid_info.strides.y, grid_info.strides.z, _x, _y, _z)

#define CUDA_REGULAR_INDEX_(stride_y, stride_z, _x, _y, _z) \
        ((int)   ((int)(_x) + \
                  (int)(_y) * stride_y + \
                  (int)(_z) * stride_z))
          
/** Use this macro instead of the CudaRegularGrid3D::neighbor() function for
 * getting the index of some coordinate offset from within the kernel. */
#define CUDA_REGULAR_NEIGHBOR(grid_info, _x, _y, _z, neigh_x, neigh_y, neigh_z) \
        CUDA_REGULAR_NEIGHBOR_(grid_info.strides.y, grid_info.strides.z, _x, _y, _z, neigh_x, neigh_y, neigh_z)

#define CUDA_REGULAR_NEIGHBOR_(stride_y, stride_z, _x, _y, _z, _neigh_x, _neigh_y, _neigh_z) \
        ((int)   (((int)(_x) + (_neigh_x)) + \
                  ((int)(_y) + (_neigh_y)) * stride_y + \
                  ((int)(_z) + (_neigh_z)) * stride_z))

// IMPLEMENTATIONS

template<typename value_t>
CudaRegularGrid3D<value_t>::CudaRegularGrid3D(coord3 dimensions) :
Grid<value_t, coord3>(dimensions,
                       sizeof(value_t)*dimensions.x*dimensions.y*dimensions.z) {
    this->init();
}

template<typename value_t>
int CudaRegularGrid3D<value_t>::index(coord3 coord) {
    CudaRegularGrid3DInfo<value_t> gridinfo = this->get_gridinfo();
    return CUDA_REGULAR_INDEX(gridinfo, coord.x, coord.y, coord.z);
}

template<typename value_t>
int CudaRegularGrid3D<value_t>::neighbor(coord3 coord, coord3 offs) {
    CudaRegularGrid3DInfo<value_t> gridinfo = this->get_gridinfo();
    return CUDA_REGULAR_NEIGHBOR(gridinfo, coord.x, coord.y, coord.z, offs.x, offs.y, offs.z);
}

template<typename value_t>
CudaRegularGrid3DInfo<value_t> CudaRegularGrid3D<value_t>::get_gridinfo() {
    coord3 dimensions = this->dimensions;
    return { .strides = {.y = (int)dimensions.x, .z = (int)dimensions.x*(int)dimensions.y } };
}

#endif