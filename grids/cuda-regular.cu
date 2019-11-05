#ifndef CUDA_REGULAR_GRID_H
#define CUDA_REGULAR_GRID_H
#include "grid.cu"
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
    
    CudaRegularGrid3D(coord3 dimensions);
    
    int index(coord3 coord); /**< Reimplemented to use the kernel macro. */
    
    using RegularGrid3D<value_t>::neighbor;
    
    int neighbor(coord3 coord, coord3 offset); /**< Reimplemented to use the kernel macro. */
    
    virtual CudaGridInfo<value_t> get_gridinfo(); /**< Return grid info struct required by kernel macros. */

};

/** Use this macro to get data (value) from a grid inside kernels. */
#define CUDA_REGULAR(grid_info, coords) \
    (grid_info.data[CUDA_REGULAR_INDEX(grid_info, coords)])

/** Use this to get the value of a neighbor. */
#define CUDA_REGULAR_NEIGH(grid_info, coords, x, y, z) \
    (grid_info.data[CUDA_REGULAR_NEIGHBOR(grid_info, coords, x, y, z)])

/** Use this macro instead of the CudaRegularGrid3D::index() function for
 * getting the index of some coordinates inside the kernel. */
#define CUDA_REGULAR_INDEX(grid_info, coords) \
        ((int)   (coords.x + \
                  coords.y * grid_info.strides.y + \
                  coords.z * grid_info.strides.z))
/*#define CUDA_REGULAR_INDEX(grid_info, coords) \
((int)   (coords.x + \
        coords.y * grid_info.dimensions.x + \
        coords.z * grid_info.dimensions.y * grid_info.dimensions.x))*/
          

/** Use this macro instead of the CudaRegularGrid3D::neighbor() function for
 * getting the index of some coordinate offset from within the kernel. */
#define CUDA_REGULAR_NEIGHBOR(grid_info, coords, _x, _y, _z) \
        ((int)   ((coords.x + (_x)) + \
                  (coords.y + (_y)) * grid_info.strides.y + \
                  (coords.z + (_z)) * grid_info.strides.z))
/*#define CUDA_REGULAR_NEIGHBOR(grid_info, coords, x, y, z) \
((int)   ((coords.x+x) + \
        (coords.y+y) * grid_info.dimensions.x + \
        (coords.z+z) * grid_info.dimensions.y * grid_info.dimensions.x))*/

// IMPLEMENTATIONS

template<typename value_t>
CudaRegularGrid3D<value_t>::CudaRegularGrid3D(coord3 dimensions) :
Grid<value_t, coord3>(),
RegularGrid3D<value_t>(),
CudaBaseGrid<value_t, coord3>(dimensions, dimensions.x*dimensions.y*dimensions.z) {
    this->allocate();
}

template<typename value_t>
int CudaRegularGrid3D<value_t>::index(coord3 coord) {
    CudaGridInfo<value_t> gridinfo = this->get_gridinfo();
    return CUDA_REGULAR_INDEX(gridinfo, coord);
}

template<typename value_t>
int CudaRegularGrid3D<value_t>::neighbor(coord3 coord, coord3 offs) {
    CudaGridInfo<value_t> gridinfo = this->get_gridinfo();
    return CUDA_REGULAR_NEIGHBOR(gridinfo, coord, offs.x, offs.y, offs.z);
}

template<typename value_t>
CudaGridInfo<value_t> CudaRegularGrid3D<value_t>::get_gridinfo() {
    coord3 dimensions = this->dimensions;
    return { .data = this->data,
             .dimensions = dimensions,
             .strides = coord3(1, dimensions.x, dimensions.x*dimensions.y) };
}

#endif