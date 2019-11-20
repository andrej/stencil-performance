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
     value_t *data; /**< Pointer to the allocated CUDA memory, i.e. this->data */
     coord3 dimensions;
     coord3 strides;
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

/** Get value at given data index (not coord.) */
#define CUDA_REGULAR_AT(grid_info, index) \
    (grid_info.data[index])

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
          
/** Use this macro instead of the CudaRegularGrid3D::neighbor() function for
 * getting the index of some coordinate offset from within the kernel. */
#define CUDA_REGULAR_NEIGHBOR(grid_info, coords, _x, _y, _z) \
        ((int)   ((coords.x + (_x)) + \
                  (coords.y + (_y)) * grid_info.strides.y + \
                  (coords.z + (_z)) * grid_info.strides.z))

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
    return CUDA_REGULAR_INDEX(gridinfo, coord);
}

template<typename value_t>
int CudaRegularGrid3D<value_t>::neighbor(coord3 coord, coord3 offs) {
    CudaRegularGrid3DInfo<value_t> gridinfo = this->get_gridinfo();
    return CUDA_REGULAR_NEIGHBOR(gridinfo, coord, offs.x, offs.y, offs.z);
}

template<typename value_t>
CudaRegularGrid3DInfo<value_t> CudaRegularGrid3D<value_t>::get_gridinfo() {
    coord3 dimensions = this->dimensions;
    return { .data = this->data,
             .dimensions = dimensions,
             .strides = coord3(1, dimensions.x, dimensions.x*dimensions.y) };
}

#endif