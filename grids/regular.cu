#ifndef REGULAR_GRID_H
#define REGULAR_GRID_H
#include <assert.h>
#include <memory>
#include <algorithm>
#include "grid.cu"
#include "coord3-base.cu"

/** A regular cartesian grid of squares.
 * 
 * Memory layout: first x, then y, then z, so when we go through the memory
 * one-by-one, x varies fastest, y second and z slowest. An allocator can be
 * provided to use instead of the standard malloc.
 * 
 *    /---/---/---/---/|
 *   / 16/ 17/ 18/ 19/ |
 *  / 8 / 9 / 10/ 11/| |
 * |---|---|---|---| | |
 * | 0 | 1 | 2 | 3 | | / 
 * |---|---|---|---| |/
 * | 4 | 5 | 6 | 7 | /
 * |---|---|---|---|/
 * 
 * Neighbors: Each inner cell has six neighbors that touch faces. Edge cells
 * have four neighbors. Corner cells have three neighbors.
 */
template<typename value_t>
class RegularGrid3D : 
virtual public Coord3BaseGrid<value_t> {

    public:

    RegularGrid3D();
    RegularGrid3D(coord3 dimensions, size_t padding=0);

    int index(coord3 coords);

    /** In a regular grid, each cell has between zero (for one-cell grid) and
     * six neighbors. */
    size_t num_neighbors(coord3 coords);

    int neighbor(coord3 coords, coord3 offset);
    int neighbor_of_index(int index, coord3 offset);

    /** Returns the X-, Y- and Z- strides. */
    coord3 get_strides();

    /** Padding: Each coordinate is padded by this many *additional* bytes.
     * Might help with cache locality. Default 0. */
    size_t padding;

};

// IMPLEMENTATIONS
template<typename value_t>
RegularGrid3D<value_t>::RegularGrid3D() {}

template<typename value_t>
RegularGrid3D<value_t>::RegularGrid3D(coord3 dimensions, size_t padding) : 
Grid<value_t, coord3>(dimensions,
                      sizeof(value_t)*dimensions.x*dimensions.y*dimensions.z*(1+padding)),
padding(padding) {
    this->init();
}

#define GRID_REGULAR_INDEX(stride_y, stride_z, x, y, z) \
        (  (x) \
         + (y) * (stride_y) \
         + (z) * (stride_z) )
template<typename value_t>
int RegularGrid3D<value_t>::index(coord3 coords) {
    coord3 M = this->dimensions;
    return (int) GRID_REGULAR_INDEX(M.x, M.x*M.y,
                                    coords.x, coords.y, coords.z);
}

#define GRID_REGULAR_NEIGHBOR(stride_y, stride_z, x, y, z, neigh_x, neigh_y, neigh_z) \
        ( (x) + (neigh_x) + \
         ((y) + (neigh_y)) * (stride_y) + \
         ((z) + (neigh_z)) * (stride_z) )
template<typename value_t>
int RegularGrid3D<value_t>::neighbor(coord3 coords, coord3 offs) {
    coord3 M = this->dimensions;
    return (int) GRID_REGULAR_NEIGHBOR(M.x, M.x*M.y,
                                       coords.x, coords.y, coords.z,
                                       offs.x, offs.y, offs.z);
}

#define GRID_REGULAR_NEIGHBOR_OF_INDEX(stride_y, stride_z, index, neigh_x, neigh_y, neigh_z) \
        ( (index) + GRID_REGULAR_INDEX(stride_y, stride_z, neigh_x, neigh_y, neigh_z) )
template<typename value_t>
int RegularGrid3D<value_t>::neighbor_of_index(int index, coord3 offs) {
    coord3 M = this->dimensions;
    return (int) GRID_REGULAR_NEIGHBOR_OF_INDEX(M.x, M.x*M.y,
                                                index,
                                                offs.x, offs.y, offs.z);
}

template<typename value_t>
size_t RegularGrid3D<value_t>::num_neighbors(coord3 coords) {
    coord3 dims = this->dimensions;
    return   (coords.x > 0) // left neighbor
           + (coords.x < dims.x-1) // right neighbor
           + (coords.y > 0) // top neighbor
           + (coords.y < dims.y-1) // right neighbor
           + (coords.z > 0) // front neighbor
           + (coords.z < dims.z-1); // back neighbor
}

template<typename value_t>
coord3 RegularGrid3D<value_t>::get_strides() {
    return coord3(1,
                  this->dimensions.x,
                  this->dimensions.x*this->dimensions.y);
}

#endif