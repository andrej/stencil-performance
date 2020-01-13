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

    RegularGrid3D(coord3 dimensions, coord3 halo);

    int index(coord3 coords);
    coord3 coordinate(int index);
    int zero_offset; // position in data of (0, 0, 0)
    
    using Grid<value_t, coord3>::neighbor;
    int neighbor(int index, coord3 offset);

    /** Returns the X-, Y- and Z- strides. */
    coord3 get_strides();

    /** Halo */
    coord3 halo;

};

// IMPLEMENTATIONS

template<typename value_t>
RegularGrid3D<value_t>::RegularGrid3D(coord3 dimensions, coord3 halo) : 
Grid<value_t, coord3>(dimensions,
                      sizeof(value_t)*(dimensions.x+2*halo.x)*(dimensions.y+2*halo.y)*(dimensions.z+2*halo.z),
                      halo) {
    this->init();
    coord3 strides = this->get_strides();
    this->zero_offset = halo.x*strides.x + halo.y*strides.y + halo.z*strides.z;
}

template<typename value_t>
int RegularGrid3D<value_t>::index(coord3 coords) {
    coord3 M = this->get_strides();
    return this->zero_offset + (coords.x + coords.y * M.y + coords.z * M.z);
}

template<typename value_t>
coord3 RegularGrid3D<value_t>::coordinate(int index) {
    coord3 M = this->get_strides();
    index -= this->zero_offset;
    return coord3(index % M.y,
                  (index % (M.z)) / M.y,
                  index / (M.z));
}

#define GRID_REGULAR_NEIGHBOR(stride_y, stride_z, index, neigh_x, neigh_y, neigh_z) \
        ( (index) + neigh_x + stride_y * neigh_y + stride_z * neigh_z )
template<typename value_t>
int RegularGrid3D<value_t>::neighbor(int index, coord3 offs) {
    coord3 M = this->get_strides();
    return (int) GRID_REGULAR_NEIGHBOR(M.y, M.z,
                                       index,
                                       offs.x, offs.y, offs.z);
}

template<typename value_t>
coord3 RegularGrid3D<value_t>::get_strides() {
    return coord3(1,
                  this->dimensions.x + 2 * this->halo.x,
                  (this->dimensions.x + 2 * this->halo.x) * (this->dimensions.y + 2 * this->halo.y));
}

#endif