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

    protected:
    RegularGrid3D() {};
    RegularGrid3D(coord3 dimensions, coord3 halo=coord3(0, 0, 0));

    public:
    static RegularGrid3D *create(coord3 dimensions, coord3 halo=coord3(0, 0, 0));

    int index(coord3 coords);
    coord3 coordinate(int index);
    int padding; // pad data so that (0, 0, 0) is at alignment boundary (pad the first halo)
    int alignment = 32; // strides are mutliple of this; together with padding this ensures that first cell in a row is alwas at multiple of alignment -> coalescing memory accesses
    int zero_offset; // position in data of (0, 0, 0)
    
    using Grid<value_t, coord3>::neighbor;
    int neighbor(int index, coord3 offset);

    /** Returns the X-, Y- and Z- strides. */
    coord3 get_strides();

};

// IMPLEMENTATIONS

template<typename value_t>
RegularGrid3D<value_t>::RegularGrid3D(coord3 dimensions, coord3 halo) {
    this->dimensions = dimensions;
    this->halo = halo;
    coord3 strides = this->get_strides();
    int first = this->halo.x*strides.x + this->halo.y*strides.y + this->halo.z*strides.z;
    int align = this->alignment / sizeof(value_t);
    if(first % align != 0) {
        this->padding = align - first % align;
    }
    this->zero_offset = this->padding + first;
    this->size = sizeof(value_t) * (this->padding + (dimensions.z+2*halo.z) * strides.z);
}

template<typename value_t>
RegularGrid3D<value_t> *RegularGrid3D<value_t>::create(coord3 dimensions, coord3 halo) {
    RegularGrid3D<value_t> *obj = new RegularGrid3D<value_t>(dimensions, halo);
    obj->init(); // Calls default Grid initializer and allocates space
    return obj;
}

#define GRID_REGULAR_INDEX(stride_y, stride_z, x, y, z) \
        ( (x) + (y) * stride_y + (z) * stride_z )
template<typename value_t>
int RegularGrid3D<value_t>::index(coord3 coords) {
    assert(-this->halo.x <= coords.x && coords.x < this->dimensions.x+this->halo.x &&
           -this->halo.y <= coords.y && coords.y < this->dimensions.y+this->halo.y &&
           -this->halo.z <= coords.z && coords.z < this->dimensions.z+this->halo.z);
    coord3 M = this->get_strides();
    return this->zero_offset + GRID_REGULAR_INDEX(M.y, M.z, coords.x, coords.y, coords.z);
}

template<typename value_t>
coord3 RegularGrid3D<value_t>::coordinate(int index) {
    assert(0 <= index < this->size/sizeof(value_t));
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
    int sx = 1;
    int sy = roundup(this->dimensions.x + 2 * this->halo.x, this->alignment / sizeof(value_t));
    int sz = roundup(sy * (this->dimensions.y + 2 * this->halo.y), this->alignment / sizeof(value_t));
    return coord3(sx, sy, sz);
}

#endif