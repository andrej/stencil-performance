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
 * 
 * Author: André Rösti
 * Date: 2019-10-27
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

    /** Only the direct face touching cells are considered neighbors. Diagonal
     * "neighbors" such as (-1, -1, 0) can however be accessed as well through
     * this method, but please note that this is not supported in other Grids.
     */
    int neighbor(coord3 coords, coord3 offset);

    /** This overloaded function returns the neighbor for a given relation,
     * like left, right, top, bottom, front and back. It is more convenient
     * to use if a specific neighbor needs to be accessed because the index of
     * neighbor is different depending on whether the cell at coords is a inner
     * cell, an edge cell or a corner cell.
     * 
     * Multiple relations can also be OR-ed together, making it possible to
     * access diagonal neighbors, e.g. with rel = top | left. But beware of
     * nonsensical combinations like left | right (which will actually just
     * give you coords itself, the "middle"). */
    //virtual int neighbor(coord3 coords, coord3_rel rel);

    /** Padding: Each coordinate is padded by this many *additional* bytes.
     * Might help with cache locality. Default 0. */
    size_t padding;

};

// IMPLEMENTATIONS
template<typename value_t>
RegularGrid3D<value_t>::RegularGrid3D() {
    
}

template<typename value_t>
RegularGrid3D<value_t>::RegularGrid3D(coord3 dimensions, size_t padding) : 
Grid<value_t, coord3>(dimensions,
                      sizeof(value_t)*dimensions.x*dimensions.y*dimensions.z*(1+padding)),
padding(padding) {
    this->init();
}

template<typename value_t>
int RegularGrid3D<value_t>::index(coord3 coords) {
    coord3 M = this->dimensions;
    return (int)((coords.x) +
                 (coords.y) * (M.x) +
                 (coords.z) * (M.y) * (M.x))
                 * (1+this->padding);
}

template<typename value_t>
int RegularGrid3D<value_t>::neighbor(coord3 coords, coord3 offs) {
    return this->index(coord3(coords.x + offs.x,
                              coords.y + offs.y,
                              coords.z + offs.z));
}

/*template<typename value_t>
int RegularGrid3D<value_t>::neighbor(coord3 coords, coord3_rel rel) {
    return this->neighbor(coords, coord3( - (rel&left>0)  + (rel&right>0),
                                          - (rel&top>0)   + (rel&bottom>0),
                                          - (rel&front>0) + (rel&back>0)));
}*/

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

#endif