#ifndef UNSTRUCTURED_GRID_H
#define UNSTRUCTURED_GRID_H
#include <assert.h>
#include <memory>
#include "grids/grid.cu"
#include "grids/coord3-base.cu"

/** Unstructured Grid in 2 coords, structured in one coord
 *
 * In the two dimensions X and Y, the grid can have arbitrary neighborship
 * relations, but each cell is limited to at most four neighbors in these
 * four dimensions. In the Z axis, the grid is regular.
 *
 * Neighbors in the X and Y plane are identified by (-1, 0, 0), (1, 0, 0),
 * (0, -1, 0) and (0, 1, 0) offsets.
 *
 * Memory layout: (where {x} is the index of x and [x] is the value of x)
 * [Cell X, Y, Z] {X-1 neighbor} {Y-1 neighbor} {X+1 neighbor} {Y+1 neighbor}
 *
 * Cell values for X, Y, Z are laid out as in the regular grid, just with
 * additional space around it for the neighborship relations.
 */
template<typename value_t>
class UnstructuredGrid3D : 
virtual public Coord3BaseGrid<value_t> {

    public: 
    UnstructuredGrid3D();
    UnstructuredGrid3D(coord3 dimensions);
    int index(coord3 coord);
    size_t num_neighbors(coord3 coord);
    int neighbor(coord3 coord, coord3 offset);

    /** Set B as the neighbor of A at offset (seen from A) offs. This also
     * adds A as a neighbor to B (neighborships are symmetric), but at the
     * inverted offset. */
    void add_neighbor(coord3 A, coord3 B, coord3 offset);

    /** Delete neighbor B from A and vice versa. */
    void del_neighbor(coord3 A, coord3 offset);

    /** Gives the index in the data array where the pointer to coord's
     * neighbor is stored. Modify the value at this index to change the 
     * neighbors of coord. */
    int neighbor_pointer(coord3 coord, coord3 offs);

    /** "Fake" a regular grid by adding the neighborship relations that a
     * regular grid would have, i.e. add top, left, right, bottom, front and
     * back neighbors as in a regular grid. */
    void add_regular_neighbors();

    /** Return a new grid with regular neighborship relations. */
    static UnstructuredGrid3D<value_t> *create_regular(coord3 dims);

};

// IMPLEMENTATIONS

template<typename value_t>
UnstructuredGrid3D<value_t>::UnstructuredGrid3D() {}

template<typename value_t>
UnstructuredGrid3D<value_t>::UnstructuredGrid3D(coord3 dimensions) :
Grid<value_t, coord3>(dimensions, dimensions.x*dimensions.y*dimensions.z*5) {
}

template<typename value_t>
int UnstructuredGrid3D<value_t>::index(coord3 coord) {
    int N = 4;
    coord3 M = this->dimensions;
    return (int)(  coord.x*(N+1)
                    + coord.y*(M.x*(N+1))
                    + coord.z*M.y*(M.x*(N+1)) );
}

template<typename value_t>
size_t UnstructuredGrid3D<value_t>::num_neighbors(coord3 coord) {
    int ret = 4;
    value_t *cell=this->data + this->index(coord);
    for(value_t *ptr = cell+1; ptr < cell+5; ptr++) {
        if(ptr == NULL) {
            ret--;
        }
    }
    return ret;
}

template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor(coord3 coord, coord3 offs) {
    assert(offs.z == 0 || (offs.x == 0 & offs.y == 0)); // only one neighbor at a time, no diagonals
    if(offs.z == 0) {
        return this->data[this->neighbor_pointer(coord, offs)];
    }
    return this->index(coord3(coord.x, coord.y, coord.z + offs.z));
}

template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor_pointer(coord3 coord, coord3 offs) {
    int cell = this->index(coord3(coord.x, coord.y, coord.z));
    assert(offs.x == 0 || offs.y == 0);
    return cell +     (offs.x == -1)
                + 2 * (offs.y == -1)
                + 3 * (offs.x == +1)
                + 4 * (offs.y == +1);
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(coord3 A, coord3 B, coord3 offs) {
    assert(offs.z == 0); // Regular in Z-axis; this neighborship cannot be changed
    int A_idx = this->index(A);
    int A_neigh_idx = this->neighbor_pointer(A, offs);
    int B_idx = this->index(B);
    int B_neigh_idx = this->neighbor_pointer(B, -offs);
    this->data[A_neigh_idx] = B_idx;
    this->data[B_neigh_idx] = A_idx;
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::del_neighbor(coord3 A, coord3 offs) {
    assert(offs.z == 0); // Regular in Z-axis; this neighborship cannot be changed
    int A_idx = this->index(A);
    int A_neigh_idx = this->neighbor_pointer(A, offs);
    int B_idx = this->neighbor(A, offs);
    coord3 B = coord3(B_idx % this->dimensions.x,
                      B_idx / this->dimensions.x,
                      B_idx / (this->dimensions.x*this->dimensions.y));
    int B_neigh_idx = this->neighbor_pointer(B, -offs);
    this->data[A_neigh_idx] = NULL;
    this->data[B_neigh_idx] = NULL;
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::add_regular_neighbors() {
    coord3 dims = this->dimensions;
    for(int z = 0; z<dims.z; z++) {
        for(int y = 0; y<dims.y; y++) {
            for(int x = 1; x<dims.x; x++) {
                this->add_neighbor(coord3(x, y, z), coord3(x-1, y, z), coord3(-1, 0, 0));
            }
            for(int x = 0; x<dims.x-1; x++) {
                this->add_neighbor(coord3(x, y, z), coord3(x+1, y, z), coord3(0, 1, 0));
            }
        }
        for(int x = 0; x<dims.x; x++) {
            for(int y = 1; y<dims.y; y++) {
                this->add_neighbor(coord3(x, y, z), coord3(x, y-1, z), coord3(0, -1, 0));
            }
            for(int y = 0; y<dims.y-1; y++) {
                this->add_neighbor(coord3(x, y, z), coord3(x, y+1, z), coord3(0, 1, 0));
            }
        }
    }
}

template<typename value_t>
UnstructuredGrid3D<value_t> *UnstructuredGrid3D<value_t>::create_regular(coord3 dims) {
    UnstructuredGrid3D<value_t> *grid = new UnstructuredGrid3D<value_t>(dims);
    grid->add_regular_neighbors();
    return grid;
}

#endif