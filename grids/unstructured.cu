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
 * (0, -1, 0) and (0, 1, 0) offsets. In the Z axis, neighbors are as in the
 * regular grid, i.e. their value is at +/- xsize*ysize.
 *
 * Memory layout: The values are stored in the first part of the memory block
 * pointed to by data just as in the regular grid, accessible by their
 * (x, y, z)-coordinates. 
 *
 * The neighborship relations are stored in the second part of the memory block.
 * Neighborship relations are stored for one 2D level. First we list all left
 * neighbors, then all top neighbors, then all right neighbors thenn all bottom
 * neighbors.
 *
 * Alternatively, a different memory location can be given for the neighborships
 * in the constructor. In that case, no memory is allocated for a neighborship
 * table. This can be useful for multiple grids sharing identical neighborships.
 */
template<typename value_t>
class UnstructuredGrid3D : 
virtual public Coord3BaseGrid<value_t> {

    public: 
    UnstructuredGrid3D();
    void init();
    UnstructuredGrid3D(coord3 dimensions);
    UnstructuredGrid3D(coord3 dimensions, int *neighbor_data);
    int index(coord3 coord);
    size_t num_neighbors(coord3 coord);
    int neighbor(coord3 coord, coord3 offset);
    int neighbor_of_index(int index, coord3 offset);
    
    /** == data; pointer to the values. Use this instead of accessing data
     * directly so we can be flexible if we want to put neighborship relations
     * or values first in our data-block. */
    value_t *values;

    /** Pointer to the neighbor memory block, where pointers to values of 
     * neighbors are stored. */
    int *neighbor_data;

    /** Static function that calculates how much memory required. */
    static size_t space_req(coord3 dimensions, bool use_external_neighbor_data=false);

    /** Set B as the neighbor of A at offset (seen from A) offs. This also
     * adds A as a neighbor to B (neighborships are symmetric), but at the
     * inverted offset. */
    void add_neighbor(coord3 A, coord3 B, coord3 offset);

    /** Delete neighbor B from A and vice versa. */
    void del_neighbor(coord3 A, coord3 offset);

    /** Returns the index into neighbor_data where the index of the 
     * requested neighbor is defined.
     *
     * Let x be the return value of this function, then neighbor_data[x]
     * contains the index in data of the value of the requested neighbor at
     * offset, i.e. data[neighbor_data[neighbor_pointer(...)]] gives value,
     * neighbor_data[neighbor_pointer(...)] gives index and
     * neighbor_pointer(...) only gives index into *neighbor_pointer* array. */
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
Grid<value_t, coord3>(dimensions, 
                      UnstructuredGrid3D<value_t>::space_req(dimensions)) {
    this->neighbor_data = NULL;
    this->init();

}

template<typename value_t>
UnstructuredGrid3D<value_t>::UnstructuredGrid3D(coord3 dimensions, int *neighbor_data) :
Grid<value_t, coord3>(dimensions,
                      UnstructuredGrid3D<value_t>::space_req(dimensions, true)),
neighbor_data(neighbor_data) {
    this->init();
}

/* Initialization. */
template<typename value_t>
void UnstructuredGrid3D<value_t>::init(){
    this->Grid<value_t, coord3>::init(); // this allocates this->data
    this->values = this->data; // values are in the first part of our data block
    coord3 dimensions = this->dimensions;
    if(this->neighbor_data == NULL) {
        int value_size = dimensions.x * dimensions.y * dimensions.z;
        this->neighbor_data = (int*) (this->data + value_size);
        int *end = (int *) ((char *)this->data + this->size );
        for(int *ptr = this->neighbor_data; ptr < end; ptr++) {
            *ptr = -1; // initialize to -1 to siginfy: "no neighbor set"
        }
    }
}

/* Space Req. */
template<typename value_t>
size_t UnstructuredGrid3D<value_t>::space_req(coord3 dimensions, bool use_external_neighbor_data) {
    return (  sizeof(value_t) * dimensions.x*dimensions.y*dimensions.z /* for the values */
            + (use_external_neighbor_data ? 0 : sizeof(int) * 4 * dimensions.x*dimensions.y) ); /* for the ptrs */
}

/* Index. */
#define GRID_UNSTR_INDEX(y_stride, z_stride, x, y, z) \
        (   (x) \
          + (y) * y_stride \
          + (z) * z_stride )
template<typename value_t>
int UnstructuredGrid3D<value_t>::index(coord3 coord) {
    coord3 M = this->dimensions;
    return (int) GRID_UNSTR_INDEX(M.x, M.x*M.y,
                                  coord.x, coord.y, coord.z);
}

/* Num Neighbors. */
template<typename value_t>
size_t UnstructuredGrid3D<value_t>::num_neighbors(coord3 coord) {
    int ret = 4;
    int start=this->neighbor_pointer(coord, coord3(-1, 0, 0));
    for(int idx=start; idx < start+4; idx++) {
        if(this->neighbor_data[idx] == -1) {
            // -1 signifies no neighbor set
            ret--;
        }
    }
    return ret;
}

/** Gives a pointer into the neighborship_data array. */
// 2D cases for Z=0
#define GRID_UNSTR_2D_NEIGHBOR_PTR_OF_INDEX(z_stride, index, x, y) \
        ( (index) + (z_stride) * 1 * ((y) == -1) \
                  + (z_stride) * 2 * ((x) == +1) \
                  + (z_stride) * 3 * ((y) == +1) )
#define GRID_UNSTR_2D_NEIGHBOR_PTR(y_stride, z_stride, x, y, neigh_x, neigh_y) \
        GRID_UNSTR_2D_NEIGHBOR_PTR_OF_INDEX(z_stride, \
                                            GRID_UNSTR_INDEX(y_stride, z_stride, x, y, 0), \
                                            neigh_x, neigh_y)
// All cases Z>=0
#define GRID_UNSTR_NEIGHBOR_PTR_OF_INDEX(z_stride, index, x, y) \
        GRID_UNSTR_2D_NEIGHBOR_PTR_OF_INDEX(z_stride, (index) % (z_stride), x, y)
#define GRID_UNSTR_NEIGHBOR_PTR(y_stride, z_stride, x, y, neigh_x, neigh_y) \
        GRID_UNSTR_2D_NEIGHBOR_PTR_OF_INDEX(z_stride, \
                                            GRID_UNSTR_INDEX(y_stride, z_stride, x, y, 0), \
                                            neigh_x, neigh_y)
template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor_pointer(coord3 coord, coord3 offs) {
    assert(offs.x == 0 || offs.y == 0); /* no diagonal neighbors */
    assert(offs.x >= -1 && offs.x <= 1 && offs.y >= -1 && offs.y <= 1);
    assert(offs.z == 0);
    coord3 M = this->dimensions;
    return GRID_UNSTR_NEIGHBOR_PTR(M.x, M.x*M.y,
                                   coord.x, coord.y,
                                   offs.x, offs.y);
}

/** Gives the index of the neighbor. */
#define GRID_UNSTR_NEIGHBOR(neighbor_data, y_stride, z_stride, x, y, z, neigh_x, neigh_y, neigh_z) \
    ( (neigh_x != 0 || neigh_y != 0 ? neighbor_data[GRID_UNSTR_NEIGHBOR_PTR(y_stride, z_stride, x, y, neigh_x, neigh_y)] \
                        : GRID_UNSTR_INDEX(y_stride, z_stride, x, y, 0))\
        + (z + neigh_z) * z_stride)
template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor(coord3 coord, coord3 offs) {
    assert(offs.z == 0 || (offs.x == 0 && offs.y == 0)); // only one neighbor at a time, no diagonals
    coord3 M = this->dimensions;
    return GRID_UNSTR_NEIGHBOR(this->neighbor_data,
                               M.x, M.x*M.y,
                               coord.x, coord.y, coord.z,
                               offs.x, offs.y, offs.z);
}

/** Gives the index of the neighbor given an index (not coordinate). */
#define GRID_UNSTR_2D_NEIGHBOR_OF_INDEX(neighbor_data, z_stride, index, x, y) \
     ( neighbor_data[GRID_UNSTR_2D_NEIGHBOR_PTR_OF_INDEX(z_stride, index, x, y)] )
#define GRID_UNSTR_NEIGHBOR_OF_INDEX(neighbor_data, z_stride, index, x, y, z) \
     ( neighbor_data[GRID_UNSTR_NEIGHBOR_PTR_OF_INDEX(z_stride, index, x, y)]\
            + ((index) / (z_stride)) * (z_stride) /* round off non-Z component of index */ \
            + (z) * (z_stride) )
template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor_of_index(int index, coord3 offs) {
    coord3 M = this->dimensions;
    return GRID_UNSTR_NEIGHBOR_OF_INDEX(this->neighbor_data,
                                        M.x*M.y,
                                        index,
                                        offs.x, offs.y, offs.z);
}

/* Add neighbor. */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(coord3 A, coord3 B, coord3 offs) {
    assert(offs.z == 0); // Regular in Z-axis; this neighborship cannot be changed
    int A_idx = this->index(coord3(A.x, A.y, 0));
    int A_neigh_idx = this->neighbor_pointer(A, offs);
    int B_idx = this->index(coord3(B.x, B.y, 0));
    int B_neigh_idx = this->neighbor_pointer(B, -offs);
    this->neighbor_data[A_neigh_idx] = B_idx;
    this->neighbor_data[B_neigh_idx] = A_idx;
}

/* Delete neighbor. */
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
    // value -1 siginifies no neighbor set
    this->neighbor_data[A_neigh_idx] = -1;
    this->neighbor_data[B_neigh_idx] = -1;
}

/* Add same neighbors as in a regular grid. */
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

/* Simulate regular grid with neighbor lookup overhead. */
template<typename value_t>
UnstructuredGrid3D<value_t> *UnstructuredGrid3D<value_t>::create_regular(coord3 dims) {
    UnstructuredGrid3D<value_t> *grid = new UnstructuredGrid3D<value_t>(dims);
    grid->add_regular_neighbors();
    return grid;
}

#endif