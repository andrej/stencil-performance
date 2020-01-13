#ifndef UNSTRUCTURED_GRID_H
#define UNSTRUCTURED_GRID_H
#include <assert.h>
#include <stdlib.h>
#include <memory>
#include <map>
#include <stdexcept>
#include "grids/grid.cu"
#include "grids/coord3-base.cu"

/** Unstructured Grid in 2 coords, structured in one Y-coordinate
 *
 * Cell grids can still be accessed by coordinates, but it is up to the user of
 * the grid to determine which coordinates are used and how they are related
 * to one another (neighborhsip relations).
 *
 * The data layout for coordinates in memory is arbitrary. When importing from
 * a regular grid, a row-major data layout can be used but the mapping from
 * coordinates to memory indices is not fixed.
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
 * neighbors, etc. If neighbor_store_depth > 1, neighbors of neighbors etc. are
 * stored as well in the same fashion.
 *
 * Alternatively, a different memory location can be given for the neighborships
 * in the constructor. In that case, no memory is allocated for a neighborship
 * table. This can be useful for multiple grids sharing identical neighborships.
 */
template<typename value_t>
class UnstructuredGrid3D : 
virtual public Coord3BaseGrid<value_t> {

    public: 
    
    /** Neighborship data pointers can be reused from a different grid. In this
     * case, pass in a pointer as neighborships.
     *
     * neighbor_store_depth gives how many direct pointers to neighbors are
     * stored in memory, i.e. for neighbor_store_depth=2, pointers to
     * neighbors and pointers to neighbors of neighbors are stored. */
    UnstructuredGrid3D(coord3 dimensions, coord3 halo, int *neighborships=NULL, int neighbor_store_depth=1);

    void init();
    
    int index(coord3 coord);
    coord3 coordinate(int index);

    using Grid<value_t, coord3>::neighbor;
    int neighbor(int index, coord3 offset);

    /** Pointer to the neighbor memory block, where pointers to values of 
     * neighbors are stored. */
    int *neighborships;
    int neighbor_store_depth=1;

    /** Mapping from X-Y-coordinates to indices. */
    std::map<coord3, int> *indices = NULL;

    /** Static function that calculates how much memory required. */
    static size_t space_req(coord3 dimensions, int neighbor_store_depth=1);
    
    /** Add a node to the grid. As this is an unstructured grid, there is not
     * necessarily a node for every coordinate within the dimensions range. */
    void add_node(coord3 coord, value_t value = 0, int new_index = -1);
    void del_node(coord3 coord);

    /** Set B as the neighbor of A at offset (seen from A) offs. This also
     * adds A as a neighbor to B (neighborships are symmetric), but at the
     * inverted offset. */
    void add_neighbor(coord3 A, coord3 B, coord3 offset);

    /** Delete neighbor B from A and vice versa. */
    void del_neighbor(coord3 A, coord3 offset);

    /** Returns the index into neighborships where the index of the 
     * requested neighbor is defined.
     *
     * Let x be the return value of this function, then neighborships[x]
     * contains the index in data of the value of the requested neighbor at
     * offset, i.e. data[neighborships[neighbor_pointer(...)]] gives value,
     * neighborships[neighbor_pointer(...)] gives index and
     * neighbor_pointer(...) only gives index into *neighbor_pointer* array. */
    int neighbor_pointer(coord3 coord, coord3 offs);

    /** "Fake" a regular grid by adding the neighborship relations that a
     * regular grid would have, i.e. add top, left, right, bottom, front and
     * back neighbors as in a regular grid. */
    enum layout_t { rowmajor };
    void add_regular_nodes(layout_t layout=rowmajor, coord3 halo=coord3(0, 0, 0))
    void add_regular_neighbors();

    /** Return a new grid with regular neighborship relations. */
    static UnstructuredGrid3D<value_t> *create_regular(coord3 dims, coord3 halo=coord3(0, 0, 0), layout_t layout=rowmajor);

};

// IMPLEMENTATIONS

template<typename value_t>
UnstructuredGrid3D<value_t>::UnstructuredGrid3D(coord3 dimensions, coord3 halo, int *neighborships, int neighbor_store_depth) :
Grid<value_t, coord3>(dimensions, 
                      UnstructuredGrid3D<value_t>::space_req(dimensions, halo, (neighborships == NULL ? 0 : neighbor_store_depth))),
neighborships(neighborships),
neighbor_store_depth(neighbor_store_depth) {
    this->neighborships = NULL;
    this->init();
}

/* Initialization */
template<typename value_t>
void UnstructuredGrid3D<value_t>::init(){
    this->Grid<value_t, coord3>::init(); // this allocates this->data
    coord3 inner = this->dimensions;
    coord3 outer = this->dimensions + 2 * this->halo;
    int inner_sz = inner.x * inner.y * inner.z;
    int outer_sz = outer.x * outer.y * outer.z;
    int halo_sz = outer_sz - inner_sz;
    this->values = this->data; // values are in the first part of our data block
    // initialize empty neighborship pointers
    if(this->neighborships == NULL) {
        this->neighborships = (int*) &this->data[outer_sz];
        int *end = (int *) ((char *)this->data + this->size);
        for(int *ptr = this->neighborships; ptr < end; ptr++) {
            *ptr = -1; // initialize to -1 to siginfy: "no neighbor set"
        }
    }
    this->indices = new std::map<coord3, int>();
    this->coordinates = new std::map<int, coord3>();
}

/* Space Req */
template<typename value_t>
size_t UnstructuredGrid3D<value_t>::space_req(coord3 dimensions, coord3 halo, int n) {
    coord3 outer = dimensions + 2 * halo; 
    const int stored_neighbors_per_node = 2 * n * (n + 1); /* https://oeis.org/A046092 */
    return (  sizeof(value_t) * outer.x * outer.y * outer.z /* for the values */
            + sizeof(int) * stored_neighbors_per_node * outer.x * outer.y ); /* for the ptrs */
}

/* Index */
template<typename value_t>
int UnstructuredGrid3D<value_t>::index(coord3 coord) {
    if(this->indices->count(coord) != 1) {
        throw std::runtime_error("coordinate not in this grid");
    }
    return this->indices[coord];
}

/* Coordinates */
template<typename value_t>
coord3 UnstructuredGrid3D<value_t>::coordinates(int index) {
    if(this->coordinates->count(index) != 1) {
        throw std::runtime_error("index has no associated coordinate in this grid");
    }
    return this->coordinates[coord];
}

/** Gives a pointer into the neighborship_data array 
 * x * ( -2 * (x<0) + 1 ): gives absolute value
 * x * ( -2 * (x<0) + 1 ) * 2 - (x<0): maps negative numbers to odd positions, positive to even
 * TODO: nice interleaving of offsets automatically, not hardcoded for depth=2 like now
 */
#define GRID_UNSTR_2D_NEIGHBOR_PTR(z_stride, index, x, y) /* 2D for case Z=0 */ \
        ( (index) + (z_stride) * (   0 * ((x) == -1 && (y) == 0)\
                                   + 1 * ((x) ==  0 && (y) == -1) \
                                   + 2 * ((x) == +1 && (y) == 0 ) \
                                   + 3 * ((x) ==  0 && (y) == +1) \
                                   + 4 * ((x) == -1 && (y) == -1) \
                                   + 5 * ((x) == -1 && (y) == +1) \
                                   + 6 * ((x) == +1 && (y) == -1) \
                                   + 7 * ((x) == +1 && (y) == +1) \
                                   + 8 * ((x) == +2 && (y) == 0) \
                                   + 9 * ((x) == -2 && (y) == 0) \
                                   +10 * ((x) ==  0 && (y) == -2) \
                                   +11 * ((x) ==  0 && (y) == +2)) )
#define GRID_UNSTR_NEIGHBOR_PTR(z_stride, index, x, y) /* Cases Z>=0 */ \
        GRID_UNSTR_2D_NEIGHBOR_PTR(z_stride, (index) % (z_stride), x, y)
template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor_pointer(int index, coord3 offs) {
    assert(offs.z == 0);
    assert(abs(offs.x) + abs(offs.y) <= this->neighbor_store_depth);
    coord3 M = this->dimensions;
    return GRID_UNSTR_NEIGHBOR_PTR(M.x*M.y, index, offs.x, offs.y);
}

/** Gives the index of the desired neighbor */
#define GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, index, x, y) /* 2D for case Z=0 */ \
     ( neighborships[GRID_UNSTR_2D_NEIGHBOR_PTR(z_stride, index, x, y)] )
#define GRID_UNSTR_NEIGHBOR(neighborships, z_stride, index, x, y, z) /* Cases Z>=0 */ \
     ( neighborships[GRID_UNSTR_NEIGHBOR_PTR(z_stride, index, x, y)] \
            + ((index) / (z_stride)) * (z_stride) /* round off non-Z component of index */ \
            + (z) * (z_stride) )
template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor(int index, coord3 offs) {
    assert(offs.z == 0 || (offs.x == 0 && offs.y == 0)); // only one neighbor at a time, no diagonals
    coord3 M = this->dimensions;
    return GRID_UNSTR_NEIGHBOR(this->neighborships, M.x, M.x*M.y,
                               index, offs.x, offs.y, offs.z);
}

/* Add node */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_node(coord3 coord, value_t value, int new_index) {
    if(new_index == -1) {
        int new_index = this->indices->size();
    }
    this->indices->emplace(coord, new_index);
    this->coordinates->emplace(new_index, coord);
}

/* Delete node */
template<typename value_t>
void UnstructuredGrid3D<value_t>::del_node(coord3 coord) {
    int index = this->index(coord);
    this->indices->erase(coord);
    this->coordinates->erase(index);
}

/* Add neighbor */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(coord3 A, coord3 B, coord3 offs) {
    return this->add_neighbor(this->index(A), this->index(B), offs);
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(int A, int B, coord3 offs) {
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(int A, int B, coord3 offs, int depth) {
    if(offs.x == 0 && offs.y == 0 && offs.z == 0 || depth <= 0) {
        return;
    }
    // Add neighbor to directly involved nodes at offsets
    int A_neigh_idx = this->neighbor_pointer(A, offs);
    int B_neigh_idx = this->neighbor_pointer(B, -offs);
    this->neighborships[A_neigh_idx] = B_idx;
    this->neighborships[B_neigh_idx] = A_idx;
    
    // Recursively update connected neighbors of neighbors
    for(int z_dist = -1; z_dist <= 1; z_dist++) {
        for(int y_dist = -1; y_dist <= 1; y_dist++){
            for(int x_dist = -1; x_dist <= 1; x_dist++) {
                coord3 dist(x_dist, y_dist, z_dist);
                if(dist == coord3(0, 0, 0) || dist == offs) {
                    // this is inefficient
                    continue;
                }
                int neigh_of_neigh = this->neighbor(A, coord3(x_dist, y_dist, z_dist));
                this->add_neighbor(neigh_of_neigh, B, offs - dist, depth - 1)
            }
        }
    }
}


/* Add all the nodes of a regular grid */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_regular_nodes(coord3 halo, layout_t layout) {
    coord3 inner = this->dimensions;
    coord3 outer = this->dimensions + 2*halo;
    /* The halo is added at the very beginning of memory. This way, when iterating
     * from index(0, 0, 0) through memory, no halo cell is ever touched. */
    int new_index = 0;
    for(int z = 0; z<halo.x; z++) {
        for(int y = 0; y<halo.y; y++) {
            for(int x = 0; x<halo.z; x++, new_index++) {
                this->add_node(coord3(x, y, z), 0, new_index);
            }
        }
    }
    for(int z = inner.z+halo.z; z<outer.z; z++) {
        for(int y = inner.y+halo.y; y<outer.y; y++) {
            for(int x = inner.x+halo.x; x<outer.x; x++) {
                this->add_node(coord3(x, y, z), 0, new_index);
            }
        }
    }
    /* Inner fields. */
    for(int z = 0; z<inner.z; z++) {
        for(int y = 0; y<inner.y; y++) {
            for(int x = 0; x<inner.x; x++) {
                this->add_node(coord3(x, y, z), 0, new_index);
                new_index++;
            }
        }
    }
}

/* Add same neighbors as in a regular grid */
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

/* Simulate regular grid with neighbor lookup overhead */
template<typename value_t>
UnstructuredGrid3D<value_t> *UnstructuredGrid3D<value_t>::create_regular(coord3 dims, coord3 halo, layout_t layout) {
    UnstructuredGrid3D<value_t> *grid = new UnstructuredGrid3D<value_t>(dims);
    grid->add_regular_nodes(layout, halo);
    grid->add_regular_neighbors();
    return grid;
}

#endif