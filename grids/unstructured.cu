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

    protected:
    /** Neighborship data pointers can be reused from a different grid. In this
     * case, pass in a pointer as neighborships.
     *
     * neighbor_store_depth gives how many direct pointers to neighbors are
     * stored in memory, i.e. for neighbor_store_depth=2, pointers to
     * neighbors and pointers to neighbors of neighbors are stored. */
    UnstructuredGrid3D() {};
    UnstructuredGrid3D(coord3 dimensions, coord3 halo=coord3(0, 0, 0), int neighbor_store_depth=1, int *neighborships=NULL);

    virtual void init();
    
    public: 
    
    static UnstructuredGrid3D<value_t> *create(coord3 dimensions, coord3 halo=coord3(0, 0, 0), int neighbor_store_depth=1, int *neighborships=NULL);
    
    /** Return a new grid with regular neighborship relations. */
    enum layout_t { rowmajor };
    static UnstructuredGrid3D<value_t> *create_regular(coord3 dims, coord3 halo=coord3(0, 0, 0), layout_t layout=rowmajor, int neighbor_store_depth=1);

    int index(coord3 coord);
    coord3 coordinate(int index);

    using Grid<value_t, coord3>::neighbor;
    int neighbor(int index, coord3 offset);
    
    /** Pointer to where the actual values are stored. Alias for data, but might
     * change if we move neighborship data to beginning of data block. */
    value_t *values;

    /** Pointer to the neighbor memory block, where pointers to values of 
     * neighbors are stored. */
    int *neighborships;
    int neighbor_store_depth=1;

    /** Mapping from X-Y-coordinates to indices. */
    std::map<coord3, int> indices;
    std::map<int, coord3> coordinates;

    /** Add a node to the grid. As this is an unstructured grid, there is not
     * necessarily a node for every coordinate within the dimensions range. */
    void add_node(coord3 coord, value_t value = 0, int new_index = -1);
    void del_node(coord3 coord);

    /** Set B as the neighbor of A at offset (seen from A) offs. This also
     * adds A as a neighbor to B (neighborships are symmetric), but at the
     * inverted offset. */
    void add_neighbor(int A, int B, coord3 offset);
    void add_neighbor(coord3 A, coord3 B, coord3 offset);
    void add_neighbor(int A, int B, coord3 offset, int depth);

    bool has_neighbor(int A, coord3 offset);

    /** Returns the index into neighborships where the index of the 
     * requested neighbor is defined.
     *
     * Let x be the return value of this function, then neighborships[x]
     * contains the index in data of the value of the requested neighbor at
     * offset, i.e. data[neighborships[neighbor_pointer(...)]] gives value,
     * neighborships[neighbor_pointer(...)] gives index and
     * neighbor_pointer(...) only gives index into *neighbor_pointer* array. */
    int neighbor_pointer(int index, coord3 offs);

    /** "Fake" a regular grid by adding the neighborship relations that a
     * regular grid would have, i.e. add top, left, right, bottom, front and
     * back neighbors as in a regular grid. */
    void add_regular_nodes(layout_t layout=rowmajor);
    void add_regular_neighbors();

    /** Grid is regular in Z-dimension. Halo is at the beginning of data. The 
     * following functions return relevant numbers. */
    int z_stride();
    int halo_size();
    bool is_index_in_halo(int index);
    bool is_coordinate_in_halo(coord3 coord);

};

// IMPLEMENTATIONS

template<typename value_t>
UnstructuredGrid3D<value_t>::UnstructuredGrid3D(coord3 dimensions, coord3 halo, int neighbor_store_depth, int *neighborships) :
neighborships(neighborships),
neighbor_store_depth(neighbor_store_depth) {
    coord3 outer = dimensions + 2 * halo; 
    const int stored_neighbors_per_node = 
        (neighborships == NULL ? 
            2 * neighbor_store_depth * (neighbor_store_depth + 1) : 
            0); /* https://oeis.org/A046092 */
    int sz = (  sizeof(value_t) * outer.x * outer.y * outer.z /* for the values */
              + sizeof(int) * stored_neighbors_per_node * outer.x * outer.y ); /* for the ptrs */
    this->dimensions = dimensions;
    this->halo = halo;
    this->size = sz;
}

template<typename value_t>
UnstructuredGrid3D<value_t> *UnstructuredGrid3D<value_t>::create(coord3 dims, coord3 ha, int nsd, int *neigh) {
    UnstructuredGrid3D<value_t> *obj = new UnstructuredGrid3D<value_t>(dims, ha, nsd, neigh);
    obj->init();
    return obj;
}

/* Simulate regular grid with neighbor lookup overhead */
template<typename value_t>
UnstructuredGrid3D<value_t> *UnstructuredGrid3D<value_t>::create_regular(coord3 dims, coord3 halo, layout_t layout, int neighbor_store_depth) {
    UnstructuredGrid3D<value_t> *obj = new UnstructuredGrid3D<value_t>(dims, halo, neighbor_store_depth);
    obj->init();
    obj->add_regular_nodes(layout);
    obj->add_regular_neighbors();
    return obj;
}

/* Initialization */
template<typename value_t>
void UnstructuredGrid3D<value_t>::init(){
    this->Grid<value_t, coord3>::init(); // this allocates this->data
    //coord3 inner = this->dimensions;
    coord3 outer = this->dimensions + 2*this->halo;
    //int inner_sz = inner.x * inner.y * inner.z;
    int outer_sz = outer.x * outer.y * outer.z;
    //int halo_sz = outer_sz - inner_sz;
    this->values = this->data; // values are in the first part of our data block
    // initialize empty neighborship pointers
    if(this->neighborships == NULL) {
        this->neighborships = (int*) &this->data[outer_sz];
        int *end = (int *) ((char *)this->data + this->size);
        for(int *ptr = this->neighborships; ptr < end; ptr++) {
            *ptr = 0; // offset=0 <=> pointing to itself means no neighbor set
        }
    }
}

template<typename value_t>
int UnstructuredGrid3D<value_t>::z_stride() {
    return (this->dimensions.x + 2*this->halo.x) * (this->dimensions.y + 2*this->halo.y);
}

template<typename value_t>
int UnstructuredGrid3D<value_t>::halo_size() {
    return    4 * this->halo.x * this->halo.y  /* outer corners */
            + 2 * this->dimensions.x * this->halo.y /* inner top/bottom edges */
            + 2 * this->dimensions.y * this->halo.x /* inner left/right edges */;
}

template<typename value_t>
bool UnstructuredGrid3D<value_t>::is_index_in_halo(int index) {
    return (index % this->z_stride()) < this->halo_size();
}

template<typename value_t>
bool UnstructuredGrid3D<value_t>::is_coordinate_in_halo(coord3 coord) {
    return ! ( 0 <= coord.x && coord.x < this->dimensions.x &&
               0 <= coord.y && coord.y < this->dimensions.y );
}

/* Index */
template<typename value_t>
int UnstructuredGrid3D<value_t>::index(coord3 coord) {
    coord3 coord_2d(coord.x, coord.y, 0);
    if(this->indices.count(coord_2d) != 1) {
        char msg[100];
        snprintf(msg, 100, "Unstructured grid: coordinate (%d, %d, %d) not in this grid (dimensions %d, %d, %d. halo %d, %d %d.)",
                 coord.x, coord.y, coord.z, this->dimensions.x, this->dimensions.y, this->dimensions.z,
                 this->halo.x, this->halo.y, this->halo.z);
        throw std::runtime_error(msg);
    }
    int idx = this->indices[coord_2d] + this->z_stride() * (coord.z + this->halo.z);
    return idx;
}

/* Coordinates */
template<typename value_t>
coord3 UnstructuredGrid3D<value_t>::coordinate(int index) {
    const int z_stride = this->z_stride();
    int index_2d = index % z_stride;
    if(this->coordinates.count(index_2d) != 1) {
        throw std::runtime_error("Unstructured grid: index has no associated coordinate in this grid");
    }
    coord3 coord = this->coordinates[index_2d];
    coord.z = index / z_stride - this->halo.z;
    return coord;
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
    return GRID_UNSTR_NEIGHBOR_PTR(this->z_stride(), index, offs.x, offs.y);
}

/** Gives the index of the desired neighbor */
#define GRID_UNSTR_2D_NEIGHBOR(neighborships, z_stride, index, x, y) /* 2D for case Z=0 */ \
     ( (index) \
       + (x!=0 || y!=0 ? neighborships[GRID_UNSTR_2D_NEIGHBOR_PTR(z_stride, index, x, y)] : 0 ) )
#define GRID_UNSTR_NEIGHBOR(neighborships, z_stride, index, x, y, z) /* Cases Z>=0 */ \
     ( (index) \
       + (x!=0 || y!=0 ? neighborships[GRID_UNSTR_NEIGHBOR_PTR(z_stride, index, x, y)] : 0) \
       + (z) * (z_stride) )
template<typename value_t>
int UnstructuredGrid3D<value_t>::neighbor(int index, coord3 offs) {
    assert(offs.z == 0 || (offs.x == 0 && offs.y == 0)); // only one neighbor at a time, no diagonals
    return GRID_UNSTR_NEIGHBOR(this->neighborships, this->z_stride(),
                               index, offs.x, offs.y, offs.z);
}

template<typename value_t>
bool UnstructuredGrid3D<value_t>::has_neighbor(int A, coord3 offs) {
    coord3 coord = this->coordinate(A);
    int neigh_ptr = this->neighbor_pointer(A, offs);
    return 0 < neigh_ptr
           && neigh_ptr < (((char*)this->data+this->size) - (char*)this->neighborships)/sizeof(int)
           && this->neighborships[neigh_ptr] != 0;
}

/* Add node */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_node(coord3 coord, value_t value, int new_index) {
    assert(coord.z == 0); // enforce regularity in Z-dimension
    if(new_index == -1) {
        new_index = this->indices.size();
    }
    if(this->is_coordinate_in_halo(coord) && !this->is_index_in_halo(new_index)) {
        char msg[256];
        snprintf(msg, 256, "Coordinate (%d, %d, %d) is in halo (%d, %d, %d) but supplied index (%d) is not in the first portion of memory reserved for the halo (< %d, z_stride %d).",
                 coord.x, coord.y, coord.z,
                 this->halo.x, this->halo.y, this->halo.z,
                 new_index, this->halo_size(), this->z_stride());
        throw std::runtime_error(msg);
    }
    if(!this->is_coordinate_in_halo(coord) && this->is_index_in_halo(new_index)) {
        throw std::runtime_error("Trying to map coordinate to index which is reserved for halo, but coordinate is not in halo.");
    }
    if(this->indices.count(coord) > 0) {
        char msg[100];
        snprintf(msg, 100, "Unstructured Grid: Node (%d, %d, %d) already in grid.", coord.x, coord.y, coord.z);
        throw std::runtime_error(msg);
    }
    this->indices.emplace(coord, new_index);
    this->coordinates.emplace(new_index, coord);
}

/* Delete node */
template<typename value_t>
void UnstructuredGrid3D<value_t>::del_node(coord3 coord) {
    int index = this->index(coord);
    this->indices.erase(coord);
    this->coordinates.erase(index);
}

/* Add neighbor */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(coord3 A, coord3 B, coord3 offs) {
    return this->add_neighbor(this->index(A), this->index(B), offs);
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(int A, int B, coord3 offs) {
    this->add_neighbor(A, B, offs, this->neighbor_store_depth);
}

template<typename value_t>
void UnstructuredGrid3D<value_t>::add_neighbor(int A, int B, coord3 offs, int depth) {
    assert(offs.z == 0);
    if(offs.x == 0 && offs.y == 0 || depth <= 0) {
        return;
    }
    if(offs.x + offs.y > this->neighbor_store_depth) {
        throw std::runtime_error("Offset is farther away than neighbor store depth.");
    }
    // Add neighbor to directly involved nodes at offsets
    int A_2d = A % this->z_stride(); // round off Z-component of index
    int B_2d = B % this->z_stride();

    this->neighborships[this->neighbor_pointer(A_2d, offs)] = B_2d - A_2d;

    if(depth > 1) {
        for(int x_dist = -1; x_dist <= +1; x_dist++) {
            for(int y_dist = -1; y_dist <= +1; y_dist++) {
                coord3 dist(x_dist, y_dist, 0);
                if(abs(dist.x) + abs(dist.y) != 1) {
                    continue;
                }
                if(!this->has_neighbor(B, dist)) {
                    continue;
                }
                /* Connecting A to B requires the following updates to our deep stored neighbors:
                 *  - newly reachable deep neighbors from A (this is the B_neigh part)
                 *  - newly reachable deep neighbors from nodes connected to A (this is the A_neigh part)
                 */
                int A_neigh = this->neighbor(A, dist);
                int B_neigh = this->neighbor(B, dist);
                this->add_neighbor(A_neigh, B, offs - dist, depth - 1);
                this->add_neighbor(A, B_neigh, offs + dist, depth - 1);
            }
        }
    }

}

/* Add all the nodes of a regular grid */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_regular_nodes(layout_t layout) {
    coord3 halo = this->halo;
    coord3 inner = this->dimensions;
    /* The halo is added at the very beginning of memory. This way, when iterating
     * from index(0, 0, 0) through memory, no halo cell is ever touched. */
    int z = 0; // grid is regular in Z-coord
    int new_index = 0;
    /* The halo is allocated in memory from the outward in. This means that if we
     * run through memory from e.g. postion (-1, -1, 0) we will never iterate over
     * a cell that is farther out (e.g. (-2, -1, 0)). 
     * The Z-halo is not handled specially in any way. */
    for(int dst = std::max(halo.x, halo.y); dst > 0; dst--) { // dst is the max distance into the halo
        /* Top/bottom edge of enclosing halo rectangle. */
        if(dst <= halo.x) {
            for(int x = -dst; x < inner.x+dst; x++) {
                this->add_node(coord3(x, -dst, z), 0, new_index);
                new_index++;
                this->add_node(coord3(x, inner.y+dst-1, z), 0, new_index);
                new_index++;
            }
        }
        /* Left/right edge of enclosing halo rectangle. */
        if(dst <= halo.y) {
            for(int y = -dst+1; y < inner.y+dst-1; y++) {
                this->add_node(coord3(-dst, y, z), 0, new_index);
                new_index++;
                this->add_node(coord3(inner.x+dst-1, y, z), 0, new_index);
                new_index++;
            }
        }
    }
    /* Inner fields. */
    assert(new_index == this->halo_size());
    for(int y = 0; y<inner.y; y++) {
        for(int x = 0; x<inner.x; x++) {
            this->add_node(coord3(x, y, z), 0, new_index);
            new_index++;
        }
    }
}

/* Add same neighbors as in a regular grid */
template<typename value_t>
void UnstructuredGrid3D<value_t>::add_regular_neighbors() {
    coord3 dims = this->dimensions;
    coord3 halo = this->halo;
    int z = 0; // neighborship relations are the same across all Z levels
    for(int y = -halo.y; y<dims.y+halo.y; y++) {
        for(int x = -halo.x + 1; x<dims.x+halo.x; x++) { // left neighbors
            this->add_neighbor(coord3(x, y, z), coord3(x-1, y, z), coord3(-1, 0, 0));
            this->add_neighbor(coord3(x-1, y, z), coord3(x, y, z), coord3(+1, 0, 0));
        }
    }
    for(int x = -halo.x; x<dims.x+halo.x; x++) {
        for(int y = -halo.y + 1; y<dims.y+halo.y; y++) {
            this->add_neighbor(coord3(x, y, z), coord3(x, y-1, z), coord3(0, -1, 0));
            this->add_neighbor(coord3(x, y-1, z), coord3(x, y, z), coord3(0, +1, 0));
        }
    }
}

#endif