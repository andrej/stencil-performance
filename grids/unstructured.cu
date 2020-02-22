#ifndef UNSTRUCTURED_GRID_H
#define UNSTRUCTURED_GRID_H
#include <assert.h>
#include <stdlib.h>
#include <memory>
#include <map>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <vector>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include "grids/grid.cu"
#include "grids/coord3-base.cu"
#include "grids/zcurve-util.cu"
#include "coord3.cu"

#define DEFAULT_Z_CURVE_WIDTH 4

enum layout_t { rowmajor, zcurve, random_layout };
enum halo_layout_t { halo_layout_spiral, halo_layout_rowmajor };

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
template<typename value_t, typename neigh_ptr_t = int>
class UnstructuredGrid3D : 
virtual public Coord3BaseGrid<value_t> {

    public:

    /** Neighborship data pointers can be reused from a different grid. In this
     * case, pass in a pointer as neighborships.
     *
     * neighbor_store_depth gives how many direct pointers to neighbors are
     * stored in memory, i.e. for neighbor_store_depth=2, pointers to
     * neighbors and pointers to neighbors of neighbors are stored. */
    UnstructuredGrid3D() {};
    UnstructuredGrid3D(coord3 dimensions, coord3 halo=coord3(0, 0, 0), int neighbor_store_depth=1, neigh_ptr_t *neighborships=NULL, bool use_prototypes=false);

    virtual void init();
    
    static UnstructuredGrid3D<value_t, neigh_ptr_t> *create(coord3 dimensions, coord3 halo=coord3(0, 0, 0), int neighbor_store_depth=1, neigh_ptr_t *neighborships=NULL, bool use_prototypes = false);
    
    /** Return a new grid with regular neighborship relations. */
    static UnstructuredGrid3D<value_t, neigh_ptr_t> *create_regular(coord3 dims, coord3 halo=coord3(0, 0, 0), layout_t layout=rowmajor, int neighbor_store_depth=1, unsigned char z_curve_width = DEFAULT_Z_CURVE_WIDTH, bool use_prototypes = false);

    int index(coord3 coord);
    coord3 coordinate(int index);

    using Grid<value_t, coord3>::neighbor;
    int neighbor(int index, coord3 offset);
    
    /** Pointer to where the actual values are stored. Alias for data, but might
     * change if we move neighborship data to beginning of data block. */
    value_t *values;

    /** Pointer to the neighbor memory block, where pointers to values of 
     * neighbors are stored. */
    neigh_ptr_t *neighborships;
    int neighbor_store_depth=1;

    /** Mapping from X-Y-coordinates to indices. */
    std::map<coord3, int> indices;
    std::map<int, coord3> coordinates;

    /** As an optimization, the neighborships table can be compressed. In that
     * case, all nodes which share the same relative neighborship offsets
     * (i.e. would be all in a regular grid) request the same pointer in this
     * prototype array that stores the typical neighbor offsets.
     */
    bool use_prototypes = false;
    neigh_ptr_t *prototypes = NULL;
    int n_prototypes = 0;
    void compress();

    /** Add a node to the grid. As this is an unstructured grid, there is not
     * necessarily a node for every coordinate within the dimensions range. */
    void add_node(coord3 coord, int new_index = -1);
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
    void add_regular_nodes(layout_t layout=rowmajor, unsigned char z_curve_width = DEFAULT_Z_CURVE_WIDTH);
    void add_regular_halo(halo_layout_t halo_layout=halo_layout_rowmajor);
    void add_regular_neighbors();

    /** Grid is regular in Z-dimension. Halo is at the beginning of data. The 
     * following functions return relevant numbers. */
    int z_stride();
    int neigh_stride(); // give stride between separate neighborship tables (top, left, ...)
    int halo_size();
    int n_stored_neighbors();
    bool is_index_in_halo(int index);
    bool is_coordinate_in_halo(coord3 coord);

    /** Give indices of our data block, as we have also a neighborship block
     * which should not be overwritten by data values! */
    virtual int values_start();
    virtual int values_stop();

    /** For debugging */
    void print_neighborships();
    void print_prototypes();

    /** Serialize the additional fields this class needs to operate when serializing. */
    template<class Archive> void serialize(Archive &ar, const unsigned int version);

    /** Link another unstructured grid to use the exact same neighborship relations and prototypes as this one. */
    void link(UnstructuredGrid3D<value_t, neigh_ptr_t> *other);

};

// IMPLEMENTATIONS

template<typename value_t, typename neigh_ptr_t>
UnstructuredGrid3D<value_t, neigh_ptr_t>::UnstructuredGrid3D(coord3 dimensions, coord3 halo, int neighbor_store_depth, neigh_ptr_t *neighborships, bool use_prototypes) :
neighborships(neighborships),
neighbor_store_depth(neighbor_store_depth),
use_prototypes(use_prototypes) {
    this->dimensions = dimensions;
    this->halo = halo;
    coord3 outer = dimensions + 2 * halo; 
    const int stored_neighbors_per_node = (neighborships == NULL ? this->n_stored_neighbors() : 0);
    int sz = (  sizeof(value_t) * this->values_stop() /* for the values */
              + sizeof(neigh_ptr_t) * stored_neighbors_per_node * outer.x * outer.y /* for the ptrs */
              + (use_prototypes ? sizeof(neigh_ptr_t) * outer.x * outer.y : 0) ); /* for the prototypes */
    this->size = sz;
}

template<typename value_t, typename neigh_ptr_t>
UnstructuredGrid3D<value_t, neigh_ptr_t> *UnstructuredGrid3D<value_t, neigh_ptr_t>::create(coord3 dims, coord3 ha, int nsd, neigh_ptr_t *neigh, bool use_prototypes) {
    UnstructuredGrid3D<value_t, neigh_ptr_t> *obj = new UnstructuredGrid3D<value_t, neigh_ptr_t>(dims, ha, nsd, neigh, use_prototypes);
    obj->init();
    return obj;
}

/* Simulate regular grid with neighbor lookup overhead */
template<typename value_t, typename neigh_ptr_t>
UnstructuredGrid3D<value_t, neigh_ptr_t> *UnstructuredGrid3D<value_t, neigh_ptr_t>::create_regular(coord3 dims, coord3 halo, layout_t layout, int neighbor_store_depth, unsigned char z_curve_width, bool use_prototypes) {
    UnstructuredGrid3D<value_t, neigh_ptr_t> *obj = new UnstructuredGrid3D<value_t, neigh_ptr_t>(dims, halo, neighbor_store_depth, use_prototypes);
    obj->init();
    obj->add_regular_nodes(layout, z_curve_width);
    obj->add_regular_neighbors();
    return obj;
}

/* Initialization */
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::init(){
    this->Grid<value_t, coord3>::init(); // this allocates this->data
    coord3 outer = this->dimensions + 2*this->halo;
    this->values = this->data; // values are in the first part of our data block
    // initialize empty neighborship pointers
    if(this->neighborships == NULL) {
        this->neighborships = (neigh_ptr_t *) &this->data[this->values_stop()];
        /* We do not do zero initialization here because if this was loaded from
         * serialized data the neighborships might already be set. This should
         * be zeroed out on allocation either way!
        neigh_ptr_t *end = (neigh_ptr_t *) ((char *)this->data + this->size);
        for(neigh_ptr_t *ptr = this->neighborships; ptr < end; ptr++) {
            *ptr = 0; // offset=0 <=> pointing to itself means no neighbor set
        }*/
    }
    bool clear_protos = false;
    if(this->prototypes == NULL) {
        clear_protos = true;
    }
    if(!this->n_prototypes) {
        // in case of load after serialization, this will be already set
        this->n_prototypes = outer.x*outer.y;
    }
    int offs = outer.x*outer.y * this->n_stored_neighbors();
    this->prototypes = (neigh_ptr_t *) &this->neighborships[offs];
    if(this->use_prototypes && clear_protos) {
        coord3 outer = this->dimensions + 2 * this->halo;
        for(int i = 0; i < outer.x*outer.y; i++) {
            this->prototypes[i] = i; // Default: just point to itself in neighborship table
        }
    }
}

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::z_stride() {
    return (this->dimensions.x + 2*this->halo.x) * (this->dimensions.y + 2*this->halo.y);
}

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::neigh_stride() {
    // in case no compression is used, n_prototypes == z_stride()
    return this->n_prototypes;
}

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::n_stored_neighbors() {
    int n_stored_neighbors  = 2 * this->neighbor_store_depth * (this->neighbor_store_depth + 1);
    /* https://oeis.org/A046092 */
    return n_stored_neighbors;
}

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::halo_size() {
    return    4 * this->halo.x * this->halo.y  /* outer corners */
            + 2 * this->dimensions.x * this->halo.y /* inner top/bottom edges */
            + 2 * this->dimensions.y * this->halo.x /* inner left/right edges */;
}

template<typename value_t, typename neigh_ptr_t>
bool UnstructuredGrid3D<value_t, neigh_ptr_t>::is_index_in_halo(int index) {
    return (index % this->z_stride()) < this->halo_size();
}

template<typename value_t, typename neigh_ptr_t>
bool UnstructuredGrid3D<value_t, neigh_ptr_t>::is_coordinate_in_halo(coord3 coord) {
    return ! ( 0 <= coord.x && coord.x < this->dimensions.x &&
               0 <= coord.y && coord.y < this->dimensions.y );
}

/* Index */
template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::index(coord3 coord) {
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
template<typename value_t, typename neigh_ptr_t>
coord3 UnstructuredGrid3D<value_t, neigh_ptr_t>::coordinate(int index) {
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
#define GRID_UNSTR_2D_NEIGHBOR_PTR(neigh_stride, index, x, y) /* 2D for case Z=0 */ \
        ( (index) + (neigh_stride) * (   0 * ((x) == -1 && (y) == 0)\
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
#define GRID_UNSTR_NEIGHBOR_PTR(z_stride, neigh_stride, index, x, y) /* Cases Z>=0 */ \
        GRID_UNSTR_2D_NEIGHBOR_PTR(neigh_stride, (index) % (z_stride), x, y)
template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::neighbor_pointer(int index, coord3 offs) {
    assert(offs.z == 0);
    assert(abs(offs.x) + abs(offs.y) <= this->neighbor_store_depth);
    int z_stride = this->z_stride();
    int neigh_stride = this->neigh_stride();
    if(this->use_prototypes) {
        index = this->prototypes[index];
    }
    return GRID_UNSTR_NEIGHBOR_PTR(z_stride, neigh_stride, index, offs.x, offs.y);
}

/** Gives the index of the desired neighbor */
// Using 2D neighbor saves expensive modulus operation -> must ensure Z=0!
#define GRID_UNSTR_2D_NEIGHBOR_(neighborships, neigh_stride, index, proto_index, x, y) /* 2D for case Z=0 */ \
     ( (index) \
       + (x!=0 || y!=0 ? (int)neighborships[GRID_UNSTR_2D_NEIGHBOR_PTR(neigh_stride, proto_index, x, y)] : 0 ) )
#define GRID_UNSTR_2D_NEIGHBOR(neighborships, neigh_stride, index, x, y) \
        GRID_UNSTR_2D_NEIGHBOR_(neighborships, neigh_stride, index, index, x, y)
#define GRID_UNSTR_PROTO_2D_NEIGHBOR(prototypes, neighborships, neigh_stride, index, x, y) \
        GRID_UNSTR_2D_NEIGHBOR_(neighborships, neigh_stride, index, prototypes[index], x, y)
// General purpose macro that also works for indices with Z>0
#define GRID_UNSTR_NEIGHBOR_(neighborships, z_stride, neigh_stride, index, proto_index, x, y, z) /* Cases Z>=0 */ \
     ( (index) \
       + (x!=0 || y!=0 ? neighborships[GRID_UNSTR_NEIGHBOR_PTR(z_stride, neigh_stride, proto_index, x, y)] : 0) \
       + (z) * (z_stride) )
#define GRID_UNSTR_NEIGHBOR(neighborships, z_stride, index, x, y, z) /* for grids not using compression neigh_stride = z_stride */ \
        GRID_UNSTR_NEIGHBOR_(neighborships, z_stride, z_stride, index, index, x, y, z)
#define GRID_UNSTR_PROTO_NEIGHBOR(prototypes, neighborships, z_stride, neigh_stride, index, x, y, z) \
        GRID_UNSTR_NEIGHBOR_(neighborships, z_stride, neigh_stride, index, prototypes[index % z_stride], x, y, z)

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::neighbor(int index, coord3 offs) {
    assert(offs.z == 0 || (offs.x == 0 && offs.y == 0)); // only one neighbor at a time, no diagonals
    int z_stride = this->z_stride();
    int neigh_stride = this->neigh_stride();
    if(this->use_prototypes) {
        return GRID_UNSTR_PROTO_NEIGHBOR(this->prototypes, this->neighborships, z_stride, neigh_stride,
                                         index, offs.x, offs.y, offs.z);
    }
    return GRID_UNSTR_NEIGHBOR(this->neighborships, z_stride,
                               index, offs.x, offs.y, offs.z);
}

template<typename value_t, typename neigh_ptr_t>
bool UnstructuredGrid3D<value_t, neigh_ptr_t>::has_neighbor(int A, coord3 offs) {
    coord3 coord = this->coordinate(A);
    int neigh_ptr = this->neighbor_pointer(A, offs);
    return 0 < neigh_ptr
           && neigh_ptr < (((char*)this->data+this->size) - (char*)this->neighborships)/sizeof(int)
           && this->neighborships[neigh_ptr] != 0;
}

/* Add node */
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_node(coord3 coord, int new_index) {
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
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::del_node(coord3 coord) {
    int index = this->index(coord);
    this->indices.erase(coord);
    this->coordinates.erase(index);
}

/* Add neighbor */
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_neighbor(coord3 A, coord3 B, coord3 offs) {
    return this->add_neighbor(this->index(A), this->index(B), offs);
}

template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_neighbor(int A, int B, coord3 offs) {
    this->add_neighbor(A, B, offs, this->neighbor_store_depth);
}

template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_neighbor(int A, int B, coord3 offs, int depth) {
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

/* Compress using prototypes */
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::compress() {
    if(!this->use_prototypes || !this->prototypes || !this->neighborships) {
        return;
    }

    coord3 outer = this->dimensions + 2*this->halo;
    int n_stored_neighbors  = this->n_stored_neighbors();
    int z_stride = this->z_stride();
    int neigh_stride = this->neigh_stride();

    struct new_proto_t {
        std::vector<int> neigh_offs;
        std::vector<int> cells;
    };
    std::vector<new_proto_t> new_protos;
    for(int i = 0; i < z_stride; i++) { // iterate over all cells in XY plane
        int idx = this->prototypes[i]; // in case we are already operating on compressed data
        auto match = new_protos.end();
        for(auto it = new_protos.begin(); it != new_protos.end(); ++it) { // check for a matching prototype
            bool does_match = true;
            for(int k = 0; k < n_stored_neighbors; k++) {
                if(this->neighborships[idx + k*neigh_stride] != it->neigh_offs.at(k)) {
                    does_match = false;
                    break;
                }
            }
            if(does_match) {
                match = it;
                break;
            }
        }
        if(match != new_protos.end()) { // found a match
            match->cells.push_back(i);
        } else { // no match -> add this cells neighborship relations as new prototype
            new_proto_t new_proto;
            for(int k = 0; k < n_stored_neighbors; k++) {
                new_proto.neigh_offs.push_back(this->neighborships[i + k*neigh_stride]);
            }
            new_proto.cells.push_back(i);
            new_protos.push_back(new_proto);
        }
    }

    this->n_prototypes = new_protos.size(); // determines new neigh_stride!
    neigh_stride = this->neigh_stride();

    // sort prototypes by frequency of use
    auto comparer = [](auto a, auto b) { return a.cells.size() < b.cells.size(); };
    std::sort(new_protos.begin(), new_protos.end(), comparer);

    // store new compact neighborship relations & new prototype pointers
    int i = 0;
    for(auto proto = new_protos.begin(); proto != new_protos.end(); ++proto, ++i) {
        // store neighbor pointers for this prototype at next consecutive slot i
        for(int k = 0; k < n_stored_neighbors; k++) {
            this->neighborships[i + k*neigh_stride] = proto->neigh_offs.at(k);
        }
        // update prototype pointers for all cells that are of this proto type
        for(auto cell = proto->cells.begin(); cell != proto->cells.end(); ++cell) {
            this->prototypes[*cell] = i;
        }
    }
    
}

/* Add all the nodes of a regular grid */
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_regular_nodes(layout_t layout, unsigned char z_curve_width) {
    // Add the halo nodes with their seperate layout, first of everything else
    this->add_regular_halo();

    // Create vector of all inner coordinates; the ordering determines the memory layout
    std::vector<coord3> coord_sequence;
    int z = 0;
    for(int y = 0; y<this->dimensions.y; y++) {
        for(int x = 0; x<this->dimensions.x; x++) {
            coord_sequence.push_back(coord3(x, y, z));
        }
    }

    // How this vector is sorted determines the memory layout
    // For z-curves this ensures that the data is still dense in the given index range,
    // even if the grid is not square (We simply use the Z-curve index for comparison
    // between coordinates, i.e. ordering. In the grid, the indices might be lower, not sparse).    
    if(layout == zcurve) {
        StretchedZCurveComparator comp(z_curve_width);
        std::sort(coord_sequence.begin(), coord_sequence.end(), comp);
    } else if(layout == random_layout) {
        // The first (0, 0, 0) element should not be shuffled as pointer to (0, 0, 0) is used as
        // pointer to the start of the inner data!
        std::random_shuffle(coord_sequence.begin()+1, coord_sequence.end());
    } else {
        // rowmajor; already sorted that way by the way we inserted it
    }

    // Actually add the nodes
    int new_index = this->halo_size();
    for(auto it = coord_sequence.begin(); it != coord_sequence.end(); ++it) {
        this->add_node(*it, new_index);
        new_index++;
    }
}

template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_regular_halo(halo_layout_t halo_layout) {
    coord3 halo = this->halo;
    coord3 inner = this->dimensions;
    /* The halo is added at the very beginning of each X-Y-plane memory slice.
     * This way, when iterating from index(0, 0, z) through memory until
     * (dimensions.x-1, dimensions.y-1, z), no halo cell is ever touched. */
    int z = 0; // grid is regular in Z-coord
    int new_index = 0;
    if(halo_layout == halo_layout_spiral) {
        /* The halo is allocated in memory from the outward in. This means that if we
        * run through memory from e.g. postion (-1, -1, 0) we will never iterate over
        * a cell that is farther out (e.g. (-2, -1, 0)). 
        * The Z-halo is not handled specially in any way. */
        for(int dst = std::max(halo.x, halo.y); dst > 0; dst--) { // dst is the max distance into the halo
            /* Top/bottom edge of enclosing halo rectangle. */
            if(dst <= halo.x) {
                for(int x = -dst; x < inner.x+dst; x++) {
                    this->add_node(coord3(x, -dst, z), new_index);
                    new_index++;
                    this->add_node(coord3(x, inner.y+dst-1, z), new_index);
                    new_index++;
                }
            }
            /* Left/right edge of enclosing halo rectangle. */
            if(dst <= halo.y) {
                for(int y = -dst+1; y < inner.y+dst-1; y++) {
                    this->add_node(coord3(-dst, y, z), new_index);
                    new_index++;
                    this->add_node(coord3(inner.x+dst-1, y, z), new_index);
                    new_index++;
                }
            }
        }
    } else if(halo_layout == halo_layout_rowmajor) {
        for(int y = -halo.y; y < inner.y + halo.y; y++) {
            for(int x = -halo.x; x < inner.x + halo.x; x++) {
                if(0 <= x  && x < inner.x && 0 <= y && y < inner.y) {
                    continue;
                }
                this->add_node(coord3(x, y, z), new_index);
                new_index++;
            }
        }
    }
    assert(new_index == this->halo_size());
}

/* Add same neighbors as in a regular grid */
template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::add_regular_neighbors() {
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

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::values_start() {
    return 0;
}

template<typename value_t, typename neigh_ptr_t>
int UnstructuredGrid3D<value_t, neigh_ptr_t>::values_stop() {
    coord3 outer = this->dimensions + 2 * this->halo;
    return outer.x * outer.y * outer.z;
}

template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::print_neighborships() {
    int z_stride = this->z_stride();
    int neigh_stride = this->neigh_stride();
    int n_stored_neighbors = this->n_stored_neighbors();
    fprintf(stderr, "Index       | Coordinate | Prototype  | Relative Neigh. Offs (Absolute, Coord)\n");
    for(int i = 0; i < z_stride; i++) {
        coord3 coord = this->coordinate(i);
        neigh_ptr_t proto = this->prototypes[i];
        fprintf(stderr, "%11d |%3d,%3d,%3d |%11d |", i, coord.x, coord.y, coord.z, proto);
        for(int k = 0; k < n_stored_neighbors; k++) {
            int neigh_offs = this->neighborships[proto + k*neigh_stride];
            int neigh_idx = i + neigh_offs;
            coord3 neigh_coord = this->coordinate(neigh_idx);
            fprintf(stderr, "%4d (%4d, (%3d,%3d,%3d)), ", neigh_offs, neigh_idx, neigh_coord.x, neigh_coord.y, neigh_coord.z);
        }
        fprintf(stderr, "\n");
    }
}

template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::print_prototypes() {
    int z_stride = this->z_stride();
    int neigh_stride = this->neigh_stride();
    int n_stored_neighbors = this->n_stored_neighbors();
    int protoype_frequencies[neigh_stride] = {0};
    for(int i = 0; i < z_stride; i++) {
        protoype_frequencies[this->prototypes[i]]++;
    }
    fprintf(stderr, "Prototype   | Frequency  | Relative Neigh. Offs\n");
    for(int i = 0; i < neigh_stride; i++) {
        int freq = protoype_frequencies[i];
        double rel = freq / (double)z_stride * 100;
        fprintf(stderr, "%11d | %3d (%3f%%) | ", i, freq, rel);
        for(int k = 0; k < n_stored_neighbors; k++) {
            fprintf(stderr, "%+4d, ", this->neighborships[i + k*neigh_stride]);
        }
        fprintf(stderr, "\n");
    }
}

template<typename value_t, typename neigh_ptr_t>
template<class Archive>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::serialize(Archive &ar, unsigned int version) {
    ar & this->indices;
    ar & this->coordinates;
    ar & this->neighbor_store_depth;
    ar & this->use_prototypes;
    ar & this->n_prototypes;
    // the call to base class serialize also calls init() on load!
    // therefore, it must be called last (after prototypes etc are set), otherwise those values in data will get overriden by initializer
    ar & boost::serialization::base_object<Coord3BaseGrid<value_t>>(*this); 
}

template<typename value_t, typename neigh_ptr_t>
void UnstructuredGrid3D<value_t, neigh_ptr_t>::link(UnstructuredGrid3D<value_t, neigh_ptr_t> *other) {
    this->indices = other->indices;
    this->coordinates = other->coordinates;
    this->neighborships = other->neighborships;
    this->use_prototypes = other->use_prototypes;
    this->n_prototypes = other->n_prototypes;
    this->prototypes = other->prototypes;
}

#endif