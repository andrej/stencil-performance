#ifndef GRID_H
#define GRID_H
#include <stddef.h>
#include <assert.h>
#include <memory>

/** Grid
 * 
 * This abstract class provides basic functionality for multi-dimensonal grids
 * of some data type (e.g. floats). All subclasses must support indexing into
 * this grid in one contiguous memory block, but it is left to implementations
 * to decide how to lay out the memory and where to allocate it.
 * 
 * @author André Rösti
 * @date 2019-10-27
 */
template<typename value_t, typename coord_t>
class Grid {

    public:

    Grid();
    Grid(coord_t dimensions, size_t size);
    
    /** Initialization stuff that cannot be done in the constructor because
     * access to subclass virtual funcitons is required. */
    virtual void init();

    virtual ~Grid();

    /** Size as number of items in each dimension. */
    coord_t dimensions;

    /** Space required by this grid in memory in *bytes*. */
    size_t size;

    /** Return a pointer to the first data item of the grid in memory. Note
     * this is not necessarily the item at index (0, 0, 0) -- the
     * implementations may chose to lay out the first element in memory
     * somewhere else. Use index() to get the memory location.*/
    value_t* __restrict__ data;

    /** Return the offset (as number of sizeof(value_t)-sized steps from the 
     * beginning of the memory block of a particular cell of the grid, i.e. the 
     * value of (x, y, z) in grid foo  is located at the pointer 
     * data() + index(dim3(x, y, z)). */
    virtual int index(coord_t coords) = 0;

    /** Give the number of neighbors of cell (x, y, z). For structured grids,
     * this will be the same value independent of (x, y, z), but for
     * unstructured ones this might be variable. */
    virtual size_t num_neighbors(coord_t coords) = 0;

    /** Returns the memory index of the neighbor at a given offset offs. */
    virtual int neighbor(coord_t coords, coord_t offs) = 0;

    /** Returns the memory index from the cell that is stored at the given
     * index with the given offset. */
    virtual int neighbor_of_index(int index, coord_t offs) = 0;
    
    virtual void allocate();

    virtual void deallocate();

    /** Copy all the values from the given grid B into this grid. This operation
     * is not supported between arbitrary types of grids, as conversion between
     * some coordinate systems or data types would not make sense. Rather, the
     * subclasses decide which type of import operations are allowed.
     */
    virtual void import(Grid<value_t, coord_t> *B) = 0;

    /** Get the value stored in the grid at (x, y, z). This is  convenience for 
     * *(data() + index(x, y, z)). The [] operator can also be used to access
     * elements. Implementations may provide optimized versions of get. */
    value_t get(coord_t coords);
    value_t operator[](coord_t coords);

    /** Store the value v  in the grid at (x, y, z). This is convenience for 
     * *(data() + index(x, y, z)) = v. The [] operator can also be used. */
    void set(coord_t coords, value_t v);

    /** Fill all cells in the grid with the given value. */
    virtual void fill(value_t v) = 0;

    /** Print the values of this grid. */
    virtual void print() = 0;

    /** Compare the values in this grid with the values of another grid,
     * check for equality up to some tolerance. */
    virtual bool compare(Grid<value_t, coord_t> *other, double tol=1e-5) = 0;

};

// IMPLEMENTATIONS

template<typename value_t, typename coord_t>
Grid<value_t, coord_t>::Grid() {}

template<typename value_t, typename coord_t>
Grid<value_t, coord_t>::Grid(coord_t dimensions, size_t size) :
dimensions(dimensions),
size(size) {
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::init() {
    this->allocate();
}

template<typename value_t, typename coord_t>
Grid<value_t, coord_t>::~Grid() {
    if(this->data) {
        this->deallocate();
    }
}

template<typename value_t, typename coord_t>
value_t Grid<value_t, coord_t>::get(coord_t coords) {
    return this->data[this->index(coords)];
}

template<typename value_t, typename coord_t>
value_t Grid<value_t, coord_t>::operator[] (const coord_t coords) {
    return this->get(coords);
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::set(coord_t coords, value_t value) {
    this->data[this->index(coords)] = value;
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::allocate() {
    assert(this->size > 0);
    this->data = (value_t *)calloc(this->size, 1);
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::deallocate() {
    free(this->data);
    this->data = NULL;
}
#endif