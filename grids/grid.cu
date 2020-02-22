#ifndef GRID_H
#define GRID_H
#include <stddef.h>
#include <assert.h>
#include <memory>
#include <random>
#include <iostream>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/split_member.hpp>
#include "coord3.cu"

/** Grid
 * 
 * This abstract class provides basic functionality for multi-dimensonal grids
 * of some data type (e.g. floats). All subclasses must support indexing into
 * this grid in one contiguous memory block, but it is left to implementations
 * to decide how to lay out the memory and where to allocate it.
 *
 * Coordinates provide a means of identifying cells in the grid. It is up to
 * subclasses to determine the exact meaning and relations of coordinates.
 * Subclasses must use coordinates that are within the given dimensions or at
 * most outside of the dimensions by an anmount given by the halo property.
 *
 * At the very least, subclasses must provide
 *  - Coordinate to data index translation in index()
 *  - Index to coordinate translation in coordinate()
 *  - Neighborship lookup using one neighbor() function
 *  - A static "create" method that returns an object of the type and calls init()
 */
template<typename value_t, typename coord_t>
class Grid {

    friend class boost::serialization::access;

    protected:
    // Subclasses 
    Grid() = default;
    virtual ~Grid();

    /** Initialization stuff that cannot be done in the constructor because
     * access to subclass virtual funcitons is required. */
    virtual void init();

    virtual void allocate();

    virtual void deallocate();

    public:

    /** Size as number of items in each dimension. */
    coord_t dimensions;

    /** Halo denotes how many coordinates beyond the dimensions should be
     * accessible, i.e. how many negative coordinates and how many beyond the
     * max size. This is useful to avoid bounds checks. */
    coord_t halo;

    /** Space required by this grid in memory in *bytes*. */
    size_t size;

    /** Return a pointer to the first data item of the grid in memory. Note
     * this is not necessarily the item at index (0, 0, 0) -- the
     * implementations may chose to lay out the first element in memory
     * somewhere else. Use index() to get the memory location.*/
    value_t* __restrict__ data = NULL;

    /** Return the offset (as number of sizeof(value_t)-sized steps from the 
     * beginning of the data block of a particular cell of the grid, i.e. the 
     * value of (x, y, z) in grid foo  is located at the pointer 
     * data + index(dim3(x, y, z)). */
    virtual int index(coord_t coords) = 0;
    value_t *pointer(coord_t coords); // return a pointer to the given coords
    virtual coord_t coordinate(int index) = 0;

    /** Returns the memory index of the neighbor at a given offset offs. */
    virtual int neighbor(int index, coord_t offs);
    virtual int neighbor(coord_t coords, coord_t offs);

    /** Copy all the values from the given grid B into this grid. This operation
     * is not supported between arbitrary types of grids, as conversion between
     * some coordinate systems or data types may not make sense. Rather, the
     * subclasses decide which type of import operations are allowed.
     */
    virtual void import(Grid<value_t, coord_t> *B) = 0;

    /** Get the value stored in the grid at (x, y, z). This is  convenience for 
     * *(data() + index(x, y, z)). The [] operator can also be used to access
     * elements. Implementations may provide optimized versions of get. */
    value_t get(int index);
    value_t get(coord_t coords);
    value_t operator[](int index);
    value_t operator[](coord_t coords);

    /** Store the value v  in the grid at (x, y, z). This is convenience for 
     * *(data() + index(x, y, z)) = v. The [] operator can also be used. */
    void set(int index, value_t v);
    void set(coord_t coords, value_t v);

    /** Fill all cells in the grid with the given value. */
    virtual void fill(value_t v);
    virtual void fill_random();

    /** Print the values of this grid. */
    virtual void print() = 0;

    /** Compare the values in this grid with the values of another grid,
     * check for equality up to some tolerance. */
    virtual bool compare(Grid<value_t, coord_t> *other, double tol=1e-5) = 0;

    /** Give indices into data array where the actual values start and stop.
     * Note that the values must be a contiguous block and only data values
     * may be within this range. */
    virtual int values_start();
    virtual int values_stop();

    /** Serialize grid and put into / take out of stream, i.e. for storing to files. */
    BOOST_SERIALIZATION_SPLIT_MEMBER();
    template<class Archive> void save(Archive &ar, const unsigned int version) const;
    template<class Archive> void load(Archive &ar, const unsigned int version);

};

// IMPLEMENTATIONS

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::init() {
    this->allocate();
}

template<typename value_t, typename coord_t>
Grid<value_t, coord_t>::~Grid() {
    /*if(this->data) {
        this->deallocate();
    }*/
}

template<typename value_t, typename coord_t>
value_t Grid<value_t, coord_t>::get(int index) {
    return this->data[index];
}

template<typename value_t, typename coord_t>
value_t Grid<value_t, coord_t>::get(coord_t coords) {
    return this->get(this->index(coords));
}

template<typename value_t, typename coord_t>
value_t Grid<value_t, coord_t>::operator[] (const coord_t coords) {
    return this->get(coords);
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::set(int index, value_t value) {
    this->data[index] = value;
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::set(coord_t coords, value_t value) {
    this->set(this->index(coords), value);
}

template<typename value_t, typename coord_t>
value_t *Grid<value_t, coord_t>::pointer(coord_t coords) {
    return &this->data[this->index(coords)];
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::allocate() {
    assert(this->size > 0);
    this->data = (value_t *)calloc(this->size, 1);
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::deallocate() {
    if(!this->data) {
        return;
    }
    free(this->data);
    this->data = NULL;
}

template<typename value_t, typename coord_t>
int Grid<value_t, coord_t>::neighbor(coord_t coords, coord_t offs) {
    return this->neighbor(this->index(coords), offs);
}

template<typename value_t, typename coord_t>
int Grid<value_t, coord_t>::neighbor(int index, coord_t offs) {
    return this->neighbor(this->coordinate(index), offs);
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::fill(value_t v) {
    memset(this->data, 0, this->size);
}

template<>
void Grid<double, coord3>::fill(double v) {
    memset(this->data, v, this->size);
}

template<>
void Grid<float, coord3>::fill(float v) {
    memset(this->data, v, this->size);
}

template<typename value_t, typename coord_t>
void Grid<value_t, coord_t>::fill_random() {
}

template<>
void Grid<double, coord3>::fill_random() {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, +1.0);
    const int values_start = this->values_start();
    const int values_stop = this->values_stop();
    for(int i = values_start; i < values_stop; i++) {
        this->data[i] = dist(gen);
    }
}

template<typename value_t, typename coord_t>
int Grid<value_t, coord_t>::values_start() {
    return 0;
}

template<typename value_t, typename coord_t>
int Grid<value_t, coord_t>::values_stop() {
    return this->size / sizeof(value_t);
}

template<typename value_t, typename coord_t>
template<class Archive>
void Grid<value_t, coord_t>::save(Archive &ar, const unsigned int version) const {
    ar << this->size;
    ar << this->dimensions;
    ar << this->halo;
    ar << boost::serialization::make_binary_object(this->data, this->size);
}

template<typename value_t, typename coord_t>
template<class Archive>
void Grid<value_t, coord_t>::load(Archive &ar, const unsigned int version) {
    ar >> this->size;
    ar >> this->dimensions;
    ar >> this->halo;
    this->deallocate(); // new data -> free previous
    this->init(); // new data might be larger/smaller, allocate anew
    // init also does other things in subclasses so all members are set up right
    ar >> boost::serialization::make_binary_object((char *)this->data, this->size);
}
#endif