#ifndef COORD3_GRID_H
#define COORD3_GRID_H
#include <stdio.h>
#include <cmath>
#include "grid.cu"
#include "coord3.cu"

/** This is still a fairly useless grid as it does not commit on a memory
 * layout and thus does not provide access to storing or getting data.
 * It does, however, provide some functionality that will be common to any grid
 * that uses the coord3 data type for coordinates.
 */

template<typename value_t>
class Coord3BaseGrid : 
virtual public Grid<value_t, coord3> {

    public:
    
    Coord3BaseGrid();

    /** Load values from another grid into this one, if coordinate and value
     * types are compatible. */
    //template<typename other_value_t, typename other_allocator_t>
    void import(Grid<value_t, coord3> *other) final;

    /** Print the grid (makes sense for small sizes for debugging). */
    void print();

    /** Compare. */
    bool compare(Grid<value_t, coord3> *other, double tol=1e-5);

};

// IMPLEMENTATIONS

template<typename value_t>
Coord3BaseGrid<value_t>::Coord3BaseGrid() {}

//template<typename value_t, typename allocator, typename other_value_t, typename other_allocator_t>
template<typename value_t>
void Coord3BaseGrid<value_t>::import(Grid<value_t, coord3> *other) {
    assert(this->dimensions == other->dimensions);
    assert(this->halo == other->halo);
    for(int x = -this->halo.x; x < this->dimensions.x+this->halo.x; x++) {
        for(int y = -this->halo.y; y < this->dimensions.y+this->halo.y; y++) {
            for(int z = -this->halo.z; z < this->dimensions.z+this->halo.z; z++) {
                this->set(coord3(x, y, z), (*other)[coord3(x, y, z)]);
            }
        }
    }
}

template<typename value_t>
void Coord3BaseGrid<value_t>::print() {}

template<>
void Coord3BaseGrid<double>::print() {
    for(int y = -this->halo.y; y < this->dimensions.y+this->halo.y; y++) {
        for(int x = -this->halo.x; x < this->dimensions.x+this->halo.x; x++) {
            fprintf(stderr, "[");
            for(int z = -this->halo.z; z < this->dimensions.z+this->halo.z; z++) {
                fprintf(stderr, "%5.1f", (*this)[coord3(x, y, z)]);
            }
            fprintf(stderr, "]  ");
        }
        fprintf(stderr, "\n");
    }
}

template<typename value_t>
bool Coord3BaseGrid<value_t>::compare(Grid<value_t, coord3> *other, double tol) {
    return false;
}

template<>
bool Coord3BaseGrid<double>::compare(Grid<double, coord3> *other, double tol) {
    if(this->dimensions != other->dimensions) {
        return false;
    }
    for(int x=0; x<other->dimensions.x; x++) {
        for(int y=0; y<other->dimensions.y; y++) {
            for(int z=0; z<other->dimensions.z; z++) {
                double diff = (*other)[coord3(x, y, z)] - (*this)[coord3(x, y, z)];
                if(isnan(diff) || abs(diff) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

template<>
bool Coord3BaseGrid<float>::compare(Grid<float, coord3> *other, double tol) {
    if(this->dimensions != other->dimensions) {
        return false;
    }
    for(int x=0; x<other->dimensions.x; x++) {
        for(int y=0; y<other->dimensions.y; y++) {
            for(int z=0; z<other->dimensions.z; z++) {
                double diff = (*other)[coord3(x, y, z)] - (*this)[coord3(x, y, z)];
                if(isnan(diff) || abs(diff) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

#endif