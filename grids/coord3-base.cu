#ifndef COORD3_GRID_H
#define COORD3_GRID_H
#include <stdio.h>
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

    /** Fill all the cells in the grid with the same value. */
    void fill(value_t value);

    /** Print the grid (makes sense for small sizes for debugging). */
    void print();

    /** Compare. */
    bool compare(Grid<value_t, coord3> *other, double tol=1e-5);

};

//  IMPLEMENTATIONS

template<typename value_t>
Coord3BaseGrid<value_t>::Coord3BaseGrid() {}

//template<typename value_t, typename allocator, typename other_value_t, typename other_allocator_t>
template<typename value_t>
void Coord3BaseGrid<value_t>::import(Grid<value_t, coord3> *other) {
    int N = std::min(this->dimensions.x, other->dimensions.x);
    int M = std::min(this->dimensions.y, other->dimensions.y);
    int L = std::min(this->dimensions.z, other->dimensions.z);
    for(int x = 0; x < N; x++) {
        for(int y = 0; y < M; y++) {
            for(int z = 0; z < L; z++) {
                this->set(coord3(x, y, z), (*other)[coord3(x, y, z)]);
            }
        }
    }
}

template<typename value_t>
void Coord3BaseGrid<value_t>::fill(value_t value) {
    int N = this->dimensions.x;
    int M = this->dimensions.y;
    int L = this->dimensions.z;
    for(int x = 0; x < N; x++) {
        for(int y = 0; y < M; y++) {
            for(int z = 0; z < L; z++) {
                this->set(coord3(x, y, z), value);
            }
        }
    }
}

template<typename value_t>
void Coord3BaseGrid<value_t>::print() {
    int N = this->dimensions.x;
    int M = this->dimensions.y;
    int L = this->dimensions.z;
    for(int x=0; x<N; x++) {
        for(int y=0; y<M; y++) {
            fprintf(stderr, "[");
            for(int z=0; z<L; z++) {
                fprintf(stderr, "%5.1f", (*this)[coord3(x, y, z)]);
            }
            fprintf(stderr, "]  ");
        }
        fprintf(stderr, "\n");
    }
}

template<typename value_t>
bool Coord3BaseGrid<value_t>::compare(Grid<value_t, coord3> *other, double tol) {
    if(this->dimensions != other->dimensions) {
        return false;
    }
    for(int x=0; x<other->dimensions.x; x++) {
        for(int y=0; y<other->dimensions.y; y++) {
            for(int z=0; z<other->dimensions.z; z++) {
                if(abs((*other)[coord3(x, y, z)] - (*this)[coord3(x, y, z)]) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

#endif