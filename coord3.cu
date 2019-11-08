#ifndef COORD3_H
#define COORD3_H

#include <cstdint>

/** Signed three dimensional coordinates.
 *
 * If the constructor is supplied with only X and Y coordinates, Z is set to
 * 0.
 */
template<typename T=int64_t>
struct _coord3 {
    T x;
    T y;
    T z;
    _coord3();
    _coord3(T x, T y);
    __host__ __device__
    _coord3(T x, T y, T z);
    void operator=(const _coord3<T> other);
    bool operator==(const _coord3<T> other) const;
    bool operator!=(const _coord3<T> other) const;
};

/** Type to describe neighborship relations between two 3D-coordinates.
 * 
 * Left means a negative step in X direction. Right means a positive step in X.
 * Top means a negative step in Y, bottom a positive step in Y.
 * Front means a negative step in Z, back a positive step in Z.
 * 
 * Values of this enum can be combined by ORing them together, e.g. to get a
 * top-left neighborship relation, do top | left.
 */
typedef enum {
    left     = 0b000001,
    right    = 0b000010,
    top      = 0b000100,
    bottom   = 0b001000,
    front    = 0b010000,
    back     = 0b100000
} coord3_rel;

// IMPLEMENTATIONS

/**
 * Constructors
 */

template<typename T>
_coord3<T>::_coord3(T _x, T _y, T _z) :
    x(_x), y(_y), z(_z) {}

template<typename T>
_coord3<T>::_coord3(T _x, T _y) :
    x(_x), y(_y), z(0) {}

template<typename T>
_coord3<T>::_coord3() :
    x(0), y(0), z(0) {}


/**
 * Operators
 */

template<typename T>
void _coord3<T>::operator=(const _coord3 other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
}

template<typename T>
bool _coord3<T>::operator==(const _coord3<T> other) const {
    return this->x == other.x && this->y == other.y && this->z == other.z;
}

template<typename T>
bool _coord3<T>::operator!=(const _coord3<T> other) const {
    return !this->operator==(other);
}

template<typename T>
bool operator==(const dim3 A, const _coord3<T> B) {
    return A.x == B.x && A.y == B.y && A.z == B.z;
}

template<typename T>
bool operator!=(const dim3 A, const _coord3<T> B) {
    return !operator==(A, B);
}

template<typename T>
bool operator==(const _coord3<T> A, const dim3 B) {
    return operator==(B, A);
}

template<typename T>
bool operator !=(const _coord3<T> A, const dim3 B) {
    return operator!=(B, A);
}

template<typename T>
bool operator<(const _coord3<T> A, const _coord3<T> B) {
    return A.x < B.x || 
           (A.x == B.x && A.y < B.y) ||
           (A.x == B.x && A.y == B.y && A.z < B.z);
}

template<typename T>
__device__ __host__
_coord3<T> operator+(_coord3<T> A, _coord3<T> B) {
    return _coord3<T>(A.x + B.x,
                      A.y + B.y,
                      A.z + B.z);
}

template<typename T>
__device__ __host__
_coord3<T> operator-(_coord3<T> A, _coord3<T> B) {
    return _coord3<T>(A.x - B.x,
                      A.y - B.y,
                      A.z - B.z);
}

template<typename T>
__device__ __host__
_coord3<T> operator-(_coord3<T> A) {
    return _coord3<T>(-A.x,
                      -A.y,
                      -A.z);
}

template<typename T>
__device__ __host__
_coord3<T> operator*(_coord3<T> A, int b) {
    return _coord3<T>(A.x * b,
                      A.y * b,
                      A.z * b);
}

template<typename T>
_coord3<T> operator*(int b, _coord3<T> A) {
    return operator*(A, b);
}

/** Make coord3 available as shorthand for one templated version of the
 * coordinate type. */
#ifndef COORD3_USE_SHORT
typedef _coord3<int64_t> coord3;
#else
typedef _coord3<int32_t> coord3;
#endif

#endif