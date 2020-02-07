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
    // This gives the default ordering as if ordered by indices of a row-major
    // layout: x changing fastest, z slowest
    return A.z < B.z || 
           (A.z == B.z && A.y < B.y) ||
           (A.z == B.z && A.y == B.y && A.x < B.x);
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
#ifdef COORD3_64
typedef _coord3<int64_t> coord3;
#else 
#ifdef COORD3_32
typedef _coord3<int32_t> coord3;
#else
#ifdef COORD3_16
typedef _coord3<int16_t> coord3;
#else
typedef _coord3<int> coord3;
#endif
#endif
#endif

#endif