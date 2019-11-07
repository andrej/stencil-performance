#ifndef COORD3_H
#define COORD3_H

/** Signed three dimensional coordinates.
 *
 * If the constructor is supplied with only X and Y coordinates, Z is set to
 * 0.
 */
struct coord3 {
    int x;
    int y;
    int z;
    coord3();
    coord3(int x, int y);
    __host__ __device__
    coord3(int x, int y, int z);
    //coord3 operator+(coord3* B);
    //coord3 operator-();
    //coord3 operator-(coord3* B);
    //coord3 operator*(int other);
    void operator=(const coord3 other);
    bool operator==(const coord3 other) const;
    bool operator!=(const coord3 other) const;
    //bool operator<(const coord3 other) const;
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

coord3::coord3(int _x, int _y, int _z) :
    x(_x), y(_y), z(_z) {}

coord3::coord3(int _x, int _y) :
    x(_x), y(_y), z(0) {}

coord3::coord3() :
    x(0), y(0), z(0) {}

void coord3::operator=(const coord3 other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
}

bool coord3::operator==(const coord3 other) const {
    return this->x == other.x && this->y == other.y && this->z == other.z;
}

bool coord3::operator!=(const coord3 other) const {
    return !this->operator==(other);
}

bool operator==(const dim3 A, const coord3 B) {
    return A.x == B.x && A.y == B.y && A.z == B.z;
}

bool operator!=(const dim3 A, const coord3 B) {
    return !operator==(A, B);
}

bool operator==(const coord3 A, const dim3 B) {
    return operator==(B, A);
}

bool operator !=(const coord3 A, const dim3 B) {
    return operator!=(B, A);
}

bool operator<(const coord3 A, const coord3 B) {
    return A.x < B.x || 
           (A.x == B.x && A.y < B.y) ||
           (A.x == B.x && A.y == B.y && A.z < B.z);
}

__device__ __host__
coord3 operator+(coord3 A, coord3 B) {
    return coord3(A.x + B.x,
                  A.y + B.y,
                  A.z + B.z);
}

__device__ __host__
coord3 operator-(coord3 A, coord3 B) {
    return coord3(A.x - B.x,
                  A.y - B.y,
                  A.z - B.z);
}

__device__ __host__
coord3 operator-(coord3 A) {
    return coord3(-A.x,
                  -A.y,
                  -A.z);
}

__device__ __host__
coord3 operator*(coord3 A, int b) {
    return coord3(A.x * b,
                  A.y * b,
                  A.z * b);
}

coord3 operator*(int b, coord3 A) {
    return operator*(A, b);
}

#endif