#ifndef ZCURVE_UTIL_H
#define ZCURVE_UTIL_H

#include "coord3.cu"

/** Given two integers, intertwine their bits. The first integer will have its
 * bits at the lower (even) positions. */
long intertwine(int a, int b) {

    // Get max "length" of both integers, i.e. index of most significant set bit + 1
    unsigned char n_bits = 0;
    while(a >> n_bits || b >> n_bits) {
        ++n_bits;
    }

    // spread out both a and b to have zero bits in all odd positions
    long a_stretch = 0;
    long b_stretch = 0;
    for(unsigned char i = 0; i < n_bits; i++) {
        a_stretch |= ((a>>i) & 1) << 2*i;
        b_stretch |= ((b>>i) & 1) << 2*i;
    }
            
    // intertwine all bits, x in even, y in odd position
    return a_stretch | (b_stretch << 1);

}

/** Return the regular stretched Z-curve index for coordinate X, Y. The
 * stretched Z-curve is like a regular Z-curve (Morton curve) for w=0. For w>0,
 * the bit at index w and all following ones up to the least significant one
 * are NOT intertwined with the Y coordinate.
 *
 * If x0, x1, ... denotes the least significant, second, ... bit of x, then we 
 * have for w = 0:
 * index(x, y) = yn xn ... y1 x1 y0 x0
 *
 * For w > 0:
 * index(x, y) = yn xn ... y1 xw+1 y0 xw xw-1 ... x1 x0
 */
long stretched_z_curve_index(int x, int y, unsigned char w = 0) {
    int x_upper = x >> w; 
    int x_lower = x & ((1 << w) - 1);
    long res = intertwine(x_upper, y);
    return res << w | x_lower;
}

/** Comparator gives order of given coordinates in wide Z-curved layout. */
struct StretchedZCurveComparator {
    unsigned char w = 0;
    StretchedZCurveComparator(unsigned char w=0) : w(w) {}
    bool operator() (coord3 a, coord3 b) {
        long a_idx = stretched_z_curve_index(a.x, a.y, this->w);
        long b_idx = stretched_z_curve_index(b.x, b.y, this->w);
        return (a.z < b.z) || 
               (a.z == b.z && a_idx < b_idx);
    }
};

#endif