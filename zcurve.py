#!/usr/bin/env python3
import sys
import argparse
from matplotlib import pyplot as plt

class Zcurve:
    
    def __init__(self):
        self.coordinates = []
        self.indices = []
        self.z_stride = 1

    def coordinate(self, idx):
        pass

    def index(self, x, y, z, w=4): 
        res = 0
        # The bits of x and y are intertwined, with bits from X in lower
        # position and Y in higher position. The lowest w bits of x are
        # NOT intertwined and instead run consecutively up. This allows
        # for a wider Z-curve. Then, the higher bits of X (> w) are 
        # intertwined with ALL bits of Y.
        
        x_lower = x & ((1 << w) - 1)
        x_upper = x >> w
        
        # in the following, treat upper bits as all of X
        x = x_upper
        
        # get most significant bit index i -> index of max(MSB(x), MSB(y))
        # i = 0 <=> both numbers are 0
        # i =  0 <=> LSB is 1 (0-based indexing of bits)
        bitlen = 0
        while((x >> bitlen) or (y >> bitlen)):
            bitlen += 1
                
        # spread out both X and Y coordinates to have zero bits in all odd positions
        x_stretch = 0
        y_stretch = 0
        for j in range(0, bitlen):
            x_stretch |= ((x>>j) & 1) << 2*j
            y_stretch |= ((y>>j) & 1) << 2*j
                
        # intertwine all bits, x in even, y in odd position
        res = x_stretch | (y_stretch << 1)
        
        # add back lower bits of X which are not intertwined
        res = res << w | x_lower
        
        return res
            
        
        """res = 0
        i = 0
        lower_mask = 2**w-1
        x_lower = x & lower_mask
        x >>= w
        while x > 0 or y > 0:
            res <<= 2
            res |= ((x & 0b1) << 0) | ((y & 0b1) << 1)
            x >>= 1
            y >>= 1
            i += 1
        out = 0
        while i > 0:
            out <<= 2
            out |= res & 0b11
            res >>= 2
            i -= 1
        res <<= w
        res |= x_lower
        return (out) + z*self.z_stride"""

    @classmethod
    def create(cls, sx, sy, z, w=4):
        obj = cls()
        obj.z_stride = sx*sy
        for x in range(0, sx):
            for y in range(0, sy):
                obj.coordinates.append((x, y, 0))
                obj.indices.append(obj.index(x, y, 0, w))
        return obj

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=int, default=32)
    parser.add_argument("-y", type=int, default=32)
    parser.add_argument("-w", type=int, default=4)
    args = parser.parse_args()
    curve = Zcurve.create(args.x, args.y, 1, args.w)
    points = list(zip(curve.indices, curve.coordinates))
    points.sort()
    for idx, c in points:
        print("{0:4}, {1:4}, {2:4}  -> {3:4}".format(c[0], c[1], c[2], idx))
    plt.plot([c[0] for i, c in points], [-c[1] for i, c in points])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
