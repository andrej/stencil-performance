#!/usr/bin/env python3

import sys
import argparse

def gen_blocksizes_for_prod(prod, initx=1, inity=1, initz=1):
    out = []
    assert initx > 0 and inity > 0 and initz > 0
    x = initx
    while x <= prod:
        y = inity
        while x * y <= prod:
            z = int(prod / (x*y))
            if z < initz:
                continue
            out.append((x, y, z))
            y *= 2
        x *= 2
    return out

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prod", type=int, default=512)
    parser.add_argument("-x", type=int, nargs="?", default=1) # min value
    parser.add_argument("-y", type=int, nargs="?", default=1)
    parser.add_argument("-z", type=int, nargs="?", default=1)
    parser.add_argument("-u", "--upto", action="store_true", default=False)
    args = parser.parse_args()
    res = []
    if(args.upto):
        p = 1
        while p <= args.prod:
            res += gen_blocksizes_for_prod(p, args.x, args.y, args.z)
            p *= 2
    else:
        res = gen_blocksizes_for_prod(args.prod, args.x, args.y, args.z)
    strs = map(lambda v: "{0}x{1}x{2}".format(v[0], v[1], v[2]), res)
    print(" ".join(strs))

if(__name__ == "__main__"):
    main(sys.argv)
