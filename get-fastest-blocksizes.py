#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd
import plot

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str)
parser.add_argument("-n", type=int, default=0)
parser.add_argument("-x", type=int, default=0)
parser.add_argument("-y", type=int, default=0)
parser.add_argument("-z", type=int, default=0)
parser.add_argument("--threads-prod", type=int, default=None)
parser.add_argument("-b", type=str, nargs="*", default=[])
args = parser.parse_args()

# read and rename columns
df=plot.read_data(args.i, dirty=True)

#filter data
df = df.loc[((df.loc[:, "size-x"] >= args.x) & (df.loc[:, "size-y"] >= args.y) & (df.loc[:, "size-z"] >= args.z))]
if args.threads_prod:
    df = df.loc[df.loc[:, "threads-prod"] == args.threads_prod]

# for each group, find fastest blocksize
grouped = df.groupby("benchmark")
vs = []
for grp, gdf in grouped:
 if args.b and grp.strip() not in args.b:
     continue
 gdf = gdf.sort_values(by=["kernel-median", "kernel-avg", "kernel-max", "kernel-min"])
 bs = "{0}x{1}x{2}".format(gdf.iloc[0].loc["threads-x"], gdf.iloc[0].loc["threads-y"], gdf.iloc[0].loc["threads-z"])
 vs.append(bs)
 print("{0}: {1}".format(grp, bs))
[sys.stdout.write("{0} ".format(x)) for x in set(vs)]
sys.stdout.write("\n")
