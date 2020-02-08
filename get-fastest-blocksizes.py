#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str)
parser.add_argument("-n", type=int, default=0)
parser.add_argument("-x", type=int, default=0)
parser.add_argument("-y", type=int, default=0)
parser.add_argument("-z", type=int, default=0)
parser.add_argument("-b", type=str, nargs="*", default=[])
args = parser.parse_args()

# read and rename columns
df=pd.read_csv(args.i, skiprows=2)
cs = df.columns.values
cs[0] = "bench"
cs[2] = "domain-x"
cs[3] = "domain-y"
cs[4] = "domain-z"
cs[8] = "threads-x"
cs[9] = "threads-y"
cs[10] = "threads-z"
df.columns = cs
df.columns = df.columns.str.strip()

#filter data
df = df.loc[((df.loc[:, "domain-x"] >= args.x) & (df.loc[:, "domain-y"] >= args.y) & (df.loc[:, "domain-z"] >= args.z))]

# for each group, find fastest blocksize
grouped = df.groupby("bench")
vs = []
for grp, gdf in grouped:
 if args.b and grp.strip() not in args.b:
     continue
 gdf = gdf.sort_values(by=["Median", "Average", "Maximum", "Minimum"])
 bs = "{0}x{1}x{2}".format(gdf.iloc[0].loc["threads-x"], gdf.iloc[0].loc["threads-y"], gdf.iloc[0].loc["threads-z"])
 vs.append(bs)
 print("{0}: {1}".format(grp, bs))
[sys.stdout.write("{0} ".format(x)) for x in set(vs)]
sys.stdout.write("\n")
