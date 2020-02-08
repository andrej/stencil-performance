#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONSTANTS
columns = ["benchmark", "precision",
           "size-x", "size-y", "size-z",
           "blocks-x", "blocks-y", "blocks-z",
           "threads-x", "threads-y", "threads-z",
           "kernel-avg", "kernel-median", "kernel-min", "kernel-max"]
column_titles = { "benchmark" : "Benchmark",
                  "precision" : "Precision",
                  "size-x" : "Domain Size (X)",
                  "size-y" : "Domain Size (Y)",
                  "size-z" : "Domain Size (Z)",
                  "blocks-x" : "Number of Blocks (X)",
                  "blocks-y" : "Number of Blocks (Y)",
                  "blocks-z" : "Number of Blocks (Z)",
                  "threads-x" : "Number of Threads (X)",
                  "threads-y" : "Number of Threads (Y)",
                  "threads-z" : "Number of Threads (Z)",
                  "kernel-avg" : "Average Kernel Runtime",
                  "kernel-median" : "Median Kernel Runtime",
                  "kernel-min" : "Minimal Kernel Runtime",
                  "kernel-max" : "Maximal Kernel Runtime",
                  "size-prod" : "Domain Size",
                  "blocks-prod" : "Number of Blocks",
                  "threads-prod" : "Number of Threads" }
column_units = { "kernel-avg" : "μs",
                 "kernel-median" : "μs",
                 "kernel-min" : "μs",
                 "kernel-max" : "μs" }
bench_markers = {"naive" : "o",
                 "idxvar" : "*",
                 "idxvar-kloop" : "v",
                 "idxvar-kloop-sliced" : "^",
                 "idxvar-shared" : "P",
                 "idxvar-no-chase" : "8",
                 "idxvar-kloop-no-chase" : "p",
                 "idxvar-kloop-sliced-no-chase" : "H",
                 "idxvar-shared-no-chase" : "x" }
bench_colors = { "regular-naive" : "C1",
                 "regular-idxvar" : "C2",
                 "regular-idxvar-kloop" : "C3",
                 "regular-idxvar-kloop-sliced" : "C4",
                 "regular-idxvar-shared" : "C5",
                 "unstr-naive" : "C1",
                 "unstr-idxvar" : "C2",
                 "unstr-idxvar-kloop" : "C3",
                 "unstr-idxvar-kloop-sliced" : "C4",
                 "unstr-idxvar-shared" : "C5", 
                 "unstr-idxvar-no-chase" : "C1",
                 "unstr-idxvar-kloop-no-chase" : "C2",
                 "unstr-idxvar-kloop-sliced-no-chase" : "C3",
                 "unstr-idxvar-shared-no-chase" : "C4" }
bench_linestyles = { "regular" : "--",
                     "unstr" : ":" }

"""
Filter input data to only include measurements in given range of blocksizes, only
for the given benchmark names, etc.
"""
def filter_data(data, blocksize_min=None, blocksize_max=None,
                domainsize_min=None, domainsize_max=None,
                precision=None):
    if blocksize_min or blocksize_max or domainsize_min or domainsize_max:
        supmin_block = data["threads-prod"] >= blocksize_min if blocksize_min else True
        submax_block = data["threads-prod"] <= blocksize_max if blocksize_max else True
        supmin_domain = data["size-prod"] >= domainsize_min if domainsize_min else True
        submax_domain = data["size-prod"] <= domainsize_max if domainsize_max else True
        data = data[np.logical_and(np.logical_and(supmin_block, submax_block),
                                   np.logical_and(supmin_domain, submax_domain))]
    if precision:
        data = data[data["precision"] == precision]
    return data

"""
Given benchmark name, return style properties associated with that benchmark.
"""
def bench_name(bench):
    if isinstance(bench, str):
        return bench
    elif isinstance(bench, tuple):
        return bench[0]
    else:
        return str(bench)
def bench_marker(bench, default="o"):
    bench = bench_name(bench)
    return ([v for k, v in bench_markers.items() if bench.endswith(k)] + [default])[0]
def bench_color(bench, default="C6"):
    bench = bench_name(bench)
    return ([v for k, v in bench_colors.items() if bench.endswith(k)] + [default])[0]
def bench_linestyle(bench, default="-"):
    bench = bench_name(bench)
    return ([v for k, v in bench_linestyles.items() if k in bench] + [default])[0]
def bench_label(bench):
    if(isinstance(bench, str)):
        return bench
    return ", ".join(bench)

"""
"""
def plot_agg(grouped, ax, ys=[], bar=False, agg=np.min):
    data = []
    xs = []
    labels = []
    for grp, df in grouped:
        data.append(np.reshape(df.loc[:, ys].to_numpy(), -1))
        labels.append(bench_label(grp))
    xs = np.arange(len(data))
    ys = [agg(ys) for ys in data]
    colors = [bench_color(b) for b, x in grouped]
    markers = [bench_marker(b) for b, x in grouped]
    ax.grid(axis="x")
    if bar:
        #bars
        ax.bar(xs, ys, color=colors)
    else:
        # box & whiskers
        ax.boxplot(data, positions=xs, whis="range", whiskerprops={"linestyle":"--"}, medianprops={"linewidth":2.0, "color":"black"})
    # draw the markers too
    for x, y, c, m in zip(xs, ys, colors, markers):
        y = y/2.0 if bar else y
        c = "white" if bar else c
        ax.plot([x], [y], color=c, marker=m, zorder=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, horizontalalignment="right")

"""
Scatter plot of block sizes (products) to execution time.
"""
def plot_sequence(grouped, ax, x="threads-prod", y="kernel-median"):
    ax.grid(axis="x")
    for grp, df in grouped:
        blocks_times = df.loc[:, [x, y]].to_numpy()
        blocks_times = blocks_times[np.argsort(blocks_times[:,0], axis=0)] 
        ax.plot(blocks_times[:,0], blocks_times[:,1], 
                linestyle=bench_linestyle(grp),
                marker=bench_marker(grp),
                color=bench_color(grp),
                label=bench_label(grp))
    # put legend outside of graph
    box = ax.get_position()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    

"""
Given a data dictionary, return limits based on min/max values.
"""
def get_limits(data, col="kernel-median",
               outliers_min=0, outliers_max=0,
               scale_min=None, scale_max=None, padding=0.1):
    vals = list(data.loc[:,col])
    vals.sort()
    for i in range(0, min(outliers_min, max(0, len(vals)-1))):
        del vals[0]
    for i in range(0, min(outliers_max, max(0, len(vals)-1))):
        del vals[-1]
    set_min_diff = 0 if scale_min == None else abs(vals[0]-scale_min)
    set_max_diff = 0 if scale_max == None else abs(vals[-1]-scale_max)
    ymin = vals[0]-set_max_diff if scale_min == None else scale_min
    ymax = vals[-1]+set_min_diff if scale_max == None else scale_max
    spread = abs(ymax-ymin)
    return ymin - padding*spread, ymax + padding*spread

"""
Relate all the data to some base line.
Operates on the given data array *in place*!
"""
def baseline(data, grouped, base_bench, y="kernel-median", ys=[]):
    ys = ys + [y]
    groups = grouped.groups
    base_groups = [x for x in groups if x[0] == base_bench or x == base_bench]
    new_ys = data.loc[:, ys].copy()
    for group in groups:
        base = np.NaN
        for other_group in base_groups:
            if isinstance(other_group, str) or other_group[1:] == group[1:]:
                # matching base group
                other_indices = groups[other_group]
                base = data.loc[other_indices, ys].min(axis=0)
        indices = groups[group]
        new_ys.loc[indices, :] /= base
    data.loc[:, ys] = new_ys

"""
Used to aggregate together all data which will be plotted together into one data point.
"""
def aggregate(data, over=["benchmark"], fun="min", y="kernel-median"):
    grps = data.groupby(over)
    if fun == "min":
        return data.loc[grps[y].idxmin()]
    elif fun == "max":
        return data.loc[grps[y].idxmax()]
    else:
        raise ValueError()

"""
Main
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=argparse.FileType("r"),
                        default=sys.stdin)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-b", "--benchmark", nargs="*", type=str, default=None)
    parser.add_argument("--exclude", nargs="*", type=str, default=None)
    parser.add_argument("-g", "--groupby", nargs="*", default=["precision"]) 
        # groups additionally to benchmark & precision
    parser.add_argument("-x", type=str, nargs="?", default="size-z")
    parser.add_argument("-y", type=str, default="kernel-median")
    parser.add_argument("-p", "--plot", nargs="*", type=str, default=["box", "line"])

    parser.add_argument("--plot-size", nargs=2, type=float, default=plt.rcParams.get("figure.figsize") ) #[11.69, 8.27])
    parser.add_argument("--scale-min", nargs="?", type=float, default=None)
    parser.add_argument("--scale-max", nargs="?", type=float, default=None)
    parser.add_argument("--logscale", action="store_true", default=False)
    parser.add_argument("--outliers-max", type=int, default=0) # disregard N outliers in axis scale computation
    parser.add_argument("--outliers-min", type=int, default=0)
    parser.add_argument("--precision", type=str, nargs="?", default="double")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--agg", type=str, default="min") 
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--threads-min", type=int, default=None)
    parser.add_argument("--threads-max", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--size-min", type=int, default=None)
    parser.add_argument("--size-max", type=int, default=None)
    parser.add_argument("--show", action="store_true", default=False)
    args = parser.parse_args()
    out = args.output
    if out == None:
        out = os.path.join(os.path.dirname(args.input.name),
                           (os.path.basename(args.input.name).rsplit(".csv", 1)[0]
                            + ".pdf"))
    
    if args.threads:
        args.threads_min = args.threads_max = args.threads
        if "threads-prod" in args.groupby:
            args.groupby.remove("threads-prod")
    if args.size:
        args.size_min = args.size_max = args.size
        if "size-prod" in args.groupby:
            args.groupby.remove("size-prod")
    if args.precision and "precision" in args.groupby:
        args.groupby.remove("precision")
    if not args.scale_min and not args.baseline:
        args.scale_min = 0

    # READ and filter data
    data = pd.read_csv(args.input, skiprows=3, header=None)
    runs_columns = ["run-{0}".format(x) for x in range(0, len(data.columns)-len(columns))]
    data.set_axis(columns + runs_columns, axis=1, inplace=True)

    data.rename(columns=dict(enumerate(columns)),inplace=True)
    data["benchmark"] = data["benchmark"].str.strip()
    data["precision"] = data["precision"].str.strip()

    # FILTER unwanted benchmarks
    if args.benchmark:
        data.drop(data[data["benchmark"].apply(lambda x: x not in args.benchmark)].index,
                  inplace=True)
    if args.exclude:
        data.drop(data[data["benchmark"].apply(lambda x: x in args.exclude)].index,
                  inplace=True)

    # EXTEND data by products
    data.insert(len(data.columns), "size-prod", 
                np.prod(data[["size-x", "size-y", "size-z"]], axis=1))
    data.insert(len(data.columns), "blocks-prod",
                np.prod(data[["blocks-x", "blocks-y", "blocks-z"]], axis=1))
    data.insert(len(data.columns), "threads-prod",
                np.prod(data[["threads-x", "threads-y", "threads-z"]], axis=1))
    data = filter_data(data, args.threads_min, args.threads_max, 
                       args.size_min, args.size_max,
                       args.precision)
    unit = ""
    if args.y in column_units:
        unit = column_units[args.y]

    group_cols = ["benchmark"] + args.groupby

    # BASELINE scale
    if args.baseline:
        grouped_line = data.groupby(group_cols + ([args.x] if args.x else []))
        baseline(data, grouped_line, args.baseline, y=args.y, ys=runs_columns)
        unit = "relative to baseline"

    data = aggregate(data, group_cols + ([args.x] if args.x else []), args.agg, args.y)
    grouped = data.groupby(group_cols) # contains multiple data points (one for each X) p g.

    # PLOT SCALE 
    ymin, ymax = get_limits(data, 
                            outliers_min=args.outliers_min,
                            outliers_max=args.outliers_max,
                            scale_min=args.scale_min,
                            scale_max=args.scale_max)

    # PLOT LAYOUT
    n_rows = len(args.plot)
    f = plt.gcf()
    f.set_size_inches(args.plot_size[0], args.plot_size[1])
    f.subplots_adjust(hspace=0.6)
    plt.style.use("seaborn")
    for row_i in range(0, n_rows):
        plot = args.plot[row_i]
        ax = plt.subplot(n_rows, 1, row_i+1)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_ylabel(column_titles[args.y] + " [" + unit + "]")
        if args.logscale:
            ax.set_yscale("log")

        # ACTUAL PLOTTING
        if plot == "box" or plot == "bar":
            ax.set_ylim(ymin=ymin, ymax=ymax)
            if args.logscale:
                ax.set_yscale("log")
            aggfun = np.median
            #if args.agg == "min":
            #    aggfun = np.min
            #elif args.agg == "max":
            #    aggfun = np.max
            plot_agg(grouped, ax, ys=runs_columns, bar=plot=="bar", agg=aggfun)

        elif plot == "line":
            if not args.x:
                continue
            xticks = data.loc[:, args.x].unique()
            ax.set_xticks(xticks)
            plot_sequence(grouped, ax, x=args.x, y=args.y)
            ax.set_xlabel(column_titles[args.x])

    plt.savefig(out, bbox_inches="tight")
    if(args.show):
        plt.tight_layout()
        plt.show()

if(__name__ == "__main__"):
	main()
