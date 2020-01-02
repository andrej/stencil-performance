#!/usr/bin/env python3
import sys
import argparse
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Filter input data to only include measurements in given range of blocksizes, only
for the given benchmark names, etc.
"""
def filter_data(data, blocksize_min=None, blocksize_max=None,
                domainsize_min=None, domainsize_max=None,
                precision=None):
    if not blocksize_min and not blocksize_max and not domainsize_min and not domainsize_max:
        return data
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
Plots average, minimum and maximum run times (both kernel and total) aggregated
for each benchmark type.
"""
def plot_avgminmax(grouped, ax, y="kernel-median", 
                   bar_size=1, group_spacing=2, bar_spacing=0):
    avgs, mins, maxs = [], [], []
    xs = []
    labels = []
    for grp, df in grouped:
        avgs.append(np.average(df.loc[:,y].to_numpy()))
        mins.append(np.min(df.loc[:,y].to_numpy()))
        maxs.append(np.max(df.loc[:,y].to_numpy()))
        labels.append(grp)
    x = 0
    for i in range(0, len(labels)):
        xs.append(x)
        x += group_spacing + 2*bar_spacing + 3*bar_size
    xs = np.array(xs)
    offs = 0
    ax.bar(xs, avgs, bar_size, label="average")
    offs += bar_size+bar_spacing
    ax.bar(xs+offs, mins, bar_size, label="fastest")
    offs += bar_size+bar_spacing
    ax.bar(xs+offs, maxs, bar_size, label="slowest")
    xtickstart = bar_spacing + bar_size #(3.0*bar_size + 2*bar_spacing) / 2.0
    xtickstep = xs[1] if len(xs) > 1 else 0
    xticks = [xtickstart+xtickstep*i for i in range(0, len(labels))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=15, horizontalalignment="right")
    ax.legend()

"""
Scatter plot of block sizes (products) to execution time.
"""
def plot_scatter_blocksize(grouped, ax, x="threads-prod", y="kernel-median"):
    for grp, df in grouped:
        blocks_times = df.loc[:, [x, y]].to_numpy()
        blocks_times = blocks_times[np.argsort(blocks_times[:,0], axis=0)] 
        ax.plot(blocks_times[:,0], blocks_times[:,1], linestyle=":", marker="o", label=grp)
    # put legend outside of graph
    box = ax.get_position()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1),
              ncol=3, fancybox=True, shadow=True)

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
def baseline(data, grouped, base_bench, y="kernel-avg"):
    groups = grouped.groups
    base_groups = [x for x in groups if x[0] == base_bench or x == base_bench]
    new_y = data.loc[:, y].copy()
    for group in groups:
        base = np.NaN
        for other_group in base_groups:
            if isinstance(other_group, str) or other_group[1:] == group[1:]:
                # matching base group
                other_indices = groups[other_group]
                base = data.loc[other_indices, y].min()
        indices = groups[group]
        new_y[indices] /= base
    data.loc[:, y] = new_y


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
    parser.add_argument("-g", "--groupby", nargs="*", default=["precision", "size-prod"]) 
        # groups additionally to benchmark & precision
    parser.add_argument("-x", type=str, default="threads-prod")
    parser.add_argument("-y", type=str, default="kernel-median")
    parser.add_argument("-p", "--plot", nargs="*", type=str, default=["bar", "line"])

    parser.add_argument("--plot-size", nargs=2, type=float, default=[11.69, 8.27])
    parser.add_argument("--scale-min", nargs="?", type=float, default=None)
    parser.add_argument("--scale-max", nargs="?", type=float, default=None)
    parser.add_argument("--logscale", action="store_true", default=False)
    parser.add_argument("--outliers-max", type=int, default=0) # disregard N outliers in axis scale computation
    parser.add_argument("--outliers-min", type=int, default=0)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--baseline-per-x", action="store_true", default=False)
    parser.add_argument("--agg", type=str, default=None) # if there are multiple entries for the same block size, fold them together (min, max, avg, none -> show all)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--threads-min", type=int, default=None)
    parser.add_argument("--threads-max", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--size-min", type=int, default=None)
    parser.add_argument("--size-max", type=int, default=None)
    args = parser.parse_args()
    out = args.output
    if out == None:
        out = os.path.join(os.path.dirname(args.input.name),
                           (os.path.basename(args.input.name).rsplit(".csv", 1)[0]
                            + ".pdf"))
    
    if args.threads:
        args.threads_min = args.threads_max = args.threads
    if args.size:
        args.size_min = args.size_max = args.size

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

    # READ and filter data
    data = pd.read_csv(args.input, skiprows=3, names=columns, header=None)
    data["benchmark"] = data["benchmark"].str.strip()
    data["precision"] = data["precision"].str.strip()

    # FILTER unwanted benchmarks
    if args.benchmark:
        data.drop(data[data["benchmark"].apply(lambda x: x not in args.benchmark)].index,
                  inplace=True)
    if args.exclude:
        data.drop(data[data["benchmark"].apply(lambda x: x in args.exclude)].index,
                  inplace=True)

    # EXTEND AND GROUP data
    data.insert(len(data.columns), "size-prod", 
                np.prod(data[["size-x", "size-y", "size-z"]], axis=1))
    data.insert(len(data.columns), "blocks-prod",
                np.prod(data[["blocks-x", "blocks-y", "blocks-z"]], axis=1))
    data.insert(len(data.columns), "threads-prod",
                np.prod(data[["threads-x", "threads-y", "threads-z"]], axis=1))
    data = filter_data(data, args.threads_min, args.threads_max, 
                       args.size_min, args.size_max,
                       args.precision)


    # AGGREGATE data so there is only one value for each X in line plot
    group_cols = ["benchmark"] + args.groupby
    x_grouped = data.groupby(group_cols + [args.x])
    if not args.agg or args.agg == "min":
        data = data.loc[x_grouped[args.y].idxmin()]
    elif args.agg == "max":
        data = data.loc[x_grouped[args.y].idxmax()]
    else:
        # TODO median?
        raise ValueError()

    # GROUP data; each group will be one line / entry in bar plot
    grouped = data.groupby(group_cols)
    x_grouped = data.groupby(group_cols + [args.x])
    
    # BASELINE scale
    if args.baseline:
        baseline(data, grouped if not args.baseline_per_x
                       else x_grouped,
                 args.baseline, y=args.y)


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
    for row_i in range(0, n_rows):

        plot = args.plot[row_i]
        ax = plt.subplot(n_rows, 1, row_i+1)
        xticks = data.loc[:, args.x].unique()
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xticks(xticks)
        ax.set_ylabel(column_titles[args.y])
        if args.logscale:
            ax.set_yscale("log")

        # ACTUAL PLOTTING
        if plot == "bar":
            ax.grid(axis="y")
            ax.set_ylim(ymin=ymin, ymax=ymax)
            ax.set_xticks(xticks)
            if args.logscale:
                ax.set_yscale("log")
            plot_avgminmax(grouped, ax, y=args.y)

        elif plot == "line":
            ax.grid()
            plot_scatter_blocksize(grouped, ax, x=args.x, y=args.y)
            ax.set_xlabel(column_titles[args.x])

    plt.savefig(out)

if(__name__ == "__main__"):
	main()
