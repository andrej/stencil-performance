#!/usr/bin/env python3
import sys
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt

"""
Parses the output produced by the gridbenchmark tool.
Expects a CSV-like format, i.e. comma-delimited lines of values. After an
(optional) header row, there must follow rows of 13 columns each, first column
contianing the benchmark name (string), and the next 12 containing float values.

Returns a dict of numpy arrays { benchmark name -> numpy array }
"""
def parse_input(f, name_strip=""):
    ret = {} # map benchmark name -> benchmark values
    # benchmark values = 2D array, row-major, of all the measurements (numpy)
    for line in f:
        cells = line.split(",")
        if len(cells) != 13:
            # might be header row or some other invalid format
            continue
        benchmark_name = None
        benchmark_values = []
        for i, cell in enumerate(cells):
            cell = cell.strip()
            v = None
            if i == 0:
                benchmark_name = cell.strip(name_strip)
                continue
            else:
                try:
                    if 1 <= i < 7:
                        v = int(cell)
                    else:
                        v = float(cell)
                except ValueError:
                    continue
            benchmark_values.append(v)
        if not benchmark_name or len(benchmark_values) != 12:
            # ignore invald rows
            continue
        if benchmark_name not in ret.keys():
            ret[benchmark_name] = []
        ret[benchmark_name].append(benchmark_values)
    for benchmark in ret:
        ret[benchmark] = np.array(ret[benchmark])
    return ret

"""
Plots average, minimum and maximum run times (both kernel and total) aggregated
for each benchmark type.
"""
def plot_avgminmax(data, ax, v_col=6, bar_size=1, group_spacing=2, bar_spacing=0):
    avgs, mins, maxs = [], [], []
    xs = []
    labels = [] # six values per label
    for bench in data:
        v = data[bench]
        avgs.append(np.average(v[:,v_col]))
        mins.append(np.min(v[:,v_col]))
        maxs.append(np.max(v[:,v_col]))
        labels.append(bench)
    x = 0
    for i in range(0, len(avgs)):
        xs.append(x)
        x += group_spacing + 2*bar_spacing + 3*bar_size
    xs = np.array(xs)
    offs = 0
    ax.bar(xs, avgs, bar_size, label="all block sizes")
    offs += bar_size+bar_spacing
    ax.bar(xs+offs, mins, bar_size, label="fastest block size")
    offs += bar_size+bar_spacing
    ax.bar(xs+offs, maxs, bar_size, label="slowest block size")
    xtickstart = bar_spacing + bar_size #(3.0*bar_size + 2*bar_spacing) / 2.0
    xtickstep = xs[1]
    xticks = [xtickstart+xtickstep*i for i in range(0, len(labels))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=12.5)
    ax.legend()
    ax.set_ylabel("Execution time [s]")

"""
Scatter plot of block sizes (products) to execution time.
"""
def plot_scatter_blocksize(data, ax, f_x=lambda v:v[3]*v[4]*v[5], f_y=lambda v:v[9]):
    # first, create 2D array of product of blocks -> execution time
    # this allows for sorting by number of blocks later
    # array needs to be sorted so that connecting lines look right
    blocks_times = {}
    for bench in data:
        dat = data[bench]
        blocks_times = []
        for v in dat:
            blocks_times.append([f_x(v), f_y(v)])
        blocks_times = np.array(blocks_times)
        #blocks_times = np.sort(blocks_times, 0) # this is wrong (breaks assoc)
        ax.plot(blocks_times[:,0], blocks_times[:,1], linestyle=":", marker="o", label=bench)
    ax.legend()

"""
Given a data dictionary, return limits based on min/max values.
"""
def get_limits(data, n=10, col=9, outliers_min=0, outliers_max=0):
    vals = set()
    for b in data:
        bench = data[b]
        vals.update( bench[:,col] )
    vals = list(vals)
    vals.sort()
    for i in range(0, min(outliers_min, max(0, len(vals)-1))):
        del vals[0]
    for i in range(0, min(outliers_max, max(0, len(vals)-1))):
        del vals[-1]
    return vals[0], vals[-1]

"""
Return X ticks for the given data, one tick for each data point.
"""
def get_xticks(data, f_x=lambda v:v[3]*v[4]*v[5]):
    ticks = set()
    for b in data:
        bench = data[b]
        for row in bench:
            v = f_x(row)
            ticks.add(v)
    ticks = list(ticks)
    ticks.sort()
    return ticks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=argparse.FileType("r"),
                        default=sys.stdin)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-x", "--exclude", nargs="*", type=str, default=None)
    parser.add_argument("-l", "--left", nargs="*", type=str, default=None)
    parser.add_argument("-r", "--right", nargs="*", type=str, default=None)
    parser.add_argument("-s", "--strip", nargs="?", type=str, default=None)
    parser.add_argument("--logscale", action="store_true", default=False)
    parser.add_argument("--outliers-max", type=int, default=0) # disregard N outliers in axis scale computation
    parser.add_argument("--outliers-min", type=int, default=0)
    args = parser.parse_args()
    out = args.output
    if out == None:
        out = os.path.join(os.path.dirname(args.input.name),
                           (os.path.basename(args.input.name).rsplit(".csv", 1)[0]
                            + ".pdf"))
    data = parse_input(args.input, args.strip)
    if args.exclude:
        for excl in args.exclude:
            if excl in data:
                del data[excl]

    data_left = {}
    data_right = {}
    if args.left:
        for incl in args.left:
            if incl in data:
                data_left[incl] = data[incl]
    if args.right:
        for incl in args.right:
            if incl in data:
                data_right[incl] = data[incl]

    numcols = 2
    if not data_left and not data_right:
        data_left = data
        numcols = 1

    # same scale for left and right graph
    ymin, ymax = get_limits({**data_left, **data_right},
            outliers_min=args.outliers_min,
            outliers_max=args.outliers_max)
    xticks = get_xticks({**data_left, **data_right})

    f = plt.gcf()
    plt.subplots_adjust(hspace=0.4)
    f.set_size_inches(11.69, 8.27)

    for col in range(0, numcols):
        d = data_left if col == 0 else data_right

        ax = plt.subplot(2, numcols, col+1)
        ax.set_title("Kernel-only execution time (all block sizes)")
        ax.grid(axis="y")
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xticks(xticks)
        if args.logscale:
            ax.set_yscale("log")
        plot_avgminmax(d, ax, v_col=9)

        ax = plt.subplot(2, numcols, numcols+col+1)
        ax.set_title("Average kernel execution time")
        ax.grid()
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xticks(xticks)
        if args.logscale:
            ax.set_yscale("log")
        plot_scatter_blocksize(d, ax)
        ax.set_xlabel("Total number of threads (X*Y*Z)")
        ax.set_ylabel("Execution time [s]")

    plt.savefig(out)

if(__name__ == "__main__"):
	main()
