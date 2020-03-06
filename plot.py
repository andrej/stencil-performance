#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import re
import argparse
import collections
import numbers
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONSTANTS
columns = [ "benchmark", "precision",
            "size-x", "size-y", "size-z",
            "blocks-x", "blocks-y", "blocks-z",
            "threads-x", "threads-y", "threads-z",
            "kernel-avg", "kernel-median", "kernel-min", "kernel-max" ]
column_types = { "benchmark" : str,
            "precision" : str,
            "size-x" : int, # "Int64",
            "size-y" : int, # "Int64",
            "size-z" : int, # "Int64",
            "blocks-x" : int, # "Int64",
            "blocks-y" : int, # "Int64",
            "blocks-z" : int, # "Int64",
            "threads-x" : int, # "Int64",
            "threads-y" : int, # "Int64",
            "threads-z" : int, # "Int64",
            "kernel-avg" : float,
            "kernel-median" : float,
            "kernel-min" : float,
            "kernel-max" : float }
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
                  "kernel-avg" : "Average Runtime",
                  "kernel-median" : "Median Runtime",
                  "kernel-min" : "Minimal Runtime",
                  "kernel-max" : "Maximal Runtime",
                  "size-prod" : "Domain Size",
                  "blocks-prod" : "Number of Blocks",
                  "threads-prod" : "Number of Threads" }
column_units = { "kernel-avg" : "μs",
                 "kernel-median" : "μs",
                 "kernel-min" : "μs",
                 "kernel-max" : "μs" }

variants = ["naive", "idxvar", "idxvar-shared", "idxvar-kloop", "idxvar-kloop-sliced"]
stencils = ["hdiff", "laplap", "fastwaves"]
storage_types = ["", "comp", "no-chase", "z-curves", "no-chase-z-curves", "comp-no-chase", "comp-z-curves", "comp-no-chase-z-curves"]

variant_labels = {"naive" : "naive",
                  "idxvar" : "index variables",
                  "idxvar-kloop" : "z-loop",
                  "idxvar-kloop-sliced" : "sliced z-loop",
                  "idxvar-shared" : "shared" }

stencil_colors = {"hdiff-regular" : "C1",
                  "hdiff-unstr" : "C2",
                  "laplap-regular" : "C3",
                  "laplap-unstr" : "C4",
                  "fastwaves-regular" : "C5",
                  "fastwaves-unstr" : "C6"}
stencil_markers = {"hdiff-regular" : "o",
                  "hdiff-unstr" : "*",
                  "laplap-regular" : "v",
                  "laplap-unstr" : "^",
                  "fastwaves-regular" : "P",
                  "fastwaves-unstr" : "8"}

storage_colors = {"" : "C1",
                  "comp" : "C2",
                  "no-chase" : "C3",
                  "z-curves" : "C4",
                  "no-chase-z-curves" : "C5",
                  "comp-no-chase" : "C6",
                  "comp-z-curves" : "grey",
                  "comp-no-chase-z-curves" : "darkorange",
                  "regular" : "black"}
storage_markers = {"" : "o",
                   "comp" : "*",
                   "no-chase" : "v",
                   "z-curves" : "^",
                   "no-chase-z-curves" : "P",
                   "comp-no-chase" : "8",
                   "comp-z-curves" : "p",
                   "comp-no-chase-z-curves" : "H"}

variant_colors = {"naive" : "C1",
                  "idxvar" : "C2",
                  "idxvar-kloop" : "C3",
                  "idxvar-kloop-sliced" : "C4",
                  "idxvar-shared" : "C5",
                  "regular" : "black"}
variant_markers = {"naive" : "o",
                   "idxvar" : "*",
                   "idxvar-kloop" : "v",
                   "idxvar-kloop-sliced" : "^",
                   "idxvar-shared" : "P"}

bench_markers = variant_markers
bench_colors = variant_colors
bench_linestyles = { "regular" : "-",
                     "unstr" : ":" }

def float_or_nan(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")

def read_data(path, dirty=False):
    # parse CSV
    data = pd.read_csv(path, skiprows=3, header=None, error_bad_lines=False)
    
    # rename columns as in global variables above
    runs_columns = ["run-{0}".format(x) for x in range(0, len(data.columns)-len(columns))]
    data.set_axis(columns + runs_columns, axis=1, inplace=True)
    
    # type conversion
    string_columns = [x for x in columns if column_types[x] in [str]]
    data[string_columns] = data[string_columns].apply(lambda s: s.str.strip(), axis=1)
    if dirty:
        numeric_columns = [x for x in columns if column_types[x] in ["Int64", int, float]] + runs_columns
        data[numeric_columns] = data[numeric_columns].applymap(lambda x: float_or_nan(x))
        data = data.dropna()
    data = data.astype(column_types)
    
    # SORT alphabetically by benchmark
    data.sort_values(by="benchmark", inplace=True)
    
    # add products
    data.insert(len(data.columns), "size-prod", 
                np.prod(data[["size-x", "size-y", "size-z"]], axis=1))
    data.insert(len(data.columns), "blocks-prod",
                np.prod(data[["blocks-x", "blocks-y", "blocks-z"]], axis=1))
    data.insert(len(data.columns), "threads-prod",
                np.prod(data[["threads-x", "threads-y", "threads-z"]], axis=1))
    
    return data

def filter_benchs(data, args):
    if args.benchmark:
        data.drop(data[data["benchmark"].apply(lambda x: [True for y in args.benchmark if re.search(y, x)] == [])].index,
                  inplace=True)
    if args.exclude:
        data.drop(data[data["benchmark"].apply(lambda x: x in args.exclude)].index,
                  inplace=True)
    if args.stencil:
        if np.any([s not in stencils for s in args.stencil]):
            raise ValueError("invalid input for stencils")
        data.drop(data[data["benchmark"].apply(lambda x: longest_match(stencils, x) not in args.stencil)].index, inplace=True)
    if args.variant:
        if np.any([v not in variants for v in args.variant]):
            raise ValueError("invalid input for variant")
        data.drop(data[data["benchmark"].apply(lambda x: longest_match(variants, x) not in args.variant)].index, inplace=True)
    if args.comp != "both":
        if args.comp not in ["comp", "no-comp"]:
            raise ValueError("invalid input for comp")
        data.drop(data[data["benchmark"].apply(lambda x: "regular" not in x and "comp" not in x if args.comp == "comp" else "comp" in x)].index, inplace=True)
    if args.chase != "both":
        if args.chase not in ["chase", "no-chase"]:
            raise ValueError("invalid input for chase")
        data.drop(data[data["benchmark"].apply(lambda x: "regular" not in x and "no-chase" in x if args.chase == "chase" else "no-chase" not in x)].index, inplace=True)
    if args.z_curves != "both":
        if args.z_curves not in ["z-curves", "no-z-curves"]:
            raise ValueError("invalid input for z-curves")
        data.drop(data[data["benchmark"].apply(lambda x: "regular" not in x and "z-curves" not in x if args.z_curves == "z-curves" else "z-curves" in x)].index, inplace=True)
    

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
Return pretty print name for benchmark
Use all_benchs to erradicate common prefixes (i.e. when only laplap is plotted, don't print "laplap" for each one)
"""
def bench_label(bench, all_benchs=[]):
    if(isinstance(bench, str)):
        return bench_pretty_label(bench, all_benchs)
    return ", ".join([bench_pretty_label(b, all_benchs) for b in bench])
    
def bench_pretty_label(bench, all_benchs=[]):
    regular = False
    stencil = None
    variant = None
    chasing = True
    z_curves = False
    compressed = False
    
    regular = "regular" in bench
    has_both_unstructured = np.any(["regular" in b for b in all_benchs]) and np.any(["unstr" in b for b in all_benchs])
    stencil = longest_match(stencils, bench)
    has_other_stencils = np.any([not longest_match(stencils, b) == stencil for b in all_benchs])
    variant = longest_match(variants, bench)
    has_other_variants = np.any([not longest_match(variants, b) == variant for b in all_benchs])
    chasing = "no-chase" not in bench
    has_both_chasing = np.any(["no-chase" in b for b in all_benchs]) and np.any(["no-chase" not in b for b in all_benchs])
    z_curves = "z-curves" in bench
    has_both_layouts = np.any(["z-curves" in b for b in all_benchs]) and np.any(["z-curves" not in b for b in all_benchs])
    compressed = "comp" in bench
    has_both_compressed = np.any(["comp" in b for b in all_benchs]) and np.any(["comp" not in b for b in all_benchs])
    
    out = []
    if regular:
        out.append("regular")
    #if has_both_unstructured:
        #if regular:
        #    out.append("regular")
        #else:
        #    out.append("unstructured")
    if has_other_stencils:
        out.append(stencil)
    if has_other_variants:
        out.append(variant_labels[variant] if variant in variant_labels else variant)
    if has_both_chasing and not regular:
        if(chasing):
            out.append("chasing")
        else:
            out.append("non-chasing")
    if has_both_layouts and not regular:
        if(z_curves):
            out.append("z-curves")
        else:
            out.append("row-major")
    if has_both_compressed and not regular:
        if(compressed):
            out.append("compressed")
        else:
            out.append("uncompressed")
    return ", ".join(out)
        


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
def longest_match(lst, key, default=""):
    haystack = sorted(lst, key=len, reverse=True)
    return ([v for v in haystack if v in key] + [default])[0]
def bench_marker(bench, default="o"):
    bench = bench_name(bench)
    return bench_markers[longest_match(bench_markers.keys(), bench, default)] # ([v for k, v in bench_markers.items() if bench.endswith(k)] + [default])[0]
def bench_color(bench, default="C6"):
    bench = bench_name(bench)
    return bench_colors[longest_match(bench_colors.keys(), bench, default)] #([v for k, v in bench_colors.items() if bench.endswith(k)] + [default])[0]
def bench_linestyle(bench, default="-"):
    bench = bench_name(bench)
    return ([v for k, v in bench_linestyles.items() if k in bench] + [default])[0]

"""
"""
def plot_agg(grouped, ax, ys=[], bar=False, agg=np.median):
    data = []
    xs = []
    labels = []
    for grp, df in grouped:
        if bar:
            data.append(agg(np.median(df.loc[:, ys], axis=1)))
        else:
            data.append(np.reshape(df.loc[:, ys].to_numpy(), -1))
        labels.append(grp)
    labels = [bench_label(b, labels) for b in labels]
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
    #for x, y, c, m in zip(xs, ys, colors, markers):
    #    y = y/2.0 if bar else y
    #    c = "white" if bar else c
    #    ax.plot([x], [y], color=c, marker=m, zorder=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, horizontalalignment="right")

"""
Scatter plot of block sizes (products) to execution time.
"""
def plot_sequence(grouped, ax, x="threads-prod", y="kernel-median"):
    ax.grid(axis="x")
    labels = np.unique([grp for grp, df in grouped])
    for grp, df in grouped:
        blocks_times = df.loc[:, [x, y]].to_numpy()
        blocks_times = blocks_times[np.argsort(blocks_times[:,0], axis=0)] 
        ax.plot(blocks_times[:,0], blocks_times[:,1], 
                linestyle=bench_linestyle(grp),
                marker=bench_marker(grp),
                color=bench_color(grp),
                label=bench_label(grp, labels))
    # put legend outside of graph
    box = ax.get_position()
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2)
    

"""
"""
def plot_grouped_bars(grouped, ax, ys=[], bench_groups=[], agg=np.median, special_group="regular-idxvar"):
    w = 1 # width of bars
    sp = 1 # spacing between bar groups
    if not bench_groups or bench_groups == ["storage"]:
        bench_groups = storage_types
    if bench_groups == ["variant"]:
        bench_groups = variants
    grps = {}
    if special_group:
        grps[special_group] = [(b, special_group, df) for b, df in grouped if special_group in b]
    for x in bench_groups:
        grps[x] = [ (b, b.replace(x, "").replace("--", "-").strip("-"), df)
                    for b, df in grouped if longest_match(bench_groups, b) == x and not special_group in b ]
    # sort bars inside groups by new shortened name for consistency
    for grp in grps:
        grps[grp].sort(key=lambda x:x[1])
    #grps = { x : v for x in grps if v }
    grp_els = np.unique(np.concatenate([[el for b, el, df in grps[grp]] for grp in grps]))
    
    # Set up arrays for each legend, i.e. all same-colored/same-category elements ACROSS groups
    label_xs = []
    labels = []
    el_xs = collections.OrderedDict()
    el_ys = collections.OrderedDict()
    el_colors = collections.OrderedDict()
    
    x = 0
    for i, grp in enumerate(grps):
        grp_x = x
        grp_width = 0
        for j, (bench_name, bench_legend, df) in enumerate(grps[grp]):
            if bench_legend not in el_xs:
                el_xs[bench_legend] = []
                el_ys[bench_legend] = []
                el_colors[bench_legend] = []
            y = agg(np.median(df.loc[df["benchmark"] == bench_name, ys].to_numpy(), axis=1))
            el_xs[bench_legend].append(x)
            el_ys[bench_legend].append(y)
            el_colors[bench_legend] = bench_color(bench_name)
            x += w
            grp_width += w
        grp_x += grp_width / 2
        if grp_width > 0:
            label_xs.append(grp_x)
            labels.append(grp)
            x += sp
    
    nonspecial_labels = [label for label in labels if label != special_group]
    labels = [bench_label(label, nonspecial_labels) if label != special_group else bench_label(special_group) for label in labels]
    
    # plot bars + their legends
    bars = []
    legends = []
    for bench_legend in grp_els:
        bar = ax.bar(el_xs[bench_legend], el_ys[bench_legend], color=el_colors[bench_legend])
        bars.append(bar)
        legends.append(bench_legend)
    nonspecial_legends = [legend for legend in legends if legend != special_group]
    legends = [bench_label(legend, nonspecial_legends) if legend != special_group else bench_label(special_group) for legend in legends]
    ax.grid(axis="x")
    ax.legend(bars, legends, ncol=2)
    
    # plot category labels
    ax.set_xticks(label_xs)
    ax.set_xticklabels(labels, rotation=15, horizontalalignment="right")
    
    # for debugging and making sure the correct values are plotted
    #for bench_legend in grp_els:
    #    for i, group_label in enumerate(labels):
    #        if i > len(el_ys[bench_legend]):
    #            break
    #        print("{0} {1}: {2:4f}".format(bench_legend, group_label, el_ys[bench_legend][i]))

    
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
    #print(ys)
    ys = list(ys) + [y]
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
        idxmin = grps[y].idxmin()
        if idxmin.empty:
            return data
        else:
            return data.loc[idxmin]
    elif fun == "max":
        return data.loc[grps[y].idxmax()]
    else:
        raise ValueError()

"""
Do the plotting
"""
def plot(plt, args, data, group_cols, ymin, ymax, unit, runs_columns):
    # DATA
    data = aggregate(data, group_cols + ([args.x] if args.x else []), args.agg, args.y)
    grouped = data.groupby(group_cols, sort=False)
    
    #aggfun = np.median # FIXME
    #if args.agg == "min":
    aggfun = np.median
    
    if args.title:
        plt.title(" ".join(args.title))
    
    # PLOT 
    n_rows = len(args.plot)
    for row_i in range(0, n_rows):
        plot = args.plot[row_i]
        ax = plt.subplot(n_rows, 1, row_i+1)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_ylabel(column_titles[args.y] + " [" + unit + "]")
        if args.logscale:
            ax.set_yscale("log")

        # ACTUAL PLOTTING
        if plot == "box" or plot == "bar":
            plot_agg(grouped, ax, ys=runs_columns, bar=plot=="bar", agg=aggfun)

        elif plot == "line":
            if not args.x:
                continue
            xticks = data.loc[:, args.x].unique()
            ax.set_xticks(xticks)
            plot_sequence(grouped, ax, x=args.x, y=args.y)
            ax.set_xlabel(column_titles[args.x])
        
        elif plot == "grouped":
            ax.set_ylim(ymin=ymin, ymax=ymax)
            plot_grouped_bars(grouped, ax, ys=["kernel-median"], bench_groups=args.bar_groups, agg=aggfun)
            
        
"""
Main
"""
def main():
    global bench_colors, bench_markers
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=argparse.FileType("r"),
                        default=sys.stdin)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-b", "--benchmark", nargs="*", type=str, default=None)
    parser.add_argument("--stencil", nargs="*", type=str, default=None)
    parser.add_argument("--variant", nargs="*", type=str, default=None)
    parser.add_argument("--comp", type=str, default="both") # both, comp, no-comp
    parser.add_argument("--z-curves", type=str, default="both") # both, z-curves, no-z-curves
    parser.add_argument("--chase", type=str, default="both") # both, chase, no-chase
    parser.add_argument("--exclude", nargs="*", type=str, default=None)
    parser.add_argument("-g", "--groupby", nargs="*", default=["precision"]) 
        # groups additionally to benchmark & precision
    parser.add_argument("-x", type=str, default="size-z")
    parser.add_argument("-y", type=str, default="kernel-median")
    parser.add_argument("-p", "--plot", nargs="*", type=str, default=["box", "line"])
    parser.add_argument("--bar-groups", nargs="*", type=str, default=[])
    parser.add_argument("--plot-size", nargs=2, type=float, default= [6, 4]) # plt.rcParams.get("figure.figsize") #[11.69, 8.27])
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
    parser.add_argument("--animate", action="store_true", default=False)
    parser.add_argument("--color", type=str, default="variant")
    parser.add_argument("--marker", type=str, default="stencil")
    parser.add_argument("--dirty", action="store_true", default=False)
    parser.add_argument("--title", nargs="*", type=str, default=[])
    args = parser.parse_args()
    out = args.output
    if out == None:
        out = os.path.join(os.path.dirname(args.input.name),
                           (os.path.basename(args.input.name).rsplit(".csv", 1)[0]
                            + ".pdf"))
    
    if args.color == "storage":
        bench_colors = storage_colors
    if args.marker == "storage":
        bench_markers = storage_markers
    if args.color == "stencil":
        bench_colors = stencil_colors
    if args.marker == "stencil":
        bench_markers = stencil_markers
    
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
    data = read_data(args.input, args.dirty)
    runs_columns = data.columns[len(columns):len(data.columns)]

    # FILTER unwanted benchmarks
    filter_benchs(data, args)
    
    # EXTEND data by products
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

    # PLOT SCALE 
    ymin, ymax = get_limits(data, 
                            outliers_min=args.outliers_min,
                            outliers_max=args.outliers_max,
                            scale_min=args.scale_min,
                            scale_max=args.scale_max)

    # PLOT LAYOUT
    plt.style.use("seaborn")
    todrop = [[]]
    if(args.animate):
        for i, b in enumerate(args.benchmark):
            i += 1 
            todrop.append(args.benchmark[len(args.benchmark)-i:len(args.benchmark)])
        todrop = todrop[0:-1]
    subplotpars = None
    for i, drop in enumerate(todrop):
        f = plt.gcf()
        f.set_size_inches(args.plot_size[0], args.plot_size[1])
        f.subplots_adjust(hspace=0.6)
        data.drop(data[data["benchmark"].apply(lambda x: x in drop)].index, inplace=True)
        plot(plt, args, data, group_cols, ymin, ymax, unit, runs_columns)
        if(args.animate):
            out_split = os.path.splitext(out)
            out_ = "{0}-{1}{2}".format(out_split[0], i, out_split[1])
        else:
            out_ = out
        if not subplotpars:
            f.tight_layout()
            subplotpars = f.subplotpars
        else:
            s = subplotpars
            f.subplots_adjust(left=s.left, right=s.right, top=s.top, bottom=s.bottom, wspace=s.wspace, hspace=s.hspace)
        plt.savefig(out_, dpi=300)
        if(args.show):
            plt.show()
        plt.clf()
        plt.close()

if(__name__ == "__main__"):
	main()
