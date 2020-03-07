# Utilities for working with reformatted output CSVs and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Column lists for quicker access
class col:
    stencil         = [ "stencil" ]
    gridtype        = [ "unstructured" ]
    variant         = [ "variant" ]
    z_curves        = [ "z-curves" ]
    no_chase        = [ "no-chase" ]
    comp            = [ "comp" ]
    size            = [ "size-x", "size-y", "size-z" ]
    threads         = [ "threads-x", "threads-y", "threads-z" ]
    blocks          = [ "blocks-x", "blocks-y", "blocks-z" ]
    measurements    = [ "min", "max", "avg", "median" ]
    launchconfig    = threads + blocks
    access          = stencil + gridtype + variant
    storage         = z_curves + no_chase + comp
    category        = access + storage
    problem         = stencil + size + gridtype
    benchmark       = problem + variant + storage
    a               = benchmark + threads + measurements

# Return dataframe reduced to one entry with minimal value to minimize per group
def groupmin(df, by=col.problem, minimize="median"):
    return df.loc[df.groupby(by)[minimize].idxmin()]
    
# Return series of relative values relative to minimum element in same group
def relmingroup(df, by=col.problem, to="median"):
    grpmins = groupmin(df, by, minimize=to)
    return relgroup(df, grpmins, by, to)

# Return series of relative values relative to specific element per group
def relgroup(df, df_base, by=col.size, to="median"):
    cols = by + [to]
    tmp = df.merge(df_base[cols], on=by, how="left")
    assert len(tmp) == len(df) # if this fails, the input df_base had multiple matches for a base "by"; maybe use aggregate .min()?
    tmp.index = df.index
    return tmp[to + "_x"] / tmp[to + "_y"]


# ####################
# NAME PRETTY PRINTING
# ####################

# Definitions for prettyprinting
stencil_names = { "laplap"    : "Laplace-of-Laplace",
                  "hdiff"     : "Horizontal diffusion",
                  "fastwaves" : "Fastwaves" }
variant_names = { "idxvar"    : "index temporaries",
                  "idxvar-kloop" : "z-loop",
                  "idxvar-kloop-sliced" : "sliced z-loop" }
column_titles = { "pretty" : "Benchmark",
                  "size-x" : "Domain size (X)",          "size-y" : "Domain size (Y)",          "size-z" : "Domain size (Z)",
                  "blocks-x" : "Number of blocks (X)",   "blocks-y" : "Number of blocks (Y)",   "blocks-z" : "Number of blocks (Z)",
                  "threads-x" : "Number of threads (X)", "threads-y" : "Number of threads (Y)", "threads-z" : "Number of threads (Z)",
                  "size-prod" : "Domain size",
                  "blocks-prod" : "Number of blocks",
                  "threads-prod" : "Number of threads",
                  "kernel-avg" : "Average runtime [μs]",
                  "kernel-median" : "Median runtime [μs]",
                  "kernel-min" : "Minimal runtime [μs]",
                  "kernel-max" : "Maximal runtime [μs]", }

# Pretty-print names for graphs
# cols: Columns to be considered for generating pretty-print name (make sure to exclude any measurement values that vary for rows of the same data set)
# The cols array also prescribes the order of the pretty-printed name
# join: How the name parts built from the columns are joined
# fmt: default format to be applied for columns with no formatter specified in formatters
# formatters: dict column_name -> function(column_value); the function must return a string that will be used as the pretty print value for that column
def pretty(df, cols=col.category, **kwargs):
    columns = [ x for x in cols if x in df and len(np.unique(df[x])) > 1 ]
    return df.apply(lambda row: pretty_cb(row, columns, **kwargs), axis=1)

# Graph title: all unvarying components of names, equal for entire graph
def title(df, cols=col.category, **kwargs):
    columns = [ x for x in cols if x in df and len(np.unique(df[x])) == 1 ]
    return pretty_cb(df.iloc[0], columns, **kwargs)

def pretty_cb(row, cols=col.category, fmt="{0}: {1}", join=",  ", 
              formatters={ "unstructured" : lambda x, r: "unstructured" if x else "regular",
                           "z-curves"     : lambda x, r: ("z-curves" if x else "row-major") if r["unstructured"] else None,
                           "no-chase"     : lambda x, r: ("non-chasing" if x else "chasing") if r["unstructured"] else None,
                           "comp"         : lambda x, r: ("compressed" if x else "uncompressed") if r["unstructured"] else None,
                           "stencil"      : lambda x, r: stencil_names[x] if x in stencil_names else x.capitalize(),
                           "variant"      : lambda x, r: variant_names[x] if x in variant_names else x }):
    out = []
    for col in cols:
        if col not in row:
            continue
        if col in formatters:
            out.append(formatters[col](row[col], row))
        else:
            out.append(fmt.format(col, row[col]))
    return join.join([x for x in out if x])

# ##############
# PLOTTING
# ##############

# Setup common plot params
def plotinit(w=6, h=4):
    plt.style.use("seaborn")
    f = plt.gcf()
    f.set_size_inches(w, h)
    plt.grid()

def plotdone():
    plotlegend()
    f = plt.gcf()
    f.tight_layout()
    plt.show()

def saveplot(f):
    f = plt.gcf()
    f.tight_layout()
    plt.savefig(f, dpi=300)

def plotlegend(ncol=2, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=ncol, **kwargs)


# Line plot
def lineplot(df, by=col.category, x="threads-z", y="median",
             label="pretty", color="color", marker="marker", ax=None,
             xticks=True,
             **kwargs):
    if not ax:
        ax = plt.gca()
    mins = groupmin(df, by + [x], minimize=y) # ensure one point per line & X
    lines = mins.groupby(by)
    for i, data in lines:
        ax.plot(data[x], data[y],
                label=data[label].iat[0] if label in data else None,
                color=data[color].iat[0] if color in data else None,
                marker=data[marker].iat[0] if marker in data else None,
                **kwargs)
    if xticks:
        ax.set_xticks(np.unique(df[x]))
    ax.set_xlabel(column_titles[x] if x in column_titles else x)
    ax.set_ylabel(column_titles[y] if y in column_titles else y)
    ax.set_title(title(mins))

# Grouped bars plot
# cat: categories, these are differentiated by colors + legend
# grp: groups, these appear togheter as bars next to each other
def barplot(df, cat=col.access, grp=col.storage + col.gridtype, y="median",
            color="color", ax=None, 
            w=1, s=1.6, **kwargs):
    if not ax:
        ax = plt.gca()
    mins = groupmin(df, cat + grp, minimize=y) # ensure one point per bar
    
    groups = mins.groupby(grp)
    group_counts = groups[y].count()
    group_inner_no = groups.cumcount()
    group_numbers = pd.Series(np.arange(0, len(group_counts)), index=group_counts.index)
    group_offsets = pd.Series(np.roll(group_counts.cumsum(), 1), index=group_counts.index)
    group_offsets.iloc[0] = 0
    group_labels = pretty(group_counts.reset_index(), cols=grp)
    categories = mins.groupby(cat, as_index=False)
    category_labels = pretty(mins, cols=cat)
    
    for i, (category, data) in enumerate(categories):
        inner_no = [group_inner_no.loc[x] for x in data.index]
        group_no = data[grp].apply(lambda x: group_numbers.loc[tuple(x[grp])], axis=1)
        group_offs = data[grp].apply(lambda x: group_offsets.loc[tuple(x[grp])], axis=1)
        xs = group_no*s + group_offs*w + inner_no*w
        label = category_labels.loc[data.iloc[0].name]
        ax.bar(x=xs.values, height=data[y].values, label=label,
               color=data[color] if color in data else None)
    
    ax.set_xticks(group_numbers*s + group_offsets*w + group_counts*w*0.5)
    ax.set_xticklabels(group_labels, rotation=15, horizontalalignment="right")


# ##############
# COLORS/MARKERS
# ##############

# Return dataframe of markers/colors
def dfmap(df, cols, mapping, default=""):
    return df[cols].apply(lambda x: mapping[tuple(x)] if tuple(x) in mapping
                                    else default, axis=1)

# Add markers/colors to dataframe
def add_colors_markers(df, color="variant", marker=None, default=""):
    if marker == None and color:
        marker = color
    colfun = (     colors_variant if color == "variant"
              else colors_stencil if color == "stencil"
              else colors_storage if color == "storage"
              else lambda x: default)
    marfun = (     markers_variant if marker == "variant"
              else markers_stencil if marker == "stencil"
              else markers_storage if marker == "storage"
              else lambda x: default)
    tmp = df.copy()
    tmp["color"] = colfun(df)
    tmp["marker"] = marfun(df)
    return tmp

# Variant markers; use col.variant
def colors_variant(df):
    return dfmap(df, cols = col.variant,
                 mapping = { ("naive",) : "C1",
                             ("idxvar",) : "C2",
                             ("idxvar-kloop",) : "C3",
                             ("idxvar-kloop-sliced",) : "C4",
                             ("idxvar-shared",) : "C5",
                             ("regular",) : "black" })

def markers_variant(df):
    return dfmap(df, cols = col.variant,
                 mapping = { ("naive",) : "o",
                             ("idxvar",) : "*",
                             ("idxvar-kloop",) : "v",
                             ("idxvar-kloop-sliced",) : "^",
                             ("idxvar-shared",) : "P"} )

def colors_stencil(df):
    return dfmap(df, cols = col.stencil + col.gridtype,
                 mapping = { ("hdiff", False) : "C1",
                             ("hdiff", True) : "C2",
                             ("laplap", False) : "C3",
                             ("laplap", True) : "C4",
                             ("fastwaves", False) : "C5",
                             ("fastwaves", True) : "C6"} )
def markers_stencil(df):
    return dfmap(df, cols = col.stencil + col.gridtype,
                 mapping = { ("hdiff", False) : "o",
                             ("hdiff", True) : "*",
                             ("laplap", False) : "v",
                             ("laplap", True) : "^",
                             ("fastwaves", False) : "P",
                             ("fastwaves", True) : "8"} )

def colors_storage(df):
    return dfmap(df, cols = col.z_curves + col.no_chase + col.comp,
                 mapping = { (False, False, False) : "C1",
                             (False, False, True) : "C2",
                             (False, True, False) : "C3",
                             (True, False, False) : "C4",
                             (True, True, False) : "C5",
                             (False, True, True) : "C6",
                             (True, False, True) : "grey",
                             (True, True, True) : "darkorange" } )

def markers_storage(df):
    return dfmap(df, cols = col.z_curves + col.no_chase + col.comp,
                 mapping = { (False, False, False) : "o",
                             (False, False, True) : "*",
                             (False, True, False) : "v",
                             (True, False, False) : "^",
                             (True, True, False) : "P",
                             (False, True, True) : "8",
                             (True, False, True) : "p",
                             (True, True, True) : "H" } )