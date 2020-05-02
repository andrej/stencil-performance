#!/usr/bin/env python3
import sys
import argparse
import pandas as pd
import numpy as np

# Input Schema
name_col = "benchmark"
input_columns = [ name_col, "precision",
                  "size-x", "size-y", "size-z",
                  "blocks-x", "blocks-y", "blocks-z",
                  "threads-x", "threads-y", "threads-z",
                  "avg", "median", "min", "max" ]
input_column_types = { name_col : str,
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
                       "avg" : float,
                       "median" : float,
                       "min" : float,
                       "max" : float }

# Generate Columns From Benchmark Name String
# Lists Must Be Ordered By Priority; First Match Will Be Result For Column
gen_str_columns = { "stencil" : ["hdiff", "laplap", "fastwaves"],
                    "variant" : ["naive", "idxvar-kloop-sliced", "idxvar-kloop", "idxvar-shared", "idxvar"] }
gen_bool_columns = { "unstructured" : ["unstr"],
                     "comp" : ["comp"],
                     "z-curves" : ["z-curves"],
                     "no-chase" : ["no-chase"] }
gen_fun_columns = { "threads-prod" : lambda d: d["threads-x"]*d["threads-y"]*d["threads-z"] }

# Output Schema
output_columns = [ "stencil", "precision", "variant", "unstructured", "z-curves", "comp", "no-chase",
                   "size-x", "size-y", "size-z",
                   "blocks-x", "blocks-y", "blocks-z",
                   "threads-prod", "threads-x", "threads-y", "threads-z",
                   "median", "avg", "min", "max" ]

# try to convert to float, if impossible return nan; used to drop invalid rows later
def float_or_nan(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


# read and parse data to correct type; drop invalid
def read_df(path, dirty=False):
    # parse CSV
    df = pd.read_csv(path, skiprows=3, header=None, error_bad_lines=False)
    
    # rename columns as in global variables above
    runs_columns = ["run-{0}".format(x) for x in range(0, len(df.columns)-len(input_columns))]
    df.set_axis(input_columns + runs_columns, axis=1, inplace=True)
    
    # type conversion
    string_columns = [x for x in input_columns if input_column_types[x] == str]
    df[string_columns] = df[string_columns].apply(lambda s: s.str.strip(), axis=1)
    if dirty:
        numeric_columns = [x for x in input_columns if input_column_types[x] in ["Int64", int, float]] + runs_columns
        df[numeric_columns] = df[numeric_columns].applymap(lambda x: float_or_nan(x))
        df = df.dropna()
    df = df.astype(input_column_types)
    return df


# generate new columns according to globals above
def gen_columns(df):
    default_str = ""
    for col in gen_str_columns:
        vals = gen_str_columns[col]
        df[col] = df[name_col].apply(lambda d: ([x for x in vals if x in d] + [default_str])[0])
    for col in gen_bool_columns:
        vals = gen_bool_columns[col]
        df[col] = df[name_col].apply(lambda d: np.any([x in d for x in vals]))
    for col in gen_fun_columns:
        df[col] = df.apply(gen_fun_columns[col], axis=1)


# main
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input", type=argparse.FileType("r"))
    arg_parser.add_argument("-o", "--output", type=argparse.FileType("w"), default=sys.stdout)
    arg_parser.add_argument("-d", "--dirty", action="store_true", default=False)
    args = arg_parser.parse_args()
    df = read_df(args.input, args.dirty)
    gen_columns(df)
    df = df[output_columns]
    df.to_csv(args.output)
    return 0
    
if __name__ == '__main__':
    sys.exit(main())
