#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "coord3.cu"
#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-ref.cu"
#include "benchmarks/hdiff-cpu-unstr.cu"
#include "benchmarks/hdiff-cuda.cu"
#include "benchmarks/hdiff-cuda-seq.cu"
#include "benchmarks/hdiff-cuda-unstr-seq.cu"
#include "benchmarks/hdiff-cuda-unstr.cu"

/** Benchmark results type: Vector of results for each individual benchmark.
 * The benchmark result at the first index contains the total running time.
 * At the second index is the time required for running the stencil itself.
 * At the third index is the time required for running the setup + teardown. */
typedef std::vector<Benchmark<double> *> benchmark_list_t;

/** List of all available benchmarks for mapping args to these values. */
typedef enum {all_benchs, 
              hdiff_ref,
              hdiff_ref_unstr,
              hdiff_cuda_regular,
              hdiff_cuda_regular_kloop,
              hdiff_cuda_sequential,
              hdiff_cuda_unstr_naive,
              hdiff_cuda_unstr_kloop,
              hdiff_cuda_unstr_idxvars,
              hdiff_cuda_unstr_seq,
              unspecified} 
benchmark_type_t;

/** Struct describing the benchmark to be run (default arguments) */
struct args_t {
    benchmark_type_t type = unspecified;
    std::vector<coord3> sizes; // default can be found in parse_args() function
    int runs = 20;
    std::vector<coord3> numthreads; // default can be found in parse_args() function
    std::vector<coord3> numblocks; // default can be found in parse_args() function
    bool print = false; // print output of benchmarked grids to stdout (makes sense for small grids)
    bool skip_errors = false; // skip printing output for erroneous benchmarks
    bool no_header = false; // print no header in the output table
};

/** Populates the list of available benchmarks. */
void get_benchmark_identifiers(std::map<std::string, benchmark_type_t> *ret) {
    (*ret)["all"] = all_benchs;
    (*ret)["hdiff-ref"] = hdiff_ref;
    (*ret)["hdiff-ref-unstr"] = hdiff_ref_unstr;
    (*ret)["hdiff-regular"] = hdiff_cuda_regular;
    (*ret)["hdiff-regular-kloop"] = hdiff_cuda_regular_kloop;
    (*ret)["hdiff-regular-seq"] = hdiff_cuda_sequential;
    (*ret)["hdiff-unstr-naive"] = hdiff_cuda_unstr_naive;
    (*ret)["hdiff-unstr-kloop"] = hdiff_cuda_unstr_kloop;
    (*ret)["hdiff-unstr-idxvars"] = hdiff_cuda_unstr_idxvars;
    (*ret)["hdiff-unstr-seq"] = hdiff_cuda_unstr_seq;
}

/** Helper function to parse input strings of the format %dx%dx%d, e.g. 16x16x1
 * into a vector of coord3s. Returns by how much the pointer passed in was
 * increased, i.e. how many strings directly following contained coord3s. */
int scan_coord3(char **strs, std::vector<coord3> *ret) {
    int i = 0;
    while(1) {
        char *str=strs[i];
        coord3 to_add;
        int c = sscanf(str, "%dx%dx%d", &to_add.x, &to_add.y, &to_add.z);
        if(c == 0) {
            break;
        }
        if(c <= 2) {
            to_add.z = 1;
        }
        if(c <= 1) {
            to_add.y = 1;
        }
        ret->push_back(to_add);
        i++;
    }
    return i;
}

/** Very simple arg parser; if multiple valid arguments for the same setting
 * are passed, the last one wins. */
args_t parse_args(int argc, char** argv) {
    args_t ret;
    std::map<std::string, benchmark_type_t> benchmark_identifiers;
    get_benchmark_identifiers(&benchmark_identifiers);
    for(int i=1; i<argc; i++) {
        std::string arg = std::string(argv[i]);
        if(benchmark_identifiers.count(arg) > 0) {
            ret.type = benchmark_identifiers[arg];
        } else if(arg == "--size" && i+1 < argc) {
            i += scan_coord3(&(argv[i+1]), &ret.sizes);
        } else if(arg == "--runs") {
            sscanf(argv[i+1], "%d", &ret.runs);
            i += 1;
        } else if(arg == "--threads" && i+1 < argc) {
            i += scan_coord3(&(argv[i+1]), &ret.numthreads);
        } else if(arg == "--blocks" && i+1 < argc) {
            i += scan_coord3(&(argv[i+1]), &ret.numblocks);
        } else if(arg == "--print") {
            ret.print = true;
        } else if(arg == "--skip-errors") {
            ret.skip_errors = true;
        } else if(arg == "--no-header") {
            ret.no_header = true;
        } else {
            fprintf(stderr, "Unrecognized argument %s.\n", arg.c_str());
            exit(1);
        }
    }
    // Default numthreads/numblocks if none of both are given
    if(ret.numthreads.empty() && ret.numblocks.empty()) {
        ret.numthreads.push_back(coord3(0, 0, 0)); //auto calculate
        ret.numblocks.push_back(coord3(32, 32, 32));
    }
    if(ret.numthreads.empty()) {
        // setting to default 0, 0, 0, benchmark class will calculate correct value
        // for data size itself
        ret.numthreads.push_back(coord3(0, 0, 0));
    }
    if(ret.numblocks.empty()) {
        ret.numblocks.push_back(coord3(0, 0, 0));
    }
    if(ret.sizes.empty()) {
        ret.sizes.push_back(coord3(32, 32, 32));
    }
    return ret;
}

/** Create the benchmark class for one of the available types. */
Benchmark<double> *create_benchmark(benchmark_type_t type, coord3 size, coord3 numthreads, coord3 numblocks, int runs, bool quiet) {
    Benchmark<double> *ret = NULL;
    switch(type) {
        case hdiff_ref:
        ret = new HdiffReferenceBenchmark(size);
        break;
        case hdiff_ref_unstr:
        ret = new HdiffCPUUnstrBenchmark(size);
        break;
        case hdiff_cuda_regular:
        ret = new HdiffCudaBenchmark(size);
        break;
        case hdiff_cuda_regular_kloop:
        ret = new HdiffCudaBenchmark(size, HdiffCudaRegular::kloop);
        break;
        case hdiff_cuda_sequential:
        ret = new HdiffCudaSequentialBenchmark(size);
        break;
        case hdiff_cuda_unstr_naive:
        ret = new HdiffCudaUnstrBenchmark(size);
        break;
        case hdiff_cuda_unstr_kloop:
        ret = new HdiffCudaUnstrBenchmark(size, HdiffCudaUnstr::UnstrKloop);
        break;
        case hdiff_cuda_unstr_idxvars:
        ret = new HdiffCudaUnstrBenchmark(size, HdiffCudaUnstr::UnstrIdxvars);
        break;
        case hdiff_cuda_unstr_seq:
        ret = new HdiffCudaUnstructuredSequentialBenchmark(size);
        break;
    }
    ret->_numthreads = dim3(numthreads.x, numthreads.y, numthreads.z);
    ret->_numblocks = dim3(numblocks.x, numblocks.y, numblocks.z);
    ret->runs = runs;
    ret->quiet = quiet;
    return ret;
}

/** From the given arguments, create a vector of benchmarks to execute. */
benchmark_list_t *create_benchmarks(args_t args) {
    std::vector<benchmark_type_t> types;
    if(args.type == all_benchs) {
        for(int it = all_benchs; it < unspecified; it++) {
            if(it == all_benchs) {
                continue;
            }
            types.push_back((benchmark_type_t)it);
        }
    } else {
        types.push_back(args.type);
    }

    benchmark_list_t *ret = new benchmark_list_t();
    for(auto it=types.begin(); it != types.end(); ++it) {
        benchmark_type_t type = *it;
        for(auto s_it = args.sizes.begin(); s_it != args.sizes.end(); ++s_it) {
            coord3 size = *s_it;
            int added = 0;
            for(auto t_it = args.numthreads.begin(); t_it != args.numthreads.end(); ++t_it) {
                coord3 numthreads = *t_it;
                for(auto b_it = args.numblocks.begin(); b_it != args.numblocks.end(); ++b_it) {
                    coord3 numblocks = *b_it;
                    Benchmark<double> *bench = create_benchmark(type,
                                                                size,
                                                                numthreads,
                                                                numblocks,
                                                                args.runs,
                                                                !args.print);
                    // only add benchmark if it respected the requested
                    // numthreads/numblocks; it can happen that less threads/
                    // blocks than requested are used if the benchmark does not
                    // support it
                    if(numthreads != coord3(0, 0, 0) && numthreads != bench->numthreads()) {
                        continue;
                    }
                    if(numblocks != coord3(0, 0, 0) && numblocks != bench->numblocks()) {
                        continue;
                    }
                    ret->push_back(bench);
                    added++;
                }
            }
            if(added == 0) {
                Benchmark<double> *bench = create_benchmark(type,
                                                            size,
                                                            args.numthreads[0],
                                                            args.numblocks[0],
                                                            args.runs,
                                                            !args.print);
                ret->push_back(bench);
            }
        }
    }

    return ret;
}

/** Create the benchmark described in bench_info, execute it and then return
 * its performance metrics. */
void run_benchmark(Benchmark<double> *bench, bool quiet = false) {
    if(!quiet) {
        dim3 numblocks = bench->numblocks();
        dim3 numthreads = bench->numthreads();
        fprintf(stderr, "Running '%s' on grid size (%d, %d, %d), blocks (%d, %d, %d), threads (%d, %d, %d), runs %d.\n",
                bench->name.c_str(), bench->size.x, bench->size.y, bench->size.z,
                numblocks.x, numblocks.y, numblocks.z,
                numthreads.x, numthreads.y, numthreads.z,
                bench->runs);
    }
    try {
        bench->execute();
    } catch (std::runtime_error e) {
        bench->error = true;
        if(!quiet) {
            fprintf(stderr, "    Error: %s\n", e.what());
        }
    }
}

/** Pretty print the results in a table (format is CSV-compatible, can be exported into Excel). */
void prettyprint(benchmark_list_t *benchmarks, bool skip_errors=false, bool header=true) {
    if(header) {
        printf("Benchmark                , Blocks       , Threads      , Total execution time           , Kernel-only execution time     \n");
        printf("                         ,   X,   Y,   Z,   X,   Y,   Z,   Average,   Minimum,   Maximum,   Average,   Minimum,   Maximum\n");
    }
    for(auto it=benchmarks->begin(); it != benchmarks->end(); ++it) {
        Benchmark<double> *bench = *it;
        if(bench->error && skip_errors) {
            continue;
        }
        dim3 numblocks = bench->numblocks();
        dim3 numthreads = bench->numthreads();
        printf("%-25s,%4d,%4d,%4d,%4d,%4d,%4d,%10.6f,%10.6f,%10.6f,%10.6f,%10.6f,%10.6f%s\n",
               bench->name.c_str(),
               numblocks.x, numblocks.y, numblocks.z,
               numthreads.x, numthreads.y, numthreads.z,
               bench->results.total.avg, bench->results.total.min, bench->results.total.max,
               bench->results.kernel.avg, bench->results.kernel.min, bench->results.kernel.max,
               (bench->error ? ", (Error)" : ""));
    }
}

/** Print usage notice and exit. */
void usage(int argc, char** argv) {
    fprintf(stderr,
"Usage: %s [--size N,M,L] [--{min,max,step}blocks N,M,L] [--print] BENCHMARK \n \
 Benchmarks: all, hdiff-ref, ...\n", argv[0]);
    exit(1);
}

/** Main */
int main(int argc, char** argv) {
    args_t args = parse_args(argc, argv);
    if(args.type == unspecified) {
        usage(argc, argv);
        return 1;
    }
    benchmark_list_t *benchmarks = create_benchmarks(args);
    for(auto it=benchmarks->begin(); it != benchmarks->end(); ++it) {
        run_benchmark(*it);
    }
    fprintf(stderr, "\n");
    prettyprint(benchmarks, args.skip_errors, !args.no_header);
    return 0;
}
