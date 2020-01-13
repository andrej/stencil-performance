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
//#include "benchmarks/hdiff-cpu-unstr.cu"
#include "benchmarks/hdiff-cuda-regular.cu"
#include "benchmarks/hdiff-cuda-regular-unfused.cu"
#include "benchmarks/hdiff-cuda-unstr.cu"
#include "benchmarks/hdiff-cuda-unstr-unfused.cu"
#include "benchmarks/fastwaves-ref.cu"
#include "benchmarks/fastwaves-regular.cu"
#include "benchmarks/fastwaves-unstr.cu"
#include "benchmarks/laplap-regular.cu"
#include "benchmarks/laplap-unstr.cu"


/** List of all available benchmarks for mapping args to these values. */
typedef enum {all_benchs, 
    // Horizontal Diffusion
              hdiff_ref,
              hdiff_cuda_regular_unfused,
              hdiff_cuda_unstr_unfused,              
              hdiff_cuda_regular_naive,
              hdiff_cuda_regular_iloop,
              hdiff_cuda_regular_jloop,
              hdiff_cuda_regular_kloop,
              hdiff_cuda_regular_idxvar,
              hdiff_cuda_regular_idxvar_kloop,
              hdiff_cuda_regular_idxvar_shared,
              hdiff_cuda_regular_coop,
              hdiff_cuda_regular_shared,
              hdiff_cuda_regular_shared_kloop,
              hdiff_cuda_unstr_naive,
              hdiff_cuda_unstr_idxvar,
              hdiff_cuda_unstr_idxvar_kloop,
              hdiff_cuda_unstr_idxvar_shared,
              fastwaves_ref,
              fastwaves_regular_unfused,
              fastwaves_unstr_unfused,
              fastwaves_regular_naive,
              fastwaves_regular_idxvar,
              fastwaves_regular_kloop,
              fastwaves_unstr_naive,
              fastwaves_unstr_idxvar,
              fastwaves_unstr_kloop,
              laplap_regular_unfused,
              laplap_regular_naive,
              laplap_regular_idxvar,
              laplap_regular_idxvar_kloop,
              laplap_regular_idxvar_kloop_sliced,
              laplap_regular_idxvar_shared,
              laplap_regular_shared,
              laplap_unstr_unfused,
              laplap_unstr_naive,
              laplap_unstr_idxvar,
              laplap_unstr_idxvar_kloop,
              laplap_unstr_idxvar_kloop_sliced,
              laplap_unstr_idxvar_shared,
              unspecified} 
benchmark_type_t;

/** Benchmarks can be run in single or double precision, this enum is used to differentiate the two. */
typedef enum {single_prec, double_prec} precision_t;

/** Type describing a benchmark type + its benchmark-specific arguments */
struct benchmark_params_t {
    benchmark_type_t type;
    precision_t precision;
    int argc;
    char **argv;
};

/** Benchmark results type: Vector of results for each individual benchmark.
 * The benchmark result at the first index contains the total running time.
 * At the second index is the time required for running the stencil itself.
 * At the third index is the time required for running the setup + teardown. */
 typedef struct { Benchmark *obj; benchmark_params_t params; } benchmark_t;
 typedef std::vector<benchmark_t> benchmark_list_t;

/** Struct used to store all parsed arguments from command line. */
struct args_t {
    std::vector<benchmark_params_t> types;
    std::vector<coord3> sizes; // default can be found in parse_args() function
    int runs = 20;
    std::vector<coord3> numthreads; // default can be found in parse_args() function
    std::vector<coord3> numblocks; // default can be found in parse_args() function
    bool print = false; // print output of benchmarked grids to stdout (makes sense for small grids)
    bool skip_errors = false; // skip printing output for erroneous benchmarks
    bool no_header = false; // print no header in the output table
    bool no_verify = false; // skip verification
    std::vector<precision_t> precisions;
};

void get_benchmark_identifiers(std::map<std::string, benchmark_type_t> *ret);
int scan_coord3(char **strs, int n, std::vector<coord3> *ret);
args_t parse_args(int argc, char** argv);
Benchmark *create_benchmark(benchmark_params_t type, coord3 size, coord3 numthreads, coord3 numblocks, int runs, bool quiet, bool no_verify);
benchmark_list_t *create_benchmarks(args_t args);
void run_benchmark(Benchmark *bench, bool quiet = false);
void prettyprint(benchmark_t *benchmark, bool skip_errors=false);
void usage(int argc, char** argv);
int main(int argc, char** argv);


// IMPLEMENTATIONS

/** Populates the list of available benchmarks. */
void get_benchmark_identifiers(std::map<std::string, benchmark_type_t> *ret) {
    for(int i=all_benchs; i < unspecified; i++) {
        benchmark_type_t type = (benchmark_type_t)i;
        std::string name;
        if(type == all_benchs) {
            name = "all";
        } else {
            // create benchmark simply to ask for its name
            benchmark_params_t param_bench = { .type = type };
            Benchmark *bench = create_benchmark(param_bench, coord3(1, 1, 1), coord3(1, 1, 1), coord3(1, 1, 1), 1, true, true);
            name = bench->name;
            delete bench;
        }
        (*ret)[name] = type;
    }
}

/** Helper function to parse input strings of the format %dx%dx%d, e.g. 16x16x1
 * into a vector of coord3s. Returns by how much the pointer passed in was
 * increased, i.e. how many strings directly following contained coord3s. */
int scan_coord3(char **strs, int n, std::vector<coord3> *ret) {
    int i = 0;
    for(; i<n; i++) {
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
    }
    return i;
}

/** Very simple arg parser; if multiple valid arguments for the same setting
 * are passed, the last one wins. */
args_t parse_args(int argc, char** argv) {
    args_t ret;
    std::map<std::string, benchmark_type_t> benchmark_identifiers;
    get_benchmark_identifiers(&benchmark_identifiers);

    benchmark_params_t current_bench;
    current_bench.type = unspecified; // unspecified used for general arguments that apply to all benchmarks
    // note once a benchmark identifier was present, only benchmark-specific arguments may follow

    for(int i = 1; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if(benchmark_identifiers.count(arg) > 0) {
            // benchmark identifier: start a new benchmark-specific entry
            // all parameters beyond this point must be considered benchmark-specific
            if(current_bench.type != unspecified) {
                // add previous current bench as its arguments are now finished (delimited by next identifier)
                ret.types.push_back(current_bench);
            }
            current_bench.type = benchmark_identifiers[arg];
            current_bench.argc = 0;
            current_bench.argv = argv + i + 1;
        } else if(current_bench.type == unspecified) {
            if(arg == "--size" && i+1 < argc) {
                i += scan_coord3(&(argv[i+1]), argc-i-1, &ret.sizes);
            } else if(arg == "--runs" && i+1 < argc) {
                sscanf(argv[i+1], "%d", &ret.runs);
                i += 1;
            } else if(arg == "--threads" && i+1 < argc) {
                i += scan_coord3(&(argv[i+1]), argc-i-1, &ret.numthreads);
            } else if(arg == "--blocks" && i+1 < argc) {
                i += scan_coord3(&(argv[i+1]), argc-i-1, &ret.numblocks);
            } else if(arg == "--print") {
                ret.print = true;
            } else if(arg == "--skip-errors") {
                ret.skip_errors = true;
            } else if(arg == "--no-header") {
                ret.no_header = true;
            } else if(arg == "--no-verify") {
                ret.no_verify = true;
            } else if(arg == "--single-prec") {
                ret.precisions.push_back(single_prec);
            } else if(arg == "--double-prec") {
                ret.precisions.push_back(double_prec);
            } else {
                    fprintf(stderr, "Unrecognized or incomplete argument %s.\n", arg.c_str());
                    exit(1);
            }
        } else {
            current_bench.argc++;
        }
    }
    // push last arguments, if any
    if(current_bench.type != unspecified) {
        // add previous current bench as its arguments are now finished (delimited by next identifier)
        ret.types.push_back(current_bench);
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
    if(ret.precisions.empty()) {
        ret.precisions.push_back(double_prec);
    }
    return ret;
}

/** Create the benchmark class for one of the available types. */
Benchmark *create_benchmark(benchmark_params_t param_bench, coord3 size,
                            coord3 numthreads, coord3 numblocks, int runs,
                            bool quiet, bool no_verify) {
    Benchmark *ret = NULL;
    precision_t precision = param_bench.precision;
    switch(param_bench.type) {
        case hdiff_ref:
        ret = (precision == single_prec ?
               (Benchmark *) new HdiffReferenceBenchmark<float>(size) :
               (Benchmark *) new HdiffReferenceBenchmark<double>(size) );
        break;
        //case hdiff_ref_unstr:
        //ret = (precision == single_prec ?
        //    (Benchmark *) new HdiffCPUUnstrBenchmark<float>(size) :
        //    (Benchmark *) new HdiffCPUUnstrBenchmark<double>(size) );
        break;
        case hdiff_cuda_regular_unfused:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularUnfusedBenchmark<float>(size) :
            (Benchmark *) new HdiffCudaRegularUnfusedBenchmark<double>(size) );
        break;
        case hdiff_cuda_unstr_unfused:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaUnstructuredUnfusedBenchmark<float>(size) :
            (Benchmark *) new HdiffCudaUnstructuredUnfusedBenchmark<double>(size) );
        break;
        case hdiff_cuda_regular_naive:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size) );
        break;
        case hdiff_cuda_regular_iloop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::iloop) : 
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::iloop) );
        break;
        case hdiff_cuda_regular_jloop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::jloop) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::jloop) );
        break;
        case hdiff_cuda_regular_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::kloop) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::kloop) );
        break;
        case hdiff_cuda_regular_idxvar:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::idxvar) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::idxvar) );
        break;
        case hdiff_cuda_regular_idxvar_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::idxvar_kloop) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::idxvar_kloop) );
        break;
        case hdiff_cuda_regular_idxvar_shared:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::idxvar_shared) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::idxvar_shared) );
        break;
        case hdiff_cuda_regular_coop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::coop) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::coop) );
        break;
        case hdiff_cuda_regular_shared:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::shared) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::shared) );
        break;
        case hdiff_cuda_regular_shared_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaRegularBenchmark<float>(size, HdiffCudaRegular::shared_kloop) :
            (Benchmark *) new HdiffCudaRegularBenchmark<double>(size, HdiffCudaRegular::shared_kloop) );
        break;
        case hdiff_cuda_unstr_naive:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaUnstrBenchmark<float>(size, HdiffCudaUnstr::naive) :
            (Benchmark *) new HdiffCudaUnstrBenchmark<double>(size, HdiffCudaUnstr::naive) );
        break;
        case hdiff_cuda_unstr_idxvar:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaUnstrBenchmark<float>(size) :
            (Benchmark *) new HdiffCudaUnstrBenchmark<double>(size) );
        break;
        case hdiff_cuda_unstr_idxvar_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaUnstrBenchmark<float>(size, HdiffCudaUnstr::idxvar_kloop) :
            (Benchmark *) new HdiffCudaUnstrBenchmark<double>(size, HdiffCudaUnstr::idxvar_kloop) );
        break;
        case hdiff_cuda_unstr_idxvar_shared:
        ret = (precision == single_prec ?
            (Benchmark *) new HdiffCudaUnstrBenchmark<float>(size, HdiffCudaUnstr::idxvar_shared) :
            (Benchmark *) new HdiffCudaUnstrBenchmark<double>(size, HdiffCudaUnstr::idxvar_shared) );
        break;
        case fastwaves_ref:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesRefBenchmark<float>(size) :
            (Benchmark *) new FastWavesRefBenchmark<double>(size) );
        break;
        case fastwaves_regular_unfused:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesRegularBenchmark<float>(size, FastWavesRegularBenchmarkNamespace::unfused) :
            (Benchmark *) new FastWavesRegularBenchmark<double>(size, FastWavesRegularBenchmarkNamespace::unfused) );
        break;
        case fastwaves_unstr_unfused:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesUnstrBenchmark<float>(size, FastWavesUnstrBenchmarkNamespace::unfused) :
            (Benchmark *) new FastWavesUnstrBenchmark<double>(size, FastWavesUnstrBenchmarkNamespace::unfused) );
        break;
        case fastwaves_regular_naive:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesRegularBenchmark<float>(size, FastWavesRegularBenchmarkNamespace::naive) :
            (Benchmark *) new FastWavesRegularBenchmark<double>(size, FastWavesRegularBenchmarkNamespace::naive) );
        break;
        case fastwaves_regular_idxvar:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesRegularBenchmark<float>(size, FastWavesRegularBenchmarkNamespace::idxvar) :
            (Benchmark *) new FastWavesRegularBenchmark<double>(size, FastWavesRegularBenchmarkNamespace::idxvar) );
        break;
        case fastwaves_regular_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesRegularBenchmark<float>(size, FastWavesRegularBenchmarkNamespace::kloop) :
            (Benchmark *) new FastWavesRegularBenchmark<double>(size, FastWavesRegularBenchmarkNamespace::kloop) );
        break;
        case fastwaves_unstr_naive:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesUnstrBenchmark<float>(size, FastWavesUnstrBenchmarkNamespace::naive) :
            (Benchmark *) new FastWavesUnstrBenchmark<double>(size, FastWavesUnstrBenchmarkNamespace::naive) );
        break;
        case fastwaves_unstr_idxvar:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesUnstrBenchmark<float>(size, FastWavesUnstrBenchmarkNamespace::idxvar) :
            (Benchmark *) new FastWavesUnstrBenchmark<double>(size, FastWavesUnstrBenchmarkNamespace::idxvar) );
        break;
        case fastwaves_unstr_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new FastWavesUnstrBenchmark<float>(size, FastWavesUnstrBenchmarkNamespace::kloop) :
            (Benchmark *) new FastWavesUnstrBenchmark<double>(size, FastWavesUnstrBenchmarkNamespace::kloop) );
        break;
        case laplap_regular_unfused:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::unfused) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::unfused) );
        break;
        case laplap_regular_naive:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::naive) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::naive) );
        break;
        case laplap_regular_idxvar:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::idxvar) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::idxvar) );
        break;
        case laplap_regular_idxvar_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::idxvar_kloop) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::idxvar_kloop) );
        break;
        case laplap_regular_idxvar_kloop_sliced:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::idxvar_kloop_sliced) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::idxvar_kloop_sliced) );
        break;
        case laplap_regular_idxvar_shared:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::idxvar_shared) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::idxvar_shared) );
        break;
        case laplap_regular_shared:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapRegularBenchmark<float>(size, LapLapRegular::shared) :
            (Benchmark *) new LapLapRegularBenchmark<double>(size, LapLapRegular::shared) );
        break;
        case laplap_unstr_unfused:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapUnstrBenchmark<float>(size, LapLapUnstr::unfused) :
            (Benchmark *) new LapLapUnstrBenchmark<double>(size, LapLapUnstr::unfused) );
        break;
        case laplap_unstr_naive:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapUnstrBenchmark<float>(size, LapLapUnstr::naive) :
            (Benchmark *) new LapLapUnstrBenchmark<double>(size, LapLapUnstr::naive) );
        break;
        case laplap_unstr_idxvar:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapUnstrBenchmark<float>(size, LapLapUnstr::idxvar) :
            (Benchmark *) new LapLapUnstrBenchmark<double>(size, LapLapUnstr::idxvar) );
        break;
        case laplap_unstr_idxvar_kloop:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapUnstrBenchmark<float>(size, LapLapUnstr::idxvar_kloop) :
            (Benchmark *) new LapLapUnstrBenchmark<double>(size, LapLapUnstr::idxvar_kloop) );
        break;
        case laplap_unstr_idxvar_kloop_sliced:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapUnstrBenchmark<float>(size, LapLapUnstr::idxvar_kloop_sliced) :
            (Benchmark *) new LapLapUnstrBenchmark<double>(size, LapLapUnstr::idxvar_kloop_sliced) );
        break;
        case laplap_unstr_idxvar_shared:
        ret = (precision == single_prec ?
            (Benchmark *) new LapLapUnstrBenchmark<float>(size, LapLapUnstr::idxvar_shared) :
            (Benchmark *) new LapLapUnstrBenchmark<double>(size, LapLapUnstr::idxvar_shared) );
        break;
        default:
        return NULL;
    }
    ret->_numthreads = dim3(numthreads.x, numthreads.y, numthreads.z);
    ret->_numblocks = dim3(numblocks.x, numblocks.y, numblocks.z);
    ret->runs = runs;
    ret->quiet = quiet;
    ret->do_verify = !no_verify;
    ret->argc = param_bench.argc;
    ret->argv = param_bench.argv;
    ret->parse_args();
    return ret;
}

/** From the given arguments, create a vector of benchmarks to execute. */
benchmark_list_t *create_benchmarks(args_t args) {
    std::vector<benchmark_params_t> types;
    for(auto it=args.types.begin(); it != args.types.end(); ++it) {
        if(it->type == all_benchs) {
            types.clear();
            for(int it = all_benchs; it < unspecified; it++) {
                if(it == all_benchs || it == hdiff_ref /*|| it == hdiff_ref_unstr*/) {
                    // reference and unstructured cpu are not included in "all" benchmarks
                    continue;
                }
                benchmark_params_t param_bench = { .type = (benchmark_type_t)it };
                types.push_back(param_bench);
            }
            break;
        }
        types.push_back(*it);
    }

    benchmark_list_t *ret = new benchmark_list_t();
    for(auto it=types.begin(); it != types.end(); ++it) {
        benchmark_type_t type = it->type;
        int argc = it->argc;
        char **argv = it->argv;
        for(auto s_it = args.sizes.begin(); s_it != args.sizes.end(); ++s_it) {
            coord3 size = *s_it;
            for(auto p_it = args.precisions.begin(); p_it != args.precisions.end(); ++p_it) {
                precision_t precision = *p_it;
                int added = 0;
                benchmark_params_t params = { .type = type,
                                              .precision = precision,
                                              .argc = argc,
                                              .argv = argv};
                for(auto t_it = args.numthreads.begin(); t_it != args.numthreads.end(); ++t_it) {
                    coord3 numthreads = *t_it;
                    for(auto b_it = args.numblocks.begin(); b_it != args.numblocks.end(); ++b_it) {
                        coord3 numblocks = *b_it;

                        Benchmark *bench = create_benchmark(params, size, numthreads,
                                                            numblocks, args.runs,
                                                            !args.print, args.no_verify);
                        // Skip if creation somehow failed
                        if(!bench) {
                            continue;
                        }
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
                        benchmark_t add = {.obj = bench, .params = params};
                        ret->push_back(add);
                        added++;
                    }
                }
                if(added == 0) {
                    Benchmark *bench = create_benchmark(params, size,
                                                        args.numthreads[0], args.numblocks[0],
                                                        args.runs, !args.print,
                                                        args.no_verify);
                    benchmark_t add = {.obj = bench, .params = params};
                    ret->push_back(add);
                }
            }
        }
    }

    return ret;
}

/** Create the benchmark described in bench_info, execute it and then return
 * its performance metrics. */
void run_benchmark(Benchmark *bench, bool quiet) {
    if(!quiet) {
        dim3 numblocks = bench->numblocks();
        dim3 numthreads = bench->numthreads();
    }
    try {
        bench->execute();
    } catch (std::runtime_error e) {
        bench->error = true;
        if(!quiet) {
            fprintf(stderr, "Error: %s\n", e.what());
        }
    }
}

/** Pretty print the results in a table (format is CSV-compatible, can be exported into Excel). */
void prettyprint(benchmark_t *it, bool skip_errors) {
    benchmark_params_t params = it->params;
    Benchmark *bench = it->obj;
    if(bench->error && skip_errors) {
        return;
    }
    dim3 numblocks = bench->numblocks();
    dim3 numthreads = bench->numthreads();
    printf("%-28s,%10s,%4d,%4d,%4d,%4d,%4d,%4d,%4d,%4d,%4d,%10.0f,%10.0f,%10.0f,%10.0f%s\n",
            bench->name.c_str(),
            (params.precision == single_prec ? "single" : "double"),
            bench->size.x, bench->size.y, bench->size.z,
            numblocks.x, numblocks.y, numblocks.z,
            numthreads.x, numthreads.y, numthreads.z,
            bench->results.runtime.avg, bench->results.runtime.median, bench->results.runtime.min, bench->results.runtime.max,
            (bench->error ? ", (Error)" : ""));
    fflush(stdout);
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
    if(args.types.empty()) {
        usage(argc, argv);
        return 1;
    }
    benchmark_list_t *benchmarks = create_benchmarks(args);
    // Print command that was used to generate these benchmarks for reproducibility
    if(!args.no_header) {
        for(int i = 0; i < argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        printf("Benchmark                   , Precision, Domain size,,, Blocks     ,,, Threads    ,,, Kernel-only execution time                \n");
        printf("                            ,          ,   X,   Y,   Z,   X,   Y,   Z,   X,   Y,   Z,   Average,    Median,   Minimum,   Maximum\n");
    }
    for(auto it=benchmarks->begin(); it != benchmarks->end(); ++it) {
        run_benchmark(it->obj);
        prettyprint(&*it, args.skip_errors);
        delete it->obj;
    }
    delete benchmarks;
    return 0;
}
