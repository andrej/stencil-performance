#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include <chrono>

#include "benchmarks/benchmark.cu"
#include "benchmarks/hdiff-ref.cu"
#include "benchmarks/hdiff-cpu-unstr.cu"
#include "benchmarks/hdiff-cuda.cu"
#include "benchmarks/hdiff-cuda-seq.cu"
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
              hdiff_cuda_sequential,
              hdiff_cuda_unstr_naive,
              hdiff_cuda_unstr_idxvars,
              unspecified} 
benchmark_type_t;

/** Struct describing the benchmark to be run (default arguments) */
struct benchmark_info_t {
    benchmark_type_t type = unspecified;
    int N = 1024;
    int M = 1024;
    int L = 80;
    int runs = 20;
    int numblocks_N = 32;
    int numblocks_M = 8;
    int numblocks_L = 8;
    bool print = false;
};

/** Populates the list of available benchmarks. */
void get_benchmark_identifiers(std::map<std::string, benchmark_type_t> *ret) {
    (*ret)["all"] = all_benchs;
    (*ret)["hdiff-ref"] = hdiff_ref;
    (*ret)["hdiff-ref-unstr"] = hdiff_ref_unstr;
    (*ret)["hdiff-cuda-regular"] = hdiff_cuda_regular;
    (*ret)["hdiff-cuda-seq"] = hdiff_cuda_sequential;
    (*ret)["hdiff-cuda-unstr"] = hdiff_cuda_unstr_naive;
    (*ret)["hdiff-cuda-unstr-idxvars"] = hdiff_cuda_unstr_idxvars;
}

/** Create the benchmark class for one of the available types. */
Benchmark<double> *create_benchmark(benchmark_type_t type, int N, int M, int L, int numblocks_N, int numblocks_M, int numblocks_L) {
    Benchmark<double> *ret = NULL;
    coord3 size = coord3(N, M, L);
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
        case hdiff_cuda_sequential:
        ret = new HdiffCudaSequentialBenchmark(size);
        break;
        case hdiff_cuda_unstr_naive:
        ret = new HdiffCudaUnstrBenchmark(size);
        break;
        case hdiff_cuda_unstr_idxvars:
        ret = new HdiffCudaUnstrBenchmark(size, HdiffCudaUnstr::Idxvars);
        break;
    }
    ret->_numblocks = dim3(numblocks_N, numblocks_M, numblocks_L);
    return ret;
}

/** Very simple arg parser; if multiple valid arguments for the same setting
 * are passed, the last one wins. */
benchmark_info_t parse_args(int argc, char** argv) {
    benchmark_info_t ret;
    std::map<std::string, benchmark_type_t> benchmark_identifiers;
    get_benchmark_identifiers(&benchmark_identifiers);
    for(int i=1; i<argc; i++) {
        std::string arg = std::string(argv[i]);
        if(benchmark_identifiers.count(arg) > 0) {
            ret.type = benchmark_identifiers[arg];
        } else if(arg == "--size" && i+1 < argc) {
            sscanf(argv[i+1], "%d,%d,%d", &ret.N, &ret.M, &ret.L);
            i += 1;
        } else if(arg == "--runs" && i+1 < argc) {
            sscanf(argv[i+1], "%d", &ret.runs);
            i += 1;
        } else if(arg == "--numblocks" && i+1 < argc) {
            sscanf(argv[i+1], "%d,%d,%d", &ret.numblocks_N, &ret.numblocks_M, &ret.numblocks_L);
            i += 1;
        } else if(arg == "--print") {
            ret.print = true;
        } else {
            fprintf(stderr, "Unrecognized argument %s.\n", arg.c_str());
            exit(1);
        }
    }
    return ret;
}

/** Print usage notice and exit. */
void usage(int argc, char** argv) {
    fprintf(stderr,
"Usage: %s [--size N,M,L] [--numblocks N,M,L] [--print] BENCHMARK \n \
 Benchmarks: all, ...\n", argv[0]);
    exit(1);
}

/** Pretty print the results in a table (format is CSV-compatible, can be exported into Excel). */
void prettyprint(benchmark_list_t *benchmarks, bool header=true) {
    if(header) {
        printf("Benchmark         , AvgTot, MinTot, MaxTot, AvgKnl, MinKnl, MaxKnl\n");
    }
    for(auto it=benchmarks->begin(); it != benchmarks->end(); ++it) {
        Benchmark<double> *bench = *it;
        printf("%-18s, % 2.3f, % 2.3f, % 2.3f, % 2.3f, % 2.3f, % 2.3f%s\n",
               bench->name.c_str(),
               bench->results.total.avg,
               bench->results.total.min,
               bench->results.total.max,
               bench->results.kernel.avg,
               bench->results.kernel.min,
               bench->results.kernel.max,
               (bench->error ? ", (Error)" : ""));
    }
}

/** Create the benchmark described in bench_info, execute it and then return
 * its performance metrics. */
Benchmark<double> *run_benchmark(benchmark_info_t bench_info, bool quiet=false) {
    Benchmark<double> *bench = create_benchmark(bench_info.type,
                                        bench_info.N, bench_info.M, bench_info.L,
                                        bench_info.numblocks_N, bench_info.numblocks_M, bench_info.numblocks_L);
    if(!quiet) {
        fprintf(stderr, "Running Benchmark,    N,    M,    L, blocksize_N, blocksize_M, blocksize_L, runs\n");
        fprintf(stderr, "%17s, %4d, %4d, %4d, %11d, %11d, %11d, %4d\n", bench->name.c_str(), bench_info.N, bench_info.M, bench_info.L, bench_info.numblocks_N, bench_info.numblocks_M, bench_info.numblocks_L, bench_info.runs);
    }
    bench->execute(bench_info.runs, !bench_info.print || quiet);
    return bench;
}

/** Main */
int main(int argc, char** argv) {
    benchmark_info_t bench = parse_args(argc, argv);
    if(bench.type == unspecified) {
        usage(argc, argv);
        return 1;
    }
    benchmark_list_t results;
    if(bench.type == all_benchs) {
        for(int it = hdiff_ref; it < unspecified; it++) {
            benchmark_type_t type = (benchmark_type_t) it;
            bench.type = type;
            Benchmark<double> *res = run_benchmark(bench);
            results.push_back(res);
        }
    } else {
        Benchmark<double> *res = run_benchmark(bench);
        results.push_back(res);
    }
    fprintf(stderr, "\nResults:\n");
    prettyprint(&results);
    return 0;
}
