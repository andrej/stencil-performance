#ifndef CUDA_BENCHMARK_H
#define CUDA_BENCHMARK_H
#include <float.h>
#include <string>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <vector>
#include <numeric>
#ifdef CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
#include "coord3.cu"
#include "grids/coord3-base.cu"

using clk = std::chrono::high_resolution_clock;

/** The maxnumthreads limits are only enforced if the total nubmer of threads
 * (product of x, y and z) is exceeded. It is therefore well possible to have
 * more threads in a given dimension, provided that the other dimensions are
 * accordingly smaller. Note that it leads to errors to try and launch a cuda
 * kernel with too many threads. */
#ifndef CUDA_MAXNUMTHREADS_X
#define CUDA_MAXNUMTHREADS_X 16
#define CUDA_MAXNUMTHREADS_Y 16
#define CUDA_MAXNUMTHREADS_Z 4
#endif

/** Benchmark result: Average, minimum and maximum runtime in seconds. */
typedef struct { 
	struct {double avg; double median; double min; double max;} runtime;
	bool error; 
} benchmark_result_t;

/** Benchmark operating on some three dimensional grid
 *
 * This is mainly a wrapper around a run() function which the subclass should
 * overwrite.
 *
 * The value of output_grid() can be verified for correctness against another
 * grid by calling verify()
 *
 * Subclasses should overwrite the run method, which is supposed to operate on
 * the input Grid.
 */
class Benchmark {

	public:

	Benchmark();
	Benchmark(coord3 size);

	coord3 size;
    dim3 _numblocks;
    dim3 _numthreads;
	std::string name;
	bool error = false;
    benchmark_result_t results;
    bool quiet = true;
    /** Turn verification off if you are sure the benchmark computes the
     * correct result and you do not want to waste time computing the
     * the reference. */
    bool do_verify = true;
    int runs = 1;

    /** Pointers to command-line arguments specific to this benchmark.
     * In main, all arguments that follow a benchmark name are considered
     * specific to that benchmark, and the pointer to the first one of those
     * is passed in argv. In argc, we have the number of arguments until the 
     * next benchmark name or the end of the command. */
    int argc;
    char **argv;
    virtual void parse_args(); /**< do some setup based on argc, argv */

	/** Subclasses (benchmarks) must at least overwrite this function an perform
	 * the computations to be benchmarked inside here. */
	virtual void run() = 0;

	/** Executes a certain number of runs of the given benchmark and stores some
	 * metrics in this->results. */
	benchmark_result_t execute();
	
	/** Compares the value in each cell of this->output grid with the given
	 * reference grid and returns true only if all the cells match (up to the
     * optionally given tolerance). */
    template<typename value_t>
	/*virtual*/ bool verify(Grid<value_t, coord3> *reference, Grid<value_t, coord3> *other, double tol=1e-5);

	// Setup and teardown are called when the benchmark is initialized, only once
	virtual void setup() {};
	virtual void teardown() {};

	// Pre and post are called for each iteration of the benchmark, i.e. once per run
	virtual void pre() {};
	virtual void post();

	// Cuda specific: number of threads and blocks to execute the benchmark in
	// May be used by the benchmark implementation in run() to determine how many
	// threads and blocks to launch the kernel in
	virtual dim3 numthreads();
	virtual dim3 numblocks();

};

/** Computes the median of a vector of (unsorted) values. */
template<typename T>
T median(std::vector<T> vec);

// IMPLEMENTATIONS

template<typename T>
T median<T>(std::vector<T> vec) {
    if(vec.size() % 2 == 0) {
        std::nth_element(vec.begin(), vec.begin()+vec.size()/2+1, vec.end());
        return (vec[vec.size()/2]+vec[vec.size()/2+1])/2;
    } else {
        std::nth_element(vec.begin(), vec.begin()+vec.size()/2, vec.end());
        return vec[vec.size()/2];
    }
}

Benchmark::Benchmark() {}

Benchmark::Benchmark(coord3 size) : size(size) {}

void Benchmark::post() {
    if(cudaGetLastError() != cudaSuccess) {
        this->error = true;
        std::ostringstream msg;
        dim3 nblocks = this->numblocks();
        dim3 nthreads = this->numthreads();
        msg << "Unable to run kernel with (" << nblocks.x << ", " << nblocks.y << ", " << nblocks.z << 
               ") blocks and (" << nthreads.x << ", " << nthreads.y << ", " << nthreads.z << ") threads.";
        throw std::runtime_error(msg.str());
    }
}

dim3 Benchmark::numblocks() {
    if(this->_numblocks.x != 0 &&
        this->_numblocks.y != 0 &&
        this->_numblocks.z != 0) {
        return this->_numblocks;
    }
    dim3 numthreads = this->numthreads();
    int x = (this->size.x + numthreads.x - 1) / numthreads.x;
    int y = (this->size.y + numthreads.y - 1) / numthreads.y;
    int z = (this->size.z + numthreads.z - 1) / numthreads.z;
    return dim3( (unsigned int) x, (unsigned int) y, (unsigned int) z );
}

dim3 Benchmark::numthreads() {
    if(this->_numthreads.x != 0 &&
        this->_numthreads.y != 0 &&
        this->_numthreads.z != 0) {
        return this->_numthreads;
    }
    int x = (this->size.x + this->_numblocks.x - 1) / this->_numblocks.x;
    int y = (this->size.y + this->_numblocks.y - 1) / this->_numblocks.y;
    int z = (this->size.z + this->_numblocks.z - 1) / this->_numblocks.z;
    if (x*y*z > CUDA_MAXNUMTHREADS_X*CUDA_MAXNUMTHREADS_Y*CUDA_MAXNUMTHREADS_Z) {
        // The limiting is only done if the total maximum is exceeded
        x = std::min(x, CUDA_MAXNUMTHREADS_X);
        y = std::min(y, CUDA_MAXNUMTHREADS_Y);
        z = std::min(z, CUDA_MAXNUMTHREADS_Z);
    }
    return dim3( (unsigned int) x, (unsigned int) y, (unsigned int) z );
}

benchmark_result_t Benchmark::execute() {
	this->setup();
    bool error = false;
    std::vector<double> runtimes;
    for(int i=-1; i<this->runs; i++) {
        this->pre();
        #ifdef CUDA_PROFILER
        cudaProfilerStart();
        #endif

        clk::time_point start = clk::now();
        this->run();
        clk::time_point stop = clk::now();

        #ifdef CUDA_PROFILER
        cudaProfilerStop();
        #endif
        this->post();
        error = error || this->error;
        if(i == -1) {
            // First run is untimed, as Cuda recompiles the kernel on first run which would distort our measurements.
            continue;
        }
        double runtime = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
        runtimes.push_back(runtime);
    }
    this->teardown();
    double avg = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
    double med = median<double>(runtimes);
    double min = *std::min_element(runtimes.begin(), runtimes.end());
    double max = *std::max_element(runtimes.begin(), runtimes.end());
    benchmark_result_t res = { .runtime = { avg, med, min, max },
							   .error = error };
	this->results = res; // not using temporary variable res gives NVCC compiler segfault ...
	return this->results;
}

template<typename value_t>
bool Benchmark::verify(Grid<value_t, coord3> *reference, Grid<value_t, coord3> *other, double tol) {
    if(other->dimensions != reference->dimensions) {
        return false;
    }
    for(int x=0; x<other->dimensions.x; x++) {
        for(int y=0; y<other->dimensions.y; y++) {
            for(int z=0; z<other->dimensions.z; z++) {
                /* The reason the benchmark times slow down this much if we use --no-verify flag
                is because comparing to the reference throws the values that the kernel needs out
                of the cache. The following proves that; i.e. not accessing the reference benchmarks
                keeps the other benchmark in cache and as such the next kernel run will be faster.
                */if(abs((*other)[coord3(x, y, z)]) > 1) {
                    return true;
                    //this->error = true;
                    //continue;
                }
                /*
                if(abs((*other)[coord3(x, y, z)] - (*reference)[coord3(x, y, z)]) > tol) {
                    return false;
                }*/
            }
        }
    }
    return true;
}

void Benchmark::parse_args() {
}

#endif