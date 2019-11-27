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
#include "coord3.cu"
#include "grids/coord3-base.cu"

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
	struct {double avg; double median; double min; double max; } total;
	struct {double avg; double median; double min; double max; } kernel;
	bool error; 
} benchmark_result_t;

/** Benchmark operating on some three dimensional grid
 *
 * This is mainly a wrapper around a run() function which the subclass should
 * overwrite. This run() function should operate on the provided input grid and
 * (whose size is given in input_size) and should transform it into something
 * else and store that in output_grid.
 *
 * The value of output_grid() can be verified for correctness against another
 * grid by calling verify()
 *
 * Subclasses should overwrite the run method, which is supposed to operate on
 * the input Grid.
 */
template<typename value_t>
class Benchmark {

	public:

	Benchmark();
	Benchmark(coord3 size);

	Grid<value_t, coord3> *input;
	Grid<value_t, coord3> *output;

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
	 * the computations to be benchmarked inside here. Computations should be
	 * performed on this->input and stored to this->output. */
	virtual void run() = 0;

	/** Executes a certain number of runs of the given benchmark and stores some
	 * metrics in this->results. */
	benchmark_result_t execute();
	
	/** Compares the value in each cell of this->output grid with the given
	 * reference grid and returns true only if all the cells match (up to the
	 * optionally given tolerance). */
	virtual bool verify(Grid<value_t, coord3> *reference, Grid<value_t, coord3> *other=NULL, double tol=1e-8);

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
template<typename value_t>
value_t median(std::vector<value_t> vec);

// IMPLEMENTATIONS

template<typename value_t>
value_t median(std::vector<value_t> vec) {
    if(vec.size() % 2 == 0) {
        std::nth_element(vec.begin(), vec.begin()+vec.size()/2+1, vec.end());
        return (vec[vec.size()/2]+vec[vec.size()/2+1])/2;
    } else {
        std::nth_element(vec.begin(), vec.begin()+vec.size()/2, vec.end());
        return vec[vec.size()/2];
    }
}

template<typename value_t>
Benchmark<value_t>::Benchmark() {}

template<typename value_t>
Benchmark<value_t>::Benchmark(coord3 size) : size(size) {}

template<typename value_t>
void Benchmark<value_t>::post() {
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

template<typename value_t>
dim3 Benchmark<value_t>::numblocks() {
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

template<typename value_t>
dim3 Benchmark<value_t>::numthreads() {
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

template<typename value_t>
benchmark_result_t Benchmark<value_t>::execute() {
	using clock = std::chrono::high_resolution_clock;
	this->setup();
    double avg, min, max;
    double kernel_avg, kernel_min, kernel_max;
    min = DBL_MAX;
    kernel_min = DBL_MAX;
    bool error = false;
    std::vector<double> total_times;
    std::vector<double> kernel_times;
    for(int i=-1; i<this->runs; i++) {
        auto start = clock::now();
        this->pre();
        auto kernel_start = clock::now();
        this->run();
        auto kernel_stop = clock::now();
        this->post();
        error = error || this->error;
        if(i == -1) {
            // First run is untimed, as Cuda recompiles the kernel on first run
            // which would distort our measurements.
            continue;
        }
        auto stop = clock::now();
        double total_time = std::chrono::duration<double>(stop-start).count();
        double kernel_time = std::chrono::duration<double>(kernel_stop-kernel_start).count();
        total_times.push_back(total_time);
        kernel_times.push_back(kernel_time);
        avg += total_time;
        min = std::min(total_time, min);
        max = std::max(total_time, max);
        kernel_avg += kernel_time;
        kernel_min = std::min(kernel_time, min);
        kernel_max = std::max(kernel_time, max);
        if(!this->quiet) {
            fprintf(stderr, "Benchmark %s, Run #%d Results\n", this->name.c_str(), i+1);
            this->output->print();
        }
    }
    this->teardown();
    avg /= runs;
    kernel_avg /= runs;
    double med = median(total_times);
    double kernel_med = median(kernel_times);
    benchmark_result_t res = { .total = { avg, med, min, max },
                               .kernel = { kernel_avg, kernel_med, kernel_min, kernel_max },
							   .error = error };
	this->results = res; // not using temporary variable res gives NVCC compiler segfault ...
	return this->results;
}

template<typename value_t>
bool Benchmark<value_t>::verify(Grid<value_t, coord3> *reference, Grid<value_t, coord3> *other, double tol) {
    if(!other) {
        other = this->output;
    }
    if(other->dimensions != reference->dimensions) {
        return false;
    }
    for(int x=0; x<other->dimensions.x; x++) {
        for(int y=0; y<other->dimensions.y; y++) {
            for(int z=0; z<other->dimensions.z; z++) {
                if(abs((*other)[coord3(x, y, z)] - (*reference)[coord3(x, y, z)]) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

template<typename value_t>
void Benchmark<value_t>::parse_args() {
}

#endif