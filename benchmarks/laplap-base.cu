#ifndef LAPLAP_BASE_BENCHMARK_H
#define LAPLAP_BASE_BENCHMARK_H
#include <random>
#include "benchmarks/benchmark.cu"
#include "grids/regular.cu"

/** Lap-Lap */
template<typename value_t>
class LapLapBaseBenchmark : public Benchmark {

    public:

    // The padding option currently only applies to regular grids
    LapLapBaseBenchmark(coord3 size);
 
    CudaBaseGrid<value_t, coord3> *input = NULL;
    CudaBaseGrid<value_t, coord3> *output = NULL;

    // Intermediate result (used only by non-fused kernel)
    CudaBaseGrid<value_t, coord3> *intermediate = NULL;

    void setup();
    void teardown();
    void pre();
    void post();
    bool verify(double tol=1e-5);

};

template<typename value_t>
LapLapBaseBenchmark<value_t>::LapLapBaseBenchmark(coord3 size) :
Benchmark(size) {}

template<typename value_t>
void LapLapBaseBenchmark<value_t>::setup() {
    std::default_random_engine gen;
    std::uniform_real_distribution<value_t> dist(-1.0, +1.0);
    Coord3BaseGrid<value_t> *in = dynamic_cast<Coord3BaseGrid<value_t> *>(this->input);
    for(int i = 0; i < this->size.x; i++) {
        for(int j = 0; j < this->size.y; j++) {
            for(int k = 0; k < this->size.z; k++) {
                coord3 p(i, j, k);
                in->set(p, dist(gen));
            }
        }
    }
    this->Benchmark::setup();
}

template<typename value_t>
void LapLapBaseBenchmark<value_t>::teardown() {
    delete this->input;
    delete this->output;
    this->Benchmark::teardown();
}

template<typename value_t>
void LapLapBaseBenchmark<value_t>::pre() {
    this->input->prefetchToDevice();
    this->output->prefetchToDevice();
    if(this->intermediate) {
        this->intermediate->prefetchToDevice();
    }
}

template<typename value_t>
void LapLapBaseBenchmark<value_t>::post() {
    this->output->prefetchToHost();
}

template<typename value_t>
bool LapLapBaseBenchmark<value_t>::verify(double tol) {
    // Calculate reference
    auto lap = [](Coord3BaseGrid<value_t> *in, coord3 p){
        return 4 * (*in)[p] - (*in)[coord3(p.x - 1, p.y, p.z)]
                            - (*in)[coord3(p.x + 1, p.y, p.z)]
                            - (*in)[coord3(p.x, p.y - 1, p.z)]
                            - (*in)[coord3(p.x, p.y + 1, p.z)];
    };
    
    // First iteration
    coord3 halo1(1, 1, 0);
    Coord3BaseGrid<value_t> *in = dynamic_cast<Coord3BaseGrid<value_t> *>(this->input);
    CudaRegularGrid3D<value_t> *lap1 = new CudaRegularGrid3D<value_t>(this->size-halo1, halo1);
    for(int i = 0; i < lap1->size.x; i++) {
        for(int j = 0; j < lap1->size.y; j++) {
            for(int k = 0; k < lap1->size.z; k++) {
                coord3 p(i, j, k);
                lap1->set(p, lap(in, p));
            }
        }
    }

    // Second iteration
    coord3 halo2(2, 2, 0);
    CudaRegularGrid3D<value_t> *lap2 = new CudaRegularGrid3D<value_t>(this->size-halo2, halo2);
    for(int i = 0; i < lap2->size.x; i++) {
        for(int j = 0; j < lap2->size.y; j++) {
            for(int k = 0; k < lap2->size.z; k++) {
                coord3 p(i, j, k);
                lap2->set(p, lap(lap1, p));
            }
        }
    }
    // Compare
    bool ret = this->output->compare(lap2, tol);
    if(!ret && !this->quiet) {
        fprintf(stderr, "Reference ---------------------------------------\n");
        lap2->print();
        fprintf(stderr, "Output ------------------------------------------\n");
        this->output->print();
    }
    return ret;
}

#endif