#ifndef LAPLAP_BASE_BENCHMARK_H
#define LAPLAP_BASE_BENCHMARK_H
#include <random>
#include <stdlib.h>
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
    
    // size w.o. halo
    coord3 inner_size;

    void setup();
    void teardown();
    void pre();
    void post();
    bool verify(double tol=1e-5);

};

template<typename value_t>
LapLapBaseBenchmark<value_t>::LapLapBaseBenchmark(coord3 size) :
Benchmark(size) {
    this->inner_size = this->size - 2*coord3(2, 2, 0);
}

template<typename value_t>
void LapLapBaseBenchmark<value_t>::setup() {
    std::default_random_engine gen;
    std::uniform_real_distribution<value_t> dist(-1.0, +1.0);
    Coord3BaseGrid<value_t> *in = dynamic_cast<Coord3BaseGrid<value_t> *>(this->input);
    for(int i = -in->halo.x; i < in->dimensions.x+in->halo.x; i++) {
        for(int j = -in->halo.y; j < in->dimensions.y+in->halo.y; j++) {
            for(int k = -in->halo.z; k < in->dimensions.z+in->halo.z; k++) {
                coord3 p(i, j, k);
                in->set(p, dist(gen));
                in->set(p, 100*abs(i) + 10*abs(j) + 1*abs(k));
            }
        }
    }
    this->output->fill(0.0);
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
    coord3 halo2(2, 2, 0);
    Coord3BaseGrid<value_t> *in = dynamic_cast<Coord3BaseGrid<value_t> *>(this->input);
    CudaRegularGrid3D<value_t> *lap1 = CudaRegularGrid3D<value_t>::create(this->inner_size, halo2);
    for(int i = -halo1.x; i < lap1->dimensions.x + halo1.x; i++) {
        for(int j = -halo1.y; j < lap1->dimensions.y + halo1.y; j++) {
            for(int k = -halo1.z; k < lap1->dimensions.z + halo1.z; k++) {
                coord3 p(i, j, k);
                lap1->set(p, lap(in, p));
            }
        }
    }

    // Second iteration
    CudaRegularGrid3D<value_t> *lap2 = CudaRegularGrid3D<value_t>::create(this->inner_size, halo2);
    for(int i = 0; i < lap2->dimensions.x; i++) {
        for(int j = 0; j < lap2->dimensions.y; j++) {
            for(int k = 0; k < lap2->dimensions.z; k++) {
                coord3 p(i, j, k);
                lap2->set(p, lap(lap1, p));
            }
        }
    }
    // Compare
    bool ret = this->output->compare(lap2, tol);
    if(!ret && !this->quiet) {
        fprintf(stderr, "Input -------------------------------------------\n");
        this->input->print();
        fprintf(stderr, "Reference Lap1 ----------------------------------\n");
        lap1->print();
        fprintf(stderr, "Reference ---------------------------------------\n");
        lap2->print();
        fprintf(stderr, "Output ------------------------------------------\n");
        this->output->print();
    }
    return ret;
}

#endif