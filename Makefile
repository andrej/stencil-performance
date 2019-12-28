CXX=nvcc
CPPFLAGS=-I./ -arch sm_60 -O3 -DHDIFF_NO_GRIDSTRIDE -DCUDA_PROFILER -DNDEBUG -lineinfo --ptxas-options=-v
CPPDEBUGFLAGS=-g -G -DHDIFF_DEBUG
SRCS=$(wildcard *.cu)
SRCS_BENCHMARKS=$(wildcard benchmarks/*.cu)
SRCS_GRIDS=$(wildcard grids/*.cu)
OBJS=$(SRCS:%.cu=%.o)
OBJS_BENCHMARKS=benchmarks/benchmark.o benchmarks/hdiff-ref.o #$(SRCS_BENCHMARKS:benchmarks/%.cu=benchmarks/%.o)
OBJS_GRIDS=grids/grid.o grids/coord3-base.o grids/regular.o grids/unstructured.o grids/cuda-regular.o grids/cuda-unstructured.o #$(SRCS_GRIDS:grids/%.cu=grids/%.o)

gridbenchmark: $(SRCS) $(SRCS_BENCHMARKS) $(SRCS_GRIDS)
	$(CXX) $(CPPFLAGS) $(CPPDEBUGFLAGS) -o gridbenchmark main.cu

nodebug: $(SRCS) $(SRCS_BENCHMARKS) $(SRCS_GRIDS)
	$(CXX) $(CPPFLAGS) -o gridbenchmark main.cu

#gridbenchmark: $(OBJS) $(OBJS_BENCHMARKS) $(OBJS_GRIDS) 
#	$(CXX) $(CPPFLAGS) -o gridbenchmark $(OBJS_GRIDS) $(OBJS_BENCHMARKS) $(OBJS) 

#%.o: %.cu
#	$(CXX) $(CPPFLAGS) -c -o $@ $<

#benchmarks/%.o: benchmarks/%.cu
#	$(CXX) $(CPPFLAGS) -c -o $@ $<

#grids/%.o: grids/%.cu
#	$(CXX) $(CPPFLAGS) -c -o $@ $<

clean:
	rm ./gridbenchmark
