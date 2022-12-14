CXX=nvcc
CPPFLAGS=-I./ -arch sm_60 -DCUDA_PROFILER -DNDEBUG -L./boost/build/lib/ -l:libboost_serialization.a #--ptxas-options=-v #-lineinfo #
CPPDEBUGFLAGS=-g -G -DHDIFF_DEBUG
SRCS=$(wildcard *.cu)
SRCS_BENCHMARKS=$(wildcard benchmarks/*.cu)
SRCS_GRIDS=$(wildcard grids/*.cu)
SRCS_KERNELS=$(wildcard kernels/*.cu)

gridbenchmark: $(SRCS) $(SRCS_BENCHMARKS) $(SRCS_GRIDS) $(SRCS_KERNELS) flush
	$(CXX) $(CPPFLAGS) $(CPPDEBUGFLAGS) $(ARGS) -o gridbenchmark main.cu

nodebug: $(SRCS) $(SRCS_BENCHMARKS) $(SRCS_GRIDS) $(SRCS_KERNELS) flush
	$(CXX) $(CPPFLAGS) $(ARGS) -O3 -o gridbenchmark main.cu

clean:
	rm ./gridbenchmark

flush:
	rm -r ./.grid-cache
	mkdir ./.grid-cache