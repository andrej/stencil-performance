CXX=nvcc
CPPFLAGS=-I./ -arch sm_60 -DCUDA_PROFILER -DNDEBUG #-lineinfo #--ptxas-options=-v
CPPDEBUGFLAGS=-g -G -DHDIFF_DEBUG
SRCS=$(wildcard *.cu)
SRCS_BENCHMARKS=$(wildcard benchmarks/*.cu)
SRCS_GRIDS=$(wildcard grids/*.cu)
SRCS_KERNELS=$(wildcard kernels/*.cu)

gridbenchmark: $(SRCS) $(SRCS_BENCHMARKS) $(SRCS_GRIDS) $(SRCS_KERNELS)
	$(CXX) $(CPPFLAGS) $(CPPDEBUGFLAGS) $(ARGS) -o gridbenchmark main.cu

nodebug: $(SRCS) $(SRCS_BENCHMARKS) $(SRCS_GRIDS) $(SRCS_KERNELS)
	$(CXX) $(CPPFLAGS) $(ARGS) -O3 -o gridbenchmark main.cu



clean:
	rm ./gridbenchmark
