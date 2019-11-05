# GridBenchmark

Utilities for executing benchmarks on grids.

## Structure

There are two main components: Benchmarks and Grids. Benchmarks operate on 
grids. Benchmarks provide a run() method and are timed.

Grids are completely independent of Benchmarks. They provide a means for
getting and setting values in cells with some type of neighborship relations.
Different types of grids are used for different types of memory layouts and 
neighborhsip relations.

### Files

 - main.cu: Entry point into the program that contains all the glue code.
   Reads command line arguments, initializes the appropriate Benchmark(s),
   executes them and prints the results in CSV-format to stdout. Some progress
   information is also printed to stderr. Piping the output of the
   gridbenchmark binary into a file should produce a valid CSV file.

 - coord3.cu: Some utilities for working with 3D-coordinates.

 - benchmarks/: Contains base class for benchmarks as well as several concrete
   benchmarks. A benchmark must provide a run method that will be benchmarked.

   - benchmark.h: Abstract base class. Provides means for timing the execution
     of the run method as well as a function for comparing the output of a
     benchmark to some other grid (for verification, called verify()).
   
   - hdiff-ref.cu: Reference implementation of horizontal diffusion kernel,
     runs on the CPU and produces the verification output as a regular grid.

 - grids/: Contains implementations of different types of grids. A grid is a
   data structure that supports getting and setting values at certain
   coordinates and has some notion of neighborship relations. The base class
   is held very abstract to make it possible to accomodate for different types
   of coordinate systems, neighborship relations, memory layouts and cell value
   types.

    - grid.cu: Abstract base class as described above

    - coord3-base.cu: Abstract base class for grids which use 3D-coordinates;
      Provides functionality that is common to all 3D-grids
    
    - cuda-base.cu: Abstract base class for grids which are to be laid out in
      Cuda memory. Essentially replaces the default Grid allocation methods
      with the Cuda ones. Additionally provides a struct to pass information
      about Grids to kernels, as classes cannot be accessed within Cuda code.
    
    - regular.cu: A regular grid, wherein each cell has six direct neighbors,
      laid out in memory in the most obvious way.
    
    - unstructured.cu: A regular grid in Z-direction, but irregular in X and Y.
      It supports arbitrary neighborship relations for any cell, but right now
      it is assumed that these relations are the same among all Z-levels
      (FIXME either specify this or make it more flexible)
    
    - cuda-*.cu: Cuda versions of regular and unstructured grids.


## Todo

 - Fix calculations of numthreads/numblocks / make it more flexible via the
   command line

 - Unstructured grids: Fix wether neighborship relations are the same among all
   Z-levels. In that case we can save some storage.

 - Make main.cu more modular, so it is easier to add and remove Benchmarks

 - Improve argument parsing of main.cu