#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdexcept>

/** Round up to the closest greater multiple. */
int roundup(int value, int multiple) {
    if(value % multiple == 0) {
        return value;
    }
    return value + multiple - value % multiple;
}

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

#define CUDA_CHECK(cmd) do { \
        cudaError_t err = cmd; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "%s, line %d: %s. %s.\n", __FILE__, __LINE__, \
                cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define CUDA_THROW(cmd) do { \
        cudaError_t err = cmd; \
        if(err != cudaSuccess) { \
            char msg[128] = ""; \
            snprintf(msg, 128, "%s, line %d: %s. %s.", \
                     __FILE__, __LINE__, \
                     cudaGetErrorName(err), cudaGetErrorString(err)); \
            throw std::runtime_error(msg); \
        } \
    } while(0)

#define CUDA_THROW_LAST() CUDA_THROW(cudaGetLastError())

#endif