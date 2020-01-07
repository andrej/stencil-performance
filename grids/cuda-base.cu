#ifndef CUDA_BASE_H
#define CUDA_BASE_H

#include <stdexcept>
#include "cuda-util.cu"

/** Cuda Base Grid
 *
 * Provides allocation of memory and a struct to pass information about a grid
 * to kernels. Abstract basis for grids implemented in Cuda.
 */
template<typename value_t, typename coord_t>
class CudaBaseGrid : 
virtual public Grid<value_t, coord_t>
{
    public:
    
    CudaBaseGrid();
    ~CudaBaseGrid();    

    virtual void allocate();
    
    virtual void deallocate();

    /** A *blocking* call that synchronizes the current state of the grid to
     * the device or host. Call before host/device accesses to prevent page
     * faults inside the kernel. */
    void prefetch(int device = -1);
    void prefetchToDevice();
    void prefetchToHost();

    void setSmemBankSize(int sz=-1);

};

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::CudaBaseGrid() { }

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::~CudaBaseGrid() {
    if(this->data) {
        this->deallocate();
    }
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::allocate() {
    value_t *ptr;
    CUDA_THROW( cudaMallocManaged(&ptr, this->size) );
    this->setSmemBankSize();
    this->data = ptr;
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::setSmemBankSize(int sz) {
    if(sz == -1) {
        sz = sizeof(value_t);
    }
    if (sz == 4) {
        CUDA_THROW( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );
    } else if(sz == 8) {
        CUDA_THROW( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
    }
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::deallocate() {
    cudaFree(this->data);
    this->data = NULL;
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::prefetch(int device) {
    if(device == -1) {
        CUDA_THROW( cudaGetDevice(&device) );
    }
    CUDA_THROW( cudaMemPrefetchAsync(this->data, this->size, device, 0) );
    CUDA_THROW( cudaDeviceSynchronize() );
}


template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::prefetchToDevice() {
    this->prefetch();
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::prefetchToHost() {
    this->prefetch(cudaCpuDeviceId);
}

#endif