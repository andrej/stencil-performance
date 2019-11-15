#include <stdexcept>

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

};

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::CudaBaseGrid() { }

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::~CudaBaseGrid() {
    this->deallocate();
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::allocate() {
    value_t *ptr;
    if (cudaMallocManaged(&ptr, this->size) != cudaSuccess) {
        throw std::runtime_error("Unable to allocate cuda memory.");
    }
    if (sizeof(value_t) == 4) {
        if (cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) != cudaSuccess) {
            throw std::runtime_error("Unable to set cuda device shared mem config.");
        }
    } else if (sizeof(value_t) == 8) {
        if (cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) != cudaSuccess) {
            throw std::runtime_error("Unable to set cuda device shared mem config.");
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        // Not entirely sure if this synchronization is required.
        throw std::runtime_error("Unable to synchronize cuda devices after memory allocation.");
    }
    this->data = ptr;
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::deallocate() {
    cudaFree(this->data);
}