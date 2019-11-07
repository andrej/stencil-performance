#include <stdexcept>

/** This struct is used to pass grid information to the kernel (because we
 * cannot use the C++ classes inside the kernel). Pass this struct
 * to the appropriate kernel macros.
 */
 template<typename value_t>
 struct CudaGridInfo {
     value_t *data; /**< Pointer to the allocated CUDA memory, i.e. this->data */
     coord3 dimensions;
     coord3 strides;
 };

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
    CudaBaseGrid(coord_t dimensions, size_t size);
    ~CudaBaseGrid();    

    virtual void allocate();
    
    virtual void deallocate();

    virtual CudaGridInfo<value_t> get_gridinfo(); /**< Return grid info struct required by kernel macros. */
};

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::CudaBaseGrid() { }

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::CudaBaseGrid(coord_t dimensions, size_t size) {
    this->dimensions = dimensions;
    this->size = size;
    this->allocate();
}

template<typename value_t, typename coord_t>
CudaBaseGrid<value_t, coord_t>::~CudaBaseGrid() {
    this->deallocate();
}

template<typename value_t, typename coord_t>
CudaGridInfo<value_t> CudaBaseGrid<value_t, coord_t>::get_gridinfo() {
    return { .data = this->data,
             .dimensions = this->dimensions,
             .strides = coord3() };
}

template<typename value_t, typename coord_t>
void CudaBaseGrid<value_t, coord_t>::allocate() {
    value_t *ptr;
    if (cudaMallocManaged(&ptr, this->size*sizeof(value_t)) != cudaSuccess) {
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