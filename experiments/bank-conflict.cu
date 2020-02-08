/** Shared memory bank conflicts
 * This experiment invstigates how accesses to shared memory should be strided to avoid
 * bank conflicts.
 *
 * Shared memory can be accessed by threads simultaneously if the addresses are in different banks.
 * Otherwise the accesses are executed sequentially. The Tesla V100 has compute capability 7.0, here
 * we have 32 banks where each consecutive 32-bit word is mapped onto successive banks.
 *
 * Command:
 * nvprof -e shared_ld_bank_conflict,shared_st_bank_conflict  ./bank-conflict 13
 *
 * For values coprime to 32 -> few bank conflicts   ~    20`000
 * For multiples of 32 -> many conflicts            ~ 3`600`000
 * For 16 -> about half conflicts of 32             ~ 1`800`000
 * For 8 -> about a quarter of conflicts of 32      ~    90`000
 */
#include <stdio.h>

__global__
void kernel(int *input, int *output, int smem_stride, int w, int h) {
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    const int k = threadIdx.z + blockIdx.z*blockDim.z;
    const int idx = i + j*w + k*w*h;

    // sizeof(int) = 4 byte = 32-bit-word -> one int per shared memory bank
    extern __shared__ int smem[];
    const int local_idx = (threadIdx.x + threadIdx.y * blockDim.x) * smem_stride;
    const bool is_first = k % 8 == 0;

    // threads at z = 0 store, all others just read
    int value;
    if(is_first) {
        value = input[idx];
        smem[local_idx] = value;
    } 
    __syncthreads();
    if(!is_first) {
        value = smem[local_idx];
    }

    output[idx] = value;
}

int main(int argc, char **argv) {
    // takes as argument the smem stride
    int smem_stride = 1;
    if(argc > 1) {
        sscanf(argv[1], "%d", &smem_stride);
    }

    // data size
    int x = 256;
    int y = 256;
    int z = 64;
    dim3 threads(32, 1, 8);
    dim3 blocks(x/threads.x, y/threads.y, z/threads.z);
    int global_size = x*y*z*sizeof(int);
    int smem_size = 32*1*smem_stride*sizeof(int);

    // allocate some data
    int *input;
    int *output;
    cudaMallocManaged(&input, global_size*sizeof(int));
    //cudaMemset(input, 7, global_size*sizeof(int));
    cudaMallocManaged(&output, global_size*sizeof(int));
    cudaDeviceSynchronize();

    // run kernels
    for(int i = 0; i < 20; i++) {
        kernel<<<dim3(8, 256, 8), threads, smem_size>>>(input, output, smem_stride, x, y);
        cudaDeviceSynchronize();
    }

    printf("input[0] = %d\noutput[0] = %d\n", input[0], output[0]);

    // cleanup
    cudaFree(input);
    cudaFree(output);
}