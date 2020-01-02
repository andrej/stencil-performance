#include <chrono>
#include <stdio.h>

__global__
void kernel(double *data_in, double *data_out) {
    for(int i = blockIdx.x*blockDim.x + threadIdx.x;
            i < gridDim.x*blockDim.x;
            i += gridDim.x) {
        data_out[i] = 2*data_in[i];
    }
}

int main(int argc, char** argv) {
    bool do_read = false;
    bool use_unified = false;
    if(argc > 1 && argv[1][0] == '1') {
        do_read = true;
    }
    if(argc > 2 && argv[2][0] == '1') {
        use_unified = true;
    }
    if(do_read) {
        printf("do read, ");
    } else {
        printf("no read, ");
    }
    if(use_unified) {
        printf("unified memory\n");
    } else {
        printf("manual memory allocation and copying\n");
    }
    int device = -1;
    cudaGetDevice(&device);
    int N=256*256*64; //1<<16;
    double *data_in;
    double *data_out;
    double *data_in_dev;
    double *data_out_dev;
    if(!use_unified) { // manual
        cudaMallocHost(&data_in, N*sizeof(double));
        cudaMallocHost(&data_out, N*sizeof(double));
        cudaMalloc(&data_in_dev, N*sizeof(double));
        cudaMalloc(&data_out_dev, N*sizeof(double));
    } else { // unified memory
        cudaMallocManaged(&data_in, N*sizeof(double));
        cudaMallocManaged(&data_out, N*sizeof(double));
        data_in_dev = data_in;
        data_out_dev = data_out;
        cudaMemPrefetchAsync(data_in, N*sizeof(double), device, NULL);
        cudaMemPrefetchAsync(data_out, N*sizeof(double), device, NULL);
    }
    for(int i = 0; i < N; i++) {
        data_in[i] = i;
    }
    if(!use_unified) {
        cudaMemcpy(data_in_dev, data_in, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(data_out_dev, data_out, N*sizeof(double), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<5; i++) {
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        kernel<<<1024, 1, 1>>>(data_in_dev, data_out_dev);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        if(do_read) {
            if(!use_unified) {
                //cudaMemcpy(data_in, data_in_dev, N*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(data_out, data_out_dev, N*sizeof(double), cudaMemcpyDeviceToHost);
            }
            for(int i = 0; i < N; i++) {
                data_in[i] = data_out[i];
            }
            if(!use_unified) {
                cudaMemcpy(data_in_dev, data_in, N*sizeof(double), cudaMemcpyHostToDevice);
                //cudaMemcpy(data_out_dev, data_out, N*sizeof(double), cudaMemcpyHostToDevice);        
            } else {
                cudaMemPrefetchAsync(data_in, N*sizeof(double), device, NULL);
                //cudaMemPrefetchAsync(data_out, N*sizeof(double), device, NULL);
            }
        }
        if(i == 0) {
            continue; // this was the warmup round
        }
        printf("%d nanoseconds\n", duration.count());
    }
    cudaFree(data_in);
    cudaFree(data_out);
    if(!use_unified) {
        cudaFree(data_in_dev);
        cudaFree(data_out_dev);
    }
}