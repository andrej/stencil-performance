#include <chrono>
#include <stdio.h>

#define CHECK(cmd) do { \
        cudaError_t err = cmd; \
        if(err != cudaSuccess) { \
            printf("%s, line %d:\n%s\n%s\n", __FILE__, __LINE__, \
                cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_LAST() CHECK(cudaGetLastError())

__global__
void kernel(const double *data_in, double *data_out, int N) {
    //const int i = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i = blockIdx.x*blockDim.x + threadIdx.x;
            i < N;
            i += gridDim.x*blockDim.x) {
        data_out[i] = 2*data_in[i];
    }
}

int main(int argc, char** argv) {
    bool do_read = false;
    bool use_unified = false;
    bool do_prefetch = true;
    if(argc > 1 && argv[1][0] == '1') {
        do_read = true;
    }
    if(argc > 2 && argv[2][0] == '1') {
        use_unified = true;
    }
    if(argc > 3 && argv[3][0] == '0') {
        do_prefetch = false;
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
        //CHECK( cudaMallocHost(&data_in, N*sizeof(double)) );
        //CHECK( cudaMallocHost(&data_out, N*sizeof(double)) );
        data_in = (double*) malloc(N*sizeof(double));
        data_out = (double*) malloc(N*sizeof(double));
        CHECK( cudaMalloc(&data_in_dev, N*sizeof(double)) );
        CHECK( cudaMalloc(&data_out_dev, N*sizeof(double)) );
    } else { // unified memory
        CHECK( cudaMallocManaged(&data_in, N*sizeof(double)) );
        CHECK( cudaMallocManaged(&data_out, N*sizeof(double)) );
        data_in_dev = data_in;
        data_out_dev = data_out;
    }
    for(int i = 0; i < N; i++) {
        data_in[i] = i;
    }
    for(int i=0; i<5; i++) {
        // COPY HOST -> DEVICE
        if(i == 0 || do_read) {
            if(!use_unified) {
                CHECK( cudaMemcpy(data_in_dev, data_in, N*sizeof(double), cudaMemcpyHostToDevice) );
            } else if(do_prefetch) {
                CHECK( cudaMemPrefetchAsync(data_in_dev, N*sizeof(double), device) );
                CHECK( cudaMemPrefetchAsync(data_out_dev, N*sizeof(double), device) );
                CHECK( cudaDeviceSynchronize() );
            }
        }

        // EXECUTE KERNEL
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        kernel<<<1, 1024>>>(data_in_dev, data_out_dev, N);
        CHECK_LAST();
        CHECK( cudaDeviceSynchronize() );
        std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

        if(do_read) {
            // COPY DEVICE -> HOST
            if(!use_unified) {
                CHECK( cudaMemcpy(data_out, data_out_dev, N*sizeof(double), cudaMemcpyDeviceToHost) );
            } else if(do_prefetch) {
                CHECK( cudaMemPrefetchAsync(data_in_dev, N*sizeof(double), cudaCpuDeviceId) );
                CHECK( cudaMemPrefetchAsync(data_out_dev, N*sizeof(double), cudaCpuDeviceId) );
                CHECK( cudaDeviceSynchronize() );
            }
            // MANIPULATE/READ
            for(int i = 0; i < N; i++) {
                data_in[i] = data_out[i];
            }
        }

        printf("%d nanoseconds\n", duration.count());
    }
    if(!use_unified) {
        free(data_in);
        free(data_out);
    }
    CHECK( cudaFree(data_in_dev) );
    CHECK( cudaFree(data_out_dev) );
}