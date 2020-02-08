/** Coalescing of data with a halo / Alignment
 * This experiment investigates how data needs to be aligned such that all memory accesses coalesce.
 * The issue is with the bounds checks. 32-byte loads coalesce into one vector instruction. If the
 * halo is aligned, but not the inner values, then these accesses do not coalesce well: part of the
 * vector load instruction is then wasted on predicated-off threads (in the halo).
 */
#include <stdio.h>

__global__
void knl_interleaved_halo(double *in, double *out, int sx, int sy, int sz, int dx, int dy, int dz, int halo) {
    int x = blockIdx.x*blockDim.x + threadIdx.x + halo;
    int y = blockIdx.y*blockDim.y + threadIdx.y + halo;
    int z = blockIdx.z*blockDim.z + threadIdx.z + halo;
    int idx = x*sx + y*sy + z*sz;
    if(x >= dx-halo || y >= dy-halo || z >= dz-halo) {
        return;
    }
    out[idx] = 2*in[idx];
}

__global__
void knl_halo_at_start(double *in, double *out, int sz, int sxy, int lxy) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int idx = x + y*blockDim.x*gridDim.x + z*sxy;
    if(x + y*blockDim.x*gridDim.x >= lxy || z >= sz) {
        return;
    }
    out[idx] = 2*in[idx];
}

int roundup(int v, int multiple) {
    if(v % multiple == 0) {
        return v;
    }
    return v + multiple - v % multiple;
}

int main(int argc, char** argv) {
    int halo = 3;
    int x = 256 + 2*halo;
    int y = 256 + 2*halo;
    int z = 64 + 2*halo;
    double *in;
    double *out;

    // values for interleaved halo
    // start_offs: offset such that first regular element is first in 64byte-block.
    // otherwise, e.g. for halo=2, first element is at i=2, i.e. first two vector elements of load instruction UNUSED
    int align = 128 / sizeof(double);
    int stride_y = roundup(x, align);
    int stride_z = roundup(stride_y*y, align);
    int padding = align - (halo+halo*stride_y+halo*stride_z) % align;
    printf("%d, %d, %d, %d\n", 1, stride_y, stride_z, padding);

    // values for halo at front
    int halo_len = 2*halo*x + 2*halo*(y-2*halo);

    cudaMallocManaged(&in, (stride_z*z+padding)*sizeof(double));
    cudaMallocManaged(&out, (x*y*z+padding)*sizeof(double));
    for(int run=0; run<2; run++) {
        knl_interleaved_halo<<<dim3(8, 32, 16), dim3(32, 1, 4)>>>(in+padding, out+padding, 1, stride_y, stride_z, x, y, z, halo);
        cudaDeviceSynchronize();
        knl_halo_at_start<<<dim3(8, 32, 16), dim3(32, 1, 4)>>>(in+halo*x*y, out+halo*x*y, z-2*halo, x*y, x*y-halo_len);
        cudaDeviceSynchronize();
    }
}