#include <stdio.h>
#include "../../commons/h5bench_util.h"
#include "cuda_kernel.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void kernel(float *d_x, float *d_y, float *d_z, float *d_px, float *d_py, float *d_pz,
  int *d_id_1, float *d_id_2, long particle_cnt, volatile int *kernel_flag) {

  int32_t total_threads = blockDim.x * gridDim.x;
  int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for( int32_t i = gid; i < particle_cnt; i+=total_threads ) {
    d_x[i] += 0.2;
    d_y[i] += 0.2;
    d_z[i] += 0.2;
    d_px[i] += 0.2;
    d_py[i] += 0.2;
    d_pz[i] += 0.2;
    d_id_1[i] += 1;
    d_id_2[i] += 0.2;
  }

  // wait for cpu to tell kernel to finish, or keep running
  while(!*kernel_flag);
}

void kernel_call(data_contig_md *data, volatile int *kernel_flag, cudaStream_t stream_id) {
  dim3 threadsperblock = 128;
  dim3 blockspergrid = 80;

  //dim3 blockspergrid = dim3(ceil((double)numparticles/256), 1, 1);
  //printf("cuda kernel launch with %d blocks of %d threads\n", blockspergrid, threadsperblock);
  // kernel<<<threadsperblock, blockspergrid, 0, stream_id>>>(data, kernel_flag);

  // kernel<<<threadsperblock, blockspergrid, 0, stream_id>>>(
    // data->d_x, data->d_y, data->d_z, data->d_px, data->d_py, data->d_pz, data->d_id_1, data->d_id_2, data->particle_cnt,
    // kernel_flag);

  // todo: false postive cufile error?
  //runtime_api_call(cudapeekatlasterror());
}
