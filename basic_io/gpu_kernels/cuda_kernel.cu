#include <stdio.h>
#include "../../commons/h5bench_util.h"
#include "cuda_kernel.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void kernel(data_contig_md *data, volatile int *kernel_flag) {

  int32_t total_threads = blockDim.x * gridDim.x;
  int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for( int32_t i = gid; i < data->particle_cnt; i+=total_threads ) {
    data->d_x[i] += 0.2;
    data->d_y[i] += 0.2;
    data->d_z[i] += 0.2;
    data->d_px[i] += 0.2;
    data->d_py[i] += 0.2;
    data->d_pz[i] += 0.2;
    data->d_id_1[i] += 1;
    data->d_id_2[i] += 0.2;
  }

  // wait for cpu to tell kernel to finish, or keep running
  while(!*kernel_flag);
}

void kernel_call(data_contig_md *data, volatile int *kernel_flag, cudaStream_t stream_id) {
  dim3 threadsperblock = 128;
  dim3 blockspergrid = 80;

  //dim3 blockspergrid = dim3(ceil((double)numparticles/256), 1, 1);
  //printf("cuda kernel launch with %d blocks of %d threads\n", blockspergrid, threadsperblock);
  kernel<<<threadsperblock, blockspergrid, 0, stream_id>>>(data, kernel_flag);

  // todo: false postive cufile error?
  //runtime_api_call(cudapeekatlasterror());
}
