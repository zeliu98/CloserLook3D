#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

// input: query_xyz(b, m, 3) support_xyz(b, n, 3) query_mask(b, m) support_mask(b, n)
// output: idx(b, m, 1), idx_mask(b, m, 1)
__global__ void masked_nearest_query_kernel(int b, int n, int m,
                                            const float *__restrict__ query_xyz,
                                            const float *__restrict__ support_xyz,
                                            const int *__restrict__ query_mask,
                                            const int *__restrict__ support_mask,
                                            int *__restrict__ idx,
                                            int *__restrict__ idx_mask) {
  int batch_index = blockIdx.x;
  support_xyz += batch_index * n * 3;
  query_xyz += batch_index * m * 3;
  support_mask += batch_index * n;
  query_mask += batch_index * m;


  idx += m * batch_index;
  idx_mask += m * batch_index;
  

  int index = threadIdx.x;
  int stride = blockDim.x;


  for (int j = index; j < m; j += stride) {
    int query_mk = query_mask[j];
    
    float query_x = query_xyz[j * 3 + 0];
    float query_y = query_xyz[j * 3 + 1];
    float query_z = query_xyz[j * 3 + 2];
    
    float min_dist = 100;
    int min_idx = -1;
    for (int k = 0; k < n; ++k) {
      int mk = support_mask[k];
      if (mk == 0){
        break;
      }
      float x = support_xyz[k * 3 + 0];
      float y = support_xyz[k * 3 + 1];
      float z = support_xyz[k * 3 + 2];
      float d2 = (query_x - x) * (query_x - x) + (query_y - y) * (query_y - y) +
                 (query_z - z) * (query_z - z);
      if(d2 < min_dist){
      min_dist = d2;
      min_idx = k;
      }
    }
    idx[j] = min_idx;
    if (query_mk == 0){
      idx_mask[j] = 0;
    } else{
      idx_mask[j] = 1;
    }
    
  }
}

void masked_nearest_query_kernel_wrapper(int b, int n, int m,  
                                        const float *query_xyz,
                                        const float *support_xyz, 
                                        const int *query_mask,
                                        const int *support_mask,  
                                        int *idx, int *idx_mask) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  masked_nearest_query_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, query_xyz, support_xyz, query_mask, support_mask, idx, idx_mask);

  CUDA_CHECK_ERRORS();
}
