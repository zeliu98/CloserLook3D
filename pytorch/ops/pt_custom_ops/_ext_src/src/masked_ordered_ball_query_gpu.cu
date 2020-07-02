#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "cuda_utils.h"

// input: query_xyz(b, m, 3) support_xyz(b, n, 3) query_mask(b, m) support_mask(b, n)
// temp: dists(b, m, 3*nsample) tempidxs: (b, m, 3*nsample)
// output: idx(b, m, nsample), idx_mask(b, m, nsample)
__global__ void masked_ordered_query_ball_point_kernel(int b, int n, int m, float radius,
                                                       int nsample,
                                                       const float *__restrict__ query_xyz,
                                                       const float *__restrict__ support_xyz,
                                                       const int *__restrict__ query_mask,
                                                       const int *__restrict__ support_mask,
                                                       int *__restrict__ idx,
                                                       int *__restrict__ idx_mask,
                                                       float *__restrict__ dists,
                                                       int *__restrict__ tempidxs) {
  int batch_index = blockIdx.x;
  support_xyz += batch_index * n * 3;
  query_xyz += batch_index * m * 3;
  support_mask += batch_index * n;
  query_mask += batch_index * m;


  idx += m * nsample * batch_index;
  idx_mask += m * nsample * batch_index;
  dists +=  m * 3 * nsample * batch_index;
  tempidxs += m * 3 * nsample * batch_index;
  

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    int query_mk = query_mask[j];

    float query_x = query_xyz[j * 3 + 0];
    float query_y = query_xyz[j * 3 + 1];
    float query_z = query_xyz[j * 3 + 2];
    
    int cnt = 0;
    float min_dist = radius2;
    int min_idx = 0;
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
      if (d2 < radius2){
        if(d2 < min_dist){
          min_dist = d2;
          min_idx = k;
        }
        
        if(cnt >= (3 * nsample)) continue;

        dists[j * 3 * nsample + cnt]  = d2;
        tempidxs[j * 3 * nsample + cnt] = k;
        cnt++;
      }      
    }

    if (cnt >= (3 * nsample) && min_idx>tempidxs[j * 3 * nsample + cnt - 1]){
      tempidxs[j * 3 * nsample + cnt - 1] = min_idx;
      dists[j * 3 * nsample + cnt - 1]  = min_dist;
    }

    thrust::sort_by_key(thrust::device, dists+j*3*nsample, dists+j*3*nsample+cnt, tempidxs+j*3*nsample);

    for(int i=0; i < cnt && i < nsample; i++){
      idx[j * nsample + i] = tempidxs[j * 3 * nsample + i];
      idx_mask[j * nsample + i] = 1;
    }
    for (int i = cnt; i < nsample; i++) {
      idx[j * nsample + i] = tempidxs[j * 3 * nsample + (i % cnt)];
      idx_mask[j * nsample + i] = 0;
    }
    
    // if query point is a padding point
    if (query_mk == 0){
      for (int l = 0; l < nsample; ++l) {
        idx_mask[j * nsample + l] = 0;
      }
    }

  }
}

void masked_ordered_query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                                    int nsample, const float *query_xyz,
                                                    const float *support_xyz, 
                                                    const int *query_mask,
                                                    const int *support_mask,  
                                                    int *idx, int *idx_mask,
                                                    float *dists, int *tempidxs) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  masked_ordered_query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, query_xyz, support_xyz, query_mask, support_mask, idx, idx_mask, dists, tempidxs);

  CUDA_CHECK_ERRORS();
}
