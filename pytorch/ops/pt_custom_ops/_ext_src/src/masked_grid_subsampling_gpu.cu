#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "cuda_utils.h"


// Input dataset: (b, n, 3) mask: (b, n)
// Ouput subxyz (b, m, 3) submask: (b, m)
__global__ void masked_grid_subsampling_kernel(int n, int m, float sampleDl, 
                                               const float *__restrict__ dataset, 
                                               const int *__restrict__ mask, 
                                               float *__restrict__ subxyz, 
                                               int *__restrict__ submask, 
                                               int *__restrict__ mapidxs,
                                               int *__restrict__ tempidxs,
                                               float *__restrict__ temp_subxyz) {
  //  for one pointcloud, return the map ind for each point

  // move pointer to current batch
  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  mask += batch_index * n;
  subxyz +=  batch_index * m * 3;
  submask += batch_index * m;
  mapidxs += batch_index * n;
  tempidxs += batch_index * n;
  temp_subxyz +=  batch_index * n * 3;
  
  float minx, miny, minz;
  float maxx, maxy, maxz;
  minx = dataset[0]; miny=dataset[1]; minz=dataset[2];
  maxx = dataset[0]; maxy=dataset[1]; maxz=dataset[2];
  for(int i = 1; i < n; i++){
    float x, y, z;
    x = dataset[i * 3 + 0];
    y = dataset[i * 3 + 1];
    z = dataset[i * 3 + 2];
    if(x>maxx) maxx=x;
    if(y>maxy) maxy=y;
    if(z>maxz) maxz=z;
    if(x<minx) minx=x;
    if(y<miny) miny=y;
    if(z<minz) minz=z;
  }

  float originx = floor(minx * (1/sampleDl)) * sampleDl;
  float originy = floor(miny * (1/sampleDl)) * sampleDl;
  float originz = floor(minz * (1/sampleDl)) * sampleDl;

  int sampleNX = (int)floor((maxx - originx) / sampleDl) + 1;
  int sampleNY = (int)floor((maxy - originy) / sampleDl) + 1;
  int sampleNZ = (int)floor((maxz - originz) / sampleDl) + 1;

  
  int iX, iY, iZ, mapIdx;
  int mask_ind=0;
  for(int i = 0; i < n; i++){
    float x, y, z;
    int mk;
    x = dataset[i * 3 + 0];
    y = dataset[i * 3 + 1];
    z = dataset[i * 3 + 2];
    mk = mask[i];
    if (mk == 0) break;  
    // Position of point in sample map
    iX = (int)floor((x - originx) / sampleDl);
    iY = (int)floor((y - originy) / sampleDl);
    iZ = (int)floor((z - originz) / sampleDl);

    mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;
    mapidxs[i] = mapIdx;
    tempidxs[i] = i;
    mask_ind++;
  }
  thrust::sort_by_key(thrust::device, mapidxs, mapidxs + mask_ind, tempidxs);

  int top, end, cur_mapid;
  float xs,ys,zs;
  float pnum;
  int j;
  
  cur_mapid = mapidxs[0];
  j = tempidxs[0];
  xs = dataset[j * 3 + 0];
  ys = dataset[j * 3 + 1];
  zs = dataset[j * 3 + 2];
  pnum = 1;
  top = 0;
  for(int i = 1; i < mask_ind; i++){
    mapIdx = mapidxs[i];
    j = tempidxs[i];
    if(mapIdx == cur_mapid){
      xs += dataset[j * 3 + 0];
      ys += dataset[j * 3 + 1];
      zs += dataset[j * 3 + 2];
      pnum+=1;
    } else{
      xs /= pnum;
      ys /= pnum;
      zs /= pnum;
      temp_subxyz[top * 3 + 0] = xs;
      temp_subxyz[top * 3 + 1] = ys;
      temp_subxyz[top * 3 + 2] = zs;
      top++;
      
      xs = dataset[j * 3 + 0];
      ys = dataset[j * 3 + 1];
      zs = dataset[j * 3 + 2];
      pnum = 1;
      cur_mapid = mapIdx;
    }
  }
  xs /= pnum;
  ys /= pnum;
  zs /= pnum;
  temp_subxyz[top * 3 + 0] = xs;
  temp_subxyz[top * 3 + 1] = ys;
  temp_subxyz[top * 3 + 2] = zs;
  top++;
  end = top;

  // shuffle 
  int a = 17; 
  int b = 139; 
  int mod = 256;
  mapidxs[0] = mapidxs[0] % mod;
  tempidxs[0] = 0;
  for(int i = 1; i < end; i++){
    mapidxs[i] = (a * mapidxs[i-1] + b ) % mod;
    tempidxs[i] = i;
  }
  
  thrust::sort_by_key(thrust::device, mapidxs, mapidxs + end, tempidxs);
  
  // form output
  for(int i = 0; i < end && i < m; i++){
    j = tempidxs[i];
    subxyz[i * 3 + 0] = temp_subxyz[j * 3 + 0];
    subxyz[i * 3 + 1] = temp_subxyz[j * 3 + 1];
    subxyz[i * 3 + 2] = temp_subxyz[j * 3 + 2];
    submask[i] = 1;
  }
  // padding with true sub point
  for(int i = end; i < m; i++){
    subxyz[i * 3 + 0] = subxyz[(i % end) * 3 + 0];
    subxyz[i * 3 + 1] = subxyz[(i % end) * 3 + 1];
    subxyz[i * 3 + 2] = subxyz[(i % end) * 3 + 2];
    submask[i] = 0;
  }

}

void masked_grid_subsampling_kernel_wrapper(int b, int n, int m, float sampleDl, const float *dataset, const int *mask, float *subxyz, int *submask, int *mapidxs, int *tempidxs, float *temp_subxyz) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  masked_grid_subsampling_kernel<<<b, 1, 0, stream>>>(n, m, sampleDl, dataset, mask, subxyz, submask,  mapidxs, tempidxs, temp_subxyz);

  CUDA_CHECK_ERRORS();
}
