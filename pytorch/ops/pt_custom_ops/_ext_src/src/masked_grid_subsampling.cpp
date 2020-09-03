#include "masked_grid_subsampling.h"
#include "utils.h"
#include <iostream>

void masked_grid_subsampling_kernel_wrapper(int b, int n, int m, float sampleDl,
                                            const float *dataset, const int *mask,
                                            float *subxyz, int *submask,
                                            int *mapidxs, int *tempidxs, float *temp_subxyz);




std::vector<at::Tensor> masked_grid_subsampling(at::Tensor points, at::Tensor mask, const int nsamples, const float sampleDl){
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);
  CHECK_CONTIGUOUS(mask);
  CHECK_IS_INT(mask);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples, 3},
                   at::device(points.device()).dtype(at::ScalarType::Float));
  at::Tensor output_mask =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor mapidx =
      torch::zeros({points.size(0), points.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Int));
  at::Tensor tempidx =
      torch::zeros({points.size(0), points.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Int));
  at::Tensor temp_subxyz =
      torch::zeros({points.size(0), points.size(1), 3},
                   at::device(points.device()).dtype(at::ScalarType::Float));


  if (points.device().is_cuda()) {
    masked_grid_subsampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, sampleDl, points.data_ptr<float>(), mask.data_ptr<int>(), output.data_ptr<float>(), output_mask.data_ptr<int>(), mapidx.data_ptr<int>(), tempidx.data_ptr<int>(), temp_subxyz.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }
  return {output, output_mask};
}
