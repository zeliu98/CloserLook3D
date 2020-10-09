#include "masked_ordered_ball_query.h"
#include "utils.h"
#include <vector>

void masked_ordered_query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                            int nsample, const float *query_xyz,
                                            const float *support_xyz,
                                            const int *query_mask,
                                            const int *support_mask,
                                            int *idx, int *idx_mask,
                                            float *dists, int *tempidxs);

std::vector<at::Tensor> masked_ordered_ball_query(at::Tensor query_xyz, at::Tensor support_xyz,
                                                  at::Tensor query_mask,at::Tensor support_mask,
                                                  const float radius,const int nsample) {
  CHECK_CONTIGUOUS(query_xyz);
  CHECK_CONTIGUOUS(support_xyz);
  CHECK_CONTIGUOUS(query_mask);
  CHECK_CONTIGUOUS(support_mask);
  CHECK_IS_FLOAT(query_xyz);
  CHECK_IS_FLOAT(support_xyz);
  CHECK_IS_INT(query_mask);
  CHECK_IS_INT(support_mask);

  if (query_xyz.device().is_cuda()) {
    CHECK_CUDA(support_xyz);
    CHECK_CUDA(query_mask);
    CHECK_CUDA(support_mask);
  }

  at::Tensor idx =
      torch::zeros({query_xyz.size(0), query_xyz.size(1), nsample},
                   at::device(query_xyz.device()).dtype(at::ScalarType::Int));
  at::Tensor idx_mask =
      torch::zeros({query_xyz.size(0), query_xyz.size(1), nsample},
                   at::device(query_xyz.device()).dtype(at::ScalarType::Int));

  at::Tensor dists =
      torch::zeros({query_xyz.size(0), query_xyz.size(1), 3*nsample},
                   at::device(query_xyz.device()).dtype(at::ScalarType::Float));

  at::Tensor tempidxs =
      torch::zeros({query_xyz.size(0), query_xyz.size(1), 3*nsample},
                   at::device(query_xyz.device()).dtype(at::ScalarType::Int));


  if (query_xyz.device().is_cuda()) {
    masked_ordered_query_ball_point_kernel_wrapper(support_xyz.size(0), support_xyz.size(1), query_xyz.size(1),
                                                  radius, nsample, query_xyz.data_ptr<float>(),
                                                  support_xyz.data_ptr<float>(), query_mask.data_ptr<int>(),
                                                  support_mask.data_ptr<int>(),
                                                  idx.data_ptr<int>(), idx_mask.data_ptr<int>(),
                                                  dists.data_ptr<float>(), tempidxs.data_ptr<int>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return {idx, idx_mask};
}
