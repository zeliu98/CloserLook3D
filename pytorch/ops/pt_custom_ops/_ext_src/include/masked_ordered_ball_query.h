#pragma once
#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> masked_ordered_ball_query(at::Tensor query_xyz, at::Tensor support_xyz, 
                                                  at::Tensor query_mask,at::Tensor support_mask,
                                                  const float radius,const int nsample);
