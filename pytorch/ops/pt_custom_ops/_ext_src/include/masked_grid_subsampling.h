#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> masked_grid_subsampling(at::Tensor points, at::Tensor mask, const int nsamples, const float sampleDl);