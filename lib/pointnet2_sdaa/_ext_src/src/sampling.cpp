// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  points = points.contiguous();
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.device()==at::kPrivateUse1) {
    CHECK_SDAA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

//  if (points.device()==at::kPrivateUse1) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
//  } else {
//    AT_ASSERT(false, "CPU not supported");
//  }
//  std::cout << "Finish gather_points......."<< std::endl;

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.device()==at::kPrivateUse1) {
    CHECK_SDAA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

//  if (grad_out.device()==at::kPrivateUse1) {
  gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                    idx.size(1), grad_out.data<float>(),
                                    idx.data<int>(), output.data<float>());
//  } else {
//    AT_ASSERT(false, "CPU not supported");
//  }

  std::cout << "Finish gather_points_grad......."<< std::endl;

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  points = points.contiguous();
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));
  
                  
  furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        tmp.data<float>(), output.data<int>());

//  std::cout << "Finish furthest......."<< std::endl;

  return output;
}
