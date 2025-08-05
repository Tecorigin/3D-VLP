// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
//#include <aten/sdaa/SDAAContext.h>
#include <torch/extension.h>


#define CHECK_SDAA(x)                                    \
  do {                                                   \
    AT_ASSERT(x.device()==at::kPrivateUse1, #x " must be a SDAA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                          \
  do {                                                               \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                               \
  do {                                                \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Int, \
              #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                               \
  do {                                                  \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Float, \
              #x " must be a float tensor");            \
  } while (0)
