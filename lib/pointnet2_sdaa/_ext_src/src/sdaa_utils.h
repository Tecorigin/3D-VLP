// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
#include <cmath>

//#include <cuda.h>
#include <sdaa_runtime.h>

#include <vector>

#if defined(__SDAA__) && defined(__clang__)

struct uint3 {
  unsigned int x, y, z;
};

typedef struct uint3 uint3;

struct dim3 {
  unsigned int x, y, z;
#if defined(__cplusplus)
#if __cplusplus >= 201103L
  __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1,
                                     unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
  __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ constexpr operator uint3(void) const {
    return uint3{x, y, z};
  }
#else
  __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1,
                           unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
  __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ operator uint3(void) const {
    uint3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
  }
#endif // __cplusplus >= 201103L
#endif // __cplusplus
};
#endif

#define SDAA_CHECK_ERRORS()                                           \
  do {                                                                \
    sdaaError_t err = sdaaGetLastError();                             \
    if (sdaaSuccess != err) {                                         \
      fprintf(stderr, "SDAA kernel failed : %s\n%s at L:%d in %s\n",  \
              sdaaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)

#endif
