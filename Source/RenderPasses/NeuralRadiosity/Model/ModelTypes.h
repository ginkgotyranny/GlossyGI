#pragma once

#if defined(TCNN_HALF_PRECISION) && TCNN_HALF_PRECISION && defined(__CUDACC__)
#include <cuda_fp16.h>
#elif defined(TCNN_HALF_PRECISION) && TCNN_HALF_PRECISION
struct __half;
#endif

#if defined(TCNN_HALF_PRECISION) && TCNN_HALF_PRECISION
using precision_t = __half;
#else
using precision_t = float;
#endif