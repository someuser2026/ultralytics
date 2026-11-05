#pragma once

// Keep this header usable by both MSVC (bindings) and NVCC (kernels).
#if defined(_MSC_VER) && !defined(__CUDACC__)
#define SSW_RESTRICT __restrict
#else
#define SSW_RESTRICT __restrict__
#endif

namespace ssw {
// Same float rounding as the original M_LOG2E conversion, without CRT macros.
inline constexpr float kLog2E = 1.44269504088896340736f;
}
