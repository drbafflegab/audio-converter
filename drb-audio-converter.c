#include "drb-audio-converter.h"

/* -------------------------------------------------------------------------- */

_Static_assert(sizeof(int) >= 4, "");

/* -------------------------------------------------------------------------- */

#pragma mark - Dependencies

#if defined(__x86_64__)
#  if defined(__SSE4_1__)
#    define SIMD_SSE
#  endif
#elif defined(__aarch64__)
#  if defined(__ARM_NEON__)
#    define SIMD_NEON
#  endif
#endif

#if defined(SIMD_SSE)
#  include <x86intrin.h> // For `__m128`, `_mm_add_ps`, `_mm_mul_ps`, ...
#elif defined(SIMD_NEON)
#  include <arm_neon.h> // For `float32x4_t`, `vaddq_f32`, `vmulq_f32`, ...
#endif

#if defined(DEBUG)
#  include <assert.h> // For `assert`.
#endif

#include <string.h> // For `memcpy` and `memset`.

/* -------------------------------------------------------------------------- */

#pragma mark - Assert Macro

#if defined(DEBUG)
#  define ASSERT assert
#else
#  define ASSERT(...)
#endif

/* -------------------------------------------------------------------------- */

#pragma mark - Print Conversion Flag

#if 0 // Enable to print the conversion into the console during construction.
#  define PRINT_CONVERSION
#endif

/* -------------------------------------------------------------------------- */

#pragma mark - Min and Max Functions

static inline int min (int const x, int const y) { return x < y ? x : y; }
static inline int max (int const x, int const y) { return x > y ? x : y; }

/* -------------------------------------------------------------------------- */

typedef long long Ticks;

enum { tick_rate = 141120000LL }; // Can be computed from 2^9 * 3^2 * 5^4 * 7^2.

_Static_assert(tick_rate % drb_audio_converter_sampling_rate_8000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_11025 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_16000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_22050 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_32000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_44100 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_48000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_60000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_88200 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_96000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_120000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_176400 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_192000 == 0, "");
_Static_assert(tick_rate % drb_audio_converter_sampling_rate_240000 == 0, "");

/* -------------------------------------------------------------------------- */

#pragma mark - Bump Allocator

enum { cache_line_size = 64 };

static long cache_align (long const size)
{
    return (size + cache_line_size - 1) & -cache_line_size;
}

typedef struct Bump_Allocator
{
    unsigned char * memory;
    long offset;
}
Bump_Allocator;

static void bump_allocator_construct
    (
        Bump_Allocator * const allocator,
        void * const memory
    )
{
    allocator->memory = memory;
    allocator->offset = 0;
}

static void * alloc (Bump_Allocator * const allocator, long const size)
{
    ASSERT(size >= 0);

    long const offset = allocator->offset;

    allocator->offset += cache_align(size);

    return allocator->memory ? allocator->memory + offset : 0;
}

/* -------------------------------------------------------------------------- */

#pragma mark - Format, Processor, and Pass Definitions

enum
{
    channel_count_1 = 1,
    channel_count_2 = 2,
    channel_count_3 = 3,
    channel_count_4 = 4,
    channel_count_5 = 5,
    channel_count_6 = 6,
    channel_count_7 = 7,
    channel_count_8 = 8
};

enum
{
    layout_interleaved = 1,
    layout_deinterleaved = 2
};

enum
{
    block_size_1 = 1,
    block_size_4 = 4,
    block_size_16 = 16,
    block_size_64 = 64,
    block_size_256 = 256,
    block_size_1024 = 1024,
    block_size_4096 = 4096
};

typedef struct Processor
{
    int (* pushed_target_frame_count)
        (
            void const * state,
            int source_frame_count,
            double * latency
        );

    int (* pulled_source_frame_count)
        (
            void const * state,
            int target_frame_count,
            double * latency
        );

    int /* <- pushed target frame count */ (* push)
        (
            void * state,
            float * restrict source_samples,
            float * restrict target_samples,
            int source_frame_count
        );

    int /* <- pulled source frame count */ (* pull)
        (
            void * state,
            float * restrict source_samples,
            float * restrict target_samples,
            int target_frame_count
        );

    void * state;
}
Processor;

typedef struct Format
{
    int sampling_rate;
    int channel_count;
    int layout;
    int block_size;
    int max_block_count;
}
Format;

typedef struct Pass
{
    Processor (* create_processor)
        (
            void const * configuration,
            Bump_Allocator * allocator,
            Format const * source_format,
            Format const * target_format
        );

    void (* restrain_formats)
        (
            void const * configuration,
            Format * source_format,
            Format * target_format
        );

    void const * configuration;
}
Pass;

/* -------------------------------------------------------------------------- */

#pragma mark - Windows

typedef struct Window { float * samples; } Window;

enum
{
    window_8p_size  =  8 * 2 * sizeof(float),
    window_16p_size = 16 * 2 * sizeof(float),
    window_24p_size = 24 * 2 * sizeof(float),
    window_32p_size = 32 * 2 * sizeof(float),
    window_40p_size = 40 * 2 * sizeof(float),
    window_48p_size = 48 * 2 * sizeof(float),
    window_128p_size = 128 * 2 * sizeof(float)
};

#define WINDOW_WRITE_IMPL(WINDOW_SIZE)                                         \
static inline void window_##WINDOW_SIZE##p_write                               \
    (                                                                          \
        Window * const window,                                                 \
        float const sample,                                                    \
        int const offset                                                       \
    )                                                                          \
{                                                                              \
    window->samples[offset +           0] = sample;                            \
    window->samples[offset + WINDOW_SIZE] = sample;                            \
}                                                                              \

WINDOW_WRITE_IMPL( 8) // => `window_8p_write`
WINDOW_WRITE_IMPL(16) // => `window_16p_write`
WINDOW_WRITE_IMPL(24) // => `window_24p_write`
WINDOW_WRITE_IMPL(32) // => `window_32p_write`
WINDOW_WRITE_IMPL(40) // => `window_40p_write`
WINDOW_WRITE_IMPL(48) // => `window_48p_write`
WINDOW_WRITE_IMPL(128) // => `window_128p_write`

#define WINDOW_NEXT_INDEX_IMPL(WINDOW_SIZE)                                    \
static inline int window_##WINDOW_SIZE##p_next_index (int const index)         \
{                                                                              \
    return (index + 1) & (WINDOW_SIZE - 1);                                    \
}                                                                              \

WINDOW_NEXT_INDEX_IMPL( 8) // => `window_8p_next_index`
WINDOW_NEXT_INDEX_IMPL(16) // => `window_16p_next_index`
WINDOW_NEXT_INDEX_IMPL(32) // => `window_32p_next_index`
WINDOW_NEXT_INDEX_IMPL(128) // => `window_128p_next_index`

static inline int window_24p_next_index (int const index)
{
    return index == (24 - 1) ? 0 : index + 1;
}

static inline int window_40p_next_index (int const index)
{
    return index == (40 - 1) ? 0 : index + 1;
}

static inline int window_48p_next_index (int const index)
{
    return index == (48 - 1) ? 0 : index + 1;
}

/* -------------------------------------------------------------------------- */

#pragma mark - Unify

static void unify (int * const source_property, int * const target_property)
{
    if (*source_property == 0 && *target_property > 0)
    {
        *source_property = *target_property;
    }
    else if (*source_property > 0 && *target_property == 0)
    {
        *target_property = *source_property;
    }

    ASSERT(*source_property == *target_property);
}

/* -------------------------------------------------------------------------- */

#pragma mark - Convolution

#if defined(SIMD_SSE)

static inline float convolve_2
    (
        float const kernel [static const 2],
        float const window [static const 2]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];

    return acc;
}

#define CONVOLVE_FUNC(TAP_COUNT)                                               \
static inline float convolve_##TAP_COUNT                                       \
    (                                                                          \
        float const kernel [static const TAP_COUNT],                           \
        float const window [static const TAP_COUNT]                            \
    )                                                                          \
{                                                                              \
    _Static_assert(TAP_COUNT % 4 == 0, "");                                    \
                                                                               \
    __m128 acc = _mm_mul_ps(_mm_load_ps(kernel), _mm_load_ps(window));         \
                                                                               \
    for (int tap = 4; tap < TAP_COUNT; tap += 4)                               \
    {                                                                          \
        __m128 const a = _mm_load_ps(kernel + tap);                            \
        __m128 const b = _mm_load_ps(window + tap);                            \
                                                                               \
        acc = _mm_add_ps(acc, _mm_mul_ps(a, b));                               \
    }                                                                          \
                                                                               \
    acc = _mm_hadd_ps(acc, acc);                                               \
    acc = _mm_hadd_ps(acc, acc);                                               \
                                                                               \
    return _mm_cvtss_f32(acc);                                                 \
}                                                                              \

CONVOLVE_FUNC(4) // => `convolve_4`
CONVOLVE_FUNC(8) // => `convolve_8`
CONVOLVE_FUNC(16) // => `convolve_16`
CONVOLVE_FUNC(24) // => `convolve_24`
CONVOLVE_FUNC(32) // => `convolve_32`
CONVOLVE_FUNC(40) // => `convolve_40`
CONVOLVE_FUNC(48) // => `convolve_48`
CONVOLVE_FUNC(128) // => `convolve_128`

#elif defined(SIMD_NEON)

static inline float convolve_2
    (
        float const kernel [static const 2],
        float const window [static const 2]
    )
{
    return vaddv_f32(vmul_f32(vld1_f32(kernel), vld1_f32(window)));
}

#define CONVOLVE_FUNC(TAP_COUNT)                                               \
static inline float convolve_##TAP_COUNT                                       \
    (                                                                          \
        float const kernel [static const TAP_COUNT],                           \
        float const window [static const TAP_COUNT]                            \
    )                                                                          \
{                                                                              \
    _Static_assert(TAP_COUNT % 4 == 0, "");                                    \
                                                                               \
    float32x4_t acc = vmulq_f32(vld1q_f32(kernel), vld1q_f32(window));         \
                                                                               \
    for (int tap = 4; tap < TAP_COUNT; tap += 4)                               \
    {                                                                          \
        acc = vfmaq_f32(acc, vld1q_f32(kernel + tap), vld1q_f32(window + tap));\
    }                                                                          \
                                                                               \
    return vaddvq_f32(acc);                                                    \
}                                                                              \

CONVOLVE_FUNC(4) // => `convolve_4`
CONVOLVE_FUNC(8) // => `convolve_8`
CONVOLVE_FUNC(16) // => `convolve_16`
CONVOLVE_FUNC(24) // => `convolve_24`
CONVOLVE_FUNC(32) // => `convolve_32`
CONVOLVE_FUNC(40) // => `convolve_40`
CONVOLVE_FUNC(48) // => `convolve_48`
CONVOLVE_FUNC(128) // => `convolve_128`

#else // No SIMD

static inline float convolve_2
    (
        float const kernel [static const 2],
        float const window [static const 2]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];

    return acc;
}

#define CONVOLVE_FUNC(TAP_COUNT)                                               \
static inline float convolve_##TAP_COUNT                                       \
    (                                                                          \
        float const kernel [static const TAP_COUNT],                           \
        float const window [static const TAP_COUNT]                            \
    )                                                                          \
{                                                                              \
    _Static_assert(TAP_COUNT % 4 == 0, "");                                    \
                                                                               \
    float acc = 0.0f;                                                          \
                                                                               \
    for (int tap = 0; tap < TAP_COUNT; tap += 4)                               \
    {                                                                          \
        acc += kernel[tap + 0] * window[tap + 0];                              \
        acc += kernel[tap + 1] * window[tap + 1];                              \
        acc += kernel[tap + 2] * window[tap + 2];                              \
        acc += kernel[tap + 3] * window[tap + 3];                              \
    }                                                                          \
                                                                               \
    return acc;                                                                \
}                                                                              \

CONVOLVE_FUNC(4) // => `convolve_4`
CONVOLVE_FUNC(8) // => `convolve_8`
CONVOLVE_FUNC(16) // => `convolve_16`
CONVOLVE_FUNC(24) // => `convolve_24`
CONVOLVE_FUNC(32) // => `convolve_32`
CONVOLVE_FUNC(40) // => `convolve_40`
CONVOLVE_FUNC(48) // => `convolve_48`

#endif // SIMD

/* -------------------------------------------------------------------------- */

#pragma mark - Interpolation

enum
{
    kernel_count_8p_linear  = 1,
    kernel_count_16p_linear = 1,
    kernel_count_24p_linear = 1,
    kernel_count_32p_linear = 1,
    kernel_count_8p_cubic   = 32,
    kernel_count_16p_cubic  = 32,
    kernel_count_24p_cubic  = 32,
    kernel_count_32p_cubic  = 32,
};

static float const kernels_8p_linear  [kernel_count_8p_linear ][ 8];
static float const kernels_16p_linear [kernel_count_16p_linear][16];
static float const kernels_24p_linear [kernel_count_16p_linear][24];
static float const kernels_32p_linear [kernel_count_32p_linear][32];
static float const kernels_8p_cubic   [kernel_count_16p_cubic ][ 8];
static float const kernels_16p_cubic  [kernel_count_16p_cubic ][16];
static float const kernels_24p_cubic  [kernel_count_16p_cubic ][24];
static float const kernels_32p_cubic  [kernel_count_32p_cubic ][32];

#if 0

static inline float interpolate_drop
    (
        float const blend,
        float const window [static const 1]
    )
{
    (void)blend;

    static float const transform [1][1] =
    {
        { 1.0f }
    };

    return transform[0][0] * window[0];
}

#endif

static inline float interpolate_linear
    (
        float const blend,
        float const window [static const 2]
    )
{
    // Uses linear interpolation to interpolate between two samples in a 2-point
    // window. Say the window is [`p0`, `p1`], `interpolate_linear` will blend
    // between `p0` and `p1` according to the value provided in the `blend`
    // parameter, which must be in the range [0; 1].

    static float const transform [2][2] =
    {
        { +1.0f,  0.0f },
        { -1.0f, +1.0f }
    };

    float const acc_0 = convolve_2(transform[0], window);
    float const acc_1 = convolve_2(transform[1], window);

    return acc_1 * blend + acc_0;
}

static inline float interpolate_cubic
    (
        float const blend,
        float const window [static const 4]
    )
{
    // Uses a Catmull–Rom spline to interpolate between the two middle samples
    // in a 4-sample window. Say the window is [`p0`, `p1`, `p2`, `p3`],
    // this function will blend between `p1` and `p2` according to the value
    // provided in the `blend` parameter, which must be in the range [0; 1].

    static float const transform [4][4] =
    {
        {         0.0f, 0.5f * +2.0f,         0.0f,         0.0f },
        { 0.5f * -1.0f,         0.0f, 0.5f * +1.0f,         0.0f },
        { 0.5f * +2.0f, 0.5f * -5.0f, 0.5f * +4.0f, 0.5f * -1.0f },
        { 0.5f * -1.0f, 0.5f * +3.0f, 0.5f * -3.0f, 0.5f * +1.0f }
    };

    float const acc_0 = convolve_4(transform[0], window);
    float const acc_1 = convolve_4(transform[1], window);
    float const acc_2 = convolve_4(transform[2], window);
    float const acc_3 = convolve_4(transform[3], window);

    return (((acc_3 * blend + acc_2) * blend) + acc_1) * blend + acc_0;
}

// Uses a bank of sincs to interpolate between the two middle samples in an 8,
// 16, 24, or 32-sample window. Say the window is [`p0`, `p1`, ..., `p7`, `p8`],
// this function will blend between `p3` and `p4` according to the value
// provided in the `blend` parameter, which must be in the range [0; 1].
#define INTERPOLATE_SINC_DROP_IMPL(KERNEL_SIZE)                                \
static inline float interpolate_##KERNEL_SIZE##p_drop                          \
    (                                                                          \
        double const blend,                                                    \
        float const window [static const KERNEL_SIZE])                         \
{                                                                              \
    static double const scale = kernel_count_##KERNEL_SIZE##p_drop;            \
                                                                               \
    int const index = (int)(blend * scale);                                    \
                                                                               \
    ASSERT(index < kernel_count_##KERNEL_SIZE##p_linear);                      \
                                                                               \
    float const * const kernel = kernels_##KERNEL_SIZE##p_drop[index];         \
                                                                               \
    return kernel[0] * window[0];                                              \
}                                                                              \

// Uses a bank of sincs to interpolate between the two middle samples in an 8,
// 16, 24, or 32-sample window. Say the window is [`p0`, `p1`, ..., `p7`, `p8`],
// this function will blend between `p3` and `p4` according to the value
// provided in the `blend` parameter, which must be in the range [0; 1].
#define INTERPOLATE_SINC_LINEAR_IMPL(KERNEL_SIZE)                              \
static inline float interpolate_##KERNEL_SIZE##p_linear                        \
    (                                                                          \
        double const blend,                                                    \
        float const window [static const KERNEL_SIZE])                         \
{                                                                              \
    static double const scale = kernel_count_##KERNEL_SIZE##p_linear - 1;      \
                                                                               \
    int const index = (int)(blend * scale);                                    \
                                                                               \
    ASSERT(index < kernel_count_##KERNEL_SIZE##p_linear - 1);                  \
                                                                               \
    float const * const kernel_0 = kernels_##KERNEL_SIZE##p_linear[index + 0]; \
    float const * const kernel_1 = kernels_##KERNEL_SIZE##p_linear[index + 1]; \
                                                                               \
    _Alignas(8) float const points [2] =                                       \
    {                                                                          \
        convolve_##KERNEL_SIZE(kernel_0, window),                              \
        convolve_##KERNEL_SIZE(kernel_1, window)                               \
    };                                                                         \
                                                                               \
    double const sub_blend = blend * scale - (double)index;                    \
                                                                               \
    return interpolate_linear(sub_blend, points);                              \
}                                                                              \

INTERPOLATE_SINC_LINEAR_IMPL( 8) // => `interpolate_8p_linear`
INTERPOLATE_SINC_LINEAR_IMPL(16) // => `interpolate_16p_linear`
INTERPOLATE_SINC_LINEAR_IMPL(24) // => `interpolate_24p_linear`
INTERPOLATE_SINC_LINEAR_IMPL(32) // => `interpolate_32p_linear`

// Uses a bank of sincs to interpolate between the two middle samples in an 8,
// 16, 24, or 32-sample window. Say the window is [`p0`, `p1`, ..., `p7`, `p8`],
// this function will blend between `p3` and `p4` according to the value
// provided in the `blend` parameter, which must be in the range [0; 1].
#define INTERPOLATE_SINC_CUBIC_IMPL(KERNEL_SIZE)                               \
static inline float interpolate_##KERNEL_SIZE##p_cubic                         \
    (                                                                          \
        double const blend,                                                    \
        float const window [static const KERNEL_SIZE])                         \
{                                                                              \
    static double const scale = kernel_count_##KERNEL_SIZE##p_cubic - 3;       \
                                                                               \
    int const index = (int)(blend * scale);                                    \
                                                                               \
    ASSERT(index < kernel_count_##KERNEL_SIZE##p_cubic - 3);                   \
                                                                               \
    float const * const kernel_0 = kernels_##KERNEL_SIZE##p_cubic[index + 0];  \
    float const * const kernel_1 = kernels_##KERNEL_SIZE##p_cubic[index + 1];  \
    float const * const kernel_2 = kernels_##KERNEL_SIZE##p_cubic[index + 2];  \
    float const * const kernel_3 = kernels_##KERNEL_SIZE##p_cubic[index + 3];  \
                                                                               \
    _Alignas(16) float const points [4] =                                      \
    {                                                                          \
        convolve_##KERNEL_SIZE(kernel_0, window),                              \
        convolve_##KERNEL_SIZE(kernel_1, window),                              \
        convolve_##KERNEL_SIZE(kernel_2, window),                              \
        convolve_##KERNEL_SIZE(kernel_3, window)                               \
    };                                                                         \
                                                                               \
    double const sub_blend = blend * scale - (double)index;                    \
                                                                               \
    return interpolate_cubic(sub_blend, points);                               \
}                                                                              \

INTERPOLATE_SINC_CUBIC_IMPL( 8) // => `interpolate_8p_linear`
INTERPOLATE_SINC_CUBIC_IMPL(16) // => `interpolate_16p_linear`
INTERPOLATE_SINC_CUBIC_IMPL(24) // => `interpolate_24p_linear`
INTERPOLATE_SINC_CUBIC_IMPL(32) // => `interpolate_32p_linear`

/* -------------------------------------------------------------------------- */

#pragma mark - Layout

static int interleave_2
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 2 + 0] = source_samples_0[frame];
        target_samples[frame * 2 + 1] = source_samples_1[frame];
    }

    return frame_count;
}

static int interleave_3
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 3 + 0] = source_samples_0[frame];
        target_samples[frame * 3 + 1] = source_samples_1[frame];
        target_samples[frame * 3 + 2] = source_samples_2[frame];
    }

    return frame_count;
}

static int interleave_4
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 4 + 0] = source_samples_0[frame];
        target_samples[frame * 4 + 1] = source_samples_1[frame];
        target_samples[frame * 4 + 2] = source_samples_2[frame];
        target_samples[frame * 4 + 3] = source_samples_3[frame];
    }

    return frame_count;
}

static int interleave_5
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 5 + 0] = source_samples_0[frame];
        target_samples[frame * 5 + 1] = source_samples_1[frame];
        target_samples[frame * 5 + 2] = source_samples_2[frame];
        target_samples[frame * 5 + 3] = source_samples_3[frame];
        target_samples[frame * 5 + 4] = source_samples_4[frame];
    }

    return frame_count;
}

static int interleave_6
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;
    float * const source_samples_5 = source_samples + frame_count * 5;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 6 + 0] = source_samples_0[frame];
        target_samples[frame * 6 + 1] = source_samples_1[frame];
        target_samples[frame * 6 + 2] = source_samples_2[frame];
        target_samples[frame * 6 + 3] = source_samples_3[frame];
        target_samples[frame * 6 + 4] = source_samples_4[frame];
        target_samples[frame * 6 + 5] = source_samples_5[frame];
    }

    return frame_count;
}

static int interleave_7
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;
    float * const source_samples_5 = source_samples + frame_count * 5;
    float * const source_samples_6 = source_samples + frame_count * 6;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 7 + 0] = source_samples_0[frame];
        target_samples[frame * 7 + 1] = source_samples_1[frame];
        target_samples[frame * 7 + 2] = source_samples_2[frame];
        target_samples[frame * 7 + 3] = source_samples_3[frame];
        target_samples[frame * 7 + 4] = source_samples_4[frame];
        target_samples[frame * 7 + 5] = source_samples_5[frame];
        target_samples[frame * 7 + 6] = source_samples_6[frame];
    }

    return frame_count;
}

static int interleave_8
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;
    float * const source_samples_5 = source_samples + frame_count * 5;
    float * const source_samples_6 = source_samples + frame_count * 6;
    float * const source_samples_7 = source_samples + frame_count * 7;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 8 + 0] = source_samples_0[frame];
        target_samples[frame * 8 + 1] = source_samples_1[frame];
        target_samples[frame * 8 + 2] = source_samples_2[frame];
        target_samples[frame * 8 + 3] = source_samples_3[frame];
        target_samples[frame * 8 + 4] = source_samples_4[frame];
        target_samples[frame * 8 + 5] = source_samples_5[frame];
        target_samples[frame * 8 + 6] = source_samples_6[frame];
        target_samples[frame * 8 + 7] = source_samples_7[frame];
    }

    return frame_count;
}

static int deinterleave_2
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 2 + 0];
        target_samples_1[frame] = source_samples[frame * 2 + 1];
    }

    return frame_count;
}

static int deinterleave_3
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 3 + 0];
        target_samples_1[frame] = source_samples[frame * 3 + 1];
        target_samples_2[frame] = source_samples[frame * 3 + 2];
    }

    return frame_count;
}

static int deinterleave_4
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 4 + 0];
        target_samples_1[frame] = source_samples[frame * 4 + 1];
        target_samples_2[frame] = source_samples[frame * 4 + 2];
        target_samples_3[frame] = source_samples[frame * 4 + 3];
    }

    return frame_count;
}

static int deinterleave_5
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 5 + 0];
        target_samples_1[frame] = source_samples[frame * 5 + 1];
        target_samples_2[frame] = source_samples[frame * 5 + 2];
        target_samples_3[frame] = source_samples[frame * 5 + 3];
        target_samples_4[frame] = source_samples[frame * 5 + 4];
    }

    return frame_count;
}

static int deinterleave_6
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;
    float * const target_samples_5 = target_samples + frame_count * 5;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 6 + 0];
        target_samples_1[frame] = source_samples[frame * 6 + 1];
        target_samples_2[frame] = source_samples[frame * 6 + 2];
        target_samples_3[frame] = source_samples[frame * 6 + 3];
        target_samples_4[frame] = source_samples[frame * 6 + 4];
        target_samples_5[frame] = source_samples[frame * 6 + 5];
    }

    return frame_count;
}

static int deinterleave_7
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;
    float * const target_samples_5 = target_samples + frame_count * 5;
    float * const target_samples_6 = target_samples + frame_count * 6;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 7 + 0];
        target_samples_1[frame] = source_samples[frame * 7 + 1];
        target_samples_2[frame] = source_samples[frame * 7 + 2];
        target_samples_3[frame] = source_samples[frame * 7 + 3];
        target_samples_4[frame] = source_samples[frame * 7 + 4];
        target_samples_5[frame] = source_samples[frame * 7 + 5];
        target_samples_6[frame] = source_samples[frame * 7 + 6];
    }

    return frame_count;
}

static int deinterleave_8
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;
    float * const target_samples_5 = target_samples + frame_count * 5;
    float * const target_samples_6 = target_samples + frame_count * 6;
    float * const target_samples_7 = target_samples + frame_count * 7;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 8 + 0];
        target_samples_1[frame] = source_samples[frame * 8 + 1];
        target_samples_2[frame] = source_samples[frame * 8 + 2];
        target_samples_3[frame] = source_samples[frame * 8 + 3];
        target_samples_4[frame] = source_samples[frame * 8 + 4];
        target_samples_5[frame] = source_samples[frame * 8 + 5];
        target_samples_6[frame] = source_samples[frame * 8 + 6];
        target_samples_7[frame] = source_samples[frame * 8 + 7];
    }

    return frame_count;
}

static Processor interleaver_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration, (void)allocator;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->channel_count >= 2);
    ASSERT(target_format->channel_count >= 2);
    ASSERT(source_format->layout == layout_deinterleaved);
    ASSERT(target_format->layout == layout_interleaved);
    ASSERT(source_format->block_size == target_format->block_size);
    ASSERT(source_format->max_block_count == target_format->max_block_count);

    int (* const callbacks [])(void *, float *, float *, int) =
    {
        [2] = interleave_2,
        [3] = interleave_3,
        [4] = interleave_4,
        [5] = interleave_5,
        [6] = interleave_6,
        [7] = interleave_7,
        [8] = interleave_8
    };

    return (Processor)
    {
        .push = callbacks[source_format->channel_count],
        .pull = callbacks[target_format->channel_count]
    };
}

static Processor deinterleaver_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration, (void)allocator;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->channel_count >= 2);
    ASSERT(target_format->channel_count >= 2);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_deinterleaved);
    ASSERT(source_format->block_size == target_format->block_size);
    ASSERT(source_format->max_block_count == target_format->max_block_count);

    int (* const callbacks [])(void *, float *, float *, int) =
    {
        [2] = deinterleave_2,
        [3] = deinterleave_3,
        [4] = deinterleave_4,
        [5] = deinterleave_5,
        [6] = deinterleave_6,
        [7] = deinterleave_7,
        [8] = deinterleave_8
    };

    return (Processor)
    {
        .push = callbacks[source_format->channel_count],
        .pull = callbacks[target_format->channel_count]
    };
}

static void interleaver_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->layout == 0)
    {
        source_format->layout = layout_deinterleaved;
    }

    if (target_format->layout == 0)
    {
        target_format->layout = layout_interleaved;
    }

    ASSERT(source_format->layout == layout_deinterleaved);
    ASSERT(target_format->layout == layout_interleaved);

    unify(&source_format->sampling_rate, &target_format->sampling_rate);
    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->block_size, &target_format->block_size);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

static void deinterleaver_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->layout == 0)
    {
        source_format->layout = layout_interleaved;
    }

    if (target_format->layout == 0)
    {
        target_format->layout = layout_deinterleaved;
    }

    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_deinterleaved);

    unify(&source_format->sampling_rate, &target_format->sampling_rate);
    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->block_size, &target_format->block_size);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

static const Pass interleaver =
{
    .create_processor = interleaver_create_processor,
    .restrain_formats = interleaver_restrain_formats
};

static const Pass deinterleaver =
{
    .create_processor = deinterleaver_create_processor,
    .restrain_formats = deinterleaver_restrain_formats
};

/* -------------------------------------------------------------------------- */

#pragma mark - Sinc Resampling

static int const resampler_sinc_mode_8p_linear  = 0;
static int const resampler_sinc_mode_16p_linear = 1;
static int const resampler_sinc_mode_24p_linear = 2;
static int const resampler_sinc_mode_32p_linear = 3;
static int const resampler_sinc_mode_8p_cubic   = 4;
static int const resampler_sinc_mode_16p_cubic  = 5;
static int const resampler_sinc_mode_24p_cubic  = 6;
static int const resampler_sinc_mode_32p_cubic  = 7;

typedef struct Resampler_Sinc
{
    Ticks source_delta;
    Ticks target_delta;
    Ticks phase;
    double latency;
    double scale;
    int channel_count;
    int index;
    _Alignas(cache_line_size) Window windows [];
}
Resampler_Sinc;

static int resampler_sinc_target_frame_count
    (
        void const * const state,
        int const source_frame_count,
        double * const latency
    )
{
    Resampler_Sinc const * const resampler = state;

    Ticks count = resampler->target_delta - resampler->phase;

    count += resampler->source_delta * source_frame_count;
    count /= resampler->target_delta;

    *latency += resampler->latency - (resampler->phase - 1) * (1.0 / tick_rate);

    return (int)count;
}

static int resampler_sinc_source_frame_count
    (
        void const * const state,
        int const target_frame_count,
        double * const latency
    )
{
    Resampler_Sinc const * const resampler = state;

    Ticks count = resampler->phase - resampler->target_delta;

    count += resampler->target_delta * target_frame_count;
    count /= resampler->source_delta;

    *latency += resampler->latency - resampler->phase * (1.0 / tick_rate);

    return (int)count;
}

#define RESAMPLER_SINC_UPDATE_WINDOWS_IMPL(KERNEL_SIZE)                        \
static inline void resampler_sinc_update_windows_##KERNEL_SIZE##p              \
    (                                                                          \
        Resampler_Sinc * const resampler,                                      \
        int const source_frame,                                                \
        float const * const source_samples                                     \
    )                                                                          \
{                                                                              \
    for (int channel = 0; channel < resampler->channel_count; channel++)       \
    {                                                                          \
        Window * const window = &resampler->windows[channel];                  \
                                                                               \
        int const index = resampler->channel_count * source_frame + channel;   \
                                                                               \
        window_##KERNEL_SIZE##p_write                                          \
        (                                                                      \
            window,                                                            \
            source_samples[index],                                             \
            resampler->index                                                   \
        );                                                                     \
    }                                                                          \
                                                                               \
    resampler->index = window_##KERNEL_SIZE##p_next_index(resampler->index);   \
}                                                                              \

RESAMPLER_SINC_UPDATE_WINDOWS_IMPL( 8) // => `resampler_sinc_update_windows_8p`
RESAMPLER_SINC_UPDATE_WINDOWS_IMPL(16) // => `resampler_sinc_update_windows_16p`
RESAMPLER_SINC_UPDATE_WINDOWS_IMPL(24) // => `resampler_sinc_update_windows_24p`
RESAMPLER_SINC_UPDATE_WINDOWS_IMPL(32) // => `resampler_sinc_update_windows_32p`

#define RESAMPLE_SINC_FRAME_IMPL(KERNEL_SIZE, INTERPOLATION)                   \
static inline void resample_sinc_frame_##KERNEL_SIZE##p_##INTERPOLATION        \
    (                                                                          \
        Resampler_Sinc * const resampler,                                      \
        int const target_frame,                                                \
        float * const target_samples,                                          \
        double const blend                                                     \
    )                                                                          \
{                                                                              \
    for (int channel = 0; channel < resampler->channel_count; channel++)       \
    {                                                                          \
        Window const * const window = &resampler->windows[channel];            \
                                                                               \
        int const index = resampler->channel_count * target_frame + channel;   \
                                                                               \
        float const * const slice = window->samples + resampler->index;        \
                                                                               \
        target_samples[index] = interpolate_##KERNEL_SIZE##p_##INTERPOLATION   \
        (                                                                      \
            blend,                                                             \
            slice                                                              \
        );                                                                     \
    }                                                                          \
}                                                                              \

RESAMPLE_SINC_FRAME_IMPL( 8, linear) // => `resample_sinc_frame_8p_linear`
RESAMPLE_SINC_FRAME_IMPL(16, linear) // => `resample_sinc_frame_16p_linear`
RESAMPLE_SINC_FRAME_IMPL(24, linear) // => `resample_sinc_frame_24p_linear`
RESAMPLE_SINC_FRAME_IMPL(32, linear) // => `resample_sinc_frame_32p_linear`
RESAMPLE_SINC_FRAME_IMPL( 8, cubic ) // => `resample_sinc_frame_8p_cubic`
RESAMPLE_SINC_FRAME_IMPL(16, cubic ) // => `resample_sinc_frame_16p_cubic`
RESAMPLE_SINC_FRAME_IMPL(24, cubic ) // => `resample_sinc_frame_24p_cubic`
RESAMPLE_SINC_FRAME_IMPL(32, cubic ) // => `resample_sinc_frame_32p_cubic`

#define RESAMPLER_SINC_PUSH_IMPL(KERNEL_SIZE, INTERPOLATION)                   \
static int resampler_sinc_push_##KERNEL_SIZE##p_##INTERPOLATION                \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    Resampler_Sinc * const resampler = state;                                  \
                                                                               \
    int source_frame = 0, target_frame = 0;                                    \
                                                                               \
    while (source_frame < source_frame_count)                                  \
    {                                                                          \
        while (resampler->phase <= resampler->source_delta)                    \
        {                                                                      \
            double const blend = (resampler->phase - 1) * resampler->scale;    \
                                                                               \
            ASSERT(0.0f <= blend && blend < 1.0f);                             \
                                                                               \
            resample_sinc_frame_##KERNEL_SIZE##p_##INTERPOLATION               \
            (                                                                  \
                resampler,                                                     \
                target_frame,                                                  \
                target_samples,                                                \
                blend                                                          \
            );                                                                 \
                                                                               \
            resampler->phase += resampler->target_delta;                       \
                                                                               \
            target_frame++;                                                    \
        }                                                                      \
                                                                               \
        resampler_sinc_update_windows_##KERNEL_SIZE##p                         \
        (                                                                      \
            resampler,                                                         \
            source_frame,                                                      \
            source_samples                                                     \
        );                                                                     \
                                                                               \
        resampler->phase -= resampler->source_delta;                           \
                                                                               \
        source_frame++;                                                        \
    }                                                                          \
                                                                               \
    ASSERT(resampler->phase >= 0);                                             \
                                                                               \
    return target_frame;                                                       \
}                                                                              \

RESAMPLER_SINC_PUSH_IMPL( 8, linear) // => `resampler_sinc_push_8p_linear`
RESAMPLER_SINC_PUSH_IMPL(16, linear) // => `resampler_sinc_push_16p_linear`
RESAMPLER_SINC_PUSH_IMPL(24, linear) // => `resampler_sinc_push_24p_linear`
RESAMPLER_SINC_PUSH_IMPL(32, linear) // => `resampler_sinc_push_32p_linear`
RESAMPLER_SINC_PUSH_IMPL( 8, cubic) // => `resampler_sinc_push_8p_cubic`
RESAMPLER_SINC_PUSH_IMPL(16, cubic ) // => `resampler_sinc_push_16p_cubic`
RESAMPLER_SINC_PUSH_IMPL(24, cubic) // => `resampler_sinc_push_24p_cubic`
RESAMPLER_SINC_PUSH_IMPL(32, cubic ) // => `resampler_sinc_push_32p_cubic`

#define RESAMPLER_SINC_PULL_IMPL(KERNEL_SIZE, INTERPOLATION)                   \
static int resampler_sinc_pull_##KERNEL_SIZE##p_##INTERPOLATION                \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    Resampler_Sinc * const resampler = state;                                  \
                                                                               \
    int source_frame = 0, target_frame = 0;                                    \
                                                                               \
    while (target_frame < target_frame_count)                                  \
    {                                                                          \
        while (resampler->phase >= resampler->source_delta)                    \
        {                                                                      \
            resampler_sinc_update_windows_##KERNEL_SIZE##p                     \
            (                                                                  \
                resampler,                                                     \
                source_frame,                                                  \
                source_samples                                                 \
            );                                                                 \
                                                                               \
            resampler->phase -= resampler->source_delta;                       \
                                                                               \
            source_frame++;                                                    \
        }                                                                      \
                                                                               \
        double const blend = resampler->phase * resampler->scale;              \
                                                                               \
        ASSERT(0.0f <= blend && blend < 1.0f);                                 \
                                                                               \
        resample_sinc_frame_##KERNEL_SIZE##p_##INTERPOLATION                   \
        (                                                                      \
            resampler,                                                         \
            target_frame,                                                      \
            target_samples,                                                    \
            blend                                                              \
        );                                                                     \
                                                                               \
        resampler->phase += resampler->target_delta;                           \
                                                                               \
        target_frame++;                                                        \
    }                                                                          \
                                                                               \
    ASSERT(resampler->phase >= 0);                                             \
                                                                               \
    return source_frame;                                                       \
}                                                                              \

RESAMPLER_SINC_PULL_IMPL( 8, linear) // => `resampler_sinc_pull_8p_linear`
RESAMPLER_SINC_PULL_IMPL(16, linear) // => `resampler_sinc_pull_16p_linear`
RESAMPLER_SINC_PULL_IMPL(24, linear) // => `resampler_sinc_pull_24p_linear`
RESAMPLER_SINC_PULL_IMPL(32, linear) // => `resampler_sinc_pull_32p_linear`
RESAMPLER_SINC_PULL_IMPL( 8, cubic ) // => `resampler_sinc_pull_8p_cubic`
RESAMPLER_SINC_PULL_IMPL(16, cubic ) // => `resampler_sinc_pull_16p_cubic`
RESAMPLER_SINC_PULL_IMPL(24, cubic ) // => `resampler_sinc_pull_24p_cubic`
RESAMPLER_SINC_PULL_IMPL(32, cubic ) // => `resampler_sinc_pull_32p_cubic`

static Processor resampler_sinc_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration;

    int const * const mode = configuration;

    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);

    int const channel_count = source_format->channel_count;
    long const windows_size = channel_count * sizeof(Window);

    Resampler_Sinc * const resampler = alloc
    (
        allocator,
        sizeof(Resampler_Sinc) + windows_size
    );

    if (resampler)
    {
        static double const latencies [] =
        {
            [resampler_sinc_mode_8p_linear ] =  8.0 / 2.0 + 1.0,
            [resampler_sinc_mode_16p_linear] = 16.0 / 2.0 + 1.0,
            [resampler_sinc_mode_24p_linear] = 24.0 / 2.0 + 1.0,
            [resampler_sinc_mode_32p_linear] = 32.0 / 2.0 + 1.0,
            [resampler_sinc_mode_8p_cubic  ] =  8.0 / 2.0 + 1.0,
            [resampler_sinc_mode_16p_cubic ] = 16.0 / 2.0 + 1.0,
            [resampler_sinc_mode_24p_cubic ] = 24.0 / 2.0 + 1.0,
            [resampler_sinc_mode_32p_cubic ] = 32.0 / 2.0 + 1.0
        };

        resampler->source_delta = tick_rate / source_format->sampling_rate;
        resampler->target_delta = tick_rate / target_format->sampling_rate;
        resampler->phase = resampler->source_delta;
        resampler->latency = latencies[*mode] / source_format->sampling_rate;
        resampler->scale = 1.0 / resampler->source_delta;
        resampler->channel_count = channel_count;
        resampler->index = 0;
    }

    for (int channel = 0; channel < channel_count; channel++)
    {
        static long const window_sizes [] =
        {
            [resampler_sinc_mode_8p_linear ] = window_8p_size,
            [resampler_sinc_mode_16p_linear] = window_16p_size,
            [resampler_sinc_mode_24p_linear] = window_24p_size,
            [resampler_sinc_mode_32p_linear] = window_32p_size,
            [resampler_sinc_mode_8p_cubic  ] = window_8p_size,
            [resampler_sinc_mode_16p_cubic ] = window_16p_size,
            [resampler_sinc_mode_24p_cubic ] = window_24p_size,
            [resampler_sinc_mode_32p_cubic ] = window_32p_size
        };

        float * const samples = alloc(allocator, window_sizes[*mode]);

        if (resampler)
        {
            resampler->windows[channel].samples = samples;
        }

        if (samples)
        {
            memset(samples, 0, window_sizes[*mode]);
        }
    }

    int (* const callbacks [][2])(void *, float *, float *, int) =
    {
        { resampler_sinc_push_8p_linear , resampler_sinc_pull_8p_linear  },
        { resampler_sinc_push_16p_linear, resampler_sinc_pull_16p_linear },
        { resampler_sinc_push_24p_linear, resampler_sinc_pull_24p_linear },
        { resampler_sinc_push_32p_linear, resampler_sinc_pull_32p_linear },
        { resampler_sinc_push_8p_cubic  , resampler_sinc_pull_8p_cubic   },
        { resampler_sinc_push_16p_cubic , resampler_sinc_pull_16p_cubic  },
        { resampler_sinc_push_24p_cubic , resampler_sinc_pull_24p_cubic  },
        { resampler_sinc_push_32p_cubic , resampler_sinc_pull_32p_cubic  }
    };

    return (Processor)
    {
        .pushed_target_frame_count = resampler_sinc_target_frame_count,
        .pulled_source_frame_count = resampler_sinc_source_frame_count,
        .push = callbacks[*mode][0],
        .pull = callbacks[*mode][1],
        .state = resampler
    };
}

static void resampler_sinc_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (target_format->block_size == 0)
    {
        target_format->block_size = 1;
    }

    ASSERT(target_format->block_size == 1);

    if (source_format->sampling_rate > 0 && target_format->sampling_rate > 0)
    {
        ASSERT(source_format->sampling_rate / target_format->sampling_rate < 2);
        ASSERT(target_format->sampling_rate / source_format->sampling_rate < 2);

        bool const source_format_resolved = source_format->max_block_count > 0;
        bool const target_format_resolved = target_format->max_block_count > 0;

        if (source_format->block_size > 0 && target_format->block_size > 0)
        {
            if (!source_format_resolved && target_format_resolved)
            {
                long count = target_format->max_block_count;

                count *= target_format->block_size;
                count *= source_format->sampling_rate;
                count /= target_format->sampling_rate;
                count /= source_format->block_size;

                source_format->max_block_count = (int)count;
            }
            else if (source_format_resolved && !target_format_resolved)
            {
                long count = source_format->max_block_count;

                count *= source_format->block_size;
                count *= target_format->sampling_rate;
                count += source_format->sampling_rate - 1;
                count /= source_format->sampling_rate;
                count /= target_format->block_size;

                target_format->max_block_count = (int)count;
            }

            ASSERT
            (
                 source_format->block_size
                    * source_format->max_block_count
                        * target_format->sampling_rate
              <
                 target_format->block_size
                    * target_format->max_block_count
                        * source_format->sampling_rate
            );

            ASSERT
            (
                 source_format->block_size
                    * (source_format->max_block_count + 1)
                        * target_format->sampling_rate
              >=
                 target_format->block_size
                    * target_format->max_block_count
                        * source_format->sampling_rate
            );
        }
    }

    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
}

#define RESAMPLER_SINC_IMPL(KERNEL_SIZE, INTERPOLATION)                        \
static const Pass resampler_sinc_##KERNEL_SIZE##p_##INTERPOLATION =            \
{                                                                              \
    .create_processor = resampler_sinc_create_processor,                       \
    .restrain_formats = resampler_sinc_restrain_formats,                       \
    .configuration = &resampler_sinc_mode_##KERNEL_SIZE##p_##INTERPOLATION     \
};                                                                             \

RESAMPLER_SINC_IMPL( 8, linear) // => `resampler_sinc_8p_linear`
RESAMPLER_SINC_IMPL(16, linear) // => `resampler_sinc_16p_linear`
RESAMPLER_SINC_IMPL(24, linear) // => `resampler_sinc_24p_linear`
RESAMPLER_SINC_IMPL(32, linear) // => `resampler_sinc_32p_linear`
RESAMPLER_SINC_IMPL( 8, cubic ) // => `resampler_sinc_8p_cubic`
RESAMPLER_SINC_IMPL(16, cubic ) // => `resampler_sinc_16p_cubic`
RESAMPLER_SINC_IMPL(24, cubic ) // => `resampler_sinc_24p_cubic`
RESAMPLER_SINC_IMPL(32, cubic ) // => `resampler_sinc_32p_cubic`

/* -------------------------------------------------------------------------- */

#pragma mark - 2X FIR Resampling

static float const coeffs_2x_fir_16 [8];
static float const coeffs_2x_fir_32 [16];
static float const coeffs_2x_fir_48 [24];
static float const coeffs_2x_fir_64 [32];
static float const coeffs_2x_fir_80 [40];
static float const coeffs_2x_fir_96 [48];
static float const coeffs_2x_fir_256 [128];

static int const resampler_2x_fir_order_16 = 0;
static int const resampler_2x_fir_order_32 = 1;
static int const resampler_2x_fir_order_48 = 2;
static int const resampler_2x_fir_order_64 = 3;
static int const resampler_2x_fir_order_80 = 4;
static int const resampler_2x_fir_order_96 = 5;
static int const resampler_2x_fir_order_256 = 6;

typedef struct Resampler_2X_FIR
{
    double latency;
    int channel_count;
    int index;
    Window windows [];
}
Resampler_2X_FIR;

static int resampler_2x_fir_frame_count_mul_2
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Resampler_2X_FIR const * const resampler = state;

    *latency += resampler->latency;

    return frame_count * 2;
}

static int resampler_2x_fir_frame_count_div_2
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Resampler_2X_FIR const * const resampler = state;

    *latency += resampler->latency;

    return frame_count / 2;
}

#define UPSAMPLE_2X_FIR_IMPL(ORDER, HALF, QUARTER)                             \
static void upsample_2x_fir_##ORDER                                            \
    (                                                                          \
        Resampler_2X_FIR * const resampler,                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const channel_count = resampler->channel_count;                        \
                                                                               \
    for (int frame = 0; frame < source_frame_count; frame++)                   \
    {                                                                          \
        for (int channel = 0; channel < channel_count; channel++)              \
        {                                                                      \
            int const index = frame * channel_count + channel;                 \
                                                                               \
            float const input_sample = source_samples[index];                  \
                                                                               \
            Window * const window = &resampler->windows[channel];              \
                                                                               \
            window_##HALF##p_write(window, input_sample, resampler->index);    \
                                                                               \
            float const * const slice = window->samples + resampler->index;    \
                                                                               \
            float const * const coeffs = coeffs_2x_fir_##ORDER;                \
                                                                               \
            float const acc_0 = slice[QUARTER] * 0.5f;                         \
            float const acc_1 = convolve_##HALF(coeffs, slice + 1);            \
                                                                               \
            float const output_sample_0 = acc_0 * 2.0f;                        \
            float const output_sample_1 = acc_1 * 2.0f;                        \
                                                                               \
            int const index_0 = (frame * 2 + 0) * channel_count + channel;     \
            int const index_1 = (frame * 2 + 1) * channel_count + channel;     \
                                                                               \
            target_samples[index_0] = output_sample_0;                         \
            target_samples[index_1] = output_sample_1;                         \
        }                                                                      \
                                                                               \
        resampler->index = window_##HALF##p_next_index(resampler->index);      \
    }                                                                          \
}                                                                              \

UPSAMPLE_2X_FIR_IMPL(16,  8,  4) // => `upsample_2x_fir_16`
UPSAMPLE_2X_FIR_IMPL(32, 16,  8) // => `upsample_2x_fir_32`
UPSAMPLE_2X_FIR_IMPL(48, 24, 12) // => `upsample_2x_fir_48`
UPSAMPLE_2X_FIR_IMPL(64, 32, 16) // => `upsample_2x_fir_64`
UPSAMPLE_2X_FIR_IMPL(80, 40, 20) // => `upsample_2x_fir_80`
UPSAMPLE_2X_FIR_IMPL(96, 48, 24) // => `upsample_2x_fir_96`
UPSAMPLE_2X_FIR_IMPL(256, 128, 64) // => `upsample_2x_fir_256`

#define DOWNSAMPLE_2X_FIR_IMPL(ORDER, HALF, QUARTER)                           \
static void downsample_2x_fir_##ORDER                                          \
    (                                                                          \
        Resampler_2X_FIR * const resampler,                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const channel_count = resampler->channel_count;                        \
                                                                               \
    for (int frame = 0; frame < target_frame_count; frame++)                   \
    {                                                                          \
        for (int channel = 0; channel < channel_count; channel++)              \
        {                                                                      \
            int const index_0 = (frame * 2 + 0) * channel_count + channel;     \
            int const index_1 = (frame * 2 + 1) * channel_count + channel;     \
                                                                               \
            float const input_sample_0 = source_samples[index_0];              \
            float const input_sample_1 = source_samples[index_1];              \
                                                                               \
            Window * const window_0 = &resampler->windows[channel * 2 + 0];    \
            Window * const window_1 = &resampler->windows[channel * 2 + 1];    \
                                                                               \
            float const * const slice_0 = window_0->samples + resampler->index;\
            float const * const slice_1 = window_1->samples + resampler->index;\
                                                                               \
            float const * const coeffs = coeffs_2x_fir_##ORDER;                \
                                                                               \
            float const acc_0 = slice_0[QUARTER] * 0.5f;                       \
            float const acc_1 = convolve_##HALF(coeffs, slice_1);              \
                                                                               \
            window_##HALF##p_write(window_0, input_sample_0, resampler->index);\
            window_##HALF##p_write(window_1, input_sample_1, resampler->index);\
                                                                               \
            float const output_sample = acc_0 + acc_1;                         \
                                                                               \
            int const index = frame * channel_count + channel;                 \
                                                                               \
            target_samples[index] = output_sample;                             \
        }                                                                      \
                                                                               \
        resampler->index = window_##HALF##p_next_index(resampler->index);      \
    }                                                                          \
}                                                                              \

DOWNSAMPLE_2X_FIR_IMPL(16,  8,  4) // => `downsample_2x_fir_16`
DOWNSAMPLE_2X_FIR_IMPL(32, 16,  8) // => `downsample_2x_fir_32`
DOWNSAMPLE_2X_FIR_IMPL(48, 24, 12) // => `downsample_2x_fir_48`
DOWNSAMPLE_2X_FIR_IMPL(64, 32, 16) // => `downsample_2x_fir_64`
DOWNSAMPLE_2X_FIR_IMPL(80, 40, 20) // => `downsample_2x_fir_80`
DOWNSAMPLE_2X_FIR_IMPL(96, 48, 24) // => `downsample_2x_fir_96`
DOWNSAMPLE_2X_FIR_IMPL(256, 128, 64) // => `downsample_2x_fir_256`

#define UPSAMPLER_2X_FIR_PUSH_IMPL(ORDER)                                      \
static int upsampler_2x_fir_push_##ORDER                                       \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const target_frame_count = source_frame_count * 2;                     \
                                                                               \
    upsample_2x_fir_##ORDER                                                    \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        source_frame_count                                                     \
    );                                                                         \
                                                                               \
    return target_frame_count;                                                 \
}                                                                              \

UPSAMPLER_2X_FIR_PUSH_IMPL(16) // => `upsampler_2x_fir_push_16`
UPSAMPLER_2X_FIR_PUSH_IMPL(32) // => `upsampler_2x_fir_push_32`
UPSAMPLER_2X_FIR_PUSH_IMPL(48) // => `upsampler_2x_fir_push_48`
UPSAMPLER_2X_FIR_PUSH_IMPL(64) // => `upsampler_2x_fir_push_64`
UPSAMPLER_2X_FIR_PUSH_IMPL(80) // => `upsampler_2x_fir_push_80`
UPSAMPLER_2X_FIR_PUSH_IMPL(96) // => `upsampler_2x_fir_push_96`
UPSAMPLER_2X_FIR_PUSH_IMPL(256) // => `upsampler_2x_fir_push_256`

#define UPSAMPLER_2X_FIR_PULL_IMPL(ORDER)                                      \
static int upsampler_2x_fir_pull_##ORDER                                       \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const source_frame_count = target_frame_count / 2;                     \
                                                                               \
    upsample_2x_fir_##ORDER                                                    \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        source_frame_count                                                     \
    );                                                                         \
                                                                               \
    return source_frame_count;                                                 \
}                                                                              \

UPSAMPLER_2X_FIR_PULL_IMPL(16) // => `upsampler_2x_fir_pull_16`
UPSAMPLER_2X_FIR_PULL_IMPL(32) // => `upsampler_2x_fir_pull_32`
UPSAMPLER_2X_FIR_PULL_IMPL(48) // => `upsampler_2x_fir_pull_48`
UPSAMPLER_2X_FIR_PULL_IMPL(64) // => `upsampler_2x_fir_pull_64`
UPSAMPLER_2X_FIR_PULL_IMPL(80) // => `upsampler_2x_fir_pull_80`
UPSAMPLER_2X_FIR_PULL_IMPL(96) // => `upsampler_2x_fir_pull_96`
UPSAMPLER_2X_FIR_PULL_IMPL(256) // => `upsampler_2x_fir_pull_256`

#define DOWNSAMPLER_2X_FIR_PUSH_IMPL(ORDER)                                    \
static int downsampler_2x_fir_push_##ORDER                                     \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const target_frame_count = source_frame_count / 2;                     \
                                                                               \
    downsample_2x_fir_##ORDER                                                  \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        target_frame_count                                                     \
    );                                                                         \
                                                                               \
    return target_frame_count;                                                 \
}                                                                              \

DOWNSAMPLER_2X_FIR_PUSH_IMPL(16) // => `downsampler_2x_fir_push_16`
DOWNSAMPLER_2X_FIR_PUSH_IMPL(32) // => `downsampler_2x_fir_push_32`
DOWNSAMPLER_2X_FIR_PUSH_IMPL(48) // => `downsampler_2x_fir_push_48`
DOWNSAMPLER_2X_FIR_PUSH_IMPL(64) // => `downsampler_2x_fir_push_64`
DOWNSAMPLER_2X_FIR_PUSH_IMPL(80) // => `downsampler_2x_fir_push_80`
DOWNSAMPLER_2X_FIR_PUSH_IMPL(96) // => `downsampler_2x_fir_push_96`
DOWNSAMPLER_2X_FIR_PUSH_IMPL(256) // => `downsampler_2x_fir_push_256`

#define DOWNSAMPLER_2X_FIR_PULL_IMPL(ORDER)                                    \
static int downsampler_2x_fir_pull_##ORDER                                     \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const source_frame_count = target_frame_count * 2;                     \
                                                                               \
    downsample_2x_fir_##ORDER                                                  \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        target_frame_count                                                     \
    );                                                                         \
                                                                               \
    return source_frame_count;                                                 \
}                                                                              \

DOWNSAMPLER_2X_FIR_PULL_IMPL(16) // => `downsampler_2x_fir_pull_16`
DOWNSAMPLER_2X_FIR_PULL_IMPL(32) // => `downsampler_2x_fir_pull_32`
DOWNSAMPLER_2X_FIR_PULL_IMPL(48) // => `downsampler_2x_fir_pull_48`
DOWNSAMPLER_2X_FIR_PULL_IMPL(64) // => `downsampler_2x_fir_pull_64`
DOWNSAMPLER_2X_FIR_PULL_IMPL(80) // => `downsampler_2x_fir_pull_80`
DOWNSAMPLER_2X_FIR_PULL_IMPL(96) // => `downsampler_2x_fir_pull_96`
DOWNSAMPLER_2X_FIR_PULL_IMPL(256) // => `downsampler_2x_fir_pull_256`

static Processor upsampler_2x_fir_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    int const * const order = configuration;

    ASSERT(source_format->sampling_rate * 2 == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);
    ASSERT(source_format->block_size * 2 == target_format->block_size);
    ASSERT(target_format->max_block_count == target_format->max_block_count);

    long const windows_size = source_format->channel_count * sizeof(Window);

    Resampler_2X_FIR * const resampler = alloc
    (
        allocator,
        sizeof(Resampler_2X_FIR) + windows_size
    );

    if (resampler)
    {
        double const latencies [] =
        {
            [resampler_2x_fir_order_16] =  8.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_32] = 16.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_48] = 24.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_64] = 32.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_80] = 40.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_96] = 48.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_256] = 128.0 / target_format->sampling_rate
        };

        resampler->latency = latencies[*order];
        resampler->channel_count = source_format->channel_count;
        resampler->index = 0;
    }

    for (int channel = 0; channel < source_format->channel_count; channel++)
    {
        long const window_sizes [] =
        {
            [resampler_2x_fir_order_16] = window_8p_size,
            [resampler_2x_fir_order_32] = window_16p_size,
            [resampler_2x_fir_order_48] = window_24p_size,
            [resampler_2x_fir_order_64] = window_32p_size,
            [resampler_2x_fir_order_80] = window_40p_size,
            [resampler_2x_fir_order_96] = window_48p_size,
            [resampler_2x_fir_order_256] = window_128p_size
        };

        float * const samples = alloc(allocator, window_sizes[*order]);

        if (resampler)
        {
            resampler->windows[channel].samples = samples;
        }

        if (samples)
        {
            memset(samples, 0, window_sizes[*order]);
        }
    }

    int (* const callbacks [][2])(void *, float *, float *, int) =
    {
        { upsampler_2x_fir_push_16, upsampler_2x_fir_pull_16 },
        { upsampler_2x_fir_push_32, upsampler_2x_fir_pull_32 },
        { upsampler_2x_fir_push_48, upsampler_2x_fir_pull_48 },
        { upsampler_2x_fir_push_64, upsampler_2x_fir_pull_64 },
        { upsampler_2x_fir_push_80, upsampler_2x_fir_pull_80 },
        { upsampler_2x_fir_push_96, upsampler_2x_fir_pull_96 },
        { upsampler_2x_fir_push_256, upsampler_2x_fir_pull_256 }
    };

    return (Processor)
    {
        .pushed_target_frame_count = resampler_2x_fir_frame_count_mul_2,
        .pulled_source_frame_count = resampler_2x_fir_frame_count_div_2,
        .push = callbacks[*order][0],
        .pull = callbacks[*order][1],
        .state = resampler
    };
}

static Processor downsampler_2x_fir_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    int const * const order = configuration;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate * 2);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);
    ASSERT(source_format->block_size == target_format->block_size * 2);
    ASSERT(target_format->max_block_count == target_format->max_block_count);

    long const windows_size = source_format->channel_count * 2 * sizeof(Window);

    Resampler_2X_FIR * const resampler = alloc
    (
        allocator,
        sizeof(Resampler_2X_FIR) + windows_size
    );

    if (resampler)
    {
        double const latencies [] =
        {
            [resampler_2x_fir_order_16] =  8.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_32] = 16.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_48] = 24.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_64] = 32.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_80] = 40.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_96] = 48.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_256] = 128.0 / source_format->sampling_rate
        };

        resampler->latency = latencies[*order];
        resampler->channel_count = source_format->channel_count;
        resampler->index = 0;
    }

    for (int channel = 0; channel < source_format->channel_count; channel++)
    {
        long const window_sizes [] =
        {
            [resampler_2x_fir_order_16] = window_8p_size,
            [resampler_2x_fir_order_32] = window_16p_size,
            [resampler_2x_fir_order_48] = window_24p_size,
            [resampler_2x_fir_order_64] = window_32p_size,
            [resampler_2x_fir_order_80] = window_40p_size,
            [resampler_2x_fir_order_96] = window_48p_size,
            [resampler_2x_fir_order_256] = window_128p_size
        };

        float * const samples_0 = alloc(allocator, window_sizes[*order]);
        float * const samples_1 = alloc(allocator, window_sizes[*order]);

        if (resampler)
        {
            resampler->windows[channel * 2 + 0].samples = samples_0;
            resampler->windows[channel * 2 + 1].samples = samples_1;
        }

        if (samples_0 && samples_1)
        {
            memset(samples_0, 0, window_sizes[*order]);
            memset(samples_1, 0, window_sizes[*order]);
        }
    }

    int (* const callbacks [][2])(void *, float *, float *, int) =
    {
        { downsampler_2x_fir_push_16, downsampler_2x_fir_pull_16 },
        { downsampler_2x_fir_push_32, downsampler_2x_fir_pull_32 },
        { downsampler_2x_fir_push_48, downsampler_2x_fir_pull_48 },
        { downsampler_2x_fir_push_64, downsampler_2x_fir_pull_64 },
        { downsampler_2x_fir_push_80, downsampler_2x_fir_pull_80 },
        { downsampler_2x_fir_push_96, downsampler_2x_fir_pull_96 },
        { downsampler_2x_fir_push_256, downsampler_2x_fir_pull_256 }
    };

    return (Processor)
    {
        .pushed_target_frame_count = resampler_2x_fir_frame_count_div_2,
        .pulled_source_frame_count = resampler_2x_fir_frame_count_mul_2,
        .push = callbacks[*order][0],
        .pull = callbacks[*order][1],
        .state = resampler
    };
}

static void upsampler_2x_fir_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->sampling_rate == 0 && target_format->sampling_rate > 0)
    {
        source_format->sampling_rate = target_format->sampling_rate / 2;
    }

    if (source_format->sampling_rate > 0 && target_format->sampling_rate == 0)
    {
        target_format->sampling_rate = source_format->sampling_rate * 2;
    }

    ASSERT(source_format->sampling_rate * 2 == target_format->sampling_rate);

    if (source_format->block_size == 0 && target_format->block_size > 0)
    {
        source_format->block_size = target_format->block_size / 2;
    }

    if (source_format->block_size > 0 && target_format->block_size == 0)
    {
        target_format->block_size = source_format->block_size * 2;
    }

    ASSERT(source_format->block_size * 2 == target_format->block_size);

    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

static void downsampler_2x_fir_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->sampling_rate == 0 && target_format->sampling_rate > 0)
    {
        source_format->sampling_rate = target_format->sampling_rate * 2;
    }

    if (source_format->sampling_rate > 0 && target_format->sampling_rate == 0)
    {
        target_format->sampling_rate = source_format->sampling_rate / 2;
    }

    ASSERT(source_format->sampling_rate == target_format->sampling_rate * 2);

    if (source_format->block_size == 0 && target_format->block_size > 0)
    {
        source_format->block_size = target_format->block_size * 2;
    }

    if (source_format->block_size > 0 && target_format->block_size == 0)
    {
        target_format->block_size = source_format->block_size / 2;
    }

    ASSERT(source_format->block_size == target_format->block_size * 2);

    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

#define UPSAMPLER_2X_FIR_IMPL(ORDER)                                           \
static const Pass upsampler_2x_fir_##ORDER =                                   \
{                                                                              \
    .create_processor = upsampler_2x_fir_create_processor,                     \
    .restrain_formats = upsampler_2x_fir_restrain_formats,                     \
    .configuration = &resampler_2x_fir_order_##ORDER                           \
};                                                                             \

UPSAMPLER_2X_FIR_IMPL(16) // <= `upsampler_2x_fir_16`
UPSAMPLER_2X_FIR_IMPL(32) // <= `upsampler_2x_fir_32`
UPSAMPLER_2X_FIR_IMPL(48) // <= `upsampler_2x_fir_48`
UPSAMPLER_2X_FIR_IMPL(64) // <= `upsampler_2x_fir_64`
UPSAMPLER_2X_FIR_IMPL(80) // <= `upsampler_2x_fir_80`
UPSAMPLER_2X_FIR_IMPL(96) // <= `upsampler_2x_fir_96`
UPSAMPLER_2X_FIR_IMPL(256) // <= `upsampler_2x_fir_256`

#define DOWNSAMPLER_2X_FIR_IMPL(ORDER)                                         \
static const Pass downsampler_2x_fir_##ORDER =                                 \
{                                                                              \
    .create_processor = downsampler_2x_fir_create_processor,                   \
    .restrain_formats = downsampler_2x_fir_restrain_formats,                   \
    .configuration = &resampler_2x_fir_order_##ORDER                           \
};                                                                             \

DOWNSAMPLER_2X_FIR_IMPL(16) // <= `downsampler_2x_fir_16`
DOWNSAMPLER_2X_FIR_IMPL(32) // <= `downsampler_2x_fir_32`
DOWNSAMPLER_2X_FIR_IMPL(48) // <= `downsampler_2x_fir_48`
DOWNSAMPLER_2X_FIR_IMPL(64) // <= `downsampler_2x_fir_64`
DOWNSAMPLER_2X_FIR_IMPL(80) // <= `downsampler_2x_fir_80`
DOWNSAMPLER_2X_FIR_IMPL(96) // <= `downsampler_2x_fir_96`
DOWNSAMPLER_2X_FIR_IMPL(256) // <= `downsampler_2x_fir_256`

/* -------------------------------------------------------------------------- */

#pragma mark - Slicing

typedef struct Slicer
{
    double sampling_time;
    int channel_count;
    int block_size;
    int index;
    _Alignas(cache_line_size) float window [];
}
Slicer;

static int slicer_target_frame_count
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Slicer const * const slicer = state;

    int const block_count = (slicer->index + frame_count) / slicer->block_size;

    *latency += slicer->index * slicer->sampling_time;

    return block_count * slicer->block_size;
}

static int slicer_source_frame_count
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Slicer const * const slicer = state;

    int const block_count = (slicer->index + frame_count) / slicer->block_size;

    *latency += (slicer->block_size - slicer->index) * slicer->sampling_time;

    return block_count * slicer->block_size;
}

static int slicer_push
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const source_frame_count
    )
{
    Slicer * const slicer = state;

    int const block_size = slicer->block_size;

    int source_frame = 0, target_frame_count = 0;

    if (source_frame < source_frame_count)
    {
        int const frames_left_in_block = block_size - slicer->index;
        int const frame_count = min(frames_left_in_block, source_frame_count);

        memcpy
        (
            slicer->window + slicer->index * slicer->channel_count,
            source_samples + source_frame * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        source_frame += frame_count;
        slicer->index += frame_count;
    }

    if (slicer->index == block_size)
    {
        int const frames_left = source_frame_count - source_frame;
        int const block_count = frames_left / block_size;

        target_frame_count = (block_count + 1) * block_size;

        memcpy
        (
            target_samples,
            slicer->window,
            block_size * slicer->channel_count * sizeof(float)
        );

        memcpy
        (
            target_samples + block_size * slicer->channel_count,
            source_samples + source_frame * slicer->channel_count,
            block_count * block_size * slicer->channel_count * sizeof(float)
        );

        source_frame += block_count * block_size;
        slicer->index = 0;
    }

    if (source_frame < source_frame_count)
    {
        int const frames_left = source_frame_count - source_frame;
        int const frame_count = frames_left;

        memcpy
        (
            slicer->window + slicer->index * slicer->channel_count,
            source_samples + source_frame * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        source_frame += frame_count;
        slicer->index += frame_count;
    }

    ASSERT(source_frame == source_frame_count);

    return target_frame_count;
}

static int slicer_pull
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const target_frame_count
    )
{
    Slicer * const slicer = state;

    int const block_size = slicer->block_size;

    int target_frame = 0, source_frame_count = 0;

    if (target_frame < target_frame_count)
    {
        int const frames_left_in_block = block_size - slicer->index;
        int const frame_count = min(frames_left_in_block, target_frame_count);

        memcpy
        (
            target_samples + target_frame * slicer->channel_count,
            slicer->window + slicer->index * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        target_frame += frame_count;
        slicer->index += frame_count;
    }

    if (slicer->index == block_size)
    {
        int const frames_left = target_frame_count - target_frame;
        int const block_count = frames_left / block_size;

        source_frame_count = (block_count + 1) * block_size;

        memcpy
        (
            target_samples + target_frame * slicer->channel_count,
            source_samples,
            block_count * block_size * slicer->channel_count * sizeof(float)
        );

        memcpy
        (
            slicer->window,
            source_samples + block_count * block_size * slicer->channel_count,
            block_size * slicer->channel_count * sizeof(float)
        );

        target_frame += block_count * block_size;
        slicer->index = 0;
    }

    if (target_frame < target_frame_count)
    {
        int const frames_left = target_frame_count - target_frame;
        int const frame_count = frames_left;

        memcpy
        (
            target_samples + target_frame * slicer->channel_count,
            slicer->window + slicer->index * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        target_frame += frame_count;
        slicer->index += frame_count;
    }

    ASSERT(target_frame == target_frame_count);

    return source_frame_count;
}

static Processor slicer_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);

    int const source_block_size = source_format->block_size;
    int const target_block_size = target_format->block_size;

    int const channel_count = source_format->channel_count;
    int const max_block_size = max(source_block_size, target_block_size);
    int const window_size = max_block_size * channel_count * sizeof(float);

    Slicer * const slicer = alloc
    (
        allocator,
        sizeof(Slicer) + window_size
    );

    if (slicer)
    {
        slicer->sampling_time = 1.0 / source_format->sampling_rate;
        slicer->channel_count = channel_count;
        slicer->block_size = max_block_size;
        slicer->index = max_block_size / 2;
        memset(slicer->window, 0, window_size);
    }

    return (Processor)
    {
        .pushed_target_frame_count = slicer_target_frame_count,
        .pulled_source_frame_count = slicer_source_frame_count,
        .push = slicer_push,
        .pull = slicer_pull,
        .state = slicer
    };
}

static void slicer_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->block_size > 0 && target_format->block_size > 0)
    {
        bool const source_format_resolved = source_format->max_block_count > 0;
        bool const target_format_resolved = target_format->max_block_count > 0;

        if (!source_format_resolved && target_format_resolved)
        {
            long count = target_format->max_block_count;

            count *= target_format->block_size;
            count /= source_format->block_size;

            source_format->max_block_count = (int)count;
        }
        else if (source_format_resolved && !target_format_resolved)
        {
            long count = source_format->max_block_count;

            count *= source_format->block_size;
            count += target_format->block_size - 1;
            count /= target_format->block_size;

            target_format->max_block_count = (int)count;
        }

        ASSERT
        (
             source_format->block_size * source_format->max_block_count
          <=
             target_format->block_size * target_format->max_block_count
        );

        ASSERT
        (
             source_format->block_size * (source_format->max_block_count + 1)
          >=
             target_format->block_size * target_format->max_block_count
        );
    }

    unify(&source_format->sampling_rate, &target_format->sampling_rate);
    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
}

static const Pass slicer =
{
    .create_processor = slicer_create_processor,
    .restrain_formats = slicer_restrain_formats
};

/* -------------------------------------------------------------------------- */

#pragma mark - Passes

enum
{
    pass_tag_interleaver,
    pass_tag_deinterleaver,
    pass_tag_upsampler_2x_fir_16,
    pass_tag_upsampler_2x_fir_32,
    pass_tag_upsampler_2x_fir_48,
    pass_tag_upsampler_2x_fir_64,
    pass_tag_upsampler_2x_fir_80,
    pass_tag_upsampler_2x_fir_96,
    pass_tag_upsampler_2x_fir_256,
    pass_tag_downsampler_2x_fir_16,
    pass_tag_downsampler_2x_fir_32,
    pass_tag_downsampler_2x_fir_48,
    pass_tag_downsampler_2x_fir_64,
    pass_tag_downsampler_2x_fir_80,
    pass_tag_downsampler_2x_fir_96,
    pass_tag_downsampler_2x_fir_256,
    pass_tag_resampler_sinc_8p_linear,
    pass_tag_resampler_sinc_16p_linear,
    pass_tag_resampler_sinc_24p_linear,
    pass_tag_resampler_sinc_32p_linear,
    pass_tag_resampler_sinc_8p_cubic,
    pass_tag_resampler_sinc_16p_cubic,
    pass_tag_resampler_sinc_24p_cubic,
    pass_tag_resampler_sinc_32p_cubic,
    pass_tag_slicer
};

static Pass make_pass (short pass_tag)
{
    switch (pass_tag)
    {
      case pass_tag_interleaver: return interleaver;
      case pass_tag_deinterleaver: return deinterleaver;
      case pass_tag_upsampler_2x_fir_16: return upsampler_2x_fir_16;
      case pass_tag_upsampler_2x_fir_32: return upsampler_2x_fir_32;
      case pass_tag_upsampler_2x_fir_48: return upsampler_2x_fir_48;
      case pass_tag_upsampler_2x_fir_64: return upsampler_2x_fir_64;
      case pass_tag_upsampler_2x_fir_80: return upsampler_2x_fir_80;
      case pass_tag_upsampler_2x_fir_96: return upsampler_2x_fir_96;
      case pass_tag_upsampler_2x_fir_256: return upsampler_2x_fir_256;
      case pass_tag_downsampler_2x_fir_16: return downsampler_2x_fir_16;
      case pass_tag_downsampler_2x_fir_32: return downsampler_2x_fir_32;
      case pass_tag_downsampler_2x_fir_48: return downsampler_2x_fir_48;
      case pass_tag_downsampler_2x_fir_64: return downsampler_2x_fir_64;
      case pass_tag_downsampler_2x_fir_80: return downsampler_2x_fir_80;
      case pass_tag_downsampler_2x_fir_96: return downsampler_2x_fir_96;
      case pass_tag_downsampler_2x_fir_256: return downsampler_2x_fir_256;
      case pass_tag_resampler_sinc_8p_linear: return resampler_sinc_8p_linear;
      case pass_tag_resampler_sinc_16p_linear: return resampler_sinc_16p_linear;
      case pass_tag_resampler_sinc_24p_linear: return resampler_sinc_24p_linear;
      case pass_tag_resampler_sinc_32p_linear: return resampler_sinc_32p_linear;
      case pass_tag_resampler_sinc_8p_cubic: return resampler_sinc_8p_cubic;
      case pass_tag_resampler_sinc_16p_cubic: return resampler_sinc_16p_cubic;
      case pass_tag_resampler_sinc_24p_cubic: return resampler_sinc_24p_cubic;
      case pass_tag_resampler_sinc_32p_cubic: return resampler_sinc_32p_cubic;
      case pass_tag_slicer: return slicer;
    }

    ASSERT(false);

    return (Pass) { 0 };
}

short const inverted_passes [] =
{
    [pass_tag_interleaver] = pass_tag_deinterleaver,
    [pass_tag_deinterleaver] = pass_tag_interleaver,
    [pass_tag_upsampler_2x_fir_16] = pass_tag_downsampler_2x_fir_16,
    [pass_tag_upsampler_2x_fir_32] = pass_tag_downsampler_2x_fir_32,
    [pass_tag_upsampler_2x_fir_48] = pass_tag_downsampler_2x_fir_48,
    [pass_tag_upsampler_2x_fir_64] = pass_tag_downsampler_2x_fir_64,
    [pass_tag_upsampler_2x_fir_80] = pass_tag_downsampler_2x_fir_80,
    [pass_tag_upsampler_2x_fir_96] = pass_tag_downsampler_2x_fir_96,
    [pass_tag_upsampler_2x_fir_256] = pass_tag_downsampler_2x_fir_256,
    [pass_tag_downsampler_2x_fir_16] = pass_tag_upsampler_2x_fir_16,
    [pass_tag_downsampler_2x_fir_32] = pass_tag_upsampler_2x_fir_32,
    [pass_tag_downsampler_2x_fir_48] = pass_tag_upsampler_2x_fir_48,
    [pass_tag_downsampler_2x_fir_64] = pass_tag_upsampler_2x_fir_64,
    [pass_tag_downsampler_2x_fir_80] = pass_tag_upsampler_2x_fir_80,
    [pass_tag_downsampler_2x_fir_96] = pass_tag_upsampler_2x_fir_96,
    [pass_tag_downsampler_2x_fir_256] = pass_tag_upsampler_2x_fir_256,
    [pass_tag_resampler_sinc_8p_linear] = pass_tag_resampler_sinc_8p_linear,
    [pass_tag_resampler_sinc_16p_linear] = pass_tag_resampler_sinc_16p_linear,
    [pass_tag_resampler_sinc_24p_linear] = pass_tag_resampler_sinc_24p_linear,
    [pass_tag_resampler_sinc_32p_linear] = pass_tag_resampler_sinc_32p_linear,
    [pass_tag_resampler_sinc_8p_cubic] = pass_tag_resampler_sinc_8p_cubic,
    [pass_tag_resampler_sinc_16p_cubic] = pass_tag_resampler_sinc_16p_cubic,
    [pass_tag_resampler_sinc_24p_cubic] = pass_tag_resampler_sinc_24p_cubic,
    [pass_tag_resampler_sinc_32p_cubic] = pass_tag_resampler_sinc_32p_cubic,
    [pass_tag_slicer] = pass_tag_slicer
};

static void invert_passes (short passes [const], int const pass_count)
{
    for (int index = 0; index < pass_count; index++)
    {
        passes[index] = inverted_passes[passes[index]];
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Conversion Resolution

enum { max_pass_count = 10, max_format_count = max_pass_count + 1 };

typedef struct Sampling_Rate_Conversion
{
    int up_2x_pass_count;
    int down_2x_pass_count;
    bool needs_fractional_resampling;
}
Sampling_Rate_Conversion;

static void resolve_sampling_rate_conversion
    (
        int const source_sampling_rate,
        int const target_sampling_rate,
        bool const oversample,
        Sampling_Rate_Conversion * const sampling_rate_conversion
    )
{
    int factor = 0;

    if (source_sampling_rate < target_sampling_rate)
    {
        while
        (
            (float)target_sampling_rate / (source_sampling_rate * (1 << factor))
          >
            (float)(source_sampling_rate * (2 << factor)) / target_sampling_rate
        )
        {
            factor += 1;
        }

        sampling_rate_conversion->up_2x_pass_count = factor;

        if (source_sampling_rate * (1 << factor) != target_sampling_rate)
        {
            sampling_rate_conversion->needs_fractional_resampling = true;

            if (oversample)
            {
                sampling_rate_conversion->up_2x_pass_count += 1;
                sampling_rate_conversion->down_2x_pass_count += 1;
            }
        }
    }
    else // target_sampling_rate > source_sampling_rate
    {
        while
        (
            (float)source_sampling_rate / (target_sampling_rate * (1 << factor))
          >
            (float)(target_sampling_rate * (2 << factor)) / source_sampling_rate
        )
        {
            factor += 1;
        }

        sampling_rate_conversion->down_2x_pass_count = factor;

        if (target_sampling_rate * (1 << factor) != source_sampling_rate)
        {
            sampling_rate_conversion->needs_fractional_resampling = true;

            if (oversample)
            {
                sampling_rate_conversion->up_2x_pass_count += 1;
                sampling_rate_conversion->down_2x_pass_count += 1;
            }
        }
    }
}

typedef struct DrB_Audio_Conversion
{
    Format formats [max_format_count];
    short passes [max_pass_count];
    int pass_count, format_count;
}
DrB_Audio_Conversion;

static int resolve_passes
    (
        Format const * const source_format,
        Format const * const target_format,
        DrB_Audio_Converter_Quality const quality,
        short passes [const]
    )
{
    enum { max_2x_pass_count = 6 };

    static short const upsamplers_2x [][max_2x_pass_count] =
    {
        [drb_audio_converter_quality_poor] =
        {
            pass_tag_upsampler_2x_fir_48, pass_tag_upsampler_2x_fir_32,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16
        },
        [drb_audio_converter_quality_fine] =
        {
            pass_tag_upsampler_2x_fir_64, pass_tag_upsampler_2x_fir_48,
            pass_tag_upsampler_2x_fir_32, pass_tag_upsampler_2x_fir_16,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16
        },
        [drb_audio_converter_quality_good] =
        {
            pass_tag_upsampler_2x_fir_80, pass_tag_upsampler_2x_fir_64,
            pass_tag_upsampler_2x_fir_48, pass_tag_upsampler_2x_fir_32,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16
        },
        [drb_audio_converter_quality_best] =
        {
            pass_tag_upsampler_2x_fir_96, pass_tag_upsampler_2x_fir_80,
            pass_tag_upsampler_2x_fir_64, pass_tag_upsampler_2x_fir_48,
            pass_tag_upsampler_2x_fir_32, pass_tag_upsampler_2x_fir_16
        }
    };

    static short const downsamplers_2x [][max_2x_pass_count] =
    {
        [drb_audio_converter_quality_poor] =
        {
            pass_tag_downsampler_2x_fir_48, pass_tag_downsampler_2x_fir_32,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16
        },
        [drb_audio_converter_quality_fine] =
        {
            pass_tag_downsampler_2x_fir_64, pass_tag_downsampler_2x_fir_48,
            pass_tag_downsampler_2x_fir_32, pass_tag_downsampler_2x_fir_16,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16
        },
        [drb_audio_converter_quality_good] =
        {
            pass_tag_downsampler_2x_fir_80, pass_tag_downsampler_2x_fir_64,
            pass_tag_downsampler_2x_fir_48, pass_tag_downsampler_2x_fir_32,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16
        },
        [drb_audio_converter_quality_best] =
        {
            pass_tag_downsampler_2x_fir_96, pass_tag_downsampler_2x_fir_80,
            pass_tag_downsampler_2x_fir_64, pass_tag_downsampler_2x_fir_48,
            pass_tag_downsampler_2x_fir_32, pass_tag_downsampler_2x_fir_16
        }
    };

    static short const resamplers [] =
    {
        [drb_audio_converter_quality_poor] = pass_tag_resampler_sinc_8p_cubic,
        [drb_audio_converter_quality_fine] = pass_tag_resampler_sinc_16p_cubic,
        [drb_audio_converter_quality_good] = pass_tag_resampler_sinc_24p_cubic,
        [drb_audio_converter_quality_best] = pass_tag_resampler_sinc_32p_cubic
    };

    Sampling_Rate_Conversion sampling_rate_conversion = { 0 };

    bool const needs_oversampling = quality > drb_audio_converter_quality_poor;

    resolve_sampling_rate_conversion
    (
        source_format->sampling_rate,
        target_format->sampling_rate,
        needs_oversampling,
        &sampling_rate_conversion
    );

    ASSERT(sampling_rate_conversion.up_2x_pass_count < max_2x_pass_count);
    ASSERT(sampling_rate_conversion.down_2x_pass_count < max_2x_pass_count);

    int pass_count = 0;
    int up_2x_index = 0;
    int down_2x_index = sampling_rate_conversion.down_2x_pass_count - 1;

    if (source_format->layout != layout_interleaved)
    {
        if (source_format->channel_count > 1)
        {
            passes[pass_count++] = pass_tag_interleaver;
        }
    }

    for (int it = 0; it < sampling_rate_conversion.up_2x_pass_count; it++)
    {
        passes[pass_count++] = upsamplers_2x[quality][up_2x_index++];
    }

    if (sampling_rate_conversion.needs_fractional_resampling)
    {
        passes[pass_count++] = resamplers[quality];
    }

    passes[pass_count++] = pass_tag_slicer; // TODO: Only add when needed.

    for (int it = 0; it < sampling_rate_conversion.down_2x_pass_count; it++)
    {
        passes[pass_count++] = downsamplers_2x[quality][down_2x_index--];
    }

    if (target_format->layout != layout_interleaved)
    {
        if (target_format->channel_count > 1)
        {
            passes[pass_count++] = pass_tag_deinterleaver;
        }
    }

    ASSERT(pass_count <= max_pass_count);

    return pass_count;
}

static int resolve_formats
    (
        Format const * const source_format,
        Format const * const target_format,
        short const passes [const],
        int const pass_count,
        Format formats [const]
    )
{
    // Set the first format to the source format, the last format to the target
    // format, and clear the rest:

    formats[0] = *source_format;

    if (formats[0].channel_count == channel_count_1)
    {
        formats[0].layout = layout_interleaved;
    }

    for (int index = 1; index < pass_count; index++)
    {
        formats[index] = (Format){ 0 };
    }

    formats[pass_count] = *target_format;

    if (formats[pass_count].channel_count == channel_count_1)
    {
        formats[pass_count].layout = layout_interleaved;
    }

    // Do a few iterations:

    for (int it = 0; it < 2; it++)
    {
        // Forward pass:

        for (int index = 0; index < pass_count; index++)
        {
            Pass const pass = make_pass(passes[index]);

            pass.restrain_formats
            (
                pass.configuration,
                &formats[index],
                &formats[index + 1]
            );
        }

        // Backward pass:

        for (int index = pass_count - 1; index >= 0; index--)
        {
            Pass const pass = make_pass(passes[index]);

            pass.restrain_formats
            (
                pass.configuration,
                &formats[index],
                &formats[index + 1]
            );
        }
    }

    if (pass_count == 0)
    {
        formats[0].block_size = max
        (
            source_format->block_size,
            target_format->block_size
        );

        formats[0].max_block_count = max
        (
            source_format->max_block_count,
            target_format->max_block_count
        );
    }

    // The format count is always equal to the pass count plus one.

    return pass_count + 1;
}

static void resolve_conversion
    (
        Format const * const source_format,
        Format const * const target_format,
        DrB_Audio_Converter_Direction const direction,
        DrB_Audio_Converter_Quality const quality,
        DrB_Audio_Conversion * const conversion
    )
{
    switch (direction)
    {
      case drb_audio_converter_direction_push:

        conversion->pass_count = resolve_passes
        (
            source_format,
            target_format,
            quality,
            conversion->passes
        );

        conversion->format_count = resolve_formats
        (
            source_format,
            target_format,
            conversion->passes,
            conversion->pass_count,
            conversion->formats
        );

        break;

      case drb_audio_converter_direction_pull:

        conversion->pass_count = resolve_passes
        (
            target_format,
            source_format,
            quality,
            conversion->passes
        );

        conversion->format_count = resolve_formats
        (
            target_format,
            source_format,
            conversion->passes,
            conversion->pass_count,
            conversion->formats
        );

        invert_passes(conversion->passes, conversion->pass_count);

        break;
    }
}

static int required_buffer_size
    (
        Format const * const formats,
        int format_count
    )
{
    int length = 0;

    for (int index = 0; index < format_count; index++)
    {
        int const channel_count = formats[index].channel_count;
        int const block_size = formats[index].block_size;
        int const max_block_count = formats[index].max_block_count;

        length = max(length, channel_count * block_size * max_block_count);
    }

    return length * sizeof(float);
}

/* -------------------------------------------------------------------------- */

#pragma mark - Validation

static bool any (bool const propositions [], int const count)
{
    bool acc = false;

    for (int index = 0; index < count; index++)
    {
        acc |= propositions[index];
    }

    return acc;
}

static bool all (bool const propositions [], int const count)
{
    bool acc = true;

    for (int index = 0; index < count; index++)
    {
        acc &= propositions[index];
    }

    return acc;
}

static bool validate_sampling_rate (int const sampling_rate)
{
    bool const propositions [] =
    {
        sampling_rate == drb_audio_converter_sampling_rate_8000,
        sampling_rate == drb_audio_converter_sampling_rate_11025,
        sampling_rate == drb_audio_converter_sampling_rate_16000,
        sampling_rate == drb_audio_converter_sampling_rate_22050,
        sampling_rate == drb_audio_converter_sampling_rate_32000,
        sampling_rate == drb_audio_converter_sampling_rate_44100,
        sampling_rate == drb_audio_converter_sampling_rate_48000,
        sampling_rate == drb_audio_converter_sampling_rate_60000,
        sampling_rate == drb_audio_converter_sampling_rate_88200,
        sampling_rate == drb_audio_converter_sampling_rate_96000,
        sampling_rate == drb_audio_converter_sampling_rate_120000,
        sampling_rate == drb_audio_converter_sampling_rate_176400,
        sampling_rate == drb_audio_converter_sampling_rate_192000,
        sampling_rate == drb_audio_converter_sampling_rate_240000
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_channel_count (int const channel_count)
{
    bool const propositions [] =
    {
        channel_count == channel_count_1,
        channel_count == channel_count_2,
        channel_count == channel_count_3,
        channel_count == channel_count_4,
        channel_count == channel_count_5,
        channel_count == channel_count_6,
        channel_count == channel_count_7,
        channel_count == channel_count_8
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_frame_layout (int const frame_layout)
{
    bool const propositions [] =
    {
        frame_layout == layout_interleaved,
        frame_layout == layout_deinterleaved
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_block_size (int const block_size)
{
    bool const propositions [] =
    {
        block_size == block_size_1,
        block_size == block_size_4,
        block_size == block_size_16,
        block_size == block_size_64,
        block_size == block_size_256,
        block_size == block_size_1024,
        block_size == block_size_4096
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_max_block_count (int const max_block_count)
{
    return 0 <= max_block_count && max_block_count <= 8192;
}

static bool validate_format (Format const * const format)
{
    bool const propositions [] =
    {
        validate_sampling_rate(format->sampling_rate),
        validate_channel_count(format->channel_count),
        validate_frame_layout(format->layout),
        validate_block_size(format->block_size),
        validate_max_block_count(format->max_block_count)
    };

    return all(propositions, sizeof(propositions));
}

static bool validate_conversion
    (
        Format const * const source_format,
        Format const * const target_format
    )
{
    bool const propositions [] =
    {
        validate_format(source_format),
        validate_format(target_format),
        source_format->block_size * source_format->max_block_count <= 8192,
        target_format->block_size * target_format->max_block_count <= 8192
    };

    return all(propositions, sizeof(propositions));
}

/* -------------------------------------------------------------------------- */

#pragma mark - Packing/Unpacking

static void pack_buffers_into_flat_array
    (
        DrB_Audio_Converter_Buffer const * const buffers,
        float * restrict const samples,
        int const frame_count,
        int const offset,
        int const channel_count,
        int const layout
    )
{
    switch (layout)
    {
      case layout_interleaved:
        memcpy
        (
            samples,
            buffers[0].samples + offset,
            frame_count * channel_count * sizeof(float)
        );
        break;
      case layout_deinterleaved:
        for (int channel = 0; channel < channel_count; channel++)
        {
            memcpy
            (
                samples + channel * frame_count,
                buffers[channel].samples + offset,
                frame_count * sizeof(float)
            );
        }
        break;
    }
}

static void unpack_flat_array_into_buffers
    (
        float const * restrict const samples,
        DrB_Audio_Converter_Buffer const * const buffers,
        int const frame_count,
        int const offset,
        int const channel_count,
        int const layout
    )
{
    switch (layout)
    {
      case layout_interleaved:
        memcpy
        (
            buffers[0].samples + offset,
            samples,
            frame_count * channel_count * sizeof(float)
        );
        break;
      case layout_deinterleaved:
        for (int channel = 0; channel < channel_count; channel++)
        {
            memcpy
            (
                buffers[channel].samples + offset,
                samples + channel * frame_count,
                frame_count * sizeof(float)
            );
        }
        break;
    }
}

static void make_buffers_point_to_flat_array
    (
        float * const samples,
        DrB_Audio_Converter_Buffer * const buffers,
        int const frame_count,
        int const channel_count
    )
{
    for (int channel = 0; channel < channel_count; channel++)
    {
        buffers[channel].samples = samples + channel * frame_count;
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Converter Definition

typedef void Converter_Function
    (
        struct DrB_Audio_Converter * converter,
        void * work_memory,
        DrB_Audio_Converter_Buffer const target_buffers [],
        int frame_count
    );

struct DrB_Audio_Converter
{
    Converter_Function * convert;
    DrB_Audio_Converter_Data_Callback data_callback;
    Format source_format, target_format;
    long buffer_length;
    int pass_count;
    int throttling;
    _Alignas(cache_line_size) Processor processors [];
};

/* -------------------------------------------------------------------------- */

#pragma mark - Conversion Push/Pull

static inline void swap_sample_pointers
    (
        float * restrict * restrict const buffer_a,
        float * restrict * restrict const buffer_b
    )
{
    float * const buffer_a_copy = *buffer_a;
    float * const buffer_b_copy = *buffer_b;

    *buffer_a = buffer_b_copy;
    *buffer_b = buffer_a_copy;
}

static void converter_push_frames
    (
        DrB_Audio_Converter * const converter,
        void * const work_memory,
        DrB_Audio_Converter_Buffer const source_buffers [const],
        int const frame_count
    )
{
    DrB_Audio_Converter_Buffer target_buffers [max_channel_count];

    int blocks [max_format_count];

    ASSERT(converter);

    float * source_samples = work_memory;
    float * target_samples = source_samples + converter->buffer_length;

    for (int offset = 0; offset < frame_count; offset += blocks[0])
    {
        double latency = offset / (float)converter->source_format.sampling_rate;

        blocks[0] = min(frame_count - offset, converter->throttling);

        for (int index = 0; index < converter->pass_count; index++)
        {
            ASSERT(blocks[index] <= converter->buffer_length);

            Processor const processor = converter->processors[index];

            if (processor.pushed_target_frame_count)
            {
                blocks[index + 1] = processor.pushed_target_frame_count
                (
                    processor.state,
                    blocks[index],
                    &latency
                );
            }
            else
            {
                blocks[index + 1] = blocks[index];
            }
        }

        pack_buffers_into_flat_array
        (
            source_buffers,
            source_samples,
            blocks[0],
            offset,
            converter->source_format.channel_count,
            converter->source_format.layout
        );

        for (int index = 0; index < converter->pass_count; index++)
        {
            Processor const processor = converter->processors[index];

            int const processed_target_frame_count = processor.push
            (
                processor.state,
                source_samples,
                target_samples,
                blocks[index]
            );

            ASSERT(processed_target_frame_count == blocks[index + 1]);

            swap_sample_pointers(&source_samples, &target_samples);
        }

        make_buffers_point_to_flat_array
        (
            source_samples,
            target_buffers,
            blocks[converter->pass_count],
            converter->target_format.channel_count
        );

        converter->data_callback.process
        (
            converter->data_callback.user_data,
            latency,
            target_buffers,
            blocks[converter->pass_count]
        );
    }
}

static void converter_pull_frames
    (
        DrB_Audio_Converter * const converter,
        void * const work_buffer,
        DrB_Audio_Converter_Buffer const target_buffers [const],
        int const frame_count
    )
{
    DrB_Audio_Converter_Buffer source_buffers [max_channel_count];

    int blocks [max_format_count];

    ASSERT(converter);

    float * source_samples = work_buffer;
    float * target_samples = source_samples + converter->buffer_length;

    for (int offset = 0; offset < frame_count; offset += blocks[0])
    {
        double latency = offset / (float)converter->target_format.sampling_rate;

        blocks[0] = min(frame_count - offset, converter->throttling);

        for (int index = 0; index < converter->pass_count; index++)
        {
            ASSERT(blocks[index] <= converter->buffer_length);

            Processor const processor = converter->processors[index];

            if (processor.pulled_source_frame_count)
            {
                blocks[index + 1] = processor.pulled_source_frame_count
                (
                    processor.state,
                    blocks[index],
                    &latency
                );
            }
            else
            {
                blocks[index + 1] = blocks[index];
            }
        }

        make_buffers_point_to_flat_array
        (
            source_samples,
            source_buffers,
            blocks[converter->pass_count],
            converter->source_format.channel_count
        );

        converter->data_callback.process
        (
            converter->data_callback.user_data,
            latency,
            source_buffers,
            blocks[converter->pass_count]
        );

        for (int index = converter->pass_count - 1; index >= 0; index--)
        {
            Processor const processor = converter->processors[index];

            int const processed_source_frame_count = processor.pull
            (
                processor.state,
                source_samples,
                target_samples,
                blocks[index]
            );

            ASSERT(processed_source_frame_count == blocks[index + 1]);

            swap_sample_pointers(&source_samples, &target_samples);
        }

        unpack_flat_array_into_buffers
        (
            source_samples,
            target_buffers,
            blocks[0],
            offset,
            converter->target_format.channel_count,
            converter->target_format.layout
        );
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Converter Construction

static void converter_construct
    (
        Bump_Allocator * const allocator,
        DrB_Audio_Conversion const * const conversion,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Data_Callback const data_callback
    )
{
    Format const * const formats = conversion->formats;
    short const * const passes = conversion->passes;
    int const pass_count = conversion->pass_count;
    int const format_count = conversion->format_count;

    DrB_Audio_Converter * const converter = alloc
    (
        allocator,
        sizeof(DrB_Audio_Converter) + pass_count * sizeof(Processor)
    );

    int const buffer_size = required_buffer_size(formats, format_count);
    int const throttling = formats[0].block_size * formats[0].max_block_count;

    ASSERT(throttling > 0);

    if (converter)
    {
        switch (direction)
        {
          case drb_audio_converter_direction_push:
            converter->convert = converter_push_frames;
            converter->source_format = formats[0];
            converter->target_format = formats[pass_count];
            break;
          case drb_audio_converter_direction_pull:
            converter->convert = converter_pull_frames;
            converter->source_format = formats[pass_count];
            converter->target_format = formats[0];
            break;
        }

        converter->data_callback = data_callback;
        converter->buffer_length = cache_align(buffer_size) / sizeof(float);
        converter->pass_count = pass_count;
        converter->throttling = throttling;
    }

    for (int index = 0; index < pass_count; index++)
    {
        Pass const pass = make_pass(passes[index]);

        Processor processor;

        switch (direction)
        {
          case drb_audio_converter_direction_push:
            processor = pass.create_processor
            (
                pass.configuration,
                allocator,
                &formats[index + 0],
                &formats[index + 1]
            );
            break;
          case drb_audio_converter_direction_pull:
            processor = pass.create_processor
            (
                pass.configuration,
                allocator,
                &formats[index + 1],
                &formats[index + 0]
            );
            break;
        }

        if (converter)
        {
            converter->processors[index] = processor;
        }
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Printing

#if defined(DEBUG) && defined(PRINT_CONVERSION)

#include <stdio.h>

static void print_format (Format const * const format)
{
    char const * const layout =
        format->layout == layout_interleaved
            ? "interleaved"
            : "de-interleaved";

    printf("sampling rate   = %d\n", format->sampling_rate);
    printf("channel count   = %d\n", format->channel_count);
    printf("layout          = %s\n", layout);
    printf("block size      = %d\n", format->block_size);
    printf("max block count = %d\n", format->max_block_count);
}

static void print_pass (int pass)
{
    static char const * const descriptions [] =
    {
        [pass_tag_interleaver] = "interleaver",
        [pass_tag_deinterleaver] = "deinterleaver",
        [pass_tag_upsampler_2x_fir_16] = "upsampler_2x_fir_16",
        [pass_tag_upsampler_2x_fir_32] = "upsampler_2x_fir_32",
        [pass_tag_upsampler_2x_fir_48] = "upsampler_2x_fir_48",
        [pass_tag_upsampler_2x_fir_64] = "upsampler_2x_fir_64",
        [pass_tag_upsampler_2x_fir_80] = "upsampler_2x_fir_80",
        [pass_tag_upsampler_2x_fir_96] = "upsampler_2x_fir_96",
        [pass_tag_upsampler_2x_fir_256] = "upsampler_2x_fir_256",
        [pass_tag_downsampler_2x_fir_16] = "downsampler_2x_fir_16",
        [pass_tag_downsampler_2x_fir_32] = "downsampler_2x_fir_32",
        [pass_tag_downsampler_2x_fir_48] = "downsampler_2x_fir_48",
        [pass_tag_downsampler_2x_fir_64] = "downsampler_2x_fir_64",
        [pass_tag_downsampler_2x_fir_80] = "downsampler_2x_fir_80",
        [pass_tag_downsampler_2x_fir_96] = "downsampler_2x_fir_96",
        [pass_tag_downsampler_2x_fir_256] = "downsampler_2x_fir_256",
        [pass_tag_resampler_sinc_8p_linear] = "resampler_sinc_8p_linear",
        [pass_tag_resampler_sinc_16p_linear] = "resampler_sinc_16p_linear",
        [pass_tag_resampler_sinc_24p_linear] = "resampler_sinc_24p_linear",
        [pass_tag_resampler_sinc_32p_linear] = "resampler_sinc_32p_linear",
        [pass_tag_resampler_sinc_8p_cubic] = "resampler_sinc_8p_cubic",
        [pass_tag_resampler_sinc_16p_cubic] = "resampler_sinc_16p_cubic",
        [pass_tag_resampler_sinc_24p_cubic] = "resampler_sinc_24p_cubic",
        [pass_tag_resampler_sinc_32p_cubic] = "resampler_sinc_32p_cubic",
        [pass_tag_slicer] = "slicer"
    };

    printf("%s\n", descriptions[pass]);
}

static void print_conversion (DrB_Audio_Conversion const * const conversion)
{
    printf("source\n\n");

    for (int index = 0; index < conversion->pass_count; index++)
    {
        print_format(&conversion->formats[index]);
        printf("\n");
        print_pass(conversion->passes[index]);
        printf("\n");
    }

    print_format(&conversion->formats[conversion->pass_count]);
    printf("\n");
    printf("target\n");
    printf("\n");
}

#endif // defined(DEBUG) && defined(PRINT_CONVERSION)

/* -------------------------------------------------------------------------- */

#pragma mark - Converter API Functions

extern _Bool drb_audio_converter_alignment_and_size
    (
        int source_sampling_rate,
        int target_sampling_rate,
        int channel_count,
        int max_frame_count,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Quality quality,
        long * alignment,
        long * size
    )
{
    int const block_size = 1;

    int const source_block_size =
        direction == drb_audio_converter_direction_pull ? block_size : 1;

    int const target_block_size =
        direction == drb_audio_converter_direction_push ? block_size : 1;

    int const source_max_block_count =
        direction == drb_audio_converter_direction_pull ? max_frame_count : 0;

    int const target_max_block_count =
        direction == drb_audio_converter_direction_push ? max_frame_count : 0;

    Format const source_format =
    {
        .sampling_rate = source_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = source_block_size,
        .max_block_count = source_max_block_count
    };

    Format const target_format =
    {
        .sampling_rate = target_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = target_block_size,
        .max_block_count = target_max_block_count
    };

    // Check that the conversion is valid.

    if (!validate_conversion(&source_format, &target_format))
    {
        return 0;
    }

    // Resolve the conversion.

    DrB_Audio_Conversion conversion = { 0 };

    resolve_conversion
    (
        &source_format,
        &target_format,
        direction,
        quality,
        &conversion
    );

    // Set up an empty allocator in order to estimate the required memory size.

    DrB_Audio_Converter_Data_Callback const data_callback = { 0 };

    Bump_Allocator bump_allocator;

    bump_allocator_construct(&bump_allocator, 0);

    converter_construct(&bump_allocator, &conversion, direction, data_callback);

    *alignment = cache_line_size;
    *size = bump_allocator.offset;

    return true;
}

extern DrB_Audio_Converter * drb_audio_converter_construct
    (
        void * memory,
        int source_sampling_rate,
        int target_sampling_rate,
        int channel_count,
        int max_frame_count,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Quality quality,
        DrB_Audio_Converter_Data_Callback data_callback
    )
{
    int const block_size = 1;

    int const source_block_size =
        direction == drb_audio_converter_direction_pull ? block_size : 1;

    int const target_block_size =
        direction == drb_audio_converter_direction_push ? block_size : 1;

    int const source_max_block_count =
        direction == drb_audio_converter_direction_pull ? max_frame_count : 0;

    int const target_max_block_count =
        direction == drb_audio_converter_direction_push ? max_frame_count : 0;

    Format const source_format =
    {
        .sampling_rate = source_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = source_block_size,
        .max_block_count = source_max_block_count
    };

    Format const target_format =
    {
        .sampling_rate = target_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = target_block_size,
        .max_block_count = target_max_block_count
    };

    // Check that the conversion is valid.

    if (!validate_conversion(&source_format, &target_format))
    {
        return 0;
    }

    // Resolve the conversion.

    DrB_Audio_Conversion conversion = { 0 };

    resolve_conversion
    (
        &source_format,
        &target_format,
        direction,
        quality,
        &conversion
    );

  #if defined(DEBUG) && defined(PRINT_CONVERSION)
    print_conversion(&conversion);
  #endif

    Bump_Allocator bump_allocator;

    bump_allocator_construct(&bump_allocator, memory);

    converter_construct(&bump_allocator, &conversion, direction, data_callback);

    // Cast the pointer to the memory implicitly to a converter and return it.

    return memory;
}

extern void drb_audio_converter_work_memory_alignment_and_size
    (
        DrB_Audio_Converter * converter,
        long * alignment,
        long * size
    )
{
    *alignment = cache_line_size;
    *size = converter->buffer_length * 2 * sizeof(float);
}

extern void drb_audio_converter_process
    (
        DrB_Audio_Converter * const converter,
        void * const work_memory,
        DrB_Audio_Converter_Buffer const buffers [const],
        int const frame_count
    )
{
    converter->convert(converter, work_memory, buffers, frame_count);
}

/* -------------------------------------------------------------------------- */

#pragma mark - Kernels

_Alignas(64) static float const coeffs_2x_fir_16 [8] =
{
    -0.0001199071f, +0.0050233615f, -0.0454816414f, +0.2905768050f,
    +0.2905768050f, -0.0454816414f, +0.0050233615f, -0.0001199071f
};

_Alignas(64) static float const coeffs_2x_fir_32 [16] =
{
    -0.0000108295f, +0.0001999758f, -0.0012197609f, +0.0047545301f,
    -0.0141238117f, +0.0355866439f, -0.0863425372f, +0.3111579909f,
    +0.3111579909f, -0.0863425372f, +0.0355866439f, -0.0141238117f,
    +0.0047545301f, -0.0012197609f, +0.0001999758f, -0.0000108295f
};

_Alignas(64) static float const coeffs_2x_fir_48 [24] =
{
    -0.0000029696f, +0.0000399690f, -0.0001883739f, +0.0006249117f,
    -0.0016744538f, +0.0038602660f, -0.0079648422f, +0.0151605471f,
    -0.0274429195f, +0.0493349697f, -0.0968589350f, +0.3151128616f,
    +0.3151128616f, -0.0968589350f, +0.0493349697f, -0.0274429195f,
    +0.0151605471f, -0.0079648422f, +0.0038602660f, -0.0016744538f,
    +0.0006249117f, -0.0001883739f, +0.0000399690f, -0.0000029696f
};

_Alignas(64) static float const coeffs_2x_fir_64 [32] =
{
    -0.0000012146f, +0.0000141345f, -0.0000568089f, +0.0001666104f,
    -0.0004088859f, +0.0008855596f, -0.0017440518f, +0.0031867119f,
    -0.0054838807f, +0.0090004334f, -0.0142617463f, +0.0221301322f,
    -0.0343195801f, +0.0551936171f, -0.1008084555f, +0.3165079191f,
    +0.3165079191f, -0.1008084555f, +0.0551936171f, -0.0343195801f,
    +0.0221301322f, -0.0142617463f, +0.0090004334f, -0.0054838807f,
    +0.0031867119f, -0.0017440518f, +0.0008855596f, -0.0004088859f,
    +0.0001666104f, -0.0000568089f, +0.0000141345f, -0.0000012146f
};

_Alignas(64) static float const coeffs_2x_fir_80 [40] =
{
    -0.0000006121f, +0.0000065923f, -0.0000239814f, +0.0000642740f,
    -0.0001468086f, +0.0003008955f, -0.0005681180f, +0.0010046723f,
    -0.0016837438f, +0.0026982588f, -0.0041650018f, +0.0062323398f,
    -0.0090963283f, +0.0130356862f, -0.0184908195f, +0.0262557595f,
    -0.0380076767f, +0.0581153610f, -0.1026860286f, +0.3171555468f,
    +0.3171555468f, -0.1026860286f, +0.0581153610f, -0.0380076767f,
    +0.0262557595f, -0.0184908195f, +0.0130356862f, -0.0090963283f,
    +0.0062323398f, -0.0041650018f, +0.0026982588f, -0.0016837438f,
    +0.0010046723f, -0.0005681180f, +0.0003008955f, -0.0001468086f,
    +0.0000642740f, -0.0000239814f, +0.0000065923f, -0.0000006121f
};

_Alignas(64) static float const coeffs_2x_fir_96 [48] =
{
    -0.0000003509f, +0.0000036098f, -0.0000123073f, +0.0000308976f,
    -0.0000666586f, +0.0001303586f, -0.0002369979f, +0.0004065870f,
    -0.0006649309f, +0.0010444142f, -0.0015848434f, +0.0023345053f,
    -0.0033517786f, +0.0047079372f, -0.0064923410f, +0.0088222963f,
    -0.0118622146f, +0.0158622292f, -0.0212409738f, +0.0287808457f,
    -0.0401598844f, +0.0597612362f, -0.1037193303f, +0.3175078537f,
    +0.3175078537f, -0.1037193303f, +0.0597612362f, -0.0401598844f,
    +0.0287808457f, -0.0212409738f, +0.0158622292f, -0.0118622146f,
    +0.0088222963f, -0.0064923410f, +0.0047079372f, -0.0033517786f,
    +0.0023345053f, -0.0015848434f, +0.0010444142f, -0.0006649309f,
    +0.0004065870f, -0.0002369979f, +0.0001303586f, -0.0000666586f,
    +0.0000308976f, -0.0000123073f, +0.0000036098f, -0.0000003509f
};

_Alignas(64) static float const coeffs_2x_fir_256 [128] =
{
    -0.0000000181f, +0.0000001675f, -0.0000004854f, +0.0000010045f,
    -0.0000017722f, +0.0000028521f, -0.0000043249f, +0.0000062902f,
    -0.0000088681f, +0.0000122013f, -0.0000164564f, +0.0000218262f,
    -0.0000285316f, +0.0000368240f, -0.0000469870f, +0.0000593388f,
    -0.0000742347f, +0.0000920685f, -0.0001132755f, +0.0001383344f,
    -0.0001677690f, +0.0002021512f, -0.0002421026f, +0.0002882967f,
    -0.0003414616f, +0.0004023825f, -0.0004719040f, +0.0005509339f,
    -0.0006404465f, +0.0007414872f, -0.0008551779f, +0.0009827237f,
    -0.0011254216f, +0.0012846706f, -0.0014619855f, +0.0016590136f,
    -0.0018775567f, +0.0021195980f, -0.0023873376f, +0.0026832376f,
    -0.0030100795f, +0.0033710391f, -0.0037697835f, +0.0042105982f,
    -0.0046985549f, +0.0052397352f, -0.0058415327f, +0.0065130670f,
    -0.0072657586f, +0.0081141436f, -0.0090770515f, +0.0101793492f,
    -0.0114545959f, +0.0129492146f, -0.0147292984f, +0.0168922210f,
    -0.0195875325f, +0.0230571427f, -0.0277193359f, +0.0343647996f,
    -0.0446886955f, +0.0630996856f, -0.1057650419f, +0.3181969882f,
    +0.3181969882f, -0.1057650419f, +0.0630996856f, -0.0446886955f,
    +0.0343647996f, -0.0277193359f, +0.0230571427f, -0.0195875325f,
    +0.0168922210f, -0.0147292984f, +0.0129492146f, -0.0114545959f,
    +0.0101793492f, -0.0090770515f, +0.0081141436f, -0.0072657586f,
    +0.0065130670f, -0.0058415327f, +0.0052397352f, -0.0046985549f,
    +0.0042105982f, -0.0037697835f, +0.0033710391f, -0.0030100795f,
    +0.0026832376f, -0.0023873376f, +0.0021195980f, -0.0018775567f,
    +0.0016590136f, -0.0014619855f, +0.0012846706f, -0.0011254216f,
    +0.0009827237f, -0.0008551779f, +0.0007414872f, -0.0006404465f,
    +0.0005509339f, -0.0004719040f, +0.0004023825f, -0.0003414616f,
    +0.0002882967f, -0.0002421026f, +0.0002021512f, -0.0001677690f,
    +0.0001383344f, -0.0001132755f, +0.0000920685f, -0.0000742347f,
    +0.0000593388f, -0.0000469870f, +0.0000368240f, -0.0000285316f,
    +0.0000218262f, -0.0000164564f, +0.0000122013f, -0.0000088681f,
    +0.0000062902f, -0.0000043249f, +0.0000028521f, -0.0000017722f,
    +0.0000010045f, -0.0000004854f, +0.0000001675f, -0.0000000181f
};

_Alignas(64) static float const kernels_8p_cubic [32][8] =
{
    {
        +0.0002597675f, -0.0039207942f, +0.0252813482f, +0.9976142235f,
        -0.0224072833f, +0.0033759653f, -0.0002028858f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0002028858f, +0.0033759653f, -0.0224072833f, +0.9976142235f,
        +0.0252813482f, -0.0039207942f, +0.0002597675f, -0.0000000762f
    },
    {
        -0.0003549887f, +0.0062188612f, -0.0419547282f, +0.9904817710f,
        +0.0534170529f, -0.0083886329f, +0.0005820920f, -0.0000006154f
    },
    {
        -0.0004626033f, +0.0085486928f, -0.0586884681f, +0.9786769174f,
        +0.0843518729f, -0.0133949381f, +0.0009719698f, -0.0000020958f
    },
    {
        -0.0005320226f, +0.0103923915f, -0.0726841373f, +0.9623222726f,
        +0.1179939134f, -0.0189191675f, +0.0014334389f, -0.0000050129f
    },
    {
        -0.0005693707f, +0.0117825051f, -0.0840441436f, +0.9415870311f,
        +0.1542140211f, -0.0249277566f, +0.0019692793f, -0.0000098768f
    },
    {
        -0.0005804673f, +0.0127559191f, -0.0928946799f, +0.9166845637f,
        +0.1928457270f, -0.0313732031f, +0.0025806987f, -0.0000172073f
    },
    {
        -0.0005707216f, +0.0133526329f, -0.0993825376f, +0.8878693936f,
        +0.2336857581f, -0.0381933275f, +0.0032670113f, -0.0000275246f
    },
    {
        -0.0005450558f, +0.0136146100f, -0.1036717835f, +0.8554336095f,
        +0.2764951337f, -0.0453107385f, +0.0040253171f, -0.0000413342f
    },
    {
        -0.0005078543f, +0.0135847166f, -0.1059403649f, +0.8197027759f,
        +0.3210008449f, -0.0526325333f, +0.0048501897f, -0.0000591070f
    },
    {
        -0.0004629377f, +0.0133057627f, -0.1063767014f, +0.7810314124f,
        +0.3668981130f, -0.0600502573f, +0.0057333854f, -0.0000812528f
    },
    {
        -0.0004135573f, +0.0128196536f, -0.1051763240f, +0.7397981173f,
        +0.4138532056f, -0.0674401466f, +0.0066635808f, -0.0001080870f
    },
    {
        -0.0003624081f, +0.0121666586f, -0.1025386141f, +0.6964004181f,
        +0.4615067821f, -0.0746636711f, +0.0076261527f, -0.0001397917f
    },
    {
        -0.0003116559f, +0.0113847990f, -0.0986636928f, +0.6512494350f,
        +0.5094777289f, -0.0815683908f, +0.0086030112f, -0.0001763709f
    },
    {
        -0.0002629760f, +0.0105093563f, -0.0937495048f, +0.6047644449f,
        +0.5573674344f, -0.0879891352f, +0.0095724973f, -0.0002176007f
    },
    {
        -0.0002176007f, +0.0095724973f, -0.0879891352f, +0.5573674344f,
        +0.6047644449f, -0.0937495048f, +0.0105093563f, -0.0002629760f
    },
    {
        -0.0001763709f, +0.0086030112f, -0.0815683908f, +0.5094777289f,
        +0.6512494350f, -0.0986636928f, +0.0113847990f, -0.0003116559f
    },
    {
        -0.0001397917f, +0.0076261527f, -0.0746636711f, +0.4615067821f,
        +0.6964004181f, -0.1025386141f, +0.0121666586f, -0.0003624081f
    },
    {
        -0.0001080870f, +0.0066635808f, -0.0674401466f, +0.4138532056f,
        +0.7397981173f, -0.1051763240f, +0.0128196536f, -0.0004135573f
    },
    {
        -0.0000812528f, +0.0057333854f, -0.0600502573f, +0.3668981130f,
        +0.7810314124f, -0.1063767014f, +0.0133057627f, -0.0004629377f
    },
    {
        -0.0000591070f, +0.0048501897f, -0.0526325333f, +0.3210008449f,
        +0.8197027759f, -0.1059403649f, +0.0135847166f, -0.0005078543f
    },
    {
        -0.0000413342f, +0.0040253171f, -0.0453107385f, +0.2764951337f,
        +0.8554336095f, -0.1036717835f, +0.0136146100f, -0.0005450558f
    },
    {
        -0.0000275246f, +0.0032670113f, -0.0381933275f, +0.2336857581f,
        +0.8878693936f, -0.0993825376f, +0.0133526329f, -0.0005707216f
    },
    {
        -0.0000172073f, +0.0025806987f, -0.0313732031f, +0.1928457270f,
        +0.9166845637f, -0.0928946799f, +0.0127559191f, -0.0005804673f
    },
    {
        -0.0000098768f, +0.0019692793f, -0.0249277566f, +0.1542140211f,
        +0.9415870311f, -0.0840441436f, +0.0117825051f, -0.0005693707f
    },
    {
        -0.0000050129f, +0.0014334389f, -0.0189191675f, +0.1179939134f,
        +0.9623222726f, -0.0726841373f, +0.0103923915f, -0.0005320226f
    },
    {
        -0.0000020958f, +0.0009719698f, -0.0133949381f, +0.0843518729f,
        +0.9786769174f, -0.0586884681f, +0.0085486928f, -0.0004626033f
    },
    {
        -0.0000006154f, +0.0005820920f, -0.0083886329f, +0.0534170529f,
        +0.9904817710f, -0.0419547282f, +0.0062188612f, -0.0003549887f
    },
    {
        -0.0000000762f, +0.0002597675f, -0.0039207942f, +0.0252813482f,
        +0.9976142235f, -0.0224072833f, +0.0033759653f, -0.0002028858f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0002028858f, +0.0033759653f, -0.0224072833f,
        +0.9976142235f, +0.0252813482f, -0.0039207942f, +0.0002597675f
    }
};

_Alignas(64) static float const kernels_16p_cubic [32][16] =
{
    {
        +0.0000142372f, -0.0001222265f, +0.0005689651f, -0.0018891417f,
        +0.0050754959f, -0.0122638191f, +0.0327412870f, +0.9979374556f,
        -0.0301752646f, +0.0115458264f, -0.0047636357f, +0.0017529852f,
        -0.0005182446f, +0.0001080204f, -0.0000117807f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000117807f, +0.0001080204f, -0.0005182446f, +0.0017529852f,
        -0.0047636357f, +0.0115458264f, -0.0301752646f, +0.9979374556f,
        +0.0327412870f, -0.0122638191f, +0.0050754959f, -0.0018891417f,
        +0.0005689651f, -0.0001222265f, +0.0000142372f, -0.0000000095f
    },
    {
        -0.0000212377f, +0.0002016965f, -0.0009828203f, +0.0033561752f,
        -0.0091739032f, +0.0222759309f, -0.0576769779f, +0.9917661662f,
        +0.0679146713f, -0.0251331479f, +0.0104144883f, -0.0038978103f,
        +0.0011846264f, -0.0002582448f, +0.0000310269f, -0.0000000758f
    },
    {
        -0.0000285336f, +0.0002811408f, -0.0013918123f, +0.0047989121f,
        -0.0131962971f, +0.0321077070f, -0.0824243242f, +0.9815349991f,
        +0.1053604338f, -0.0384813103f, +0.0159620092f, -0.0060064202f,
        +0.0018418555f, -0.0004073410f, +0.0000504203f, -0.0000002550f
    },
    {
        -0.0000338536f, +0.0003466893f, -0.0017442842f, +0.0060734677f,
        -0.0168033627f, +0.0409738500f, -0.1043632345f, +0.9673248576f,
        +0.1448939482f, -0.0521681919f, +0.0216568422f, -0.0081925113f,
        +0.0025344075f, -0.0005684830f, +0.0000724164f, -0.0000006010f
    },
    {
        -0.0000373982f, +0.0003988727f, -0.0020402120f, +0.0071749533f,
        -0.0199746479f, +0.0488223507f, -0.1234661426f, +0.9492478724f,
        +0.1863068468f, -0.0660411847f, +0.0274319851f, -0.0104308793f,
        +0.0032549260f, -0.0007403036f, +0.0000969544f, -0.0000011634f
    },
    {
        -0.0000393780f, +0.0004383875f, -0.0022804092f, +0.0081011958f,
        -0.0226965652f, +0.0556163129f, -0.1397314714f, +0.9274462843f,
        +0.2293684592f, -0.0799363198f, +0.0332152172f, -0.0126937555f,
        +0.0039949671f, -0.0009210902f, +0.0001239070f, -0.0000019861f
    },
    {
        -0.0000400071f, +0.0004660666f, -0.0024664455f, +0.0088525824f,
        -0.0249621715f, +0.0613336021f, -0.1531828621f, +0.9020910290f,
        +0.2738275104f, -0.0936795786f, +0.0389297706f, -0.0149510380f,
        +0.0047450430f, -0.0011087789f, +0.0001530733f, -0.0000031058f
    },
    {
        -0.0000394985f, +0.0004828506f, -0.0026005590f, +0.0094318791f,
        -0.0267708702f, +0.0659663364f, -0.1638681598f, +0.8733800404f,
        +0.3194140663f, -0.1070883754f, +0.0444950995f, -0.0171705714f,
        +0.0054946854f, -0.0013009540f, +0.0001841735f, -0.0000045496f
    },
    {
        -0.0000380595f, +0.0004897602f, -0.0026855661f, +0.0098440277f,
        -0.0281280474f, +0.0695202339f, -0.1718581730f, +0.8415362932f,
        +0.3658417060f, -0.1199731976f, +0.0498277448f, -0.0193184765f,
        +0.0062325316f, -0.0014948552f, +0.0002168439f, -0.0000063340f
    },
    {
        -0.0000358882f, +0.0004878694f, -0.0027247691f, +0.0100959241f,
        -0.0290446470f, +0.0720138275f, -0.1772452254f, +0.8068056046f,
        +0.4128099039f, -0.1321393907f, +0.0548422843f, -0.0213595271f,
        +0.0069464297f, -0.0016873911f, +0.0002506334f, -0.0000084630f
    },
    {
        -0.0000331701f, +0.0004782815f, -0.0027218637f, +0.0101961838f,
        -0.0295366969f, +0.0734775655f, -0.1801415221f, +0.7694542222f,
        +0.4600065953f, -0.1433890707f, +0.0594523625f, -0.0232575705f,
        +0.0076235672f, -0.0018751595f, +0.0002850013f, -0.0000109264f
    },
    {
        -0.0000300758f, +0.0004621060f, -0.0026808482f, +0.0101548987f,
        -0.0296247927f, +0.0739528105f, -0.1806773531f, +0.7297662246f,
        +0.5071109014f, -0.1535231460f, +0.0635717883f, -0.0249759894f,
        +0.0082506182f, -0.0020544763f, +0.0003193171f, -0.0000136986f
    },
    {
        -0.0000267592f, +0.0004404382f, -0.0026059351f, +0.0099833884f,
        -0.0293335506f, +0.0734907557f, -0.1789991574f, +0.6880407631f,
        +0.5537959863f, -0.1623434278f, +0.0671156908f, -0.0264782005f,
        +0.0088139118f, -0.0022214120f, +0.0003528617f, -0.0000167374f
    },
    {
        -0.0000233559f, +0.0004143413f, -0.0025014665f, +0.0096939513f,
        -0.0286910368f, +0.0721512743f, -0.1752674740f, +0.6445891772f,
        +0.5997320140f, -0.1696548069f, +0.0700017192f, -0.0277281836f,
        +0.0092996180f, -0.0023718355f, +0.0003848309f, -0.0000199827f
    },
    {
        -0.0000199827f, +0.0003848309f, -0.0023718355f, +0.0092996180f,
        -0.0277281836f, +0.0700017192f, -0.1696548069f, +0.5997320140f,
        +0.6445891772f, -0.1752674740f, +0.0721512743f, -0.0286910368f,
        +0.0096939513f, -0.0025014665f, +0.0004143413f, -0.0000233559f
    },
    {
        -0.0000167374f, +0.0003528617f, -0.0022214120f, +0.0088139118f,
        -0.0264782005f, +0.0671156908f, -0.1623434278f, +0.5537959863f,
        +0.6880407631f, -0.1789991574f, +0.0734907557f, -0.0293335506f,
        +0.0099833884f, -0.0026059351f, +0.0004404382f, -0.0000267592f
    },
    {
        -0.0000136986f, +0.0003193171f, -0.0020544763f, +0.0082506182f,
        -0.0249759894f, +0.0635717883f, -0.1535231460f, +0.5071109014f,
        +0.7297662246f, -0.1806773531f, +0.0739528105f, -0.0296247927f,
        +0.0101548987f, -0.0026808482f, +0.0004621060f, -0.0000300758f
    },
    {
        -0.0000109264f, +0.0002850013f, -0.0018751595f, +0.0076235672f,
        -0.0232575705f, +0.0594523625f, -0.1433890707f, +0.4600065953f,
        +0.7694542222f, -0.1801415221f, +0.0734775655f, -0.0295366969f,
        +0.0101961838f, -0.0027218637f, +0.0004782815f, -0.0000331701f
    },
    {
        -0.0000084630f, +0.0002506334f, -0.0016873911f, +0.0069464297f,
        -0.0213595271f, +0.0548422843f, -0.1321393907f, +0.4128099039f,
        +0.8068056046f, -0.1772452254f, +0.0720138275f, -0.0290446470f,
        +0.0100959241f, -0.0027247691f, +0.0004878694f, -0.0000358882f
    },
    {
        -0.0000063340f, +0.0002168439f, -0.0014948552f, +0.0062325316f,
        -0.0193184765f, +0.0498277448f, -0.1199731976f, +0.3658417060f,
        +0.8415362932f, -0.1718581730f, +0.0695202339f, -0.0281280474f,
        +0.0098440277f, -0.0026855661f, +0.0004897602f, -0.0000380595f
    },
    {
        -0.0000045496f, +0.0001841735f, -0.0013009540f, +0.0054946854f,
        -0.0171705714f, +0.0444950995f, -0.1070883754f, +0.3194140663f,
        +0.8733800404f, -0.1638681598f, +0.0659663364f, -0.0267708702f,
        +0.0094318791f, -0.0026005590f, +0.0004828506f, -0.0000394985f
    },
    {
        -0.0000031058f, +0.0001530733f, -0.0011087789f, +0.0047450430f,
        -0.0149510380f, +0.0389297706f, -0.0936795786f, +0.2738275104f,
        +0.9020910290f, -0.1531828621f, +0.0613336021f, -0.0249621715f,
        +0.0088525824f, -0.0024664455f, +0.0004660666f, -0.0000400071f
    },
    {
        -0.0000019861f, +0.0001239070f, -0.0009210902f, +0.0039949671f,
        -0.0126937555f, +0.0332152172f, -0.0799363198f, +0.2293684592f,
        +0.9274462843f, -0.1397314714f, +0.0556163129f, -0.0226965652f,
        +0.0081011958f, -0.0022804092f, +0.0004383875f, -0.0000393780f
    },
    {
        -0.0000011634f, +0.0000969544f, -0.0007403036f, +0.0032549260f,
        -0.0104308793f, +0.0274319851f, -0.0660411847f, +0.1863068468f,
        +0.9492478724f, -0.1234661426f, +0.0488223507f, -0.0199746479f,
        +0.0071749533f, -0.0020402120f, +0.0003988727f, -0.0000373982f
    },
    {
        -0.0000006010f, +0.0000724164f, -0.0005684830f, +0.0025344075f,
        -0.0081925113f, +0.0216568422f, -0.0521681919f, +0.1448939482f,
        +0.9673248576f, -0.1043632345f, +0.0409738500f, -0.0168033627f,
        +0.0060734677f, -0.0017442842f, +0.0003466893f, -0.0000338536f
    },
    {
        -0.0000002550f, +0.0000504203f, -0.0004073410f, +0.0018418555f,
        -0.0060064202f, +0.0159620092f, -0.0384813103f, +0.1053604338f,
        +0.9815349991f, -0.0824243242f, +0.0321077070f, -0.0131962971f,
        +0.0047989121f, -0.0013918123f, +0.0002811408f, -0.0000285336f
    },
    {
        -0.0000000758f, +0.0000310269f, -0.0002582448f, +0.0011846264f,
        -0.0038978103f, +0.0104144883f, -0.0251331479f, +0.0679146713f,
        +0.9917661662f, -0.0576769779f, +0.0222759309f, -0.0091739032f,
        +0.0033561752f, -0.0009828203f, +0.0002016965f, -0.0000212377f
    },
    {
        -0.0000000095f, +0.0000142372f, -0.0001222265f, +0.0005689651f,
        -0.0018891417f, +0.0050754959f, -0.0122638191f, +0.0327412870f,
        +0.9979374556f, -0.0301752646f, +0.0115458264f, -0.0047636357f,
        +0.0017529852f, -0.0005182446f, +0.0001080204f, -0.0000117807f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0000117807f, +0.0001080204f, -0.0005182446f,
        +0.0017529852f, -0.0047636357f, +0.0115458264f, -0.0301752646f,
        +0.9979374556f, +0.0327412870f, -0.0122638191f, +0.0050754959f,
        -0.0018891417f, +0.0005689651f, -0.0001222265f, +0.0000142372f
    }
};

_Alignas(64) static float const kernels_24p_cubic [32][24] =
{
    {
        +0.0000033144f, -0.0000209910f, +0.0000798387f, -0.0002338727f,
        +0.0005748807f, -0.0012439113f, +0.0024495327f, -0.0045199995f,
        +0.0080939427f, -0.0149662792f, +0.0343262200f, +0.9979973240f,
        -0.0318593467f, +0.0142966758f, -0.0077748779f, +0.0043392521f,
        -0.0023433708f, +0.0011834015f, -0.0005427604f, +0.0002185407f,
        -0.0000735260f, +0.0000188839f, -0.0000028063f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000028063f, +0.0000188839f, -0.0000735260f, +0.0002185407f,
        -0.0005427604f, +0.0011834015f, -0.0023433708f, +0.0043392521f,
        -0.0077748779f, +0.0142966758f, -0.0318593467f, +0.9979973240f,
        +0.0343262200f, -0.0149662792f, +0.0080939427f, -0.0045199995f,
        +0.0024495327f, -0.0012439113f, +0.0005748807f, -0.0002338727f,
        +0.0000798387f, -0.0000209910f, +0.0000033144f, -0.0000000028f
    },
    {
        -0.0000051172f, +0.0000355820f, -0.0001402267f, +0.0004199031f,
        -0.0010483357f, +0.0022945952f, -0.0045566386f, +0.0084528537f,
        -0.0151514077f, +0.0277905079f, -0.0611219134f, +0.9920041847f,
        +0.0709654097f, -0.0304553129f, +0.0164205964f, -0.0091717408f,
        +0.0049788618f, -0.0025352555f, +0.0011760909f, -0.0004808896f,
        +0.0001653406f, -0.0000439673f, +0.0000071402f, -0.0000000224f
    },
    {
        -0.0000069535f, +0.0000500567f, -0.0001997236f, +0.0006025858f,
        -0.0015124149f, +0.0033233465f, -0.0066184825f, +0.0123003897f,
        -0.0220580368f, +0.0403627372f, -0.0876824355f, +0.9820651127f,
        +0.1097405537f, -0.0463076883f, +0.0248873056f, -0.0139025690f,
        +0.0075594333f, -0.0038596835f, +0.0017971530f, -0.0007385310f,
        +0.0002557187f, -0.0000687613f, +0.0000114700f, -0.0000000750f
    },
    {
        -0.0000083432f, +0.0000623081f, -0.0002517679f, +0.0007654320f,
        -0.0019314560f, +0.0042609450f, -0.0085103569f, +0.0158462501f,
        -0.0284314902f, +0.0519097820f, -0.1114606930f, +0.9682538750f,
        +0.1504526450f, -0.0623526392f, +0.0333959862f, -0.0186566958f,
        +0.0101607794f, -0.0052017031f, +0.0024309532f, -0.0010039595f,
        +0.0003500502f, -0.0000951590f, +0.0000162848f, -0.0000001762f
    },
    {
        -0.0000093200f, +0.0000723707f, -0.0002962347f, +0.0009076231f,
        -0.0023026917f, +0.0051002440f, -0.0102166050f, +0.0190598760f,
        -0.0342172021f, +0.0623437547f, -0.1324016689f, +0.9506728031f,
        +0.1928820316f, -0.0784094115f, +0.0418440948f, -0.0233757758f,
        +0.0127508293f, -0.0065448293f, +0.0030698041f, -0.0012740408f,
        +0.0004472797f, -0.0001228985f, +0.0000215531f, -0.0000003398f
    },
    {
        -0.0000099218f, +0.0000803100f, -0.0003331156f, +0.0010286690f,
        -0.0026241236f, +0.0058356775f, -0.0117245319f, +0.0219159352f,
        -0.0393696295f, +0.0715927976f, -0.1504754667f, +0.9294518630f,
        +0.2367899788f, -0.0942887757f, +0.0501256871f, -0.0279995362f,
        +0.0152962545f, -0.0078717557f, +0.0037055186f, -0.0015453689f,
        +0.0005462256f, -0.0001516701f, +0.0000272302f, -0.0000005775f
    },
    {
        -0.0000101901f, +0.0000862186f, -0.0003625099f, +0.0011283939f,
        -0.0028945063f, +0.0064632554f, -0.0130244366f, +0.0243944269f,
        -0.0438524507f, +0.0796012403f, -0.1656769911f, +0.9047474747f,
        +0.2819204347f, -0.1097946714f, +0.0581325547f, -0.0324664563f,
        +0.0177628446f, -0.0091645441f, +0.0043294942f, -0.0018142982f,
        +0.0006455881f, -0.0001811176f, +0.0000332571f, -0.0000008984f
    },
    {
        -0.0000101683f, +0.0000902122f, -0.0003846154f, +0.0012069187f,
        -0.0031133231f, +0.0069805397f, -0.0141096059f, +0.0264807174f,
        -0.0476386478f, +0.0863295806f, -0.1780253982f, +0.8767410952f,
        +0.3280019845f, -0.1247259707f, +0.0657554308f, -0.0367144873f,
        +0.0201159107f, -0.0104048325f, +0.0049328089f, -0.0020769795f,
        +0.0007439606f, -0.0002108394f, +0.0000395610f, -0.0000013087f
    },
    {
        -0.0000099007f, +0.0000924257f, -0.0003997180f, +0.0012646409f,
        -0.0032807525f, +0.0073866010f, -0.0149762701f, +0.0281655088f,
        -0.0507104777f, +0.0917542956f, -0.1875633263f, +0.8456375775f,
        +0.3747499743f, -0.1388783446f, +0.0728852517f, -0.0406818075f,
        +0.0223207119f, -0.0115740576f, +0.0055063250f, -0.0023294028f,
        +0.0008398422f, -0.0002403912f, +0.0000460543f, -0.0000018109f
    },
    {
        -0.0000094318f, +0.0000930092f, -0.0004081806f, +0.0013022106f,
        -0.0033976276f, +0.0076819573f, -0.0156235240f, +0.0294447433f,
        -0.0530593343f, +0.0958674873f, -0.1943559165f, +0.8116633236f,
        +0.4218687880f, -0.1520462140f, +0.0794144622f, -0.0443076031f,
        +0.0243428992f, -0.0126536909f, +0.0060408025f, -0.0025674435f,
        +0.0009316541f, -0.0002692897f, +0.0000526353f, -0.0000024037f
    },
    {
        -0.0000088047f, +0.0000921243f, -0.0004104323f, +0.0013205063f,
        -0.0034653890f, +0.0078684977f, -0.0160532150f, +0.0303194471f,
        -0.0546855091f, +0.0986763736f, -0.1984896390f, +0.7750642500f,
        +0.4690542534f, -0.1640247678f, +0.0852383507f, -0.0475328662f,
        +0.0261489746f, -0.0136254854f, +0.0065270196f, -0.0027869147f,
        +0.0010177571f, -0.0002970167f, +0.0000591884f, -0.0000030815f
    },
    {
        -0.0000080611f, +0.0000899403f, -0.0004069570f, +0.0013206070f,
        -0.0034860316f, +0.0079493917f, -0.0162698005f, +0.0307955176f,
        -0.0555978548f, +0.1002026338f, -0.2000709375f, +0.7361035848f,
        +0.5159961578f, -0.1746120273f, +0.0902563986f, -0.0503012008f,
        +0.0277067564f, -0.0144717289f, +0.0069558992f, -0.0029836220f,
        +0.0010964721f, -0.0003230251f, +0.0000655847f, -0.0000038345f
    },
    {
        -0.0000072398f, +0.0000866310f, -0.0003982815f, +0.0013037653f,
        -0.0034620479f, +0.0079289853f, -0.0162801796f, +0.0308834592f,
        -0.0558133600f, +0.1004816186f, -0.1992247116f, +0.6950595211f,
        +0.5623808472f, -0.1836109370f, +0.0943736309f, -0.0525596267f,
        +0.0289858477f, -0.0151755034f, +0.0073186402f, -0.0031534228f,
        +0.0011661023f, -0.0003467444f, +0.0000716832f, -0.0000046481f
    },
    {
        -0.0000063765f, +0.0000823707f, -0.0003849653f, +0.0012713778f,
        -0.0033963663f, +0.0078126882f, -0.0160934999f, +0.0305980721f,
        -0.0553566427f, +0.0995614398f, -0.1960926535f, +0.6522227481f,
        +0.6078938857f, -0.1908314595f, +0.0975019503f, -0.0542593720f,
        +0.0299581009f, -0.0157209447f, +0.0076068516f, -0.0032922882f,
        +0.0012249570f, -0.0003675889f, +0.0000773321f, -0.0000055033f
    },
    {
        -0.0000055033f, +0.0000773321f, -0.0003675889f, +0.0012249570f,
        -0.0032922882f, +0.0076068516f, -0.0157209447f, +0.0299581009f,
        -0.0542593720f, +0.0975019503f, -0.1908314595f, +0.6078938857f,
        +0.6522227481f, -0.1960926535f, +0.0995614398f, -0.0553566427f,
        +0.0305980721f, -0.0160934999f, +0.0078126882f, -0.0033963663f,
        +0.0012713778f, -0.0003849653f, +0.0000823707f, -0.0000063765f
    },
    {
        -0.0000046481f, +0.0000716832f, -0.0003467444f, +0.0011661023f,
        -0.0031534228f, +0.0073186402f, -0.0151755034f, +0.0289858477f,
        -0.0525596267f, +0.0943736309f, -0.1836109370f, +0.5623808472f,
        +0.6950595211f, -0.1992247116f, +0.1004816186f, -0.0558133600f,
        +0.0308834592f, -0.0162801796f, +0.0079289853f, -0.0034620479f,
        +0.0013037653f, -0.0003982815f, +0.0000866310f, -0.0000072398f
    },
    {
        -0.0000038345f, +0.0000655847f, -0.0003230251f, +0.0010964721f,
        -0.0029836220f, +0.0069558992f, -0.0144717289f, +0.0277067564f,
        -0.0503012008f, +0.0902563986f, -0.1746120273f, +0.5159961578f,
        +0.7361035848f, -0.2000709375f, +0.1002026338f, -0.0555978548f,
        +0.0307955176f, -0.0162698005f, +0.0079493917f, -0.0034860316f,
        +0.0013206070f, -0.0004069570f, +0.0000899403f, -0.0000080611f
    },
    {
        -0.0000030815f, +0.0000591884f, -0.0002970167f, +0.0010177571f,
        -0.0027869147f, +0.0065270196f, -0.0136254854f, +0.0261489746f,
        -0.0475328662f, +0.0852383507f, -0.1640247678f, +0.4690542534f,
        +0.7750642500f, -0.1984896390f, +0.0986763736f, -0.0546855091f,
        +0.0303194471f, -0.0160532150f, +0.0078684977f, -0.0034653890f,
        +0.0013205063f, -0.0004104323f, +0.0000921243f, -0.0000088047f
    },
    {
        -0.0000024037f, +0.0000526353f, -0.0002692897f, +0.0009316541f,
        -0.0025674435f, +0.0060408025f, -0.0126536909f, +0.0243428992f,
        -0.0443076031f, +0.0794144622f, -0.1520462140f, +0.4218687880f,
        +0.8116633236f, -0.1943559165f, +0.0958674873f, -0.0530593343f,
        +0.0294447433f, -0.0156235240f, +0.0076819573f, -0.0033976276f,
        +0.0013022106f, -0.0004081806f, +0.0000930092f, -0.0000094318f
    },
    {
        -0.0000018109f, +0.0000460543f, -0.0002403912f, +0.0008398422f,
        -0.0023294028f, +0.0055063250f, -0.0115740576f, +0.0223207119f,
        -0.0406818075f, +0.0728852517f, -0.1388783446f, +0.3747499743f,
        +0.8456375775f, -0.1875633263f, +0.0917542956f, -0.0507104777f,
        +0.0281655088f, -0.0149762701f, +0.0073866010f, -0.0032807525f,
        +0.0012646409f, -0.0003997180f, +0.0000924257f, -0.0000099007f
    },
    {
        -0.0000013087f, +0.0000395610f, -0.0002108394f, +0.0007439606f,
        -0.0020769795f, +0.0049328089f, -0.0104048325f, +0.0201159107f,
        -0.0367144873f, +0.0657554308f, -0.1247259707f, +0.3280019845f,
        +0.8767410952f, -0.1780253982f, +0.0863295806f, -0.0476386478f,
        +0.0264807174f, -0.0141096059f, +0.0069805397f, -0.0031133231f,
        +0.0012069187f, -0.0003846154f, +0.0000902122f, -0.0000101683f
    },
    {
        -0.0000008984f, +0.0000332571f, -0.0001811176f, +0.0006455881f,
        -0.0018142982f, +0.0043294942f, -0.0091645441f, +0.0177628446f,
        -0.0324664563f, +0.0581325547f, -0.1097946714f, +0.2819204347f,
        +0.9047474747f, -0.1656769911f, +0.0796012403f, -0.0438524507f,
        +0.0243944269f, -0.0130244366f, +0.0064632554f, -0.0028945063f,
        +0.0011283939f, -0.0003625099f, +0.0000862186f, -0.0000101901f
    },
    {
        -0.0000005775f, +0.0000272302f, -0.0001516701f, +0.0005462256f,
        -0.0015453689f, +0.0037055186f, -0.0078717557f, +0.0152962545f,
        -0.0279995362f, +0.0501256871f, -0.0942887757f, +0.2367899788f,
        +0.9294518630f, -0.1504754667f, +0.0715927976f, -0.0393696295f,
        +0.0219159352f, -0.0117245319f, +0.0058356775f, -0.0026241236f,
        +0.0010286690f, -0.0003331156f, +0.0000803100f, -0.0000099218f
    },
    {
        -0.0000003398f, +0.0000215531f, -0.0001228985f, +0.0004472797f,
        -0.0012740408f, +0.0030698041f, -0.0065448293f, +0.0127508293f,
        -0.0233757758f, +0.0418440948f, -0.0784094115f, +0.1928820316f,
        +0.9506728031f, -0.1324016689f, +0.0623437547f, -0.0342172021f,
        +0.0190598760f, -0.0102166050f, +0.0051002440f, -0.0023026917f,
        +0.0009076231f, -0.0002962347f, +0.0000723707f, -0.0000093200f
    },
    {
        -0.0000001762f, +0.0000162848f, -0.0000951590f, +0.0003500502f,
        -0.0010039595f, +0.0024309532f, -0.0052017031f, +0.0101607794f,
        -0.0186566958f, +0.0333959862f, -0.0623526392f, +0.1504526450f,
        +0.9682538750f, -0.1114606930f, +0.0519097820f, -0.0284314902f,
        +0.0158462501f, -0.0085103569f, +0.0042609450f, -0.0019314560f,
        +0.0007654320f, -0.0002517679f, +0.0000623081f, -0.0000083432f
    },
    {
        -0.0000000750f, +0.0000114700f, -0.0000687613f, +0.0002557187f,
        -0.0007385310f, +0.0017971530f, -0.0038596835f, +0.0075594333f,
        -0.0139025690f, +0.0248873056f, -0.0463076883f, +0.1097405537f,
        +0.9820651127f, -0.0876824355f, +0.0403627372f, -0.0220580368f,
        +0.0123003897f, -0.0066184825f, +0.0033233465f, -0.0015124149f,
        +0.0006025858f, -0.0001997236f, +0.0000500567f, -0.0000069535f
    },
    {
        -0.0000000224f, +0.0000071402f, -0.0000439673f, +0.0001653406f,
        -0.0004808896f, +0.0011760909f, -0.0025352555f, +0.0049788618f,
        -0.0091717408f, +0.0164205964f, -0.0304553129f, +0.0709654097f,
        +0.9920041847f, -0.0611219134f, +0.0277905079f, -0.0151514077f,
        +0.0084528537f, -0.0045566386f, +0.0022945952f, -0.0010483357f,
        +0.0004199031f, -0.0001402267f, +0.0000355820f, -0.0000051172f
    },
    {
        -0.0000000028f, +0.0000033144f, -0.0000209910f, +0.0000798387f,
        -0.0002338727f, +0.0005748807f, -0.0012439113f, +0.0024495327f,
        -0.0045199995f, +0.0080939427f, -0.0149662792f, +0.0343262200f,
        +0.9979973240f, -0.0318593467f, +0.0142966758f, -0.0077748779f,
        +0.0043392521f, -0.0023433708f, +0.0011834015f, -0.0005427604f,
        +0.0002185407f, -0.0000735260f, +0.0000188839f, -0.0000028063f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0000028063f, +0.0000188839f, -0.0000735260f,
        +0.0002185407f, -0.0005427604f, +0.0011834015f, -0.0023433708f,
        +0.0043392521f, -0.0077748779f, +0.0142966758f, -0.0318593467f,
        +0.9979973240f, +0.0343262200f, -0.0149662792f, +0.0080939427f,
        -0.0045199995f, +0.0024495327f, -0.0012439113f, +0.0005748807f,
        -0.0002338727f, +0.0000798387f, -0.0000209910f, +0.0000033144f
    }
};

_Alignas(64) static float const kernels_32p_cubic [32][32] =
{
    {
        +0.0000012645f, -0.0000067946f, +0.0000224320f, -0.0000592696f,
        +0.0001354267f, -0.0002779601f, +0.0005248199f, -0.0009271590f,
        +0.0015533014f, -0.0024979610f, +0.0039061912f, -0.0060399793f,
        +0.0094887898f, -0.0160340981f, +0.0348973545f, +0.9980182787f,
        -0.0324688367f, +0.0153931209f, -0.0091849261f, +0.0058605138f,
        -0.0037898167f, +0.0024200032f, -0.0015011736f, +0.0008931230f,
        -0.0005034712f, +0.0002652818f, -0.0001284086f, +0.0000557190f,
        -0.0000208343f, +0.0000061808f, -0.0000010818f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000010818f, +0.0000061808f, -0.0000208343f, +0.0000557190f,
        -0.0001284086f, +0.0002652818f, -0.0005034712f, +0.0008931230f,
        -0.0015011736f, +0.0024200032f, -0.0037898167f, +0.0058605138f,
        -0.0091849261f, +0.0153931209f, -0.0324688367f, +0.9980182787f,
        +0.0348973545f, -0.0160340981f, +0.0094887898f, -0.0060399793f,
        +0.0039061912f, -0.0024979610f, +0.0015533014f, -0.0009271590f,
        +0.0005248199f, -0.0002779601f, +0.0001354267f, -0.0000592696f,
        +0.0000224320f, -0.0000067946f, +0.0000012645f, -0.0000000012f
    },
    {
        -0.0000019828f, +0.0000117116f, -0.0000399064f, +0.0001073872f,
        -0.0002485605f, +0.0005152066f, -0.0009803506f, +0.0017427091f,
        -0.0029340187f, +0.0047357111f, -0.0074219332f, +0.0114781442f,
        -0.0179693531f, +0.0299983759f, -0.0623715136f, +0.9920875035f,
        +0.0720625023f, -0.0325493562f, +0.0191780947f, -0.0121919281f,
        +0.0078847572f, -0.0050457431f, +0.0031413256f, -0.0018780675f,
        +0.0010652546f, -0.0005656298f, +0.0002764734f, -0.0001215100f,
        +0.0000462618f, -0.0000141538f, +0.0000027095f, -0.0000000094f
    },
    {
        -0.0000027078f, +0.0000165689f, -0.0000570867f, +0.0001545864f,
        -0.0003593845f, +0.0007474069f, -0.0014259322f, +0.0025401311f,
        -0.0042836961f, +0.0069228469f, -0.0108581105f, +0.0167941732f,
        -0.0262642439f, +0.0436828569f, -0.0895942203f, +0.9822507145f,
        +0.1113126084f, -0.0493742406f, +0.0289588411f, -0.0183849603f,
        +0.0118894812f, -0.0076137015f, +0.0047456404f, -0.0028417551f,
        +0.0016151325f, -0.0008597757f, +0.0004215934f, -0.0001860652f,
        +0.0000712550f, -0.0000220148f, +0.0000043292f, -0.0000000316f
    },
    {
        -0.0000032650f, +0.0000207412f, -0.0000722804f, +0.0001969809f,
        -0.0004599821f, +0.0009598425f, -0.0018360835f, +0.0032777097f,
        -0.0055368629f, +0.0089594224f, -0.0140635597f, +0.0217551320f,
        -0.0339889340f, +0.0563285222f, -0.1140474994f, +0.9685792243f,
        +0.1524439164f, -0.0663268640f, +0.0387173115f, -0.0245455488f,
        +0.0158724270f, -0.0101710034f, +0.0063469827f, -0.0038066817f,
        +0.0021679066f, -0.0011569305f, +0.0005691040f, -0.0002522071f,
        +0.0000971433f, -0.0000303015f, +0.0000061130f, -0.0000000740f
    },
    {
        -0.0000036648f, +0.0000242282f, -0.0000854264f, +0.0002343176f,
        -0.0005496306f, +0.0011508111f, -0.0022072742f, +0.0039487735f,
        -0.0066817850f, +0.0108259326f, -0.0170072549f, +0.0263132746f,
        -0.0410717922f, +0.0678329093f, -0.1356665369f, +0.9511719921f,
        +0.1952331456f, -0.0832164873f, +0.0483363121f, -0.0305983271f,
        +0.0197844725f, -0.0126859712f, +0.0079254716f, -0.0047608622f,
        +0.0027167232f, -0.0014534276f, +0.0007172029f, -0.0003191422f,
        +0.0001236264f, -0.0000389244f, +0.0000080461f, -0.0000001425f
    },
    {
        -0.0000039198f, +0.0000270400f, -0.0000964957f, +0.0002664240f,
        -0.0006277834f, +0.0013189554f, -0.0025365958f, +0.0045477043f,
        -0.0077084263f, +0.0125055128f, -0.0196621911f, +0.0304269704f,
        -0.0474507593f, +0.0781096722f, -0.1544112244f, +0.9301547546f,
        +0.2394390841f, -0.0998451493f, +0.0576964130f, -0.0364669358f,
        +0.0235758735f, -0.0151264486f, +0.0094608367f, -0.0056920032f,
        +0.0032544977f, -0.0017454402f, +0.0008639854f, -0.0003860180f,
        +0.0001503745f, -0.0000477810f, +0.0000101091f, -0.0000002417f
    },
    {
        -0.0000040443f, +0.0000291962f, -0.0001054898f, +0.0002932065f,
        -0.0006940681f, +0.0014632641f, -0.0028217731f, +0.0050699679f,
        -0.0086085142f, +0.0139840604f, -0.0220055878f, +0.0340610154f,
        -0.0530737661f, +0.0870889414f, -0.1702659926f, +0.9056789241f,
        +0.2848043664f, -0.1160094089f, +0.0666772507f, -0.0420749059f,
        +0.0271968506f, -0.0174601824f, +0.0109326585f, -0.0065876465f,
        +0.0037739967f, -0.0020290245f, +0.0010074656f, -0.0004519321f,
        +0.0001770302f, -0.0000567561f, +0.0000122775f, -0.0000003753f
    },
    {
        -0.0000040537f, +0.0000307247f, -0.0001124389f, +0.0003146466f,
        -0.0007482824f, +0.0015830695f, -0.0030611663f, +0.0055121300f,
        -0.0093755812f, +0.0152503190f, -0.0240190360f, +0.0371868591f,
        -0.0578990264f, +0.0947175064f, -0.1832394230f, +0.8779202636f,
        +0.3310574185f, -0.1315021858f, +0.0751588798f, -0.0473465686f,
        +0.0305981951f, -0.0196552177f, +0.0123206187f, -0.0074353219f,
        +0.0042679248f, -0.0023001659f, +0.0011455986f, -0.0005159412f,
        +0.0002032129f, -0.0000657233f, +0.0000145225f, -0.0000005454f
    },
    {
        -0.0000039641f, +0.0000316605f, -0.0001173988f, +0.0003307972f,
        -0.0007903882f, +0.0016780403f, -0.0032537663f, +0.0058718582f,
        -0.0100049828f, +0.0162959253f, -0.0256885889f, +0.0397827472f,
        -0.0618952059f, +0.1009588230f, -0.1933636434f, +0.8470773515f,
        +0.3779145552f, -0.1461146806f, +0.0830231596f, -0.0522079819f,
        +0.0337318847f, -0.0216803022f, +0.0136047578f, -0.0082227040f,
        +0.0047290161f, -0.0025548284f, +0.0012763049f, -0.0005770723f,
        +0.0002285221f, -0.0000745454f, +0.0000168103f, -0.0000007529f
    },
    {
        -0.0000037923f, +0.0000320450f, -0.0001204491f, +0.0003417769f,
        -0.0008205037f, +0.0017481703f, -0.0033991812f, +0.0061479103f,
        -0.0104938934f, +0.0171154204f, -0.0270047965f, +0.0418337815f,
        -0.0650414694f, +0.1057928503f, -0.2006935166f, +0.8133698514f,
        +0.4250822109f, -0.1596383586f, +0.0901551619f, -0.0565878630f,
        +0.0365517032f, -0.0235052933f, +0.0147657362f, -0.0089377738f,
        +0.0051501286f, -0.0027890067f, +0.0013974965f, -0.0006343344f,
        +0.0002525421f, -0.0000830761f, +0.0000191031f, -0.0000009968f
    },
    {
        -0.0000035546f, +0.0000319243f, -0.0001216900f, +0.0003477655f,
        -0.0008388934f, +0.0017937639f, -0.0034976161f, +0.0063401091f,
        -0.0108412791f, +0.0177062244f, -0.0279626859f, +0.0433318972f,
        -0.0673274067f, +0.1092157212f, -0.2053056334f, +0.7770366035f,
        +0.4722592828f, -0.1718669778f, +0.0964445838f, -0.0604185172f,
        +0.0390138560f, -0.0251015642f, +0.0157850964f, -0.0095689817f,
        +0.0055243410f, -0.0029987796f, +0.0015071050f, -0.0006867310f,
        +0.0002748471f, -0.0000911614f, +0.0000213585f, -0.0000012744f
    },
    {
        -0.0000032671f, +0.0000313486f, -0.0001212397f, +0.0003489973f,
        -0.0008459576f, +0.0018154180f, -0.0035498463f, +0.0064493058f,
        -0.0110478500f, +0.0180685782f, -0.0285616886f, +0.0442757611f,
        -0.0687528444f, +0.1112392555f, -0.2072971213f, +0.7383335558f,
        +0.5191395661f, -0.1825986398f, +0.1017871517f, -0.0636367499f,
        +0.0410775747f, -0.0264424041f, +0.0166455225f, -0.0101054108f,
        +0.0058450504f, -0.0031803656f, +0.0016031094f, -0.0007332736f,
        +0.0002950064f, -0.0000986415f, +0.0000235308f, -0.0000015811f
    },
    {
        -0.0000029453f, +0.0000303707f, -0.0001192314f, +0.0003457548f,
        -0.0008422197f, +0.0018140007f, -0.0035571840f, +0.0064773308f,
        -0.0111159924f, +0.0182054520f, -0.0288055172f, +0.0446705934f,
        -0.0693275455f, +0.1118903228f, -0.2067842857f, +0.6975315552f,
        +0.5654142596f, -0.1916378464f, +0.1060859992f, -0.0661847528f,
        +0.0427057015f, -0.0275034058f, +0.0173310942f, -0.0105369371f,
        +0.0061060692f, -0.0033301778f, +0.0016835662f, -0.0007729959f,
        +0.0003125909f, -0.0001053529f, +0.0000255708f, -0.0000019107f
    },
    {
        -0.0000026035f, +0.0000290450f, -0.0001158103f, +0.0003383623f,
        -0.0008283135f, +0.0017906283f, -0.0035214398f, +0.0064269349f,
        -0.0110496830f, +0.0181224240f, -0.0287019943f, +0.0445279186f,
        -0.0690708056f, +0.1112100673f, -0.2039010961f, +0.6549140203f,
        +0.6107745167f, -0.1987975381f, +0.1092530039f, -0.0680109518f,
        +0.0438652479f, -0.0282628379f, +0.0178275310f, -0.0108543862f,
        +0.0063017213f, -0.0034448798f, +0.0017466391f, -0.0008049685f,
        +0.0003271787f, -0.0001111306f, +0.0000274267f, -0.0000022548f
    },
    {
        -0.0000022548f, +0.0000274267f, -0.0001111306f, +0.0003271787f,
        -0.0008049685f, +0.0017466391f, -0.0034448798f, +0.0063017213f,
        -0.0108543862f, +0.0178275310f, -0.0282628379f, +0.0438652479f,
        -0.0680109518f, +0.1092530039f, -0.1987975381f, +0.6107745167f,
        +0.6549140203f, -0.2039010961f, +0.1112100673f, -0.0690708056f,
        +0.0445279186f, -0.0287019943f, +0.0181224240f, -0.0110496830f,
        +0.0064269349f, -0.0035214398f, +0.0017906283f, -0.0008283135f,
        +0.0003383623f, -0.0001158103f, +0.0000290450f, -0.0000026035f
    },
    {
        -0.0000019107f, +0.0000255708f, -0.0001053529f, +0.0003125909f,
        -0.0007729959f, +0.0016835662f, -0.0033301778f, +0.0061060692f,
        -0.0105369371f, +0.0173310942f, -0.0275034058f, +0.0427057015f,
        -0.0661847528f, +0.1060859992f, -0.1916378464f, +0.5654142596f,
        +0.6975315552f, -0.2067842857f, +0.1118903228f, -0.0693275455f,
        +0.0446705934f, -0.0288055172f, +0.0182054520f, -0.0111159924f,
        +0.0064773308f, -0.0035571840f, +0.0018140007f, -0.0008422197f,
        +0.0003457548f, -0.0001192314f, +0.0000303707f, -0.0000029453f
    },
    {
        -0.0000015811f, +0.0000235308f, -0.0000986415f, +0.0002950064f,
        -0.0007332736f, +0.0016031094f, -0.0031803656f, +0.0058450504f,
        -0.0101054108f, +0.0166455225f, -0.0264424041f, +0.0410775747f,
        -0.0636367499f, +0.1017871517f, -0.1825986398f, +0.5191395661f,
        +0.7383335558f, -0.2072971213f, +0.1112392555f, -0.0687528444f,
        +0.0442757611f, -0.0285616886f, +0.0180685782f, -0.0110478500f,
        +0.0064493058f, -0.0035498463f, +0.0018154180f, -0.0008459576f,
        +0.0003489973f, -0.0001212397f, +0.0000313486f, -0.0000032671f
    },
    {
        -0.0000012744f, +0.0000213585f, -0.0000911614f, +0.0002748471f,
        -0.0006867310f, +0.0015071050f, -0.0029987796f, +0.0055243410f,
        -0.0095689817f, +0.0157850964f, -0.0251015642f, +0.0390138560f,
        -0.0604185172f, +0.0964445838f, -0.1718669778f, +0.4722592828f,
        +0.7770366035f, -0.2053056334f, +0.1092157212f, -0.0673274067f,
        +0.0433318972f, -0.0279626859f, +0.0177062244f, -0.0108412791f,
        +0.0063401091f, -0.0034976161f, +0.0017937639f, -0.0008388934f,
        +0.0003477655f, -0.0001216900f, +0.0000319243f, -0.0000035546f
    },
    {
        -0.0000009968f, +0.0000191031f, -0.0000830761f, +0.0002525421f,
        -0.0006343344f, +0.0013974965f, -0.0027890067f, +0.0051501286f,
        -0.0089377738f, +0.0147657362f, -0.0235052933f, +0.0365517032f,
        -0.0565878630f, +0.0901551619f, -0.1596383586f, +0.4250822109f,
        +0.8133698514f, -0.2006935166f, +0.1057928503f, -0.0650414694f,
        +0.0418337815f, -0.0270047965f, +0.0171154204f, -0.0104938934f,
        +0.0061479103f, -0.0033991812f, +0.0017481703f, -0.0008205037f,
        +0.0003417769f, -0.0001204491f, +0.0000320450f, -0.0000037923f
    },
    {
        -0.0000007529f, +0.0000168103f, -0.0000745454f, +0.0002285221f,
        -0.0005770723f, +0.0012763049f, -0.0025548284f, +0.0047290161f,
        -0.0082227040f, +0.0136047578f, -0.0216803022f, +0.0337318847f,
        -0.0522079819f, +0.0830231596f, -0.1461146806f, +0.3779145552f,
        +0.8470773515f, -0.1933636434f, +0.1009588230f, -0.0618952059f,
        +0.0397827472f, -0.0256885889f, +0.0162959253f, -0.0100049828f,
        +0.0058718582f, -0.0032537663f, +0.0016780403f, -0.0007903882f,
        +0.0003307972f, -0.0001173988f, +0.0000316605f, -0.0000039641f
    },
    {
        -0.0000005454f, +0.0000145225f, -0.0000657233f, +0.0002032129f,
        -0.0005159412f, +0.0011455986f, -0.0023001659f, +0.0042679248f,
        -0.0074353219f, +0.0123206187f, -0.0196552177f, +0.0305981951f,
        -0.0473465686f, +0.0751588798f, -0.1315021858f, +0.3310574185f,
        +0.8779202636f, -0.1832394230f, +0.0947175064f, -0.0578990264f,
        +0.0371868591f, -0.0240190360f, +0.0152503190f, -0.0093755812f,
        +0.0055121300f, -0.0030611663f, +0.0015830695f, -0.0007482824f,
        +0.0003146466f, -0.0001124389f, +0.0000307247f, -0.0000040537f
    },
    {
        -0.0000003753f, +0.0000122775f, -0.0000567561f, +0.0001770302f,
        -0.0004519321f, +0.0010074656f, -0.0020290245f, +0.0037739967f,
        -0.0065876465f, +0.0109326585f, -0.0174601824f, +0.0271968506f,
        -0.0420749059f, +0.0666772507f, -0.1160094089f, +0.2848043664f,
        +0.9056789241f, -0.1702659926f, +0.0870889414f, -0.0530737661f,
        +0.0340610154f, -0.0220055878f, +0.0139840604f, -0.0086085142f,
        +0.0050699679f, -0.0028217731f, +0.0014632641f, -0.0006940681f,
        +0.0002932065f, -0.0001054898f, +0.0000291962f, -0.0000040443f
    },
    {
        -0.0000002417f, +0.0000101091f, -0.0000477810f, +0.0001503745f,
        -0.0003860180f, +0.0008639854f, -0.0017454402f, +0.0032544977f,
        -0.0056920032f, +0.0094608367f, -0.0151264486f, +0.0235758735f,
        -0.0364669358f, +0.0576964130f, -0.0998451493f, +0.2394390841f,
        +0.9301547546f, -0.1544112244f, +0.0781096722f, -0.0474507593f,
        +0.0304269704f, -0.0196621911f, +0.0125055128f, -0.0077084263f,
        +0.0045477043f, -0.0025365958f, +0.0013189554f, -0.0006277834f,
        +0.0002664240f, -0.0000964957f, +0.0000270400f, -0.0000039198f
    },
    {
        -0.0000001425f, +0.0000080461f, -0.0000389244f, +0.0001236264f,
        -0.0003191422f, +0.0007172029f, -0.0014534276f, +0.0027167232f,
        -0.0047608622f, +0.0079254716f, -0.0126859712f, +0.0197844725f,
        -0.0305983271f, +0.0483363121f, -0.0832164873f, +0.1952331456f,
        +0.9511719921f, -0.1356665369f, +0.0678329093f, -0.0410717922f,
        +0.0263132746f, -0.0170072549f, +0.0108259326f, -0.0066817850f,
        +0.0039487735f, -0.0022072742f, +0.0011508111f, -0.0005496306f,
        +0.0002343176f, -0.0000854264f, +0.0000242282f, -0.0000036648f
    },
    {
        -0.0000000740f, +0.0000061130f, -0.0000303015f, +0.0000971433f,
        -0.0002522071f, +0.0005691040f, -0.0011569305f, +0.0021679066f,
        -0.0038066817f, +0.0063469827f, -0.0101710034f, +0.0158724270f,
        -0.0245455488f, +0.0387173115f, -0.0663268640f, +0.1524439164f,
        +0.9685792243f, -0.1140474994f, +0.0563285222f, -0.0339889340f,
        +0.0217551320f, -0.0140635597f, +0.0089594224f, -0.0055368629f,
        +0.0032777097f, -0.0018360835f, +0.0009598425f, -0.0004599821f,
        +0.0001969809f, -0.0000722804f, +0.0000207412f, -0.0000032650f
    },
    {
        -0.0000000316f, +0.0000043292f, -0.0000220148f, +0.0000712550f,
        -0.0001860652f, +0.0004215934f, -0.0008597757f, +0.0016151325f,
        -0.0028417551f, +0.0047456404f, -0.0076137015f, +0.0118894812f,
        -0.0183849603f, +0.0289588411f, -0.0493742406f, +0.1113126084f,
        +0.9822507145f, -0.0895942203f, +0.0436828569f, -0.0262642439f,
        +0.0167941732f, -0.0108581105f, +0.0069228469f, -0.0042836961f,
        +0.0025401311f, -0.0014259322f, +0.0007474069f, -0.0003593845f,
        +0.0001545864f, -0.0000570867f, +0.0000165689f, -0.0000027078f
    },
    {
        -0.0000000094f, +0.0000027095f, -0.0000141538f, +0.0000462618f,
        -0.0001215100f, +0.0002764734f, -0.0005656298f, +0.0010652546f,
        -0.0018780675f, +0.0031413256f, -0.0050457431f, +0.0078847572f,
        -0.0121919281f, +0.0191780947f, -0.0325493562f, +0.0720625023f,
        +0.9920875035f, -0.0623715136f, +0.0299983759f, -0.0179693531f,
        +0.0114781442f, -0.0074219332f, +0.0047357111f, -0.0029340187f,
        +0.0017427091f, -0.0009803506f, +0.0005152066f, -0.0002485605f,
        +0.0001073872f, -0.0000399064f, +0.0000117116f, -0.0000019828f
    },
    {
        -0.0000000012f, +0.0000012645f, -0.0000067946f, +0.0000224320f,
        -0.0000592696f, +0.0001354267f, -0.0002779601f, +0.0005248199f,
        -0.0009271590f, +0.0015533014f, -0.0024979610f, +0.0039061912f,
        -0.0060399793f, +0.0094887898f, -0.0160340981f, +0.0348973545f,
        +0.9980182787f, -0.0324688367f, +0.0153931209f, -0.0091849261f,
        +0.0058605138f, -0.0037898167f, +0.0024200032f, -0.0015011736f,
        +0.0008931230f, -0.0005034712f, +0.0002652818f, -0.0001284086f,
        +0.0000557190f, -0.0000208343f, +0.0000061808f, -0.0000010818f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0000010818f, +0.0000061808f, -0.0000208343f,
        +0.0000557190f, -0.0001284086f, +0.0002652818f, -0.0005034712f,
        +0.0008931230f, -0.0015011736f, +0.0024200032f, -0.0037898167f,
        +0.0058605138f, -0.0091849261f, +0.0153931209f, -0.0324688367f,
        +0.9980182787f, +0.0348973545f, -0.0160340981f, +0.0094887898f,
        -0.0060399793f, +0.0039061912f, -0.0024979610f, +0.0015533014f,
        -0.0009271590f, +0.0005248199f, -0.0002779601f, +0.0001354267f,
        -0.0000592696f, +0.0000224320f, -0.0000067946f, +0.0000012645f
    }
};

/* -------------------------------------------------------------------------- */
