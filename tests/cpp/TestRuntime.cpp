#include <Kun/Context.hpp>
#include <Kun/Module.hpp>

using namespace kun;

static constexpr size_t num_splits = 4;

static void stage1(Context *__ctx, size_t __stock_idx, size_t __total_time,
                   size_t __start, size_t __length) {
    float *base =
        &__ctx->buffers[0].ptr[__total_time / num_splits * __stock_idx];
    float *out =
        &__ctx->buffers[1].ptr[__total_time / num_splits * __stock_idx];
    for (size_t i = 0; i < __length; i++) {
        out[i + __start] = base[i + __start];
    }
}

static void stage2(Context *__ctx, size_t __stock_idx, size_t __total_time,
                   size_t __start, size_t __length) {
    float *base =
        &__ctx->buffers[1].ptr[__total_time / num_splits * __stock_idx];
    float *out =
        &__ctx->buffers[2].ptr[__total_time / num_splits * __stock_idx];
    for (size_t i = 0; i < __length; i++) {
        out[i + __start] = base[i + __start] * 2;
    }
}

static void stage3(Context *__ctx, size_t __stock_idx, size_t __total_time,
                   size_t __start, size_t __length) {
    float *base1 =
        &__ctx->buffers[1].ptr[__total_time / num_splits * __stock_idx];
    float *base2 =
        &__ctx->buffers[2].ptr[__total_time / num_splits * __stock_idx];
    float *out =
        &__ctx->buffers[3].ptr[__total_time / num_splits * __stock_idx];
    for (size_t i = 0; i < __length; i++) {
        out[i + __start] = base1[i + __start] + base2[i + __start];
    }
}

template <typename T, std::size_t N>
constexpr std::size_t arraySize(const T(&)[N]) { return N; }

static BufferInfo buffers[]{
    {0, "input", BufferKind::INPUT},
    {1, "t1", BufferKind::TEMP},
    {2, "t2", BufferKind::TEMP},
    {3, "output", BufferKind::OUTPUT},
};

namespace {
static constexpr size_t stage1_num_dep = 2;
extern Stage *stage1_dep[stage1_num_dep];
static constexpr size_t stage2_num_dep = 1;
extern Stage *stage2_dep[stage1_num_dep];
} // namespace

static BufferInfo *stage1_in_buf[] = {&buffers[0]};
static BufferInfo *stage1_out_buf[] = {&buffers[1]};
static BufferInfo *stage2_in_buf[] = {&buffers[1]};
static BufferInfo *stage2_out_buf[] = {&buffers[2]};
static BufferInfo *stage3_in_buf[] = {&buffers[1], &buffers[2]};
static BufferInfo *stage3_out_buf[] = {&buffers[3]};

static Stage stages[] = {
    {/*f*/ stage1, /*dependers*/ stage1_dep, /*num_dependers*/ stage1_num_dep,
     /*in_buffers*/ stage1_in_buf, /*num_in_buffers*/ 1,
     /*out_buffers*/ stage1_out_buf, /*num_out_buffers*/ 1, /*pending_out*/ 2,
     /*num_tasks*/ num_splits, /*id*/ 0},
    {/*f*/ stage2, /*dependers*/ stage2_dep, /*num_dependers*/ stage2_num_dep,
     /*in_buffers*/ stage2_in_buf, /*num_in_buffers*/ 1,
     /*out_buffers*/ stage2_out_buf, /*num_out_buffers*/ 1, /*pending_out*/ 1,
     /*num_tasks*/ num_splits, /*id*/ 2},
    {/*f*/ stage1, /*dependers*/ nullptr, /*num_dependers*/ 0,
     /*in_buffers*/ stage3_in_buf, /*num_in_buffers*/ 2,
     /*out_buffers*/ stage3_out_buf, /*num_out_buffers*/ 1, /*pending_out*/ 0,
     /*num_tasks*/ num_splits, /*id*/ 3},
};

namespace {
Stage *stage1_dep[] = {&stages[1], &stages[2]};
Stage *stage2_dep[] = {&stages[2]};
} // namespace


KUN_API Module testRuntimeModule{
    arraySize(stages),
    stages,
    arraySize(buffers),
    buffers
};