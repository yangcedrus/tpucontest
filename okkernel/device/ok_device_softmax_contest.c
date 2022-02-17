#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
#define NPU_NUM okk_npu_num()
#define NO_USE 0
typedef struct {
    int N, C, H, W;
    unsigned long long output_addr;
    unsigned long long input_addr;
} __attribute__((packed)) param_t;

void softmax_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 shape = {.n=param->N, .c=param->C, .h=param->H, .w=param->W};
    dim4 stride;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);

    local_addr_t input_addr, exp_addr, output_addr;
    input_addr = 0;
    exp_addr = input_addr + shape.n * stride.n * sizeof(float);
    output_addr = exp_addr + shape.n * stride.n * sizeof(float);

    OKKERNEL_ASSERT(output_addr + shape.n * stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);


    // get max value


    // sub max value


    // cal exp(X)
    okk_bdc_exp(exp_addr, input_addr, output_addr, &shape);

    // cal sum(exp(X))
    x32 C = {1.0};
    // dim4 kernel_shape = {.n=1,.c=param->C,.h=1,.w=param->C*2};
    // okk_bdc_32bit_set_C(output_addr, C, &kernel_shape, NULL);

    dim4 kernel_shape = {.n=(param->C + 1) / 2, .c=param->C, .h=1, .w=2};
    dim4 kernel_stride;
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    okk_bdc_32bit_set_C(output_addr, C, &kernel_shape, &kernel_stride);
    // OKKERNEL_LOG("kernel shape is [%d,%d,%d,%d]", kernel_shape_2IC.n, kernel_shape_2IC.c, kernel_shape_2IC.h, kernel_shape_2IC.w);
    // OKKERNEL_LOG("kernel stride is [%d,%d,%d,%d]", kernel_stride_2IC.n, kernel_stride_2IC.c, kernel_stride_2IC.h, kernel_stride_2IC.w);

    dim4 kernel_shape_2IC = {.n = (param->C + 1) / 2, .c = param->C, .h = 1, .w = 1};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    okk_bdc_conv2d(input_addr, exp_addr, output_addr, NULL, &shape, param->C, 1, 1, &stride, &kernel_stride_2IC, false, false, NULL, NULL, NULL);

    // div exp(x) by sum
    // OKKERNEL_LOG("stride is [%d,%d,%d,%d]", stride.n, stride.c, stride.h, stride.w);
    // OKKERNEL_LOG("broadcast_stride is [%d,%d,%d,%d]", broadcast_stride.n, broadcast_stride.c, broadcast_stride.h, broadcast_stride.w);
    okk_bdc_div(output_addr, exp_addr, input_addr, &shape, &stride, &stride, &stride);

    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);

    // dim4 outshape={.n=1,.c=1,.h=1,.w=param->C};
    // dim4 outstride={.n=2,.c=2,.h=2,.w=1};
    // kernel_shape_2IC.w=1;
    // okk_gdma_32bit_cpy_L2S(param->output_addr, input_addr, &kernel_shape_2IC, NULL, &kernel_stride_2IC);
    
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
