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

void conduct_softmax(local_addr_t input_addr, local_addr_t exp_addr, local_addr_t output_addr, dim4 shape, dim4 stride){
    // TODO handle overflow
    // get max value
    // sub max value

    // sub constant 10
    okk_bdc_sub_C(input_addr, input_addr, 10.0, &shape, &stride, &stride);

    // cal exp(X)
    okk_bdc_exp(exp_addr, input_addr, output_addr, &shape);

    // cal sum(exp(X))
    x32 C = {1.0};
    dim4 kernel_shape = {.n=1, .c=64, .h=1, .w=2};
    dim4 kernel_stride;
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    // OKKERNEL_LOG("kernel shape is [%d,%d,%d,%d]", kernel_shape.n, kernel_shape.c, kernel_shape.h, kernel_shape.w);
    // OKKERNEL_LOG("kernel stride is [%d,%d,%d,%d]", kernel_stride.n, kernel_stride.c, kernel_stride.h, kernel_stride.w);
    // OKKERNEL_LOG("input addr:%d, exp addr:%d, output addr:%d, total size:%d, mem size:%d", input_addr, exp_addr, output_addr, output_addr + kernel_shape.n * kernel_stride.n * sizeof(float), LOCAL_MEM_SIZE);
    OKKERNEL_ASSERT(output_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);
    okk_bdc_32bit_set_C(output_addr, C, &kernel_shape, &kernel_stride);

    // dim4 kernel_shape_2IC = {.n = (shape.c + 1) / 2, .c = shape.c, .h = 1, .w = 1};
    // dim4 kernel_stride_2IC;
    // okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    dim4 kernel_stride_2IC = {.n=0,.c=0,.h=0,.w=0};
    // OKKERNEL_LOG("kernel shape is [%d,%d,%d,%d]", kernel_shape_2IC.n, kernel_shape_2IC.c, kernel_shape_2IC.h, kernel_shape_2IC.w);
    // OKKERNEL_LOG("kernel stride is [%d,%d,%d,%d]", kernel_stride_2IC.n, kernel_stride_2IC.c, kernel_stride_2IC.h, kernel_stride_2IC.w);

    okk_bdc_conv2d(input_addr, exp_addr, output_addr, NULL, &shape, shape.c, 1, 1, &stride, &kernel_stride_2IC, false, false, NULL, NULL, NULL);

    // div exp(x) by sum
    // OKKERNEL_LOG("stride is [%d,%d,%d,%d]", stride.n, stride.c, stride.h, stride.w);
    // OKKERNEL_LOG("broadcast_stride is [%d,%d,%d,%d]", broadcast_stride.n, broadcast_stride.c, broadcast_stride.h, broadcast_stride.w);
    okk_bdc_div(output_addr, exp_addr, input_addr, &shape, &stride, &stride, &stride);
}

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

    // OKKERNEL_LOG("input addr:%d, exp addr:%d, output addr:%d, total size:%d, mem size:%d", input_addr, exp_addr, output_addr, output_addr + shape.n * stride.n * sizeof(float), LOCAL_MEM_SIZE);

    // OKKERNEL_ASSERT(output_addr + shape.n * stride.n * sizeof(float) <= LOCAL_MEM_SIZE);
    if (output_addr + shape.n * stride.n * sizeof(float) <= LOCAL_MEM_SIZE)
    {
        // common situation
        okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);

        conduct_softmax(input_addr, exp_addr, output_addr, shape, stride);

        okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    }else
    {
        // split data
        if((LOCAL_MEM_SIZE) / (3 * stride.n * sizeof(float)) != 0)
        {
            // OKKERNEL_LOG("data can be split by N");
            int group_size = LOCAL_MEM_SIZE / (3 * stride.n * sizeof(float));
            // OKKERNEL_LOG("group size:%d", group_size);
            int temp_num = 0;
            int tensor_size = param->C * param->H * param->W * sizeof(float);
            while(temp_num < param->N)
            {
                if(temp_num + group_size < param->N)
                    shape.n = group_size;
                else
                    shape.n = param->N - temp_num;

                input_addr = 0;
                exp_addr = input_addr + shape.n * stride.n * sizeof(float);
                output_addr = exp_addr + shape.n * stride.n * sizeof(float);

                int system_offset = temp_num * tensor_size;

                okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + system_offset, &shape, NULL, NULL);

                conduct_softmax(input_addr, exp_addr, output_addr, shape, stride);

                okk_gdma_32bit_cpy_L2S(param->output_addr + system_offset, output_addr, &shape, NULL, NULL);

                temp_num += group_size;
            }
            
        }else
        {
            // TODO dynamic split
            // int size_per_N = LOCAL_MEM_SIZE / shape.n;
            // int wh_perNperC = size_per_N / DIV_UP(shape.c, okk_npu_num());

            // split hw to 2*2 just for tpucontest
            dim4 new_shape = {.n=param->N, .c=param->C, .h=(param->H + 1)/2, .w=(param->W + 1)/2};
            dim4 new_stride;
            okk_128_byte_aligned_stride_for_32bit(&new_stride, 0, &new_shape);

            if(LOCAL_MEM_SIZE / (3*new_shape.n * new_stride.n * sizeof(float)) != 0)
            {
                // split data by HW
                OKKERNEL_LOG("data can be split by HW");
            }else
            {
                // split data by N & HW
                // TODO catch error for too big tensor
                OKKERNEL_LOG("data can be split by N & HW");
                int group_size = LOCAL_MEM_SIZE / (3 * new_stride.n * sizeof(float));
                // OKKERNEL_LOG("group size:%d", group_size);
                int temp_num = 0;
                int tensor_size = new_shape.c * new_shape.h * new_shape.w * sizeof(float);
                for(int i = 0; i < 4; i++)
                {
                    int sys_offset = (i/2)*((param->H + 1)/2) * param->N * param->C * param->W * sizeof(float) + (i%2) * ((param->W + 1)/2) * sizeof(float);
                    dim4 sys_stride = {.n=param->C * param->H * param->W, .c=param->H * param->W, .h=param->W, .w=1};

                }
            }
        }
    }
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
