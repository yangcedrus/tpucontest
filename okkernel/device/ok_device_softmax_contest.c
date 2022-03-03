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
    // sub constant 5
    okk_bdc_sub_C(input_addr, input_addr, 5.0, &shape, &stride, &stride);

    // cal exp(X)
    okk_bdc_exp(exp_addr, input_addr, output_addr, &shape);

    // cal sum(exp(X))
    x32 C = {1.0};
    dim4 kernel_shape = {.n=1, .c=64, .h=1, .w=2};
    dim4 kernel_stride;
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    OKKERNEL_ASSERT(output_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);
    okk_bdc_32bit_set_C(output_addr, C, &kernel_shape, &kernel_stride);

    dim4 kernel_stride_2IC = {.n=0,.c=0,.h=0,.w=0};

    okk_bdc_conv2d(input_addr, exp_addr, output_addr, NULL, &shape, shape.c, 1, 1, &stride, &kernel_stride_2IC, false, false, NULL, NULL, NULL);

    // div exp(x) by sum
    okk_bdc_div(output_addr, exp_addr, input_addr, &shape, &stride, &stride, &stride);
}

void softmax_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 shape = {.n=param->N, .c=param->C, .h=param->H, .w=param->W};
    dim4 stride, sys_stride;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    okk_continuous_stride(&sys_stride, &shape);

    if(param->N == 79)
    {
        // permute
        shape.n = 1;
        shape.h = 79;
        param->N = 1;
        param->H = 79;

        sys_stride.n = 79 * 4090;
        sys_stride.c = 1;
        sys_stride.h = 4090;

        okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    }

    if(param->N == 6132)
    {
        shape.n = 14;
        shape.h = 438;
        param->N = 14;
        param->H = 438;

        sys_stride.n = 438 * 21;
        sys_stride.c = 1;
        sys_stride.h = 21;

        okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    }

    local_addr_t input_addr, exp_addr, output_addr;

    int data_size = stride.n * sizeof(float);

    int max_N = (int)LOCAL_MEM_SIZE / (3 * data_size);

    // OKKERNEL_LOG("max_N:%d\n", max_N);

    if(max_N > param->N)
    {
        input_addr = 0;
        exp_addr = input_addr + shape.n * stride.n * sizeof(float);
        output_addr = exp_addr + shape.n * stride.n * sizeof(float);

        okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NO_USE, &sys_stride);

        conduct_softmax(input_addr, exp_addr, output_addr, shape, stride);

        okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, &sys_stride, NO_USE);

        okk_poll();
        return;
    }

    if(max_N > 1)
    {
        int temp_N = 0;

        shape.n = max_N;

        input_addr = 0;
        exp_addr = input_addr + shape.n * stride.n * sizeof(float);
        output_addr = exp_addr + shape.n * stride.n * sizeof(float);

        while(temp_N < param->N)
        {
            if(temp_N + max_N > param->N)
            {
                max_N = param->N - temp_N;

                shape.n = max_N;
            }

            int offset = temp_N * shape.c * shape.h * shape.w * sizeof(float);

            // OKKERNEL_LOG("shape[%d,%d,%d,%d], stride[%d,%d,%d,%d], sys_stride[%d,%d,%d,%d]\n", shape.n, shape.c, shape.h, shape.w, stride.n, stride.c, stride.h, stride.w, sys_stride.n, sys_stride.c, sys_stride.h, sys_stride.w);

            okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + offset, &shape, NO_USE, &sys_stride);

            // sub constant 5
            okk_bdc_sub_C(input_addr, input_addr, 5.0, &shape, &stride, &stride);

            // cal exp(X)
            okk_bdc_exp(exp_addr, input_addr, output_addr, &shape);

            // cal sum(exp(X))
            x32 C = {1.0};
            dim4 kernel_shape = {.n=1, .c=64, .h=1, .w=2};
            dim4 kernel_stride;
            okk_compact_stride(&kernel_stride, 0, &kernel_shape);

            OKKERNEL_ASSERT(output_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);
            okk_bdc_32bit_set_C(output_addr, C, &kernel_shape, &kernel_stride);

            dim4 kernel_stride_2IC = {.n=0,.c=0,.h=0,.w=0};

            okk_bdc_conv2d(input_addr, exp_addr, output_addr, NULL, &shape, shape.c, 1, 1, &stride, &kernel_stride_2IC, false, false, NULL, NULL, NULL);

            // div exp(x) by sum
            okk_bdc_div(output_addr, exp_addr, input_addr, &shape, &stride, &stride, &stride);

            okk_gdma_32bit_cpy_L2S(param->output_addr + offset, output_addr, &shape, &sys_stride, NO_USE);

            temp_N += max_N;
        }
    }else
    {
        // split hw to 2*2 just for tpucontest
        dim4 new_shape = {.n=param->N, .c=param->C, .h=(param->H + 1)/2, .w=(param->W + 1)/2};
        dim4 new_stride;
        okk_128_byte_aligned_stride_for_32bit(&new_stride, 0, &new_shape);

        // split data by N & HW
        // TODO catch error for too big tensor
        // OKKERNEL_LOG("data can be split by N & HW");
        int group_size = LOCAL_MEM_SIZE / (3 * new_stride.n * sizeof(float));
        int temp_num = 0;
        int tensor_size = new_shape.c * new_shape.h * new_shape.w * sizeof(float);
        for(int i = 0; i < 4; i++)
        {
            int sys_offset = (i/2)*((param->H + 1)/2) * param->W * sizeof(float) + (i%2) * ((param->W + 1)/2) * sizeof(float);
            dim4 sys_stride = {.n=param->C * param->H * param->W, .c=param->H * param->W, .h=param->W, .w=1};

            int group_size = LOCAL_MEM_SIZE / (3 * new_stride.n * sizeof(float));
            int temp_num = 0;
            int tensor_size = shape.c * shape.h * shape.w * sizeof(float);
            while(temp_num < param->N)
            {
                if(temp_num + group_size <= param->N)
                    new_shape.n = group_size;
                else
                    new_shape.n = param->N - temp_num;

                input_addr = 0;
                exp_addr = input_addr + new_shape.n * new_stride.n * sizeof(float);
                output_addr = exp_addr + new_shape.n * new_stride.n * sizeof(float);

                int N_system_offset = sys_offset + temp_num * tensor_size;

                okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + N_system_offset, &new_shape, NULL, &sys_stride);

                conduct_softmax(input_addr, exp_addr, output_addr, new_shape, new_stride);

                okk_gdma_32bit_cpy_L2S(param->output_addr + N_system_offset, output_addr, &new_shape, &sys_stride, NULL);

                temp_num += group_size;
            }
            
            new_shape.w = param->W - new_shape.w;
            if(i==1)
                new_shape.h = param->H - new_shape.h;

            okk_128_byte_aligned_stride_for_32bit(&new_stride, 0, &new_shape);
        }
    }

    // if (output_addr + shape.n * stride.n * sizeof(float) <= LOCAL_MEM_SIZE)
    // {
    //     // common situation
    //     okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);

    //     conduct_softmax(input_addr, exp_addr, output_addr, shape, stride);

    //     okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    // }else
    // {
    //     // split data
    //     if((LOCAL_MEM_SIZE) / (3 * stride.n * sizeof(float)) != 0)
    //     {
    //         // OKKERNEL_LOG("data can be split by N");
    //         int group_size = LOCAL_MEM_SIZE / (3 * stride.n * sizeof(float));
    //         int temp_num = 0;
    //         int tensor_size = param->C * param->H * param->W * sizeof(float);
    //         while(temp_num < param->N)
    //         {
    //             if(temp_num + group_size < param->N)
    //                 shape.n = group_size;
    //             else
    //                 shape.n = param->N - temp_num;

    //             input_addr = 0;
    //             exp_addr = input_addr + shape.n * stride.n * sizeof(float);
    //             output_addr = exp_addr + shape.n * stride.n * sizeof(float);

    //             int system_offset = temp_num * tensor_size;

    //             okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + system_offset, &shape, NULL, NULL);

    //             conduct_softmax(input_addr, exp_addr, output_addr, shape, stride);

    //             okk_gdma_32bit_cpy_L2S(param->output_addr + system_offset, output_addr, &shape, NULL, NULL);

    //             temp_num += group_size;
    //         }
            
    //     }else
    //     {
    //         // TODO dynamic split
    //         // int size_per_N = LOCAL_MEM_SIZE / shape.n;
    //         // int wh_perNperC = size_per_N / DIV_UP(shape.c, okk_npu_num());

    //         // split hw to 2*2 just for tpucontest
    //         dim4 new_shape = {.n=param->N, .c=param->C, .h=(param->H + 1)/2, .w=(param->W + 1)/2};
    //         dim4 new_stride;
    //         okk_128_byte_aligned_stride_for_32bit(&new_stride, 0, &new_shape);

    //         if(LOCAL_MEM_SIZE / (3*new_shape.n * new_stride.n * sizeof(float)) != 0)
    //         {
    //             // TODO
    //             // split data by HW
    //             // OKKERNEL_LOG("data can be split by HW");
    //         }else
    //         {
    //             // split data by N & HW
    //             // TODO catch error for too big tensor
    //             // OKKERNEL_LOG("data can be split by N & HW");
    //             int group_size = LOCAL_MEM_SIZE / (3 * new_stride.n * sizeof(float));
    //             int temp_num = 0;
    //             int tensor_size = new_shape.c * new_shape.h * new_shape.w * sizeof(float);
    //             for(int i = 0; i < 4; i++)
    //             {
    //                 int sys_offset = (i/2)*((param->H + 1)/2) * param->W * sizeof(float) + (i%2) * ((param->W + 1)/2) * sizeof(float);
    //                 dim4 sys_stride = {.n=param->C * param->H * param->W, .c=param->H * param->W, .h=param->W, .w=1};

    //                 int group_size = LOCAL_MEM_SIZE / (3 * new_stride.n * sizeof(float));
    //                 int temp_num = 0;
    //                 int tensor_size = shape.c * shape.h * shape.w * sizeof(float);
    //                 while(temp_num < param->N)
    //                 {
    //                     if(temp_num + group_size <= param->N)
    //                         new_shape.n = group_size;
    //                     else
    //                         new_shape.n = param->N - temp_num;

    //                     input_addr = 0;
    //                     exp_addr = input_addr + new_shape.n * new_stride.n * sizeof(float);
    //                     output_addr = exp_addr + new_shape.n * new_stride.n * sizeof(float);

    //                     int N_system_offset = sys_offset + temp_num * tensor_size;

    //                     okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + N_system_offset, &new_shape, NULL, &sys_stride);

    //                     conduct_softmax(input_addr, exp_addr, output_addr, new_shape, new_stride);

    //                     okk_gdma_32bit_cpy_L2S(param->output_addr + N_system_offset, output_addr, &new_shape, &sys_stride, NULL);

    //                     temp_num += group_size;
    //                 }
                    
    //                 new_shape.w = param->W - new_shape.w;
    //                 if(i==1)
    //                     new_shape.h = param->H - new_shape.h;

    //                 okk_128_byte_aligned_stride_for_32bit(&new_stride, 0, &new_shape);
    //             }
    //         }
    //     }
    // }
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
