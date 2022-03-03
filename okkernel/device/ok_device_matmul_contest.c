#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
// #define LOCAL_MEM_SIZE 800
#define NPU_NUM okk_npu_num()
#define NO_USE 0
typedef struct {
    int left_rows, left_cols, right_cols;
    unsigned long long output_addr;
    unsigned long long left_addr;
    unsigned long long right_addr;
} __attribute__((packed)) param_t;

void matmul_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);
    
    dim4 left_shape = {.n=param->left_rows, .c=DIV_UP(param->left_cols, left_cols_per_channel), .h=1, .w=left_cols_per_channel};
    dim4 right_shape = {.n=param->left_cols, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};
    dim4 output_shape = {.n=param->left_rows, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};

    dim4 left_stride, right_stride, output_stride;

    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    // int left_size = left_shape.n * left_stride.n * sizeof(float);
    int right_size = right_shape.n * right_stride.n * sizeof(float);
    // int output_size = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t left_addr, right_addr, output_addr;
    left_addr = 0;

    int max_left_row = 0;
    if ((long int)LOCAL_MEM_SIZE - right_size < 0)
        max_left_row = -1;
    else
        max_left_row = ((long int)LOCAL_MEM_SIZE - right_size) / ((left_stride.n + output_stride.n) * (int)sizeof(float));

    // OKKERNEL_LOG("max_N:%d\n", max_N);

    if(max_left_row >= 2)
    {
        int temp_left_row = 0;

        left_shape.n = max_left_row;
        output_shape.n = max_left_row;

        right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
        output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

        okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, param->left_cols, param->right_cols, right_cols_per_channel, param->right_cols);

        while(temp_left_row < param->left_rows)
        {
            if(temp_left_row + max_left_row > param->left_rows)
            {
                max_left_row = param->left_rows - temp_left_row;

                left_shape.n = max_left_row;
                output_shape.n = max_left_row;
            }

            int sys_left_offset = temp_left_row * param->left_cols * sizeof(float);
            int sys_output_offset = temp_left_row * param->right_cols * sizeof(float);

            okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr + sys_left_offset, max_left_row, param->left_cols, left_cols_per_channel, param->left_cols);

            okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, max_left_row, param->left_cols, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

            okk_gdma_32bit_matrix_L2S(param->output_addr + sys_output_offset, output_addr, max_left_row, param->right_cols, right_cols_per_channel, param->right_cols);

            temp_left_row += max_left_row;
        }
    }else{
        int max_left_col = MIN(4096, param->left_cols);
        int max_right_col = MIN(2048, param->right_cols);

        left_cols_per_channel = DIV_UP(max_left_col, NPU_NUM);
        right_cols_per_channel = DIV_UP(max_right_col, NPU_NUM);

        // adjust shape
        left_shape.c = DIV_UP(max_left_col, left_cols_per_channel);
        left_shape.w = left_cols_per_channel;
        right_shape.n = max_left_col;
        right_shape.c = DIV_UP(max_right_col, right_cols_per_channel);
        right_shape.w = right_cols_per_channel;
        output_shape.c = right_shape.c;
        output_shape.w = right_cols_per_channel;

        // adjust stride
        okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
        okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
        okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

        // cal max_N
        // left_size = left_shape.n * left_stride.n * sizeof(float);
        right_size = right_shape.n * right_stride.n * sizeof(float);
        // output_size = output_shape.n * output_stride.n * sizeof(float);

        // OKKERNEL_LOG("LOCAL:%d, right size:%d", LOCAL_MEM_SIZE, right_size);

        max_left_row = ((long int)LOCAL_MEM_SIZE - right_size) / ((left_stride.n + output_stride.n) * (int)sizeof(float));

        // OKKERNEL_LOG("max left row:%d\n", max_left_row);

        max_left_row = MIN(max_left_row, param->left_rows);

        left_shape.n = max_left_row;
        output_shape.n = max_left_row;

        right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
        output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

        OKKERNEL_ASSERT(output_addr + output_shape.n * output_stride.n *sizeof(float) <= LOCAL_MEM_SIZE);

        int temp_left_row = 0;

        while(temp_left_row < param->left_rows)
        {
            // OKKERNEL_LOG("temp left row:%d\n", temp_left_row);
            if(temp_left_row + max_left_row > param->left_rows)
            {
                
                max_left_row = param->left_rows - temp_left_row;

                // adjust shape, left row-> left row, output row
                left_shape.n = max_left_row;
                output_shape.n = max_left_row;
            }

            int temp_right_col = 0;
            while(temp_right_col < param->right_cols)
            {
                // OKKERNEL_LOG("temp right col:%d\n", temp_right_col);
                if(temp_right_col + max_right_col > param->right_cols)
                {
                    max_right_col = param->right_cols - temp_right_col;

                    // adjust shape, right col->rigth col, output col
                    right_cols_per_channel = DIV_UP(max_right_col, NPU_NUM);
                    right_shape.c = DIV_UP(max_right_col, right_cols_per_channel);
                    right_shape.w = right_cols_per_channel;
                    output_shape.c = right_shape.c;
                    output_shape.w = right_cols_per_channel;

                    // adjust stride
                    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
                    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
                }

                int temp_left_col = 0;

                x32 C = {0.0};
                // OKKERNEL_LOG("before set C\n");
                // OKKERNEL_LOG("output addr:%d shape[%d,%d,%d,%d], stride[%d,%d,%d,%d]\n", output_addr, output_shape.n, output_shape.c, output_shape.h, output_shape.w, output_stride.n, output_stride.c, output_stride.h, output_stride.w);
                okk_bdc_32bit_set_C(output_addr, C, &output_shape, &output_stride);

                int output_offset = (temp_left_row * param->right_cols + temp_right_col) * sizeof(float);

                // OKKERNEL_LOG("temp left col:%d\n", temp_left_col);

                while(temp_left_col < param->left_cols)
                {
                    // OKKERNEL_LOG("temp left col:%d\n", temp_left_col);
                    if(temp_left_col + max_left_col > param->left_cols)
                    {
                        max_left_col = param->left_cols - temp_left_col;

                        // adjust shape
                        left_cols_per_channel = DIV_UP(max_left_col, NPU_NUM);
                        left_shape.c = DIV_UP(max_left_col, left_cols_per_channel);
                        left_shape.w = left_cols_per_channel;
                        right_shape.n = max_left_col;

                        // adjust stride
                        okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
                    }

                    int left_offset = (temp_left_row * param->left_cols + temp_left_col) * sizeof(float);
                    int right_offset = (temp_left_col * param->right_cols + temp_right_col) * sizeof(float);

                    okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr + left_offset, left_shape.n, max_left_col, left_cols_per_channel, param->left_cols);
                    okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + right_offset, right_shape.n, max_right_col, right_cols_per_channel, param->right_cols);

                    okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, left_shape.n, right_shape.n, max_right_col, left_cols_per_channel, right_cols_per_channel, false, true);

                    okk_poll();
                    okk_initialize();
                    temp_left_col += max_left_col;
                }
                okk_gdma_32bit_matrix_L2S(param->output_addr + output_offset, output_addr, output_shape.n, max_right_col, right_cols_per_channel, param->right_cols);
                

                temp_right_col += max_right_col;
            }
            temp_left_row += max_left_row;
        }
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(matmul_contest);
