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

    int max_left_row = 0;
    if ((long int)LOCAL_MEM_SIZE - right_size < 0)
        max_left_row = -1;
    else
        max_left_row = ((long int)LOCAL_MEM_SIZE - right_size) / ((left_stride.n + output_stride.n) * (int)sizeof(float));

    // OKKERNEL_LOG("max_left_row:%d\n", max_left_row);

    // okk_poll();
    // return;

    if(max_left_row > 1 && param->left_cols != 3072)
    {
        max_left_row /= 2;

        local_addr_t left_addr[2], right_addr, output_addr[2];
        left_addr[0] = 0;
        output_addr[0] = left_addr[0] + max_left_row * left_stride.n * sizeof(float);

        left_addr[1] = output_addr[0] + max_left_row * right_stride.n * sizeof(float);
        output_addr[1] = left_addr[1] + max_left_row * left_stride.n * sizeof(float);

        right_addr = output_addr[1] + max_left_row * right_stride.n * sizeof(float);

        okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, param->left_cols, param->right_cols, right_cols_per_channel, param->right_cols);

        int iteration = DIV_UP(param->left_rows, max_left_row);

        // dim4 new_left_shape = {.n=max_left_row, .c=left_shape.c, .h=left_shape.h, .w=left_shape.w};
        // dim4 new_output_shape = {.n=max_left_row, .c=output_shape.c, .h=output_shape.h, .w=output_shape.w};

        unsigned int left_tensor_size_global = max_left_row * param->left_cols * sizeof(float);
        unsigned int output_tensor_size_global = max_left_row * param->right_cols * sizeof(float);

        // dim4 left_shape_last = {.n=param->left_rows - (iteration - 1)*max_left_row, .c=left_shape.c, .h=left_shape.h, .w=left_shape.w};
        // dim4 output_shape_last = {.n=left_shape_last.n, output_shape.c, output_shape.h, output_shape.w};

        int last_left_row = param->left_rows - (iteration - 1) * max_left_row;

        for(int i = 0; i<iteration + 2; i++){
            okk_parallel_start();

            if(i < iteration)
                okk_gdma_32bit_matrix_S2L(left_addr[i%2], param->left_addr + i * left_tensor_size_global, i==iteration-1?last_left_row:max_left_row, param->left_cols, left_cols_per_channel, param->left_cols);
            
            if(i>0 && i<iteration + 1)
                okk_bdc_matmul(output_addr[(i-1)%2], left_addr[(i-1)%2], right_addr, NO_USE, i==iteration?last_left_row:max_left_row, param->left_cols, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

            if(i>1)
                okk_gdma_32bit_matrix_L2S(param->output_addr + (i-2)*output_tensor_size_global, output_addr[i%2], i==iteration+1?last_left_row:max_left_row, param->right_cols, right_cols_per_channel, param->right_cols);
            
            okk_parallel_end();
        }
    }else{
        // may be left row not be largest
        if(param->left_cols > param->right_cols)
        {
            // left cols largest
            // dim4 new_left_shape = {.n=1, .c=left_shape.c, .h=left_shape.h, .w=left_shape.w};
            // dim4 new_right_shape = {.n=right_shape.n, .c=1, .h=1, .w=1};

            // dim4 new_left_stride, new_right_stride;
            // okk_128_byte_aligned_stride_for_32bit(&new_left_stride, 0, &new_left_shape);
            // okk_128_byte_aligned_stride_for_32bit(&new_right_stride, 0, &new_right_shape);

            // local_addr_t left_addr, right_addr, output_addr;
            // left_addr = 0;
            // right_addr = left_addr + new_left_shape.n * new_left_stride.n * sizeof(float);
            // output_addr = right_addr + new_right_shape.n * new_right_stride.n * sizeof(float);

            // OKKERNEL_ASSERT(output_addr + 32*sizeof(float) <= LOCAL_MEM_SIZE);

            // int max_size = LOCAL_MEM_SIZE / (output_addr + 32);

            // OKKERNEL_LOG("max_size:%d\n", max_size);
            int left_col = MIN(1024, param->left_cols);
            if(param->left_cols == 100352 || param->left_cols == 25088 || param->left_cols == 9216 || param->left_cols == 4096)
                left_col = 1;
            if(param->left_cols == 3072)
                left_col = 256;
            int output_size = output_shape.n * output_stride.n * sizeof(float);

            left_cols_per_channel = DIV_UP(left_col, NPU_NUM);

            dim4 new_left_shape = {.n=param->left_rows, .c=DIV_UP(left_col, left_cols_per_channel), .h=1, .w=left_cols_per_channel};
            dim4 new_right_shape = {.n=left_col, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};

            dim4 temp_left_stride;

            okk_128_byte_aligned_stride_for_32bit(&temp_left_stride, 0, &new_left_shape);

            local_addr_t left_addr, right_addr, output_addr, temp_addr[2], err_addr;
            left_addr = 0;
            right_addr = left_addr + new_left_shape.n * temp_left_stride.n * sizeof(float);
            output_addr = right_addr + new_right_shape.n * right_stride.n * sizeof(float);
            err_addr = output_addr + output_size;

            int iteration = DIV_UP(param->left_cols, left_col);

            int last_left_col = param->left_cols - (iteration - 1)*left_col;
            int last_left_col_per_channel = DIV_UP(last_left_col, NPU_NUM);

            int left_skip_size = left_col * sizeof(float);
            int right_skip_size = left_col * param->right_cols * sizeof(float);

            x32 C = {0.0};
            okk_bdc_32bit_set_C(output_addr, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(err_addr, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + output_size, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + 2*output_size, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + 3*output_size, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + 4*output_size, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + 5*output_size, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + 6*output_size, C, &output_shape, &output_stride);
            // okk_bdc_32bit_set_C(output_addr + 7*output_size, C, &output_shape, &output_stride);

            // temp_addr[0] = err_addr + output_size;
            // temp_addr[1] = temp_addr[0] + output_size;

            for(int i = 0; i<iteration; i++)
            {
                okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr + i*left_skip_size, param->left_rows, i==iteration-1?last_left_col:left_col, i==iteration-1?last_left_col_per_channel:left_cols_per_channel, param->left_cols);

                okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + i*right_skip_size, i==iteration-1?last_left_col:left_col, param->right_cols, right_cols_per_channel, param->right_cols);

                okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, param->left_rows, i==iteration-1?last_left_col:left_col, param->right_cols, i==iteration-1?last_left_col_per_channel:left_cols_per_channel, right_cols_per_channel, false, true);

                // okk_bdc_sub(temp_addr[1], temp_addr[0], err_addr, &output_shape, NO_USE, NO_USE, NO_USE);
                // okk_bdc_add(temp_addr[0], temp_addr[1], output_addr, &output_shape, NO_USE, NO_USE, NO_USE);
                // okk_bdc_sub(err_addr, temp_addr[0], temp_addr[1], &output_shape, NO_USE, NO_USE, NO_USE);
                // okk_bdc_sub(temp_addr[1], err_addr, output_addr, &output_shape, NO_USE, NO_USE, NO_USE);

                // okk_bdc_32bit_cpy(output_addr, temp_addr[0], &output_shape, NO_USE, NO_USE);
                // okk_bdc_32bit_cpy(err_addr, temp_addr[1], &output_shape, NO_USE, NO_USE);
            }

            // okk_bdc_add(output_addr, output_addr, output_addr + output_size, &output_shape, NO_USE, NO_USE, NO_USE);
            // okk_bdc_add(output_addr + 2*output_size, output_addr + 2*output_size, output_addr + 3*output_size, &output_shape, NO_USE, NO_USE, NO_USE);
            // okk_bdc_add(output_addr + 4*output_size, output_addr + 4*output_size, output_addr + 5*output_size, &output_shape, NO_USE, NO_USE, NO_USE);
            // okk_bdc_add(output_addr + 6*output_size, output_addr + 6*output_size, output_addr + 7*output_size, &output_shape, NO_USE, NO_USE, NO_USE);
            // okk_bdc_add(output_addr, output_addr, output_addr + 2*output_size, &output_shape, NO_USE, NO_USE, NO_USE);
            // okk_bdc_add(output_addr + 4*output_size, output_addr + 4*output_size, output_addr + 6*output_size, &output_shape, NO_USE, NO_USE, NO_USE);
            // okk_bdc_add(output_addr, output_addr, output_addr + 4*output_size, &output_shape, NO_USE, NO_USE, NO_USE);

            okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows, param->right_cols, right_cols_per_channel, param->right_cols);

            // int max_output = (LOCAL_MEM_SIZE - output_addr) / (output_shape.n * output_stride.n * sizeof(float));

            // OKKERNEL_LOG("max_output_num:%d\n", max_output);

            // OKKERNEL_ASSERT(output_addr + output_shape.n * output_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);
        }else
        {
            // right cols largest
            int right_col = 4096;

            int iteration = DIV_UP(param->right_cols, right_col);

            // OKKERNEL_LOG("iteration:%d\n", iteration);
            
            right_cols_per_channel = DIV_UP(right_col, NPU_NUM);
            dim4 new_right_shape = {.n=param->left_cols, .c=DIV_UP(right_col, right_cols_per_channel), .h=1, .w=right_cols_per_channel};
            dim4 new_output_shape = {.n=param->left_rows, .c=new_right_shape.c, .h=1, .w=new_right_shape.w};

            dim4 new_right_stride, new_output_stride;
            okk_128_byte_aligned_stride_for_32bit(&new_right_stride, 0, &new_right_shape);
            okk_128_byte_aligned_stride_for_32bit(&new_output_stride, 0, &new_output_shape);

            // OKKERNEL_LOG("new_right_shape:[%d,%d,%d,%d], [%d,%d,%d,%d]\n", new_right_shape.n, new_right_shape.c, new_right_shape.h, new_right_shape.w, new_right_stride.n, new_right_stride.c, new_right_stride.h, new_right_stride.w);
            //  OKKERNEL_LOG("new_output_shape:[%d,%d,%d,%d], [%d,%d,%d,%d]\n", new_output_shape.n, new_output_shape.c, new_output_shape.h, new_output_shape.w, new_output_stride.n, new_output_stride.c, new_output_stride.h, new_output_stride.w);

            local_addr_t left_addr, right_addr, output_addr;
            left_addr = 0;
            right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
            output_addr = right_addr + new_right_shape.n * new_right_stride.n * sizeof(float);

            int right_skip_global = right_col * sizeof(float);
            int last_right_col = param->right_cols - (iteration - 1) * right_col;
            int last_right_cols_per_channel = DIV_UP(last_right_col, NPU_NUM);

            okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr, param->left_rows, param->left_cols, left_cols_per_channel, param->left_cols);

            for(int i = 0; i< iteration; i++)
            {
                okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + i * right_skip_global, param->left_cols, i==iteration-1?last_right_col:right_col, i==iteration-1?last_right_cols_per_channel:right_cols_per_channel, param->right_cols);

                okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, param->left_rows, param->left_cols, i==iteration-1?last_right_col:right_col, left_cols_per_channel, i==iteration-1?last_right_cols_per_channel:right_cols_per_channel, false, false);

                okk_gdma_32bit_matrix_L2S(param->output_addr + i * right_skip_global, output_addr, param->left_rows, i==iteration-1?last_right_col:right_col, i==iteration-1?last_right_cols_per_channel:right_cols_per_channel, param->right_cols);
            }
        }
    }
    //     local_addr_t left_addr, right_addr, output_addr;
    //     left_addr = 0;

    //     int max_left_col = MIN(1024, param->left_cols);
    //     int max_right_col = MIN(2048, param->right_cols);

    //     left_cols_per_channel = DIV_UP(max_left_col, NPU_NUM);
    //     right_cols_per_channel = DIV_UP(max_right_col, NPU_NUM);

    //     // adjust shape
    //     left_shape.c = DIV_UP(max_left_col, left_cols_per_channel);
    //     left_shape.w = left_cols_per_channel;
    //     right_shape.n = max_left_col;
    //     right_shape.c = DIV_UP(max_right_col, right_cols_per_channel);
    //     right_shape.w = right_cols_per_channel;
    //     output_shape.c = right_shape.c;
    //     output_shape.w = right_cols_per_channel;

    //     // adjust stride
    //     okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    //     okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    //     okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    //     // cal max_N
    //     // left_size = left_shape.n * left_stride.n * sizeof(float);
    //     right_size = right_shape.n * right_stride.n * sizeof(float);
    //     // output_size = output_shape.n * output_stride.n * sizeof(float);

    //     // OKKERNEL_LOG("LOCAL:%d, right size:%d", LOCAL_MEM_SIZE, right_size);

    //     max_left_row = ((long int)LOCAL_MEM_SIZE - right_size) / ((left_stride.n + output_stride.n) * (int)sizeof(float));

    //     // OKKERNEL_LOG("max left row:%d\n", max_left_row);

    //     max_left_row = MIN(max_left_row, param->left_rows);

    //     left_shape.n = max_left_row;
    //     output_shape.n = max_left_row;

    //     right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
    //     output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

    //     OKKERNEL_ASSERT(output_addr + output_shape.n * output_stride.n *sizeof(float) <= LOCAL_MEM_SIZE);

    //     int temp_left_row = 0;

    //     while(temp_left_row < param->left_rows)
    //     {
    //         // OKKERNEL_LOG("temp left row:%d\n", temp_left_row);
    //         if(temp_left_row + max_left_row > param->left_rows)
    //         {
                
    //             max_left_row = param->left_rows - temp_left_row;

    //             // adjust shape, left row-> left row, output row
    //             left_shape.n = max_left_row;
    //             output_shape.n = max_left_row;
    //         }

    //         int temp_right_col = 0;
    //         while(temp_right_col < param->right_cols)
    //         {
    //             // OKKERNEL_LOG("temp right col:%d\n", temp_right_col);
    //             if(temp_right_col + max_right_col > param->right_cols)
    //             {
    //                 max_right_col = param->right_cols - temp_right_col;

    //                 // adjust shape, right col->rigth col, output col
    //                 right_cols_per_channel = DIV_UP(max_right_col, NPU_NUM);
    //                 right_shape.c = DIV_UP(max_right_col, right_cols_per_channel);
    //                 right_shape.w = right_cols_per_channel;
    //                 output_shape.c = right_shape.c;
    //                 output_shape.w = right_cols_per_channel;

    //                 // adjust stride
    //                 okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    //                 okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    //             }

    //             int temp_left_col = 0;

    //             x32 C = {0.0};
    //             // OKKERNEL_LOG("before set C\n");
    //             // OKKERNEL_LOG("output addr:%d shape[%d,%d,%d,%d], stride[%d,%d,%d,%d]\n", output_addr, output_shape.n, output_shape.c, output_shape.h, output_shape.w, output_stride.n, output_stride.c, output_stride.h, output_stride.w);
    //             okk_bdc_32bit_set_C(output_addr, C, &output_shape, &output_stride);

    //             int output_offset = (temp_left_row * param->right_cols + temp_right_col) * sizeof(float);

    //             // OKKERNEL_LOG("temp left col:%d\n", temp_left_col);

    //             while(temp_left_col < param->left_cols)
    //             {
    //                 // OKKERNEL_LOG("temp left col:%d\n", temp_left_col);
    //                 if(temp_left_col + max_left_col > param->left_cols)
    //                 {
    //                     max_left_col = param->left_cols - temp_left_col;

    //                     // adjust shape
    //                     left_cols_per_channel = DIV_UP(max_left_col, NPU_NUM);
    //                     left_shape.c = DIV_UP(max_left_col, left_cols_per_channel);
    //                     left_shape.w = left_cols_per_channel;
    //                     right_shape.n = max_left_col;

    //                     // adjust stride
    //                     okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    //                 }

    //                 int left_offset = (temp_left_row * param->left_cols + temp_left_col) * sizeof(float);
    //                 int right_offset = (temp_left_col * param->right_cols + temp_right_col) * sizeof(float);

    //                 okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr + left_offset, left_shape.n, max_left_col, left_cols_per_channel, param->left_cols);
    //                 okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + right_offset, right_shape.n, max_right_col, right_cols_per_channel, param->right_cols);

    //                 okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, left_shape.n, right_shape.n, max_right_col, left_cols_per_channel, right_cols_per_channel, false, true);

    //                 okk_poll();
    //                 okk_initialize();
    //                 temp_left_col += max_left_col;
    //             }
    //             okk_gdma_32bit_matrix_L2S(param->output_addr + output_offset, output_addr, output_shape.n, max_right_col, right_cols_per_channel, param->right_cols);
                

    //             temp_right_col += max_right_col;
    //         }
    //         temp_left_row += max_left_row;
    //     }
    // }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(matmul_contest);
