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

    // global_addr_t left_addr, right_addr, output_addr,  error_addr, temp_sum_addr, temp_addr;

    // int mid_length = 1000;

    // int temp_left_cols_per_channel = DIV_UP(mid_length, NPU_NUM);

    // left_shape.c = DIV_UP(mid_length, temp_left_cols_per_channel);
    // left_shape.w = temp_left_cols_per_channel;

    // right_shape.n = mid_length;
    // output_shape.n = mid_length;

    // okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    // left_addr = 0;
    // right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
    // output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

    // OKKERNEL_ASSERT(output_addr + output_shape.n * output_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

    // error_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    // temp_sum_addr = error_addr + output_shape.n * output_stride.n * sizeof(float);
    // temp_addr = temp_sum_addr + output_shape.n * output_stride.n * sizeof(float);

    // OKKERNEL_ASSERT(temp_addr + output_shape.n * output_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

    // int temp_mid = 0;

    // x32 C = {0.0};
    // // okk_bdc_32bit_set_C(error_addr, C, &output_shape, &output_stride);
    // okk_bdc_32bit_set_C(output_addr, C, &output_shape, &output_stride);

    // while(temp_mid < param->left_cols)
    // {
    //     // OKKERNEL_LOG("temp_mid:%d\n", temp_mid);
    //     if(temp_mid + mid_length > param->left_cols){
    //         mid_length = param->left_cols - temp_mid;
    //         temp_left_cols_per_channel = DIV_UP(mid_length, NPU_NUM);

    //         left_shape.c = DIV_UP(mid_length, temp_left_cols_per_channel);
    //         left_shape.w = temp_left_cols_per_channel;

    //         right_shape.n = mid_length;
    //     }

    //     long sys_left_offset = temp_mid * sizeof(float);
    //     long sys_right_offset = temp_mid * param->right_cols * sizeof(float);
    //     long sys_otuput_offset = 0;

    //     // okk_gdma_32bit_cpy_S2L(left_addr, param->left_addr + sys_left_offset, &left_shape, NO_USE, sys_left_stride);
    //     okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr + sys_left_offset, param->left_rows, mid_length, temp_left_cols_per_channel, param->left_cols);
    //     okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + sys_right_offset, mid_length, param->right_cols, right_cols_per_channel, param->right_cols);

    //     okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, param->left_rows, mid_length, param->right_cols, temp_left_cols_per_channel, right_cols_per_channel, false, true);

    //     // 误差消除累加
    //     // // temp = temp_sum - err_of_previous_step
    //     // okk_bdc_sub(temp_addr, temp_sum_addr, error_addr, &output_shape, &output_stride, &output_stride, &output_stride);

    //     // // temp_sum =  output + temp + err_of_this_step
    //     // okk_bdc_add(temp_sum_addr, output_addr, temp_addr, &output_shape, &output_stride, &output_stride, &output_stride);

    //     // // err = temp_sum - output
    //     // okk_bdc_sub(error_addr, temp_sum_addr, output_addr, &output_shape, &output_stride, &output_stride, &output_stride);

    //     // // output = temp_sum.copy()
    //     // okk_bdc_32bit_cpy(output_addr, temp_sum_addr, &output_shape, &output_stride, &output_stride);

    //     // // temp_sum = err - temp
    //     // okk_bdc_sub(temp_sum_addr, error_addr, temp_addr, &output_shape, &output_stride, &output_stride, &output_stride);
    //     // okk_bdc_32bit_cpy(error_addr, temp_sum_addr, &output_shape, &output_stride, & output_stride);

    //     temp_mid += mid_length;

    // }
    // okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows, param->right_cols, right_cols_per_channel, param->right_cols);

    
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(matmul_contest);


// void matmul_contest(const void *args) {
//     okk_initialize();
//     param_t *param = (param_t *)args;
    
//     // N should be > 2, C shoule be 64*n, H*W shoule be 16*2
//     // TODO mutli loop

//     //TODO 3 case split
//     // left row: gdma split left row
//     // right col: gdma split right col
//     // left col: 
//     int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
//     int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);

//     dim4 left_shape = {.n=param->left_rows, .c=DIV_UP(param->left_cols, left_cols_per_channel), .h=1, .w=left_cols_per_channel};
//     dim4 right_shape = {.n=param->left_cols, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};
//     dim4 output_shape = {.n=param->left_rows, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};

//     dim4 left_stride, right_stride, output_stride;

//     okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
//     okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
//     okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

//     int left_size = left_shape.n * left_stride.n * sizeof(float);
//     int right_size = right_shape.n * right_stride.n * sizeof(float);
//     int output_size = output_shape.n * output_stride.n * sizeof(float);

//     // OKKERNEL_LOG("left: [%d, %d, %d, %d], stride: [%d, %d, %d, %d]", left_shape.n, left_shape.c, left_shape.h, left_shape.w, left_stride.n, left_stride.c, left_stride.h, left_stride.w);
//     // OKKERNEL_LOG("right: [%d, %d, %d, %d], stride: [%d, %d, %d, %d]", right_shape.n, right_shape.c, right_shape.h, right_shape.w, right_stride.n, right_stride.c, right_stride.h, right_stride.w);
//     // OKKERNEL_LOG("output: [%d, %d, %d, %d], stride: [%d, %d, %d, %d]", output_shape.n, output_shape.c, output_shape.h, output_shape.w, output_stride.n, output_stride.c, output_stride.h, output_stride.w);
//     // OKKERNEL_LOG("left size:%d, right size:%d, output size:%d", left_size, right_size, output_size);

//     // assume n*k, k*m
//     int nkm = -1;
//     if(param->left_rows >= param->left_cols)
//     {
//         if(param->left_rows >= param->right_cols)
//             nkm = 0;
//         else
//             nkm = 2;
//     }else
//     {
//         if(param->left_cols >= param->right_cols)
//             nkm = 1;
//         else
//             nkm = 2;
//     }

//     if(nkm == 0)
//     {
//         // split left matrix rows
//         // OKKERNEL_LOG("left matrix row biggest");

//         int max_left_rows = (LOCAL_MEM_SIZE - right_size) / ((left_stride.n + output_stride.n) * sizeof(float));
//         int temp_row = 0;

//         // OKKERNEL_LOG("left matrix max row:%d", max_left_rows);

//         local_addr_t left_addr, right_addr, output_addr;
//         left_addr = 0;
//         right_addr = left_addr + max_left_rows * left_stride.n * sizeof(float);
//         output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

//         // cpy rigth matrix to local memory
//         okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, param->left_cols, param->right_cols, right_cols_per_channel, param->right_cols);

//         // TODO catch too big matrix
//         OKKERNEL_ASSERT(output_addr + max_left_rows * output_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

//         while(temp_row < param->left_rows)
//         {
//             long long left_offset = temp_row * param->left_cols * sizeof(float);
//             long long output_offset = temp_row * param->right_cols * sizeof(float);
//             if(temp_row + max_left_rows < param->left_rows)
//                 left_shape.n = max_left_rows;
//             else
//                 left_shape.n = param->left_rows - temp_row;

//             okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr+left_offset, left_shape.n, param->left_cols, left_cols_per_channel, param->left_cols);

//             okk_bdc_matmul(output_addr, left_addr, right_addr, NULL, left_shape.n, param->left_cols, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

//             okk_gdma_32bit_matrix_L2S(param->output_addr+output_offset, output_addr, left_shape.n, param->right_cols, right_cols_per_channel, param->right_cols);

//             temp_row += max_left_rows;
//         }
//     }else
//     {
//         if(nkm == 1)
//         {
//             // left matrix col biggest
//             // split left matrix col

//             // OKKERNEL_LOG("right matrix row biggest");

//             // OKKERNEL_LOG("left size:%d, output size:%d", left_size, output_size);

//             int max_right_rows = (LOCAL_MEM_SIZE - left_size - output_size) / (int)(right_stride.n * sizeof(float));
//             int temp_row = 0;

//             // OKKERNEL_LOG("right matrix max row:%d", max_right_rows);

//             local_addr_t left_addr, right_addr, output_addr;
//             left_addr = 0;
//             right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
//             output_addr = right_addr + max_right_rows * right_stride.n * sizeof(float);
            
//             x32 C = {0.0};
//             okk_bdc_32bit_set_C(output_addr, C, &output_shape, &output_stride);

//             // OKKERNEL_LOG("reset 0.0");

//             // TODO catch too big matrix
//             OKKERNEL_ASSERT(output_addr + left_shape.n * output_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

//             while(temp_row < param->left_cols)
//             {

//                 // OKKERNEL_LOG("temp right row:%d", temp_row);
//                 long long left_offset = temp_row * sizeof(float);
//                 long long right_offset = temp_row * param->right_cols * sizeof(float);

//                 if(temp_row + max_right_rows < param->left_cols)
//                     right_shape.n = max_right_rows;
//                 else
//                     right_shape.n = param->left_cols - temp_row;

//                 left_cols_per_channel = DIV_UP(right_shape.n, NPU_NUM);
//                 left_shape.c = DIV_UP(right_shape.n, left_cols_per_channel);
//                 left_shape.w = left_cols_per_channel;

//                 okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);

//                 // OKKERNEL_LOG("left: [%d, %d, %d, %d], stride: [%d, %d, %d, %d]", left_shape.n, left_shape.c, left_shape.h, left_shape.w, left_stride.n, left_stride.c, left_stride.h, left_stride.w);

//                 okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr+left_offset, param->left_rows, right_shape.n, left_cols_per_channel, param->left_cols);
//                 okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr+right_offset, right_shape.n, param->right_cols, right_cols_per_channel, param->right_cols);

//                 // okk_bdc_mul_C(left_addr, left_addr, 100, &left_shape, &left_stride, &left_stride);
//                 // okk_bdc_mul_C(right_addr, right_addr, 100, &right_shape, &right_stride, &right_stride);

//                 // okk_gdma_32bit_cpy_S2L(left_addr, param->left_addr+left_offset, &left_shape, &left_stride, NULL);
//                 // // OKKERNEL_LOG("finish cpy left matrix part");
//                 // okk_gdma_32bit_cpy_S2L(right_addr, param->right_addr+right_offset, &right_shape, &right_stride, NULL);
//                 // OKKERNEL_LOG("finish cpy right matrix part");
                
//                 // if(temp_row != 0)
//                 okk_bdc_matmul(output_addr, left_addr, right_addr, NULL, left_shape.n, right_shape.n, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, true);

//                 temp_row += max_right_rows;

//                 // break;
//             }

//             // okk_bdc_div_C(output_addr, output_addr, 10000, &output_shape, &output_stride, &output_stride);

//             // cpy output to system memory
//             okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows, param->right_cols, right_cols_per_channel, param->right_cols);

//         }else
//         {
//             // output matrix biggest
//             // split both rows and cols

//             // OKKERNEL_LOG("right matrix row biggest");

//             int max_right_cols = 0; // not max, max | 1024
//             // OKKERNEL_LOG("local mem size:%d, right rows:%d, div:%d", LOCAL_MEM_SIZE, param->left_cols, (LOCAL_MEM_SIZE / param->left_cols));
//             for(int j=1; j <= (LOCAL_MEM_SIZE / param->left_cols) / 16; j++)
//             {
//                 right_shape.c = 64;
//                 right_shape.w = 16*j;
//                 output_shape.c = 64;
//                 output_shape.w = 16*j;

//                 okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
//                 okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

//                 // OKKERNEL_LOG("temp size:%d, total size:%d", left_size + right_shape.n * right_stride.n * sizeof(float) + output_shape.n * output_stride.n * sizeof(float), LOCAL_MEM_SIZE);

//                 if(left_size + right_shape.n * right_stride.n * sizeof(float) + output_shape.n * output_stride.n * sizeof(float) < LOCAL_MEM_SIZE && (j-1)*1024 < param->right_cols)
//                 {
//                     // OKKERNEL_LOG("final size:%d, total size:", left_size + right_shape.n * right_stride.n * sizeof(float) + output_shape.n * output_stride.n * sizeof(float), LOCAL_MEM_SIZE);
//                     max_right_cols = j * 1024;
//                 }
//                 else
//                 {
//                     j--;
//                     right_shape.c = 64;
//                     right_shape.w = 16*j;
//                     output_shape.c = 64;
//                     output_shape.w = 16*j;

//                     okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
//                     okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

//                     break;
//                 }
//             }

//             // OKKERNEL_LOG("max right col:%d", max_right_cols);

//             if(max_right_cols == 0)
//                 return;
            
//             int temp_col = 0;

//             local_addr_t left_addr, right_addr, output_addr;
//             left_addr = 0;
//             right_addr = left_addr + left_size;
//             output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);


//             // OKKERNEL_LOG("left addr:%d, right addr:%d, output addr:%d, final size:%d", left_addr, right_addr, output_addr, );
//             OKKERNEL_ASSERT(output_addr + output_shape.n * output_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

//             //cpy left matrix
//             okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr, param->left_rows, param->left_cols, left_cols_per_channel, param->left_cols);

//             while(temp_col < param->right_cols)
//             {
//                 long long right_offset = temp_col * sizeof(float);

//                 // OKKERNEL_LOG("temp col:%d, right offset:%llu", temp_col, right_offset);

//                 if(temp_col + max_right_cols > param->right_cols)
//                 {
//                     max_right_cols = param->right_cols - temp_col;
//                     right_cols_per_channel = DIV_UP(max_right_cols, 64);
//                     right_shape.c = DIV_UP(max_right_cols, right_cols_per_channel);
//                     right_shape.w = right_cols_per_channel;

//                     output_shape.c = right_shape.c;
//                     output_shape.w = right_shape.w;
//                 }else
//                 {
//                     right_cols_per_channel = DIV_UP(max_right_cols, 64);
//                     right_shape.c = DIV_UP(max_right_cols, right_cols_per_channel);
//                     right_shape.w = right_cols_per_channel;

//                     output_shape.c = right_shape.c;
//                     output_shape.w = right_shape.w;
//                 }

//                 // OKKERNEL_LOG("max right cols:%d", max_right_cols);
//                 // OKKERNEL_LOG("right: [%d, %d, %d, %d], stride: [%d, %d, %d, %d]", right_shape.n, right_shape.c, right_shape.h, right_shape.w, right_stride.n, right_stride.c, right_stride.h, right_stride.w);
//                 // OKKERNEL_LOG("output: [%d, %d, %d, %d], stride: [%d, %d, %d, %d]", output_shape.n, output_shape.c, output_shape.h, output_shape.w, output_stride.n, output_stride.c, output_stride.h, output_stride.w);

//                 okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + right_offset, param->left_cols, max_right_cols, right_cols_per_channel, param->right_cols);

//                 okk_bdc_matmul(output_addr, left_addr, right_addr, NULL, param->left_rows, param->left_cols, max_right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

//                 okk_gdma_32bit_matrix_L2S(param->output_addr+right_offset, output_addr, param->left_rows, max_right_cols, right_cols_per_channel, param->right_cols);

//                 temp_col += max_right_cols;
//             }
//         }
//     }

//     // local_addr_t left_addr, right_addr, output_addr;
//     // left_addr = 0;
//     // right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
//     // output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

//     // OKKERNEL_LOG("rigth addr:%d, output addr:%d, final addr:%d, total size:%d", right_addr, output_addr, output_addr + left_shape.n * right_stride.n * sizeof(float), LOCAL_MEM_SIZE);

//     // OKKERNEL_ASSERT(output_addr + left_shape.n * right_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

//     // okk_gdma_32bit_cpy_S2L(left_addr, param->left_addr, &left_shape, NULL, NULL);
//     // okk_gdma_32bit_cpy_S2L(right_addr, param->right_addr, &right_shape, NULL, NULL);

//     // okk_bdc_matmul(output_addr, left_addr, right_addr, NULL, param->left_rows, param->left_cols, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

//     // okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &output_shape, NULL, NULL);

//     okk_poll();
// }
// OKKERNEL_FUNC_REGISTER(matmul_contest);
