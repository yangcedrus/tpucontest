#include "okk.h"
#include "math.h"
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
    int N, IC, OC, H, W;
    int kernel_h, kernel_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    unsigned long long output_addr;
    unsigned long long input_addr;
    unsigned long long kernel_addr;
} __attribute__((packed)) param_t;

void conv2d_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int IC_new = (param->IC + 1) / 2;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    dim4 output_stride, input_stride, kernel_stride;

    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    dim4 system_input_stride = {.n=input_shape.c*input_shape.h*input_shape.w, .c=input_shape.h*input_shape.w, .h=input_shape.w, .w=1};
    dim4 system_output_stride = {.n=output_shape.c*output_shape.h*output_shape.w, .c=output_shape.h*output_shape.w, .h=output_shape.w, .w=1};

    // cal max H*W
    int input_channel_per_npu = DIV_UP(param->IC, NPU_NUM);
    int output_channel_per_npu = DIV_UP(param->OC, NPU_NUM);
    long kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    long output_size = output_shape.n * output_stride.n * sizeof(float);
    long max_HW = ((LOCAL_MEM_SIZE - kernel_size) / (input_shape.n * input_channel_per_npu * sizeof(float) + output_shape.n * output_channel_per_npu * sizeof(float))) / 32 * 32;

    OKKERNEL_LOG("rest size:%d\n", LOCAL_MEM_SIZE - kernel_size);
    OKKERNEL_LOG("max HW:%ld\n", max_HW);
    
    // TODO avoid overflow by data overlap
    int patches = DIV_UP(param->H * param->W, max_HW);

    OKKERNEL_LOG("patches:%d\n", patches);

    int row_patches = (int)sqrt(patches);
    int col_patches = DIV_UP(patches, row_patches);

    OKKERNEL_LOG("row patch:%d, col patch:%d\n", row_patches, col_patches);

    int patch_row_iter_size = output_shape.h / row_patches;
    int patch_col_iter_size = output_shape.w / col_patches;

    int patch_row = (patch_row_iter_size - 1) * param->stride_h + kernel_h_ext;
    int patch_col = (patch_col_iter_size - 1) * param->stride_w + kernel_w_ext;

    int temp_output_h = (patch_row + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    int temp_output_w = (patch_col + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 temp_input_shape = {.n = param->N, .c=param->IC, .h=patch_row, .w=patch_col};
    dim4 temp_output_shape = {.n=param->N, .c=param->OC, .h=temp_output_h, .w=temp_output_w};

    dim4 temp_input_stride, temp_output_stride;

    okk_128_byte_aligned_stride_for_32bit(&temp_input_stride, 0, &temp_input_shape);
    okk_128_byte_aligned_stride_for_32bit(&temp_output_stride, 0 , &temp_output_shape);

    local_addr_t input_addr, kernel_addr, output_addr;

    output_addr = 0;
    input_addr = output_addr + temp_output_shape.n * temp_output_stride.n * sizeof(float);
    kernel_addr = input_addr + temp_input_shape.n * temp_input_stride.n * sizeof(float);

    OKKERNEL_ASSERT(output_addr + kernel_shape.n * kernel_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);

    for(int row_p = 0; row_p < row_patches; row_p++)
    {
        int patch_row = (patch_row_iter_size - 1) * param->stride_h + kernel_h_ext;
        if (row_p == 0)
            patch_row -= param->pad_top;
        if (row_p == row_patches - 1)
            patch_row -= param->pad_bottom;
        
        for(int col_p = 0; col_p < col_patches; col_p++)
        {

            OKKERNEL_LOG("patch idx:[%d, %d]\n", row_p, col_p);
            
            int patch_col = (patch_col_iter_size - 1) * param->stride_w + kernel_w_ext;

            // cal offset
            long input_offset = MAX((row_p*patch_row_iter_size*param->stride_h - param->pad_top) * param->W * (long)sizeof(float), 0) + MAX((col_p * patch_col_iter_size * param->stride_w - param->pad_left) * (long)sizeof(float), 0);
            long output_offset = (row_p*patch_row_iter_size) * output_shape.w + col_p * patch_col_iter_size * sizeof(float);

            OKKERNEL_LOG("input offset:%d, output offset:%d\n", input_offset, output_offset);

            // cal temp patch H&W
            if(col_p == 0)
                patch_col -= param->pad_left;
            if(col_p == col_patches - 1)
                patch_col -= param->pad_right;

            Padding padding = {
                .left=(col_p==0)?param->pad_left:0, .right=(col_p==col_patches-1)?param->pad_right:0,
                .top=(row_p==0)?param->pad_top:0, .bottom=(row_p==row_patches-1)?param->pad_bottom:0
            };

            dim4 patch_input_shape = {.n=param->N, .c=param->IC, .h=patch_row, .w=patch_col};
            int patch_output_h = (patch_row + padding.top + padding.bottom - kernel_h_ext) / param->stride_h + 1;
            int patch_output_w = (patch_col + padding.left + padding.right - kernel_w_ext) / param->stride_w + 1;
            dim4 patch_output_shape = {.n=param->N, .c=param->OC, .h=patch_output_h, .w=patch_output_w};

            dim4 patch_input_stride, patch_output_stride;

            okk_128_byte_aligned_stride_for_32bit(&patch_input_stride, 0, &patch_input_shape);
            okk_128_byte_aligned_stride_for_32bit(&patch_output_stride, 0, &patch_output_shape);

            // input_addr = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
            // output_addr = input_addr + patch_input_shape.n * patch_input_stride.n * sizeof(float);

            okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr+input_offset, &patch_input_shape, NO_USE, &system_input_stride);

            dim2 stride = {.h = param->stride_h, .w = param->stride_w};
            dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
            dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
            dim4 kernel_stride_2IC;
            okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

            OKKERNEL_LOG("input shape:[%d, %d, %d, %d], stride: [%d, %d, %d, %d]\n", patch_input_shape.n, patch_input_shape.c, patch_input_shape.h, patch_input_shape.w, patch_input_stride.n, patch_input_stride.c, patch_input_stride.h, patch_input_stride.w);
            OKKERNEL_LOG("output shape:[%d, %d, %d, %d], stride: [%d, %d, %d, %d]\n", patch_output_shape.n ,patch_output_shape.c, patch_output_shape.h, patch_output_shape.w, patch_output_stride.n, patch_output_stride.c, patch_output_stride.h, patch_output_stride.w);
            OKKERNEL_LOG("padding: [%d, %d, %d, %d]\n", padding.left, padding.right, padding.top, padding.bottom);

            // x32 C = {1.0};
            // okk_bdc_32bit_set_C(kernel_addr, C, &kernel_shape, &kernel_stride);
            
            okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, &patch_input_shape, param->OC, param->kernel_h, param->kernel_w, &patch_input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

            okk_gdma_32bit_cpy_L2S(param->output_addr+output_offset, output_addr, &patch_output_shape, &system_output_stride, NO_USE);

            // okk_gdma_32bit_cpy_L2S(param->output_addr+output_offset, input_addr, &patch_input_shape, &system_output_stride, &patch_input_stride);
            // okk_gdma_32bit_cpy_L2S(param->output_addr+output_offset, kernel_addr, &kernel_shape, &system_output_stride, &kernel_stride);

            break;
        }
        break;
    }

    // int seen_input_row = 0, seen_input_col = 0;
    // int now_row = 0, now_col = 0;

    // while(seen_input_row < param->H)
    // {
    //     while(seen_input_col < param->W)
    //     {
    //         break;
    //     }
    //     break;
    // }

    // const int IC_new = (param->IC + 1) / 2;
    // const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    // const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    // const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    // const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    // dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    // dim4 input_shape = {.n = param->N, .c = param->IC, .h = param->H, .w = param->W};
    // dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    // dim4 output_stride, input_stride, kernel_stride;

    // okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    // OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);

    // okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &input_shape, NULL, NULL);
    // okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    // Padding padding = {
    //     .top = param->pad_top, .bottom = param->pad_bottom,
    //     .left = param->pad_left, .right = param->pad_right
    // };
    // dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    // dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    // dim4 kernel_stride_2IC;
    // okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    // okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, &input_shape, param->OC, param->kernel_h, param->kernel_w, &input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

    // okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &output_shape, NULL, NULL);


    // dim4 output_shape = {.n = param->N, .c = param->OC, .h = 2, .w = 2};
    // dim4 input_shape = {.n = param->N, .c = param->IC, .h = 5, .w = 5};
    // dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    // dim4 output_stride, input_stride, kernel_stride;

    // okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    // local_addr_t input_addr, output_addr, kernel_addr;
    // input_addr = 0;
    // output_addr = input_addr + input_shape.n * input_stride.n * sizeof(float);
    // kernel_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);

    // okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    // int row_total_iter = 4, col_total_iter = 4;
    // int row_patch_iter = 2, col_patch_iter = 2;
    // int row_per_patch = 5, col_per_patch = 5;

    // dim4 patch_output_shape = {output_shape.n , output_shape.c, 4, 4};

    // int row_iter = 0, col_iter = 0;
    // int now_row = 0, now_col = 0;

    // while(row_iter < row_total_iter)
    // {
    //     now_row = row_per_patch;
    //     if(row_iter == 0)
    //         now_row = row_per_patch - param->pad_top;
    //     if(row_iter + row_patch_iter >= row_total_iter)
    //         now_row = row_per_patch - param->pad_bottom;
    //     input_shape.h = now_row;
        
    //     col_iter = 0;
    //     while(col_iter < col_total_iter)
    //     {
    //         now_col = col_per_patch;
    //         if(col_iter == 0)
    //             now_col -= param->pad_left;
    //         if(col_iter + col_patch_iter >= col_total_iter)
    //             now_col -= param->pad_right;
    //         input_shape.w = now_col;

    //         long long input_offset = ;
    //         long long output_offset = ;
    //         Padding padding = {.top = 0, .bottom = 0, .left = 0, .right = 0};

    //         if(temp_input_row == 0)
    //             padding.top = param->pad_top;

    //         if(temp_input_col == 0)
    //             padding.left = param->pad_left;

    //         if(temp_input_row + row_per_patch > param->H)
    //             padding.bottom = param->pad_bottom;

    //         if(temp_input_col + col_per_patch > param->W)
    //             padding.right = param->pad_right;

    //         dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    //         dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    //         dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    //         dim4 kernel_stride_2IC;
    //         okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    //         okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, &input_shape, param->OC, param->kernel_h, param->kernel_w, &input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

    //         col_iter = col_patch_iter;
    //     }
    //     row_iter += row_patch_iter;
    // }

    // int start_row = 0, start_col = 0;

    

    // while(start_row < param->H)
    // {
    //     // if(temp_input_row + row_per_patch > param->H)
    //     //     now_row = param->H - temp_input_row;
    //     // else
    //     //     now_row = row_per_patch;

    //     if(temp_input_row == 0 || temp_input_row + )

    //     temp_input_col = 0;
    //     while(temp_input_col < param->W)
    //     {
    //         if(temp_input_col + col_per_patch > param->W)
    //             now_col = param->W - temp_input_col;
    //         else
    //             now_col = col_per_patch;
            
    //         long long input_offset = temp_input_row * param->W * sizeof(float) + temp_input_col * sizeof(float);
    //         long long output_offset = (temp_input_row + param->pad_top) / row_per_patch;
    //         Padding padding = {.top = 0, .bottom = 0, .left = 0, .right = 0};

    //         if(temp_input_row == 0)
    //             padding.top = param->pad_top;

    //         if(temp_input_col == 0)
    //             padding.left = param->pad_left;

    //         if(temp_input_row + row_per_patch > param->H)
    //             padding.bottom = param->pad_bottom;

    //         if(temp_input_col + col_per_patch > param->W)
    //             padding.right = param->pad_right;

    //         dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    //         dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    //         dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    //         dim4 kernel_stride_2IC;
    //         okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    //         okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, &input_shape, param->OC, param->kernel_h, param->kernel_w, &input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

    //         temp_input_col += now_col;
    //     }

    //     temp_input_row += now_row;
    // }

    // // output is 64-byte aligned layout
    // local_addr_t output_addr = 0;
    // okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // // input is 64-byte aligned layout
    // local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    // okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // // kernel is compact layout
    // local_addr_t kernel_addr = input_addr + input_shape.n * input_stride.n * sizeof(float);
    // okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    // // check local memory exceeded
    // // OKKERNEL_LOG()
    // OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);
    // // copy input from global memory to local memory
    // okk_gdma_32bit_cpy_S2L(
    //     input_addr,
    //     param->input_addr,
    //     &input_shape,
    //     NULL,
    //     NULL);
    // // copy kernel from global memory to local memory
    // okk_gdma_32bit_cpy_S2L(
    //     kernel_addr,
    //     param->kernel_addr,
    //     &kernel_shape,
    //     &kernel_stride,
    //     NULL);
    // // conv2d
    // Padding padding = {
    //     .top = param->pad_top, .bottom = param->pad_bottom,
    //     .left = param->pad_left, .right = param->pad_right
    // };
    // dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    // dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // // view the data type of kernel as fp32x2
    // dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    // dim4 kernel_stride_2IC;
    // okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    // okk_bdc_conv2d(
    //     output_addr,
    //     input_addr,
    //     kernel_addr,
    //     NO_USE,
    //     &input_shape,
    //     param->OC,
    //     param->kernel_h,
    //     param->kernel_w,
    //     &input_stride,
    //     &kernel_stride_2IC,
    //     false,
    //     false,
    //     &padding,
    //     &stride,
    //     &dilation);
    // // copy output from local memory to global memory
    // okk_gdma_32bit_cpy_L2S(
    //     param->output_addr,
    //     output_addr,
    //     &output_shape,
    //     NULL,
    //     NULL);
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(conv2d_contest);
