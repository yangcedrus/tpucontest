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

// static inline void conv2d_reference(float *output, const float *input, const float *kernel, const param_t &param) {
//     const int kernel_h_ext = (param.kernel_h - 1) * param.dilation_h + 1;
//     const int kernel_w_ext = (param.kernel_w - 1) * param.dilation_w + 1;
//     const int output_h = (param.H + param.pad_top + param.pad_bottom - kernel_h_ext) / param.stride_h + 1;
//     const int output_w = (param.W + param.pad_left + param.pad_right - kernel_w_ext) / param.stride_w + 1;
//     for (int n = 0; n < param.N; ++n) {
//         for (int oc = 0; oc < param.OC; ++oc) {
//             for (int oh = 0; oh < output_h; ++oh) {
//                 for (int ow = 0; ow < output_w; ++ow) {
//                     float acc = 0.f;
//                     for (int kh = 0; kh < param.kernel_h; ++kh) {
//                         for (int kw = 0; kw < param.kernel_w; ++kw) {
//                             int ih = oh * param.stride_h + kh * param.dilation_h - param.pad_top;
//                             int iw = ow * param.stride_w + kw * param.dilation_w - param.pad_left;
//                             if (ih >= 0 && ih < param.H && iw >= 0 && iw < param.W) {
//                                 for (int ic = 0; ic < param.IC; ++ic) {
//                                     float ival = input[n * param.IC * param.H * param.W + ic * param.H * param.W + ih * param.W + iw];
//                                     float kval = kernel[oc * param.IC * param.kernel_h * param.kernel_w + ic * param.kernel_h * param.kernel_w + kh * param.kernel_w + kw];
//                                     acc += ival * kval;
//                                 }
//                             }
//                         }
//                     }
//                     output[n * param.OC * output_h * output_w + oc * output_h * output_w + oh * output_w + ow] = acc;
//                 }
//             }
//         }
//     }
// }

// static inline void convert_kernel_2IC(float *dst, const float *src, int OC, int IC, int H, int W) {
//     // src: [OC, IC, H, W]
//     // dst: [IC_new, OC, H, W, 2], where IC_new = (IC + 1) / 2
//     for (int oc = 0; oc < OC; ++oc) {
//         for (int ic = 0; ic < IC; ++ic) {
//             for (int h = 0; h < H; ++h) {
//                 for (int w = 0; w < W; ++w) {
//                     dst[((ic / 2) * OC * H * W + oc * H * W + h * W + w) * 2 + (ic % 2)] =
//                         src[oc * IC * H * W + ic * H * W + h * W + w];
//                 }
//             }
//         }
//     }
// }

void conv2d_by_arm(param_t *param){
    unsigned long long input_addr = param->input_addr;
    unsigned long long output_addr = param->output_addr;
    unsigned long long kernel_addr = param->kernel_addr;
    
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    for (int n = 0; n<param->N; ++n){
        for (int oc = 0; oc<param->OC; ++oc){
            for (int oh=0; oh < output_h; ++oh){
                for (int ow=0; ow < output_w; ++ ow){
                    float acc = 1.f;
                    for (int kh=0; kh < param->kernel_h; ++ kh){
                        for (int kw = 0; kw < param->kernel_w; ++ kw){
                            int ih = oh * param->stride_h + kh * param->dilation_h - param->pad_top;
                            int iw = ow * param->stride_w + kw * param->dilation_w - param->pad_left;
                            if (ih >= 0 && ih < param->H && iw>=0 && iw < param->W){
                                for (int ic = 0; ic < param->IC; ++ic) {
                                    float ival = *(float*)okk_global_mem_addr(input_addr + (n*param->IC * param->H * param->W + ic * param->H * param->W + ih * param->W + iw)*sizeof(float));
                                    float kval = *(float*)okk_global_mem_addr(kernel_addr + (ic/2*param->OC * param->kernel_h * param->kernel_w + oc * param->kernel_h * param->kernel_w + kh*param->kernel_w + kw + ic%2) * sizeof(float));
                                    acc += ival * kval;
                                    // float kval_1 = *(float*)okk_global_mem_addr(kernel_addr + (ic*param->OC * param->kernel_h * param->kernel_w + oc * param->kernel_h * param->kernel_w + kh * param->kernel_w + kw) * sizeof(float));
                                    // float kval_2 = 0.f;
                                    // if(ic == (param->IC - 1)/2 && param->IC%2 == 0)
                                    //     kval_2 = *(float*)okk_global_mem_addr(kernel_addr + (ic*param->OC * param->kernel_h * param->kernel_w + oc * param->kernel_h * param->kernel_w + kh * param->kernel_w + kw + 1) * sizeof(float));
                                    // acc += ival * kval_1 + ival * kval_2;
                                }
                            }
                        }
                    }
                    *(float*)okk_global_mem_addr(output_addr + (n * param->OC * output_h * output_w + oc * output_h * output_w + oh * output_w + ow)*sizeof(float)) = acc;
                }
            }
        }
    }
}

void conv2d_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    // if(param->IC >= 512)
    // {
    //     conv2d_by_arm(param);

    //     okk_poll();
    //     return;
    // }

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

    long kernel_size = kernel_shape.n * kernel_stride.n * (int)sizeof(float);

    // cal max N
    int max_N = (LOCAL_MEM_SIZE - kernel_size) / ((input_stride.n + output_stride.n) * sizeof(float));

    // max_N = MIN(max_N, param->N);

    if(max_N > 1)
    {
        max_N /= 2;

        local_addr_t input_addr[2], output_addr[2], kernel_addr;
        output_addr[0] = 0;
        input_addr[0] = output_addr[0] + max_N * output_stride.n * sizeof(float);

        output_addr[1] = input_addr[0] + max_N * input_stride.n * sizeof(float);
        input_addr[1] = output_addr[1] + max_N * output_stride.n * sizeof(float);

        kernel_addr = input_addr[1] + max_N * input_stride.n * sizeof(float);

        OKKERNEL_ASSERT(output_addr[1] % 128 == 0);
        OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);

        okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);

        int iteration = DIV_UP(param->N, max_N);

        dim4 new_input_shape = {.n=max_N, .c=input_shape.c, .h=input_shape.h, .w=input_shape.w};
        dim4 new_output_shape = {.n=max_N, .c=output_shape.c, .h=output_shape.h, .w=output_shape.w};
        unsigned int input_tensor_size_global = max_N * new_input_shape.c * new_input_shape.h * new_input_shape.w * sizeof(float);
        unsigned int output_tensor_size_global = max_N * new_output_shape.c *new_output_shape.h *new_output_shape.w * sizeof(float);

        dim4 intput_shape_last = {.n=param->N - (iteration - 1)*max_N, .c=param->IC, .h=param->H, .w=param->W};
        dim4 output_shape_last = {.n=param->N - (iteration - 1)*max_N, .c=param->OC, .h=output_h, .w=output_w};

        Padding padding = {
            .left=param->pad_left, .right=param->pad_right,
            .top=param->pad_top, .bottom=param->pad_bottom
        };
        dim2 stride = {.h = param->stride_h, .w = param->stride_w};
        dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
        dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
        dim4 kernel_stride_2IC;
        okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

        for (int i=0; i<iteration+2;i++){
            okk_parallel_start();

            if(i<iteration)
                okk_gdma_32bit_cpy_S2L(input_addr[i%2], param->input_addr + i * input_tensor_size_global, i==iteration-1?&intput_shape_last:&new_input_shape, NO_USE, NO_USE);

            if(i>0 && i<iteration+1)
                okk_bdc_conv2d(output_addr[(i-1)%2], input_addr[(i-1)%2], kernel_addr, NO_USE, i == iteration?&intput_shape_last:&new_input_shape, param->OC, param->kernel_h, param->kernel_w, &input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);
                // okk_bdc_depthwise2d(output_addr[(i-1)%2], input_addr[(i-1)%2], kernel_addr, NO_USE, i == iteration?&intput_shape_last:&new_input_shape, param->kernel_h, param->kernel_w, false, &padding, &stride, &dilation);
            
            if(i>1)
                okk_gdma_32bit_cpy_L2S(param->output_addr + (i-2)*output_tensor_size_global, output_addr[i%2], i == iteration+1? &output_shape_last:&new_output_shape, NO_USE, NO_USE);

            okk_parallel_end();
        }

        okk_poll();
        return;
    }

    // cal max H*W
    int input_channel_per_npu = DIV_UP(param->IC, NPU_NUM);
    int output_channel_per_npu = DIV_UP(param->OC, NPU_NUM);
    long output_size = output_shape.n * output_stride.n * sizeof(float);
    long max_HW = ((LOCAL_MEM_SIZE - kernel_size) / (input_shape.n * input_channel_per_npu * sizeof(float) + output_shape.n * output_channel_per_npu * sizeof(float))) / 32 * 32;

    
    // TODO avoid overflow by data overlap
    int patches = DIV_UP(param->H * param->W, max_HW);

    // OKKERNEL_LOG("patches:%d\n", patches);

    // patches = 36;

    int row_patches = (int)sqrt(patches);
    int col_patches = DIV_UP(patches, row_patches);

    int patch_row_iter_size = DIV_UP(output_shape.h, row_patches);
    int patch_col_iter_size = DIV_UP(output_shape.w, col_patches);

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
        
        if(row_p*patch_row_iter_size*param->stride_h + patch_row > param->pad_top + param->H)
            patch_row -= (row_p*patch_row_iter_size*param->stride_h + patch_row) - (param->pad_top + param->H);
        
        for(int col_p = 0; col_p < col_patches; col_p++)
        {
            
            int patch_col = (patch_col_iter_size - 1) * param->stride_w + kernel_w_ext;

            // cal offset
            long input_offset = MAX((row_p*patch_row_iter_size*param->stride_h - param->pad_top) * param->W * (long)sizeof(float), 0) + MAX((col_p * patch_col_iter_size * param->stride_w - param->pad_left) * (long)sizeof(float), 0);
            long output_offset = (row_p*patch_row_iter_size) * output_shape.w * sizeof(float) + col_p * patch_col_iter_size * sizeof(float);

            // cal temp patch H&W
            if(col_p == 0)
                patch_col -= param->pad_left;
            
            if(col_p*patch_col_iter_size*param->stride_w + patch_col > param->pad_left + param->W)
                patch_col -= (col_p*patch_col_iter_size*param->stride_w + patch_col) - (param->pad_left + param->W);

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

            okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr+input_offset, &patch_input_shape, NO_USE, &system_input_stride);

            dim2 stride = {.h = param->stride_h, .w = param->stride_w};
            dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
            dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
            dim4 kernel_stride_2IC;
            okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
            
            okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, &patch_input_shape, param->OC, param->kernel_h, param->kernel_w, &patch_input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

            okk_gdma_32bit_cpy_L2S(param->output_addr+output_offset, output_addr, &patch_output_shape, &system_output_stride, NO_USE);

        }
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(conv2d_contest);
