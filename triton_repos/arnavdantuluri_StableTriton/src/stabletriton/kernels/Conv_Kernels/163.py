import torch
import triton
import triton.language as tl
from conv_utils import _unpack, conv_heuristics


@conv_heuristics()
@triton.jit
def _kernel_delta_x_hwc(
    x,
    w,
    y,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    stride_biasn,
    delta_xh_ptr,
    delta_xw_ptr,
    delta_xc_ptr,
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_H: tl.constexpr,
):

    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W

    if not CONV1X1_NHWC:
        delta_xh_ptrs = delta_xh_ptr + off_x_crs
        delta_xw_ptrs = delta_xw_ptr + off_x_crs
        delta_xc_ptrs = delta_xc_ptr + off_x_crs
        delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
        off_x_crs_unpacked = (
            delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
        )
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
        delta_xh = 0
        delta_xw = 0

    mask_x = (
        (off_x_n < BATCH)[:, None]
        & (off_x_crs < CRS)[None, :]
        & (off_x_h[:, None] + delta_xh[None, :] >= 0)
        & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
        & (off_x_w[:, None] + delta_xw[None, :] >= 0)
        & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
    )

    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        acc += tl.dot(matrix_x, matrix_w)

        w_ptrs += BLOCK_K

        off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
        if not CONV1X1_NHWC:
            delta_xh_ptrs += BLOCK_K
            delta_xw_ptrs += BLOCK_K
            delta_xc_ptrs += BLOCK_K
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = (
                delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
            )
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs += BLOCK_K

        mask_x = (
            (off_x_n < BATCH)[:, None]
            & (off_x_crs < CRS)[None, :]
            & (off_x_h[:, None] + delta_xh[None, :] >= 0)
            & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
            & (off_x_w[:, None] + delta_xw[None, :] >= 0)
            & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
        )
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(y.dtype.element_ty)

    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)

    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

    return


@conv_heuristics()
@triton.jit
def _kernel_delta_x(
    x,
    w,
    y,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    stride_biasn,
    delta_x_ptr,
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_H: tl.constexpr,
):

    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W

    if not CONV1X1_NHWC:
        delta_x_ptrs = delta_x_ptr + off_x_crs
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]

    mask_x = (
        (off_x_n < BATCH)
        & (off_x_h >= 0)
        & (off_x_h < IN_H)
        & (off_x_w >= 0)
        & (off_x_w < IN_W)
    )[:, None] & (off_x_crs < CRS)[None, :]

    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        acc += tl.dot(matrix_x, matrix_w)

        w_ptrs += BLOCK_K

        if not CONV1X1_NHWC:
            delta_x_ptrs += BLOCK_K
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            x_ptrs += BLOCK_K

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_h >= 0)
            & (off_x_h < IN_H)
            & (off_x_w >= 0)
            & (off_x_w < IN_W)
        )[:, None] & (off_x_crs < CRS)[None, :]
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(y.dtype.element_ty)

    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)

    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)


class _conv:
    kernel = _kernel_delta_x_hwc

    @staticmethod
    def _delta_x_ptr_hwc(
        IN_C,
        KERNEL_H,
        KERNEL_W,
        dilation_h,
        dilation_w,
        stride_wc,
        stride_wh,
        stride_ww,
        stride_xc,
        stride_xh,
        stride_xw,
        device,
    ):

        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c

        return (
            r_dilation_h,
            r_dilation_w,
            r_inc,
        )

    @staticmethod
    def _delta_x_ptr(
        IN_C,
        KERNEL_H,
        KERNEL_W,
        dilation_h,
        dilation_w,
        stride_wc,
        stride_wh,
        stride_ww,
        stride_xc,
        stride_xh,
        stride_xw,
        device,
    ):

        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        delta_x = (
            r_dilation_h * stride_xh + r_dilation_w * stride_xw + r_inc * stride_xc
        )
        return delta_x

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):

        device = x.device

        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        xn, xc, xh, xw = 0, 1, 2, 3
        yn, yc, yh, yw = 0, 1, 2, 3
        wn, wc, wh, ww = 0, 1, 2, 3

        kernel_size = [shape_w[wh], shape_w[ww]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert (
            not shape_bias or shape_bias[0] == shape_w[wn]
        ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
        assert (
            shape_x[xc] == in_channel
        ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

        assert (
            len(stride)
            == len(padding)
            == len(dilation)
            == len(output_padding)
            == len(kernel_size)
            == len(input_size)
        )

        shape_y = [0] * 4
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        shape_y[yh] = (
            input_size[0]
            + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1)
            - 1
            + stride[0]
        ) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (
            input_size[1]
            + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1)
            - 1
            + stride[1]
        ) // stride[1] + 2 * output_padding[1]

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_H = shape_x[xh]
        IN_W = shape_x[xw]
        KERNEL_N = shape_w[wn]
        KERNEL_H = shape_w[wh]
        KERNEL_W = shape_w[ww]
        OUT_H = shape_y[yh]
        OUT_W = shape_y[yw]

        y = torch.empty(shape_y, device=device, dtype=x.dtype)

        stride_x = x.stride()
        stride_w = w.stride()
        stride_bias = bias.stride() if shape_bias else None
        stride_biasn = stride_bias[0] if stride_bias else None

        if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
            y = y.to(memory_format=torch.channels_last)
        stride_y = y.stride()

        ACC_TYPE = (
            tl.float32
            if x.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )

        CONV1X1_NHWC = False
        if stride_x[xc] == 1 and KERNEL_H == 1 and KERNEL_W == 1:
            CONV1X1_NHWC = True

        DELTA_X_PTR_HWC = (
            False
            if (
                (padding[0] == 0 and padding[1] == 0)
                or (KERNEL_H == 1 and KERNEL_W == 1)
            )
            else True
        )
        if not CONV1X1_NHWC:
            if DELTA_X_PTR_HWC:
                delta_xh, delta_xw, delta_xc = _conv._delta_x_ptr_hwc(
                    IN_C,
                    KERNEL_H,
                    KERNEL_W,
                    dilation[0],
                    dilation[1],
                    stride_w[wc],
                    stride_w[wh],
                    stride_w[ww],
                    stride_x[xc],
                    stride_x[xh],
                    stride_x[xw],
                    device,
                )
            else:
                delta_x = _conv._delta_x_ptr(
                    IN_C,
                    KERNEL_H,
                    KERNEL_W,
                    dilation[0],
                    dilation[1],
                    stride_w[wc],
                    stride_w[wh],
                    stride_w[ww],
                    stride_x[xc],
                    stride_x[xh],
                    stride_x[xw],
                    device,
                )
        else:
            delta_x = None
            delta_xh, delta_xw, delta_xc = None, None, None

        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                triton.cdiv(KERNEL_N, META["BLOCK_N"]),
            )

        if CONV1X1_NHWC or not DELTA_X_PTR_HWC:
            _kernel_delta_x[grid](
                x,
                w,
                y,
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                stride_biasn,
                delta_x,
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                ACC_TYPE=ACC_TYPE,
                CONV1X1_NHWC=CONV1X1_NHWC,
                GROUP_H=1,
            )

        else:

            _kernel_delta_x_hwc[grid](
                x,
                w,
                y,
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                stride_biasn,
                delta_xh,
                delta_xw,
                delta_xc,
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                ACC_TYPE=ACC_TYPE,
                CONV1X1_NHWC=CONV1X1_NHWC,
                GROUP_H=1,
            )

        if bias is not None:
            if len(bias.shape) == 1:
                bias = bias.reshape([1, bias.shape[0], 1, 1])
            y += bias
        return y

    @staticmethod
    def forward(
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
    ):
        if groups != 1:
            print(f"Do not support groups = {groups}")
            return
        if transposed:
            print("Do not support transposed")
        return _conv._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )


conv = _conv.forward

conv_torch = torch.nn.Conv2d(
    128, 256, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1)
).cuda()
x = torch.rand((2, 128, 64, 64)).cuda()
torch_out = conv_torch(x)
triton_out = conv(x, conv_torch.weight, conv_torch.bias)
