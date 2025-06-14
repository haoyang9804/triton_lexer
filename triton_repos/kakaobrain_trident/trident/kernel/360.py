import triton
import triton.language as tl


class GroupNorm:
    @staticmethod
    @triton.heuristics(
        {
            "require_group_boundary_check": lambda args: args["group_size"]
            % args["group_block_size"],
            "require_x_boundary_check": lambda args: args["x_size"]
            % args["x_block_size"],
        }
    )
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        rstd_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        group_size: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        num_groups: tl.int32,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        eps: tl.float32,
        dtype: tl.constexpr,
        group_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
        require_group_boundary_check: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // num_groups
        group = pid % num_groups
        num_elements = group_size * x_size
        batch_offset = batch * num_groups * num_elements
        group_offset = batch_offset + group * num_elements
        output_block_ptr = tl.make_block_ptr(
            output_ptr + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        rstd_block_ptr = tl.make_block_ptr(
            rstd_ptr + batch * num_groups,
            shape=(group_size,),
            strides=(1,),
            offsets=(group,),
            block_shape=(1,),
            order=(0,),
        )
        mean_block_ptr = tl.make_block_ptr(
            mean_ptr + batch * num_groups,
            shape=(group_size,),
            strides=(1,),
            offsets=(group,),
            block_shape=(1,),
            order=(0,),
        )

        if require_group_boundary_check | require_x_boundary_check:
            input = tl.load(
                input_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            mean = tl.sum(
                tl.view(input / num_elements, (1, group_block_size * x_block_size)), 1
            )
            group_condition = tl.arange(0, group_block_size) < group_size
            x_condition = tl.arange(0, x_block_size) < x_size
            condition = group_condition[:, None] & x_condition[None, :]
            centered_mean = tl.where(condition, input - mean, 0)
        else:
            input = tl.load(input_block_ptr)
            mean = tl.sum(
                tl.view(input / num_elements, (1, group_block_size * x_block_size)), 1
            )
            centered_mean = input - mean

        var = tl.sum(
            tl.view(
                centered_mean * centered_mean / num_elements,
                (1, group_block_size * x_block_size),
            ),
            1,
        )
        rstd = tl.math.rsqrt(var + eps)
        output = centered_mean * rstd

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(y_size, 1),
                strides=(1, y_size),
                offsets=(group * group_size, 0),
                block_shape=(group_block_size, 1),
                order=(0, 1),
            )
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
            output *= weight

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(y_size, 1),
                strides=(1, y_size),
                offsets=(group * group_size, 0),
                block_shape=(group_block_size, 1),
                order=(0, 1),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(0,))
            output += bias

        if require_group_boundary_check | require_x_boundary_check:
            tl.store(output_block_ptr, output.to(dtype), boundary_check=(0, 1))
        else:
            tl.store(output_block_ptr, output.to(dtype))

        tl.store(rstd_block_ptr, rstd.to(dtype))
        tl.store(mean_block_ptr, mean.to(dtype))

    @staticmethod
    @triton.heuristics(
        {
            "require_group_boundary_check": lambda args: args["group_size"]
            % args["group_block_size"],
            "require_x_boundary_check": lambda args: args["x_size"]
            % args["x_block_size"],
        }
    )
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_staging_ptr: tl.tensor,
        grad_bias_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        group_size: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        num_groups: tl.int32,
        weight_ptr: tl.tensor,
        rstd_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        dtype: tl.constexpr,
        group_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
        require_group_boundary_check: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // num_groups
        group = pid % num_groups
        num_elements = group_size * x_size
        batch_offset = batch * num_groups * num_elements
        group_offset = batch_offset + group * num_elements
        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr + group_offset,
            shape=(group_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(group_block_size, x_block_size),
            order=(1, 0),
        )
        rstd_block_ptr = tl.make_block_ptr(
            rstd_ptr + batch * num_groups,
            shape=(group_size,),
            strides=(1,),
            offsets=(group,),
            block_shape=(1,),
            order=(0,),
        )
        mean_block_ptr = tl.make_block_ptr(
            mean_ptr + batch * num_groups,
            shape=(group_size,),
            strides=(1,),
            offsets=(group,),
            block_shape=(1,),
            order=(0,),
        )

        rstd = tl.load(rstd_block_ptr)
        mean = tl.load(mean_block_ptr)

        if require_group_boundary_check | require_x_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1))
        else:
            grad_output = tl.load(grad_output_block_ptr)

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(y_size, 1),
                strides=(1, y_size),
                offsets=(group * group_size, 0),
                block_shape=(group_block_size, 1),
                order=(0, 1),
            )
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
            grad_norm = weight * grad_output
        else:
            grad_norm = grad_output

        if require_group_boundary_check | require_x_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(0, 1))
            group_condition = tl.arange(0, group_block_size) < group_size
            x_condition = tl.arange(0, x_block_size) < x_size
            condition = group_condition[:, None] & x_condition[None, :]
            centered_mean = tl.where(condition, input - mean, 0)
            grad_std = tl.sum(
                tl.view(
                    grad_norm * centered_mean, (1, group_block_size * x_block_size)
                ),
                1,
            )
            grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / (x_size * group_size)
            grad_distance = 2 * centered_mean * grad_var
            grad_centered_mean = tl.where(
                condition, grad_norm * rstd + grad_distance, 0
            )
            grad_mean = (
                -tl.sum(
                    tl.view(grad_centered_mean, (1, group_block_size * x_block_size)), 1
                )
                / num_elements
            )
            grad_input = grad_centered_mean + grad_mean
            tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0, 1))
        else:
            input = tl.load(input_block_ptr)
            centered_mean = input - mean
            grad_std = tl.sum(
                tl.view(
                    grad_norm * centered_mean, (1, group_block_size * x_block_size)
                ),
                1,
            )
            grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / (x_size * group_size)
            grad_distance = 2 * centered_mean * grad_var
            grad_centered_mean = grad_norm * rstd + grad_distance
            grad_mean = (
                -tl.sum(
                    tl.view(grad_centered_mean, (1, group_block_size * x_block_size)), 1
                )
                / num_elements
            )
            grad_input = grad_centered_mean + grad_mean
            tl.store(grad_input_block_ptr, grad_input.to(dtype))

        if grad_weight_staging_ptr is not None:
            norm = centered_mean * rstd
            grad_weight = tl.sum(norm * grad_output, 1)
            offset = batch * y_size + group * group_size
            grad_weight_staging_block_ptr = tl.make_block_ptr(
                grad_weight_staging_ptr + offset,
                shape=(group_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(group_block_size,),
                order=(0,),
            )

            if require_group_boundary_check:
                tl.store(
                    grad_weight_staging_block_ptr,
                    grad_weight.to(dtype),
                    boundary_check=(0,),
                )
            else:
                tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))

        if grad_bias_staging_ptr is not None:
            grad_bias = tl.sum(grad_output, 1)
            offset = batch * y_size + group * group_size
            grad_bias_staging_block_ptr = tl.make_block_ptr(
                grad_bias_staging_ptr + offset,
                shape=(group_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(group_block_size,),
                order=(0,),
            )

            if require_group_boundary_check:
                tl.store(
                    grad_bias_staging_block_ptr,
                    grad_bias.to(dtype),
                    boundary_check=(0,),
                )
            else:
                tl.store(grad_bias_staging_block_ptr, grad_bias.to(dtype))
