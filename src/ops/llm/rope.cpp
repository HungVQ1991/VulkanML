#include "ops/llm/rope.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::llm
{

static void ensureTensorTarget(Tensor &t, Execution_Target target, const Shape &shape, Data_Type dtype)
{
    if (t.getExecutionTarget() != target || t.getDataType() != dtype || t.getShape() != shape)
    {
        t = Tensor(shape, dtype, target);
    }
    else
    {
        t.reshape(shape);
    }
}

void applyRoPE(const Tensor &input, Tensor &output, uint32_t seq_len, uint32_t head_dim,
               int direction, float base, uint32_t num_heads, uint32_t mode)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    Execution_Target target = input.getExecutionTarget();
    Shape in_shape = input.getShape();
    Shape out_shape;
    if (mode == 1)
    {
        out_shape = Shape{in_shape[0], num_heads, seq_len, head_dim};
    }
    else if (mode == 2)
    {
        out_shape = Shape{in_shape[0], seq_len, num_heads, head_dim};
    }
    else
    {
        out_shape = in_shape;
    }

    bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type out_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : input.getDataType();

    ensureTensorTarget(output, target, out_shape, out_dtype);

    if (target == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        size_t total = input.getTotalElements();
        size_t total_pairs = total / 2;
        size_t D = head_dim;
        size_t S = seq_len;
        size_t H = num_heads;

        for (size_t pair_id = 0; pair_id < total_pairs; ++pair_id)
        {
            size_t base_idx = pair_id * 2;
            size_t c = base_idx % D;
            size_t pair_idx = c / 2;
            size_t token_pos = 0;
            size_t in_idx = base_idx;
            size_t out_idx = base_idx;

            if (mode == 1)
            {
                token_pos = (base_idx / D) % S;
                size_t h = (base_idx / (D * S)) % H;
                size_t b = base_idx / (D * S * H);
                in_idx = b * (S * H * D) + token_pos * (H * D) + h * D + c;
                out_idx = base_idx;
            }
            else if (mode == 2)
            {
                token_pos = (base_idx / D) % S;
                size_t h = (base_idx / (D * S)) % H;
                size_t b = base_idx / (D * S * H);
                in_idx = base_idx;
                out_idx = b * (S * H * D) + token_pos * (H * D) + h * D + c;
            }
            else
            {
                token_pos = (base_idx / D) % S;
                in_idx = base_idx;
                out_idx = base_idx;
            }

            float theta = std::pow(base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(D));
            float alpha = static_cast<float>(direction) * static_cast<float>(token_pos) * theta;
            float cos_a = std::cos(alpha);
            float sin_a = std::sin(alpha);

            float x0 = in_data[in_idx];
            float x1 = in_data[in_idx + 1];

            (*out_cpu.getStorageBuffer())[out_idx]     = x0 * cos_a - x1 * sin_a;
            (*out_cpu.getStorageBuffer())[out_idx + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        uint32_t total = static_cast<uint32_t>(effective_self->getTotalElements());
        uint32_t total_pairs = total / 2;

        struct RoPE_Constants
        {
            uint32_t total_pairs;
            uint32_t seq_len;
            uint32_t head_dim;
            int32_t direction;
            float base;
            uint32_t num_heads;
            uint32_t mode;
        } pcs{total_pairs, seq_len, head_dim, direction, base, num_heads, mode};

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::ROPE_FP16 : Compute_Pipeline::ROPE,
                                    {buf_self, out_gpu.getStorageBuffer()},
                                    pcs, (total_pairs + 255) / 256, 1, 1);
    }
}

} // namespace ops::llm
