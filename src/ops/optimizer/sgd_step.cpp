#include "ops/optimizer/sgd_step.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::optimizer
{

void sgdStep(Tensor &parameter,
             const Tensor &gradient,
             float learning_rate,
             float max_gradient,
             float inv_scale,
             const std::shared_ptr<gpu::vector> &dynamic_params_buffer,
             float weight_decay,
             float momentum,
             Tensor *velocity)
{
    auto param_impl = parameter.getImplementation();
    auto grad_impl = gradient.getImplementation();

    if (!param_impl || !grad_impl)
    {
        Logger::logMessage("ops::optimizer::sgdStep: Null implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::OPTIMIZER_STEP);
        return;
    }

    if (gradient.getExecutionTarget() != parameter.getExecutionTarget())
    {
        throw std::invalid_argument("ops::optimizer::sgdStep: Execution_Target mismatch");
    }

    if (parameter.getDataType() == Data_Type::TERNARY_INT2 ||
        gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::optimizer::sgdStep: TERNARY_INT2 is not supported");
    }

    param_impl->validateSameDimensions(*grad_impl);

    bool use_momentum = (momentum > 0.0f && velocity != nullptr);
    if (use_momentum)
    {
        auto vel_impl = velocity->getImplementation();
        if (!vel_impl)
        {
            throw std::invalid_argument("ops::optimizer::sgdStep: Null velocity implementation");
        }
        if (velocity->getExecutionTarget() != parameter.getExecutionTarget())
        {
            throw std::invalid_argument("ops::optimizer::sgdStep: Velocity Execution_Target mismatch");
        }
        if (velocity->getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::optimizer::sgdStep: TERNARY_INT2 velocity is not supported");
        }
        param_impl->validateSameDimensions(*vel_impl);
    }

    size_t total_elements = parameter.getTotalElements();
    if (total_elements == 0)
    {
        return;
    }

    if (parameter.getExecutionTarget() == Execution_Target::CPU)
    {
        auto &param_cpu = Cpu_Tensor_Impl::castToCpu(*param_impl);
        if (!param_cpu.isContiguous() || param_cpu.getByteOffset() != 0)
        {
            throw std::invalid_argument("ops::optimizer::sgdStep: Non-contiguous parameter view on CPU");
        }

        Cpu_Tensor_Impl *v_cpu = nullptr;
        if (use_momentum)
        {
            v_cpu = &Cpu_Tensor_Impl::castToCpu(*velocity->getImplementation());
            if (!v_cpu->isContiguous() || v_cpu->getByteOffset() != 0)
            {
                throw std::invalid_argument("ops::optimizer::sgdStep: Non-contiguous velocity view on CPU");
            }
        }

        std::vector<float> grad_vals = gradient.getData();

        auto loadCpuState = [total_elements](Cpu_Tensor_Impl &impl, std::vector<float> &temp) -> std::vector<float> *
        {
            if (impl.getDataType() == Data_Type::FLOAT16)
            {
                temp.assign(total_elements, 0.0f);
                const auto &fp16_buf = impl.getStorageBufferFp16();
                if (fp16_buf && !fp16_buf->empty())
                {
                    size_t copy_count = std::min(total_elements, fp16_buf->size());
                    convertFp16ToFp32(fp16_buf->data(), temp.data(), copy_count);
                }
                return &temp;
            }
            auto &fp32_buf = impl.getStorageBuffer();
            if (!fp32_buf)
            {
                fp32_buf = std::make_shared<std::vector<float>>(total_elements, 0.0f);
            }
            else if (fp32_buf.use_count() > 1)
            {
                fp32_buf = std::make_shared<std::vector<float>>(*fp32_buf);
            }
            if (fp32_buf->size() < total_elements)
            {
                fp32_buf->resize(total_elements, 0.0f);
            }
            impl.getStorageBufferFp16().reset();
            impl.getStorageBufferU32().reset();
            return fp32_buf.get();
        };

        auto storeCpuFp16State = [total_elements](Cpu_Tensor_Impl &impl, const std::vector<float> &vals)
        {
            std::vector<float> clamped(total_elements);
            for (size_t i = 0; i < total_elements; ++i)
            {
                clamped[i] = std::clamp(vals[i], -65504.0f, 65504.0f);
            }
            auto &fp16_buf = impl.getStorageBufferFp16();
            if (!fp16_buf || fp16_buf.use_count() > 1 || fp16_buf->size() != total_elements)
            {
                fp16_buf = std::make_shared<std::vector<float16_t>>(total_elements);
            }
            convertFp32ToFp16(clamped.data(), fp16_buf->data(), total_elements);
            impl.getStorageBuffer().reset();
            impl.getStorageBufferU32().reset();
        };

        std::vector<float> temp_param;
        std::vector<float> temp_v;
        std::vector<float> *param_vec = loadCpuState(param_cpu, temp_param);
        std::vector<float> *v_vec = use_momentum ? loadCpuState(*v_cpu, temp_v) : nullptr;

        for (size_t i = 0; i < total_elements; ++i)
        {
            float g = grad_vals[i] * inv_scale;
            if (std::isnan(g) || std::isinf(g))
            {
                continue;
            }
            if (max_gradient > 0.0f)
            {
                g = std::clamp(g, -max_gradient, max_gradient);
            }
            float step_dir = g;
            if (use_momentum)
            {
                (*v_vec)[i] = momentum * (*v_vec)[i] + g;
                step_dir = (*v_vec)[i];
            }
            if (weight_decay > 0.0f)
            {
                (*param_vec)[i] -= learning_rate * (step_dir + weight_decay * (*param_vec)[i]);
            }
            else
            {
                (*param_vec)[i] -= learning_rate * step_dir;
            }
        }

        if (param_cpu.getDataType() == Data_Type::FLOAT16)
        {
            storeCpuFp16State(param_cpu, temp_param);
        }
        if (use_momentum && v_cpu->getDataType() == Data_Type::FLOAT16)
        {
            storeCpuFp16State(*v_cpu, temp_v);
        }

        Logger::logMessage(Input_Format{"ops::optimizer::sgdStep (CPU): elements={}, lr={}, max_grad={}",
                                        total_elements, learning_rate, max_gradient},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }
    else
    {
        auto &param_gpu = Gpu_Tensor_Impl::castToGpu(*param_impl);
        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);

        if (!param_gpu.isContiguous() || param_gpu.getByteOffset() != 0)
        {
            throw std::invalid_argument("ops::optimizer::sgdStep: Non-contiguous parameter view on GPU");
        }

        Gpu_Tensor_Impl *v_gpu = nullptr;
        if (use_momentum)
        {
            v_gpu = &Gpu_Tensor_Impl::castToGpu(*velocity->getImplementation());
            if (!v_gpu->isContiguous() || v_gpu->getByteOffset() != 0)
            {
                throw std::invalid_argument("ops::optimizer::sgdStep: Non-contiguous velocity view on GPU");
            }
            if (!v_gpu->getStorageBuffer())
            {
                throw std::runtime_error("ops::optimizer::sgdStep: Missing velocity GPU storage buffer");
            }
        }

        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        if (!param_gpu.getStorageBuffer() || !effective_grad->getStorageBuffer())
        {
            throw std::runtime_error("ops::optimizer::sgdStep: Missing GPU storage buffer");
        }

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        std::shared_ptr<gpu::vector> dyn_buf = dynamic_params_buffer;
        if (!dyn_buf)
        {
            dyn_buf = engine.allocateDynamicOptimizerParamsBuffer(learning_rate, 1.0f, 1.0f, inv_scale, current_frame);
        }

        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::optimizer::sgdStep: Element count exceeds UINT32_MAX");
        }
        uint32_t total_u32 = static_cast<uint32_t>(total_elements);
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total_elements + 255ull) / 256ull);

        auto castToFp32 = [&](const std::shared_ptr<gpu::vector> &src) -> std::shared_ptr<gpu::vector>
        {
            auto tmp = std::make_shared<gpu::vector>(engine.getContext(), total_elements, Data_Type::FLOAT32);
            param_gpu.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {src, tmp}, total_u32, wg_x, wg_y, 1);
            return tmp;
        };

        bool param_is_fp16 = (param_gpu.getDataType() == Data_Type::FLOAT16);
        std::shared_ptr<gpu::vector> param_buf_fp32 = param_is_fp16
            ? castToFp32(param_gpu.getStorageBuffer())
            : param_gpu.getStorageBuffer();

        std::shared_ptr<gpu::vector> grad_buf_fp32 = (effective_grad->getDataType() == Data_Type::FLOAT16)
            ? castToFp32(effective_grad->getStorageBuffer())
            : effective_grad->getStorageBuffer();

        bool v_is_fp16 = use_momentum && (v_gpu->getDataType() == Data_Type::FLOAT16);
        std::shared_ptr<gpu::vector> v_buf_fp32 = nullptr;
        if (use_momentum)
        {
            v_buf_fp32 = v_is_fp16 ? castToFp32(v_gpu->getStorageBuffer()) : v_gpu->getStorageBuffer();
        }

        bool use_fp16_cache = !param_is_fp16 &&
                              (engine.isCooperativeMatrixEnabled() ||
                               param_gpu.getCachedFp16Storage() != nullptr);

        if (use_momentum)
        {
            struct SGD_Momentum_Constants
            {
                uint32_t total_elements;
                float max_gradient;
                float weight_decay;
                float momentum;
            } constants{total_u32, max_gradient, weight_decay, momentum};

            if (param_is_fp16)
            {
                param_gpu.pushToGraph(Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16,
                                      {param_buf_fp32, grad_buf_fp32, v_buf_fp32, dyn_buf, param_gpu.getStorageBuffer()},
                                      constants, wg_x, wg_y, 1);
                param_gpu.setFp16CacheDirty(true);
            }
            else if (use_fp16_cache)
            {
                auto fp16_buf = param_gpu.ensureCachedFp16Storage(total_elements);
                param_gpu.pushToGraph(Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16,
                                      {param_buf_fp32, grad_buf_fp32, v_buf_fp32, dyn_buf, fp16_buf},
                                      constants, wg_x, wg_y, 1);
                param_gpu.setFp16CacheDirty(false);
            }
            else
            {
                param_gpu.pushToGraph(Compute_Pipeline::SGD_MOMENTUM_UPDATE,
                                      {param_buf_fp32, grad_buf_fp32, v_buf_fp32, dyn_buf},
                                      constants, wg_x, wg_y, 1);
                param_gpu.setFp16CacheDirty(true);
            }

            if (v_is_fp16)
            {
                v_gpu->pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16, {v_buf_fp32, v_gpu->getStorageBuffer()}, total_u32, wg_x, wg_y, 1);
            }
            v_gpu->setFp16CacheDirty(true);
            v_gpu->clearHostCache();
        }
        else
        {
            struct SGD_Constants
            {
                uint32_t total_elements;
                float max_gradient;
                float weight_decay;
            } constants{total_u32, max_gradient, weight_decay};

            if (param_is_fp16)
            {
                param_gpu.pushToGraph(Compute_Pipeline::SGD_UPDATE_FP16,
                                      {param_buf_fp32, grad_buf_fp32, dyn_buf, param_gpu.getStorageBuffer()},
                                      constants, wg_x, wg_y, 1);
                param_gpu.setFp16CacheDirty(true);
            }
            else if (use_fp16_cache)
            {
                auto fp16_buf = param_gpu.ensureCachedFp16Storage(total_elements);
                param_gpu.pushToGraph(Compute_Pipeline::SGD_UPDATE_FP16,
                                      {param_buf_fp32, grad_buf_fp32, dyn_buf, fp16_buf},
                                      constants, wg_x, wg_y, 1);
                param_gpu.setFp16CacheDirty(false);
            }
            else
            {
                param_gpu.pushToGraph(Compute_Pipeline::SGD_UPDATE,
                                      {param_buf_fp32, grad_buf_fp32, dyn_buf},
                                      constants, wg_x, wg_y, 1);
                param_gpu.setFp16CacheDirty(true);
            }
        }
        param_gpu.clearHostCache();
    }
}

}
