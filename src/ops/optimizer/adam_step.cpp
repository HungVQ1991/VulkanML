#include "ops/optimizer/adam_step.h"

#include <algorithm>
#include <cmath>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::optimizer
{

void adamStep(Tensor &parameter,
              const Tensor &gradient,
              Tensor &first_moment,
              Tensor &second_moment,
              float learning_rate,
              float beta1,
              float beta2,
              float epsilon,
              size_t timestep,
              float max_gradient,
              float inv_scale,
              float weight_decay)
{
    auto param_impl = parameter.getImplementation();
    auto grad_impl = gradient.getImplementation();
    auto m_impl = first_moment.getImplementation();
    auto v_impl = second_moment.getImplementation();

    if (!param_impl || !grad_impl || !m_impl || !v_impl)
    {
        Logger::logMessage("ops::optimizer::adamStep: Null implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::OPTIMIZER_STEP);
        return;
    }

    param_impl->validateSameDimensions(*grad_impl);
    param_impl->validateSameDimensions(*m_impl);
    param_impl->validateSameDimensions(*v_impl);

    size_t total_elements = parameter.getTotalElements();
    if (total_elements == 0)
    {
        return;
    }

    if (parameter.getExecutionTarget() == Execution_Target::CPU)
    {
        auto &param_cpu = Cpu_Tensor_Impl::castToCpu(*param_impl);
        auto &m_cpu = Cpu_Tensor_Impl::castToCpu(*m_impl);
        auto &v_cpu = Cpu_Tensor_Impl::castToCpu(*v_impl);

        if (!param_cpu.getStorageBuffer())
        {
            param_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        if (!m_cpu.getStorageBuffer())
        {
            m_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        if (!v_cpu.getStorageBuffer())
        {
            v_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }

        const auto &grad_data = gradient.getData();
        auto &param_buf = *param_cpu.getStorageBuffer();
        auto &m_buf = *m_cpu.getStorageBuffer();
        auto &v_buf = *v_cpu.getStorageBuffer();

        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));

        for (size_t i = 0; i < total_elements; ++i)
        {
            float raw_g = grad_data[i] * inv_scale;
            if (std::isnan(raw_g) || std::isinf(raw_g))
            {
                continue;
            }
            float g = (max_gradient > 0.0f) ? std::clamp(raw_g, -max_gradient, max_gradient) : raw_g;
            m_buf[i] = beta1 * m_buf[i] + (1.0f - beta1) * g;
            v_buf[i] = beta2 * v_buf[i] + (1.0f - beta2) * (g * g);

            float m_hat = m_buf[i] / bc1;
            float v_hat = v_buf[i] / bc2;
            float step_val = m_hat / (std::sqrt(v_hat) + epsilon);
            if (weight_decay > 0.0f)
            {
                param_buf[i] -= learning_rate * (step_val + weight_decay * param_buf[i]);
            }
            else
            {
                param_buf[i] -= learning_rate * step_val;
            }
        }

        Logger::logMessage(Input_Format{"ops::optimizer::adamStep (CPU): step={}, lr={}",
                                        timestep, learning_rate},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }
    else
    {
        auto &param_gpu = Gpu_Tensor_Impl::castToGpu(*param_impl);
        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        const auto &m_gpu = Gpu_Tensor_Impl::castToGpu(*m_impl);
        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v_impl);

        size_t effective_t = std::max<size_t>(timestep, 1);
        float bc1 = std::max(1.0f - std::pow(beta1, static_cast<float>(effective_t)), 1e-8f);
        float bc2 = std::max(1.0f - std::pow(beta2, static_cast<float>(effective_t)), 1e-8f);

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0f / bc1, 1.0f / std::sqrt(bc2), inv_scale, current_frame);

        struct Adam_Constants
        {
            uint32_t total_elements;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float weight_decay;
        } constants{static_cast<uint32_t>(total_elements), beta1, beta2, epsilon, max_gradient, weight_decay};

        auto dyn_buf = engine.getDynamicOptimizerBuffer(current_frame);
        bool use_fp16 = (param_gpu.getDataType() == Data_Type::FLOAT16 ||
                         Execution_Engine::getInstance().isCooperativeMatrixEnabled() ||
                         param_gpu.getCachedFp16Storage() != nullptr);
        if (use_fp16)
        {
            if (!param_gpu.getCachedFp16Storage() || param_gpu.getCachedFp16Storage()->getSize() < total_elements)
            {
                param_gpu.setCachedFp16Storage(std::make_shared<gpu::vector>(engine.getContext(), total_elements, Data_Type::FLOAT16));
            }
            param_gpu.pushToGraph(Compute_Pipeline::ADAM_UPDATE_FP16,
                                  {param_gpu.getStorageBuffer(), effective_grad->getStorageBuffer(), m_gpu.getStorageBuffer(), v_gpu.getStorageBuffer(), dyn_buf, param_gpu.getCachedFp16Storage()},
                                  constants, (static_cast<uint32_t>(total_elements) + 255) / 256);
            param_gpu.setFp16CacheDirty(false);
        }
        else
        {
            param_gpu.pushToGraph(Compute_Pipeline::ADAM_UPDATE,
                                  {param_gpu.getStorageBuffer(), effective_grad->getStorageBuffer(), m_gpu.getStorageBuffer(), v_gpu.getStorageBuffer(), dyn_buf},
                                  constants, (static_cast<uint32_t>(total_elements) + 255) / 256);
            param_gpu.setFp16CacheDirty(true);
        }
    }
}

} // namespace ops::optimizer
