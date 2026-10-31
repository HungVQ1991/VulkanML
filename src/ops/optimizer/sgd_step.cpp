#include "ops/optimizer/sgd_step.h"

#include <algorithm>
#include <cmath>

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
             float inv_scale)
{
    auto param_impl = parameter.getImplementation();
    auto grad_impl = gradient.getImplementation();

    if (!param_impl || !grad_impl)
    {
        Logger::logMessage("ops::optimizer::sgdStep: Null implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::OPTIMIZER_STEP);
        return;
    }

    param_impl->validateSameDimensions(*grad_impl);
    size_t total_elements = parameter.getTotalElements();
    if (total_elements == 0)
    {
        return;
    }

    if (parameter.getExecutionTarget() == Execution_Target::CPU)
    {
        auto &param_cpu = Cpu_Tensor_Impl::castToCpu(*param_impl);
        const auto &grad_data = gradient.getData();

        if (!param_cpu.getStorageBuffer())
        {
            param_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        auto &param_buf = *param_cpu.getStorageBuffer();

        for (size_t i = 0; i < total_elements; ++i)
        {
            float g = grad_data[i] * inv_scale;
            if (std::isnan(g) || std::isinf(g))
            {
                continue;
            }
            if (max_gradient > 0.0f)
            {
                g = std::clamp(g, -max_gradient, max_gradient);
            }
            param_buf[i] -= learning_rate * g;
        }

        Logger::logMessage(Input_Format{"ops::optimizer::sgdStep (CPU): elements={}, lr={}, max_grad={}",
                                        total_elements, learning_rate, max_gradient},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }
    else
    {
        auto &param_gpu = Gpu_Tensor_Impl::castToGpu(*param_impl);
        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0f, 1.0f, inv_scale, current_frame);
        auto dyn_buf = engine.getDynamicOptimizerBuffer(current_frame);

        struct SGD_Constants
        {
            uint32_t total_elements;
            float max_gradient;
        } constants{static_cast<uint32_t>(total_elements), max_gradient};

        param_gpu.pushToGraph(Compute_Pipeline::SGD_UPDATE,
                              {param_gpu.getStorageBuffer(), effective_grad->getStorageBuffer(), dyn_buf},
                              constants, (static_cast<uint32_t>(total_elements) + 255) / 256);
        param_gpu.setFp16CacheDirty(true);
    }
}

} // namespace ops::optimizer
