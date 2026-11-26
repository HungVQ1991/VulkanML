#include "ops/optimizer/adam_step.h"

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
              float weight_decay,
              const std::shared_ptr<gpu::vector> &dynamic_params_buffer)
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

    if (gradient.getExecutionTarget() != parameter.getExecutionTarget() ||
        first_moment.getExecutionTarget() != parameter.getExecutionTarget() ||
        second_moment.getExecutionTarget() != parameter.getExecutionTarget())
    {
        throw std::invalid_argument("ops::optimizer::adamStep: Execution_Target mismatch");
    }

    if (parameter.getDataType() == Data_Type::TERNARY_INT2 ||
        gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        first_moment.getDataType() == Data_Type::TERNARY_INT2 ||
        second_moment.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::optimizer::adamStep: TERNARY_INT2 is not supported");
    }

    if (!std::isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(learning_rate) || learning_rate < 0.0f ||
        !std::isfinite(max_gradient) || max_gradient < 0.0f ||
        !std::isfinite(weight_decay) || weight_decay < 0.0f ||
        !std::isfinite(inv_scale) || inv_scale < 0.0f)
    {
        throw std::invalid_argument("ops::optimizer::adamStep: Invalid Adam hyperparameter");
    }

    param_impl->validateSameDimensions(*grad_impl);
    param_impl->validateSameDimensions(*m_impl);
    param_impl->validateSameDimensions(*v_impl);

    size_t total_elements = parameter.getTotalElements();
    if (total_elements == 0)
    {
        return;
    }

    size_t effective_t = std::max<size_t>(timestep, 1);
    float bc1 = std::max(1.0f - std::pow(beta1, static_cast<float>(effective_t)), 1e-8f);
    float bc2 = std::max(1.0f - std::pow(beta2, static_cast<float>(effective_t)), 1e-8f);

    if (parameter.getExecutionTarget() == Execution_Target::CPU)
    {
        auto &param_cpu = Cpu_Tensor_Impl::castToCpu(*param_impl);
        auto &m_cpu = Cpu_Tensor_Impl::castToCpu(*m_impl);
        auto &v_cpu = Cpu_Tensor_Impl::castToCpu(*v_impl);

        if (!param_cpu.isContiguous() || param_cpu.getByteOffset() != 0 ||
            !m_cpu.isContiguous() || m_cpu.getByteOffset() != 0 ||
            !v_cpu.isContiguous() || v_cpu.getByteOffset() != 0)
        {
            throw std::invalid_argument("ops::optimizer::adamStep: Non-contiguous parameter or optimizer state view on CPU");
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
        std::vector<float> temp_m;
        std::vector<float> temp_v;

        std::vector<float> *param_vec = loadCpuState(param_cpu, temp_param);
        std::vector<float> *m_vec = loadCpuState(m_cpu, temp_m);
        std::vector<float> *v_vec = loadCpuState(v_cpu, temp_v);

        for (size_t i = 0; i < total_elements; ++i)
        {
            float raw_g = grad_vals[i] * inv_scale;
            if (std::isnan(raw_g) || std::isinf(raw_g))
            {
                continue;
            }
            float g = (max_gradient > 0.0f) ? std::clamp(raw_g, -max_gradient, max_gradient) : raw_g;
            (*m_vec)[i] = beta1 * (*m_vec)[i] + (1.0f - beta1) * g;
            (*v_vec)[i] = beta2 * (*v_vec)[i] + (1.0f - beta2) * (g * g);

            float m_hat = (*m_vec)[i] / bc1;
            float v_hat = (*v_vec)[i] / bc2;
            float step_val = m_hat / (std::sqrt(v_hat) + epsilon);
            if (weight_decay > 0.0f)
            {
                (*param_vec)[i] -= learning_rate * (step_val + weight_decay * (*param_vec)[i]);
            }
            else
            {
                (*param_vec)[i] -= learning_rate * step_val;
            }
        }

        if (param_cpu.getDataType() == Data_Type::FLOAT16)
        {
            storeCpuFp16State(param_cpu, temp_param);
        }
        if (m_cpu.getDataType() == Data_Type::FLOAT16)
        {
            storeCpuFp16State(m_cpu, temp_m);
        }
        if (v_cpu.getDataType() == Data_Type::FLOAT16)
        {
            storeCpuFp16State(v_cpu, temp_v);
        }

        Logger::logMessage(Input_Format{"ops::optimizer::adamStep (CPU): step={}, lr={}",
                                        timestep, learning_rate},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }
    else
    {
        auto &param_gpu = Gpu_Tensor_Impl::castToGpu(*param_impl);
        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto &m_gpu = Gpu_Tensor_Impl::castToGpu(*m_impl);
        auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v_impl);

        if (!param_gpu.isContiguous() || param_gpu.getByteOffset() != 0 ||
            !m_gpu.isContiguous() || m_gpu.getByteOffset() != 0 ||
            !v_gpu.isContiguous() || v_gpu.getByteOffset() != 0)
        {
            throw std::invalid_argument("ops::optimizer::adamStep: Non-contiguous parameter or optimizer state view on GPU");
        }

        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        if (!param_gpu.getStorageBuffer() || !effective_grad->getStorageBuffer() ||
            !m_gpu.getStorageBuffer() || !v_gpu.getStorageBuffer())
        {
            throw std::runtime_error("ops::optimizer::adamStep: Missing GPU storage buffer");
        }

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        std::shared_ptr<gpu::vector> dyn_buf = dynamic_params_buffer;
        if (!dyn_buf)
        {
            dyn_buf = engine.allocateDynamicOptimizerParamsBuffer(learning_rate, 1.0f / bc1, 1.0f / std::sqrt(bc2), inv_scale, current_frame);
        }

        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::optimizer::adamStep: Element count exceeds UINT32_MAX");
        }
        uint32_t total_u32 = static_cast<uint32_t>(total_elements);
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total_elements + 255ull) / 256ull);
        auto [adam_wg_x, adam_wg_y] = engine.getContext().compute2DWorkgroupGrid((total_elements + 1023ull) / 1024ull);

        auto castToFp32 = [&](const std::shared_ptr<gpu::vector> &src) -> std::shared_ptr<gpu::vector>
        {
            auto tmp = std::make_shared<gpu::vector>(engine.getContext(), total_elements, Data_Type::FLOAT32);
            param_gpu.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {src, tmp}, total_u32, wg_x, wg_y, 1);
            return tmp;
        };

        bool param_is_fp16 = (param_gpu.getDataType() == Data_Type::FLOAT16);
        bool grad_is_fp16 = (effective_grad->getDataType() == Data_Type::FLOAT16);
        bool m_is_fp16 = (m_gpu.getDataType() == Data_Type::FLOAT16);
        bool v_is_fp16 = (v_gpu.getDataType() == Data_Type::FLOAT16);

        std::shared_ptr<gpu::vector> param_buf_fp32 = param_is_fp16
            ? castToFp32(param_gpu.getStorageBuffer())
            : param_gpu.getStorageBuffer();
        std::shared_ptr<gpu::vector> grad_buf_fp32 = grad_is_fp16
            ? castToFp32(effective_grad->getStorageBuffer())
            : effective_grad->getStorageBuffer();
        std::shared_ptr<gpu::vector> m_buf_fp32 = m_is_fp16
            ? castToFp32(m_gpu.getStorageBuffer())
            : m_gpu.getStorageBuffer();
        std::shared_ptr<gpu::vector> v_buf_fp32 = v_is_fp16
            ? castToFp32(v_gpu.getStorageBuffer())
            : v_gpu.getStorageBuffer();

        struct Adam_Constants
        {
            uint32_t total_elements;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float weight_decay;
        } constants{total_u32, beta1, beta2, epsilon, max_gradient, weight_decay};

        if (param_is_fp16)
        {
            param_gpu.pushToGraph(Compute_Pipeline::ADAM_UPDATE_FP16,
                                  {param_buf_fp32, grad_buf_fp32, m_buf_fp32, v_buf_fp32, dyn_buf, param_gpu.getStorageBuffer()},
                                  constants, adam_wg_x, adam_wg_y, 1);
            param_gpu.setFp16CacheDirty(true);
        }
        else
        {
            bool use_fp16_cache = (engine.isCooperativeMatrixEnabled() ||
                                   param_gpu.getCachedFp16Storage() != nullptr);
            if (use_fp16_cache)
            {
                auto fp16_buf = param_gpu.ensureCachedFp16Storage(total_elements);
                param_gpu.pushToGraph(Compute_Pipeline::ADAM_UPDATE_FP16,
                                      {param_buf_fp32, grad_buf_fp32, m_buf_fp32, v_buf_fp32, dyn_buf, fp16_buf},
                                      constants, adam_wg_x, adam_wg_y, 1);
                param_gpu.setFp16CacheDirty(false);
            }
            else
            {
                param_gpu.pushToGraph(Compute_Pipeline::ADAM_UPDATE,
                                      {param_buf_fp32, grad_buf_fp32, m_buf_fp32, v_buf_fp32, dyn_buf},
                                      constants, adam_wg_x, adam_wg_y, 1);
                param_gpu.setFp16CacheDirty(true);
            }
        }

        if (m_is_fp16)
        {
            m_gpu.pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16, {m_buf_fp32, m_gpu.getStorageBuffer()}, total_u32, wg_x, wg_y, 1);
        }
        if (v_is_fp16)
        {
            v_gpu.pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16, {v_buf_fp32, v_gpu.getStorageBuffer()}, total_u32, wg_x, wg_y, 1);
        }

        param_gpu.clearHostCache();
        m_gpu.setFp16CacheDirty(true);
        m_gpu.clearHostCache();
        v_gpu.setFp16CacheDirty(true);
        v_gpu.clearHostCache();
    }
}

void adamStepMulti4Fp16(const std::array<Adam_Multi_Item, 4> &items,
                        size_t count,
                        float learning_rate,
                        float beta1,
                        float beta2,
                        float epsilon,
                        size_t timestep,
                        float max_gradient,
                        float inv_scale,
                        float weight_decay,
                        const std::shared_ptr<gpu::vector> &dynamic_params_buffer)
{
    if (count == 0)
    {
        return;
    }
    if (count > 4)
    {
        throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: count cannot exceed 4");
    }
    if (count == 1)
    {
        const auto &it = items[0];
        if (!it.parameter || !it.gradient || !it.first_moment || !it.second_moment)
        {
            throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: Null tensor pointer");
        }
        adamStep(*it.parameter, *it.gradient, *it.first_moment, *it.second_moment,
                 learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay, dynamic_params_buffer);
        return;
    }
    if (!std::isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(learning_rate) || learning_rate < 0.0f ||
        !std::isfinite(max_gradient) || max_gradient < 0.0f ||
        !std::isfinite(weight_decay) || weight_decay < 0.0f ||
        !std::isfinite(inv_scale) || inv_scale < 0.0f)
    {
        throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: Invalid Adam hyperparameter");
    }

    Execution_Engine &engine = Execution_Engine::getInstance();
    uint32_t current_frame = engine.getContext().getCurrentFrame();
    std::shared_ptr<gpu::vector> dyn_buf = dynamic_params_buffer;
    if (!dyn_buf)
    {
        size_t effective_t = std::max<size_t>(timestep, 1);
        float bc1 = std::max(1.0f - std::pow(beta1, static_cast<float>(effective_t)), 1e-8f);
        float bc2 = std::max(1.0f - std::pow(beta2, static_cast<float>(effective_t)), 1e-8f);
        dyn_buf = engine.allocateDynamicOptimizerParamsBuffer(learning_rate, 1.0f / bc1, 1.0f / std::sqrt(bc2), inv_scale, current_frame);
    }

    std::array<uint32_t, 4> counts{0u, 0u, 0u, 0u};
    std::vector<std::shared_ptr<gpu::vector>> buffers;
    buffers.reserve(21);
    buffers.push_back(dyn_buf);

    uint32_t max_elems = 0u;
    std::array<Gpu_Tensor_Impl *, 4> p_gpus{};
    std::array<Gpu_Tensor_Impl *, 4> m_gpus{};
    std::array<Gpu_Tensor_Impl *, 4> v_gpus{};

    for (size_t i = 0; i < count; ++i)
    {
        const auto &it = items[i];
        if (!it.parameter || !it.gradient || !it.first_moment || !it.second_moment ||
            !it.parameter->getImplementation() || !it.gradient->getImplementation() ||
            !it.first_moment->getImplementation() || !it.second_moment->getImplementation())
        {
            throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: Null tensor or implementation");
        }
        auto &p_gpu = Gpu_Tensor_Impl::castToGpu(*it.parameter->getImplementation());
        const auto &g_gpu = Gpu_Tensor_Impl::castToGpu(*it.gradient->getImplementation());
        auto &m_gpu = Gpu_Tensor_Impl::castToGpu(*it.first_moment->getImplementation());
        auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*it.second_moment->getImplementation());

        p_gpu.validateSameDimensions(g_gpu);
        p_gpu.validateSameDimensions(m_gpu);
        p_gpu.validateSameDimensions(v_gpu);

        size_t elems = it.parameter->getTotalElements();
        if (elems == 0 || elems > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: Invalid element count");
        }
        if (p_gpu.getDataType() != Data_Type::FLOAT32 || g_gpu.getDataType() != Data_Type::FLOAT32 ||
            m_gpu.getDataType() != Data_Type::FLOAT32 || v_gpu.getDataType() != Data_Type::FLOAT32)
        {
            throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: Requires FLOAT32 master/grad/moment tensors");
        }
        if (!p_gpu.isContiguous() || p_gpu.getByteOffset() != 0 ||
            !g_gpu.isContiguous() || g_gpu.getByteOffset() != 0 ||
            !m_gpu.isContiguous() || m_gpu.getByteOffset() != 0 ||
            !v_gpu.isContiguous() || v_gpu.getByteOffset() != 0 ||
            !p_gpu.getStorageBuffer() || !g_gpu.getStorageBuffer() ||
            !m_gpu.getStorageBuffer() || !v_gpu.getStorageBuffer())
        {
            throw std::invalid_argument("ops::optimizer::adamStepMulti4Fp16: Requires contiguous non-null GPU buffers");
        }
        auto fp16_buf = p_gpu.ensureCachedFp16Storage(elems);

        uint32_t elems_u32 = static_cast<uint32_t>(elems);
        counts[i] = elems_u32;
        max_elems = std::max(max_elems, elems_u32);
        p_gpus[i] = &p_gpu;
        m_gpus[i] = &m_gpu;
        v_gpus[i] = &v_gpu;

        buffers.push_back(p_gpu.getStorageBuffer());
        buffers.push_back(g_gpu.getStorageBuffer());
        buffers.push_back(m_gpu.getStorageBuffer());
        buffers.push_back(v_gpu.getStorageBuffer());
        buffers.push_back(fp16_buf);
    }

    for (size_t i = count; i < 4; ++i)
    {
        for (size_t b = 1; b <= 5; ++b)
        {
            buffers.push_back(buffers[b]);
        }
    }

    struct Adam_Multi4_Constants
    {
        uint32_t n0;
        uint32_t n1;
        uint32_t n2;
        uint32_t n3;
        float beta1;
        float beta2;
        float eps;
        float max_grad;
        float weight_decay;
    } constants{counts[0], counts[1], counts[2], counts[3], beta1, beta2, epsilon, max_gradient, weight_decay};

    uint64_t wg_x_u64 = (static_cast<uint64_t>(max_elems) + 1023ULL) / 1024ULL;
    const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
    if (wg_x_u64 > max_wg[0] || count > max_wg[2])
    {
        throw std::overflow_error("ops::optimizer::adamStepMulti4Fp16: Workgroup grid exceeds hardware limits");
    }

    p_gpus[0]->pushToGraph(Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16,
                           buffers,
                           constants,
                           static_cast<uint32_t>(wg_x_u64),
                           1u,
                           static_cast<uint32_t>(count));

    for (size_t i = 0; i < count; ++i)
    {
        p_gpus[i]->setFp16CacheDirty(false);
        p_gpus[i]->clearHostCache();
        m_gpus[i]->setFp16CacheDirty(true);
        m_gpus[i]->clearHostCache();
        v_gpus[i]->setFp16CacheDirty(true);
        v_gpus[i]->clearHostCache();
    }
}

}
