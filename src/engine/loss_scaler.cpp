#include "engine/loss_scaler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"

namespace
{
void applyGradientScale(Tensor &gradient, float factor)
{
    if (gradient.isEmpty())
    {
        return;
    }
    if (gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Loss_Scaler: TERNARY_INT2 gradients are not supported");
    }
    if (gradient.getDataType() == Data_Type::FLOAT16 && gradient.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        auto *gpu_impl = dynamic_cast<Gpu_Tensor_Impl *>(gradient.getImplementation().get());
        if (gpu_impl && gpu_impl->isContiguous() && gpu_impl->getByteOffset() == 0 && gpu_impl->getStorageBuffer())
        {
            struct Fp16_Scale_Constants
            {
                uint32_t total_elements;
                float scalar;
                uint32_t preserve_nonfinite;
            } pc{static_cast<uint32_t>(gradient.getTotalElements()), factor, 1u};

            gpu_impl->pushToGraph(
                Compute_Pipeline::MUL_SCALAR_FP16,
                {gpu_impl->getStorageBuffer(), gpu_impl->getStorageBuffer()},
                pc,
                static_cast<uint32_t>((gradient.getTotalElements() + 255) / 256));
            gpu_impl->invalidateFp16Cache();
            gpu_impl->clearHostCache();
        }
        else
        {
            std::vector<float> vals = gradient.getData();
            for (float &v : vals)
            {
                v *= factor;
            }
            gradient.uploadData(vals);
        }
        return;
    }
    gradient.mulScalar(factor, gradient);
}
}

Loss_Scaler::Loss_Scaler(float _initial_scale,
                         float _growth_factor,
                         float _backoff_factor,
                         uint32_t _growth_interval,
                         bool _enabled,
                         float _max_scale,
                         float _min_scale)
    : scale_factor(_initial_scale),
      growth_factor(_growth_factor),
      backoff_factor(_backoff_factor),
      min_scale(_min_scale),
      max_scale(_max_scale),
      growth_interval(_growth_interval),
      good_steps(0),
      is_enabled(_enabled)
{
    if (!std::isfinite(_initial_scale) || _initial_scale <= 0.0f ||
        !std::isfinite(_growth_factor) || _growth_factor < 1.0f ||
        !std::isfinite(_backoff_factor) || _backoff_factor <= 0.0f || _backoff_factor >= 1.0f ||
        _growth_interval == 0 ||
        !std::isfinite(_min_scale) || _min_scale <= 0.0f ||
        !std::isfinite(_max_scale) || _max_scale < _min_scale)
    {
        throw std::invalid_argument("Loss_Scaler::Loss_Scaler: Invalid loss scaler hyperparameter(s)");
    }
    scale_factor = std::clamp(_initial_scale, min_scale, max_scale);

    Logger::logMessage(Input_Format{"Loss_Scaler::Loss_Scaler: Initialized with scale={:.1f}, growth={:.2f}, backoff={:.2f}, interval={}, min_scale={:.1f}, max_scale={:.1f}",
                                    scale_factor, growth_factor, backoff_factor, growth_interval, min_scale, max_scale},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
}

float Loss_Scaler::scaleLoss(float loss_value) const noexcept
{
    if (!is_enabled || !std::isfinite(loss_value) || !std::isfinite(scale_factor) || scale_factor <= 0.0f)
    {
        return loss_value;
    }
    return loss_value * scale_factor;
}

void Loss_Scaler::scaleGradient(Tensor &gradient) const
{
    if (!is_enabled || !std::isfinite(scale_factor) || scale_factor <= 0.0f || scale_factor == 1.0f)
    {
        return;
    }
    applyGradientScale(gradient, scale_factor);
}

void Loss_Scaler::unscaleGradient(Tensor &gradient) const
{
    if (!is_enabled || !std::isfinite(scale_factor) || scale_factor <= 0.0f || scale_factor == 1.0f)
    {
        return;
    }
    applyGradientScale(gradient, 1.0f / scale_factor);
}

bool Loss_Scaler::hasOverflow(const Tensor &gradient) const
{
    if (gradient.isEmpty())
    {
        return false;
    }

    if (gradient.getExecutionTarget() == Execution_Target::CPU)
    {
        if (const auto *cpu_impl = dynamic_cast<const Cpu_Tensor_Impl *>(gradient.getImplementation().get()))
        {
            if (cpu_impl->getDataType() == Data_Type::TERNARY_INT2)
            {
                return false;
            }
            if (cpu_impl->isContiguous() && cpu_impl->getByteOffset() == 0)
            {
                if (cpu_impl->getDataType() == Data_Type::FLOAT32)
                {
                    if (const auto buf = cpu_impl->getStorageBuffer())
                    {
                        for (float val : *buf)
                        {
                            if (!std::isfinite(val))
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                }
                else if (cpu_impl->getDataType() == Data_Type::FLOAT16)
                {
                    if (const auto buf16 = cpu_impl->getStorageBufferFp16())
                    {
                        static_assert(sizeof(float16_t) == sizeof(uint16_t), "Unexpected float16_t size");
                        for (const float16_t &val : *buf16)
                        {
                            uint16_t bits = 0;
                            std::memcpy(&bits, &val, sizeof(uint16_t));
                            if ((bits & 0x7C00u) == 0x7C00u)
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                }
            }
        }
    }

    const auto &data = gradient.getData();
    for (float val : data)
    {
        if (!std::isfinite(val))
        {
            return true;
        }
    }
    return false;
}

bool Loss_Scaler::hasOverflow(const std::vector<Tensor> &gradients) const
{
    for (const auto &grad : gradients)
    {
        if (hasOverflow(grad))
        {
            return true;
        }
    }
    return false;
}

bool Loss_Scaler::hasOverflow(const std::vector<std::pair<Tensor *, Tensor *>> &param_grad_pairs) const
{
    for (const auto &[param, grad] : param_grad_pairs)
    {
        if (grad && hasOverflow(*grad))
        {
            return true;
        }
    }
    return false;
}

void Loss_Scaler::unscaleGradients(std::vector<std::pair<Tensor *, Tensor *>> &param_grad_pairs) const
{
    for (auto &[param, grad] : param_grad_pairs)
    {
        if (grad)
        {
            unscaleGradient(*grad);
        }
    }
}

bool Loss_Scaler::step(bool overflow_detected)
{
    if (!is_enabled)
    {
        return true;
    }

    if (overflow_detected)
    {
        scale_factor = std::max(scale_factor * backoff_factor, min_scale);
        good_steps = 0;
        Logger::logMessage(Input_Format{"Loss_Scaler::step: Overflow detected! Backed off scale factor to {:.4f}", scale_factor},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
        return false;
    }

    if (growth_interval == 0)
    {
        growth_interval = 1;
    }

    good_steps++;
    if (good_steps >= growth_interval)
    {
        scale_factor = std::min(scale_factor * growth_factor, max_scale);
        good_steps = 0;
        Logger::logMessage(Input_Format{"Loss_Scaler::step: Increased scale factor to {:.4f}", scale_factor},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
    }
    return true;
}

void Loss_Scaler::setGrowthInterval(uint32_t _growth_interval) noexcept
{
    if (_growth_interval == 0)
    {
        Logger::logMessage("Loss_Scaler::setGrowthInterval: Ignoring interval 0, clamping to 1",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
    }
    growth_interval = std::max<uint32_t>(1u, _growth_interval);
}

void Loss_Scaler::setBackoffFactor(float _backoff_factor) noexcept
{
    if (!std::isfinite(_backoff_factor) || _backoff_factor <= 0.0f || _backoff_factor >= 1.0f)
    {
        Logger::logMessage("Loss_Scaler::setBackoffFactor: Ignoring invalid backoff factor outside (0, 1)",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
        return;
    }
    backoff_factor = _backoff_factor;
}

void Loss_Scaler::setGrowthFactor(float _growth_factor) noexcept
{
    if (!std::isfinite(_growth_factor) || _growth_factor < 1.0f)
    {
        Logger::logMessage("Loss_Scaler::setGrowthFactor: Ignoring invalid growth factor < 1.0",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
        return;
    }
    growth_factor = _growth_factor;
}

void Loss_Scaler::setScaleFactor(float _scale_factor) noexcept
{
    if (!std::isfinite(_scale_factor) || _scale_factor <= 0.0f)
    {
        Logger::logMessage("Loss_Scaler::setScaleFactor: Ignoring non-finite or non-positive scale factor",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
        return;
    }
    scale_factor = std::clamp(_scale_factor, min_scale, max_scale);
}

void Loss_Scaler::setMinScale(float _min_scale) noexcept
{
    if (!std::isfinite(_min_scale) || _min_scale <= 0.0f)
    {
        Logger::logMessage("Loss_Scaler::setMinScale: Ignoring non-finite or non-positive min_scale",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
        return;
    }
    min_scale = _min_scale;
    if (max_scale < min_scale)
    {
        max_scale = min_scale;
    }
    scale_factor = std::clamp(scale_factor, min_scale, max_scale);
}

void Loss_Scaler::setMaxScale(float _max_scale) noexcept
{
    if (!std::isfinite(_max_scale) || _max_scale <= 0.0f)
    {
        Logger::logMessage("Loss_Scaler::setMaxScale: Ignoring non-finite or non-positive max_scale",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
        return;
    }
    max_scale = _max_scale;
    if (min_scale > max_scale)
    {
        min_scale = max_scale;
    }
    scale_factor = std::clamp(scale_factor, min_scale, max_scale);
}
