#include "engine/loss_scaler.h"

#include <algorithm>
#include <cmath>

#include "helper/logger.h"
#include "math/tensor.h"

Loss_Scaler::Loss_Scaler(float _initial_scale,
                         float _growth_factor,
                         float _backoff_factor,
                         uint32_t _growth_interval,
                         bool _enabled,
                         float _max_scale)
    : scale_factor(_initial_scale),
      growth_factor(_growth_factor),
      backoff_factor(_backoff_factor),
      min_scale(1.0f),
      max_scale(_max_scale),
      growth_interval(_growth_interval),
      good_steps(0),
      is_enabled(_enabled)
{
    Logger::logMessage(Input_Format{"Loss_Scaler::Loss_Scaler: Initialized with scale={:.1f}, growth={:.2f}, backoff={:.2f}, interval={}, max_scale={:.1f}",
                                    scale_factor, growth_factor, backoff_factor, growth_interval, max_scale},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
}

void Loss_Scaler::scaleGradient(Tensor &gradient) const
{
    if (!is_enabled || scale_factor == 1.0f)
    {
        return;
    }
    gradient.mulScalar(scale_factor, gradient);
}

void Loss_Scaler::unscaleGradient(Tensor &gradient) const
{
    if (!is_enabled || scale_factor == 0.0f)
    {
        return;
    }
    gradient.mulScalar(1.0f / scale_factor, gradient);
}

bool Loss_Scaler::hasOverflow(const Tensor &gradient) const
{
    if (gradient.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        return false;
    }
    const auto &data = gradient.getData();
    for (float val : data)
    {
        if (std::isnan(val) || std::isinf(val))
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
