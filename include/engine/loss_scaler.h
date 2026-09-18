#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "helper/logger.h"
#include "math/tensor.h"

class Loss_Scaler
{
private:
    float scale_factor = 1.0f;
    float growth_factor = 2.0f;
    float backoff_factor = 0.5f;
    float min_scale = 1.0f;
    float max_scale = 1024.0f;
    std::uint32_t growth_interval = 2000;
    std::uint32_t good_steps = 0;
    bool is_enabled = true;

public:
    Loss_Scaler(float _initial_scale = 1.0f,
                float _growth_factor = 2.0f,
                float _backoff_factor = 0.5f,
                std::uint32_t _growth_interval = 2000,
                bool _enabled = true,
                float _max_scale = 1024.0f)
        : scale_factor(_initial_scale),
          growth_factor(_growth_factor),
          backoff_factor(_backoff_factor),
          max_scale(_max_scale),
          growth_interval(_growth_interval),
          is_enabled(_enabled)
    {
        Logger::logMessage(Input_Format{"Loss_Scaler::Loss_Scaler: Initialized with scale={:.1f}, growth={:.2f}, backoff={:.2f}, interval={}, max_scale={:.1f}",
                                        scale_factor, growth_factor, backoff_factor, growth_interval, max_scale},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE | Log_Feature::FP16_METRICS);
    }

    float scaleLoss(float loss_value) const noexcept
    {
        if (!is_enabled)
        {
            return loss_value;
        }
        return loss_value * scale_factor;
    }

    void scaleGradient(Tensor &gradient) const
    {
        if (!is_enabled || scale_factor == 1.0f)
        {
            return;
        }
        gradient.mulScalar(scale_factor, gradient);
    }

    void unscaleGradient(Tensor &gradient) const
    {
        if (!is_enabled || scale_factor == 0.0f)
        {
            return;
        }
        gradient.mulScalar(1.0f / scale_factor, gradient);
    }

    bool hasOverflow(const Tensor &gradient) const
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

    bool hasOverflow(const std::vector<Tensor> &gradients) const
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

    bool hasOverflow(const std::vector<std::pair<Tensor *, Tensor *>> &param_grad_pairs) const
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

    void unscaleGradients(std::vector<std::pair<Tensor *, Tensor *>> &param_grad_pairs) const
    {
        for (auto &[param, grad] : param_grad_pairs)
        {
            if (grad)
            {
                unscaleGradient(*grad);
            }
        }
    }

    bool step(bool overflow_detected)
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

    std::uint32_t getGrowthInterval() const noexcept { return growth_interval; }
    std::uint32_t getGoodSteps() const noexcept { return good_steps; }
    float getBackoffFactor() const noexcept { return backoff_factor; }
    float getGrowthFactor() const noexcept { return growth_factor; }
    float getScaleFactor() const noexcept { return scale_factor; }
    float getMinScale() const noexcept { return min_scale; }
    float getMaxScale() const noexcept { return max_scale; }
    bool isEnabled() const noexcept { return is_enabled; }

    void setGrowthInterval(std::uint32_t _growth_interval) noexcept { growth_interval = _growth_interval; }
    void setBackoffFactor(float _backoff_factor) noexcept { backoff_factor = _backoff_factor; }
    void setGrowthFactor(float _growth_factor) noexcept { growth_factor = _growth_factor; }
    void setScaleFactor(float _scale_factor) noexcept { scale_factor = _scale_factor; }
    void setGoodSteps(std::uint32_t _steps) noexcept { good_steps = _steps; }
    void setMinScale(float _min_scale) noexcept { min_scale = _min_scale; }
    void setMaxScale(float _max_scale) noexcept { max_scale = _max_scale; }
    void setEnabled(bool _enabled) noexcept { is_enabled = _enabled; }
};
