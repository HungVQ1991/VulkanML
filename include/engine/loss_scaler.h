#pragma once

#include <cstdint>
#include <utility>
#include <vector>

class Tensor;

class Loss_Scaler
{
private:
    float scale_factor = 1.0f;
    float growth_factor = 2.0f;
    float backoff_factor = 0.5f;
    float min_scale = 1.0f;
    float max_scale = 1024.0f;
    uint32_t growth_interval = 2000;
    uint32_t good_steps = 0;
    bool is_enabled = true;

public:
    Loss_Scaler(float _initial_scale = 1.0f,
                float _growth_factor = 2.0f,
                float _backoff_factor = 0.5f,
                uint32_t _growth_interval = 2000,
                bool _enabled = true,
                float _max_scale = 1024.0f);

    float scaleLoss(float loss_value) const noexcept
    {
        if (!is_enabled)
        {
            return loss_value;
        }
        return loss_value * scale_factor;
    }

    void scaleGradient(Tensor &gradient) const;
    void unscaleGradient(Tensor &gradient) const;

    bool hasOverflow(const Tensor &gradient) const;
    bool hasOverflow(const std::vector<Tensor> &gradients) const;
    bool hasOverflow(const std::vector<std::pair<Tensor *, Tensor *>> &param_grad_pairs) const;
    void unscaleGradients(std::vector<std::pair<Tensor *, Tensor *>> &param_grad_pairs) const;

    bool step(bool overflow_detected);

    uint32_t getGrowthInterval() const noexcept { return growth_interval; }
    uint32_t getGoodSteps() const noexcept { return good_steps; }
    float getBackoffFactor() const noexcept { return backoff_factor; }
    float getGrowthFactor() const noexcept { return growth_factor; }
    float getScaleFactor() const noexcept { return scale_factor; }
    float getScale() const noexcept { return scale_factor; }
    float getMinScale() const noexcept { return min_scale; }
    float getMaxScale() const noexcept { return max_scale; }
    bool isEnabled() const noexcept { return is_enabled; }

    void setGrowthInterval(uint32_t _growth_interval) noexcept { growth_interval = _growth_interval; }
    void setBackoffFactor(float _backoff_factor) noexcept { backoff_factor = _backoff_factor; }
    void setGrowthFactor(float _growth_factor) noexcept { growth_factor = _growth_factor; }
    void setScaleFactor(float _scale_factor) noexcept { scale_factor = _scale_factor; }
    void setGoodSteps(uint32_t _steps) noexcept { good_steps = _steps; }
    void setMinScale(float _min_scale) noexcept { min_scale = _min_scale; }
    void setMaxScale(float _max_scale) noexcept { max_scale = _max_scale; }
    void setEnabled(bool _enabled) noexcept { is_enabled = _enabled; }
};
