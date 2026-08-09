#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilayer.h"
#include "math/matrix.h"

class GlobalAvgPool2d_Layer : public ILayer
{
private:
    std::uint32_t in_h;
    std::uint32_t in_w;
    std::uint32_t channels;

    Matrix inputs;
    Matrix outputs;

    bool has_forward;
    Execution_Target target;

public:
    GlobalAvgPool2d_Layer(std::uint32_t h, std::uint32_t w, std::uint32_t c, Execution_Target exec_target = Execution_Target::CPU)
        : in_h(h), in_w(w), channels(c), target(exec_target), inputs(0, 0, exec_target), outputs(0, 0, exec_target), has_forward(false) {}

    ~GlobalAvgPool2d_Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        LAYER_LOG_DEBUG("GlobalAvgPool2d_Layer::forward: in_h=" + std::to_string(in_h) + ", in_w=" + std::to_string(in_w) + ", channels=" + std::to_string(channels));

        inputs = input_matrix;
        outputs = inputs.globalAvgPool2d(in_h, in_w, channels);
        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("GlobalAvgPool2d_Layer::backward: Backward called before forward", LOG_ERROR, true);
            throw std::logic_error("Backward called before forward");
        }

        LAYER_LOG_DEBUG("GlobalAvgPool2d_Layer::backward: grad_output rows=" + std::to_string(gradient_output.getRows()) + ", cols=" + std::to_string(gradient_output.getCols()));

        return gradient_output.globalAvgPool2dBackward(in_h, in_w, channels);
    }

    void resetGradient() override
    {
        inputs = Matrix(0, 0, target);
        outputs = Matrix(0, 0, target);
        has_forward = false;
    }

    bool hasParameters() const override
    {
        return false;
    }

    Layer_Type getLayerType() const override
    {
        return Layer_Type::GLOBAL_AVG_POOL_2D;
    }

    void saveConfig(std::ofstream &out_file) const override
    {
        out_file.write(reinterpret_cast<const char *>(&in_h), sizeof(in_h));
        out_file.write(reinterpret_cast<const char *>(&in_w), sizeof(in_w));
        out_file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    }

    void saveInference(std::ofstream &out_file) const override {}
    void loadInference(std::ifstream &in_file) override {}

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
            return;

        Logger::logMessage("GlobalAvgPool2d_Layer::setTarget: Changing execution target from " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(target)) + " to " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(new_target)), LOG_WARNING);
        
        target = new_target;
        inputs.setExecutionTarget(new_target);
        outputs.setExecutionTarget(new_target);
    }
};