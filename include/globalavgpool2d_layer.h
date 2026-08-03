#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "ilayer.h"
#include "math/matrix.h"
#include "math/logger.h"

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
        : in_h(h), in_w(w), channels(c), target(exec_target), has_forward(false) {}

    ~GlobalAvgPool2d_Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        inputs = input_matrix;
        outputs = inputs.globalAvgPool2d(in_h, in_w, channels);
        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("GlobalAvgPool2d_Layer::backward: Backward called before forward", LOG_ERROR);
            throw std::logic_error("Backward called before forward");
        }

        return gradient_output.globalAvgPool2dBackward(in_h, in_w, channels);
    }

    void resetGradient() override
    {
        inputs = Matrix(0, 0, target);
        outputs = Matrix(0, 0, target);
        has_forward = false;
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

    void saveState(std::ofstream &out_file) const override {}
    void loadState(std::ifstream &in_file) override {}
};