#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilayer.h"
#include "math/matrix.h"

class MaxPool2d_Layer : public ILayer
{
private:
    std::uint32_t in_h;
    std::uint32_t in_w;
    std::uint32_t channels;
    std::uint32_t out_h;
    std::uint32_t out_w;
    std::uint32_t kernel_size;
    std::uint32_t stride;
    std::uint32_t padding;

    Matrix inputs;
    Matrix outputs;
    Matrix mask;
    Matrix grad_input;

    bool has_forward;
    Execution_Target target;

    void ensureBuffers(std::size_t batch_size)
    {
        std::size_t out_dim = out_h * out_w * channels;
        std::size_t in_dim = in_h * in_w * channels;

        if (outputs.getRows() != batch_size || outputs.getCols() != out_dim)
        {
            outputs = Matrix(batch_size, out_dim, target);
            mask = Matrix(batch_size, out_dim, target);
        }

        if (grad_input.getRows() != batch_size || grad_input.getCols() != in_dim)
        {
            grad_input = Matrix(batch_size, in_dim, target);
        }
    }

public:
    MaxPool2d_Layer(
        std::uint32_t h, std::uint32_t w, std::uint32_t c,
        std::uint32_t k, std::uint32_t s, std::uint32_t p,
        Execution_Target exec_target = Execution_Target::CPU)
        : in_h(h), in_w(w), channels(c), kernel_size(k), stride(s), padding(p),
          target(exec_target),
          inputs(0, 0, exec_target),
          outputs(0, 0, exec_target),
          mask(0, 0, exec_target),
          grad_input(0, 0, exec_target),
          has_forward(false)
    {
        out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    }

    ~MaxPool2d_Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        LAYER_LOG_DEBUG("MaxPool2d_Layer::forward: in_h=" + std::to_string(in_h) + ", in_w=" + std::to_string(in_w) + ", channels=" + std::to_string(channels));

        inputs = input_matrix;
        std::size_t batch_size = inputs.getRows();
        ensureBuffers(batch_size);

        inputs.maxpool2d(outputs, mask, in_h, in_w, channels, kernel_size, stride, padding);
        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("MaxPool2d_Layer::backward: Backward called before forward", LOG_ERROR, true);
            throw std::logic_error("Backward called before forward");
        }

        LAYER_LOG_DEBUG("MaxPool2d_Layer::backward: grad_output rows=" + std::to_string(gradient_output.getRows()) + ", cols=" + std::to_string(gradient_output.getCols()));

        gradient_output.maxpool2dBackward(mask, grad_input, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding);
        return grad_input;
    }

    void resetGradient() override
    {
        inputs = Matrix(0, 0, target);
        has_forward = false;
    }

    bool hasParameters() const override
    {
        return false;
    }

    Layer_Type getLayerType() const override
    {
        return Layer_Type::MAX_POOL_2D;
    }

    Matrix getOutput() override
    {
        return outputs;
    }

    void saveConfig(std::ofstream &out_file) const override
    {
        out_file.write(reinterpret_cast<const char *>(&in_h), sizeof(in_h));
        out_file.write(reinterpret_cast<const char *>(&in_w), sizeof(in_w));
        out_file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
        out_file.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
        out_file.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
        out_file.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
    }

    void saveInference(std::ofstream &out_file) const override {}
    void loadInference(std::ifstream &in_file) override {}

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
        {
            return;
        }

        Logger::logMessage("Maxpool2d_Layer::setTarget: Changing execution target from " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(target)) + " to " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(new_target)), LOG_WARNING);

        target = new_target;
        inputs.setExecutionTarget(new_target);
        outputs.setExecutionTarget(new_target);
        mask.setExecutionTarget(new_target);
        grad_input.setExecutionTarget(new_target);
    }
};