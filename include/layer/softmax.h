#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include "layer/ilayer.h"
#include "math/matrix.h"

class Softmax : public ILayer
{
private:
    Matrix inputs;
    Matrix cached_output;
    bool is_fused_with_loss;
    Execution_Target target;

public:
    explicit Softmax(bool fused = false, Execution_Target exec_target = Execution_Target::CPU)
        : inputs(0, 0, exec_target), cached_output(0, 0, exec_target), is_fused_with_loss(fused), target(exec_target) {}

    ~Softmax() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        LAYER_LOG_DEBUG("Softmax::forward: rows=" + std::to_string(input_matrix.getRows()) + ", cols=" + std::to_string(input_matrix.getCols()));

        inputs = input_matrix;
        cached_output = input_matrix.softmax();
        return cached_output;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        LAYER_LOG_DEBUG("Softmax::backward: grad_output rows=" + std::to_string(gradient_output.getRows()) + ", cols=" + std::to_string(gradient_output.getCols()) + ", fused=" + (is_fused_with_loss ? "true" : "false"));

        if (is_fused_with_loss)
            return gradient_output;
        return cached_output.softmaxBackward(gradient_output);
    }

    void resetGradient() override
    {
        inputs = Matrix(0, 0, target);
        cached_output = Matrix(0, 0, target);
    }

    bool hasParameters() const override
    {
        return false;
    }

    Layer_Type getLayerType() const override
    {
        return Layer_Type::SOFTMAX;
    }

    void saveConfig(std::ofstream &out_file) const override
    {
        std::uint8_t fused_val = is_fused_with_loss ? 1 : 0;
        out_file.write(reinterpret_cast<const char *>(&fused_val), sizeof(fused_val));
    }

    void saveInference(std::ofstream &out_file) const override {}
    void loadInference(std::ifstream &in_file) override {}

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
            return;

        Logger::logMessage("Softmax::setTarget: Changing execution target from " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(target)) + " to " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(new_target)), LOG_WARNING);

        target = new_target;
        inputs.setExecutionTarget(new_target);
        cached_output.setExecutionTarget(new_target);
    }

    Matrix getInput() override { return inputs; }
    Matrix getOutput() override { return cached_output; }
};