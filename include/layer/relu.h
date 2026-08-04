#pragma once

#include <fstream>
#include <stdexcept>

#include "helper/logger.h"
#include "layer/ilayer.h"
#include "math/matrix.h"

class ReLU : public ILayer
{
private:
    Matrix outputs;
    bool has_forward;
    Execution_Target target;

public:
    explicit ReLU(Execution_Target exec_target = Execution_Target::CPU)
        : outputs(0, 0, exec_target),
          has_forward(false),
          target(exec_target) {}

    ~ReLU() override = default;

    Matrix forward(const Matrix &inputs) override
    {
        outputs = inputs.relu();
        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("Relu::backward: Relu backward called before forward", LOG_ERROR);
            throw std::logic_error("Relu backward called before forward");
        }

        if (gradient_output.getRows() != outputs.getRows() || gradient_output.getCols() != outputs.getCols())
        {
            Logger::logMessage("Relu::backward: Relu gradient dimensions must match output dimensions", LOG_ERROR);
            throw std::invalid_argument("Relu gradient dimensions must match output dimensions");
        }

        return outputs.reluBackward(gradient_output);
    }

    void resetGradient() override
    {
        outputs = Matrix(0, 0, target);
        has_forward = false;
    }

    bool hasParameters() const override
    {
        return false;
    }

    Layer_Type getLayerType() const override
    {
        return Layer_Type::RELU;
    }

    void saveConfig(std::ofstream &out_file) const override {}

    void saveInference(std::ofstream &out_file) const override {}
    void loadInference(std::ifstream &in_file) override {}

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
            return;
        target = new_target;

        outputs.setExecutionTarget(new_target);
    }
};