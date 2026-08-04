#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <fstream>

#include "ilayer.h"
#include "math/matrix.h"
#include "math/logger.h"

class ReLU : public ILayer
{
private:
    Matrix outputs;
    bool has_forward = false;
    Execution_Target target = Execution_Target::CPU;

public:
    explicit ReLU(Execution_Target exec_target = Execution_Target::CPU)
        : outputs(0, 0, exec_target),
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

    Layer_Type getLayerType() const override { return Layer_Type::RELU; }
    void saveConfig(std::ofstream &out_file) const override {}
    void saveState(std::ofstream &out_file) const override {}
    void loadState(std::ifstream &in_file) override {}
    void setTarget(Execution_Target _target) override { target = _target; }
};