#pragma once

#include <fstream>
#include <stdexcept>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "layer/ilayer.h"
#include "math/matrix.h"

class GeLU : public ILayer
{
private:
    Matrix outputs;
    Matrix inputs;
    bool has_forward;
    Execution_Target target;

public:
    GeLU(Execution_Target exec_target = Execution_Target::CPU)
        : target(exec_target), inputs(0, 0, exec_target), outputs(0, 0, exec_target), has_forward(false) {}

    ~GeLU() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        inputs = input_matrix;
        outputs = input_matrix.gelu();
        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("GeLU::backward: GeLU backward called before forward", LOG_ERROR);
            throw std::logic_error("GeLU backward called before forward");
        }

        if (gradient_output.getRows() != outputs.getRows() || gradient_output.getCols() != outputs.getCols())
        {
            Logger::logMessage("GeLU::backward: GeLU gradient dimensions must match output dimensions", LOG_ERROR);
            throw std::invalid_argument("GeLU gradient dimensions must match output dimensions");
        }

        return inputs.geluBackward(gradient_output);
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
        return Layer_Type::GELU;
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

        inputs.setExecutionTarget(new_target);
        outputs.setExecutionTarget(new_target);
    }
};