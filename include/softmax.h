#pragma once

#include "layer.h"
#include "math/matrix.h"

class Softmax : public ILayer
{
private:
    Matrix cached_output;
    bool is_fused_with_loss;
    Execution_Target target = Execution_Target::CPU;

public:
    explicit Softmax(bool fused = false, Execution_Target exec_target = Execution_Target::CPU)
        : cached_output(0, 0, exec_target), is_fused_with_loss(fused), target(exec_target) {}

    ~Softmax() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        cached_output = input_matrix.softmax();
        return cached_output;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (is_fused_with_loss)
            return gradient_output;
        return cached_output.softmaxBackward(gradient_output);
    }

    void update(float learning_rate, float max_gradient = 1.0f) override {}

    void resetGradient() override { cached_output = Matrix(0, 0, target); }

    bool hasParameters() const override { return false; }
};