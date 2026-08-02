#pragma once

#include "math/matrix.h"

class ILayer
{
public:
    virtual ~ILayer() noexcept = default;

    virtual Matrix forward(const Matrix &input_matrix) = 0;
    virtual Matrix backward(const Matrix &gradient_output) = 0;
    virtual void update(float learning_rate, float max_gradient = 1.0f) = 0;
    virtual Matrix getWeights() const { return Matrix(0, 0); }
    virtual Matrix getBiases() const { return Matrix(0, 0); }
    virtual bool hasParameters() const { return false; }
    virtual void resetGradient() {}
    virtual Matrix getWeightsGradient() {return Matrix(0, 0); }
    virtual Matrix getInput() { return Matrix(0, 0); }
}; 