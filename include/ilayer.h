#pragma once

#include "math/matrix.h"
#include "learning_rate.h"
#include <fstream>
#include <utility>

enum class Layer_Type
{
    LINEAR,
    CONV2D,
    BATCH_NORM,
    MAX_POOL_2D,
    GLOBAL_AVG_POOL_2D,
    RELU,
    GELU,
    SOFTMAX,
    LAYER_TYPE_END
};

class ILayer
{
public:
    virtual ~ILayer() noexcept = default;
    virtual Matrix forward(const Matrix &input_matrix) = 0;
    virtual Matrix backward(const Matrix &gradient_output) = 0;
    virtual Matrix getWeights() const { return Matrix(0, 0); }
    virtual Matrix getBiases() const { return Matrix(0, 0); }
    virtual bool hasParameters() const { return false; }
    virtual void resetGradient() {}
    virtual Matrix getWeightsGradient() {return Matrix(0, 0); }
    virtual std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads() { return {}; }
    virtual Matrix getInput() { return Matrix(0, 0); }
    virtual Layer_Type getLayerType() const = 0;
    virtual void saveConfig(std::ofstream &out_file) const = 0;
    virtual void saveState(std::ofstream &out_file) const = 0;
    virtual void loadState(std::ifstream &in_file) = 0;
}; 