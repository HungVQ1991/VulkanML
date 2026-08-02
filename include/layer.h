#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>
#include <utility>
#include <algorithm>

#include "ilayer.h"
#include "math/matrix.h"
#include "math/logger.h"

class Layer : public ILayer
{
public:
    Matrix weights;
    Matrix biases;
    Matrix inputs;
    Matrix outputs;

    Matrix weights_gradient;
    Matrix biases_gradient;

    std::size_t input_dim = 0;
    std::size_t output_dim = 0;
    Execution_Target target = Execution_Target::CPU;

private:
    Matrix clipGradients(const Matrix &gradients, float max_value) const
    {
        if (target == Execution_Target::CPU)
        {
            std::vector<float> grad_data = gradients.getData();
            for (float &val : grad_data)
            {
                val = std::clamp(val, -max_value, max_value);
            }
            return Matrix(gradients.getRows(), gradients.getCols(), std::move(grad_data), target);
        }
        return gradients;
    }

public:
    Layer()
        : weights(0, 0),
          biases(0, 0),
          inputs(0, 0),
          outputs(0, 0),
          weights_gradient(0, 0),
          biases_gradient(0, 0) {}

    Layer(std::size_t input_dimension, std::size_t output_dimension, Execution_Target exec_target = Execution_Target::CPU, float init_gain = 2.0f)
        : weights(0, 0, exec_target),
          biases(0, 0, exec_target),
          inputs(0, 0, exec_target),
          outputs(0, 0, exec_target),
          weights_gradient(input_dimension, output_dimension, exec_target),
          biases_gradient(1, output_dimension, exec_target),
          input_dim(input_dimension),
          output_dim(output_dimension),
          target(exec_target)
    {
        std::vector<float> weight_data(input_dimension * output_dimension);
        std::vector<float> bias_data(output_dimension, 0.0f);

        float stddev = std::sqrt(init_gain / static_cast<float>(input_dimension));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, stddev);

        for (float &w : weight_data)
        {
            w = dist(gen);
        }

        weights = Matrix(input_dimension, output_dimension, std::move(weight_data), target);
        biases = Matrix(1, output_dimension, std::move(bias_data), target);
    }

    ~Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        if (input_matrix.getCols() != input_dim)
        {
            Logger::logMessage("Layer::forward: Input dimension mismatch", LOG_ERROR);
            throw std::invalid_argument("Input dimension mismatch");
        }

        inputs = input_matrix;
        outputs = inputs.matmulAdd(weights, biases);
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (gradient_output.getCols() != output_dim || gradient_output.getRows() != inputs.getRows())
        {
            Logger::logMessage("Layer::backward: Gradient output dimension mismatch", LOG_ERROR);
            throw std::invalid_argument("Gradient output dimension mismatch");
        }

        std::size_t batch_size = inputs.getRows();
        float inv_batch_size = 1.0f / static_cast<float>(batch_size);

        std::vector<float> ones_data(batch_size, 1.0f);
        Matrix ones_t(1, batch_size, std::move(ones_data), target);

        weights_gradient = (inputs.matmulTransA(gradient_output)) * inv_batch_size;
        biases_gradient = (ones_t * gradient_output) * inv_batch_size;
        return gradient_output.matmulTransB(weights);
    }

    void update(float learning_rate, float max_gradient = 1.0f) override
    {
        weights.sgdUpdate(weights_gradient, learning_rate, max_gradient);
        biases.sgdUpdate(biases_gradient, learning_rate, max_gradient);
    }

    Matrix getWeights() const override
    {
        return weights;
    }

    Matrix getBiases() const override
    {
        return biases;
    }

    void setWeights(const Matrix &new_weights)
    {
        if (new_weights.getRows() != input_dim || new_weights.getCols() != output_dim)
        {
            Logger::logMessage("Layer::setWeights: Dimension size of weight must match", LOG_ERROR);
            throw std::invalid_argument("Dimension size of weight must match");
        }
        weights = new_weights;
    }

    void setBiases(const Matrix &new_biases)
    {
        if (new_biases.getRows() != 1 || new_biases.getCols() != output_dim)
        {
            Logger::logMessage("Layer::setBiases: Dimension size of bias must match", LOG_ERROR);
            throw std::invalid_argument("Dimension size of bias must match");
        }
        biases = new_biases;
    }

    void resetGradient() override
    {
        weights_gradient = Matrix(input_dim, output_dim, target);
        biases_gradient = Matrix(1, output_dim, target);
        inputs = Matrix(0, 0, target);
        outputs = Matrix(0, 0, target);
    }

    bool hasParameters() const override
    {
        return true;
    }

    Matrix getWeightsGradient() override
    {
        return weights_gradient;
    }

    Matrix getInput() override { return inputs; }
};