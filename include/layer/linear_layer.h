#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ilayer.h"
#include "math/matrix.h"

class Linear_Layer : public ILayer
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

public:
    Linear_Layer()
        : weights(0, 0),
          biases(0, 0),
          inputs(0, 0),
          outputs(0, 0),
          weights_gradient(0, 0),
          biases_gradient(0, 0) {}

    Linear_Layer(std::size_t input_dimension, std::size_t output_dimension, Execution_Target exec_target = Execution_Target::CPU, float init_gain = 2.0f)
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

        float sum_w = 0.0f;
        for (float &w : weight_data)
        {
            w = dist(gen);
            sum_w += std::abs(w);
        }

        float mean_w = weight_data.empty() ? 0.0f : sum_w / static_cast<float>(weight_data.size());
        Logger::logMessage(std::format("DEBUG_INIT: Layer weights mean_abs = {:.8f}", mean_w), LOG_DEBUG);

        weights = Matrix(input_dimension, output_dimension, std::move(weight_data), target);
        biases = Matrix(1, output_dimension, std::move(bias_data), target);
    }

    ~Linear_Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        if (input_matrix.getCols() != input_dim)
        {
            Logger::logMessage("Linear_Layer::forward: Input dimension mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Input dimension mismatch");
        }

        LAYER_LOG_DEBUG("Linear_Layer::forward: batch_size=" + std::to_string(input_matrix.getRows()) + ", in_dim=" + std::to_string(input_dim) + ", out_dim=" + std::to_string(output_dim));

        inputs = input_matrix;
        outputs = Matrix(inputs.getRows(), output_dim, target);
        inputs.linearForward(weights, biases, outputs);
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (gradient_output.getCols() != output_dim || gradient_output.getRows() != inputs.getRows())
        {
            Logger::logMessage("Linear_Layer::backward: Gradient output dimension mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Gradient output dimension mismatch");
        }

        LAYER_LOG_DEBUG("Linear_Layer::backward: grad_output rows=" + std::to_string(gradient_output.getRows()) + ", cols=" + std::to_string(gradient_output.getCols()));

        std::size_t batch_size = inputs.getRows();

        inputs.linearBackwardWeightBias(gradient_output, weights_gradient, biases_gradient);

        Matrix grad_input(batch_size, input_dim, target);
        gradient_output.linearBackwardInput(weights, grad_input);
        return grad_input;
    }

    Matrix getWeights() const override { return weights; }

    Matrix getBiases() const override { return biases; }

    void setWeights(const Matrix &new_weights)
    {
        if (new_weights.getRows() != input_dim || new_weights.getCols() != output_dim)
        {
            Logger::logMessage("Linear_Layer::setWeights: Dimension size of weight must match", LOG_ERROR, true);
            throw std::invalid_argument("Dimension size of weight must match");
        }
        weights = new_weights;
    }

    void setBiases(const Matrix &new_biases)
    {
        if (new_biases.getRows() != 1 || new_biases.getCols() != output_dim)
        {
            Logger::logMessage("Linear_Layer::setBiases: Dimension size of bias must match", LOG_ERROR, true);
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

    bool hasParameters() const override { return true; }
    Matrix getWeightsGradient() override { return weights_gradient; }
    Matrix getInput() override { return inputs; }

    Layer_Type getLayerType() const override { return Layer_Type::LINEAR; }

    void saveConfig(std::ofstream &out_file) const override
    {
        std::uint32_t in_dim_val = static_cast<std::uint32_t>(input_dim);
        std::uint32_t out_dim_val = static_cast<std::uint32_t>(output_dim);
        out_file.write(reinterpret_cast<const char *>(&in_dim_val), sizeof(in_dim_val));
        out_file.write(reinterpret_cast<const char *>(&out_dim_val), sizeof(out_dim_val));
    }

    void saveInference(std::ofstream &out_file) const override
    {
        weights.saveMatrix(out_file);
        biases.saveMatrix(out_file);
    }

    void loadInference(std::ifstream &in_file) override
    {
        weights = Matrix::loadMatrix(in_file, target);
        biases = Matrix::loadMatrix(in_file, target);
        input_dim = weights.getRows();
        output_dim = weights.getCols();
    }

    void saveCheckpoint(std::ofstream &out_file) const override
    {
        weights.saveMatrix(out_file);
        biases.saveMatrix(out_file);
        weights_gradient.saveMatrix(out_file);
        biases_gradient.saveMatrix(out_file);
    }

    void loadCheckpoint(std::ifstream &in_file) override
    {
        weights = Matrix::loadMatrix(in_file, target);
        biases = Matrix::loadMatrix(in_file, target);
        weights_gradient = Matrix::loadMatrix(in_file, target);
        biases_gradient = Matrix::loadMatrix(in_file, target);
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads() override { return {{&weights, &weights_gradient}, {&biases, &biases_gradient}}; }

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
            return;

        Logger::logMessage("Linear_Layer::setTarget: Changing execution target from " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(target)) + " to " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(new_target)), LOG_WARNING);
        
        target = new_target;
        weights.setExecutionTarget(new_target);
        biases.setExecutionTarget(new_target);
        weights_gradient.setExecutionTarget(new_target);
        biases_gradient.setExecutionTarget(new_target);
    }
};