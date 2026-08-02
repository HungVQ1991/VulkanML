#pragma once

#include <vector>
#include <stdexcept>
#include <utility>

#include "ilayer.h"
#include "math/matrix.h"
#include "math/logger.h"

class Batch_Norm_Layer : public ILayer
{
private:
    std::size_t input_dim;
    float epsilon;
    float momentum;
    bool is_training;

    Matrix gamma;
    Matrix beta;
    Matrix grad_gamma;
    Matrix grad_beta;

    Matrix running_mean;
    Matrix running_var;
    Matrix batch_mean;
    Matrix batch_var;
    Matrix x_hat;

    Matrix inputs;
    bool has_forward;
    Execution_Target target;

    void initializeParameters()
    {
        std::vector<float> gamma_data(input_dim, 1.0f);
        std::vector<float> beta_data(input_dim, 0.0f);
        std::vector<float> mean_data(input_dim, 0.0f);
        std::vector<float> var_data(input_dim, 1.0f);

        gamma = Matrix(1, input_dim, std::move(gamma_data), target);
        beta = Matrix(1, input_dim, std::move(beta_data), target);
        grad_gamma = Matrix(1, input_dim, target);
        grad_beta = Matrix(1, input_dim, target);

        running_mean = Matrix(1, input_dim, std::move(mean_data), target);
        running_var = Matrix(1, input_dim, std::move(var_data), target);

        batch_mean = Matrix(0, 0, target);
        batch_var = Matrix(0, 0, target);
        x_hat = Matrix(0, 0, target);
    }

public:
    Batch_Norm_Layer(std::size_t dimension, float eps = 1e-5f, float mom = 0.1f, Execution_Target exec_target = Execution_Target::CPU)
        : input_dim(dimension),
          epsilon(eps),
          momentum(mom),
          is_training(true),
          gamma(0, 0, exec_target),
          beta(0, 0, exec_target),
          grad_gamma(0, 0, exec_target),
          grad_beta(0, 0, exec_target),
          running_mean(0, 0, exec_target),
          running_var(0, 0, exec_target),
          batch_mean(0, 0, exec_target),
          batch_var(0, 0, exec_target),
          x_hat(0, 0, exec_target),
          inputs(0, 0, exec_target),
          has_forward(false),
          target(exec_target)
    {
        initializeParameters();
    }

    ~Batch_Norm_Layer() override = default;

    void setTrainingMode(bool training)
    {
        is_training = training;
    }

    Matrix forward(const Matrix &input_matrix) override
    {
        if (input_matrix.getCols() != input_dim)
        {
            Logger::logMessage("Batch_Norm_Layer::forward: Input dimension mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Input dimension mismatch");
        }

        inputs = input_matrix;
        Matrix outputs = inputs.batchNormForward(
            gamma, beta,
            running_mean, running_var,
            batch_mean, batch_var, x_hat,
            epsilon, momentum, is_training);

        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("Batch_Norm_Layer::backward: Backward called before forward", LOG_ERROR, true);
            throw std::logic_error("Backward called before forward");
        }

        return inputs.batchNormBackward(
            gradient_output, gamma, batch_var, x_hat,
            grad_gamma, grad_beta,
            epsilon);
    }

    void update(float learning_rate, float max_gradient = 1.0f) override
    {
        gamma.sgdUpdate(grad_gamma, learning_rate, max_gradient);
        beta.sgdUpdate(grad_beta, learning_rate, max_gradient);
    }

    void resetGradient() override
    {
        grad_gamma = Matrix(1, input_dim, target);
        grad_beta = Matrix(1, input_dim, target);
        inputs = Matrix(0, 0, target);
        batch_mean = Matrix(0, 0, target);
        batch_var = Matrix(0, 0, target);
        x_hat = Matrix(0, 0, target);
        has_forward = false;
    }

    Matrix getWeights() const override
    {
        return gamma;
    }

    Matrix getBiases() const override
    {
        return beta;
    }

    Matrix getWeightsGradient() override
    {
        return grad_gamma;
    }

    Matrix getInput() override
    {
        return inputs;
    }

    bool hasParameters() const override
    {
        return true;
    }
};