#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ilayer.h"
#include "math/matrix.h"

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

    void setTrainingMode(bool training) override
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

        LAYER_LOG_DEBUG("Batch_Norm_Layer::forward: dim=" + std::to_string(input_dim) + ", mode=" + (is_training ? "Train" : "Eval"));

        inputs = input_matrix;
        if (inputs.getTarget() != target)
        {
            inputs.setExecutionTarget(target);
        }

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

        LAYER_LOG_DEBUG("Batch_Norm_Layer::backward: grad_output rows=" + std::to_string(gradient_output.getRows()) + ", cols=" + std::to_string(gradient_output.getCols()));

        Matrix grad_input = inputs.batchNormBackward(
            gradient_output, gamma, batch_var, x_hat,
            grad_gamma, grad_beta,
            epsilon);

        return grad_input;
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

    Layer_Type getLayerType() const override
    {
        return Layer_Type::BATCH_NORM;
    }

    void saveConfig(std::ofstream &out_file) const override
    {
        std::uint32_t num_features = static_cast<std::uint32_t>(input_dim);
        out_file.write(reinterpret_cast<const char *>(&num_features), sizeof(num_features));
        out_file.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        out_file.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
    }

    void saveInference(std::ofstream &out_file) const override
    {
        gamma.saveMatrix(out_file);
        beta.saveMatrix(out_file);
        running_mean.saveMatrix(out_file);
        running_var.saveMatrix(out_file);
    }

    void loadInference(std::ifstream &in_file) override
    {
        gamma = Matrix::loadMatrix(in_file, target);
        beta = Matrix::loadMatrix(in_file, target);
        running_mean = Matrix::loadMatrix(in_file, target);
        running_var = Matrix::loadMatrix(in_file, target);
    }

    void saveCheckpoint(std::ofstream &out_file) const override
    {
        gamma.saveMatrix(out_file);
        beta.saveMatrix(out_file);
        running_mean.saveMatrix(out_file);
        running_var.saveMatrix(out_file);
        grad_gamma.saveMatrix(out_file);
        grad_beta.saveMatrix(out_file);
    }

    void loadCheckpoint(std::ifstream &in_file) override
    {
        gamma = Matrix::loadMatrix(in_file, target);
        beta = Matrix::loadMatrix(in_file, target);
        running_mean = Matrix::loadMatrix(in_file, target);
        running_var = Matrix::loadMatrix(in_file, target);
        grad_gamma = Matrix::loadMatrix(in_file, target);
        grad_beta = Matrix::loadMatrix(in_file, target);
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads() override
    {
        return {{&gamma, &grad_gamma}, {&beta, &grad_beta}};
    }

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
            return;

        Logger::logMessage("Batch_Norm_Layer::setTarget: Changing execution target from " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(target)) + " to " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(new_target)), LOG_WARNING);

        target = new_target;
        gamma.setExecutionTarget(new_target);
        beta.setExecutionTarget(new_target);
        grad_gamma.setExecutionTarget(new_target);
        grad_beta.setExecutionTarget(new_target);
        running_mean.setExecutionTarget(new_target);
        running_var.setExecutionTarget(new_target);
        batch_mean.setExecutionTarget(new_target);
        batch_var.setExecutionTarget(new_target);
        x_hat.setExecutionTarget(new_target);
        inputs.setExecutionTarget(new_target);
    }
};