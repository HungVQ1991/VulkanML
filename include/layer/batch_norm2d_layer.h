#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ilayer.h"
#include "math/matrix.h"

class Batch_Norm2d_Layer : public ILayer
{
private:
    std::uint32_t in_h;
    std::uint32_t in_w;
    std::uint32_t channels;
    float epsilon;
    float momentum;
    bool is_training = true;

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
    Matrix outputs;
    Matrix grad_input;

    bool has_forward = false;
    Execution_Target target;

    void ensureBuffers(std::size_t batch_size)
    {
        std::size_t total_dim = in_h * in_w * channels;

        if (outputs.getRows() != batch_size || outputs.getCols() != total_dim)
        {
            outputs = Matrix(batch_size, total_dim, target);
            x_hat = Matrix(batch_size, total_dim, target);
        }

        if (grad_input.getRows() != batch_size || grad_input.getCols() != total_dim)
        {
            grad_input = Matrix(batch_size, total_dim, target);
        }
    }

public:
    Batch_Norm2d_Layer(
        std::uint32_t h, std::uint32_t w, std::uint32_t c,
        float eps = 1e-5f, float mom = 0.1f,
        Execution_Target exec_target = Execution_Target::CPU)
        : in_h(h), in_w(w), channels(c), epsilon(eps), momentum(mom),
          target(exec_target),
          gamma(1, c, std::vector<float>(c, 1.0f), exec_target),
          beta(1, c, std::vector<float>(c, 0.0f), exec_target),
          grad_gamma(1, c, exec_target),
          grad_beta(1, c, exec_target),
          running_mean(1, c, std::vector<float>(c, 0.0f), exec_target),
          running_var(1, c, std::vector<float>(c, 1.0f), exec_target),
          batch_mean(1, c, exec_target),
          batch_var(1, c, exec_target),
          inputs(0, 0, exec_target),
          outputs(0, 0, exec_target),
          x_hat(0, 0, exec_target),
          grad_input(0, 0, exec_target)
    {}

    ~Batch_Norm2d_Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        inputs = input_matrix;
        std::size_t batch_size = inputs.getRows();
        ensureBuffers(batch_size);

        inputs.batchNorm2dForward(
            gamma, beta,
            running_mean, running_var,
            batch_mean, batch_var, x_hat,
            outputs,
            in_h, in_w, channels,
            epsilon, momentum, is_training);

        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("Batch_Norm2d_Layer::backward: Backward called before forward", LOG_ERROR, true);
            throw std::logic_error("Backward called before forward");
        }

        gradient_output.batchNorm2dBackward(
            gamma, batch_var, x_hat,
            grad_gamma, grad_beta,
            grad_input,
            in_h, in_w, channels,
            epsilon);

        return grad_input;
    }

    void resetGradient() override
    {
        inputs = Matrix(0, 0, target);
        has_forward = false;
    }

    bool hasParameters() const override
    {
        return true;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads() override
    {
        return {{&gamma, &grad_gamma}, {&beta, &grad_beta}};
    }

    Layer_Type getLayerType() const override
    {
        return Layer_Type::BATCH_NORM_2D;
    }

    Matrix getOutput() override
    {
        return outputs;
    }

    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
        {
            return;
        }

        target = new_target;
        gamma.setExecutionTarget(new_target);
        beta.setExecutionTarget(new_target);
        grad_gamma.setExecutionTarget(new_target);
        grad_beta.setExecutionTarget(new_target);
        running_mean.setExecutionTarget(new_target);
        running_var.setExecutionTarget(new_target);
        batch_mean.setExecutionTarget(new_target);
        batch_var.setExecutionTarget(new_target);
        inputs.setExecutionTarget(new_target);
        outputs.setExecutionTarget(new_target);
        x_hat.setExecutionTarget(new_target);
        grad_input.setExecutionTarget(new_target);
    }

    void saveConfig(std::ofstream &out_file) const override
    {
        out_file.write(reinterpret_cast<const char *>(&in_h), sizeof(in_h));
        out_file.write(reinterpret_cast<const char *>(&in_w), sizeof(in_w));
        out_file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
        out_file.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        out_file.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
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
};