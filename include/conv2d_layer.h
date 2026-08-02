#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>

#include "learning_rate.h"
#include "ilayer.h"
#include "math/matrix.h"
#include "math/logger.h"

class Conv2d_Layer : public ILayer
{
private:
    std::uint32_t in_h;
    std::uint32_t in_w;
    std::uint32_t in_c;
    std::uint32_t out_c;
    std::uint32_t kernel_size;
    std::uint32_t stride;
    std::uint32_t padding;
    std::uint32_t out_h;
    std::uint32_t out_w;

    Matrix weights;
    Matrix biases;
    Matrix grad_weights;
    Matrix grad_biases;
    Matrix inputs;
    Matrix outputs;

    bool has_forward;
    Execution_Target target;

    void initializeWeights()
    {
        std::size_t num_weights = kernel_size * kernel_size * in_c * out_c;
        std::vector<float> host_weights(num_weights);
        std::vector<float> host_biases(out_c, 0.0f);

        float fan_in = static_cast<float>(kernel_size * kernel_size * in_c);
        float std_dev = std::sqrt(2.0f / fan_in);

        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, std_dev);

        for (std::size_t i = 0; i < num_weights; ++i)
        {
            host_weights[i] = dist(gen);
        }

        weights = Matrix(1, num_weights, host_weights, target);
        biases = Matrix(1, out_c, host_biases, target);
        grad_weights = Matrix(1, num_weights, target);
        grad_biases = Matrix(1, out_c, target);
    }

public:
    Conv2d_Layer(std::uint32_t h, std::uint32_t w, std::uint32_t c_in, std::uint32_t c_out, std::uint32_t k, std::uint32_t s, std::uint32_t p, Execution_Target exec_target = Execution_Target::CPU)
        : in_h(h), in_w(w), in_c(c_in), out_c(c_out), kernel_size(k), stride(s), padding(p), target(exec_target), has_forward(false)
    {
        out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
        initializeWeights();
    }

    ~Conv2d_Layer() override = default;

    Matrix forward(const Matrix &input_matrix) override
    {
        inputs = input_matrix;
        outputs = inputs.conv2d(weights, biases, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
        has_forward = true;
        return outputs;
    }

    Matrix backward(const Matrix &gradient_output) override
    {
        if (!has_forward)
        {
            Logger::logMessage("Conv2d_Layer::backward: Backward called before forward", LOG_ERROR);
            throw std::logic_error("Backward called before forward");
        }

        inputs.conv2dBackwardWeight(gradient_output, grad_weights, grad_biases, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
        return gradient_output.conv2dBackwardInput(weights, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    }

    void update(float learning_rate, float max_gradient = 0.0f) override
    {
        weights.sgdUpdate(grad_weights, learning_rate, max_gradient);
        biases.sgdUpdate(grad_biases, learning_rate, max_gradient);
    }

    void resetGradient() override
    {
        std::size_t num_weights = kernel_size * kernel_size * in_c * out_c;
        grad_weights = Matrix(1, num_weights, target);
        grad_biases = Matrix(1, out_c, target);
        inputs = Matrix(0, 0, target);
        outputs = Matrix(0, 0, target);
        has_forward = false;
    }

    Matrix getWeights() const override
    {
        return weights;
    }

    Matrix getBiases() const override
    {
        return biases;
    }

    bool hasParameters() const override
    {
        return true;
    }
};