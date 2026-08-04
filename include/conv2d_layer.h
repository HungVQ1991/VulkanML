#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
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
    Matrix weights_gradient;
    Matrix biases_gradient;
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
        weights_gradient = Matrix(1, num_weights, target);
        biases_gradient = Matrix(1, out_c, target);
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

        inputs.conv2dBackwardWeight(gradient_output, weights_gradient, biases_gradient, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
        return gradient_output.conv2dBackwardInput(weights, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    }


    void resetGradient() override
    {
        std::size_t num_weights = kernel_size * kernel_size * in_c * out_c;
        weights_gradient = Matrix(1, num_weights, target);
        biases_gradient = Matrix(1, out_c, target);
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

    Layer_Type getLayerType() const override
    {
        return Layer_Type::CONV2D;
    }

    void saveConfig(std::ofstream &out_file) const override
    {
        out_file.write(reinterpret_cast<const char *>(&in_h), sizeof(in_h));
        out_file.write(reinterpret_cast<const char *>(&in_w), sizeof(in_w));
        out_file.write(reinterpret_cast<const char *>(&in_c), sizeof(in_c));
        out_file.write(reinterpret_cast<const char *>(&out_c), sizeof(out_c));
        out_file.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
        out_file.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
        out_file.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
    }

    void saveState(std::ofstream &out_file) const override
    {
        weights.saveMatrix(out_file);
        biases.saveMatrix(out_file);
    }

    void loadState(std::ifstream &in_file) override
    {
        weights = Matrix::loadMatrix(in_file, target);
        biases = Matrix::loadMatrix(in_file, target);
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads() override { return {{&weights, &weights_gradient}, {&biases, &biases_gradient}}; }
    void setTarget(Execution_Target new_target) override
    {
        if (target == new_target)
            return;
        target = new_target;

        weights.setExecutionTarget(new_target);
        biases.setExecutionTarget(new_target);
        weights_gradient.setExecutionTarget(new_target);
        biases_gradient.setExecutionTarget(new_target);
    }
};