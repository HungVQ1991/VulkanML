#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cpu_matrix_impl.h"
#include "gpu_matrix_impl.h"
#include "helper/logger.h"
#include "impl.h"

enum class Execution_Target
{
    CPU,
    VULKAN_GPU
};

class Matrix
{
private:
    std::shared_ptr<Impl> pimpl;
    Execution_Target current_target;

public:
    explicit Matrix(Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(0, 0);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(0, 0);
        }
    }

    Matrix(std::size_t rows, std::size_t cols, Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        }
    }

    Matrix(std::size_t rows, std::size_t cols, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols, host_data);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols, host_data);
        }
    }

    Matrix(std::size_t rows, std::size_t cols, std::vector<float> &&host_data, Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(host_data));
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols, std::move(host_data));
        }
    }

    explicit Matrix(std::shared_ptr<Impl> implementation, Execution_Target target = Execution_Target::CPU)
        : pimpl(std::move(implementation)), current_target(target) {}

    ~Matrix() = default;
    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;
    Matrix(Matrix &&) noexcept = default;
    Matrix &operator=(Matrix &&) noexcept = default;

    void initShape(std::size_t rows, std::size_t cols)
    {
        if (pimpl->getRows() == rows && pimpl->getCols() == cols)
        {
            return;
        }

        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        }
    }

    void print(std::size_t max_display_rows = 10, std::size_t max_display_cols = 10) const
    {
        const auto &data_vec = getData();
        std::size_t print_rows = std::min(getRows(), max_display_rows);
        std::size_t print_cols = std::min(getCols(), max_display_cols);
    
        std::cout << "Matrix (" << getRows() << "x" << getCols() << "):\n";
    
        for (std::size_t r = 0; r < print_rows; ++r)
        {
            std::cout << "  [ ";
            for (std::size_t c = 0; c < print_cols; ++c)
            {
                std::cout << std::format("{:8.4f} ", data_vec[r * getCols() + c]);
            }
            if (getCols() > print_cols)
            {
                std::cout << "... ";
            }
            std::cout << "]\n";
        }
    
        if (getRows() > print_rows)
        {
            std::cout << "  ...\n";
        }
    }

    void saveMatrix(std::ofstream &out_file) const
    {
        if (!out_file.is_open())
        {
            Logger::logMessage("Matrix::saveMatrix: Output stream is not open", LOG_ERROR, true);
            throw std::runtime_error("Output stream is not open");
        }

        std::uint32_t rows = static_cast<std::uint32_t>(pimpl->getRows());
        std::uint32_t cols = static_cast<std::uint32_t>(pimpl->getCols());
        out_file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        out_file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

        std::vector<float> host_data = pimpl->getData();
        out_file.write(reinterpret_cast<const char *>(host_data.data()), host_data.size() * sizeof(float));
    }

    static Matrix loadMatrix(std::ifstream &in_file, Execution_Target exec_target = Execution_Target::CPU)
    {
        if (!in_file.is_open())
        {
            Logger::logMessage("Matrix::loadMatrix: Input stream is not open", LOG_ERROR, true);
            throw std::runtime_error("Input stream is not open");
        }

        std::uint32_t rows = 0;
        std::uint32_t cols = 0;
        in_file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        in_file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        std::vector<float> host_data(rows * cols);
        in_file.read(reinterpret_cast<char *>(host_data.data()), host_data.size() * sizeof(float));

        return Matrix(rows, cols, std::move(host_data), exec_target);
    }

    std::size_t getRows() const { return pimpl->getRows(); }
    std::size_t getCols() const { return pimpl->getCols(); }
    Execution_Target getTarget() const { return current_target; }

    void setExecutionTarget(Execution_Target new_target)
    {
        if (current_target == new_target) return;

        Logger::logMessage("Matrix::setExecutionTarget: Changing execution target from " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(current_target)) + " to " + static_cast<std::string>(magic_enum::enum_name<Execution_Target>(new_target)), LOG_WARNING);

        std::size_t r = getRows();
        std::size_t c = getCols();
        std::vector<float> current_data = getData();
        *this = Matrix(r, c, current_data, new_target);

        if (new_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
    }

    std::vector<float> getData() const { return pimpl->getData(); }
    Matrix operator*(const Matrix &other) const { return Matrix(pimpl->matmul(other.pimpl), current_target); }
    Matrix operator/(const Matrix &other) const { return Matrix(pimpl->matdiv(other.pimpl), current_target); }
    Matrix operator*(float scalar) const { return Matrix(pimpl->mulScalar(scalar), current_target); }
    Matrix operator/(float scalar) const { return Matrix(pimpl->divScalar(scalar), current_target); }
    Matrix operator+(const Matrix &other) const { return Matrix(pimpl->add(other.pimpl), current_target); }
    Matrix operator-(const Matrix &other) const { return Matrix(pimpl->sub(other.pimpl), current_target); }
    Matrix hadamardMul(const Matrix &other) const { return Matrix(pimpl->hadamardMul(other.pimpl), current_target); }
    Matrix hadamardDiv(const Matrix &other) const { return Matrix(pimpl->hadamardDiv(other.pimpl), current_target); }
    Matrix transpose() const { return Matrix(pimpl->transpose(), current_target); }
    Matrix inverse() const { return Matrix(pimpl->inverse(), current_target); }
    Matrix normalize() const { return Matrix(pimpl->normalize(), current_target); }
    Matrix relu() const { return Matrix(pimpl->relu(), current_target); }
    Matrix reluBackward(const Matrix &gradient) const { return Matrix(pimpl->reluBackward(gradient.pimpl), current_target); }
    Matrix gelu() const { return Matrix(pimpl->gelu(), current_target); }
    Matrix geluBackward(const Matrix &gradient) const { return Matrix(pimpl->geluBackward(gradient.pimpl), current_target); }
    Matrix softmax() const { return Matrix(pimpl->softmax(), current_target); }
    Matrix softmaxBackward(const Matrix &gradient) const { return Matrix(pimpl->softmaxBackward(gradient.pimpl), current_target); }
    void sgdUpdate(const Matrix &gradient, float learning_rate, float max_gradient = 0.0f) { pimpl->sgdUpdate(gradient.pimpl, learning_rate, max_gradient); }
    void adamUpdate(
        const Matrix &gradient,
        const Matrix &m_matrix,
        const Matrix &v_matrix,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        std::size_t timestep,
        float max_gradient = 1.0f) { pimpl->adamUpdate(gradient.pimpl, m_matrix.pimpl, v_matrix.pimpl, learning_rate, beta1, beta2, epsilon, timestep, max_gradient); }
    Matrix matmulAdd(const Matrix &other, const Matrix &biases) { return Matrix(pimpl->matmulAdd(other.pimpl, biases.pimpl), current_target); }

    Matrix conv2d(const Matrix &weights, const Matrix &bias,
                  std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
                  std::uint32_t out_c, std::uint32_t kernel_size,
                  std::uint32_t stride, std::uint32_t padding) const
    {
        return Matrix(pimpl->conv2d(weights.pimpl, bias.pimpl, in_h, in_w, in_c, out_c, kernel_size, stride, padding), current_target);
    }

    Matrix conv2dBackwardInput(const Matrix &weights,
                               std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
                               std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
                               std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        return Matrix(pimpl->conv2dBackwardInput(weights.pimpl, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding), current_target);
    }

    void conv2dBackwardWeight(const Matrix &grad_output, Matrix &grad_weights, Matrix &grad_biases,
                              std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
                              std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
                              std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        pimpl->conv2dBackwardWeight(grad_output.pimpl, grad_weights.pimpl, grad_biases.pimpl, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    }

    void maxpool2d(
        Matrix &out_result,
        Matrix &out_mask,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        pimpl->maxpool2d(*out_result.pimpl, *out_mask.pimpl, in_h, in_w, channels, kernel_size, stride, padding);
    }

    void maxpool2dBackward(
        const Matrix &mask,
        Matrix &grad_input,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t out_h, std::uint32_t out_w,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        pimpl->maxpool2dBackward(*mask.pimpl, *grad_input.pimpl, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding);
    }

    Matrix globalAvgPool2d(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const
    {
        return Matrix(pimpl->globalAvgPool2d(in_h, in_w, channels), current_target);
    }

    Matrix globalAvgPool2dBackward(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const
    {
        return Matrix(pimpl->globalAvgPool2dBackward(in_h, in_w, channels), current_target);
    }

    Matrix batchNormForward(const Matrix &gamma, const Matrix &beta,
                            Matrix &running_mean, Matrix &running_var,
                            Matrix &batch_mean, Matrix &batch_var, Matrix &x_hat,
                            float epsilon, float momentum, bool is_training) const
    {
        auto result_impl = pimpl->batchNormForward(
            gamma.pimpl, beta.pimpl,
            running_mean.pimpl, running_var.pimpl,
            batch_mean.pimpl, batch_var.pimpl, x_hat.pimpl,
            epsilon, momentum, is_training);

        return Matrix(result_impl, current_target);
    }

    Matrix batchNormBackward(const Matrix &grad_output, const Matrix &gamma,
                             const Matrix &batch_var, const Matrix &x_hat,
                             Matrix &grad_gamma, Matrix &grad_beta,
                             float epsilon) const
    {
        auto dx_impl = pimpl->batchNormBackward(
            grad_output.pimpl, gamma.pimpl, batch_var.pimpl, x_hat.pimpl,
            grad_gamma.pimpl, grad_beta.pimpl, epsilon);

        return Matrix(dx_impl, current_target);
    }

    void linearForward(
        const Matrix &weights_w,
        const Matrix &biases_b,
        Matrix &output_y) const
    {
        pimpl->linearForward(*weights_w.pimpl, *biases_b.pimpl, *output_y.pimpl);
    }

    void linearBackwardInput(
        const Matrix &weights_w,
        Matrix &grad_x) const
    {
        pimpl->linearBackwardInput(*weights_w.pimpl, *grad_x.pimpl);
    }

    void linearBackwardWeightBias(
        const Matrix &grad_y,
        Matrix &grad_w,
        Matrix &grad_b) const
    {
        pimpl->linearBackwardWeightBias(*grad_y.pimpl, *grad_w.pimpl, *grad_b.pimpl);
    }

    void batchNorm2dForward(
        const Matrix &gamma, const Matrix &beta,
        Matrix &running_mean, Matrix &running_var,
        Matrix &batch_mean, Matrix &batch_var, Matrix &x_hat,
        Matrix &output_y,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        float epsilon, float momentum, bool is_training) const
    {
        pimpl->batchNorm2dForward(
            *gamma.pimpl, *beta.pimpl,
            *running_mean.pimpl, *running_var.pimpl,
            *batch_mean.pimpl, *batch_var.pimpl, *x_hat.pimpl,
            *output_y.pimpl,
            in_h, in_w, in_c, epsilon, momentum, is_training);
    }

    void batchNorm2dBackward(
        const Matrix &gamma,
        const Matrix &batch_var,
        const Matrix &x_hat,
        Matrix &grad_gamma,
        Matrix &grad_beta,
        Matrix &grad_input,
        std::uint32_t in_h,
        std::uint32_t in_w,
        std::uint32_t in_c,
        float epsilon) const
    {
        pimpl->batchNorm2dBackward(
            *gamma.pimpl,
            *batch_var.pimpl,
            *x_hat.pimpl,
            *grad_gamma.pimpl,
            *grad_beta.pimpl,
            *grad_input.pimpl,
            in_h,
            in_w,
            in_c,
            epsilon);
    }

    Matrix cceLoss(const Matrix &target_matrix, float epsilon_val = 1e-7f) const
    {
        return Matrix(pimpl->cceLoss(target_matrix.pimpl, epsilon_val), current_target);
    }

    Matrix mseLoss(const Matrix &target_matrix) const
    {
        return Matrix(pimpl->mseLoss(target_matrix.pimpl), current_target);
    }

    Matrix maeLoss(const Matrix &target_matrix) const
    {
        return Matrix(pimpl->maeLoss(target_matrix.pimpl), current_target);
    }

    Matrix bceLoss(const Matrix &target_matrix, float epsilon_val = 1e-7f) const
    {
        return Matrix(pimpl->bceLoss(target_matrix.pimpl, epsilon_val), current_target);
    }

    float getScalar() const
    {
        const auto &host_data = getData();
        float sum_val = 0.0f;
        for (float val : host_data)
        {
            sum_val += val;
        }
        return sum_val;
    }

    void uploadData(const std::vector<float> &host_data) { pimpl->uploadData(host_data); }
    std::shared_ptr<GVector> getGVector() {return pimpl->getGVector(); }
};