#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "impl.h"

struct Matrix_Dimensions
{
    std::uint32_t rows_a;
    std::uint32_t cols_a;
    std::uint32_t cols_b;
};

struct Elementwise_Dimensions
{
    std::uint32_t total_elements;
    std::uint32_t cols;
    std::uint32_t is_broadcast;
};

struct Transpose_Dimensions
{
    std::uint32_t rows;
    std::uint32_t cols;
};

class Gpu_Matrix_Impl : public Impl
{
private:
    std::size_t rows;
    std::size_t cols;
    std::shared_ptr<GVector> storage;
    mutable std::vector<float> host_cache;

    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum pipeline_id, const std::vector<std::shared_ptr<GVector>> &buffers, const Push_Constants_Type &push_constants, std::uint32_t group_x, std::uint32_t group_y = 1, std::uint32_t group_z = 1) const
    {
        if (buffers.size() > 16)
        {
            Logger::logMessage("Gpu_Matrix_Impl::pushToGraph: Exceeded maximum supported buffer count", LOG_ERROR, true);
            throw std::runtime_error("Exceeded maximum supported buffer count");
        }

        Compute_Node node;
        node.pipeline_id = pipeline_id;
        node.buffers = buffers;

        const std::uint8_t *byte_ptr = reinterpret_cast<const std::uint8_t *>(&push_constants);
        node.push_constants_data.assign(byte_ptr, byte_ptr + sizeof(Push_Constants_Type));

        node.group_x = group_x;
        node.group_y = group_y;
        node.group_z = group_z;

        Execution_Engine::getInstance().getCurrentGraph().addNode(node);
    }

    std::shared_ptr<Gpu_Matrix_Impl> castToGpuMatrix(const std::shared_ptr<Impl> &other_impl, const std::string &error_msg) const
    {
        auto other_gpu = std::dynamic_pointer_cast<Gpu_Matrix_Impl>(other_impl);
        if (!other_gpu)
        {
            Logger::logMessage(error_msg, LOG_ERROR, true);
            throw std::invalid_argument(error_msg);
        }
        return other_gpu;
    }

    const Gpu_Matrix_Impl &castToGpuMatrix(const Impl &other_impl, const std::string &error_msg) const
    {
        try
        {
            return dynamic_cast<const Gpu_Matrix_Impl &>(other_impl);
        }
        catch (const std::bad_cast &)
        {
            Logger::logMessage(error_msg, LOG_ERROR, true);
            throw std::invalid_argument(error_msg);
        }
    }

    Gpu_Matrix_Impl &castToGpuMatrix(Impl &other_impl, const std::string &error_msg) const
    {
        try
        {
            return dynamic_cast<Gpu_Matrix_Impl &>(other_impl);
        }
        catch (const std::bad_cast &)
        {
            Logger::logMessage(error_msg, LOG_ERROR, true);
            throw std::invalid_argument(error_msg);
        }
    }

    template <typename Pipeline_Enum>
    std::shared_ptr<Impl> executeElementwise(const std::shared_ptr<Impl> &other_impl, Pipeline_Enum pipeline_id, const std::string &error_msg, bool allow_broadcast) const
    {
        auto other_gpu = castToGpuMatrix(other_impl, error_msg);

        bool is_broadcast = false;
        if (allow_broadcast)
        {
            bool same_shape = (rows == other_gpu->getRows() && cols == other_gpu->getCols());
            is_broadcast = (other_gpu->getRows() == 1 && cols == other_gpu->getCols());

            if (!same_shape && !is_broadcast)
                validateSameDimensions(*other_impl);
        }
        else
            validateSameDimensions(*other_impl);

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);

        Elementwise_Dimensions dims{
            .total_elements = static_cast<std::uint32_t>(rows * cols),
            .cols = static_cast<std::uint32_t>(cols),
            .is_broadcast = static_cast<std::uint32_t>(is_broadcast ? 1 : 0)};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::executeElementwise: total_elements={}, cols={}, is_broadcast={}", dims.total_elements, dims.cols, dims.is_broadcast));
        pushToGraph(pipeline_id, {storage, other_gpu->storage, result->storage}, dims, (dims.total_elements + 255) / 256);

        return result;
    }

public:
    Gpu_Matrix_Impl(std::size_t r, std::size_t c)
        : rows(r), cols(c)
    {
        const auto &engine = Execution_Engine::getInstance();
        storage = std::make_shared<GVector>(engine.getContext(), rows * cols);
    }

    Gpu_Matrix_Impl(std::size_t r, std::size_t c, const std::vector<float> &host_data)
        : rows(r), cols(c)
    {
        if (host_data.size() != rows * cols)
        {
            Logger::logMessage("Gpu_Matrix_Impl::Gpu_Matrix_Impl: Host data size mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Host data size mismatch");
        }
        const auto &engine = Execution_Engine::getInstance();
        storage = std::make_shared<GVector>(engine.getContext(), host_data);
    }

    Gpu_Matrix_Impl(std::size_t r, std::size_t c, std::shared_ptr<GVector> vec)
        : rows(r), cols(c), storage(std::move(vec)) {}

    ~Gpu_Matrix_Impl() override = default;

    std::size_t getRows() const override { return rows; }

    std::size_t getCols() const override { return cols; }

    const std::vector<float> &getData() const override
    {
        if (rows == 0 || cols == 0)
        {
            Logger::logMessage(std::format("[GET_DATA_DEBUG] Warning: rows or cols is 0! rows={}, cols={}", rows, cols), LOG_WARNING);
            host_cache.clear();
            return host_cache;
        }
        Execution_Engine::getInstance().getContext().flush();
        host_cache.resize(rows * cols);
        storage->downloadData(host_cache);
        return host_cache;
    }

    std::shared_ptr<GVector> getGVector() override { return storage; }

    VkBuffer getBuffer() const { return storage->getBuffer(); }

    std::shared_ptr<Impl> matmul(const std::shared_ptr<Impl> &other_impl) const override
    {
        validateMatmulDimensions(*other_impl);
        auto other_gpu = castToGpuMatrix(other_impl, "Gpu_Matrix_Impl::matmul: Target matrix is not a Gpu_Matrix_Impl");

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, other_gpu->getCols());

        Matrix_Dimensions dims{
            .rows_a = static_cast<std::uint32_t>(rows),
            .cols_a = static_cast<std::uint32_t>(cols),
            .cols_b = static_cast<std::uint32_t>(result->getCols())};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::matmul: rows_a={}, cols_a={}, cols_b={}", dims.rows_a, dims.cols_a, dims.cols_b));
        pushToGraph(MATMUL, {storage, other_gpu->storage, result->storage}, dims, (dims.cols_b + 15) / 16, (dims.rows_a + 15) / 16);
        return result;
    }

    std::shared_ptr<Impl> add(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, ADD, "Gpu_Matrix_Impl::add: Target matrix is not a Gpu_Matrix_Impl", true);
    }

    std::shared_ptr<Impl> sub(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, SUB, "Gpu_Matrix_Impl::sub: Target matrix is not a Gpu_Matrix_Impl", true);
    }

    std::shared_ptr<Impl> mulScalar(float scalar) const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Scalar_Push_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{total_elements, scalar};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::mulScalar: total_elements={}, scalar={}", constants.total_elements, constants.scalar));
        pushToGraph(MUL_SCALAR, {storage, result->storage}, constants, (total_elements + 255) / 256);

        return result;
    }

    std::shared_ptr<Impl> divScalar(float scalar) const override
    {
        if (std::abs(scalar) < 1e-8f)
        {
            Logger::logMessage("Gpu_Matrix_Impl::divScalar: Division by zero", LOG_ERROR, true);
            throw std::runtime_error("Division by zero in divScalar");
        }

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Scalar_Push_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{total_elements, 1.0f / scalar};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::divScalar: total_elements={}, scalar={}", constants.total_elements, constants.scalar));
        pushToGraph(MUL_SCALAR, {storage, result->storage}, constants, (total_elements + 255) / 256);

        return result;
    }

    std::shared_ptr<Impl> hadamardMul(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, HADAMARD_MUL, "Gpu_Matrix_Impl::hadamardMul: Target matrix is not a Gpu_Matrix_Impl", false);
    }

    std::shared_ptr<Impl> hadamardDiv(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, HADAMARD_DIV, "Gpu_Matrix_Impl::hadamardDiv: Target matrix is not a Gpu_Matrix_Impl", false);
    }

    std::shared_ptr<Impl> transpose() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(cols, rows);

        Transpose_Dimensions dims{
            .rows = static_cast<std::uint32_t>(rows),
            .cols = static_cast<std::uint32_t>(cols)};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::transpose: rows={}, cols={}", dims.rows, dims.cols));
        pushToGraph(TRANSPOSE, {storage, result->storage}, dims, (dims.cols + 15) / 16, (dims.rows + 15) / 16);

        return result;
    }

    std::shared_ptr<Impl> relu() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::relu: total_elements={}", total_elements));
        pushToGraph(RELU, {storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> reluBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        validateSameDimensions(*grad_impl);
        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::reluBackward: Target matrix is not a Gpu_Matrix_Impl");
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::reluBackward: total_elements={}", total_elements));
        pushToGraph(RELU_BACKWARD, {storage, grad_gpu->storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> gelu() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::gelu: total_elements={}", total_elements));
        pushToGraph(GELU, {storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> geluBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        validateSameDimensions(*grad_impl);
        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::geluBackward: Target matrix is not a Gpu_Matrix_Impl");
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::geluBackward: total_elements={}", total_elements));
        pushToGraph(GELU_BACKWARD, {storage, grad_gpu->storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> inverse() const override
    {
        validateSquare();

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(n, n);

        struct Inverse_Push_Constants
        {
            std::uint32_t n;
        } constants{n};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::inverse: n={}", constants.n));
        pushToGraph(MATRIX_INVERSE, {storage, result->storage}, constants, 1, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> normalize() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Normalize_Push_Constants
        {
            std::uint32_t total_elements;
        } constants{total_elements};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::normalize: total_elements={}", constants.total_elements));
        pushToGraph(NORMALIZE, {storage, result->storage}, constants, 1, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> softmax() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);

        struct Softmax_Dimensions
        {
            std::uint32_t rows;
            std::uint32_t cols;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols)};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::softmax: rows={}, cols={}", constants.rows, constants.cols));
        pushToGraph(SOFTMAX, {storage, result->storage}, constants, constants.rows, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> softmaxBackward(const std::shared_ptr<Impl> &gradient_output) const override
    {
        validateSameDimensions(*gradient_output);
        auto grad_gpu = castToGpuMatrix(gradient_output, "Gpu_Matrix_Impl::softmaxBackward: Invalid matrix type");

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);

        struct Softmax_Dimensions
        {
            std::uint32_t rows;
            std::uint32_t cols;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols)};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::softmaxBackward: rows={}, cols={}", constants.rows, constants.cols));
        pushToGraph(SOFTMAX_BACKWARD, {storage, grad_gpu->storage, result->storage}, constants, constants.rows, 1, 1);

        return result;
    }

    void sgdUpdate(const std::shared_ptr<Impl> &grad_impl, float learning_rate, float max_gradient = 0.0f) override
    {
        validateSameDimensions(*grad_impl);
        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::sgdUpdate: Target matrix is not a Gpu_Matrix_Impl");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Sgd_Push_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float max_gradient;
        } constants{total_elements, learning_rate, max_gradient};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::sgdUpdate: total_elements={}, learning_rate={}, max_gradient={}", constants.total_elements, constants.learning_rate, constants.max_gradient));
        pushToGraph(SGD_UPDATE, {storage, grad_gpu->storage}, constants, (total_elements + 255) / 256);
    }

    void adamUpdate(
        const std::shared_ptr<Impl> &grad_impl,
        const std::shared_ptr<Impl> &m_impl,
        const std::shared_ptr<Impl> &v_impl,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        std::size_t timestep,
        float max_gradient = 1.0f) override
    {
        validateSameDimensions(*grad_impl);
        validateSameDimensions(*m_impl);
        validateSameDimensions(*v_impl);

        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::adamUpdate: Invalid grad_impl matrix");
        auto m_gpu = castToGpuMatrix(m_impl, "Gpu_Matrix_Impl::adamUpdate: Invalid m_impl matrix");
        auto v_gpu = castToGpuMatrix(v_impl, "Gpu_Matrix_Impl::adamUpdate: Invalid v_impl matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        float correction1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
        float correction2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));

        float inv_correction1 = 1.0f / correction1;
        float inv_sqrt_correction2 = 1.0f / std::sqrt(correction2);

        struct Adam_Push_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float inv_correction1;
            float inv_sqrt_correction2;
        } constants{
            total_elements,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            max_gradient,
            inv_correction1,
            inv_sqrt_correction2};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::adamUpdate: total_elements={}, learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, inv_correction1={}, inv_sqrt_correction2={}",
                                     constants.total_elements, constants.learning_rate, constants.beta1, constants.beta2, constants.epsilon, constants.max_gradient, constants.inv_correction1, constants.inv_sqrt_correction2));
        pushToGraph(
            ADAM_UPDATE,
            {storage, grad_gpu->storage, m_gpu->storage, v_gpu->storage},
            constants,
            (total_elements + 255) / 256);
    }

    std::shared_ptr<Impl> matdiv(const std::shared_ptr<Impl> &other_impl) const override { return matmul(other_impl->inverse()); }

    std::shared_ptr<Impl> matmulAdd(const std::shared_ptr<Impl> &other, const std::shared_ptr<Impl> &bias) const override
    {
        validateMatmulDimensions(*other);
        auto w_gpu = castToGpuMatrix(other, "Invalid weight matrix type");
        auto b_gpu = castToGpuMatrix(bias, "Invalid bias matrix type");

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, w_gpu->cols);

        struct Push_Constants
        {
            std::uint32_t rows_x;
            std::uint32_t cols_x;
            std::uint32_t cols_w;
            std::uint32_t pad;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(w_gpu->cols),
            0};

        std::uint32_t group_x = (constants.cols_w + 15) / 16;
        std::uint32_t group_y = (constants.rows_x + 15) / 16;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::matmulAdd: rows_x={}, cols_x={}, cols_w={}, pad={}", constants.rows_x, constants.cols_x, constants.cols_w, constants.pad));
        pushToGraph(MATMUL_ADD, {storage, w_gpu->storage, b_gpu->storage, result->storage}, constants, group_x, group_y, 1);

        return result;
    }

    std::shared_ptr<Impl> conv2d(
        const std::shared_ptr<Impl> &weights,
        const std::shared_ptr<Impl> &bias,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_c, std::uint32_t kernel_size,
        std::uint32_t stride, std::uint32_t padding) const override
    {
        auto w_gpu = castToGpuMatrix(weights, "Gpu_Matrix_Impl::conv2d: Invalid weights matrix");
        auto b_gpu = castToGpuMatrix(bias, "Gpu_Matrix_Impl::conv2d: Invalid bias matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, out_h * out_w * out_c);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t in_c;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t out_c;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding};

        std::uint32_t group_x = (out_c + 15) / 16;
        std::uint32_t group_y = (out_w + 15) / 16;
        std::uint32_t group_z = batch_size * out_h;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::conv2d: batch_size={}, in_h={}, in_w={}, in_c={}, out_h={}, out_w={}, out_c={}, kernel_size={}, stride={}, padding={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.in_c, constants.out_h, constants.out_w, constants.out_c, constants.kernel_size, constants.stride, constants.padding));
        pushToGraph(CONV2D_FORWARD_PASS, {storage, w_gpu->storage, b_gpu->storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
    }

    std::shared_ptr<Impl> conv2dBackwardInput(
        const std::shared_ptr<Impl> &weights,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto w_gpu = castToGpuMatrix(weights, "Gpu_Matrix_Impl::conv2dBackwardInput: Invalid weights matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, in_h * in_w * in_c);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t in_c;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t out_c;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding};

        std::uint32_t group_x = (in_c + 15) / 16;
        std::uint32_t group_y = (in_w + 15) / 16;
        std::uint32_t group_z = batch_size * in_h;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::conv2dBackwardInput: batch_size={}, in_h={}, in_w={}, in_c={}, out_h={}, out_w={}, out_c={}, kernel_size={}, stride={}, padding={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.in_c, constants.out_h, constants.out_w, constants.out_c, constants.kernel_size, constants.stride, constants.padding));
        pushToGraph(CONV2D_BACKWARD_PASS_INPUT_GRADIENT, {storage, w_gpu->storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
    }

    void conv2dBackwardWeight(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &grad_weights,
        const std::shared_ptr<Impl> &grad_biases,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto go_gpu = castToGpuMatrix(grad_output, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid grad_output matrix");
        auto gw_gpu = castToGpuMatrix(grad_weights, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid grad_weights matrix");
        auto gb_gpu = castToGpuMatrix(grad_biases, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid grad_biases matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t in_c;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t out_c;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding};

        std::uint32_t group_x = (out_c + 15) / 16;
        std::uint32_t group_y = (in_c + 15) / 16;
        std::uint32_t group_z = kernel_size * kernel_size;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::conv2dBackwardWeight: batch_size={}, in_h={}, in_w={}, in_c={}, out_h={}, out_w={}, out_c={}, kernel_size={}, stride={}, padding={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.in_c, constants.out_h, constants.out_w, constants.out_c, constants.kernel_size, constants.stride, constants.padding));
        pushToGraph(CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT, {storage, go_gpu->storage, gw_gpu->storage, gb_gpu->storage}, constants, group_x, group_y, group_z);
    }

    void maxpool2d(
        Impl &out_result,
        Impl &out_mask,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto &result_gpu = static_cast<Gpu_Matrix_Impl &>(out_result);
        auto &mask_gpu = static_cast<Gpu_Matrix_Impl &>(out_mask);

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        struct Max_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding};

        std::uint32_t group_x = (channels + 15) / 16;
        std::uint32_t group_y = (out_w + 15) / 16;
        std::uint32_t group_z = batch_size * out_h;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::maxpool2d: batch_size={}, in_h={}, in_w={}, channels={}, out_h={}, out_w={}, kernel_size={}, stride={}, padding={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.channels, constants.out_h, constants.out_w, constants.kernel_size, constants.stride, constants.padding));
        pushToGraph(MAXPOOL2D_FORWARD, {storage, result_gpu.storage, mask_gpu.storage}, constants, group_x, group_y, group_z);
    }

    void maxpool2dBackward(
        const Impl &mask,
        Impl &grad_input,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t out_h, std::uint32_t out_w,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        const auto &mask_gpu = static_cast<const Gpu_Matrix_Impl &>(mask);
        auto &grad_in_gpu = static_cast<Gpu_Matrix_Impl &>(grad_input);

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);

        struct Max_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding};

        std::uint32_t group_x = (channels + 15) / 16;
        std::uint32_t group_y = (in_w + 15) / 16;
        std::uint32_t group_z = batch_size * in_h;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::maxpool2dBackward: batch_size={}, in_h={}, in_w={}, channels={}, out_h={}, out_w={}, kernel_size={}, stride={}, padding={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.channels, constants.out_h, constants.out_w, constants.kernel_size, constants.stride, constants.padding));
        pushToGraph(MAXPOOL2D_BACKWARD, {storage, mask_gpu.storage, grad_in_gpu.storage}, constants, group_x, group_y, group_z);
    }

    std::shared_ptr<Impl> globalAvgPool2d(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, channels);

        struct Global_Avg_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
        } constants{batch_size, in_h, in_w, channels};

        std::uint32_t group_x = (channels + 255) / 256;
        std::uint32_t group_y = batch_size;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::globalAvgPool2d: batch_size={}, in_h={}, in_w={}, channels={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.channels));
        pushToGraph(GLOBAL_AVGPOOL_FORWARD, {storage, result->storage}, constants, group_x, group_y, 1);

        return result;
    }

    std::shared_ptr<Impl> globalAvgPool2dBackward(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, in_h * in_w * channels);

        struct Global_Avg_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
        } constants{batch_size, in_h, in_w, channels};

        std::uint32_t group_x = (channels + 15) / 16;
        std::uint32_t group_y = (in_w + 15) / 16;
        std::uint32_t group_z = batch_size * in_h;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::globalAvgPool2dBackward: batch_size={}, in_h={}, in_w={}, channels={}",
                                     constants.batch_size, constants.in_h, constants.in_w, constants.channels));
        pushToGraph(GLOBAL_AVGPOOL_BACKWARD, {storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
    }

    std::shared_ptr<Impl> batchNormForward(
        const std::shared_ptr<Impl> &gamma,
        const std::shared_ptr<Impl> &beta,
        std::shared_ptr<Impl> &running_mean,
        std::shared_ptr<Impl> &running_var,
        std::shared_ptr<Impl> &batch_mean,
        std::shared_ptr<Impl> &batch_var,
        std::shared_ptr<Impl> &x_hat,
        float epsilon,
        float momentum,
        bool is_training) const override
    {
        auto gamma_gpu = castToGpuMatrix(gamma, "Invalid gamma matrix");
        auto beta_gpu = castToGpuMatrix(beta, "Invalid beta matrix");
        auto r_mean_gpu = castToGpuMatrix(running_mean, "Invalid running_mean matrix");
        auto r_var_gpu = castToGpuMatrix(running_var, "Invalid running_var matrix");

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        std::uint32_t d = static_cast<std::uint32_t>(cols);

        auto result = std::make_shared<Gpu_Matrix_Impl>(n, d);
        x_hat = std::make_shared<Gpu_Matrix_Impl>(n, d);
        auto x_hat_gpu = castToGpuMatrix(x_hat, "Invalid x_hat matrix");

        if (is_training)
        {
            batch_mean = std::make_shared<Gpu_Matrix_Impl>(1, d);
            batch_var = std::make_shared<Gpu_Matrix_Impl>(1, d);
            auto b_mean_gpu = castToGpuMatrix(batch_mean, "Invalid batch_mean matrix");
            auto b_var_gpu = castToGpuMatrix(batch_var, "Invalid batch_var matrix");

            struct Stats_Push_Constants
            {
                std::uint32_t batch_size;
                std::uint32_t dim;
                float momentum;
            } stats_consts{n, d, momentum};

            MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNormForward (stats): batch_size={}, dim={}, momentum={}", stats_consts.batch_size, stats_consts.dim, stats_consts.momentum));
            pushToGraph(BATCH_NORM_STATS_FORWARD, {storage, b_mean_gpu->storage, b_var_gpu->storage, r_mean_gpu->storage, r_var_gpu->storage}, stats_consts, (d + 255) / 256, 1, 1);

            struct Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t dim;
                float epsilon;
            } trans_consts{n * d, d, epsilon};

            MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNormForward (transform): total_elements={}, dim={}, epsilon={}", trans_consts.total_elements, trans_consts.dim, trans_consts.epsilon));
            pushToGraph(BATCH_NORM_TRANSFORM_FORWARD, {storage, b_mean_gpu->storage, b_var_gpu->storage, gamma_gpu->storage, beta_gpu->storage, result->storage, x_hat_gpu->storage}, trans_consts, (n * d + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t dim;
                float epsilon;
            } trans_consts{n * d, d, epsilon};

            MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNormForward (transform eval): total_elements={}, dim={}, epsilon={}", trans_consts.total_elements, trans_consts.dim, trans_consts.epsilon));
            pushToGraph(BATCH_NORM_TRANSFORM_FORWARD, {storage, r_mean_gpu->storage, r_var_gpu->storage, gamma_gpu->storage, beta_gpu->storage, result->storage, x_hat_gpu->storage}, trans_consts, (n * d + 255) / 256, 1, 1);
        }

        return result;
    }

    std::shared_ptr<Impl> batchNormBackward(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &gamma,
        const std::shared_ptr<Impl> &batch_var,
        const std::shared_ptr<Impl> &x_hat,
        std::shared_ptr<Impl> &grad_gamma,
        std::shared_ptr<Impl> &grad_beta,
        float epsilon) const override
    {
        auto dout_gpu = castToGpuMatrix(grad_output, "Invalid grad_output matrix");
        auto gamma_gpu = castToGpuMatrix(gamma, "Invalid gamma matrix");
        auto b_var_gpu = castToGpuMatrix(batch_var, "Invalid batch_var matrix");
        auto x_hat_gpu = castToGpuMatrix(x_hat, "Invalid x_hat matrix");

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        std::uint32_t d = static_cast<std::uint32_t>(cols);

        grad_gamma = std::make_shared<Gpu_Matrix_Impl>(1, d);
        grad_beta = std::make_shared<Gpu_Matrix_Impl>(1, d);
        auto g_gamma_gpu = castToGpuMatrix(grad_gamma, "Invalid grad_gamma matrix");
        auto g_beta_gpu = castToGpuMatrix(grad_beta, "Invalid grad_beta matrix");

        auto dx = std::make_shared<Gpu_Matrix_Impl>(n, d);

        struct Backward_Stats_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t dim;
        } b_stats_consts{n, d};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNormBackward (stats): batch_size={}, dim={}", b_stats_consts.batch_size, b_stats_consts.dim));
        pushToGraph(BATCH_NORM_STATS_BACKWARD, {dout_gpu->storage, x_hat_gpu->storage, g_gamma_gpu->storage, g_beta_gpu->storage}, b_stats_consts, (d + 255) / 256, 1, 1);

        struct Backward_Transform_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t batch_size;
            std::uint32_t dim;
            float epsilon;
        } b_trans_consts{n * d, n, d, epsilon};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNormBackward (transform): total_elements={}, batch_size={}, dim={}, epsilon={}", b_trans_consts.total_elements, b_trans_consts.batch_size, b_trans_consts.dim, b_trans_consts.epsilon));
        pushToGraph(BATCH_NORM_TRANSFORM_BACKWARD, {dout_gpu->storage, x_hat_gpu->storage, gamma_gpu->storage, g_gamma_gpu->storage, g_beta_gpu->storage, b_var_gpu->storage, dx->storage}, b_trans_consts, (n * d + 255) / 256, 1, 1);

        return dx;
    }

    void linearForward(
        const Impl &weights_w,
        const Impl &biases_b,
        Impl &output_y) const override
    {
        const auto &w_gpu = static_cast<const Gpu_Matrix_Impl &>(weights_w);
        const auto &b_gpu = static_cast<const Gpu_Matrix_Impl &>(biases_b);
        auto &y_gpu = static_cast<Gpu_Matrix_Impl &>(output_y);

        struct Push_Constants
        {
            std::uint32_t rows_x;
            std::uint32_t cols_x;
            std::uint32_t cols_w;
            std::uint32_t pad;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(w_gpu.cols),
            0};

        std::uint32_t group_x = (constants.cols_w + 15) / 16;
        std::uint32_t group_y = (constants.rows_x + 15) / 16;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::linearForward: rows_x={}, cols_x={}, cols_w={}, pad={}", constants.rows_x, constants.cols_x, constants.cols_w, constants.pad));
        pushToGraph(MATMUL_ADD, {storage, w_gpu.storage, b_gpu.storage, y_gpu.storage}, constants, group_x, group_y, 1);
    }

    void linearBackwardInput(
        const Impl &weights_w,
        Impl &grad_x) const override
    {
        const auto &w_gpu = static_cast<const Gpu_Matrix_Impl &>(weights_w);
        auto &dx_gpu = static_cast<Gpu_Matrix_Impl &>(grad_x);

        struct Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_dim;
            std::uint32_t out_dim;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(w_gpu.rows),
            static_cast<std::uint32_t>(cols)};

        std::uint32_t group_x = (constants.in_dim + 15) / 16;
        std::uint32_t group_y = (constants.batch_size + 15) / 16;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::linearBackwardInput: batch_size={}, in_dim={}, out_dim={}", constants.batch_size, constants.in_dim, constants.out_dim));
        pushToGraph(LINEAR_BACKWARD_INPUT, {storage, w_gpu.storage, dx_gpu.storage}, constants, group_x, group_y, 1);
    }

    void linearBackwardWeightBias(
        const Impl &grad_y,
        Impl &grad_w,
        Impl &grad_b) const override
    {
        const auto &dy_gpu = static_cast<const Gpu_Matrix_Impl &>(grad_y);
        auto &dw_gpu = static_cast<Gpu_Matrix_Impl &>(grad_w);
        auto &db_gpu = static_cast<Gpu_Matrix_Impl &>(grad_b);

        struct Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_dim;
            std::uint32_t out_dim;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(dy_gpu.cols)};

        std::uint32_t group_x = (constants.out_dim + 15) / 16;
        std::uint32_t group_y = (constants.in_dim + 15) / 16;

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::linearBackwardWeightBias: batch_size={}, in_dim={}, out_dim={}", constants.batch_size, constants.in_dim, constants.out_dim));
        pushToGraph(LINEAR_BACKWARD_WEIGHT_BIAS,
                    {storage, dy_gpu.storage, dw_gpu.storage, db_gpu.storage},
                    constants, group_x, group_y, 1);
    }

    void batchNorm2dForward(
        const Impl &gamma,
        const Impl &beta,
        Impl &running_mean,
        Impl &running_var,
        Impl &batch_mean,
        Impl &batch_var,
        Impl &x_hat,
        Impl &output_y,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        float epsilon, float momentum, bool is_training) const override
    {
        const auto &gamma_gpu = castToGpuMatrix(gamma, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid gamma matrix");
        const auto &beta_gpu = castToGpuMatrix(beta, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid beta matrix");
        auto &r_mean_gpu = castToGpuMatrix(running_mean, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid running_mean matrix");
        auto &r_var_gpu = castToGpuMatrix(running_var, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid running_var matrix");
        auto &b_mean_gpu = castToGpuMatrix(batch_mean, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid batch_mean matrix");
        auto &b_var_gpu = castToGpuMatrix(batch_var, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid batch_var matrix");
        auto &x_hat_gpu = castToGpuMatrix(x_hat, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid x_hat matrix");
        auto &y_gpu = castToGpuMatrix(output_y, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid output_y matrix");

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        std::uint32_t d = in_h * in_w * in_c;
        std::uint32_t spatial_count = n * in_h * in_w;

        if (is_training)
        {
            struct Stats_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                std::uint32_t spatial_count;
                float momentum;
            } stats_consts{n * d, in_c, spatial_count, momentum};

            MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNorm2dForward (stats): total_elements={}, channels={}, spatial_count={}, momentum={}", stats_consts.total_elements, stats_consts.channels, stats_consts.spatial_count, stats_consts.momentum));
            pushToGraph(BATCH_NORM2D_STATS_FORWARD, {storage, b_mean_gpu.storage, b_var_gpu.storage, r_mean_gpu.storage, r_var_gpu.storage}, stats_consts, (in_c + 255) / 256, 1, 1);

            struct Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                float epsilon;
            } trans_consts{n * d, in_c, epsilon};

            MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNorm2dForward (transform): total_elements={}, channels={}, epsilon={}", trans_consts.total_elements, trans_consts.channels, trans_consts.epsilon));
            pushToGraph(BATCH_NORM2D_TRANSFORM_FORWARD, {storage, b_mean_gpu.storage, b_var_gpu.storage, gamma_gpu.storage, beta_gpu.storage, y_gpu.storage, x_hat_gpu.storage}, trans_consts, (n * d + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                float epsilon;
            } trans_consts{n * d, in_c, epsilon};

            MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNorm2dForward (transform eval): total_elements={}, channels={}, epsilon={}", trans_consts.total_elements, trans_consts.channels, trans_consts.epsilon));
            pushToGraph(BATCH_NORM2D_TRANSFORM_FORWARD, {storage, r_mean_gpu.storage, r_var_gpu.storage, gamma_gpu.storage, beta_gpu.storage, y_gpu.storage, x_hat_gpu.storage}, trans_consts, (n * d + 255) / 256, 1, 1);
        }
    }

    void batchNorm2dBackward(
        const Impl &gamma,
        const Impl &batch_var,
        const Impl &x_hat,
        Impl &grad_gamma,
        Impl &grad_beta,
        Impl &grad_input,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        float epsilon) const override
    {
        const auto &gamma_gpu = castToGpuMatrix(gamma, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid gamma matrix");
        const auto &b_var_gpu = castToGpuMatrix(batch_var, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid batch_var matrix");
        const auto &x_hat_gpu = castToGpuMatrix(x_hat, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid x_hat matrix");
        auto &g_gamma_gpu = castToGpuMatrix(grad_gamma, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid grad_gamma matrix");
        auto &g_beta_gpu = castToGpuMatrix(grad_beta, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid grad_beta matrix");
        auto &dx_gpu = castToGpuMatrix(grad_input, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid grad_input matrix");

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        std::uint32_t d = in_h * in_w * in_c;
        std::uint32_t spatial_count = n * in_h * in_w;

        struct Backward_Stats_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t channels;
            std::uint32_t spatial_count;
        } b_stats_consts{n * d, in_c, spatial_count};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNorm2dBackward (stats): total_elements={}, channels={}, spatial_count={}", b_stats_consts.total_elements, b_stats_consts.channels, b_stats_consts.spatial_count));
        pushToGraph(BATCH_NORM2D_STATS_BACKWARD, {storage, x_hat_gpu.storage, g_gamma_gpu.storage, g_beta_gpu.storage}, b_stats_consts, (in_c + 255) / 256, 1, 1);

        struct Backward_Transform_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t channels;
            std::uint32_t spatial_count;
            float epsilon;
        } b_trans_consts{n * d, in_c, spatial_count, epsilon};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::batchNorm2dBackward (transform): total_elements={}, channels={}, spatial_count={}, epsilon={}", b_trans_consts.total_elements, b_trans_consts.channels, b_trans_consts.spatial_count, b_trans_consts.epsilon));
        pushToGraph(BATCH_NORM2D_TRANSFORM_BACKWARD, {storage, x_hat_gpu.storage, gamma_gpu.storage, g_gamma_gpu.storage, g_beta_gpu.storage, b_var_gpu.storage, dx_gpu.storage}, b_trans_consts, (n * d + 255) / 256, 1, 1);
    }

    std::shared_ptr<Impl> cceLoss(
        const std::shared_ptr<Impl> &target_impl,
        float epsilon_val) const override
    {
        validateSameDimensions(*target_impl);
        auto target_gpu = castToGpuMatrix(target_impl, "Gpu_Matrix_Impl::cceLoss: Target matrix is not a Gpu_Matrix_Impl");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);
        std::uint32_t group_x = (total_elements + 255) / 256;

        auto result = std::make_shared<Gpu_Matrix_Impl>(1, group_x);

        struct Cce_Loss_Push_Constants
        {
            std::uint32_t total_elements;
            float epsilon_val;
        } constants{total_elements, epsilon_val};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::cceLoss: total_elements={}, epsilon_val={}", constants.total_elements, constants.epsilon_val));
        pushToGraph(CCE_LOSS, {storage, target_gpu->storage, result->storage}, constants, group_x, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> mseLoss(
        const std::shared_ptr<Impl> &target_impl) const override
    {
        validateSameDimensions(*target_impl);
        auto target_gpu = castToGpuMatrix(target_impl, "Gpu_Matrix_Impl::mseLoss: Target matrix is not a Gpu_Matrix_Impl");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);
        std::uint32_t group_x = (total_elements + 255) / 256;

        auto result = std::make_shared<Gpu_Matrix_Impl>(1, group_x);

        struct Mse_Loss_Push_Constants
        {
            std::uint32_t total_elements;
        } constants{total_elements};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::mseLoss: total_elements={}", constants.total_elements));
        pushToGraph(MSE_LOSS, {storage, target_gpu->storage, result->storage}, constants, group_x, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> maeLoss(
        const std::shared_ptr<Impl> &target_impl) const override
    {
        validateSameDimensions(*target_impl);
        auto target_gpu = castToGpuMatrix(target_impl, "Gpu_Matrix_Impl::maeLoss: Target matrix is not a Gpu_Matrix_Impl");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);
        std::uint32_t group_x = (total_elements + 255) / 256;

        auto result = std::make_shared<Gpu_Matrix_Impl>(1, group_x);

        struct Mae_Loss_Push_Constants
        {
            std::uint32_t total_elements;
        } constants{total_elements};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::maeLoss: total_elements={}", constants.total_elements));
        pushToGraph(MAE_LOSS, {storage, target_gpu->storage, result->storage}, constants, group_x, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> bceLoss(
        const std::shared_ptr<Impl> &target_impl,
        float epsilon_val) const override
    {
        validateSameDimensions(*target_impl);
        auto target_gpu = castToGpuMatrix(target_impl, "Gpu_Matrix_Impl::bceLoss: Target matrix is not a Gpu_Matrix_Impl");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);
        std::uint32_t group_x = (total_elements + 255) / 256;

        auto result = std::make_shared<Gpu_Matrix_Impl>(1, group_x);

        struct Bce_Loss_Push_Constants
        {
            std::uint32_t total_elements;
            float epsilon_val;
        } constants{total_elements, epsilon_val};

        MATRIX_LOG_DEBUG(std::format("Gpu_Matrix_Impl::bceLoss: total_elements={}, epsilon_val={}", constants.total_elements, constants.epsilon_val));
        pushToGraph(BCE_LOSS, {storage, target_gpu->storage, result->storage}, constants, group_x, 1, 1);

        return result;
    }

    void uploadData(const std::vector<float> &host_data) override
    {
        if (host_data.size() != rows * cols)
        {
            Logger::logMessage("Gpu_Matrix_Impl::uploadData: Host data size mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Host data size mismatch");
        }
        storage->uploadData(host_data);
    }
};