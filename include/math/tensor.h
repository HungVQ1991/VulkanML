#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cpu_tensor_impl.h"
#include "gpu_tensor_impl.h"
#include "helper/logger.h"
#include "shape.h"
#include "tensor_impl.h"

enum class Execution_Target
{
    CPU,
    VULKAN_GPU
};

class Tensor
{
private:
    std::shared_ptr<Tensor_Impl> implementation;
    Execution_Target execution_target;

    static constexpr uint32_t TENSOR_MAGIC_HEADER = 0x7FFFFFFF;

public:    explicit Tensor(Execution_Target target = Execution_Target::CPU);
    Tensor(size_t rows, size_t columns, Execution_Target target = Execution_Target::CPU);
    Tensor(size_t rows, size_t columns, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU);
    Tensor(size_t rows, size_t columns, std::vector<float> &&host_data, Execution_Target target = Execution_Target::CPU);
    explicit Tensor(Shape shape, Execution_Target target = Execution_Target::CPU);
    Tensor(Shape shape, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU);
    Tensor(Shape shape, Data_Type type, Execution_Target target = Execution_Target::CPU);
    Tensor(Shape shape, const std::vector<float> &host_data, Data_Type type, Execution_Target target = Execution_Target::CPU);
    void to(Data_Type target_type, Tensor &output) const;
    Tensor to(Data_Type target_type) const;
    Tensor toFp16() const;
    Tensor toFp32() const;
    Tensor(std::initializer_list<size_t> shape_list, Execution_Target target = Execution_Target::CPU);
    explicit Tensor(std::shared_ptr<Tensor_Impl> impl, Execution_Target target = Execution_Target::CPU);


    ~Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;    void initializeShape(size_t rows, size_t columns);


    void initShape(size_t rows, size_t columns)
    {
        initializeShape(rows, columns);
    }    void reshape(Shape new_shape);
    Tensor permute(const std::vector<size_t> &axes_permutation) const;
    Tensor slice(size_t axis, size_t start, size_t length) const;
    void updateSlice(size_t axis, size_t start, const Tensor &source);
    Tensor gatherRows(const std::vector<int32_t> &indices) const;
    Tensor contiguous() const;
    void ensureOutputTarget(Tensor &output) const;
    void contiguous(Tensor &output) const;
    void matmul(const Tensor &other, Tensor &output) const;
    void matdiv(const Tensor &other, Tensor &output) const;
    void add(const Tensor &other, Tensor &output) const;
    void sub(const Tensor &other, Tensor &output) const;
    void mulScalar(float scalar, Tensor &output) const;
    void divScalar(float scalar, Tensor &output) const;
    void hadamardMul(const Tensor &other, Tensor &output) const;
    void hadamardDiv(const Tensor &other, Tensor &output) const;
    void transpose(Tensor &output) const;
    void inverse(Tensor &output) const;
    void normalize(Tensor &output) const;
    void relu(Tensor &output) const;
    void reluBackward(const Tensor &output_gradient, Tensor &input_gradient) const;
    void gelu(Tensor &output) const;
    void geluBackward(const Tensor &output_gradient, Tensor &input_gradient) const;
    void softmax(Tensor &output) const;
    void softmaxBackward(const Tensor &output_gradient, Tensor &input_gradient) const;
    void matmulAdd(const Tensor &other, const Tensor &biases, Tensor &output) const;
    void sgdUpdate(const Tensor &gradient, float learning_rate, float max_gradient = 0.0F, float inv_scale = 1.0F);
    void adamUpdate(const Tensor &gradient,
                    const Tensor &first_moment,
                    const Tensor &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    size_t timestep,
                    float max_gradient = 1.0F,
                    float inv_scale = 1.0F,
                    float weight_decay = 0.0F);
    void conv2d(const Tensor &weights, const Tensor &biases, Tensor &output,
                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                uint32_t output_channels, uint32_t kernel_size,
                uint32_t stride, uint32_t padding,
                Tensor *scratch = nullptr) const;
    void conv2dBackwardInput(const Tensor &weights, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                             uint32_t kernel_size, uint32_t stride, uint32_t padding) const;
    void conv2dBackwardWeight(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient,
                              uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                              uint32_t kernel_size, uint32_t stride, uint32_t padding,
                              Tensor *im2col_scratch = nullptr) const;
    void maxpool2d(Tensor &output, Tensor &output_mask,
                   uint32_t input_height, uint32_t input_width, uint32_t channels,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding) const;
    void maxpool2dBackward(const Tensor &mask, Tensor &input_gradient,
                           uint32_t input_height, uint32_t input_width, uint32_t channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) const;
    void globalAvgPool2d(Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const;
    void globalAvgPool2dBackward(Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const;
    void batchNormForward(const Tensor &gamma, const Tensor &beta,
                          Tensor &running_mean, Tensor &running_variance,
                          Tensor &batch_mean, Tensor &batch_variance,
                          Tensor &normalized_input, Tensor &output,
                          float epsilon, float momentum, bool is_training) const;
    void batchNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input,
                           Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, float epsilon) const;
    void rmsNormForward(const Tensor &gamma, Tensor &inv_rms, Tensor &output, float epsilon = 1e-5f) const;
    void rmsNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &inv_rms,
                         Tensor &gamma_gradient, Tensor &input_gradient, bool accumulate_gamma = false) const;
    void applyRoPE(Tensor &output, uint32_t seq_len, uint32_t head_dim, int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0) const;
    void swigluForward(const Tensor &b, Tensor &output) const;
    void swigluBackward(const Tensor &output_gradient, const Tensor &b, Tensor &grad_a, Tensor &grad_b) const;
    void fusedSwiGLUForward(Tensor &output) const;
    void fusedSwiGLUBackward(const Tensor &output_gradient, Tensor &input_gradient) const;
    void flashAttentionForward(const Tensor &k, const Tensor &v, Tensor &output,
                               uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                               bool is_causal = false, float scale = 0.0f,
                               Tensor *l_stats = nullptr) const;
    void flashAttentionBackward(const Tensor &k, const Tensor &v,
                                const Tensor &o, const Tensor &do_grad,
                                Tensor &dq, Tensor &dk, Tensor &dv,
                                uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                                bool is_causal = false, float scale = 0.0f,
                                const Tensor *l_stats = nullptr) const;
    void embeddingForward(const Tensor &indices, Tensor &output) const;
    Tensor embeddingForward(const Tensor &indices) const;
    void embeddingForward(const std::vector<int32_t> &indices, Tensor &output) const;
    Tensor embeddingForward(const std::vector<int32_t> &indices) const;
    void embeddingBackward(const Tensor &indices, const Tensor &output_gradient, Tensor &weight_gradient) const;
    void embeddingBackward(const std::vector<int32_t> &indices, const Tensor &output_gradient, Tensor &weight_gradient) const;
    void singleTokenAttentionForward(const Tensor &k, const Tensor &v, Tensor &output,
                                     size_t num_heads, size_t head_dim, size_t total_seq_len) const;
    Tensor singleTokenAttentionForward(const Tensor &k, const Tensor &v,
                                       size_t num_heads, size_t head_dim, size_t total_seq_len) const;
    float fusedCrossEntropyLoss(const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens = 0) const;
    float fusedCrossEntropyLoss(const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens = 0) const;
    void linearForward(const Tensor &weights, const Tensor &biases, Tensor &output) const;
    void linearBackwardInput(const Tensor &weights, Tensor &input_gradient) const;
    void linearBackwardWeightBias(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate = false) const;
    void linearBackwardWeightAdam(const Tensor &output_gradient, Tensor &weights, Tensor &first_moment, Tensor &second_moment, Tensor &bias_gradient,
                                  float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient = 1.0F, float inv_scale = 1.0F, float weight_decay = 0.0F) const;
    void batchNorm2dForward(const Tensor &gamma, const Tensor &beta,
                            Tensor &running_mean, Tensor &running_variance,
                            Tensor &batch_mean, Tensor &batch_variance,
                            Tensor &normalized_input, Tensor &output,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const;
    void batchNorm2dBackward(const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input,
                             Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const;
    void cceLoss(const Tensor &target, Tensor &output, float epsilon = 1e-7F) const;
    void mseLoss(const Tensor &target, Tensor &output) const;
    void maeLoss(const Tensor &target, Tensor &output) const;
    void bceLoss(const Tensor &target, Tensor &output, float epsilon = 1e-7F) const;
    void huberLoss(const Tensor &target, Tensor &output, float delta = 1.0F) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    Tensor hadamardMul(const Tensor &other) const;
    Tensor hadamardDiv(const Tensor &other) const;
    Tensor transpose() const;
    Tensor inverse() const;
    Tensor normalize() const;
    Tensor relu() const;
    Tensor reluBackward(const Tensor &output_gradient) const;
    Tensor gelu() const;
    Tensor geluBackward(const Tensor &output_gradient) const;
    Tensor softmax() const;
    Tensor softmaxBackward(const Tensor &output_gradient) const;
    Tensor matmulAdd(const Tensor &other, const Tensor &biases) const;
    void concatenateColumns(const Tensor &other, Tensor &output) const;
    Tensor concatenateColumns(const Tensor &other) const;
    void concatenateRows(const Tensor &other, Tensor &output) const;
    Tensor concatenateRows(const Tensor &other) const;
    void splitColumns(size_t split_index, Tensor &result_left, Tensor &result_right) const;
    std::pair<Tensor, Tensor> splitColumns(size_t split_index) const;
    void splitRows(size_t split_index, Tensor &result_up, Tensor &result_down) const;
    std::pair<Tensor, Tensor> splitRows(size_t split_index) const;
    float getScalar() const;
    void print(size_t max_display_rows = 10, size_t max_display_columns = 10) const;
    void saveTensor(std::ofstream &output_file_stream) const;
    static Tensor loadTensor(std::ifstream &input_file_stream, Execution_Target target = Execution_Target::CPU);
    void saveMatrix(std::ofstream &output_file_stream) const;
    static Tensor loadMatrix(std::ifstream &input_file_stream, Execution_Target target = Execution_Target::CPU);
    Tensor clone() const;
    void fill(float value);
    void zero();
    void invalidateFp16Cache() noexcept;
    void prewarmFp16Cache();


    const Shape &getShape() const noexcept { return implementation->getShape(); }
    const Stride &getStrides() const noexcept { return implementation->getStrides(); }
    Storage_Handle getStorage() const { const Tensor_Impl &const_implementation = *implementation; return const_implementation.getStorage(); }
    Mutable_Storage_Handle getStorage() { return implementation->getStorage(); }    std::vector<float> getData() const;

    std::shared_ptr<Tensor_Impl> getImplementation() const noexcept { return implementation; }
    size_t getTotalElements() const noexcept { return implementation->getTotalElements(); }
    size_t getColumns() const noexcept { return implementation->getColumns(); }
    size_t getRank() const noexcept { return implementation->getRank(); }
    size_t getRows() const noexcept { return implementation->getRows(); }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    Data_Type getDataType() const noexcept { return implementation->getDataType(); }
    bool isEmpty() const noexcept { return implementation->isEmpty(); }    void uploadData(const std::vector<float> &host_data);

    void setImplementation(std::shared_ptr<Tensor_Impl> _impl) noexcept { implementation = std::move(_impl); }    void setDataType(Data_Type _type) noexcept;
    void logFp16Stats(std::string_view tensor_name, Log_Level level = Log_Level::LOG_DEBUG) const;

    void setExecutionTarget(Execution_Target new_target)
    {
        if (execution_target == new_target)
        {
            return;
        }
        Shape current_shape = getShape();
        std::vector<float> current_data = getData();
        *this = Tensor(current_shape, current_data, getDataType(), new_target);
        if (new_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
    }
};;