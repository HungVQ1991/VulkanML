#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "tensor_impl.h"

struct Matrix_Dimensions
{
    uint32_t batch_count = 1;
    uint32_t rows_a = 0;
    uint32_t columns_a = 0;
    uint32_t columns_b = 0;
    uint32_t broadcast_b = 0;
};

struct Elementwise_Dimensions
{
    uint32_t total_elements = 0;
    uint32_t columns = 0;
    uint32_t is_broadcast = 0;
};

struct Transpose_Dimensions
{
    uint32_t rows = 0;
    uint32_t columns = 0;
};

struct Contiguous_Push_Constants
{
    uint32_t total_elements = 0;
    uint32_t rank = 0;
    uint32_t offset_elements = 0;
    uint32_t shape_0 = 1;
    uint32_t shape_1 = 1;
    uint32_t shape_2 = 1;
    uint32_t shape_3 = 1;
    uint32_t shape_4 = 1;
    uint32_t shape_5 = 1;
    uint32_t stride_0 = 1;
    uint32_t stride_1 = 1;
    uint32_t stride_2 = 1;
    uint32_t stride_3 = 1;
    uint32_t stride_4 = 1;
    uint32_t stride_5 = 1;
};

class Gpu_Tensor_Impl : public Tensor_Impl
{
private:
    std::shared_ptr<gpu::vector> storage;
    mutable std::vector<float> host_cache;
    mutable std::shared_ptr<gpu::vector> cached_fp16_storage;
    mutable bool is_fp16_cache_dirty = true;

    static inline bool is_graph_logging_enabled = true;

public:
    static inline size_t distinct_operations_count;

    void invalidateFp16Cache() noexcept override
    {
        is_fp16_cache_dirty = true;
    }

    void prewarmFp16Cache() override
    {
        getEffectiveFp16Storage();
    }    std::shared_ptr<gpu::vector> getEffectiveFp16Storage() const;


private:
    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum pipeline_id,
                     const std::vector<std::shared_ptr<gpu::vector>> &buffers,
                     const Push_Constants_Type &push_constants,
                     uint32_t workgroup_count_x,
                     uint32_t workgroup_count_y = 1,
                     uint32_t workgroup_count_z = 1) const
    {
        if (buffers.size() > 16)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::pushToGraph: Exceeded max buffer count"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::GRAPH_RECORDING);
            throw std::runtime_error("Exceeded maximum supported buffer count");
        }

        Compute_Node node;
        node.pipeline_id = pipeline_id;
        node.buffers = buffers;

        const auto *byte_ptr = reinterpret_cast<const std::uint8_t *>(&push_constants);
        node.push_constants_data.assign(byte_ptr, byte_ptr + sizeof(Push_Constants_Type));

        node.workgroup_count_x = workgroup_count_x;
        node.workgroup_count_y = workgroup_count_y;
        node.workgroup_count_z = workgroup_count_z;

        Execution_Engine::getInstance().getCurrentGraph().addNode(std::move(node));

        if (is_graph_logging_enabled)
        {
            is_graph_logging_enabled = Logger::logMessage(
                Input_Format{"Gpu_Tensor_Impl::pushToGraph: Op={}, Dispatch=({}, {}, {}), Buffers={}",
                             magic_enum::enum_name(pipeline_id),
                             workgroup_count_x, workgroup_count_y, workgroup_count_z,
                             buffers.size()},
                Log_Level::LOG_DEBUG, true, distinct_operations_count, Log_Feature::GRAPH_RECORDING);
        }
    }

    const Gpu_Tensor_Impl &castToGpu(const Tensor_Impl &other) const noexcept
    {
        return static_cast<const Gpu_Tensor_Impl &>(other);
    }

    Gpu_Tensor_Impl &castToGpu(Tensor_Impl &other) const noexcept
    {
        return static_cast<Gpu_Tensor_Impl &>(other);
    }    std::shared_ptr<Gpu_Tensor_Impl> ensureContiguousSelf() const;


    template <typename Pipeline_Enum>
    void executeElementwise(const Tensor_Impl &other, Tensor_Impl &output, Pipeline_Enum pipeline_id, bool is_broadcast_allowed) const
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        auto &output_gpu = castToGpu(output);

        bool is_broadcast = false;
        if (is_broadcast_allowed)
        {
            bool same_shape = (shape == effective_other->shape);
            is_broadcast = (effective_other->getRows() == 1 && getColumns() == effective_other->getColumns());
            if (!same_shape && !is_broadcast)
            {
                validateSameDimensions(*effective_other);
            }
        }
        else
        {
            validateSameDimensions(*effective_other);
        }

        bool is_fp16_pipe = (pipeline_id == Compute_Pipeline::ADD_FP16 ||
                             pipeline_id == Compute_Pipeline::SUB_FP16 ||
                             pipeline_id == Compute_Pipeline::HADAMARD_MUL_FP16 ||
                             pipeline_id == Compute_Pipeline::HADAMARD_DIV_FP16);
        if (is_fp16_pipe)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            output_gpu.setDataType(data_type);
        }
        output_gpu.reshape(shape);

        Elementwise_Dimensions dims{
            .total_elements = static_cast<uint32_t>(total_elements),
            .columns = static_cast<uint32_t>(getColumns()),
            .is_broadcast = static_cast<uint32_t>(is_broadcast ? 1 : 0)};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::executeElementwise: total={}, cols={}, broadcast={}",
                                        dims.total_elements, dims.columns, dims.is_broadcast},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);

        std::shared_ptr<gpu::vector> buf_self;
        std::shared_ptr<gpu::vector> buf_other;
        if (is_fp16_pipe)
        {
            buf_self = (effective_self->getDataType() == Data_Type::FLOAT16)
                           ? effective_self->storage
                           : effective_self->getEffectiveFp16Storage();
            buf_other = (effective_other->getDataType() == Data_Type::FLOAT16)
                            ? effective_other->storage
                            : effective_other->getEffectiveFp16Storage();
        }
        else
        {
            buf_self = effective_self->storage;
            buf_other = effective_other->storage;
        }

        pushToGraph(pipeline_id, {buf_self, buf_other, output_gpu.storage}, dims, (dims.total_elements + 255) / 256);
    }

public:    Gpu_Tensor_Impl(size_t rows, size_t columns);
    Gpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data);
    explicit Gpu_Tensor_Impl(Shape tensor_shape, Data_Type type = Data_Type::FLOAT32);
    Gpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data, Data_Type type = Data_Type::FLOAT32);
    Gpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<gpu::vector> existing_storage, size_t offset_bytes, Data_Type type = Data_Type::FLOAT32);


    ~Gpu_Tensor_Impl() noexcept override = default;    void reshape(size_t rows, size_t columns) override;
    void reshape(Shape new_shape) override;
    void permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const override;
    void slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const override;
    void updateSlice(size_t axis, size_t start, const Tensor_Impl &source) override;
    void gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const override;
    void contiguous(Tensor_Impl &output) const override;
    void to(Data_Type target_type, Tensor_Impl &output) const override;
    void matmul(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void add(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void sub(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void mulScalar(float scalar, Tensor_Impl &output) const override;
    void divScalar(float scalar, Tensor_Impl &output) const override;
    void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void transpose(Tensor_Impl &output) const override;
    void inverse(Tensor_Impl &output) const override;
    void normalize(Tensor_Impl &output) const override;
    void relu(Tensor_Impl &output) const override;
    void reluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void gelu(Tensor_Impl &output) const override;
    void geluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void softmax(Tensor_Impl &output) const override;
    void softmaxBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient = 0.0F, float inv_scale = 1.0F) override;
    void adamUpdate(const Tensor_Impl &gradient,
                    const Tensor_Impl &first_moment,
                    const Tensor_Impl &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    size_t timestep,
                    float max_gradient = 1.0F,
                    float inv_scale = 1.0F,
                    float weight_decay = 0.0F) override;
    void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override;
    void uploadData(const std::vector<float> &host_data) override;
    void zero() override;
    void fill(float value) override;
    void conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output,
                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                uint32_t output_channels, uint32_t kernel_size,
                uint32_t stride, uint32_t padding,
                Tensor_Impl *scratch = nullptr) const override;
    void conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                             uint32_t kernel_size, uint32_t stride, uint32_t padding) const override;
    void conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient,
                              uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                              uint32_t kernel_size, uint32_t stride, uint32_t padding,
                              Tensor_Impl *im2col_scratch = nullptr) const override;
    void maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask,
                   uint32_t input_height, uint32_t input_width, uint32_t channels,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding) const override;
    void maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient,
                           uint32_t input_height, uint32_t input_width, uint32_t channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) const override;
    void globalAvgPool2d(Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const override;
    void globalAvgPool2dBackward(Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const override;
    void batchNormForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                          Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                          Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                          Tensor_Impl &normalized_input, Tensor_Impl &output,
                          float epsilon, float momentum, bool is_training) const override;
    void batchNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                           Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, float epsilon) const override;
    void linearForward(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override;
    void linearBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient) const override;
    void linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient, bool accumulate = false) const override;
    void linearBackwardWeightAdam(
        const Tensor_Impl &output_gradient,
        Tensor_Impl &weights,
        Tensor_Impl &first_moment,
        Tensor_Impl &second_moment,
        Tensor_Impl &bias_gradient,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        size_t timestep,
        float max_gradient = 1.0F,
        float inv_scale = 1.0F,
        float weight_decay = 0.0F) override;
    void batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                            Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                            Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                            Tensor_Impl &normalized_input, Tensor_Impl &output,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const override;
    void batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                             Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const override;
    void cceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override;
    void mseLoss(const Tensor_Impl &target, Tensor_Impl &output) const override;
    void maeLoss(const Tensor_Impl &target, Tensor_Impl &output) const override;
    void bceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override;
    void huberLoss(const Tensor_Impl &target, Tensor_Impl &output, float delta) const override;
    void concatenateColumns(const Tensor_Impl& other, Tensor_Impl& output) const override;
    void concatenateRows(const Tensor_Impl& other, Tensor_Impl& output) const override;
    void splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override;
    void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override;
    void rmsNormForward(const Tensor_Impl &gamma, Tensor_Impl &inv_rms, Tensor_Impl &output, float epsilon) const override;
    void rmsNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &inv_rms,
                         Tensor_Impl &gamma_gradient, Tensor_Impl &input_gradient, bool accumulate_gamma = false) const override;
    void applyRoPE(Tensor_Impl &output, uint32_t seq_len, uint32_t head_dim, int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0) const override;
    void swigluForward(const Tensor_Impl &b, Tensor_Impl &output) const override;
    void swigluBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &b, Tensor_Impl &grad_a, Tensor_Impl &grad_b) const override;
    void fusedSwiGLUForward(Tensor_Impl &output) const override;
    void fusedSwiGLUBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void flashAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output,
                               uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                               bool is_causal = false, float scale = 0.0f,
                               Tensor_Impl *l_stats = nullptr) const override;
    void flashAttentionBackward(const Tensor_Impl &k, const Tensor_Impl &v,
                                const Tensor_Impl &o, const Tensor_Impl &do_grad,
                                Tensor_Impl &dq, Tensor_Impl &dk, Tensor_Impl &dv,
                                uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                                bool is_causal = false, float scale = 0.0f,
                                const Tensor_Impl *l_stats = nullptr) const override;
    void embeddingForward(const Tensor_Impl &indices, Tensor_Impl &output) const override;
    void embeddingBackward(const Tensor_Impl &indices, const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient) const override;
    void singleTokenAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output,
                                     size_t num_heads, size_t head_dim, size_t total_seq_len) const override;
    float fusedCrossEntropyLoss(const Tensor_Impl &targets, Tensor_Impl &d_logits, uint32_t valid_tokens = 0) const override;
    const std::vector<float> &getData() const noexcept override;
    Mutable_Storage_Handle getStorage() override;


    Storage_Handle getStorage() const override { return storage; }
    std::shared_ptr<gpu::vector> getVector() override { return storage; }
    bool isEmpty() const noexcept override { return !storage || storage->isEmpty(); }

};;