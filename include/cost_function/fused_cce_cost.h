#pragma once

#include <cstdint>
#include <fstream>
#include <vector>

#include "cost_function/icost_function.h"
#include "math/tensor.h"

class Fused_Cross_Entropy final : public ICost_Function
{
private:
    int32_t ignore_index;

public:
    explicit Fused_Cross_Entropy(int32_t ignore_idx = -1) noexcept
        : ignore_index(ignore_idx)
    {
    }

    Cost_Type getType() const noexcept override
    {
        return Cost_Type::FUSED_CCE;
    }

    bool isFused() const noexcept override
    {
        return true;
    }

    bool supportsIntegerTargets() const noexcept override
    {
        return true;
    }

    int32_t getIgnoreIndex() const noexcept
    {
        return ignore_index;
    }

    void setIgnoreIndex(int32_t ignore_idx) noexcept
    {
        ignore_index = ignore_idx;
    }

    float computeLoss(const Tensor& prediction_tensor, const Tensor& target_tensor) const override
    {
        Tensor dummy_grad;
        return prediction_tensor.fusedCrossEntropyLoss(target_tensor, dummy_grad, 0);
    }

    void computeGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const
    {
        prediction_tensor.fusedCrossEntropyLoss(target_tensor, gradient_out, 0);
    }

    Tensor computeGradient(const Tensor& prediction_tensor, const Tensor& target_tensor) const override
    {
        Tensor gradient_out;
        computeGradient(prediction_tensor, target_tensor, gradient_out);
        return gradient_out;
    }

    float computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const override
    {
        return prediction_tensor.fusedCrossEntropyLoss(target_tensor, gradient_out, 0);
    }

    float computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out, uint32_t valid_tokens) const
    {
        return prediction_tensor.fusedCrossEntropyLoss(target_tensor, gradient_out, valid_tokens);
    }

    float computeLossAndGradient(const Tensor& prediction_tensor, const std::vector<int32_t>& target_indices, Tensor& gradient_out) const override
    {
        if (target_indices.empty())
        {
            if (gradient_out.getShape() != prediction_tensor.getShape() ||
                gradient_out.getExecutionTarget() != prediction_tensor.getExecutionTarget() ||
                gradient_out.getDataType() != prediction_tensor.getDataType())
            {
                gradient_out = Tensor(prediction_tensor.getShape(), prediction_tensor.getDataType(), prediction_tensor.getExecutionTarget());
            }
            gradient_out.zero();
            return 0.0f;
        }

        uint32_t valid_tokens = 0;
        size_t vocab_size = (prediction_tensor.getShape().getRank() == 3)
            ? prediction_tensor.getShape()[2]
            : prediction_tensor.getColumns();

        for (int32_t id : target_indices)
        {
            if (id >= 0 && id != ignore_index && static_cast<size_t>(id) < vocab_size)
            {
                valid_tokens++;
            }
        }

        if (valid_tokens == 0)
        {
            if (gradient_out.getShape() != prediction_tensor.getShape() ||
                gradient_out.getExecutionTarget() != prediction_tensor.getExecutionTarget() ||
                gradient_out.getDataType() != prediction_tensor.getDataType())
            {
                gradient_out = Tensor(prediction_tensor.getShape(), prediction_tensor.getDataType(), prediction_tensor.getExecutionTarget());
            }
            gradient_out.zero();
            return 0.0f;
        }

        if (ignore_index >= 0)
        {
            std::vector<int32_t> masked_targets = target_indices;
            for (auto& id : masked_targets)
            {
                if (id == ignore_index)
                {
                    id = -100;
                }
            }
            return prediction_tensor.fusedCrossEntropyLoss(masked_targets, gradient_out, valid_tokens);
        }

        return prediction_tensor.fusedCrossEntropyLoss(target_indices, gradient_out, valid_tokens);
    }

    void saveCheckpoint(std::ofstream& _output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char*>(&ignore_index), sizeof(ignore_index));
    }

    void loadCheckpoint(std::ifstream& _input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char*>(&ignore_index), sizeof(ignore_index));
    }
};