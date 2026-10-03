#include "cost_function/fused_cce_cost.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"


Fused_Cross_Entropy::Fused_Cross_Entropy(int32_t ignore_idx) noexcept
    : ignore_index(ignore_idx)
{
    }

float Fused_Cross_Entropy::computeLoss(const Tensor& prediction_tensor, const Tensor& target_tensor) const
{
        Tensor dummy_grad;
        return prediction_tensor.fusedCrossEntropyLoss(target_tensor, dummy_grad, 0);
    }

void Fused_Cross_Entropy::computeGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const
{
        prediction_tensor.fusedCrossEntropyLoss(target_tensor, gradient_out, 0);
    }

Tensor Fused_Cross_Entropy::computeGradient(const Tensor& prediction_tensor, const Tensor& target_tensor) const
{
        Tensor gradient_out;
        computeGradient(prediction_tensor, target_tensor, gradient_out);
        return gradient_out;
    }

float Fused_Cross_Entropy::computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const
{
        return prediction_tensor.fusedCrossEntropyLoss(target_tensor, gradient_out, 0);
    }

float Fused_Cross_Entropy::computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out, uint32_t valid_tokens) const
{
        return prediction_tensor.fusedCrossEntropyLoss(target_tensor, gradient_out, valid_tokens);
    }

float Fused_Cross_Entropy::computeLossAndGradient(const Tensor& prediction_tensor, const std::vector<int32_t>& target_indices, Tensor& gradient_out) const
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
