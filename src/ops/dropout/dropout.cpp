#include "ops/dropout/dropout.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ops::dropout
{

void dropout(const Tensor &input, Tensor &output, Tensor &mask, float dropout_rate, bool is_training, uint64_t seed)
{
    if (!is_training || dropout_rate <= 0.0f)
    {
        output = input;
        return;
    }

    if (dropout_rate >= 1.0f)
    {
        throw std::invalid_argument("ops::dropout: dropout_rate must be strictly less than 1.0f");
    }

    const size_t total_elements = input.getShape().getTotalElements();
    if (total_elements == 0)
    {
        output = input;
        mask = Tensor(input.getShape(), input.getExecutionTarget());
        return;
    }

    const float scale = 1.0f / (1.0f - dropout_rate);
    std::vector<float> mask_data(total_elements);

    std::mt19937_64 rng(seed != 0 ? seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_elements; ++i)
    {
        mask_data[i] = (dist(rng) >= dropout_rate) ? scale : 0.0f;
    }

    mask = Tensor(input.getShape(), mask_data, input.getExecutionTarget());

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const std::vector<float> input_data = input.getData();
        std::vector<float> output_data(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            output_data[i] = input_data[i] * mask_data[i];
        }
        output = Tensor(input.getShape(), std::move(output_data), Execution_Target::CPU);
    }
    else
    {
        input.hadamardMul(mask, output);
    }
}

void dropoutBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient, float dropout_rate)
{
    if (dropout_rate <= 0.0f || mask.getShape().getTotalElements() == 0)
    {
        input_gradient = output_gradient;
        return;
    }

    if (output_gradient.getExecutionTarget() == Execution_Target::CPU)
    {
        const std::vector<float> grad_data = output_gradient.getData();
        const std::vector<float> mask_data = mask.getData();
        const size_t total_elements = grad_data.size();
        std::vector<float> in_grad_data(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            in_grad_data[i] = grad_data[i] * mask_data[i];
        }
        input_gradient = Tensor(output_gradient.getShape(), std::move(in_grad_data), Execution_Target::CPU);
    }
    else
    {
        output_gradient.hadamardMul(mask, input_gradient);
    }
}

} // namespace ops::dropout
