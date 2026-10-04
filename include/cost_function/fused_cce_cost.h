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

public:    explicit Fused_Cross_Entropy(int32_t ignore_idx = -1) noexcept;


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
    }    float computeLoss(const Tensor& prediction_tensor, const Tensor& target_tensor) const override;
    void computeGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const;
    Tensor computeGradient(const Tensor& prediction_tensor, const Tensor& target_tensor) const override;
    float computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const override;
    float computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out, uint32_t valid_tokens) const;
    float computeLossAndGradient(const Tensor& prediction_tensor, const std::vector<int32_t>& target_indices, Tensor& gradient_out) const override;


    void saveCheckpoint(std::ofstream& _output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char*>(&ignore_index), sizeof(ignore_index));
    }

    void loadCheckpoint(std::ifstream& _input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char*>(&ignore_index), sizeof(ignore_index));
    }
};;