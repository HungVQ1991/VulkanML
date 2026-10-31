#pragma once

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
#include "icost_function.h"
#include "math/tensor.h"

class Asl_Cost : public ICost_Function
{
private:
    float gamma_pos = 1.0f;
    float gamma_neg = 4.0f;
    float clip_margin = 0.05f;
    float epsilon = 1e-7f;
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;

public:
    explicit Asl_Cost(
        float _gamma_pos = 1.0f,
        float _gamma_neg = 4.0f,
        float _clip_margin = 0.05f,
        float _epsilon = 1e-7f,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~Asl_Cost() noexcept override = default;

    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    float computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const override;

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&gamma_pos), sizeof(gamma_pos));
        _output_file_stream.write(reinterpret_cast<const char *>(&gamma_neg), sizeof(gamma_neg));
        _output_file_stream.write(reinterpret_cast<const char *>(&clip_margin), sizeof(clip_margin));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&gamma_pos), sizeof(gamma_pos));
        _input_file_stream.read(reinterpret_cast<char *>(&gamma_neg), sizeof(gamma_neg));
        _input_file_stream.read(reinterpret_cast<char *>(&clip_margin), sizeof(clip_margin));
        _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::ASL; }

    float getGammaPos() const noexcept { return gamma_pos; }
    float getGammaNeg() const noexcept { return gamma_neg; }
    float getClipMargin() const noexcept { return clip_margin; }
    float getEpsilon() const noexcept { return epsilon; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix) { loss_matrix = _matrix; }
    void setGammaPos(float _val) noexcept { gamma_pos = _val; }
    void setGammaNeg(float _val) noexcept { gamma_neg = _val; }
    void setClipMargin(float _val) noexcept { clip_margin = _val; }
    void setEpsilon(float _val) noexcept { epsilon = _val; }
};
