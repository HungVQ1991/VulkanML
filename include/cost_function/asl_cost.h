#pragma once

#include <cmath>
#include <ostream>
#include <istream>
#include <stdexcept>

#include "cost_function/icost_function.h"
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

    static void validateParameters(float _gp, float _gn, float _cm, float _eps)
    {
        if (!std::isfinite(_gp) || _gp < 0.0f)
        {
            throw std::invalid_argument("Asl_Cost: gamma_pos must be finite and >= 0");
        }
        if (!std::isfinite(_gn) || _gn < 0.0f)
        {
            throw std::invalid_argument("Asl_Cost: gamma_neg must be finite and >= 0");
        }
        if (!std::isfinite(_cm) || _cm < 0.0f || _cm >= 1.0f)
        {
            throw std::invalid_argument("Asl_Cost: clip_margin must be finite and in [0, 1)");
        }
        if (!std::isfinite(_eps) || _eps <= 0.0f || _eps >= 0.5f)
        {
            throw std::invalid_argument("Asl_Cost: epsilon must be finite and in (0, 0.5)");
        }
    }

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

    void saveCheckpoint(std::ostream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&gamma_pos), sizeof(gamma_pos));
        _output_file_stream.write(reinterpret_cast<const char *>(&gamma_neg), sizeof(gamma_neg));
        _output_file_stream.write(reinterpret_cast<const char *>(&clip_margin), sizeof(clip_margin));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Asl_Cost::saveCheckpoint: Failed to write checkpoint");
        }
    }

    void loadCheckpoint(std::istream &_input_file_stream) override
    {
        float gp = gamma_pos;
        float gn = gamma_neg;
        float cm = clip_margin;
        float eps = epsilon;
        _input_file_stream.read(reinterpret_cast<char *>(&gp), sizeof(gp));
        _input_file_stream.read(reinterpret_cast<char *>(&gn), sizeof(gn));
        _input_file_stream.read(reinterpret_cast<char *>(&cm), sizeof(cm));
        _input_file_stream.read(reinterpret_cast<char *>(&eps), sizeof(eps));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Asl_Cost::loadCheckpoint: Failed to read checkpoint");
        }
        validateParameters(gp, gn, cm, eps);
        gamma_pos = gp;
        gamma_neg = gn;
        clip_margin = cm;
        epsilon = eps;
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::ASL; }

    float getGammaPos() const noexcept { return gamma_pos; }
    float getGammaNeg() const noexcept { return gamma_neg; }
    float getClipMargin() const noexcept { return clip_margin; }
    float getEpsilon() const noexcept { return epsilon; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix)
    {
        if (_matrix.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Asl_Cost::setLossMatrix: TERNARY_INT2 is not supported");
        }
        loss_matrix = _matrix;
    }
    void setGammaPos(float _val)
    {
        validateParameters(_val, gamma_neg, clip_margin, epsilon);
        gamma_pos = _val;
    }
    void setGammaNeg(float _val)
    {
        validateParameters(gamma_pos, _val, clip_margin, epsilon);
        gamma_neg = _val;
    }
    void setClipMargin(float _val)
    {
        validateParameters(gamma_pos, gamma_neg, _val, epsilon);
        clip_margin = _val;
    }
    void setEpsilon(float _val)
    {
        validateParameters(gamma_pos, gamma_neg, clip_margin, _val);
        epsilon = _val;
    }
};
