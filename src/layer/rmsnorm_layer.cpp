#include "layer/rmsnorm_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "math/tensor_impl.h"
#include "ops/normalization/rmsnorm.h"

void RMSNorm_Layer::initializeParameters()
{
        std::vector<float> gamma_data(dimension, 1.0f);
        gamma = Tensor(1, dimension, std::move(gamma_data), execution_target);
        gamma_gradient_tensor = Tensor(1, dimension, execution_target);
        gamma_gradient_tensor.zero();
    }

RMSNorm_Layer::RMSNorm_Layer(size_t _dimension, float _epsilon, Execution_Target _execution_target, Data_Type _data_type)
    : dimension(_dimension),
          epsilon(_epsilon),
          gamma(0, 0, _execution_target),
          gamma_gradient_tensor(0, 0, _execution_target),
          inv_rms(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          input_tensor_fp16(0, 0, _execution_target),
          output_gradient_fp16(0, 0, _execution_target),
          input_gradient_tensor_fp16(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{
        if (_dimension == 0)
        {
            throw std::invalid_argument("RMSNorm_Layer: dimension must be > 0");
        }
        if (!std::isfinite(_epsilon) || _epsilon <= 0.0f)
        {
            throw std::invalid_argument("RMSNorm_Layer: epsilon must be finite and > 0");
        }
        if (_data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        initializeParameters();
    }

Tensor RMSNorm_Layer::forward(const Tensor &_input_tensor)
{
        if (dimension == 0)
        {
            throw std::invalid_argument("RMSNorm_Layer::forward: dimension must be > 0");
        }
        if (!std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::invalid_argument("RMSNorm_Layer::forward: epsilon must be finite and > 0");
        }
        if (!_input_tensor.getImplementation())
        {
            throw std::invalid_argument("RMSNorm_Layer::forward: Input tensor has null implementation");
        }
        if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("RMSNorm_Layer::forward: TERNARY_INT2 input is not supported");
        }
        if (_input_tensor.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("RMSNorm_Layer::forward: Execution_Target mismatch");
        }

        const Shape orig_shape = _input_tensor.getShape();
        const size_t rank = orig_shape.getRank();
        if (rank < 1)
        {
            throw std::invalid_argument("RMSNorm_Layer::forward: Input rank must be >= 1");
        }
        const size_t in_features = orig_shape[rank - 1];
        if (in_features != dimension)
        {
            Logger::logMessage(Input_Format{"RMSNorm_Layer::forward: Dimension mismatch (expected {}, got {})",
                                            dimension, in_features},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::ACTIVATION_COMPUTE);
            throw std::invalid_argument(std::format("RMSNorm input dimension mismatch (expected {}, got {})", dimension, in_features));
        }

        cache_orig_shape = orig_shape;
        cache_is_3d = (rank != 2);

        Tensor eff_input = _input_tensor;
        if (cache_is_3d)
        {
            size_t outer_rows = eff_input.getTotalElements() / dimension;
            eff_input = eff_input.view(Shape{ outer_rows, dimension });
        }

        input_tensor = eff_input;
        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *norm_input = &input_tensor;
        if (use_amp)
        {
            if (input_tensor.getDataType() != Data_Type::FLOAT16)
            {
                input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                norm_input = &input_tensor_fp16;
            }
            else
            {
                input_tensor_fp16 = input_tensor;
            }
        }
        ops::normalization::rmsNormForward(*norm_input, gamma, inv_rms, output_tensor, epsilon);

        if (cache_is_3d)
        {
            output_tensor.reshape(cache_orig_shape);
        }

        is_forward_completed = true;
        return output_tensor;
    }

Tensor RMSNorm_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            throw std::logic_error("RMSNorm_Layer::backward called before forward");
        }
        if (dimension == 0)
        {
            throw std::invalid_argument("RMSNorm_Layer::backward: dimension must be > 0");
        }
        if (!_output_gradient.getImplementation())
        {
            throw std::invalid_argument("RMSNorm_Layer::backward: Output gradient has null implementation");
        }
        if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("RMSNorm_Layer::backward: TERNARY_INT2 gradient is not supported");
        }
        if (_output_gradient.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("RMSNorm_Layer::backward: Execution_Target mismatch");
        }
        if (_output_gradient.getShape() != cache_orig_shape)
        {
            throw std::invalid_argument("RMSNorm_Layer::backward: Output gradient shape mismatch");
        }

        Tensor eff_grad = _output_gradient;
        if (cache_is_3d)
        {
            size_t outer_rows = eff_grad.getTotalElements() / dimension;
            eff_grad = eff_grad.view(Shape{ outer_rows, dimension });
        }

        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *norm_in = &input_tensor;
        const Tensor *norm_grad = &eff_grad;
        if (use_amp)
        {
            if (input_tensor.getDataType() != Data_Type::FLOAT16)
            {
                input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                norm_in = &input_tensor_fp16;
            }
            if (eff_grad.getDataType() != Data_Type::FLOAT16)
            {
                eff_grad.to(Data_Type::FLOAT16, output_gradient_fp16);
                norm_grad = &output_gradient_fp16;
            }
        }
        Tensor &target_in_grad = use_amp ? input_gradient_tensor_fp16 : input_gradient_tensor;

        if (is_trainable)
        {
            ops::normalization::rmsNormBackward(*norm_in, *norm_grad, gamma, inv_rms,
                                                gamma_gradient_tensor, target_in_grad, is_accumulated);
        }
        else
        {
            Tensor unused_gamma_grad(1, dimension, execution_target);
            ops::normalization::rmsNormBackward(*norm_in, *norm_grad, gamma, inv_rms,
                                                unused_gamma_grad, target_in_grad, false);
        }

        if (use_amp)
        {
            if (input_tensor.getDataType() == Data_Type::FLOAT16)
            {
                input_gradient_tensor = input_gradient_tensor_fp16;
            }
            else
            {
                input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
            }
        }

        if (cache_is_3d)
        {
            input_gradient_tensor.reshape(cache_orig_shape);
        }

        return input_gradient_tensor;
    }

Tensor RMSNorm_Layer::forwardResidualAdd(const Tensor &_residual, const Tensor &_branch, float _scale, Tensor &_out_residual)
{
        if (dimension == 0)
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: dimension must be > 0");
        }
        if (!std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: epsilon must be finite and > 0");
        }
        if (!_residual.getImplementation() || !_branch.getImplementation())
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: Input tensor has null implementation");
        }
        if (_residual.getDataType() == Data_Type::TERNARY_INT2 || _branch.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: TERNARY_INT2 input is not supported");
        }
        if (_residual.getExecutionTarget() != execution_target || _branch.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: Execution_Target mismatch");
        }
        if (_residual.getShape() != _branch.getShape())
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: residual and branch shapes must match");
        }

        const Shape orig_shape = _residual.getShape();
        const size_t rank = orig_shape.getRank();
        if (rank < 1)
        {
            throw std::invalid_argument("RMSNorm_Layer::forwardResidualAdd: Input rank must be >= 1");
        }
        const size_t in_features = orig_shape[rank - 1];
        if (in_features != dimension)
        {
            throw std::invalid_argument(std::format("RMSNorm input dimension mismatch (expected {}, got {})", dimension, in_features));
        }

        cache_orig_shape = orig_shape;
        cache_is_3d = (rank != 2);

        const size_t outer_rows = _residual.getTotalElements() / dimension;
        Tensor eff_res = cache_is_3d ? _residual.view(Shape{ outer_rows, dimension }) : _residual;
        Tensor eff_branch = cache_is_3d ? _branch.view(Shape{ outer_rows, dimension }) : _branch;

        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *res_ptr = &eff_res;
        const Tensor *branch_ptr = &eff_branch;
        Tensor res_fp16_tmp(execution_target);
        Tensor branch_fp16_tmp(execution_target);
        if (use_amp)
        {
            if (eff_res.getDataType() != Data_Type::FLOAT16)
            {
                eff_res.to(Data_Type::FLOAT16, res_fp16_tmp);
                res_ptr = &res_fp16_tmp;
            }
            if (eff_branch.getDataType() != Data_Type::FLOAT16)
            {
                eff_branch.to(Data_Type::FLOAT16, branch_fp16_tmp);
                branch_ptr = &branch_fp16_tmp;
            }
        }

        Tensor out_res_2d = _out_residual;
        if (cache_is_3d && out_res_2d.getImplementation() && out_res_2d.getExecutionTarget() == execution_target &&
            out_res_2d.getTotalElements() == outer_rows * dimension &&
            out_res_2d.getImplementation()->isContiguous() && out_res_2d.getImplementation()->getByteOffset() == 0)
        {
            out_res_2d = out_res_2d.view(Shape{ outer_rows, dimension });
        }

        ops::normalization::residualScaleRmsNormForward(*res_ptr, *branch_ptr, gamma, _scale, out_res_2d, inv_rms, output_tensor, epsilon);

        input_tensor = out_res_2d;
        if (use_amp && out_res_2d.getDataType() == Data_Type::FLOAT16)
        {
            input_tensor_fp16 = out_res_2d;
        }

        _out_residual = out_res_2d;
        if (cache_is_3d)
        {
            _out_residual.reshape(cache_orig_shape);
            output_tensor.reshape(cache_orig_shape);
        }

        is_forward_completed = true;
        return output_tensor;
    }

void RMSNorm_Layer::backwardResidualAdd(const Tensor &_output_gradient, const Tensor &_residual_grad,
                                        Tensor &_out_sum, Tensor *_out_scaled, float _scale)
{
        if (!is_forward_completed)
        {
            throw std::logic_error("RMSNorm_Layer::backwardResidualAdd called before forward");
        }
        if (dimension == 0)
        {
            throw std::invalid_argument("RMSNorm_Layer::backwardResidualAdd: dimension must be > 0");
        }
        if (!_output_gradient.getImplementation() || !_residual_grad.getImplementation())
        {
            throw std::invalid_argument("RMSNorm_Layer::backwardResidualAdd: Gradient tensor has null implementation");
        }
        if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2 || _residual_grad.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("RMSNorm_Layer::backwardResidualAdd: TERNARY_INT2 gradient is not supported");
        }
        if (_output_gradient.getExecutionTarget() != execution_target || _residual_grad.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("RMSNorm_Layer::backwardResidualAdd: Execution_Target mismatch");
        }
        if (_output_gradient.getShape() != cache_orig_shape || _residual_grad.getShape() != cache_orig_shape)
        {
            throw std::invalid_argument("RMSNorm_Layer::backwardResidualAdd: Gradient shape mismatch");
        }

        const size_t outer_rows = _output_gradient.getTotalElements() / dimension;
        Tensor eff_grad = cache_is_3d ? _output_gradient.view(Shape{ outer_rows, dimension }) : _output_gradient;
        Tensor eff_res_grad = cache_is_3d ? _residual_grad.view(Shape{ outer_rows, dimension }) : _residual_grad;

        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *norm_in = &input_tensor;
        const Tensor *norm_grad = &eff_grad;
        const Tensor *res_grad_ptr = &eff_res_grad;
        Tensor res_grad_fp16_tmp(execution_target);
        if (use_amp)
        {
            if (input_tensor.getDataType() != Data_Type::FLOAT16)
            {
                input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                norm_in = &input_tensor_fp16;
            }
            if (eff_grad.getDataType() != Data_Type::FLOAT16)
            {
                eff_grad.to(Data_Type::FLOAT16, output_gradient_fp16);
                norm_grad = &output_gradient_fp16;
            }
            if (eff_res_grad.getDataType() != Data_Type::FLOAT16)
            {
                eff_res_grad.to(Data_Type::FLOAT16, res_grad_fp16_tmp);
                res_grad_ptr = &res_grad_fp16_tmp;
            }
        }

        Tensor unused_gamma_grad(1, dimension, execution_target);
        Tensor &eff_gamma_grad = is_trainable ? gamma_gradient_tensor : unused_gamma_grad;
        const bool eff_accum = is_trainable ? is_accumulated : false;

        if (use_amp && input_tensor.getDataType() != Data_Type::FLOAT16)
        {
            Tensor sum_fp16(execution_target);
            Tensor scaled_fp16(execution_target);
            ops::normalization::rmsNormBackwardResidual(*norm_in, *norm_grad, gamma, inv_rms, *res_grad_ptr,
                                                        eff_gamma_grad, sum_fp16,
                                                        (_out_scaled != nullptr) ? &scaled_fp16 : nullptr,
                                                        _scale, eff_accum);
            sum_fp16.to(Data_Type::FLOAT32, _out_sum);
            if (_out_scaled != nullptr)
            {
                scaled_fp16.to(Data_Type::FLOAT32, *_out_scaled);
            }
        }
        else
        {
            ops::normalization::rmsNormBackwardResidual(*norm_in, *norm_grad, gamma, inv_rms, *res_grad_ptr,
                                                        eff_gamma_grad, _out_sum, _out_scaled, _scale, eff_accum);
        }

        if (cache_is_3d)
        {
            _out_sum.reshape(cache_orig_shape);
            if (_out_scaled != nullptr)
            {
                _out_scaled->reshape(cache_orig_shape);
            }
        }
    }

void RMSNorm_Layer::copyStateTo(RMSNorm_Layer &_dst) const
{
        _dst.gamma = gamma.clone();
        _dst.gamma_gradient_tensor = gamma_gradient_tensor.clone();
        copyBaseStateTo(_dst);
    }

std::unique_ptr<ILayer> RMSNorm_Layer::clone() const
{
        auto copy = std::make_unique<RMSNorm_Layer>(dimension, epsilon, execution_target);
        copyStateTo(*copy);
        return copy;
    }

void RMSNorm_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void RMSNorm_Layer::resetGradients()
{
        resetGradient();
        gamma_gradient_tensor.zero();
    }

void RMSNorm_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::saveConfiguration: Output stream is not open or writable");
        }
        if (dimension > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("RMSNorm_Layer::saveConfiguration: dimension exceeds UINT32_MAX");
        }
        uint32_t feature_count = static_cast<uint32_t>(dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::saveConfiguration: Failed writing configuration");
        }
    }

void RMSNorm_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::saveInference: Output stream is not open or writable");
        }
        gamma.saveTensor(_output_file_stream);
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::saveInference: Failed writing inference state");
        }
    }

void RMSNorm_Layer::loadInference(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::loadInference: Input stream is not open or readable");
        }
        Tensor loaded_g = Tensor::loadTensor(_input_file_stream, execution_target);
        if (!_input_file_stream)
        {
            throw std::runtime_error("RMSNorm_Layer::loadInference: Failed reading inference state");
        }
        if (!loaded_g.getImplementation() ||
            loaded_g.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_g.getTotalElements() != dimension)
        {
            throw std::runtime_error("RMSNorm_Layer::loadInference: Loaded gamma shape or dtype mismatch");
        }
        gamma = std::move(loaded_g);
        gamma.invalidateFp16Cache();
    }

void RMSNorm_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::saveCheckpoint: Output stream is not open or writable");
        }
        gamma.saveTensor(_output_file_stream);
        gamma_gradient_tensor.saveTensor(_output_file_stream);
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::saveCheckpoint: Failed writing checkpoint state");
        }
    }

void RMSNorm_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("RMSNorm_Layer::loadCheckpoint: Input stream is not open or readable");
        }
        Tensor loaded_g = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_gg = Tensor::loadTensor(_input_file_stream, execution_target);
        if (!_input_file_stream)
        {
            throw std::runtime_error("RMSNorm_Layer::loadCheckpoint: Failed reading checkpoint state");
        }
        if (!loaded_g.getImplementation() ||
            loaded_g.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_g.getTotalElements() != dimension ||
            !loaded_gg.getImplementation() ||
            loaded_gg.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_gg.getTotalElements() != dimension)
        {
            throw std::runtime_error("RMSNorm_Layer::loadCheckpoint: Loaded tensor shape or dtype mismatch");
        }
        gamma = std::move(loaded_g);
        gamma_gradient_tensor = std::move(loaded_gg);
        gamma.invalidateFp16Cache();
    }

std::vector<std::pair<Tensor *, Tensor *>> RMSNorm_Layer::getParametersAndGradients()
{
        if (!is_trainable)
        {
            return {};
        }
        return {{&gamma, &gamma_gradient_tensor}};
    }

size_t RMSNorm_Layer::getDimension() const noexcept
{ return dimension; }

void RMSNorm_Layer::setExecutionTarget(Execution_Target _new_target)
{
        if (execution_target == _new_target) return;
        logChangeExecutionTarget(_new_target);
        execution_target = _new_target;
        gamma.setExecutionTarget(_new_target);
        gamma_gradient_tensor.setExecutionTarget(_new_target);
        inv_rms.setExecutionTarget(_new_target);
        input_tensor.setExecutionTarget(_new_target);
        output_tensor.setExecutionTarget(_new_target);
        input_gradient_tensor.setExecutionTarget(_new_target);
        input_tensor_fp16 = Tensor(0, 0, _new_target);
        output_gradient_fp16 = Tensor(0, 0, _new_target);
        input_gradient_tensor_fp16 = Tensor(0, 0, _new_target);
    }

void RMSNorm_Layer::setMixedPrecision(bool _enable) noexcept
{
        is_mixed_precision_enabled = _enable;
        gamma.invalidateFp16Cache();
    }

void RMSNorm_Layer::setEpsilon(float _epsilon)
{
        if (!std::isfinite(_epsilon) || _epsilon <= 0.0f)
        {
            throw std::invalid_argument("RMSNorm_Layer: epsilon must be finite and > 0");
        }
        epsilon = _epsilon;
    }

void RMSNorm_Layer::invalidateWeightCache() noexcept
{
        gamma.invalidateFp16Cache();
    }
