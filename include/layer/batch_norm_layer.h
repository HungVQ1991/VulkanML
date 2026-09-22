#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class Batch_Norm_Layer : public ILayer
{
private:
    size_t input_dimension = 0;
    float epsilon = 1e-5f;
    float momentum = 0.1f;
    bool is_training = true;

    Tensor gamma;
    Tensor beta;
    Tensor gamma_gradient_tensor;
    Tensor beta_gradient_tensor;

    Tensor running_mean;
    Tensor running_variance;
    Tensor batch_mean;
    Tensor batch_variance;
    Tensor normalized_input;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

    void initializeParameters()
    {
        std::vector<float> gamma_data(input_dimension, 1.0f);
        std::vector<float> beta_data(input_dimension, 0.0f);
        std::vector<float> mean_data(input_dimension, 0.0f);
        std::vector<float> variance_data(input_dimension, 1.0f);

        gamma = Tensor(1, input_dimension, std::move(gamma_data), execution_target);
        beta = Tensor(1, input_dimension, std::move(beta_data), execution_target);
        gamma_gradient_tensor = Tensor(1, input_dimension, execution_target);
        beta_gradient_tensor = Tensor(1, input_dimension, execution_target);

        running_mean = Tensor(1, input_dimension, std::move(mean_data), execution_target);
        running_variance = Tensor(1, input_dimension, std::move(variance_data), execution_target);

        batch_mean = Tensor(1, input_dimension, execution_target);
        batch_variance = Tensor(1, input_dimension, execution_target);
    }

public:
    using ILayer::forward;
    explicit Batch_Norm_Layer(size_t _dimension,
                              float _epsilon = 1e-5f,
                              float _momentum = 0.1f,
                              Execution_Target _execution_target = Execution_Target::CPU)
        : input_dimension(_dimension),
          epsilon(_epsilon),
          momentum(_momentum),
          is_training(true),
          gamma(0, 0, _execution_target),
          beta(0, 0, _execution_target),
          gamma_gradient_tensor(0, 0, _execution_target),
          beta_gradient_tensor(0, 0, _execution_target),
          running_mean(0, 0, _execution_target),
          running_variance(0, 0, _execution_target),
          batch_mean(0, 0, _execution_target),
          batch_variance(0, 0, _execution_target),
          normalized_input(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
        initializeParameters();
    }

    ~Batch_Norm_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override
    {
        if (_input_tensor.getColumns() != input_dimension)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_Layer::forward: Input dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Input dimension mismatch");
        }

        Logger::logMessage(Input_Format{"Batch_Norm_Layer::forward: dimension={}, mode={}",
                                        input_dimension,
                                        is_training ? "Train" : "Eval"},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        if (input_tensor.getExecutionTarget() != execution_target)
        {
            input_tensor.setExecutionTarget(execution_target);
        }

        input_tensor.batchNormForward(
            gamma,
            beta,
            running_mean,
            running_variance,
            batch_mean,
            batch_variance,
            normalized_input,
            output_tensor,
            epsilon,
            momentum,
            is_training);

        is_forward_completed = true;
        return output_tensor;
    }

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Batch_Norm_Layer::forward: expected {} batched parameter tensors (gamma, beta, running_mean, running_variance), got {}",
                                             getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }
        Tensor batch_mean_tensor(getExecutionTarget());
        Tensor batch_variance_tensor(getExecutionTarget());
        Tensor normalized_input_tensor(getExecutionTarget());
        Tensor output_tensor_result(getExecutionTarget());

        _batched_input.batchNormForward(
            _batched_params[0],
            _batched_params[1],
            const_cast<Tensor &>(_batched_params[2]),
            const_cast<Tensor &>(_batched_params[3]),
            batch_mean_tensor,
            batch_variance_tensor,
            normalized_input_tensor,
            output_tensor_result,
            epsilon,
            momentum,
            is_training);

        return output_tensor_result;
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Batch_Norm_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        if (!is_accumulated)
        {
            input_tensor.batchNormBackward(
                _output_gradient,
                gamma,
                batch_variance,
                normalized_input,
                gamma_gradient_tensor,
                beta_gradient_tensor,
                input_gradient_tensor,
                epsilon);
        }
        else
        {
            Tensor step_gamma_grad(gamma_gradient_tensor.getShape(), execution_target);
            Tensor step_beta_grad(beta_gradient_tensor.getShape(), execution_target);
            input_tensor.batchNormBackward(
                _output_gradient,
                gamma,
                batch_variance,
                normalized_input,
                step_gamma_grad,
                step_beta_grad,
                input_gradient_tensor,
                epsilon);
            gamma_gradient_tensor = gamma_gradient_tensor + step_gamma_grad;
            beta_gradient_tensor = beta_gradient_tensor + step_beta_grad;
        }

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        return input_gradient_tensor;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Batch_Norm_Layer>(input_dimension, epsilon, momentum, execution_target);
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    void resetGradients() override
    {
        resetGradient();
        gamma_gradient_tensor.zero();
        beta_gradient_tensor.zero();
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        uint32_t feature_count = static_cast<uint32_t>(input_dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        beta = Tensor::loadTensor(_input_file_stream, execution_target);
        running_mean = Tensor::loadTensor(_input_file_stream, execution_target);
        running_variance = Tensor::loadTensor(_input_file_stream, execution_target);
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
        gamma_gradient_tensor.saveTensor(_output_file_stream);
        beta_gradient_tensor.saveTensor(_output_file_stream);
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        beta = Tensor::loadTensor(_input_file_stream, execution_target);
        running_mean = Tensor::loadTensor(_input_file_stream, execution_target);
        running_variance = Tensor::loadTensor(_input_file_stream, execution_target);
        gamma_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        beta_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
    }

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override
    {
        if (param_index == 0 || param_index == 3)
        {
            return [](std::mt19937&)
                {
                    return 1.0f;
                };
        }
        else if (param_index == 1 || param_index == 2)
        {
            return [](std::mt19937&)
                {
                    return 0.0f;
                };
        }

        return [](std::mt19937&)
            {
                return 0.0f;
            };
    }
    std::vector<float> getPopulationParameter(size_t param_index) const override
    {
        switch (param_index)
        {
        case 0:
            return gamma.getData();
        case 1:
            return beta.getData();
        case 2:
            return running_mean.getData();
        case 3:
            return running_variance.getData();
        default:
            throw std::out_of_range("Batch_Norm_Layer::getPopulationParameter: Parameter index out of range");
        }
    }
    std::vector<Shape> getPopulationParameterDims() const override { return { Shape{ 1, input_dimension }, Shape{ 1, input_dimension }, Shape{ 1, input_dimension }, Shape{ 1, input_dimension } }; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {true, true, false, false}; }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override { return {{&gamma, &gamma_gradient_tensor}, {&beta, &beta_gradient_tensor}}; }
    const Tensor &getWeightsGradient() const override { return gamma_gradient_tensor; }
    const Tensor &getGammaGradient() const noexcept { return gamma_gradient_tensor; }
    const Tensor &getBetaGradient() const noexcept { return beta_gradient_tensor; }
    const Tensor &getRunningVariance() const noexcept { return running_variance; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getNormalizedInput() const noexcept { return normalized_input; }
    const Tensor &getBatchVariance() const noexcept { return batch_variance; }
    const Tensor &getRunningMean() const noexcept { return running_mean; }
    const Tensor &getBatchMean() const noexcept { return batch_mean; }
    const Tensor &getWeights() const override { return gamma; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getBiases() const override { return beta; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getGamma() const noexcept { return gamma; }
    const Tensor &getBeta() const noexcept { return beta; }
    size_t getInputDimension() const noexcept { return input_dimension; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::BATCH_NORM; }
    float getMomentum() const noexcept { return momentum; }
    float getEpsilon() const noexcept { return epsilon; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }
    bool isTraining() const noexcept { return is_training; }

    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override
    {
        if (flat_data.size() != input_dimension)
        {
            throw std::invalid_argument("Batch_Norm_Layer::setPopulationParameter: Dimension mismatch");
        }

        switch (param_index)
        {
        case 0:
            gamma = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 1:
            beta = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 2:
            running_mean = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 3:
            running_variance = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        default:
            throw std::out_of_range("Batch_Norm_Layer::setPopulationParameter: Parameter index out of range");
        }
    }
    void setGammaGradient(const Tensor &_tensor) { gamma_gradient_tensor = _tensor; }
    void setBetaGradient(const Tensor &_tensor) { beta_gradient_tensor = _tensor; }
    void setRunningVariance(const Tensor &_tensor) { running_variance = _tensor; }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setNormalizedInput(const Tensor &_tensor) { normalized_input = _tensor; }
    void setBatchVariance(const Tensor &_tensor) { batch_variance = _tensor; }
    void setRunningMean(const Tensor &_tensor) { running_mean = _tensor; }
    void setBatchMean(const Tensor &_tensor) { batch_mean = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setGamma(const Tensor &_tensor) { gamma = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setBeta(const Tensor &_tensor) { beta = _tensor; }
    void setInputDimension(size_t _dimension) noexcept { input_dimension = _dimension; }
    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        gamma.setExecutionTarget(_new_execution_target);
        beta.setExecutionTarget(_new_execution_target);
        gamma_gradient_tensor.setExecutionTarget(_new_execution_target);
        beta_gradient_tensor.setExecutionTarget(_new_execution_target);
        running_mean.setExecutionTarget(_new_execution_target);
        running_variance.setExecutionTarget(_new_execution_target);
        batch_mean.setExecutionTarget(_new_execution_target);
        batch_variance.setExecutionTarget(_new_execution_target);
        normalized_input.setExecutionTarget(_new_execution_target);
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    void setMomentum(float _momentum) noexcept { momentum = _momentum; }
    void setEpsilon(float _epsilon) noexcept { epsilon = _epsilon; }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
    void setTrainingMode(bool _is_training) override { is_training = _is_training; }
    void setIsTraining(bool _is_training) noexcept { is_training = _is_training; }
};