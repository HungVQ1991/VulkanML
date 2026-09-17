#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class PPO_Actor_Critic_Layer : public ILayer
{
private:
    std::vector<std::unique_ptr<ILayer>> actor;
    std::vector<std::unique_ptr<ILayer>> critic;

    std::uint64_t actor_output_dimension = 0;
    Matrix input_matrix;
    Matrix output_matrix;
    Matrix actor_output_matrix;
    Matrix critic_output_matrix;
    Matrix current_actor;
    Matrix current_critic;
    Matrix actor_gradient;
    Matrix critic_gradient;
    Matrix actor_input_gradient;
    Matrix critic_input_gradient;
    Matrix input_gradient;

    bool is_forward_completed = false;
    bool is_accumulated = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    PPO_Actor_Critic_Layer(std::uint64_t _actor_output_dimension, Execution_Target _execution_target = Execution_Target::CPU)
        : actor_output_dimension(_actor_output_dimension),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          actor_output_matrix(0, 0, _execution_target),
          critic_output_matrix(0, 0, _execution_target),
          current_actor(0, 0, _execution_target),
          current_critic(0, 0, _execution_target),
          actor_gradient(0, 0, _execution_target),
          critic_gradient(0, 0, _execution_target),
          actor_input_gradient(0, 0, _execution_target),
          critic_input_gradient(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_forward_completed(false),
          is_accumulated(false),
          execution_target(_execution_target)
    {
    }

    ~PPO_Actor_Critic_Layer() noexcept override = default;

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addActorLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        actor.push_back(std::move(new_layer));
        return layer_reference;
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addCriticLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        critic.push_back(std::move(new_layer));
        return layer_reference;
    }

    void addActorLayer(std::unique_ptr<ILayer> _layer)
    {
        actor.push_back(std::move(_layer));
    }

    void addCriticLayer(std::unique_ptr<ILayer> _layer)
    {
        critic.push_back(std::move(_layer));
    }

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (_new_execution_target == execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);
        execution_target = _new_execution_target;

        for (auto &layer : actor)
        {
            layer->setExecutionTarget(_new_execution_target);
        }
        for (auto &layer : critic)
        {
            layer->setExecutionTarget(_new_execution_target);
        }

        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        actor_output_matrix.setExecutionTarget(_new_execution_target);
        critic_output_matrix.setExecutionTarget(_new_execution_target);
        current_actor.setExecutionTarget(_new_execution_target);
        current_critic.setExecutionTarget(_new_execution_target);
        actor_gradient.setExecutionTarget(_new_execution_target);
        critic_gradient.setExecutionTarget(_new_execution_target);
        actor_input_gradient.setExecutionTarget(_new_execution_target);
        critic_input_gradient.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }

    void setTrainingMode(bool _is_training) override
    {
        for (auto &layer : actor)
        {
            layer->setTrainingMode(_is_training);
        }
        for (auto &layer : critic)
        {
            layer->setTrainingMode(_is_training);
        }
    }

    Matrix forward(const Matrix &_input_matrix) override
    {
        input_matrix = _input_matrix;

        current_actor = input_matrix;
        for (auto &layer : actor)
        {
            current_actor = layer->forward(current_actor);
        }
        actor_output_matrix = current_actor;

        current_critic = input_matrix;
        for (auto &layer : critic)
        {
            current_critic = layer->forward(current_critic);
        }
        critic_output_matrix = current_critic;

        actor_output_matrix.concatenateCollumns(critic_output_matrix, output_matrix);

        is_forward_completed = true;
        return output_matrix;
    }

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "PPO_Actor_Critic_Layer::forward: expected {} batched parameter tensors, got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        std::size_t param_offset = 0;
        Tensor running_actor = _batched_input;
        for (const auto& layer : actor)
        {
            std::size_t param_count = layer->getPopulationParameterDims().size();
            std::vector<Tensor> sub_params(
                _batched_params.begin() + param_offset,
                _batched_params.begin() + param_offset + param_count);
            running_actor = layer->forward(running_actor, sub_params);
            param_offset += param_count;
        }

        Tensor running_critic = _batched_input;
        for (const auto& layer : critic)
        {
            std::size_t param_count = layer->getPopulationParameterDims().size();
            std::vector<Tensor> sub_params(
                _batched_params.begin() + param_offset,
                _batched_params.begin() + param_offset + param_count);
            running_critic = layer->forward(running_critic, sub_params);
            param_offset += param_count;
        }

        Tensor combined_output(_batched_input.getExecutionTarget());
        running_actor.concatenateCollumns(running_critic, combined_output);
        return combined_output;
    }

    std::vector<Shape> getPopulationParameterDims() const override
    {
        std::vector<Shape> total_dims;
        for (const auto& layer : actor)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        for (const auto& layer : critic)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        return total_dims;
    }

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        std::vector<bool> total_evolvable;
        for (const auto& layer : actor)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        for (const auto& layer : critic)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        return total_evolvable;
    }

    bool supportsPopulationBatch() const noexcept override { return true; }

    std::unique_ptr<ILayer> clone() const override
    {
        auto cloned_layer = std::make_unique<PPO_Actor_Critic_Layer>(actor_output_dimension, execution_target);
        for (const auto& layer : actor)
        {
            cloned_layer->addActorLayer(layer->clone());
        }
        for (const auto& layer : critic)
        {
            cloned_layer->addCriticLayer(layer->clone());
        }
        return cloned_layer;
    }

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override
    {
        std::size_t current_offset = 0;
        for (const auto& layer : actor)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto& layer : critic)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        std::size_t current_offset = 0;
        for (auto& layer : actor)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        for (auto& layer : critic)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::setPopulationParameter: Parameter index out of range");
    }

    bool isAccumulated() const noexcept
    {
        return is_accumulated;
    }

    void setAccumulated(bool _is_accumulated) noexcept
    {
        is_accumulated = _is_accumulated;
    }

    std::uint64_t getActorOutputDimension() const noexcept
    {
        return actor_output_dimension;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"PPO_Actor_Critic_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        if (_output_gradient.getColumns() <= actor_output_dimension || _output_gradient.getRows() != input_matrix.getRows())
        {
            Logger::logMessage(Input_Format{"PPO_Actor_Critic_Layer::backward: Output gradient dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Output gradient dimension mismatch");
        }

        _output_gradient.splitCollumns(static_cast<std::size_t>(actor_output_dimension), actor_gradient, critic_gradient);

        Matrix current_actor_gradient = actor_gradient;
        for (auto it = actor.rbegin(); it != actor.rend(); ++it)
        {
            current_actor_gradient = (*it)->backward(current_actor_gradient);
        }
        actor_input_gradient = current_actor_gradient;

        Matrix current_critic_gradient = critic_gradient;
        for (auto it = critic.rbegin(); it != critic.rend(); ++it)
        {
            current_critic_gradient = (*it)->backward(current_critic_gradient);
        }
        critic_input_gradient = current_critic_gradient;

        actor_input_gradient.add(critic_input_gradient, input_gradient);
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
        for (auto &layer : actor)
        {
            layer->resetGradient();
        }
        for (auto &layer : critic)
        {
            layer->resetGradient();
        }
    }

    bool hasParameters() const noexcept override
    {
        return true;
    }

    Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::PPO_ACTOR_CRITIC;
    }

    const Matrix &getInput() const override
    {
        return input_matrix;
    }

    const Matrix &getOutput() const override
    {
        return output_matrix;
    }

    Matrix getActorOutput() const
    {
        return actor_output_matrix;
    }

    Matrix getCriticOutput() const
    {
        return critic_output_matrix;
    }

    Execution_Target getExecutionTarget() const override
    {
        return execution_target;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        std::vector<std::pair<Matrix *, Matrix *>> all_parameters;
        for (auto &layer : actor)
        {
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        for (auto &layer : critic)
        {
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        return all_parameters;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&actor_output_dimension), sizeof(actor_output_dimension));
        std::uint64_t actor_count = static_cast<std::uint64_t>(actor.size());
        std::uint64_t critic_count = static_cast<std::uint64_t>(critic.size());
        _output_file_stream.write(reinterpret_cast<const char *>(&actor_count), sizeof(actor_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&critic_count), sizeof(critic_count));

        for (const auto &layer : actor)
        {
            layer->saveConfiguration(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            layer->saveConfiguration(_output_file_stream);
        }
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        for (const auto &layer : actor)
        {
            layer->saveInference(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            layer->saveInference(_output_file_stream);
        }
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        for (auto &layer : actor)
        {
            layer->loadInference(_input_file_stream);
        }
        for (auto &layer : critic)
        {
            layer->loadInference(_input_file_stream);
        }
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        for (const auto &layer : actor)
        {
            layer->saveCheckpoint(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            layer->saveCheckpoint(_output_file_stream);
        }
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        for (auto &layer : actor)
        {
            layer->loadCheckpoint(_input_file_stream);
        }
        for (auto &layer : critic)
        {
            layer->loadCheckpoint(_input_file_stream);
        }
    }
};