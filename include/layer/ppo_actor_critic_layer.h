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
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor actor_output_tensor;
    Tensor critic_output_tensor;
    Tensor current_actor;
    Tensor current_critic;
    Tensor actor_gradient;
    Tensor critic_gradient;
    Tensor actor_input_gradient;
    Tensor critic_input_gradient;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    PPO_Actor_Critic_Layer(std::uint64_t _actor_output_dimension, Execution_Target _execution_target = Execution_Target::CPU)
        : actor_output_dimension(_actor_output_dimension),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          actor_output_tensor(0, 0, _execution_target),
          critic_output_tensor(0, 0, _execution_target),
          current_actor(0, 0, _execution_target),
          current_critic(0, 0, _execution_target),
          actor_gradient(0, 0, _execution_target),
          critic_gradient(0, 0, _execution_target),
          actor_input_gradient(0, 0, _execution_target),
          critic_input_gradient(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
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


    Tensor forward(const Tensor &_input_tensor) override
    {
        input_tensor = _input_tensor;

        current_actor = input_tensor;
        for (auto &layer : actor)
        {
            current_actor = layer->forward(current_actor);
        }
        actor_output_tensor = current_actor;

        current_critic = input_tensor;
        for (auto &layer : critic)
        {
            current_critic = layer->forward(current_critic);
        }
        critic_output_tensor = current_critic;

        actor_output_tensor.concatenateCollumns(critic_output_tensor, output_tensor);

        is_forward_completed = true;
        return output_tensor;
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


    Tensor backward(const Tensor &_output_gradient) override
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

        if (_output_gradient.getColumns() <= actor_output_dimension || _output_gradient.getRows() != input_tensor.getRows())
        {
            Logger::logMessage(Input_Format{"PPO_Actor_Critic_Layer::backward: Output gradient dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Output gradient dimension mismatch");
        }

        _output_gradient.splitCollumns(static_cast<std::size_t>(actor_output_dimension), actor_gradient, critic_gradient);

        Tensor current_actor_gradient = actor_gradient;
        for (auto it = actor.rbegin(); it != actor.rend(); ++it)
        {
            current_actor_gradient = (*it)->backward(current_actor_gradient);
        }
        actor_input_gradient = current_actor_gradient;

        Tensor current_critic_gradient = critic_gradient;
        for (auto it = critic.rbegin(); it != critic.rend(); ++it)
        {
            current_critic_gradient = (*it)->backward(current_critic_gradient);
        }
        critic_input_gradient = current_critic_gradient;

        actor_input_gradient.add(critic_input_gradient, input_gradient_tensor);
        return input_gradient_tensor;
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

    void resetGradients() override
    {
        resetGradient();
        for (auto &layer : actor)
        {
            layer->resetGradients();
        }
        for (auto &layer : critic)
        {
            layer->resetGradients();
        }
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
            Layer_Type type = layer->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
            layer->saveConfiguration(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            Layer_Type type = layer->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
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
    std::vector<float> getPopulationParameter(std::size_t param_index) const override
    {
        std::size_t current_offset = 0;
        for (const auto &layer : actor)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto &layer : critic)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::getPopulationParameter: Parameter index out of range");
    }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override
    {
        std::vector<std::pair<Tensor *, Tensor *>> all_parameters;
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
    const std::vector<std::unique_ptr<ILayer>> &getActor() const noexcept { return actor; }
    const std::vector<std::unique_ptr<ILayer>> &getCritic() const noexcept { return critic; }
    const Tensor &getCriticInputGradient() const noexcept { return critic_input_gradient; }
    const Tensor &getActorInputGradient() const noexcept { return actor_input_gradient; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getCriticOutput() const noexcept { return critic_output_tensor; }
    const Tensor &getActorOutput() const noexcept { return actor_output_tensor; }
    const Tensor &getCurrentCritic() const noexcept { return current_critic; }
    const Tensor &getCurrentActor() const noexcept { return current_actor; }
    const Tensor &getCriticGradient() const noexcept { return critic_gradient; }
    const Tensor &getActorGradient() const noexcept { return actor_gradient; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    std::uint64_t getActorOutputDimension() const noexcept { return actor_output_dimension; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::PPO_ACTOR_CRITIC; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    bool supportsPopulationBatch() const noexcept override
    {
        for (const auto& layer : actor)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        for (const auto& layer : critic)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        return true;
    }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }

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
    void setCriticInputGradient(const Tensor &_tensor) { critic_input_gradient = _tensor; }
    void setActorInputGradient(const Tensor &_tensor) { actor_input_gradient = _tensor; }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setCriticGradient(const Tensor &_tensor) { critic_gradient = _tensor; }
    void setActorGradient(const Tensor &_tensor) { actor_gradient = _tensor; }
    void setCriticOutput(const Tensor &_tensor) { critic_output_tensor = _tensor; }
    void setActorOutput(const Tensor &_tensor) { actor_output_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setActorOutputDimension(std::uint64_t _dimension) noexcept { actor_output_dimension = _dimension; }
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

        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        actor_output_tensor.setExecutionTarget(_new_execution_target);
        critic_output_tensor.setExecutionTarget(_new_execution_target);
        current_actor.setExecutionTarget(_new_execution_target);
        current_critic.setExecutionTarget(_new_execution_target);
        actor_gradient.setExecutionTarget(_new_execution_target);
        critic_gradient.setExecutionTarget(_new_execution_target);
        actor_input_gradient.setExecutionTarget(_new_execution_target);
        critic_input_gradient.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    void setAccumulated(bool _is_accumulated) noexcept override
    {
        is_accumulated = _is_accumulated;
        for (auto &layer : actor)
        {
            layer->setAccumulated(_is_accumulated);
        }
        for (auto &layer : critic)
        {
            layer->setAccumulated(_is_accumulated);
        }
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
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
};