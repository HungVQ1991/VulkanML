#include "layer/ppo_actor_critic_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"


PPO_Actor_Critic_Layer::PPO_Actor_Critic_Layer(uint64_t _actor_output_dimension, Execution_Target _execution_target)
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

void PPO_Actor_Critic_Layer::addActorLayer(std::unique_ptr<ILayer> _layer)
{
        actor.push_back(std::move(_layer));
    }

void PPO_Actor_Critic_Layer::addCriticLayer(std::unique_ptr<ILayer> _layer)
{
        critic.push_back(std::move(_layer));
    }

Tensor PPO_Actor_Critic_Layer::forward(const Tensor &_input_tensor)
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

        actor_output_tensor.concatenateColumns(critic_output_tensor, output_tensor);

        is_forward_completed = true;
        return output_tensor;
    }

Tensor PPO_Actor_Critic_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
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

        size_t param_offset = 0;
        Tensor running_actor = _batched_input;
        for (const auto& layer : actor)
        {
            size_t param_count = layer->getPopulationParameterDims().size();
            std::vector<Tensor> sub_params(
                _batched_params.begin() + param_offset,
                _batched_params.begin() + param_offset + param_count);
            running_actor = layer->forward(running_actor, sub_params);
            param_offset += param_count;
        }

        Tensor running_critic = _batched_input;
        for (const auto& layer : critic)
        {
            size_t param_count = layer->getPopulationParameterDims().size();
            std::vector<Tensor> sub_params(
                _batched_params.begin() + param_offset,
                _batched_params.begin() + param_offset + param_count);
            running_critic = layer->forward(running_critic, sub_params);
            param_offset += param_count;
        }

        Tensor combined_output(_batched_input.getExecutionTarget());
        running_actor.concatenateColumns(running_critic, combined_output);
        return combined_output;
    }

std::unique_ptr<ILayer> PPO_Actor_Critic_Layer::clone() const
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

Tensor PPO_Actor_Critic_Layer::backward(const Tensor &_output_gradient)
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

        _output_gradient.splitColumns(static_cast<size_t>(actor_output_dimension), actor_gradient, critic_gradient);

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

void PPO_Actor_Critic_Layer::resetGradient()
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

void PPO_Actor_Critic_Layer::resetGradients()
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

void PPO_Actor_Critic_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&actor_output_dimension), sizeof(actor_output_dimension));
        uint64_t actor_count = static_cast<uint64_t>(actor.size());
        uint64_t critic_count = static_cast<uint64_t>(critic.size());
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

void PPO_Actor_Critic_Layer::saveInference(std::ofstream &_output_file_stream) const
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

void PPO_Actor_Critic_Layer::loadInference(std::ifstream &_input_file_stream)
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

void PPO_Actor_Critic_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
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

void PPO_Actor_Critic_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
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

std::function<float(std::mt19937&)> PPO_Actor_Critic_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto& layer : actor)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto& layer : critic)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }

std::vector<float> PPO_Actor_Critic_Layer::getPopulationParameter(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto &layer : actor)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto &layer : critic)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::getPopulationParameter: Parameter index out of range");
    }

std::vector<std::pair<Tensor *, Tensor *>> PPO_Actor_Critic_Layer::getParametersAndGradients()
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

const std::vector<std::unique_ptr<ILayer>> & PPO_Actor_Critic_Layer::getActor() const noexcept
{ return actor; }

const std::vector<std::unique_ptr<ILayer>> & PPO_Actor_Critic_Layer::getCritic() const noexcept
{ return critic; }

const Tensor & PPO_Actor_Critic_Layer::getCriticInputGradient() const noexcept
{ return critic_input_gradient; }

const Tensor & PPO_Actor_Critic_Layer::getActorInputGradient() const noexcept
{ return actor_input_gradient; }

const Tensor & PPO_Actor_Critic_Layer::getCriticOutput() const noexcept
{ return critic_output_tensor; }

const Tensor & PPO_Actor_Critic_Layer::getActorOutput() const noexcept
{ return actor_output_tensor; }

const Tensor & PPO_Actor_Critic_Layer::getCurrentCritic() const noexcept
{ return current_critic; }

const Tensor & PPO_Actor_Critic_Layer::getCurrentActor() const noexcept
{ return current_actor; }

const Tensor & PPO_Actor_Critic_Layer::getCriticGradient() const noexcept
{ return critic_gradient; }

const Tensor & PPO_Actor_Critic_Layer::getActorGradient() const noexcept
{ return actor_gradient; }

uint64_t PPO_Actor_Critic_Layer::getActorOutputDimension() const noexcept
{ return actor_output_dimension; }

void PPO_Actor_Critic_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        size_t current_offset = 0;
        for (auto& layer : actor)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        for (auto& layer : critic)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::setPopulationParameter: Parameter index out of range");
    }

void PPO_Actor_Critic_Layer::setCriticInputGradient(const Tensor &_tensor)
{ critic_input_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setActorInputGradient(const Tensor &_tensor)
{ actor_input_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setCriticGradient(const Tensor &_tensor)
{ critic_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setActorGradient(const Tensor &_tensor)
{ actor_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setCriticOutput(const Tensor &_tensor)
{ critic_output_tensor = _tensor; }

void PPO_Actor_Critic_Layer::setActorOutput(const Tensor &_tensor)
{ actor_output_tensor = _tensor; }

void PPO_Actor_Critic_Layer::setActorOutputDimension(uint64_t _dimension) noexcept
{ actor_output_dimension = _dimension; }

void PPO_Actor_Critic_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

void PPO_Actor_Critic_Layer::setAccumulated(bool _is_accumulated) noexcept
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

void PPO_Actor_Critic_Layer::setTrainingMode(bool _is_training)
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

void PPO_Actor_Critic_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }
