#include "training_context.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/cost_function.h"
#include "helper/layer.h"
#include "helper/learning_rate.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/optimizer.h"

namespace
{
constexpr size_t MAX_CONFIG_RECURSION_DEPTH = 64;
constexpr uint64_t MAX_CONFIG_SUBLAYER_COUNT = 10000ULL;

template <typename T>
void readChecked(std::ifstream &_stream, T &_out, const char *_err_msg)
{
    if (!_stream.read(reinterpret_cast<char *>(&_out), sizeof(T)))
    {
        throw std::runtime_error(_err_msg);
    }
}
}

void Training_Context::createCostFunction(Cost_Type _cost_type, std::ifstream &_input_file_stream, Execution_Target _execution_target)
{
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Training_Context::createCostFunction: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Input stream is not open");
    }

    Logger::logMessage(Input_Format{"Training_Context::createCostFunction: Cost type = {}",
                                    magic_enum::enum_name(_cost_type)},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    switch (_cost_type)
    {
    case Cost_Type::MSE:
        cost_function = std::make_unique<Mse_Cost>(_execution_target);
        break;
    case Cost_Type::MAE:
        cost_function = std::make_unique<Mae_Cost>(_execution_target);
        break;
    case Cost_Type::BCE:
        cost_function = std::make_unique<Bce_Cost>(1e-7f, _execution_target);
        break;
    case Cost_Type::CCE:
        cost_function = std::make_unique<Cce_Cost>(1e-7f, _execution_target);
        break;
    case Cost_Type::HUBER:
        cost_function = std::make_unique<Huber_Cost>(1.0f, _execution_target);
        break;
    case Cost_Type::FUSED_CCE:
        cost_function = std::make_unique<Fused_Cross_Entropy>();
        break;
    case Cost_Type::ASL:
        cost_function = std::make_unique<Asl_Cost>(1.0f, 4.0f, 0.05f, 1e-7f, _execution_target);
        break;
    case Cost_Type::DICE:
        cost_function = std::make_unique<Dice_Cost>(1e-7f, _execution_target);
        break;
    case Cost_Type::DICE_BCE:
        cost_function = std::make_unique<Dice_BCE_Cost>(1.0f, 1.0f, 1e-7f, _execution_target);
        break;
    default:
        Logger::logMessage("Training_Context::createCostFunction: Unknown cost type",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Unknown cost type");
    }
    cost_function->loadCheckpoint(_input_file_stream);
}

void Training_Context::createLearningRateScheduler(Decay_Mode _decay_mode, std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Training_Context::createLearningRateScheduler: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Input stream is not open");
    }

    Logger::logMessage(Input_Format{"Training_Context::createLearningRateScheduler: Decay mode = {}",
                                    magic_enum::enum_name(_decay_mode)},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    switch (_decay_mode)
    {
    case Decay_Mode::NO_DECAY:
        learning_rate_scheduler = std::make_unique<No_Decay>();
        break;
    case Decay_Mode::STEP_DECAY:
        learning_rate_scheduler = std::make_unique<Step_Decay>();
        break;
    case Decay_Mode::MULTI_STEP_DECAY:
        learning_rate_scheduler = std::make_unique<Multi_Step_Decay>();
        break;
    case Decay_Mode::EXPONENTIAL_DECAY:
        learning_rate_scheduler = std::make_unique<Exponential_Decay>();
        break;
    case Decay_Mode::COSINE_ANNEALING:
        learning_rate_scheduler = std::make_unique<Cosine_Annealing>(0.001f, 0.0f, 1);
        break;
    case Decay_Mode::POLYNOMIAL_DECAY:
        learning_rate_scheduler = std::make_unique<Polynomial_Decay>();
        break;
    case Decay_Mode::REDUCE_ON_PLATEAU:
        learning_rate_scheduler = std::make_unique<Reduce_On_Plateau>();
        break;
    default:
        Logger::logMessage("Training_Context::createLearningRateScheduler: Unknown decay mode",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Unknown decay mode");
    }
    learning_rate_scheduler->loadCheckpoint(_input_file_stream);
    Logger::logMessage(Input_Format{"Training_Context::createLearningRateScheduler: Learning rate: {}",
                                    learning_rate_scheduler->getCurrentRate()},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);
}

void Training_Context::createOptimizer(Optimizer_Type _optimizer_type, std::ifstream &_input_file_stream, Execution_Target _execution_target)
{
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Training_Context::createOptimizer: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Input stream is not open");
    }

    if (!learning_rate_scheduler)
    {
        Logger::logMessage("Training_Context::createOptimizer: Learning rate scheduler must be created before optimizer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Learning rate scheduler must be created before optimizer");
    }

    Logger::logMessage(Input_Format{"Training_Context::createOptimizer: Optimizer type = {}",
                                    magic_enum::enum_name(_optimizer_type)},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    switch (_optimizer_type)
    {
    case Optimizer_Type::SGD_OPTIMIZER:
        optimizer = std::make_unique<Sgd_Optimizer>(*learning_rate_scheduler);
        break;
    case Optimizer_Type::ADAM_OPTIMIZER:
        optimizer = std::make_unique<Adam_Optimizer>(*learning_rate_scheduler, 0.9f, 0.999f, 1e-8f, 1.0f);
        break;
    default:
        Logger::logMessage("Training_Context::createOptimizer: Unknown optimizer type",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Unknown optimizer type");
    }

    optimizer->loadCheckpoint(_input_file_stream, _execution_target);
}

static std::unique_ptr<ILayer> constructLayerFromConfigImpl(std::ifstream &_input_file_stream, Layer_Type _layer_type, Execution_Target _execution_target, size_t _depth)
{
    if (_depth > MAX_CONFIG_RECURSION_DEPTH)
    {
        throw std::runtime_error("Training_Context::constructLayerFromConfig: Sublayer recursion depth limit exceeded");
    }
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Training_Context::constructLayerFromConfig: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Input stream is not open");
    }

    Logger::logMessage(Input_Format{"Training_Context::constructLayerFromConfig: Constructing layer type = {}",
                                    magic_enum::enum_name(_layer_type)},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    switch (_layer_type)
    {
    case Layer_Type::LINEAR:
    {
        uint32_t input_dimension = 0;
        uint32_t output_dimension = 0;
        readChecked(_input_file_stream, input_dimension, "Training_Context::constructLayerFromConfig: Truncated LINEAR input_dimension");
        readChecked(_input_file_stream, output_dimension, "Training_Context::constructLayerFromConfig: Truncated LINEAR output_dimension");
        if (input_dimension == 0 || output_dimension == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: LINEAR layer dimensions must be > 0");
        }
        return std::make_unique<Linear_Layer>(input_dimension, output_dimension, _execution_target);
    }
    case Layer_Type::LORA_LINEAR:
    {
        uint32_t input_dimension = 0;
        uint32_t output_dimension = 0;
        uint32_t saved_rank = 0;
        float saved_alpha = 1.0f;
        float saved_dropout = 0.0f;
        uint8_t flags = 0;
        readChecked(_input_file_stream, input_dimension, "Training_Context::constructLayerFromConfig: Truncated LORA_LINEAR input_dimension");
        readChecked(_input_file_stream, output_dimension, "Training_Context::constructLayerFromConfig: Truncated LORA_LINEAR output_dimension");
        readChecked(_input_file_stream, saved_rank, "Training_Context::constructLayerFromConfig: Truncated LORA_LINEAR rank");
        readChecked(_input_file_stream, saved_alpha, "Training_Context::constructLayerFromConfig: Truncated LORA_LINEAR alpha");
        readChecked(_input_file_stream, saved_dropout, "Training_Context::constructLayerFromConfig: Truncated LORA_LINEAR dropout");
        readChecked(_input_file_stream, flags, "Training_Context::constructLayerFromConfig: Truncated LORA_LINEAR flags");
        if (input_dimension == 0 || output_dimension == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: LORA_LINEAR layer dimensions must be > 0");
        }
        const bool use_bias = (flags & 0x01u) != 0;
        const bool lora_on = (flags & 0x02u) != 0 && saved_rank > 0;
        const bool base_tr = (flags & 0x08u) != 0;
        const bool layer_tr = (flags & 0x10u) != 0;
        auto layer = std::make_unique<LoRA_Linear_Layer>(
            input_dimension, output_dimension,
            lora_on ? saved_rank : 0,
            saved_alpha, saved_dropout, _execution_target, 2.0f, Data_Type::FLOAT32, use_bias);
        layer->setBaseTrainable(base_tr);
        layer->setTrainable(layer_tr);
        return layer;
    }
    case Layer_Type::TERNARY_LINEAR:
    {
        uint32_t input_dimension = 0;
        uint32_t output_dimension = 0;
        readChecked(_input_file_stream, input_dimension, "Training_Context::constructLayerFromConfig: Truncated TERNARY_LINEAR input_dimension");
        readChecked(_input_file_stream, output_dimension, "Training_Context::constructLayerFromConfig: Truncated TERNARY_LINEAR output_dimension");
        if (input_dimension == 0 || output_dimension == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: TERNARY_LINEAR layer dimensions must be > 0");
        }
        return std::make_unique<Ternary_Linear_Layer>(input_dimension, output_dimension, _execution_target);
    }
    case Layer_Type::CONV2D:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t input_channels = 0;
        uint32_t output_channels = 0;
        uint32_t kernel_size = 0;
        uint32_t stride = 0;
        uint32_t padding = 0;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        readChecked(_input_file_stream, input_channels, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        readChecked(_input_file_stream, output_channels, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        readChecked(_input_file_stream, kernel_size, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        readChecked(_input_file_stream, stride, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        readChecked(_input_file_stream, padding, "Training_Context::constructLayerFromConfig: Truncated CONV2D config");
        if (input_height == 0 || input_width == 0 || input_channels == 0 ||
            output_channels == 0 || kernel_size == 0 || stride == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid zero parameter in CONV2D configuration");
        }
        return std::make_unique<Conv2d_Layer>(input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, _execution_target);
    }
    case Layer_Type::BATCH_NORM:
    {
        uint32_t feature_count = 0;
        float epsilon = 0.0f;
        float momentum = 0.0f;
        readChecked(_input_file_stream, feature_count, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM config");
        readChecked(_input_file_stream, epsilon, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM config");
        readChecked(_input_file_stream, momentum, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM config");
        if (feature_count == 0 || !std::isfinite(epsilon) || epsilon <= 0.0f ||
            !std::isfinite(momentum) || momentum < 0.0f || momentum > 1.0f)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid BATCH_NORM configuration");
        }
        return std::make_unique<Batch_Norm_Layer>(feature_count, epsilon, momentum, _execution_target);
    }
    case Layer_Type::BATCH_NORM_2D:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t channels = 0;
        float epsilon = 0.0f;
        float momentum = 0.0f;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM_2D config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM_2D config");
        readChecked(_input_file_stream, channels, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM_2D config");
        readChecked(_input_file_stream, epsilon, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM_2D config");
        readChecked(_input_file_stream, momentum, "Training_Context::constructLayerFromConfig: Truncated BATCH_NORM_2D config");
        if (input_height == 0 || input_width == 0 || channels == 0 ||
            !std::isfinite(epsilon) || epsilon <= 0.0f ||
            !std::isfinite(momentum) || momentum < 0.0f || momentum > 1.0f)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid BATCH_NORM_2D configuration");
        }
        return std::make_unique<Batch_Norm_2d_Layer>(input_height, input_width, channels, epsilon, momentum, _execution_target);
    }
    case Layer_Type::INSTANCE_NORM_2D:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t channels = 0;
        float epsilon = 0.0f;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated INSTANCE_NORM_2D config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated INSTANCE_NORM_2D config");
        readChecked(_input_file_stream, channels, "Training_Context::constructLayerFromConfig: Truncated INSTANCE_NORM_2D config");
        readChecked(_input_file_stream, epsilon, "Training_Context::constructLayerFromConfig: Truncated INSTANCE_NORM_2D config");
        if (input_height == 0 || input_width == 0 || channels == 0 ||
            !std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid INSTANCE_NORM_2D configuration");
        }
        return std::make_unique<Instance_Norm_2d_Layer>(input_height, input_width, channels, epsilon, _execution_target);
    }
    case Layer_Type::GLOBAL_AVG_POOL_2D:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t channels = 0;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated GLOBAL_AVG_POOL_2D config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated GLOBAL_AVG_POOL_2D config");
        readChecked(_input_file_stream, channels, "Training_Context::constructLayerFromConfig: Truncated GLOBAL_AVG_POOL_2D config");
        if (input_height == 0 || input_width == 0 || channels == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid GLOBAL_AVG_POOL_2D dimensions");
        }
        return std::make_unique<Global_Avg_Pool_2d_Layer>(input_height, input_width, channels, _execution_target);
    }
    case Layer_Type::MAX_POOL_2D:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t channels = 0;
        uint32_t kernel_size = 0;
        uint32_t stride = 0;
        uint32_t padding = 0;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated MAX_POOL_2D config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated MAX_POOL_2D config");
        readChecked(_input_file_stream, channels, "Training_Context::constructLayerFromConfig: Truncated MAX_POOL_2D config");
        readChecked(_input_file_stream, kernel_size, "Training_Context::constructLayerFromConfig: Truncated MAX_POOL_2D config");
        readChecked(_input_file_stream, stride, "Training_Context::constructLayerFromConfig: Truncated MAX_POOL_2D config");
        readChecked(_input_file_stream, padding, "Training_Context::constructLayerFromConfig: Truncated MAX_POOL_2D config");
        if (input_height == 0 || input_width == 0 || channels == 0 || kernel_size == 0 || stride == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid MAX_POOL_2D configuration");
        }
        return std::make_unique<Max_Pool_2d_Layer>(input_height, input_width, channels, kernel_size, stride, padding, _execution_target);
    }
    case Layer_Type::AVG_POOL_2D:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t channels = 0;
        uint32_t kernel_size = 0;
        uint32_t stride = 0;
        uint32_t padding = 0;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated AVG_POOL_2D config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated AVG_POOL_2D config");
        readChecked(_input_file_stream, channels, "Training_Context::constructLayerFromConfig: Truncated AVG_POOL_2D config");
        readChecked(_input_file_stream, kernel_size, "Training_Context::constructLayerFromConfig: Truncated AVG_POOL_2D config");
        readChecked(_input_file_stream, stride, "Training_Context::constructLayerFromConfig: Truncated AVG_POOL_2D config");
        readChecked(_input_file_stream, padding, "Training_Context::constructLayerFromConfig: Truncated AVG_POOL_2D config");
        if (input_height == 0 || input_width == 0 || channels == 0 || kernel_size == 0 || stride == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid AVG_POOL_2D configuration");
        }
        return std::make_unique<Avg_Pool_2d_Layer>(input_height, input_width, channels, kernel_size, stride, padding, _execution_target);
    }
    case Layer_Type::MULTI_SCALE_RESNET_18:
    {
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t in_c = 0;
        uint8_t d_flag = 0;
        readChecked(_input_file_stream, height, "Training_Context::constructLayerFromConfig: Truncated MULTI_SCALE_RESNET_18 config");
        readChecked(_input_file_stream, width, "Training_Context::constructLayerFromConfig: Truncated MULTI_SCALE_RESNET_18 config");
        readChecked(_input_file_stream, in_c, "Training_Context::constructLayerFromConfig: Truncated MULTI_SCALE_RESNET_18 config");
        readChecked(_input_file_stream, d_flag, "Training_Context::constructLayerFromConfig: Truncated MULTI_SCALE_RESNET_18 config");
        if (height == 0 || width == 0 || in_c == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid MULTI_SCALE_RESNET_18 dimensions");
        }
        bool use_resnet_d = (d_flag & 0x01) != 0;
        float lambda_gmp = 0.0f;
        if (d_flag & 0x80)
        {
            readChecked(_input_file_stream, lambda_gmp, "Training_Context::constructLayerFromConfig: Truncated MULTI_SCALE_RESNET_18 lambda_gmp");
            if (!std::isfinite(lambda_gmp))
            {
                throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid lambda_gmp in MULTI_SCALE_RESNET_18");
            }
        }
        auto layer = std::make_unique<Multi_Scale_ResNet18_Balanced_Layer>(height, width, in_c, _execution_target, use_resnet_d);
        layer->setLambdaGmp(lambda_gmp);
        return layer;
    }
    case Layer_Type::RES_NET_BLOCK_2D:
    {
        uint64_t main_count = 0;
        uint64_t shortcut_count = 0;
        std::uint8_t has_post_act = 0;
        readChecked(_input_file_stream, main_count, "Training_Context::constructLayerFromConfig: Truncated RES_NET_BLOCK_2D config");
        readChecked(_input_file_stream, shortcut_count, "Training_Context::constructLayerFromConfig: Truncated RES_NET_BLOCK_2D config");
        readChecked(_input_file_stream, has_post_act, "Training_Context::constructLayerFromConfig: Truncated RES_NET_BLOCK_2D config");
        if (main_count > MAX_CONFIG_SUBLAYER_COUNT || shortcut_count > MAX_CONFIG_SUBLAYER_COUNT)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: RES_NET_BLOCK_2D sublayer count exceeds limit");
        }

        auto block = std::make_unique<Res_Net_Block_2d_Layer>(_execution_target);
        for (uint64_t i = 0; i < main_count; ++i)
        {
            Layer_Type sub_type{};
            readChecked(_input_file_stream, sub_type, "Training_Context::constructLayerFromConfig: Truncated RES_NET_BLOCK_2D main sublayer type");
            block->addMainLayer(constructLayerFromConfigImpl(_input_file_stream, sub_type, _execution_target, _depth + 1));
        }
        for (uint64_t i = 0; i < shortcut_count; ++i)
        {
            Layer_Type sub_type{};
            readChecked(_input_file_stream, sub_type, "Training_Context::constructLayerFromConfig: Truncated RES_NET_BLOCK_2D shortcut sublayer type");
            block->addShortcutLayer(constructLayerFromConfigImpl(_input_file_stream, sub_type, _execution_target, _depth + 1));
        }
        if (has_post_act != 0)
        {
            Layer_Type sub_type{};
            readChecked(_input_file_stream, sub_type, "Training_Context::constructLayerFromConfig: Truncated RES_NET_BLOCK_2D post_act type");
            block->setPostActivation(constructLayerFromConfigImpl(_input_file_stream, sub_type, _execution_target, _depth + 1));
        }
        return block;
    }
    case Layer_Type::PPO_ACTOR_CRITIC:
    {
        uint64_t actor_output_dimension = 0;
        uint64_t actor_count = 0;
        uint64_t critic_count = 0;
        readChecked(_input_file_stream, actor_output_dimension, "Training_Context::constructLayerFromConfig: Truncated PPO_ACTOR_CRITIC config");
        readChecked(_input_file_stream, actor_count, "Training_Context::constructLayerFromConfig: Truncated PPO_ACTOR_CRITIC config");
        readChecked(_input_file_stream, critic_count, "Training_Context::constructLayerFromConfig: Truncated PPO_ACTOR_CRITIC config");
        if (actor_output_dimension == 0 || actor_count > MAX_CONFIG_SUBLAYER_COUNT || critic_count > MAX_CONFIG_SUBLAYER_COUNT)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid PPO_ACTOR_CRITIC configuration");
        }

        auto ppo_layer = std::make_unique<PPO_Actor_Critic_Layer>(actor_output_dimension, _execution_target);
        for (uint64_t i = 0; i < actor_count; ++i)
        {
            Layer_Type sub_type{};
            readChecked(_input_file_stream, sub_type, "Training_Context::constructLayerFromConfig: Truncated PPO_ACTOR_CRITIC actor sublayer type");
            ppo_layer->addActorLayer(constructLayerFromConfigImpl(_input_file_stream, sub_type, _execution_target, _depth + 1));
        }
        for (uint64_t i = 0; i < critic_count; ++i)
        {
            Layer_Type sub_type{};
            readChecked(_input_file_stream, sub_type, "Training_Context::constructLayerFromConfig: Truncated PPO_ACTOR_CRITIC critic sublayer type");
            ppo_layer->addCriticLayer(constructLayerFromConfigImpl(_input_file_stream, sub_type, _execution_target, _depth + 1));
        }
        return ppo_layer;
    }
    case Layer_Type::RES_NET_20:
    {
        uint32_t input_height = 0;
        uint32_t input_width = 0;
        uint32_t input_channels = 0;
        uint32_t num_classes = 0;
        readChecked(_input_file_stream, input_height, "Training_Context::constructLayerFromConfig: Truncated RES_NET_20 config");
        readChecked(_input_file_stream, input_width, "Training_Context::constructLayerFromConfig: Truncated RES_NET_20 config");
        readChecked(_input_file_stream, input_channels, "Training_Context::constructLayerFromConfig: Truncated RES_NET_20 config");
        readChecked(_input_file_stream, num_classes, "Training_Context::constructLayerFromConfig: Truncated RES_NET_20 config");
        if (input_height == 0 || input_width == 0 || input_channels == 0 || num_classes == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid RES_NET_20 configuration");
        }
        return std::make_unique<Res_Net_20_Layer>(input_height, input_width, input_channels, num_classes, _execution_target);
    }
    case Layer_Type::SOFTMAX:
    {
        std::uint8_t is_fused_value = 0;
        readChecked(_input_file_stream, is_fused_value, "Training_Context::constructLayerFromConfig: Truncated SOFTMAX config");
        bool is_fused = (is_fused_value != 0);
        return std::make_unique<Softmax_Layer>(is_fused, _execution_target);
    }
    case Layer_Type::RELU:
    {
        return std::make_unique<Relu_Layer>(_execution_target);
    }
    case Layer_Type::LEAKY_RELU:
    {
        float negative_slope = 0.01f;
        readChecked(_input_file_stream, negative_slope, "Training_Context::constructLayerFromConfig: Truncated LEAKY_RELU config");
        if (!std::isfinite(negative_slope))
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid LEAKY_RELU negative_slope");
        }
        return std::make_unique<Leaky_Relu_Layer>(negative_slope, _execution_target);
    }
    case Layer_Type::GELU:
    {
        return std::make_unique<Gelu_Layer>(_execution_target);
    }
    case Layer_Type::SIGMOID:
    {
        return std::make_unique<Sigmoid_Layer>(_execution_target);
    }
    case Layer_Type::DROPOUT:
    {
        float dropout_rate = 0.5f;
        readChecked(_input_file_stream, dropout_rate, "Training_Context::constructLayerFromConfig: Truncated DROPOUT config");
        if (!std::isfinite(dropout_rate) || dropout_rate < 0.0f || dropout_rate >= 1.0f)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid DROPOUT dropout_rate");
        }
        return std::make_unique<Dropout_Layer>(dropout_rate, _execution_target);
    }
    case Layer_Type::RMSNORM:
    {
        uint32_t feature_count = 0;
        float epsilon = 1e-5f;
        readChecked(_input_file_stream, feature_count, "Training_Context::constructLayerFromConfig: Truncated RMSNORM config");
        readChecked(_input_file_stream, epsilon, "Training_Context::constructLayerFromConfig: Truncated RMSNORM config");
        if (feature_count == 0 || !std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid RMSNORM configuration");
        }
        return std::make_unique<RMSNorm_Layer>(feature_count, epsilon, _execution_target);
    }
    case Layer_Type::SWIGLU:
    {
        return std::make_unique<SwiGLU_Layer>(_execution_target);
    }
    case Layer_Type::EMBEDDING:
    {
        uint64_t vocab_size = 0;
        uint64_t embedding_dim = 0;
        readChecked(_input_file_stream, vocab_size, "Training_Context::constructLayerFromConfig: Truncated EMBEDDING config");
        readChecked(_input_file_stream, embedding_dim, "Training_Context::constructLayerFromConfig: Truncated EMBEDDING config");
        if (vocab_size == 0 || embedding_dim == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid EMBEDDING dimensions");
        }
        return std::make_unique<Embedding_Layer>(vocab_size, embedding_dim, _execution_target);
    }
    case Layer_Type::TRANSFORMER_BLOCK:
    {
        constexpr uint64_t TRANSFORMER_BLOCK_GQA_FLAG = (1ULL << 63);
        constexpr uint64_t TRANSFORMER_BLOCK_EXT_FLAG = (1ULL << 62);
        uint64_t hidden_dim = 0;
        uint64_t num_heads_raw = 0;
        uint64_t intermediate_dim = 0;
        float rms_norm_eps = 1e-5f;
        float rope_base = 10000.0f;
        uint64_t num_layers = 1;
        float residual_scale = 1.0f;
        readChecked(_input_file_stream, hidden_dim, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        readChecked(_input_file_stream, num_heads_raw, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        readChecked(_input_file_stream, intermediate_dim, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        readChecked(_input_file_stream, rms_norm_eps, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        readChecked(_input_file_stream, rope_base, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        readChecked(_input_file_stream, num_layers, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        readChecked(_input_file_stream, residual_scale, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK config");
        const bool has_gqa = (num_heads_raw & TRANSFORMER_BLOCK_GQA_FLAG) != 0;
        const bool has_ext = (num_heads_raw & TRANSFORMER_BLOCK_EXT_FLAG) != 0;
        const uint64_t num_heads = num_heads_raw & ~(TRANSFORMER_BLOCK_GQA_FLAG | TRANSFORMER_BLOCK_EXT_FLAG);
        uint64_t num_kv_heads = num_heads;
        if (has_gqa)
        {
            readChecked(_input_file_stream, num_kv_heads, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK num_kv_heads config");
        }
        uint64_t explicit_head_dim = 0;
        uint8_t ext_flags = 0x06u;
        if (has_ext)
        {
            readChecked(_input_file_stream, explicit_head_dim, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK head_dim config");
            readChecked(_input_file_stream, ext_flags, "Training_Context::constructLayerFromConfig: Truncated TRANSFORMER_BLOCK ext_flags config");
        }
        const bool ok_head = (explicit_head_dim > 0)
            ? ((explicit_head_dim % 2) == 0)
            : ((hidden_dim % num_heads == 0) && ((hidden_dim / num_heads) % 2 == 0));
        if (hidden_dim == 0 || num_heads == 0 || intermediate_dim == 0 || num_kv_heads == 0 ||
            num_kv_heads > num_heads || (num_heads % num_kv_heads) != 0 || !ok_head ||
            num_layers == 0 || !std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0f ||
            !std::isfinite(rope_base) || rope_base <= 0.0f || !std::isfinite(residual_scale))
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid TRANSFORMER_BLOCK configuration");
        }
        Transformer_Block_Config tb_cfg;
        tb_cfg.hidden_dim = hidden_dim;
        tb_cfg.num_heads = num_heads;
        tb_cfg.head_dim = explicit_head_dim;
        tb_cfg.num_kv_heads = num_kv_heads;
        tb_cfg.intermediate_dim = intermediate_dim;
        tb_cfg.rms_norm_eps = rms_norm_eps;
        tb_cfg.rope_base = rope_base;
        tb_cfg.execution_target = _execution_target;
        tb_cfg.data_type = Data_Type::FLOAT32;
        tb_cfg.num_layers = num_layers;
        tb_cfg.residual_scale = residual_scale;
        tb_cfg.use_qk_norm = (ext_flags & 0x01u) != 0;
        tb_cfg.use_attention_bias = (ext_flags & 0x02u) != 0;
        tb_cfg.use_mlp_bias = (ext_flags & 0x04u) != 0;
        tb_cfg.gradient_checkpointing = (ext_flags & 0x08u) != 0;
        return std::make_unique<Transformer_Block>(tb_cfg);
    }
    case Layer_Type::UP_SAMPLE_2D:
    {
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t channels = 0;
        uint32_t scale_factor = 0;
        readChecked(_input_file_stream, height, "Training_Context::constructLayerFromConfig: Truncated UP_SAMPLE_2D config");
        readChecked(_input_file_stream, width, "Training_Context::constructLayerFromConfig: Truncated UP_SAMPLE_2D config");
        readChecked(_input_file_stream, channels, "Training_Context::constructLayerFromConfig: Truncated UP_SAMPLE_2D config");
        readChecked(_input_file_stream, scale_factor, "Training_Context::constructLayerFromConfig: Truncated UP_SAMPLE_2D config");
        if (height == 0 || width == 0 || channels == 0 || scale_factor == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid UP_SAMPLE_2D configuration");
        }
        return std::make_unique<Up_Sample_2d_Layer>(height, width, channels, scale_factor, _execution_target);
    }
    case Layer_Type::U_NET:
    {
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t in_c = 0;
        uint32_t out_c = 0;
        uint32_t base_c = 0;
        readChecked(_input_file_stream, height, "Training_Context::constructLayerFromConfig: Truncated U_NET config");
        readChecked(_input_file_stream, width, "Training_Context::constructLayerFromConfig: Truncated U_NET config");
        readChecked(_input_file_stream, in_c, "Training_Context::constructLayerFromConfig: Truncated U_NET config");
        readChecked(_input_file_stream, out_c, "Training_Context::constructLayerFromConfig: Truncated U_NET config");
        readChecked(_input_file_stream, base_c, "Training_Context::constructLayerFromConfig: Truncated U_NET config");
        if (height == 0 || width == 0 || in_c == 0 || out_c == 0 || base_c == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid U_NET configuration");
        }
        return std::make_unique<U_Net_Layer>(height, width, in_c, out_c, base_c, _execution_target);
    }
    case Layer_Type::NN_UNET:
    {
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t in_c = 0;
        uint32_t out_c = 0;
        uint32_t raw_bc = 0;
        readChecked(_input_file_stream, height, "Training_Context::constructLayerFromConfig: Truncated NN_UNET config");
        readChecked(_input_file_stream, width, "Training_Context::constructLayerFromConfig: Truncated NN_UNET config");
        readChecked(_input_file_stream, in_c, "Training_Context::constructLayerFromConfig: Truncated NN_UNET config");
        readChecked(_input_file_stream, out_c, "Training_Context::constructLayerFromConfig: Truncated NN_UNET config");
        readChecked(_input_file_stream, raw_bc, "Training_Context::constructLayerFromConfig: Truncated NN_UNET config");
        const bool ds_enabled = (raw_bc & NN_UNet_Layer::NN_UNET_DS_DISABLED_FLAG) == 0u;
        const uint32_t base_c = raw_bc & ~NN_UNet_Layer::NN_UNET_DS_DISABLED_FLAG;
        if (height == 0 || width == 0 || in_c == 0 || out_c == 0 || base_c == 0)
        {
            throw std::runtime_error("Training_Context::constructLayerFromConfig: Invalid NN_UNET configuration");
        }
        auto layer = std::make_unique<NN_UNet_Layer>(height, width, in_c, out_c, base_c, _execution_target);
        layer->setEnableDeepSupervision(ds_enabled);
        return layer;
    }
    default:
        Logger::logMessage("Training_Context::constructLayerFromConfig: Unsupported layer type",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Unsupported or unknown layer type");
    }
}

std::unique_ptr<ILayer> Training_Context::constructLayerFromConfig(std::ifstream &_input_file_stream, Layer_Type _layer_type, Execution_Target _execution_target)
{
    return constructLayerFromConfigImpl(_input_file_stream, _layer_type, _execution_target, 0);
}

bool Training_Context::loadHeader(std::ifstream &_input_file_stream, Execution_Target _execution_target)
{
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Training_Context::loadHeader: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        return false;
    }

    char magic_header[4] = {};
    if (!_input_file_stream.read(magic_header, 4) ||
        magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'C' || magic_header[3] != 'K')
    {
        Logger::logMessage("Training_Context::loadHeader: Invalid or truncated magic header",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        return false;
    }

    uint32_t epoch_value = 0;
    if (!_input_file_stream.read(reinterpret_cast<char *>(&epoch_value), sizeof(epoch_value)))
    {
        return false;
    }
    current_epoch = static_cast<size_t>(epoch_value);
    Logger::logMessage(Input_Format{"Training_Context::loadHeader: Current epoch = {}", current_epoch},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    Cost_Type cost_type{};
    if (!_input_file_stream.read(reinterpret_cast<char *>(&cost_type), sizeof(cost_type)))
    {
        return false;
    }
    Logger::logMessage(Input_Format{"Training_Context::loadHeader: Cost function type = {}",
                                    magic_enum::enum_name(cost_type)},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);
    createCostFunction(cost_type, _input_file_stream, _execution_target);

    Decay_Mode decay_mode{};
    if (!_input_file_stream.read(reinterpret_cast<char *>(&decay_mode), sizeof(decay_mode)))
    {
        return false;
    }
    Logger::logMessage(Input_Format{"Training_Context::loadHeader: Decay mode = {}",
                                    magic_enum::enum_name(decay_mode)},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);
    createLearningRateScheduler(decay_mode, _input_file_stream);

    Optimizer_Type optimizer_type{};
    if (!_input_file_stream.read(reinterpret_cast<char *>(&optimizer_type), sizeof(optimizer_type)))
    {
        return false;
    }
    Logger::logMessage(Input_Format{"Training_Context::loadHeader: Optimizer type = {}",
                                    magic_enum::enum_name(optimizer_type)},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);
    createOptimizer(optimizer_type, _input_file_stream, _execution_target);

    return true;
}

void Training_Context::setLearningRateScheduler(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler)
{
    if (!_learning_rate_scheduler)
    {
        Logger::logMessage("Training_Context::setLearningRateScheduler: Null learning rate scheduler pointer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::TRAINING);
        throw std::invalid_argument("Training_Context::setLearningRateScheduler: Null learning rate scheduler pointer");
    }
    learning_rate_scheduler = std::move(_learning_rate_scheduler);
    if (optimizer)
    {
        optimizer->setLearningRateScheduler(learning_rate_scheduler.get());
    }
}

void Training_Context::setOptimizer(std::unique_ptr<IOptimizer> _optimizer)
{
    if (!_optimizer)
    {
        Logger::logMessage("Training_Context::setOptimizer: Null optimizer pointer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::TRAINING);
        throw std::invalid_argument("Training_Context::setOptimizer: Null optimizer pointer");
    }
    optimizer = std::move(_optimizer);
}

void Training_Context::setCostFunction(std::unique_ptr<ICost_Function> _cost_function)
{
    if (!_cost_function)
    {
        Logger::logMessage("Training_Context::setCostFunction: Null cost function pointer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::TRAINING);
        throw std::invalid_argument("Training_Context::setCostFunction: Null cost function pointer");
    }
    cost_function = std::move(_cost_function);
}
