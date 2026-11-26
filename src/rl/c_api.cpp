#include "rl/c_api.h"

#include <cmath>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "layer/gelu.h"
#include "layer/leaky_relu_layer.h"
#include "layer/linear_layer.h"
#include "layer/relu.h"
#include "layer/sigmoid.h"
#include "helper/logger.h"
#include "neural_network.h"
#include "population.h"

namespace
{

thread_local std::string g_last_error;

void clearLastError() noexcept
{
    g_last_error.clear();
}

void setLastError(const char *msg) noexcept
{
    try
    {
        g_last_error = msg ? msg : "Unknown error";
    }
    catch (...)
    {
    }
}

void logCapiErrorNoexcept(const char *prefix, const char *detail) noexcept
{
    try
    {
        if (detail)
        {
            Logger::error(Input_Format{"{}: {}", prefix, detail}, Log_Feature::RL_AGENT);
        }
        else
        {
            Logger::error(prefix, Log_Feature::RL_AGENT);
        }
    }
    catch (...)
    {
    }
}

bool mapExecutionTarget(uint32_t raw_target, Execution_Target &out_target) noexcept
{
    switch (raw_target)
    {
    case VULKANML_TARGET_CPU:
        out_target = Execution_Target::CPU;
        return true;
    case VULKANML_TARGET_VULKAN_GPU:
        out_target = Execution_Target::VULKAN_GPU;
        return true;
    default:
        return false;
    }
}

void appendHiddenActivationLayer(Neural_Network &net, uint32_t hidden_activation, Execution_Target target)
{
    switch (hidden_activation)
    {
    case VULKANML_ACT_RELU:
        net.addLayer<Relu_Layer>(target);
        return;
    case VULKANML_ACT_LEAKY_RELU:
        net.addLayer<Leaky_Relu_Layer>(0.01f, target);
        return;
    case VULKANML_ACT_GELU:
        net.addLayer<Gelu_Layer>(target);
        return;
    case VULKANML_ACT_SIGMOID:
        net.addLayer<Sigmoid_Layer>(target);
        return;
    default:
        throw std::invalid_argument("createPopulation: unsupported hidden_activation");
    }
}

template <typename Fn>
int32_t runGuardedStatus(Fn &&fn) noexcept
{
    clearLastError();
    try
    {
        return static_cast<int32_t>(fn());
    }
    catch (const std::invalid_argument &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("RL C-API invalid argument", e.what());
        return VULKANML_ERR_INVALID_ARGUMENT;
    }
    catch (const std::out_of_range &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("RL C-API out of range", e.what());
        return VULKANML_ERR_INVALID_ARGUMENT;
    }
    catch (const std::bad_alloc &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("RL C-API out of memory", e.what());
        return VULKANML_ERR_OUT_OF_MEMORY;
    }
    catch (const std::exception &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("RL C-API exception", e.what());
        return VULKANML_ERR_RUNTIME;
    }
    catch (...)
    {
        setLastError("Unknown C++ exception");
        logCapiErrorNoexcept("RL C-API unknown C++ exception", nullptr);
        return VULKANML_ERR_UNKNOWN;
    }
}

}

extern "C"
{

const char *vulkanmlGetLastError(void)
{
    return g_last_error.c_str();
}

void vulkanmlClearLastError(void)
{
    clearLastError();
}

void *createPopulation(size_t pop_size,
                       size_t state_dim,
                       size_t action_dim,
                       const size_t *hidden_sizes,
                       size_t num_hidden_layers,
                       uint32_t hidden_activation,
                       uint32_t execution_target)
{
    clearLastError();
    try
    {
        if (pop_size == 0 || state_dim == 0 || action_dim == 0)
        {
            setLastError("createPopulation: pop_size, state_dim, and action_dim must be > 0");
            return nullptr;
        }
        if (num_hidden_layers > 0 && hidden_sizes == nullptr)
        {
            setLastError("createPopulation: hidden_sizes must not be null when num_hidden_layers > 0");
            return nullptr;
        }

        Execution_Target target = Execution_Target::CPU;
        if (!mapExecutionTarget(execution_target, target))
        {
            setLastError("createPopulation: unsupported execution_target");
            return nullptr;
        }

        Neural_Network template_net(target);
        size_t prev_dim = state_dim;
        for (size_t i = 0; i < num_hidden_layers; ++i)
        {
            if (hidden_sizes[i] == 0)
            {
                setLastError("createPopulation: every hidden layer width must be > 0");
                return nullptr;
            }
            template_net.addLayer<Linear_Layer>(prev_dim, hidden_sizes[i], target);
            appendHiddenActivationLayer(template_net, hidden_activation, target);
            prev_dim = hidden_sizes[i];
        }
        template_net.addLayer<Linear_Layer>(prev_dim, action_dim, target);

        auto pop = std::make_unique<Population>(pop_size, template_net, state_dim, action_dim, target);
        return static_cast<void *>(pop.release());
    }
    catch (const std::invalid_argument &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("createPopulation invalid argument", e.what());
        return nullptr;
    }
    catch (const std::bad_alloc &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("createPopulation out of memory", e.what());
        return nullptr;
    }
    catch (const std::exception &e)
    {
        setLastError(e.what());
        logCapiErrorNoexcept("createPopulation exception", e.what());
        return nullptr;
    }
    catch (...)
    {
        setLastError("createPopulation: unknown C++ exception");
        logCapiErrorNoexcept("createPopulation unknown C++ exception", nullptr);
        return nullptr;
    }
}

void destroyPopulation(void *pop_handle)
{
    if (!pop_handle)
    {
        return;
    }
    try
    {
        delete static_cast<Population *>(pop_handle);
    }
    catch (...)
    {
    }
}

int32_t selectPopulationActions(void *pop_handle,
                                const float *states_flat,
                                const size_t *active_indices,
                                size_t active_count,
                                size_t *actions_out)
{
    return runGuardedStatus([&]() -> VulkanML_Status
    {
        if (!pop_handle)
        {
            throw std::invalid_argument("selectPopulationActions: pop_handle must not be null");
        }
        if (active_count == 0)
        {
            return VULKANML_OK;
        }
        if (!states_flat || !actions_out)
        {
            throw std::invalid_argument("selectPopulationActions: states_flat and actions_out must not be null");
        }
        const auto *pop = static_cast<const Population *>(pop_handle);
        if (!active_indices)
        {
            if (active_count > pop->getPopulationSize())
            {
                throw std::out_of_range("selectPopulationActions: active_count exceeds population size");
            }
        }
        else
        {
            const size_t pop_sz = pop->getPopulationSize();
            for (size_t i = 0; i < active_count; ++i)
            {
                if (active_indices[i] >= pop_sz)
                {
                    throw std::out_of_range("selectPopulationActions: active_indices contains out-of-range index");
                }
            }
        }
        pop->selectBatchActions(states_flat, active_indices, active_count, actions_out);
        return VULKANML_OK;
    });
}

int32_t evolvePopulation(void *pop_handle,
                         const float *fitness_scores,
                         float elitism_ratio,
                         float mutation_rate,
                         float mutation_strength)
{
    return runGuardedStatus([&]() -> VulkanML_Status
    {
        if (!pop_handle)
        {
            throw std::invalid_argument("evolvePopulation: pop_handle must not be null");
        }
        if (!fitness_scores)
        {
            throw std::invalid_argument("evolvePopulation: fitness_scores must not be null");
        }
        if (!std::isfinite(elitism_ratio) || elitism_ratio < 0.0f || elitism_ratio > 1.0f)
        {
            throw std::invalid_argument("evolvePopulation: elitism_ratio must be finite and in [0, 1]");
        }
        if (!std::isfinite(mutation_rate) || mutation_rate < 0.0f || mutation_rate > 1.0f)
        {
            throw std::invalid_argument("evolvePopulation: mutation_rate must be finite and in [0, 1]");
        }
        if (!std::isfinite(mutation_strength) || mutation_strength < 0.0f)
        {
            throw std::invalid_argument("evolvePopulation: mutation_strength must be finite and >= 0");
        }
        auto *pop = static_cast<Population *>(pop_handle);
        pop->evolve(fitness_scores, elitism_ratio, mutation_rate, mutation_strength);
        return VULKANML_OK;
    });
}

int32_t saveBestIndividual(void *pop_handle,
                           const char *file_path,
                           const float *fitness_scores)
{
    return runGuardedStatus([&]() -> VulkanML_Status
    {
        if (!pop_handle)
        {
            throw std::invalid_argument("saveBestIndividual: pop_handle must not be null");
        }
        if (!file_path || file_path[0] == '\0')
        {
            throw std::invalid_argument("saveBestIndividual: file_path must be a non-empty string");
        }
        if (!fitness_scores)
        {
            throw std::invalid_argument("saveBestIndividual: fitness_scores must not be null");
        }
        const auto *pop = static_cast<const Population *>(pop_handle);
        if (!pop->saveBestIndividual(std::string(file_path), fitness_scores))
        {
            setLastError("saveBestIndividual: failed to write best individual checkpoint");
            return VULKANML_ERR_IO;
        }
        return VULKANML_OK;
    });
}

}
