#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef VULKANML_EXPORTS
#define ENGINE_API __declspec(dllexport)
#else
#define ENGINE_API __declspec(dllimport)
#endif
#else
#define ENGINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum VulkanML_Status
    {
        VULKANML_OK = 0,
        VULKANML_ERR_INVALID_ARGUMENT = 1,
        VULKANML_ERR_OUT_OF_MEMORY = 2,
        VULKANML_ERR_RUNTIME = 3,
        VULKANML_ERR_IO = 4,
        VULKANML_ERR_UNKNOWN = 5
    } VulkanML_Status;

    typedef enum VulkanML_Execution_Target
    {
        VULKANML_TARGET_CPU = 0,
        VULKANML_TARGET_VULKAN_GPU = 1
    } VulkanML_Execution_Target;

    typedef enum VulkanML_Activation
    {
        VULKANML_ACT_RELU = 0,
        VULKANML_ACT_LEAKY_RELU = 1,
        VULKANML_ACT_GELU = 2,
        VULKANML_ACT_SIGMOID = 3
    } VulkanML_Activation;

    ENGINE_API const char *vulkanmlGetLastError(void);

    ENGINE_API void vulkanmlClearLastError(void);

    ENGINE_API void *createPopulation(size_t pop_size,
                                      size_t state_dim,
                                      size_t action_dim,
                                      const size_t *hidden_sizes,
                                      size_t num_hidden_layers,
                                      uint32_t hidden_activation,
                                      uint32_t execution_target);

    ENGINE_API void destroyPopulation(void *pop_handle);

    ENGINE_API int32_t selectPopulationActions(void *pop_handle,
                                               const float *states_flat,
                                               const size_t *active_indices,
                                               size_t active_count,
                                               size_t *actions_out);

    ENGINE_API int32_t evolvePopulation(void *pop_handle,
                                        const float *fitness_scores,
                                        float elitism_ratio,
                                        float mutation_rate,
                                        float mutation_strength);

    ENGINE_API int32_t saveBestIndividual(void *pop_handle,
                                          const char *file_path,
                                          const float *fitness_scores);

#ifdef __cplusplus
}
#endif
