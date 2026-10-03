#include "engine/vulkan_context.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <iostream>
#include <stdexcept>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/user_preferences.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT _message_severity,
    VkDebugUtilsMessageTypeFlagsEXT _message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *_callback_data,
    void *_user_data)
{
    if (_callback_data && _callback_data->pMessage)
    {
        Logger::logMessage(Input_Format{"debugCallback: Vulkan Validation Layer: {}", _callback_data->pMessage},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
    }
    return VK_FALSE;
}

void Vulkan_Context::initializeFences()
{
    VkFenceCreateInfo fence_create_information{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (vkCreateFence(device, &fence_create_information, nullptr, &fences[i]) != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Vulkan_Context::initializeFences: Failed to create fence for frame {}", i},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
            throw std::runtime_error("Failed to create fence");
        }
    }
}

void Vulkan_Context::initializeTimelineSemaphore()
{
    if (!User_Preferences::getInstance().isTimelineSemaphoreEnabled())
    {
        is_timeline_semaphore_supported = false;
        return;
    }

    VkSemaphoreTypeCreateInfo timeline_type_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0};

    VkSemaphoreCreateInfo semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timeline_type_create_info,
        .flags = 0};

    if (vkCreateSemaphore(device, &semaphore_create_info, nullptr, &timeline_semaphore) == VK_SUCCESS)
    {
        is_timeline_semaphore_supported = true;
        Logger::logMessage("Vulkan_Context::initializeTimelineSemaphore: Timeline semaphore initialized successfully",
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
    }
    else
    {
        is_timeline_semaphore_supported = false;
        Logger::logMessage("Vulkan_Context::initializeTimelineSemaphore: Failed to create timeline semaphore",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
    }
}

void Vulkan_Context::initializeInstance()
{
    VkApplicationInfo application_information{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "Matrix_Compute",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No_Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_4};

    const char *validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
    const char *instance_extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

    VkDebugUtilsMessengerCreateInfoEXT debug_create_information{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = nullptr};

    VkInstanceCreateInfo create_information{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = IS_DEBUG_VALIDATION_ENABLED ? &debug_create_information : nullptr,
        .flags = 0,
        .pApplicationInfo = &application_information,
        .enabledLayerCount = IS_DEBUG_VALIDATION_ENABLED ? 1u : 0u,
        .ppEnabledLayerNames = IS_DEBUG_VALIDATION_ENABLED ? validation_layers : nullptr,
        .enabledExtensionCount = IS_DEBUG_VALIDATION_ENABLED ? 1u : 0u,
        .ppEnabledExtensionNames = IS_DEBUG_VALIDATION_ENABLED ? instance_extensions : nullptr};

    if (vkCreateInstance(&create_information, nullptr, &instance) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::initializeInstance: Failed to create Vulkan instance",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to create Vulkan instance");
    }
}

bool Vulkan_Context::hasRequiredDeviceLimits(VkPhysicalDevice _target_physical_device) const
{
    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(_target_physical_device, &device_properties);
    const auto &limits = device_properties.limits;

    constexpr uint32_t REQUIRED_STORAGE_BUFFERS = 32;
    constexpr uint32_t REQUIRED_PUSH_CONSTANTS = 128;

    if (limits.maxPerStageDescriptorStorageBuffers < REQUIRED_STORAGE_BUFFERS)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::hasRequiredDeviceLimits: Device lacks required maxPerStageDescriptorStorageBuffers ({} < {})",
                                       limits.maxPerStageDescriptorStorageBuffers, REQUIRED_STORAGE_BUFFERS},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        return false;
    }

    if (limits.maxDescriptorSetStorageBuffers < REQUIRED_STORAGE_BUFFERS)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::hasRequiredDeviceLimits: Device lacks required maxDescriptorSetStorageBuffers ({} < {})",
                                       limits.maxDescriptorSetStorageBuffers, REQUIRED_STORAGE_BUFFERS},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        return false;
    }

    if (limits.maxPushConstantsSize < REQUIRED_PUSH_CONSTANTS)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::hasRequiredDeviceLimits: Device lacks required maxPushConstantsSize ({} < {})",
                                       limits.maxPushConstantsSize, REQUIRED_PUSH_CONSTANTS},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        return false;
    }

    return true;
}

void Vulkan_Context::selectPhysicalDevice()
{
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

    if (device_count == 0)
    {
        Logger::logMessage("Vulkan_Context::selectPhysicalDevice: No physical devices with Vulkan support found",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    for (VkPhysicalDevice dev : devices)
    {
        VkPhysicalDeviceProperties prop;
        vkGetPhysicalDeviceProperties(dev, &prop);
        std::cout << prop.deviceName << "\n";
    }
    constexpr std::array<size_t, 5> priority_order = {2, 1, 3, 4, 0};
    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    uint32_t best_compute_family_index = 0;
    size_t best_rank = priority_order.size();

    for (const auto &device_candidate : devices)
    {
        if (!hasRequiredDeviceLimits(device_candidate))
        {
            continue;
        }

        VkPhysicalDeviceProperties device_properties;
        vkGetPhysicalDeviceProperties(device_candidate, &device_properties);

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &queue_family_count, queue_families.data());

        for (uint32_t i = 0; i < queue_family_count; ++i)
        {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                auto it = std::find(priority_order.begin(), priority_order.end(), static_cast<size_t>(device_properties.deviceType));
                if (it != priority_order.end())
                {
                    size_t current_rank = static_cast<size_t>(std::distance(priority_order.begin(), it));
                    if (current_rank < best_rank)
                    {
                        best_rank = current_rank;
                        best_device = device_candidate;
                        best_compute_family_index = i;
                    }
                }
                break;
            }
        }
    }

    if (best_device == VK_NULL_HANDLE)
    {
        Logger::logMessage("Vulkan_Context::selectPhysicalDevice: Failed to find a suitable GPU that satisfies all compute and resource limits",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to find a suitable GPU");
    }
    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(best_device, &prop);
    std::cout << static_cast<std::string>(magic_enum::enum_name<VkPhysicalDeviceType>(prop.deviceType)) << "\n";
    physical_device = best_device;
    compute_queue_family_index = best_compute_family_index;
}

void Vulkan_Context::createLogicalDevice()
{
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_information{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = compute_queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority};

    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());

    auto is_extension_available = [&available_extensions](const char *_extension_name)
    {
        return std::any_of(available_extensions.begin(), available_extensions.end(),
                           [_extension_name](const VkExtensionProperties &_props)
                           {
                               return std::strcmp(_props.extensionName, _extension_name) == 0;
                           });
    };

    std::vector<const char *> required_extensions;

    bool has_coop_mat_ext = is_extension_available(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    bool has_float16_ext = is_extension_available(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    bool has_storage16_ext = is_extension_available(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);

    void *device_create_pnext = nullptr;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = nullptr,
        .cooperativeMatrix = VK_TRUE,
        .cooperativeMatrixRobustBufferAccess = VK_FALSE};

    VkPhysicalDeviceFloat16Int8FeaturesKHR float16_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
        .pNext = nullptr,
        .shaderFloat16 = VK_TRUE,
        .shaderInt8 = VK_FALSE};

    VkPhysicalDevice16BitStorageFeatures storage_16bit_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .pNext = nullptr,
        .storageBuffer16BitAccess = VK_TRUE,
        .uniformAndStorageBuffer16BitAccess = VK_FALSE,
        .storagePushConstant16 = VK_FALSE,
        .storageInputOutput16 = VK_FALSE};

    if (has_float16_ext && has_storage16_ext)
    {
        required_extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        required_extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);

        storage_16bit_features.pNext = &float16_features;
        device_create_pnext = &storage_16bit_features;
        is_float16_supported = true;
        is_float16_enabled = true;
        Logger::logMessage("Vulkan_Context::createLogicalDevice: Float16 compute (VK_KHR_shader_float16_int8) & 16-bit storage enabled",
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::FP16_METRICS);

        if (has_coop_mat_ext)
        {
            required_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
            float16_features.pNext = &cooperative_matrix_features;
            is_cooperative_matrix_supported = true;
        }
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .pNext = nullptr,
        .timelineSemaphore = VK_TRUE};

    if (User_Preferences::getInstance().isTimelineSemaphoreEnabled())
    {
        timeline_semaphore_features.pNext = device_create_pnext;
        device_create_pnext = &timeline_semaphore_features;
    }

    VkPhysicalDeviceFeatures enabled_features{};

    VkDeviceCreateInfo device_create_information{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = device_create_pnext,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_information,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(required_extensions.size()),
        .ppEnabledExtensionNames = required_extensions.empty() ? nullptr : required_extensions.data(),
        .pEnabledFeatures = &enabled_features};

    if (vkCreateDevice(physical_device, &device_create_information, nullptr, &device) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::createLogicalDevice: Failed to create logical Vulkan device",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);
}

void Vulkan_Context::queryCooperativeMatrixSupport()
{
    if (!is_cooperative_matrix_supported)
    {
        return;
    }

    auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

    if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
    {
        is_cooperative_matrix_supported = false;
        is_cooperative_matrix_enabled = false;
        return;
    }

    uint32_t property_count = 0;
    if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &property_count, nullptr) != VK_SUCCESS || property_count == 0)
    {
        is_cooperative_matrix_supported = false;
        is_cooperative_matrix_enabled = false;
        return;
    }

    VkCooperativeMatrixPropertiesKHR default_property{};
    default_property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    default_property.pNext = nullptr;
    std::vector<VkCooperativeMatrixPropertiesKHR> properties(property_count, default_property);

    if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &property_count, properties.data()) != VK_SUCCESS)
    {
        is_cooperative_matrix_supported = false;
        is_cooperative_matrix_enabled = false;
        return;
    }

    bool found_supported_configuration = false;
    for (const auto &prop : properties)
    {
        if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16 && prop.scope == VK_SCOPE_SUBGROUP_KHR)
        {
            cooperative_matrix_properties = prop;
            found_supported_configuration = true;
            break;
        }
    }

    if (found_supported_configuration)
    {
        is_cooperative_matrix_supported = true;
        is_cooperative_matrix_enabled = true;
        Logger::logMessage("Vulkan_Context::queryCooperativeMatrixSupport: Cooperative matrix (16x16x16, Subgroup) supported and enabled",
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::FP16_METRICS);
    }
    else
    {
        is_cooperative_matrix_supported = false;
        is_cooperative_matrix_enabled = false;
        Logger::logMessage("Vulkan_Context::queryCooperativeMatrixSupport: Cooperative matrix 16x16x16 configuration not found",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
    }
}

void Vulkan_Context::createCommandPool()
{
    VkCommandPoolCreateInfo pool_create_information{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = compute_queue_family_index};

    if (vkCreateCommandPool(device, &pool_create_information, nullptr, &command_pool) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::createCommandPool: Failed to create command pool",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to create command pool");
    }
}

void Vulkan_Context::createPipelineCache()
{
    VkPipelineCacheCreateInfo cache_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .initialDataSize = 0,
        .pInitialData = nullptr};

    if (vkCreatePipelineCache(device, &cache_create_information, nullptr, &pipeline_cache) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::createPipelineCache: Failed to create pipeline cache",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to create pipeline cache");
    }
}

Vulkan_Context::Vulkan_Context()
{
    Logger::logMessage("Vulkan_Context::Vulkan_Context: Initializing Vulkan Context",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT);
    initializeInstance();
    selectPhysicalDevice();
    createLogicalDevice();
    queryCooperativeMatrixSupport();
    createPipelineCache();
    createCommandPool();
    initializeFences();
    initializeTimelineSemaphore();
    allocator = std::make_unique<Vulkan_Sub_Allocator>(device, *this, physical_device);
}

Vulkan_Context::~Vulkan_Context()
{
    Logger::logMessage("Vulkan_Context::~Vulkan_Context: Destroying Vulkan Context",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT);
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        cleanGarbage(i);

        if (fences[i] != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, fences[i], nullptr);
        }

        if (staging_buffers[i] != VK_NULL_HANDLE)
        {
            vkUnmapMemory(device, staging_allocations[i].memory);
            vkDestroyBuffer(device, staging_buffers[i], nullptr);
            allocator->free(staging_allocations[i]);
        }
    }

    if (readback_staging_buffer != VK_NULL_HANDLE)
    {
        vkUnmapMemory(device, readback_staging_allocation.memory);
        vkDestroyBuffer(device, readback_staging_buffer, nullptr);
        allocator->free(readback_staging_allocation);
        readback_staging_buffer = VK_NULL_HANDLE;
        readback_mapped_pointer = nullptr;
        readback_staging_capacity = 0;
    }

    if (timeline_semaphore != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(device, timeline_semaphore, nullptr);
        timeline_semaphore = VK_NULL_HANDLE;
    }

    allocator.reset();

    if (command_pool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, command_pool, nullptr);
    }

    if (pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    }

    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
    }

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
}

Memory_Allocation Vulkan_Context::allocateMemory(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties) const
{
    return allocator->allocate(_memory_requirements, _memory_properties, physical_device);
}

void Vulkan_Context::deferDestruction(uint32_t _used_frame, VkBuffer _buffer, const Memory_Allocation &_allocation) const
{
    if (_buffer != VK_NULL_HANDLE)
    {
        removeTransferTasksForBuffer(_buffer);
    }

    if (_buffer != VK_NULL_HANDLE || _allocation.memory != VK_NULL_HANDLE)
    {
        std::lock_guard<std::mutex> lock(garbage_mutex);
        garbage_bins[_used_frame].push_back(Resource_Garbage{
            .buffer = _buffer,
            .allocation = _allocation});
    }
    else
    {
        Logger::logMessage("Vulkan_Context::deferDestruction: Both buffer and memory handle are null",
                           Log_Level::LOG_WARNING,
                           false,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
    }
}

uint32_t Vulkan_Context::findMemoryType(uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const
{
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
    for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i)
    {
        if ((_type_filter & (1u << i)) && (physical_device_memory_properties.memoryTypes[i].propertyFlags & _memory_properties) == _memory_properties)
        {
            return i;
        }
    }
    Logger::logMessage("Vulkan_Context::findMemoryType: Failed to find suitable memory type",
                       Log_Level::LOG_ERROR,
                       true,
                       0,
                       Log_Feature::MEMORY_ALLOCATION);
    throw std::runtime_error("Failed to find suitable memory type");
}

void *Vulkan_Context::allocateStagingSpace(uint32_t _frame_index, VkDeviceSize _size, VkBuffer &_out_buffer, VkDeviceSize &_out_offset) const
{
    if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::allocateStagingSpace: frame_index out of bounds ({})", _frame_index},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
        _frame_index = _frame_index % MAX_FRAMES_IN_FLIGHT;
    }

    std::lock_guard lock(context_mutex);

    constexpr VkDeviceSize ALIGNMENT = 256;
    VkDeviceSize aligned_size = (_size + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);

    if (staging_buffers[_frame_index] == VK_NULL_HANDLE || aligned_size > staging_capacities[_frame_index])
    {
        constexpr VkDeviceSize INITIAL_STAGING_CAPACITY = 32 * 1024 * 1024;
        VkDeviceSize calculated_size = std::max(staging_capacities[_frame_index] * 2, current_offsets[_frame_index] + aligned_size + 1024 * 1024);
        VkDeviceSize new_capacity = std::max(INITIAL_STAGING_CAPACITY, calculated_size);

        Logger::logMessage(Input_Format{"Vulkan_Context::allocateStagingSpace: Reallocating staging buffer for frame {} to new capacity {} bytes", _frame_index, new_capacity},
                           Log_Level::LOG_WARNING,
                           false,
                           0,
                           Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);

        VkBufferCreateInfo buffer_create_information{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = new_capacity,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};

        VkBuffer new_buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(device, &buffer_create_information, nullptr, &new_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to create new staging buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to create staging buffer");
        }

        VkMemoryRequirements memory_requirements;
        vkGetBufferMemoryRequirements(device, new_buffer, &memory_requirements);

        Memory_Allocation new_allocation = allocateMemory(memory_requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkBindBufferMemory(device, new_buffer, new_allocation.memory, new_allocation.offset) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, new_buffer, nullptr);
            allocator->free(new_allocation);
            Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to bind staging buffer memory",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to bind staging memory");
        }

        void *new_mapped_pointer = nullptr;
        if (vkMapMemory(device, new_allocation.memory, new_allocation.offset, new_capacity, 0, &new_mapped_pointer) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, new_buffer, nullptr);
            allocator->free(new_allocation);
            Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to map staging memory",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to map staging memory");
        }

        if (staging_buffers[_frame_index] != VK_NULL_HANDLE)
        {
            if (current_offsets[_frame_index] > 0)
            {
                std::memcpy(new_mapped_pointer, staging_mapped_pointers[_frame_index], current_offsets[_frame_index]);
            }

            vkUnmapMemory(device, staging_allocations[_frame_index].memory);
            staging_garbages[_frame_index].push_back(Staging_Garbage{
                .buffer = staging_buffers[_frame_index],
                .allocation = staging_allocations[_frame_index]});

            for (auto &task : pending_transfer_tasks)
            {
                if (task.source_buffer == staging_buffers[_frame_index])
                {
                    task.source_buffer = new_buffer;
                }
            }
        }

        staging_buffers[_frame_index] = new_buffer;
        staging_allocations[_frame_index] = new_allocation;
        staging_mapped_pointers[_frame_index] = new_mapped_pointer;
        staging_capacities[_frame_index] = new_capacity;
    }

    if (current_offsets[_frame_index] + aligned_size > staging_capacities[_frame_index])
    {
        const_cast<Vulkan_Context*>(this)->executePendingTransfers();
        if (!is_frame_ready[_frame_index])
        {
            const_cast<Vulkan_Context*>(this)->prepareFrame(_frame_index);
        }
        current_offsets[_frame_index] = 0;
    }

    _out_buffer = staging_buffers[_frame_index];
    _out_offset = current_offsets[_frame_index];
    void *target_pointer = static_cast<char *>(staging_mapped_pointers[_frame_index]) + _out_offset;
    current_offsets[_frame_index] += aligned_size;

    Logger::logMessage(Input_Format{"Vulkan_Context::allocateStagingSpace: Allocated {} bytes in staging buffer for frame {}", _size, _frame_index},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);

    return target_pointer;
}

void Vulkan_Context::resetFrameFence(uint32_t _frame_index) const
{
    if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::resetFrameFence: frame_index out of bounds ({})", _frame_index},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
        return;
    }
    vkResetFences(device, 1, &fences[_frame_index]);
}

void Vulkan_Context::flush(VkFence _fence) const
{
    if (flush_callback)
    {
        Logger::logMessage("Vulkan_Context::flush: Executing flush callback",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
        flush_callback(_fence);
    }
    else
    {
        Logger::logMessage("Vulkan_Context::flush: Flush callback is not registered",
                           Log_Level::LOG_WARNING,
                           false,
                           0,
                           Log_Feature::SYNCHRONIZATION);
    }
}

void Vulkan_Context::resetStagingOffset(uint32_t _frame_index) const
{
    if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::resetStagingOffset: frame_index out of bounds ({})", _frame_index},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
        return;
    }
    std::lock_guard lock(context_mutex);
    current_offsets[_frame_index] = 0;
}

void Vulkan_Context::addTransferTask(const Buffer_Transfer_Task &_task) const
{
    std::lock_guard lock(context_mutex);
    pending_transfer_tasks.push_back(_task);
}

void Vulkan_Context::removeTransferTasksForBuffer(VkBuffer _buffer) const
{
    if (_buffer == VK_NULL_HANDLE)
    {
        return;
    }
    std::lock_guard lock(context_mutex);
    std::erase_if(pending_transfer_tasks, [_buffer](const Buffer_Transfer_Task &task) {
        return task.source_buffer == _buffer || task.destination_buffer == _buffer;
    });
}

void Vulkan_Context::clearTransferTasks() const
{
    std::lock_guard lock(context_mutex);
    pending_transfer_tasks.clear();
}

void Vulkan_Context::executePendingTransfers() const
{
    std::vector<Buffer_Transfer_Task> transfers_to_execute;
    {
        std::lock_guard lock(context_mutex);
        if (pending_transfer_tasks.empty())
        {
            return;
        }
        transfers_to_execute = std::move(pending_transfer_tasks);
        pending_transfer_tasks.clear();
    }

    VkCommandBufferAllocateInfo allocate_information{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocate_information, &command_buffer) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to allocate command buffer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to allocate command buffer");
    }

    VkCommandBufferBeginInfo begin_information{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};

    if (vkBeginCommandBuffer(command_buffer, &begin_information) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
        Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to begin command buffer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to begin command buffer");
    }

    for (const auto &task : transfers_to_execute)
    {
        if (task.destination_buffer == VK_NULL_HANDLE || task.size == 0 ||
            (task.source_buffer != VK_NULL_HANDLE && task.source_buffer == task.destination_buffer && task.source_offset == task.destination_offset))
        {
            continue;
        }

        if (task.source_buffer == VK_NULL_HANDLE)
        {
            vkCmdFillBuffer(command_buffer, task.destination_buffer, task.destination_offset, task.size, static_cast<uint32_t>(task.source_offset));
        }
        else
        {
            VkBufferCopy copy_region{
                .srcOffset = task.source_offset,
                .dstOffset = task.destination_offset,
                .size = task.size};

            vkCmdCopyBuffer(command_buffer, task.source_buffer, task.destination_buffer, 1, &copy_region);
        }

        VkBufferMemoryBarrier memory_barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = task.destination_buffer,
            .offset = task.destination_offset,
            .size = task.size};

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &memory_barrier, 0, nullptr);
    }

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
        Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to end command buffer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to end command buffer");
    }

    VkSubmitInfo submit_information{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr};

    if (vkQueueSubmit(compute_queue, 1, &submit_information, VK_NULL_HANDLE) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
        Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to submit transfer queue",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to submit transfer queue");
    }

    vkQueueWaitIdle(compute_queue);
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void Vulkan_Context::cleanGarbage(uint32_t _frame_index) const
{
    if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::cleanGarbage: frame_index out of bounds ({})", _frame_index},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        return;
    }

    {
        std::lock_guard lock(context_mutex);
        for (const auto &staging_garbage_item : staging_garbages[_frame_index])
        {
            if (staging_garbage_item.buffer != VK_NULL_HANDLE)
            {
                std::erase_if(pending_transfer_tasks, [&](const Buffer_Transfer_Task &task) {
                    return task.source_buffer == staging_garbage_item.buffer || task.destination_buffer == staging_garbage_item.buffer;
                });
                vkDestroyBuffer(device, staging_garbage_item.buffer, nullptr);
            }
            if (staging_garbage_item.allocation.memory != VK_NULL_HANDLE)
            {
                allocator->free(staging_garbage_item.allocation);
            }
        }
        staging_garbages[_frame_index].clear();
    }

    std::vector<Resource_Garbage> local_bin;
    {
        std::lock_guard<std::mutex> lock(garbage_mutex);
        local_bin.swap(garbage_bins[_frame_index]);
    }

    if (!local_bin.empty())
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::cleanGarbage: Cleaning {} garbage items for frame {}", local_bin.size(), _frame_index},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
    }

    for (const auto &garbage_item : local_bin)
    {
        if (garbage_item.buffer != VK_NULL_HANDLE)
        {
            removeTransferTasksForBuffer(garbage_item.buffer);
            vkDestroyBuffer(device, garbage_item.buffer, nullptr);
        }
        if (garbage_item.allocation.memory != VK_NULL_HANDLE)
        {
            allocator->free(garbage_item.allocation);
        }
    }
}

void Vulkan_Context::prepareFrame(uint32_t _frame_index)
{
    uint32_t target_frame = (_frame_index == UINT32_MAX) ? current_frame : _frame_index;
    if (target_frame >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    if (is_frame_ready[target_frame])
    {
        return;
    }

    if (device == VK_NULL_HANDLE || fences[target_frame] == VK_NULL_HANDLE)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::prepareFrame: Invalid device or fence handle for frame {}", target_frame},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
        throw std::runtime_error("invalid handle");
    }

    VkResult fence_status = vkGetFenceStatus(device, fences[target_frame]);
    if (fence_status == VK_NOT_READY)
    {
        vkWaitForFences(device, 1, &fences[target_frame], VK_TRUE, UINT64_MAX);
    }

    cleanGarbage(target_frame);
    is_frame_ready[target_frame] = true;
}

void Vulkan_Context::advanceFrame() const
{
    is_frame_ready[current_frame] = false;
    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    Logger::logMessage(Input_Format{"Vulkan_Context::advanceFrame: Advanced current frame to {}", current_frame},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SYNCHRONIZATION);
}

void Vulkan_Context::copyBuffer(VkBuffer _source_buffer,
                                VkBuffer _destination_buffer,
                                VkDeviceSize _size,
                                VkDeviceSize _source_offset,
                                VkDeviceSize _destination_offset) const
{
    if (_source_buffer == VK_NULL_HANDLE || _destination_buffer == VK_NULL_HANDLE || _size == 0)
    {
        Logger::logMessage("Vulkan_Context::copyBuffer: Invalid parameters provided for buffer copy",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::MEMORY_TRANSFER);
        return;
    }

    Buffer_Transfer_Task transfer_task{
        .source_buffer = _source_buffer,
        .source_offset = _source_offset,
        .destination_buffer = _destination_buffer,
        .destination_offset = _destination_offset,
        .size = _size};

    addTransferTask(transfer_task);
    Logger::logMessage(Input_Format{"Vulkan_Context::copyBuffer: Enqueued transfer task of size {} bytes", _size},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MEMORY_TRANSFER);
}

void Vulkan_Context::fillBuffer(VkBuffer _destination_buffer,
                                VkDeviceSize _size,
                                VkDeviceSize _destination_offset,
                                uint32_t _pattern) const
{
    if (_destination_buffer == VK_NULL_HANDLE || _size == 0)
    {
        Logger::logMessage("Vulkan_Context::fillBuffer: Invalid parameters provided for buffer fill",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::MEMORY_TRANSFER);
        return;
    }

    Buffer_Transfer_Task transfer_task{
        .source_buffer = VK_NULL_HANDLE,
        .source_offset = static_cast<VkDeviceSize>(_pattern),
        .destination_buffer = _destination_buffer,
        .destination_offset = _destination_offset,
        .size = _size};

    addTransferTask(transfer_task);
    Logger::logMessage(Input_Format{"Vulkan_Context::fillBuffer: Enqueued fill task of size {} bytes, pattern 0x{:08x}", _size, _pattern},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MEMORY_TRANSFER);
}

void Vulkan_Context::waitTimelineSemaphore(uint64_t target_value, uint64_t timeout_ns) const
{
    if (!isTimelineSemaphoreSupported())
    {
        if (device != VK_NULL_HANDLE && compute_queue != VK_NULL_HANDLE)
        {
            vkQueueWaitIdle(compute_queue);
        }
        return;
    }

    uint64_t wait_values[1] = { target_value };
    VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &timeline_semaphore,
        .pValues = wait_values};

    VkResult res = vkWaitSemaphores(device, &wait_info, timeout_ns);
    if (res != VK_SUCCESS)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::waitTimelineSemaphore: Wait failed with code {}", static_cast<int>(res)},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
    }
}

void *Vulkan_Context::ensureReadbackStagingBuffer(VkDeviceSize required_size, VkBuffer &out_buffer) const
{
    std::lock_guard lock(context_mutex);

    constexpr VkDeviceSize ALIGNMENT = 256;
    VkDeviceSize aligned_size = (required_size + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);

    VkDeviceSize initial_capacity = static_cast<VkDeviceSize>(User_Preferences::getInstance().getStagingPoolSizeMb()) * 1024 * 1024;
    if (initial_capacity == 0)
    {
        initial_capacity = 64 * 1024 * 1024;
    }

    if (readback_staging_buffer == VK_NULL_HANDLE || aligned_size > readback_staging_capacity)
    {
        VkDeviceSize new_capacity = std::max(initial_capacity, std::max(readback_staging_capacity * 2, aligned_size));

        if (readback_staging_buffer != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);
            vkUnmapMemory(device, readback_staging_allocation.memory);
            vkDestroyBuffer(device, readback_staging_buffer, nullptr);
            allocator->free(readback_staging_allocation);
            readback_staging_buffer = VK_NULL_HANDLE;
            readback_mapped_pointer = nullptr;
            readback_staging_capacity = 0;
        }

        VkBufferCreateInfo buffer_create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = new_capacity,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};

        if (vkCreateBuffer(device, &buffer_create_info, nullptr, &readback_staging_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::ensureReadbackStagingBuffer: Failed to create readback staging buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Vulkan_Context::ensureReadbackStagingBuffer: Failed to create readback staging buffer");
        }

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(device, readback_staging_buffer, &mem_req);

        readback_staging_allocation = allocator->allocate(mem_req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physical_device);

        if (vkBindBufferMemory(device, readback_staging_buffer, readback_staging_allocation.memory, readback_staging_allocation.offset) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, readback_staging_buffer, nullptr);
            allocator->free(readback_staging_allocation);
            readback_staging_buffer = VK_NULL_HANDLE;
            throw std::runtime_error("Vulkan_Context::ensureReadbackStagingBuffer: Failed to bind memory");
        }

        if (vkMapMemory(device, readback_staging_allocation.memory, readback_staging_allocation.offset, new_capacity, 0, &readback_mapped_pointer) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, readback_staging_buffer, nullptr);
            allocator->free(readback_staging_allocation);
            readback_staging_buffer = VK_NULL_HANDLE;
            throw std::runtime_error("Vulkan_Context::ensureReadbackStagingBuffer: Failed to map memory");
        }

        readback_staging_capacity = new_capacity;
        Logger::logMessage(Input_Format{"Vulkan_Context::ensureReadbackStagingBuffer: Allocated & persistently mapped readback staging buffer of {} MB", new_capacity / (1024 * 1024)},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
    }

    out_buffer = readback_staging_buffer;
    return readback_mapped_pointer;
}
