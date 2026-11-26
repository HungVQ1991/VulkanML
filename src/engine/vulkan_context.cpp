#include "engine/vulkan_context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <stdexcept>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/user_preferences.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *_callback_data,
    void *)
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
    if (!User_Preferences::getInstance().isTimelineSemaphoreEnabled() || !is_timeline_semaphore_feature_enabled)
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

    bool enable_validation = IS_DEBUG_VALIDATION_ENABLED;
    if (!enable_validation)
    {
        if (const char *env_val = std::getenv("VULKANML_VALIDATION"); env_val != nullptr && std::strcmp(env_val, "1") == 0)
        {
            enable_validation = true;
        }
    }
    if (enable_validation)
    {
        uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> available_layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
        bool has_layer = std::any_of(available_layers.begin(), available_layers.end(),
                                     [](const VkLayerProperties &lp)
                                     { return std::strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0; });

        uint32_t ext_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> available_exts(ext_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, available_exts.data());
        bool has_ext = std::any_of(available_exts.begin(), available_exts.end(),
                                   [](const VkExtensionProperties &ep)
                                   { return std::strcmp(ep.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0; });

        if (!has_layer || !has_ext)
        {
            Logger::logMessage("Vulkan_Context::initializeInstance: Requested validation layer or VK_EXT_debug_utils unavailable; continuing without validation",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            enable_validation = false;
        }
    }

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
        .pNext = enable_validation ? &debug_create_information : nullptr,
        .flags = 0,
        .pApplicationInfo = &application_information,
        .enabledLayerCount = enable_validation ? 1u : 0u,
        .ppEnabledLayerNames = enable_validation ? validation_layers : nullptr,
        .enabledExtensionCount = enable_validation ? 1u : 0u,
        .ppEnabledExtensionNames = enable_validation ? instance_extensions : nullptr};

    if (vkCreateInstance(&create_information, nullptr, &instance) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::initializeInstance: Failed to create Vulkan instance",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    if (enable_validation)
    {
        auto create_fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create_fn != nullptr)
        {
            if (create_fn(instance, &debug_create_information, nullptr, &debug_messenger) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::initializeInstance: Failed to register VkDebugUtilsMessengerEXT",
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT);
                debug_messenger = VK_NULL_HANDLE;
            }
        }
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
        Logger::logMessage(static_cast<std::string>(prop.deviceName), Log_Level::LOG_INFO, true, 1, Log_Feature::DEVICE_MANAGEMENT);
    }

    struct Candidate_Device
    {
        VkPhysicalDevice device = VK_NULL_HANDLE;
        VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
        uint32_t compute_family_index = 0;
    };

    std::vector<Candidate_Device> candidates;
    candidates.reserve(devices.size());

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

        uint32_t selected_queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < queue_family_count; ++i)
        {
            if (queue_families[i].queueCount == 0 || !(queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                continue;
            }
            if (!(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                selected_queue_family = i;
                break;
            }
            if (selected_queue_family == UINT32_MAX)
            {
                selected_queue_family = i;
            }
        }

        if (selected_queue_family != UINT32_MAX)
        {
            candidates.push_back(Candidate_Device{
                device_candidate,
                device_properties.deviceType,
                selected_queue_family});
        }
    }

    if (candidates.empty())
    {
        Logger::logMessage("Vulkan_Context::selectPhysicalDevice: Failed to find a suitable GPU that satisfies all compute and resource limits",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Failed to find a suitable GPU");
    }

    const Device_Preferences prefs = User_Preferences::getInstance().getDevicePreferences();
    bool has_preferred_type = true;
    VkPhysicalDeviceType preferred_vk_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    if (prefs.preferred_type == "DISCRETE_GPU")
    {
        preferred_vk_type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    }
    else if (prefs.preferred_type == "INTEGRATED_GPU")
    {
        preferred_vk_type = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }
    else if (prefs.preferred_type == "VIRTUAL_GPU")
    {
        preferred_vk_type = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
    }
    else if (prefs.preferred_type == "CPU")
    {
        preferred_vk_type = VK_PHYSICAL_DEVICE_TYPE_CPU;
    }
    else if (prefs.preferred_type == "OTHER")
    {
        preferred_vk_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    }
    else
    {
        has_preferred_type = false;
    }

    Candidate_Device chosen{};
    bool found = false;

    if (has_preferred_type)
    {
        std::vector<Candidate_Device> preferred_matches;
        for (const auto &c : candidates)
        {
            if (c.type == preferred_vk_type)
            {
                preferred_matches.push_back(c);
            }
        }
        if (!preferred_matches.empty())
        {
            uint32_t idx = prefs.preferred_device_index;
            if (idx >= preferred_matches.size())
            {
                Logger::logMessage(Input_Format{"Vulkan_Context::selectPhysicalDevice: preferred_device_index {} out of bounds for {} matching device(s); falling back to index 0",
                                                idx, preferred_matches.size()},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT);
                idx = 0;
            }
            chosen = preferred_matches[idx];
            found = true;
        }
    }

    if (!found)
    {
        constexpr std::array<VkPhysicalDeviceType, 5> priority_order = {
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
            VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
            VK_PHYSICAL_DEVICE_TYPE_CPU,
            VK_PHYSICAL_DEVICE_TYPE_OTHER};
        for (VkPhysicalDeviceType target_type : priority_order)
        {
            for (const auto &c : candidates)
            {
                if (c.type == target_type)
                {
                    chosen = c;
                    found = true;
                    break;
                }
            }
            if (found)
            {
                break;
            }
        }
    }

    if (!found)
    {
        chosen = candidates.front();
    }

    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(chosen.device, &prop);
    Logger::logMessage(Input_Format{"Vulkan_Context::selectPhysicalDevice: Selected {} ({})",
                                    prop.deviceName,
                                    magic_enum::enum_name<VkPhysicalDeviceType>(prop.deviceType)},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT);
    physical_device = chosen.device;
    compute_queue_family_index = chosen.compute_family_index;
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

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    uint32_t device_api_version = device_properties.apiVersion;
    max_compute_work_group_count[0] = std::max<uint32_t>(1u, device_properties.limits.maxComputeWorkGroupCount[0]);
    max_compute_work_group_count[1] = std::max<uint32_t>(1u, device_properties.limits.maxComputeWorkGroupCount[1]);
    max_compute_work_group_count[2] = std::max<uint32_t>(1u, device_properties.limits.maxComputeWorkGroupCount[2]);
    timestamp_period_ns = device_properties.limits.timestampPeriod;
    timestamp_valid_bits = 0;
    {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        if (queue_family_count > 0)
        {
            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
            if (compute_queue_family_index < queue_families.size())
            {
                timestamp_valid_bits = queue_families[compute_queue_family_index].timestampValidBits;
            }
        }
    }

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
    bool has_timeline_ext = is_extension_available(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    bool has_subgroup_control_ext = is_extension_available(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    bool has_push_descriptor_ext = is_extension_available(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    VkPhysicalDeviceSubgroupProperties base_subgroup_props{};
    base_subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_props{};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDevicePushDescriptorPropertiesKHR push_desc_props{};
    push_desc_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

    if (device_api_version >= VK_API_VERSION_1_1)
    {
        base_subgroup_props.pNext = properties2.pNext;
        properties2.pNext = &base_subgroup_props;
    }
    if (has_subgroup_control_ext || device_api_version >= VK_API_VERSION_1_3)
    {
        subgroup_props.pNext = properties2.pNext;
        properties2.pNext = &subgroup_props;
    }
    if (has_push_descriptor_ext)
    {
        push_desc_props.pNext = properties2.pNext;
        properties2.pNext = &push_desc_props;
    }
    if (properties2.pNext != nullptr)
    {
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    }
    default_subgroup_size = base_subgroup_props.subgroupSize;

    VkPhysicalDevice16BitStorageFeatures supported_storage16{};
    supported_storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceFloat16Int8FeaturesKHR supported_fp16{};
    supported_fp16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR supported_coop{};
    supported_coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceTimelineSemaphoreFeatures supported_timeline{};
    supported_timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceSubgroupSizeControlFeatures supported_subgroup{};
    supported_subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;

    VkPhysicalDeviceFeatures2 supported_features2{};
    supported_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    if (has_storage16_ext || device_api_version >= VK_API_VERSION_1_1)
    {
        supported_storage16.pNext = supported_features2.pNext;
        supported_features2.pNext = &supported_storage16;
    }
    if (has_float16_ext || device_api_version >= VK_API_VERSION_1_2)
    {
        supported_fp16.pNext = supported_features2.pNext;
        supported_features2.pNext = &supported_fp16;
    }
    if (has_coop_mat_ext)
    {
        supported_coop.pNext = supported_features2.pNext;
        supported_features2.pNext = &supported_coop;
    }
    if (has_timeline_ext || device_api_version >= VK_API_VERSION_1_2)
    {
        supported_timeline.pNext = supported_features2.pNext;
        supported_features2.pNext = &supported_timeline;
    }
    if (has_subgroup_control_ext || device_api_version >= VK_API_VERSION_1_3)
    {
        supported_subgroup.pNext = supported_features2.pNext;
        supported_features2.pNext = &supported_subgroup;
    }
    if (supported_features2.pNext != nullptr)
    {
        vkGetPhysicalDeviceFeatures2(physical_device, &supported_features2);
    }

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

    is_float16_supported = false;
    is_float16_enabled = false;
    is_cooperative_matrix_supported = false;
    is_cooperative_matrix_enabled = false;

    if ((has_float16_ext || device_api_version >= VK_API_VERSION_1_2) &&
        (has_storage16_ext || device_api_version >= VK_API_VERSION_1_1) &&
        supported_fp16.shaderFloat16 == VK_TRUE &&
        supported_storage16.storageBuffer16BitAccess == VK_TRUE)
    {
        if (has_float16_ext)
        {
            required_extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }
        if (has_storage16_ext)
        {
            required_extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        }

        float16_features.pNext = device_create_pnext;
        storage_16bit_features.pNext = &float16_features;
        device_create_pnext = &storage_16bit_features;
        is_float16_supported = true;
        is_float16_enabled = true;
        Logger::logMessage("Vulkan_Context::createLogicalDevice: Float16 compute (VK_KHR_shader_float16_int8) & 16-bit storage enabled",
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::FP16_METRICS);

        const bool will_enable_subgroup_control =
            (has_subgroup_control_ext || device_api_version >= VK_API_VERSION_1_3) &&
            supported_subgroup.subgroupSizeControl == VK_TRUE;
        const bool can_guarantee_32 =
            (default_subgroup_size == 32) ||
            (will_enable_subgroup_control &&
             ((subgroup_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0) &&
             subgroup_props.minSubgroupSize <= 32 &&
             subgroup_props.maxSubgroupSize >= 32);

        if (has_coop_mat_ext && supported_coop.cooperativeMatrix == VK_TRUE)
        {
            if (can_guarantee_32)
            {
                required_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
                cooperative_matrix_features.pNext = device_create_pnext;
                device_create_pnext = &cooperative_matrix_features;
                is_cooperative_matrix_supported = true;
            }
            else
            {
                Logger::logMessage(Input_Format{"Vulkan_Context::createLogicalDevice: Disabling Cooperative Matrix because device subgroup size ({}) cannot guarantee 32", default_subgroup_size},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT);
            }
        }
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .pNext = nullptr,
        .timelineSemaphore = VK_TRUE};

    is_timeline_semaphore_feature_enabled = false;
    if (User_Preferences::getInstance().isTimelineSemaphoreEnabled() &&
        (has_timeline_ext || device_api_version >= VK_API_VERSION_1_2) &&
        supported_timeline.timelineSemaphore == VK_TRUE)
    {
        if (device_api_version < VK_API_VERSION_1_2 && has_timeline_ext)
        {
            required_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        }
        timeline_semaphore_features.pNext = device_create_pnext;
        device_create_pnext = &timeline_semaphore_features;
        is_timeline_semaphore_feature_enabled = true;
    }

    VkPhysicalDeviceFeatures enabled_features{};

    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .pNext = nullptr,
        .subgroupSizeControl = VK_TRUE,
        .computeFullSubgroups = VK_FALSE};

    is_subgroup_size_control_supported = false;
    is_subgroup_size_control_enabled = false;
    is_compute_full_subgroups_supported = false;
    min_subgroup_size = 0;
    max_subgroup_size = 0;
    required_subgroup_size_stages = 0;

    if ((has_subgroup_control_ext || device_api_version >= VK_API_VERSION_1_3) &&
        supported_subgroup.subgroupSizeControl == VK_TRUE)
    {
        if (has_subgroup_control_ext)
        {
            required_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        subgroup_size_control_features.subgroupSizeControl = VK_TRUE;
        subgroup_size_control_features.computeFullSubgroups = supported_subgroup.computeFullSubgroups;
        subgroup_size_control_features.pNext = device_create_pnext;
        device_create_pnext = &subgroup_size_control_features;

        is_subgroup_size_control_supported = true;
        is_subgroup_size_control_enabled = true;
        is_compute_full_subgroups_supported = (supported_subgroup.computeFullSubgroups == VK_TRUE);
        min_subgroup_size = subgroup_props.minSubgroupSize;
        max_subgroup_size = subgroup_props.maxSubgroupSize;
        required_subgroup_size_stages = subgroup_props.requiredSubgroupSizeStages;
        Logger::logMessage(Input_Format{"Vulkan_Context::createLogicalDevice: Subgroup size control enabled (min={}, max={}, full={})",
                                       min_subgroup_size, max_subgroup_size, is_compute_full_subgroups_supported},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
    }

    is_push_descriptor_supported = false;
    is_push_descriptor_enabled = false;
    if (has_push_descriptor_ext)
    {
        if (push_desc_props.maxPushDescriptors >= 32)
        {
            is_push_descriptor_supported = true;
            if (User_Preferences::getInstance().isPushDescriptorEnabled())
            {
                required_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
                is_push_descriptor_enabled = true;
                Logger::logMessage("Vulkan_Context::createLogicalDevice: Push descriptors (VK_KHR_push_descriptor) enabled",
                                   Log_Level::LOG_INFO,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT);
            }
            else
            {
                Logger::logMessage("Vulkan_Context::createLogicalDevice: Push descriptors supported but disabled by user preferences",
                                   Log_Level::LOG_INFO,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT);
            }
        }
        else
        {
            Logger::logMessage(Input_Format{"Vulkan_Context::createLogicalDevice: Push descriptors disabled because maxPushDescriptors ({}) < 32",
                                           push_desc_props.maxPushDescriptors},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
        }
    }

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

    if (is_push_descriptor_enabled)
    {
        vkCmdPushDescriptorSetKHR_ptr = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR")
        );
        if (!vkCmdPushDescriptorSetKHR_ptr)
        {
            Logger::logMessage("Vulkan_Context::createLogicalDevice: Failed to load vkCmdPushDescriptorSetKHR proc address",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            is_push_descriptor_enabled = false;
        }
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

void Vulkan_Context::cleanUp() noexcept
{
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        cleanGarbage(i);

        if (device != VK_NULL_HANDLE && fences[i] != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, fences[i], nullptr);
            fences[i] = VK_NULL_HANDLE;
        }

        if (device != VK_NULL_HANDLE && staging_buffers[i] != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device, staging_buffers[i], nullptr);
            staging_buffers[i] = VK_NULL_HANDLE;
            if (allocator)
            {
                allocator->free(staging_allocations[i]);
            }
            staging_allocations[i] = Memory_Allocation{};
            staging_mapped_pointers[i] = nullptr;
            staging_capacities[i] = 0;
            current_offsets[i] = 0;
        }
    }

    if (device != VK_NULL_HANDLE && readback_staging_buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, readback_staging_buffer, nullptr);
        if (allocator)
        {
            allocator->free(readback_staging_allocation);
        }
        readback_staging_allocation = Memory_Allocation{};
        readback_staging_buffer = VK_NULL_HANDLE;
        readback_mapped_pointer = nullptr;
        readback_staging_capacity = 0;
    }

    if (device != VK_NULL_HANDLE && timeline_semaphore != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(device, timeline_semaphore, nullptr);
        timeline_semaphore = VK_NULL_HANDLE;
    }
    is_timeline_semaphore_supported = false;

    allocator.reset();

    if (device != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE && transfer_command_buffer != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(device, command_pool, 1, &transfer_command_buffer);
        transfer_command_buffer = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE && immediate_copy_command_buffer != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(device, command_pool, 1, &immediate_copy_command_buffer);
        immediate_copy_command_buffer = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE && immediate_copy_fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(device, immediate_copy_fence, nullptr);
        immediate_copy_fence = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, command_pool, nullptr);
        command_pool = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE && pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        pipeline_cache = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE)
    {
        if (debug_messenger != VK_NULL_HANDLE)
        {
            auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy_fn != nullptr)
            {
                destroy_fn(instance, debug_messenger, nullptr);
            }
            debug_messenger = VK_NULL_HANDLE;
        }
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

Vulkan_Context::Vulkan_Context()
{
    Logger::logMessage("Vulkan_Context::Vulkan_Context: Initializing Vulkan Context",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT);
    try
    {
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
    catch (...)
    {
        cleanUp();
        throw;
    }
}

Vulkan_Context::~Vulkan_Context()
{
    Logger::logMessage("Vulkan_Context::~Vulkan_Context: Destroying Vulkan Context",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT);
    cleanUp();
}

Memory_Allocation Vulkan_Context::allocateMemory(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties) const
{
    return allocator->allocate(_memory_requirements, _memory_properties, physical_device);
}

void Vulkan_Context::deferDestruction(uint32_t _used_frame, VkBuffer _buffer, const Memory_Allocation &_allocation) const
{
    const uint32_t safe_frame = _used_frame % MAX_FRAMES_IN_FLIGHT;
    deferDestructionMask(1u << safe_frame, _buffer, _allocation);
}

void Vulkan_Context::deferDestructionMask(uint32_t _used_frame_mask, VkBuffer _buffer, const Memory_Allocation &_allocation) const
{
    const uint32_t full_mask = (1u << MAX_FRAMES_IN_FLIGHT) - 1u;
    uint32_t mask = _used_frame_mask & full_mask;
    if (_buffer != VK_NULL_HANDLE)
    {
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        const bool has_pending_source_transfer = std::any_of(
            pending_transfer_tasks.begin(),
            pending_transfer_tasks.end(),
            [_buffer](const Buffer_Transfer_Task &task)
            {
                return task.source_buffer == _buffer;
            });
        if (has_pending_source_transfer)
        {
            const uint32_t active_frame = current_frame.load(std::memory_order_acquire) % MAX_FRAMES_IN_FLIGHT;
            mask |= (1u << active_frame);
        }
        else
        {
            removeTransferTasksForBuffer(_buffer);
        }
    }

    if (mask == 0)
    {
        mask = 1u << (current_frame.load(std::memory_order_acquire) % MAX_FRAMES_IN_FLIGHT);
    }

    if (_buffer != VK_NULL_HANDLE || _allocation.memory != VK_NULL_HANDLE)
    {
        std::lock_guard<std::mutex> lock(garbage_mutex);
        deferred_garbage.push_back(Resource_Garbage{
            .buffer = _buffer,
            .allocation = _allocation,
            .pending_frame_mask = mask});
    }
    else
    {
        Logger::logMessage("Vulkan_Context::deferDestructionMask: Both buffer and memory handle are null",
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

void* Vulkan_Context::allocateStagingSpace(uint32_t _frame_index, VkDeviceSize _size, VkBuffer& _out_buffer, VkDeviceSize& _out_offset) const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);

    if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
    {
        Logger::logMessage(Input_Format{ "Vulkan_Context::allocateStagingSpace: frame_index out of bounds ({})", _frame_index },
            Log_Level::LOG_WARNING,
            true,
            0,
            Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
        _frame_index = _frame_index % MAX_FRAMES_IN_FLIGHT;
    }

    if (!is_frame_ready[_frame_index].load(std::memory_order_acquire))
    {
        prepareFrame(_frame_index);
    }

    constexpr VkDeviceSize ALIGNMENT = 256;
    VkDeviceSize aligned_size = (_size + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);

    if (staging_buffers[_frame_index] == VK_NULL_HANDLE || current_offsets[_frame_index] + aligned_size > staging_capacities[_frame_index])
    {
        bool staging_in_use_by_pending = false;
        if (staging_buffers[_frame_index] != VK_NULL_HANDLE)
        {
            staging_in_use_by_pending = std::any_of(
                pending_transfer_tasks.begin(),
                pending_transfer_tasks.end(),
                [&](const Buffer_Transfer_Task &task)
                {
                    return task.source_buffer == staging_buffers[_frame_index];
                });
        }

        if (staging_buffers[_frame_index] != VK_NULL_HANDLE && aligned_size <= staging_capacities[_frame_index] && !staging_in_use_by_pending)
        {
            if (!is_frame_ready[_frame_index].load(std::memory_order_acquire))
            {
                prepareFrame(_frame_index);
            }
            current_offsets[_frame_index] = 0;
        }
        else
        {
            VkDeviceSize initial_staging_capacity = static_cast<VkDeviceSize>(User_Preferences::getInstance().getStagingPoolSizeMb()) * 1024ULL * 1024ULL;
            if (initial_staging_capacity == 0)
            {
                initial_staging_capacity = 32ULL * 1024ULL * 1024ULL;
            }
            VkDeviceSize calculated_size = std::max(staging_capacities[_frame_index] * 2, aligned_size + 1024 * 1024);
            VkDeviceSize new_capacity = std::max(initial_staging_capacity, calculated_size);

            Logger::logMessage(Input_Format{ "Vulkan_Context::allocateStagingSpace: Reallocating staging buffer for frame {} to new capacity {} bytes", _frame_index, new_capacity },
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
                .pQueueFamilyIndices = nullptr };

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

            void* new_mapped_pointer = new_allocation.mapped_pointer;
            if (new_mapped_pointer == nullptr)
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
                staging_garbages[_frame_index].push_back(Staging_Garbage{
                    .buffer = staging_buffers[_frame_index],
                    .allocation = staging_allocations[_frame_index] });
            }

            staging_buffers[_frame_index] = new_buffer;
            staging_allocations[_frame_index] = new_allocation;
            staging_mapped_pointers[_frame_index] = new_mapped_pointer;
            staging_capacities[_frame_index] = new_capacity;
            current_offsets[_frame_index] = 0;
        }
    }

    _out_buffer = staging_buffers[_frame_index];
    _out_offset = current_offsets[_frame_index];
    void* target_pointer = static_cast<char*>(staging_mapped_pointers[_frame_index]) + _out_offset;
    current_offsets[_frame_index] += aligned_size;

    Logger::logMessage(Input_Format{ "Vulkan_Context::allocateStagingSpace: Allocated {} bytes in staging buffer for frame {}", _size, _frame_index },
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);

    return target_pointer;
}

void Vulkan_Context::uploadToDevice(uint32_t _frame_index, const void* _source_data, VkDeviceSize _size, VkBuffer _destination_buffer, VkDeviceSize _destination_offset) const
{
    if (!_source_data || _size == 0 || _destination_buffer == VK_NULL_HANDLE)
    {
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceSize staging_offset = 0;
    void* mapped_pointer = allocateStagingSpace(_frame_index, _size, staging_buffer, staging_offset);
    std::memcpy(mapped_pointer, _source_data, static_cast<size_t>(_size));

    Buffer_Transfer_Task transfer_task{
        .source_buffer = staging_buffer,
        .source_offset = staging_offset,
        .destination_buffer = _destination_buffer,
        .destination_offset = _destination_offset,
        .size = _size};
    addTransferTask(transfer_task);
}

void Vulkan_Context::downloadFromDevice(VkBuffer _source_buffer, void* _destination_data, VkDeviceSize _copy_size, VkDeviceSize _buffer_size) const
{
    if (_source_buffer == VK_NULL_HANDLE || !_destination_data || _copy_size == 0 || _buffer_size == 0)
    {
        return;
    }

    flush();

    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    executePendingTransfers();
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    void* mapped_pointer = ensureReadbackStagingBuffer(_buffer_size, staging_buffer);
    copyBufferImmediate(_source_buffer, staging_buffer, _buffer_size);

    size_t copy_bytes = static_cast<size_t>(std::min(_copy_size, _buffer_size));
    std::memcpy(_destination_data, mapped_pointer, copy_bytes);
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
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    if (vkResetFences(device, 1, &fences[_frame_index]) != VK_SUCCESS)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::resetFrameFence: Failed to reset fence for frame {}", _frame_index},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
        throw std::runtime_error("Vulkan_Context::resetFrameFence: Failed to reset fence");
    }
}

void Vulkan_Context::flush(VkFence _fence) const
{
    std::function<void(VkFence)> callback_copy;
    {
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        callback_copy = flush_callback;
    }
    if (callback_copy)
    {
        Logger::logMessage("Vulkan_Context::flush: Executing flush callback",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
        callback_copy(_fence);
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
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    current_offsets[_frame_index] = 0;
}

void Vulkan_Context::addTransferTask(const Buffer_Transfer_Task &_task) const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    pending_transfer_tasks.push_back(_task);
}

void Vulkan_Context::removeTransferTasksForBuffer(VkBuffer _buffer) const
{
    if (_buffer == VK_NULL_HANDLE)
    {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    std::erase_if(pending_transfer_tasks, [_buffer](const Buffer_Transfer_Task &task) {
        return task.source_buffer == _buffer || task.destination_buffer == _buffer;
    });
}

void Vulkan_Context::clearTransferTasks() const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    pending_transfer_tasks.clear();
}

bool Vulkan_Context::hasPendingTransfers() const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    return !pending_transfer_tasks.empty();
}

std::vector<Buffer_Transfer_Task> Vulkan_Context::takeTransferTasks() const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    std::vector<Buffer_Transfer_Task> tasks = std::move(pending_transfer_tasks);
    pending_transfer_tasks.clear();
    return tasks;
}

void Vulkan_Context::executePendingTransfers() const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    if (pending_transfer_tasks.empty())
    {
        return;
    }
    std::vector<Buffer_Transfer_Task> transfers_to_execute = std::move(pending_transfer_tasks);
    pending_transfer_tasks.clear();

    if (transfer_command_buffer == VK_NULL_HANDLE)
    {
        VkCommandBufferAllocateInfo allocate_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};

        if (vkAllocateCommandBuffers(device, &allocate_information, &transfer_command_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to allocate command buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to allocate command buffer");
        }
    }

    VkCommandBufferBeginInfo begin_information{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};

    if (vkBeginCommandBuffer(transfer_command_buffer, &begin_information) != VK_SUCCESS)
    {
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
            (!task.is_fill && task.source_buffer != VK_NULL_HANDLE && task.source_buffer == task.destination_buffer && task.source_offset == task.destination_offset))
        {
            continue;
        }

        if (task.is_fill || task.source_buffer == VK_NULL_HANDLE)
        {
            if ((task.destination_offset % 4) != 0 || (task.size != VK_WHOLE_SIZE && (task.size % 4) != 0))
            {
                vkEndCommandBuffer(transfer_command_buffer);
                throw std::invalid_argument("Vulkan_Context::executePendingTransfers: fillBuffer offset and size must be multiples of 4 bytes");
            }
            const uint32_t pattern = task.is_fill ? task.fill_pattern : static_cast<uint32_t>(task.source_offset);
            vkCmdFillBuffer(transfer_command_buffer, task.destination_buffer, task.destination_offset, task.size, pattern);
        }
        else
        {
            VkBufferCopy copy_region{
                .srcOffset = task.source_offset,
                .dstOffset = task.destination_offset,
                .size = task.size};

            vkCmdCopyBuffer(transfer_command_buffer, task.source_buffer, task.destination_buffer, 1, &copy_region);
        }

        VkBufferMemoryBarrier memory_barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = task.destination_buffer,
            .offset = task.destination_offset,
            .size = task.size};

        vkCmdPipelineBarrier(transfer_command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &memory_barrier, 0, nullptr);
    }

    if (vkEndCommandBuffer(transfer_command_buffer) != VK_SUCCESS)
    {
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
        .pCommandBuffers = &transfer_command_buffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr};

    if (vkQueueSubmit(compute_queue, 1, &submit_information, VK_NULL_HANDLE) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to submit transfer queue",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to submit transfer queue");
    }

    if (vkQueueWaitIdle(compute_queue) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::executePendingTransfers: vkQueueWaitIdle failed",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Vulkan_Context::executePendingTransfers: vkQueueWaitIdle failed");
    }
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
                if (device != VK_NULL_HANDLE)
                {
                    vkDestroyBuffer(device, staging_garbage_item.buffer, nullptr);
                }
            }
            if (staging_garbage_item.allocation.memory != VK_NULL_HANDLE && allocator)
            {
                allocator->free(staging_garbage_item.allocation);
            }
        }
        staging_garbages[_frame_index].clear();
    }

    std::vector<Resource_Garbage> local_bin;
    {
        std::lock_guard<std::mutex> lock(garbage_mutex);
        const uint32_t clear_bit = ~(1u << _frame_index);
        std::vector<Resource_Garbage> remaining;
        remaining.reserve(deferred_garbage.size());
        for (auto &item : deferred_garbage)
        {
            item.pending_frame_mask &= clear_bit;
            if (item.pending_frame_mask == 0)
            {
                local_bin.push_back(item);
            }
            else
            {
                remaining.push_back(item);
            }
        }
        deferred_garbage.swap(remaining);
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
            if (device != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(device, garbage_item.buffer, nullptr);
            }
        }
        if (garbage_item.allocation.memory != VK_NULL_HANDLE && allocator)
        {
            allocator->free(garbage_item.allocation);
        }
    }
}

void Vulkan_Context::prepareFrame(uint32_t _frame_index) const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);

    uint32_t target_frame = (_frame_index == UINT32_MAX) ? current_frame.load(std::memory_order_acquire) : _frame_index;
    if (target_frame >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    if (is_frame_ready[target_frame].load(std::memory_order_acquire))
    {
        return;
    }

    if (device == VK_NULL_HANDLE || fences[target_frame] == VK_NULL_HANDLE)
    {
        Logger::logMessage(Input_Format{ "Vulkan_Context::prepareFrame: Invalid device or fence handle for frame {}", target_frame },
            Log_Level::LOG_ERROR,
            true,
            0,
            Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
        throw std::runtime_error("invalid handle");
    }

    VkResult fence_status = vkGetFenceStatus(device, fences[target_frame]);
    if (fence_status == VK_NOT_READY)
    {
        VkResult wait_res = vkWaitForFences(device, 1, &fences[target_frame], VK_TRUE, UINT64_MAX);
        if (wait_res != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Vulkan_Context::prepareFrame: vkWaitForFences failed with code {}", static_cast<int>(wait_res)},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
            throw std::runtime_error("Vulkan_Context::prepareFrame: vkWaitForFences failed");
        }
    }
    else if (fence_status != VK_SUCCESS)
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::prepareFrame: vkGetFenceStatus failed with code {}", static_cast<int>(fence_status)},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
        throw std::runtime_error("Vulkan_Context::prepareFrame: vkGetFenceStatus failed");
    }

    cleanGarbage(target_frame);
    resetStagingOffset(target_frame);
    is_frame_ready[target_frame].store(true, std::memory_order_release);
}

void Vulkan_Context::advanceFrame() const
{
    std::lock_guard<std::recursive_mutex> lock(context_mutex);
    uint32_t cur = current_frame.load(std::memory_order_acquire);
    is_frame_ready[cur].store(false, std::memory_order_release);
    uint32_t next = (cur + 1) % MAX_FRAMES_IN_FLIGHT;
    current_frame.store(next, std::memory_order_release);
    Logger::logMessage(Input_Format{"Vulkan_Context::advanceFrame: Advanced current frame to {}", next},
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

void Vulkan_Context::copyBufferImmediate(VkBuffer _source_buffer,
                                        VkBuffer _destination_buffer,
                                        VkDeviceSize _size,
                                        VkDeviceSize _source_offset,
                                        VkDeviceSize _destination_offset) const
{
    if (_source_buffer == VK_NULL_HANDLE || _destination_buffer == VK_NULL_HANDLE || _size == 0)
    {
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(context_mutex);

    if (immediate_copy_command_buffer == VK_NULL_HANDLE)
    {
        VkCommandBufferAllocateInfo allocate_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};

        if (vkAllocateCommandBuffers(device, &allocate_information, &immediate_copy_command_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to allocate command buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to allocate command buffer");
        }
    }

    VkCommandBufferBeginInfo begin_information{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};

    if (vkBeginCommandBuffer(immediate_copy_command_buffer, &begin_information) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to begin command buffer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to begin command buffer");
    }

    VkMemoryBarrier pre_copy_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};

    vkCmdPipelineBarrier(
        immediate_copy_command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &pre_copy_barrier,
        0,
        nullptr,
        0,
        nullptr);

    VkBufferCopy copy_region{
        .srcOffset = _source_offset,
        .dstOffset = _destination_offset,
        .size = _size};
    vkCmdCopyBuffer(immediate_copy_command_buffer, _source_buffer, _destination_buffer, 1, &copy_region);

    if (vkEndCommandBuffer(immediate_copy_command_buffer) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to end command buffer",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_TRANSFER);
        throw std::runtime_error("Failed to end command buffer");
    }

    if (isTimelineSemaphoreSupported())
    {
        uint64_t signal_val = getNextTimelineValue();
        VkSemaphore sem = getTimelineSemaphore();

        VkTimelineSemaphoreSubmitInfo timeline_submit_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 0,
            .pWaitSemaphoreValues = nullptr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_val};

        VkSubmitInfo submit_information{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_submit_info,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &immediate_copy_command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &sem};

        if (vkQueueSubmit(compute_queue, 1, &submit_information, VK_NULL_HANDLE) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to submit copy command",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to submit copy command");
        }

        waitTimelineSemaphore(signal_val);
    }
    else
    {
        if (immediate_copy_fence == VK_NULL_HANDLE)
        {
            VkFenceCreateInfo fence_create_information{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0};

            if (vkCreateFence(device, &fence_create_information, nullptr, &immediate_copy_fence) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to create fence",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::SYNCHRONIZATION);
                throw std::runtime_error("Failed to create fence");
            }
        }
        else
        {
            if (vkResetFences(device, 1, &immediate_copy_fence) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to reset immediate copy fence",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::SYNCHRONIZATION);
                throw std::runtime_error("Failed to reset immediate copy fence");
            }
        }

        VkSubmitInfo submit_information{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &immediate_copy_command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr};

        if (vkQueueSubmit(compute_queue, 1, &submit_information, immediate_copy_fence) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to submit copy command to queue",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to submit copy command");
        }

        if (vkWaitForFences(device, 1, &immediate_copy_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::copyBufferImmediate: Failed to wait for immediate copy fence",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SYNCHRONIZATION);
            throw std::runtime_error("Failed to wait for immediate copy fence");
        }
    }
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

    if ((_destination_offset % 4) != 0 || (_size != VK_WHOLE_SIZE && (_size % 4) != 0))
    {
        Logger::logMessage(Input_Format{"Vulkan_Context::fillBuffer: Unaligned offset ({}) or size ({}) for vkCmdFillBuffer", _destination_offset, _size},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_TRANSFER);
        throw std::invalid_argument("Vulkan_Context::fillBuffer: offset and size must be multiples of 4 bytes");
    }

    Buffer_Transfer_Task transfer_task{
        .source_buffer = VK_NULL_HANDLE,
        .source_offset = 0,
        .destination_buffer = _destination_buffer,
        .destination_offset = _destination_offset,
        .size = _size,
        .is_fill = true,
        .fill_pattern = _pattern};

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
            if (vkQueueWaitIdle(compute_queue) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::waitTimelineSemaphore: vkQueueWaitIdle failed",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::SYNCHRONIZATION);
                throw std::runtime_error("Vulkan_Context::waitTimelineSemaphore: vkQueueWaitIdle failed");
            }
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
        throw std::runtime_error("Vulkan_Context::waitTimelineSemaphore: vkWaitSemaphores failed");
    }
}

void* Vulkan_Context::ensureReadbackStagingBuffer(VkDeviceSize required_size, VkBuffer& out_buffer) const
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
            .pQueueFamilyIndices = nullptr };

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

        readback_mapped_pointer = readback_staging_allocation.mapped_pointer;
        if (readback_mapped_pointer == nullptr)
        {
            vkDestroyBuffer(device, readback_staging_buffer, nullptr);
            allocator->free(readback_staging_allocation);
            readback_staging_buffer = VK_NULL_HANDLE;
            throw std::runtime_error("Vulkan_Context::ensureReadbackStagingBuffer: Failed to map memory");
        }

        readback_staging_capacity = new_capacity;
        Logger::logMessage(Input_Format{ "Vulkan_Context::ensureReadbackStagingBuffer: Allocated & persistently mapped readback staging buffer of {} MB", new_capacity / (1024 * 1024) },
            Log_Level::LOG_INFO,
            true,
            0,
            Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
    }

    out_buffer = readback_staging_buffer;
    return readback_mapped_pointer;
}
