#include "engine/gpu_vector.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <stdexcept>

#include "helper/logger.h"
#include "helper/training_profiler.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_sub_allocator.h"

namespace gpu
{
    void vector::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_properties)
    {
        VkDevice device = context.getDevice();

        VkBufferCreateInfo buffer_create_information{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = usage_flags,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};

        if (vkCreateBuffer(device, &buffer_create_information, nullptr, &buffer) != VK_SUCCESS)
        {
            Logger::logMessage("gpu::vector::createBuffer: Failed to create buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Failed to create buffer");
        }

        VkMemoryRequirements memory_requirements;
        vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

        allocation = context.allocateMemory(memory_requirements, memory_properties);

        if (vkBindBufferMemory(device, buffer, allocation.memory, allocation.offset) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            const uint32_t mask = (used_frame_mask != 0) ? used_frame_mask : (1u << (used_frame_index % MAX_FRAMES_IN_FLIGHT));
            context.deferDestructionMask(mask, VK_NULL_HANDLE, allocation);
            allocation = Memory_Allocation{};
            Logger::logMessage("gpu::vector::createBuffer: Failed to bind buffer memory",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Failed to bind buffer memory");
        }

        global_buffer_mutation_epoch.fetch_add(1, std::memory_order_release);

        Logger::logMessage(Input_Format{"gpu::vector::createBuffer: ID: {}, Buffer: {:p}, Chunk: {}, Offset: {}, Size: {} bytes, Frame: {}",
                                       vector_id, static_cast<void *>(buffer), allocation.chunk_index, allocation.offset, static_cast<size_t>(size), used_frame_index},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
    }

    void vector::copyBuffer(VkBuffer source_buffer, VkBuffer destination_buffer, VkDeviceSize size) const
    {
        context.copyBufferImmediate(source_buffer, destination_buffer, size);
    }

    vector::vector(const Vulkan_Context &_context)
        : context(_context), buffer_size_in_bytes(0), element_count(0), data_type(Data_Type::FLOAT32),
          used_frame_index(_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT),
          used_frame_mask(1u << (_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT)),
          vector_id(++global_vector_counter),
          fp16_shadow_cache(std::make_shared<Fp16_Shadow_Cache>())
    {
    }

    vector::vector(const Vulkan_Context &_context, size_t _element_count, Data_Type _type)
        : context(_context), buffer_size_in_bytes(0), element_count(0), data_type(_type),
          used_frame_index(_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT),
          used_frame_mask(1u << (_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT)),
          vector_id(++global_vector_counter),
          fp16_shadow_cache(std::make_shared<Fp16_Shadow_Cache>())
    {
        allocateMemory(_element_count, _type);
    }

    vector::vector(const Vulkan_Context &_context, const std::vector<float> &_host_data)
        : context(_context), buffer_size_in_bytes(0), element_count(0), data_type(Data_Type::FLOAT32),
          used_frame_index(_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT),
          used_frame_mask(1u << (_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT)),
          vector_id(++global_vector_counter),
          fp16_shadow_cache(std::make_shared<Fp16_Shadow_Cache>())
    {
        if (_host_data.empty())
        {
            Logger::logMessage("gpu::vector::vector: Host data vector is empty during construction",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            return;
        }
        allocateMemory(_host_data.size(), Data_Type::FLOAT32);
        uploadData(_host_data);
    }

    vector::vector(const Vulkan_Context &_context, const std::vector<float16_t> &_host_data)
        : context(_context), buffer_size_in_bytes(0), element_count(0), data_type(Data_Type::FLOAT16),
          used_frame_index(_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT),
          used_frame_mask(1u << (_context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT)),
          vector_id(++global_vector_counter),
          fp16_shadow_cache(std::make_shared<Fp16_Shadow_Cache>())
    {
        if (_host_data.empty())
        {
            Logger::logMessage("gpu::vector::vector: Host data vector is empty during construction",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            return;
        }
        allocateMemory(_host_data.size(), Data_Type::FLOAT16);
        uploadData(_host_data);
    }

    vector::~vector()
    {
        freeMemory();
    }

    vector::vector(vector &&other) noexcept
        : context(other.context), buffer(other.buffer), allocation(other.allocation),
          buffer_size_in_bytes(other.buffer_size_in_bytes), element_count(other.element_count),
          data_type(other.data_type), used_frame_index(other.used_frame_index), used_frame_mask(other.used_frame_mask),
          vector_id(other.vector_id), host_mapped_pointer(other.host_mapped_pointer), is_host_mapped(other.is_host_mapped),
          fp16_shadow_cache(std::move(other.fp16_shadow_cache))
    {
        other.buffer = VK_NULL_HANDLE;
        other.allocation = Memory_Allocation{};
        other.buffer_size_in_bytes = 0;
        other.element_count = 0;
        other.data_type = Data_Type::FLOAT32;
        other.used_frame_index = 0;
        other.used_frame_mask = 0;
        other.vector_id = 0;
        other.host_mapped_pointer = nullptr;
        other.is_host_mapped = false;
        if (buffer != VK_NULL_HANDLE)
        {
            global_buffer_mutation_epoch.fetch_add(1, std::memory_order_release);
        }
    }

    vector &vector::operator=(vector &&other) noexcept
    {
        if (this != &other)
        {
            if (&context != &other.context)
            {
                Logger::logMessage("gpu::vector::operator=(vector&&): Cannot move-assign across distinct Vulkan_Context instances",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                std::terminate();
            }

            freeMemory();

            buffer = other.buffer;
            allocation = other.allocation;
            buffer_size_in_bytes = other.buffer_size_in_bytes;
            element_count = other.element_count;
            data_type = other.data_type;
            used_frame_index = other.used_frame_index;
            used_frame_mask = other.used_frame_mask;
            vector_id = other.vector_id;
            host_mapped_pointer = other.host_mapped_pointer;
            is_host_mapped = other.is_host_mapped;
            fp16_shadow_cache = std::move(other.fp16_shadow_cache);

            other.buffer = VK_NULL_HANDLE;
            other.allocation = Memory_Allocation{};
            other.buffer_size_in_bytes = 0;
            other.element_count = 0;
            other.data_type = Data_Type::FLOAT32;
            other.used_frame_index = 0;
            other.used_frame_mask = 0;
            other.vector_id = 0;
            other.host_mapped_pointer = nullptr;
            other.is_host_mapped = false;
            if (buffer != VK_NULL_HANDLE)
            {
                global_buffer_mutation_epoch.fetch_add(1, std::memory_order_release);
            }
        }
        return *this;
    }

    void vector::allocateMemory(size_t _element_count, Data_Type _type)
    {
        invalidateFp16ShadowCache();
        VkDeviceSize required_byte_size = (_type == Data_Type::TERNARY_INT2)
                                              ? static_cast<VkDeviceSize>(((_element_count + 15) / 16) * sizeof(uint32_t))
                                              : static_cast<VkDeviceSize>(_element_count * getDataTypeStorageWordBytes(_type));
        if (!is_host_mapped && required_byte_size == buffer_size_in_bytes && buffer != VK_NULL_HANDLE)
        {
            data_type = _type;
            element_count = _element_count;
            Logger::logMessage(Input_Format{"gpu::vector::allocateMemory: Requested size matches current buffer size (ID: {}), skipping", vector_id},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        freeMemory();
        data_type = _type;

        if (_element_count == 0)
        {
            element_count = 0;
            Logger::logMessage("gpu::vector::allocateMemory: Requested allocation size is 0",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        const uint64_t next_id = ++global_vector_counter;
        const uint32_t frame_idx = context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT;
        vector_id = next_id;
        used_frame_index = frame_idx;
        used_frame_mask = (1u << frame_idx);

        Logger::logMessage(Input_Format{"gpu::vector::allocateMemory: Allocating {} elements ({} bytes) for ID: {}",
                                       _element_count, static_cast<size_t>(required_byte_size), vector_id},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        createBuffer(required_byte_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        buffer_size_in_bytes = static_cast<size_t>(required_byte_size);
        element_count = _element_count;
    }

    void vector::allocateMemory(const Shape &_shape, Data_Type _type)
    {
        allocateMemoryByBytes(getTensorStorageBytes(_type, _shape), _type, _shape.getTotalElements());
    }

    void vector::allocateMemoryByBytes(size_t _byte_size, Data_Type _type, size_t _logical_elements)
    {
        invalidateFp16ShadowCache();
        if (_type == Data_Type::TERNARY_INT2 && _byte_size > 0 && _logical_elements == 0)
        {
            throw std::invalid_argument("gpu::vector::allocateMemoryByBytes: TERNARY_INT2 non-empty allocation requires explicit _logical_elements > 0");
        }
        const size_t target_elements = (_logical_elements > 0) ? _logical_elements : getElementCountFromStorageBytes(_type, _byte_size);

        if (!is_host_mapped && _byte_size == buffer_size_in_bytes && buffer != VK_NULL_HANDLE)
        {
            data_type = _type;
            element_count = target_elements;
            Logger::logMessage(Input_Format{"gpu::vector::allocateMemoryByBytes: Requested size matches current buffer size (ID: {}), skipping", vector_id},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        freeMemory();
        data_type = _type;

        if (_byte_size == 0)
        {
            element_count = 0;
            return;
        }

        const uint64_t next_id = ++global_vector_counter;
        const uint32_t frame_idx = context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT;
        vector_id = next_id;
        used_frame_index = frame_idx;
        used_frame_mask = (1u << frame_idx);

        Logger::logMessage(Input_Format{"gpu::vector::allocateMemoryByBytes: Allocating {} bytes (logical {} elements) for ID: {}",
                                       _byte_size, target_elements, vector_id},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        createBuffer(_byte_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        buffer_size_in_bytes = _byte_size;
        element_count = target_elements;
    }

    void vector::freeMemory()
    {
        invalidateFp16ShadowCache();
        if (buffer != VK_NULL_HANDLE || allocation.memory != VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{ "gpu::vector::freeMemory: ID: {}, Buffer: {:p}, Chunk: {}, Offset: {}, Size: {} bytes, Frame: {}",
                                           vector_id, static_cast<void*>(buffer), allocation.chunk_index, allocation.offset, buffer_size_in_bytes, used_frame_index },
                Log_Level::LOG_DEBUG,
                true,
                0,
                Log_Feature::MEMORY_ALLOCATION);

            host_mapped_pointer = nullptr;
            is_host_mapped = false;

            const uint32_t mask = (used_frame_mask != 0) ? used_frame_mask : (1u << (used_frame_index % MAX_FRAMES_IN_FLIGHT));
            context.deferDestructionMask(mask, buffer, allocation);

            buffer = VK_NULL_HANDLE;
            allocation = Memory_Allocation{};
            buffer_size_in_bytes = 0;
            element_count = 0;
            used_frame_mask = 0;
            global_buffer_mutation_epoch.fetch_add(1, std::memory_order_release);
        }
    }

    void vector::allocateHostVisible(size_t _byte_size, Data_Type _type)
    {
        freeMemory();
        data_type = _type;
        if (_byte_size == 0)
        {
            return;
        }

        vector_id = ++global_vector_counter;
        const uint32_t frame_idx = context.getCurrentFrame() % MAX_FRAMES_IN_FLIGHT;
        used_frame_index = frame_idx;
        used_frame_mask = (1u << frame_idx);

        createBuffer(_byte_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (allocation.mapped_pointer == nullptr)
        {
            buffer_size_in_bytes = _byte_size;
            freeMemory();
            Logger::logMessage("gpu::vector::allocateHostVisible: Failed to obtain mapped host memory pointer",
                Log_Level::LOG_ERROR, true, 0, Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Failed to map host visible memory");
        }

        host_mapped_pointer = allocation.mapped_pointer;
        buffer_size_in_bytes = _byte_size;
        element_count = getElementCountFromStorageBytes(_type, _byte_size);
        data_type = _type;
        is_host_mapped = true;
    }

    void vector::writeHostDirect(const void *_source_pointer, size_t _size_in_bytes)
    {
        if (_size_in_bytes == 0)
        {
            return;
        }
        if (!_source_pointer)
        {
            Logger::logMessage("gpu::vector::writeHostDirect: Null source pointer",
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::MEMORY_TRANSFER);
            throw std::invalid_argument("gpu::vector::writeHostDirect: Null source pointer");
        }
        if (!is_host_mapped || !host_mapped_pointer)
        {
            Logger::logMessage("gpu::vector::writeHostDirect: Buffer is not host-mapped",
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("gpu::vector::writeHostDirect: Buffer is not host-mapped");
        }
        if (_size_in_bytes > buffer_size_in_bytes)
        {
            Logger::logMessage(Input_Format{"gpu::vector::writeHostDirect: Write size ({}) exceeds buffer size ({})",
                                            _size_in_bytes, buffer_size_in_bytes},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::MEMORY_TRANSFER);
            throw std::out_of_range("gpu::vector::writeHostDirect: Write size exceeds buffer capacity");
        }
        std::memcpy(host_mapped_pointer, _source_pointer, _size_in_bytes);
        invalidateFp16ShadowCache();
    }

    void vector::uploadRawData(const void *source_pointer, size_t size_in_bytes)
    {
        invalidateFp16ShadowCache();
        if (!source_pointer || size_in_bytes == 0 || buffer_size_in_bytes == 0)
        {
            Logger::logMessage("gpu::vector::uploadRawData: Empty host data or zero GPU buffer size",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            return;
        }

        if (size_in_bytes != buffer_size_in_bytes)
        {
            Logger::logMessage("gpu::vector::uploadRawData: Host data size mismatch with GPU buffer size",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("size mismatch");
        }

        uint32_t frame_index = context.getCurrentFrame();
        context.uploadToDevice(frame_index, source_pointer, buffer_size_in_bytes, buffer, 0);
        markAsUsedInFrame(frame_index);
    }

    void vector::uploadData(const std::vector<float> &host_data)
    {
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("gpu::vector::uploadData: Use uploadData(const std::vector<uint32_t>&) for packed TERNARY_INT2 buffers");
        }
        if (data_type == Data_Type::FLOAT16)
        {
            std::vector<float16_t> fp16_data(host_data.size());
            convertFp32ToFp16(host_data.data(), fp16_data.data(), host_data.size());
            uploadRawData(fp16_data.data(), fp16_data.size() * sizeof(float16_t));
        }
        else
        {
            uploadRawData(host_data.data(), host_data.size() * sizeof(float));
        }
    }

    void vector::uploadData(const std::vector<float16_t> &host_data)
    {
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("gpu::vector::uploadData: Use uploadData(const std::vector<uint32_t>&) for packed TERNARY_INT2 buffers");
        }
        if (data_type == Data_Type::FLOAT32)
        {
            std::vector<float> fp32_data(host_data.size());
            convertFp16ToFp32(host_data.data(), fp32_data.data(), host_data.size());
            uploadRawData(fp32_data.data(), fp32_data.size() * sizeof(float));
        }
        else
        {
            uploadRawData(host_data.data(), host_data.size() * sizeof(float16_t));
        }
    }

    void vector::uploadData(const std::vector<uint32_t> &host_data)
    {
        uploadRawData(host_data.data(), host_data.size() * sizeof(uint32_t));
    }

    void vector::downloadRawData(void *destination_pointer, size_t size_in_bytes) const
    {
        if (buffer_size_in_bytes == 0 || buffer == VK_NULL_HANDLE || !destination_pointer || size_in_bytes == 0)
        {
            Logger::logMessage("gpu::vector::downloadRawData: Attempted download from zero-sized or invalid buffer",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            return;
        }

        size_t copy_bytes = std::min(size_in_bytes, buffer_size_in_bytes);
        context.downloadFromDevice(buffer, destination_pointer, copy_bytes, buffer_size_in_bytes);
    }

    void vector::downloadData(std::vector<float> &host_data) const
    {
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("gpu::vector::downloadData: Use downloadData(std::vector<uint32_t>&) for packed TERNARY_INT2 buffers");
        }
        size_t count = getSize();
        if (host_data.size() != count)
        {
            host_data.resize(count);
        }

        if (data_type == Data_Type::FLOAT16)
        {
            std::vector<float16_t> fp16_data(count);
            downloadRawData(fp16_data.data(), count * sizeof(float16_t));
            convertFp16ToFp32(fp16_data.data(), host_data.data(), count);
        }
        else
        {
            downloadRawData(host_data.data(), std::min(buffer_size_in_bytes, host_data.size() * sizeof(float)));
        }
    }

    void vector::downloadData(std::vector<float16_t> &host_data) const
    {
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("gpu::vector::downloadData: Use downloadData(std::vector<uint32_t>&) for packed TERNARY_INT2 buffers");
        }
        size_t count = getSize();
        if (host_data.size() != count)
        {
            host_data.resize(count);
        }

        if (data_type == Data_Type::FLOAT32)
        {
            std::vector<float> fp32_data(count);
            downloadRawData(fp32_data.data(), count * sizeof(float));
            convertFp32ToFp16(fp32_data.data(), host_data.data(), count);
        }
        else
        {
            downloadRawData(host_data.data(), std::min(buffer_size_in_bytes, host_data.size() * sizeof(float16_t)));
        }
    }

    void vector::downloadData(std::vector<uint32_t> &host_data) const
    {
        size_t count = buffer_size_in_bytes / sizeof(uint32_t);
        if (host_data.size() != count)
        {
            host_data.resize(count);
        }
        downloadRawData(host_data.data(), std::min(buffer_size_in_bytes, host_data.size() * sizeof(uint32_t)));
    }

    VkDevice vector::getDevice() const noexcept
    {
        return context.getDevice();
    }

    void vector::setDataType(Data_Type _type) noexcept
    {
        if (data_type != _type)
        {
            invalidateFp16ShadowCache();
        }
        if (buffer != VK_NULL_HANDLE && data_type != _type)
        {
            Logger::logMessage(Input_Format{"gpu::vector::setDataType: Mutating data_type of live buffer (ID: {}) without reallocation",
                                            vector_id},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
        }
        data_type = _type;
    }

    void vector::markAsUsedInFrame(uint32_t frame_index) noexcept
    {
        const uint32_t safe_frame = frame_index % MAX_FRAMES_IN_FLIGHT;
        used_frame_index = safe_frame;
        used_frame_mask |= (1u << safe_frame);
    }

    void vector::setUsedFrameIndex(uint32_t _frame_index) noexcept
    {
        const uint32_t safe_frame = _frame_index % MAX_FRAMES_IN_FLIGHT;
        used_frame_index = safe_frame;
        used_frame_mask |= (1u << safe_frame);
    }
}
