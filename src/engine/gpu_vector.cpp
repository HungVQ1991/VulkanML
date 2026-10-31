#include "engine/gpu_vector.h"

#include <cstring>
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
            context.deferDestruction(used_frame_index, VK_NULL_HANDLE, allocation);
            Logger::logMessage("gpu::vector::createBuffer: Failed to bind buffer memory",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Failed to bind buffer memory");
        }

        Logger::logMessage(Input_Format{"gpu::vector::createBuffer: ID: {}, Buffer: {:p}, Chunk: {}, Offset: {}, Size: {} bytes, Frame: {}",
                                       vector_id, static_cast<void *>(buffer), allocation.chunk_index, allocation.offset, buffer_size_in_bytes, used_frame_index},
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
          used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
    {
    }

    vector::vector(const Vulkan_Context &_context, size_t _element_count, Data_Type _type)
        : context(_context), buffer_size_in_bytes(0), element_count(0), data_type(_type),
          used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
    {
        allocateMemory(_element_count, _type);
    }

    vector::vector(const Vulkan_Context &_context, const std::vector<float> &_host_data)
        : context(_context), buffer_size_in_bytes(0), element_count(0), data_type(Data_Type::FLOAT32),
          used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
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
          used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
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
          data_type(other.data_type), used_frame_index(other.used_frame_index), vector_id(other.vector_id),
          host_mapped_pointer(other.host_mapped_pointer), is_host_mapped(other.is_host_mapped)
    {
        other.buffer = VK_NULL_HANDLE;
        other.allocation = Memory_Allocation{};
        other.buffer_size_in_bytes = 0;
        other.element_count = 0;
        other.data_type = Data_Type::FLOAT32;
        other.used_frame_index = 0;
        other.vector_id = 0;
        other.host_mapped_pointer = nullptr;
        other.is_host_mapped = false;
    }

    vector &vector::operator=(vector &&other) noexcept
    {
        if (this != &other)
        {
            freeMemory();

            buffer = other.buffer;
            allocation = other.allocation;
            buffer_size_in_bytes = other.buffer_size_in_bytes;
            element_count = other.element_count;
            data_type = other.data_type;
            used_frame_index = other.used_frame_index;
            vector_id = other.vector_id;
            host_mapped_pointer = other.host_mapped_pointer;
            is_host_mapped = other.is_host_mapped;

            other.buffer = VK_NULL_HANDLE;
            other.allocation = Memory_Allocation{};
            other.buffer_size_in_bytes = 0;
            other.element_count = 0;
            other.data_type = Data_Type::FLOAT32;
            other.used_frame_index = 0;
            other.vector_id = 0;
            other.host_mapped_pointer = nullptr;
            other.is_host_mapped = false;
        }
        return *this;
    }

    void vector::allocateMemory(size_t _element_count, Data_Type _type)
    {
        data_type = _type;
        element_count = _element_count;
        VkDeviceSize required_byte_size = (_type == Data_Type::TERNARY_INT2)
                                              ? static_cast<VkDeviceSize>(((_element_count + 15) / 16) * sizeof(uint32_t))
                                              : static_cast<VkDeviceSize>(_element_count * getDataTypeSize(_type));
        if (required_byte_size == buffer_size_in_bytes && buffer != VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"gpu::vector::allocateMemory: Requested size matches current buffer size (ID: {}), skipping", vector_id},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        freeMemory();

        if (_element_count == 0)
        {
            Logger::logMessage("gpu::vector::allocateMemory: Requested allocation size is 0",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        if (vector_id == 0)
        {
            vector_id = ++global_vector_counter;
        }

        buffer_size_in_bytes = required_byte_size;
        used_frame_index = context.getCurrentFrame();

        Logger::logMessage(Input_Format{"gpu::vector::allocateMemory: Allocating {} elements ({} bytes) for ID: {}",
                                       _element_count, buffer_size_in_bytes, vector_id},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        createBuffer(buffer_size_in_bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    void vector::allocateMemoryByBytes(size_t _byte_size, Data_Type _type, size_t _logical_elements)
    {
        data_type = _type;
        element_count = (_logical_elements > 0) ? _logical_elements : (_byte_size / getDataTypeSize(_type));
        if (_byte_size == buffer_size_in_bytes && buffer != VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"gpu::vector::allocateMemoryByBytes: Requested size matches current buffer size (ID: {}), skipping", vector_id},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        freeMemory();

        if (_byte_size == 0)
        {
            return;
        }

        if (vector_id == 0)
        {
            vector_id = ++global_vector_counter;
        }

        buffer_size_in_bytes = _byte_size;
        used_frame_index = context.getCurrentFrame();

        Logger::logMessage(Input_Format{"gpu::vector::allocateMemoryByBytes: Allocating {} bytes (logical {} elements) for ID: {}",
                                       buffer_size_in_bytes, element_count, vector_id},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        createBuffer(buffer_size_in_bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    void vector::freeMemory()
    {
        if (buffer != VK_NULL_HANDLE || allocation.memory != VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{ "gpu::vector::freeMemory: ID: {}, Buffer: {:p}, Chunk: {}, Offset: {}, Size: {} bytes, Frame: {}",
                                           vector_id, static_cast<void*>(buffer), allocation.chunk_index, allocation.offset, buffer_size_in_bytes, used_frame_index },
                Log_Level::LOG_DEBUG,
                true,
                0,
                Log_Feature::MEMORY_ALLOCATION);

            if (buffer != VK_NULL_HANDLE)
            {
                context.removeTransferTasksForBuffer(buffer);
            }

            host_mapped_pointer = nullptr;
            is_host_mapped = false;

            context.deferDestruction(used_frame_index, buffer, allocation);

            buffer = VK_NULL_HANDLE;
            allocation = Memory_Allocation{};
            buffer_size_in_bytes = 0;
        }
    }

    void vector::allocateHostVisible(size_t _byte_size)
    {
        freeMemory();
        if (_byte_size == 0)
        {
            return;
        }

        if (vector_id == 0)
        {
            vector_id = ++global_vector_counter;
        }

        buffer_size_in_bytes = _byte_size;
        element_count = _byte_size / sizeof(float);
        data_type = Data_Type::FLOAT32;
        used_frame_index = context.getCurrentFrame();

        createBuffer(buffer_size_in_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        host_mapped_pointer = allocation.mapped_pointer;
        if (host_mapped_pointer == nullptr)
        {
            Logger::logMessage("gpu::vector::allocateHostVisible: Failed to obtain mapped host memory pointer",
                Log_Level::LOG_ERROR, true, 0, Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Failed to map host visible memory");
        }
        is_host_mapped = true;
    }

    void vector::writeHostDirect(const void *_source_pointer, size_t _size_in_bytes)
    {
        if (host_mapped_pointer && _source_pointer && _size_in_bytes <= buffer_size_in_bytes)
        {
            std::memcpy(host_mapped_pointer, _source_pointer, _size_in_bytes);
        }
    }

    void vector::uploadRawData(const void *source_pointer, size_t size_in_bytes)
    {
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

        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkDeviceSize staging_offset = 0;
        void *mapped_pointer = context.allocateStagingSpace(frame_index, buffer_size_in_bytes, staging_buffer, staging_offset);

        std::memcpy(mapped_pointer, source_pointer, buffer_size_in_bytes);

        Buffer_Transfer_Task transfer_task{
            .source_buffer = staging_buffer,
            .source_offset = staging_offset,
            .destination_buffer = buffer,
            .destination_offset = 0,
            .size = buffer_size_in_bytes};

        context.addTransferTask(transfer_task);

        markAsUsedInFrame(frame_index);
    }

    void vector::uploadData(const std::vector<float> &host_data)
    {
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

        context.flush();

        VkBuffer staging_buffer = VK_NULL_HANDLE;
        void *mapped_pointer = context.ensureReadbackStagingBuffer(buffer_size_in_bytes, staging_buffer);

        copyBuffer(buffer, staging_buffer, buffer_size_in_bytes);

        size_t copy_bytes = std::min(size_in_bytes, buffer_size_in_bytes);
        std::memcpy(destination_pointer, mapped_pointer, copy_bytes);
    }

    void vector::downloadData(std::vector<float> &host_data) const
    {
        size_t count = getElementCount();
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
            downloadRawData(host_data.data(), buffer_size_in_bytes);
        }
    }

    void vector::downloadData(std::vector<float16_t> &host_data) const
    {
        size_t count = getElementCount();
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
            downloadRawData(host_data.data(), buffer_size_in_bytes);
        }
    }

    void vector::downloadData(std::vector<uint32_t> &host_data) const
    {
        size_t count = buffer_size_in_bytes / sizeof(uint32_t);
        if (host_data.size() != count)
        {
            host_data.resize(count);
        }
        downloadRawData(host_data.data(), buffer_size_in_bytes);
    }

    VkDevice vector::getDevice() const noexcept
    {
        return context.getDevice();
    }
}
