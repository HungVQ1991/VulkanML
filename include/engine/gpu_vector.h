#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include "data_type.h"
#include "vulkan_sub_allocator.h"

class Vulkan_Context;

namespace gpu
{
    class vector
    {
    private:
        static inline std::atomic<uint64_t> global_vector_counter{0};

        const Vulkan_Context &context;
        VkBuffer buffer = VK_NULL_HANDLE;
        Memory_Allocation allocation{};
        size_t buffer_size_in_bytes = 0;
        size_t element_count = 0;
        Data_Type data_type = Data_Type::FLOAT32;
        uint32_t used_frame_index = 0;
        uint64_t vector_id = 0;
        void *host_mapped_pointer = nullptr;
        bool is_host_mapped = false;

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_properties);
        void copyBuffer(VkBuffer source_buffer, VkBuffer destination_buffer, VkDeviceSize size) const;

    public:
        explicit vector(const Vulkan_Context &_context);
        vector(const Vulkan_Context &_context, size_t _element_count, Data_Type _type = Data_Type::FLOAT32);
        vector(const Vulkan_Context &_context, const std::vector<float> &_host_data);
        vector(const Vulkan_Context &_context, const std::vector<float16_t> &_host_data);
        ~vector();

        vector(const vector &) = delete;
        vector &operator=(const vector &) = delete;

        vector(vector &&other) noexcept;
        vector &operator=(vector &&other) noexcept;

        void allocateMemory(size_t _element_count, Data_Type _type = Data_Type::FLOAT32);
        void freeMemory();

        void allocateHostVisible(size_t _byte_size);
        void writeHostDirect(const void *_source_pointer, size_t _size_in_bytes);

        void *getHostMappedPointer() noexcept { return host_mapped_pointer; }
        const void *getHostMappedPointer() const noexcept { return host_mapped_pointer; }
        bool isHostMapped() const noexcept { return is_host_mapped; }

        void uploadRawData(const void *source_pointer, size_t size_in_bytes);
        void uploadData(const std::vector<float> &host_data);
        void uploadData(const std::vector<float16_t> &host_data);

        void downloadRawData(void *destination_pointer, size_t size_in_bytes) const;
        void downloadData(std::vector<float> &host_data) const;
        void downloadData(std::vector<float16_t> &host_data) const;

        const Vulkan_Context &getContext() const noexcept { return context; }
        const Memory_Allocation &getAllocation() const noexcept { return allocation; }
        size_t getElementCount() const noexcept { return element_count > 0 ? element_count : (buffer_size_in_bytes / getDataTypeSize(data_type)); }
        size_t getBufferSizeInBytes() const noexcept { return buffer_size_in_bytes; }
        size_t getSize() const noexcept { return getElementCount(); }
        size_t getSizeBytes() const noexcept { return buffer_size_in_bytes; }
        size_t getByteSize() const noexcept { return buffer_size_in_bytes; }
        VkDevice getDevice() const noexcept;
        uint64_t getVectorId() const noexcept { return vector_id; }
        uint64_t getId() const noexcept { return vector_id; }
        VkBuffer getBuffer() const noexcept { return buffer; }
        Data_Type getDataType() const noexcept { return data_type; }
        uint32_t getUsedFrameIndex() const noexcept { return used_frame_index; }
        bool isEmpty() const noexcept { return buffer_size_in_bytes == 0 || buffer == VK_NULL_HANDLE; }

        void setAllocation(const Memory_Allocation &_allocation) noexcept { allocation = _allocation; }
        void setBufferSizeInBytes(size_t _bytes) noexcept { buffer_size_in_bytes = _bytes; }
        void setElementCount(size_t _count) noexcept { element_count = _count; }
        void setVectorId(uint64_t _vector_id) noexcept { vector_id = _vector_id; }
        void setBuffer(VkBuffer _buffer) noexcept { buffer = _buffer; }
        void setDataType(Data_Type _type) noexcept { data_type = _type; }
        void markAsUsedInFrame(uint32_t frame_index) noexcept { used_frame_index = frame_index; }
        void setUsedFrameIndex(uint32_t _frame_index) noexcept { used_frame_index = _frame_index; }
    };
}