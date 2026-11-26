#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>

#include "data_type.h"
#include "vulkan_sub_allocator.h"

class Vulkan_Context;

namespace gpu
{
    class vector;

    struct Fp16_Shadow_Cache
    {
        mutable std::mutex mutex;
        std::shared_ptr<vector> fp16_storage;
        std::atomic<uint64_t> storage_version{1};
        uint64_t cached_version = 0;
        size_t valid_elements = 0;
    };

    class vector
    {
    private:
        static inline std::atomic<uint64_t> global_vector_counter{0};
        static inline std::atomic<uint64_t> global_buffer_mutation_epoch{1};

        const Vulkan_Context &context;
        VkBuffer buffer = VK_NULL_HANDLE;
        Memory_Allocation allocation{};
        size_t buffer_size_in_bytes = 0;
        size_t element_count = 0;
        Data_Type data_type = Data_Type::FLOAT32;
        uint32_t used_frame_index = 0;
        uint32_t used_frame_mask = 0;
        uint64_t vector_id = 0;
        void *host_mapped_pointer = nullptr;
        bool is_host_mapped = false;
        std::shared_ptr<Fp16_Shadow_Cache> fp16_shadow_cache;

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
        void allocateMemory(const Shape &_shape, Data_Type _type = Data_Type::FLOAT32);
        void allocateMemoryByBytes(size_t _byte_size, Data_Type _type = Data_Type::FLOAT32, size_t _logical_elements = 0);
        void freeMemory();

        void allocateHostVisible(size_t _byte_size, Data_Type _type = Data_Type::FLOAT32);
        void writeHostDirect(const void *_source_pointer, size_t _size_in_bytes);

        void *getHostMappedPointer() noexcept { return host_mapped_pointer; }
        const void *getHostMappedPointer() const noexcept { return host_mapped_pointer; }
        bool isHostMapped() const noexcept { return is_host_mapped; }

        void uploadRawData(const void *source_pointer, size_t size_in_bytes);
        void uploadData(const std::vector<float> &host_data);
        void uploadData(const std::vector<float16_t> &host_data);
        void uploadData(const std::vector<uint32_t> &host_data);

        void downloadRawData(void *destination_pointer, size_t size_in_bytes) const;
        void downloadData(std::vector<float> &host_data) const;
        void downloadData(std::vector<float16_t> &host_data) const;
        void downloadData(std::vector<uint32_t> &host_data) const;

        const Vulkan_Context &getContext() const noexcept { return context; }
        const Memory_Allocation &getAllocation() const noexcept { return allocation; }
        size_t getBufferSizeInBytes() const noexcept { return buffer_size_in_bytes; }
        size_t getSize() const noexcept
        {
            if (element_count > 0) return element_count;
            if (buffer_size_in_bytes == 0 || data_type == Data_Type::TERNARY_INT2) return 0;
            return getElementCountFromStorageBytes(data_type, buffer_size_in_bytes);
        }
        size_t getSizeBytes() const noexcept { return buffer_size_in_bytes; }
        size_t getByteSize() const noexcept { return buffer_size_in_bytes; }
        VkDevice getDevice() const noexcept;
        static uint64_t getGlobalBufferMutationEpoch() noexcept { return global_buffer_mutation_epoch.load(std::memory_order_acquire); }
        uint64_t getVectorId() const noexcept { return vector_id; }
        uint64_t getId() const noexcept { return vector_id; }
        VkBuffer getBuffer() const noexcept { return buffer; }
        Data_Type getDataType() const noexcept { return data_type; }
        uint32_t getUsedFrameIndex() const noexcept { return used_frame_index; }
        uint32_t getUsedFrameMask() const noexcept { return used_frame_mask; }
        bool isEmpty() const noexcept { return buffer_size_in_bytes == 0 || buffer == VK_NULL_HANDLE; }

        void setDataType(Data_Type _type) noexcept;
        void markAsUsedInFrame(uint32_t frame_index) noexcept;
        void setUsedFrameIndex(uint32_t _frame_index) noexcept;

        void invalidateFp16ShadowCache() const noexcept
        {
            if (fp16_shadow_cache)
            {
                fp16_shadow_cache->storage_version.fetch_add(1, std::memory_order_release);
            }
        }
        std::shared_ptr<Fp16_Shadow_Cache> getFp16ShadowCache() const noexcept { return fp16_shadow_cache; }
        std::shared_ptr<Fp16_Shadow_Cache> ensureFp16ShadowCache()
        {
            if (!fp16_shadow_cache)
            {
                fp16_shadow_cache = std::make_shared<Fp16_Shadow_Cache>();
            }
            return fp16_shadow_cache;
        }
    };
}
