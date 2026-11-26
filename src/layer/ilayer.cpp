#include "layer/ilayer.h"

#include <atomic>

namespace
{
std::atomic<uint64_t> g_ilayer_next_hook_id{1};
}

const Tensor &ILayer::emptyTensor()
{
    static const Tensor empty_tensor(0, 0);
    return empty_tensor;
}

uint64_t ILayer::nextHookId() noexcept
{
    return g_ilayer_next_hook_id.fetch_add(1, std::memory_order_relaxed);
}
