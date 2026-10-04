#pragma once

#include <cstddef>
#include <cstdint>

class Step_Timings
{
public:
    static Step_Timings& getInstance();

    Step_Timings(const Step_Timings&) = delete;
    Step_Timings& operator=(const Step_Timings&) = delete;
    Step_Timings(Step_Timings&&) = delete;
    Step_Timings& operator=(Step_Timings&&) = delete;

    // Major Top-Level Stages (per batch or accumulated)
    double data_prep_ms = 0.0;
    double reset_grad_ms = 0.0;
    double fwd_loss_bwd_ms = 0.0;
    double opt_step_ms = 0.0;
    double sched_step_ms = 0.0;
    double total_batch_ms = 0.0;
    double total_ms = 0.0; // Alias of total_batch_ms

    // Forward Pass & Loss Sub-operations
    double fwd_record_ms = 0.0;
    double fwd_loss_exec_ms = 0.0;
    double fwd_gpu_ms = 0.0; // Alias of fwd_loss_exec_ms
    double fwd_loss_gpu_submit_ms = 0.0;
    double fwd_loss_fence_wait_ms = 0.0;
    size_t fwd_loss_gpu_nodes = 0;
    size_t fwd_gpu_nodes = 0; // Alias of fwd_loss_gpu_nodes

    double loss_readback_ms = 0.0;
    double loss_read_ms = 0.0; // Alias of loss_readback_ms
    double mul_scalar_ms = 0.0;

    // Backward Pass Sub-operations
    double bwd_record_ms = 0.0;
    double bwd_exec_ms = 0.0;
    double bwd_gpu_ms = 0.0; // Alias of bwd_exec_ms
    double bwd_gpu_submit_ms = 0.0;
    double bwd_fence_wait_ms = 0.0;
    size_t bwd_gpu_nodes = 0;

    // Optimizer Sub-operations
    double opt_record_ms = 0.0;
    double opt_exec_ms = 0.0;
    double opt_gpu_ms = 0.0; // Alias of opt_exec_ms
    double opt_gpu_submit_ms = 0.0;
    double opt_fence_wait_ms = 0.0;
    size_t opt_gpu_nodes = 0;

    // Inter-stage pipeline fence shifts (caused by double-buffering MAX_FRAMES_IN_FLIGHT = 2)
    double bwd_waited_in_opt_ms = 0.0;
    double opt_waited_in_fwd_ms = 0.0;
    bool is_chained_bwd_opt = false;

    void syncAliases();
    void reset();
    void add(const Step_Timings& other);
    void scale(double factor);
    void printTable(size_t batch_idx, size_t total_batches, double tok_s, size_t count = 1);

private:
    Step_Timings() = default;
    ~Step_Timings() = default;
};

// #pragma once

// #include <array>
// #include <cstddef>
// #include <format>
// #include <iostream>
// #include <string_view>
// #include <utility>

// #include "helper/magic_enum.hpp"

// enum class Timing_Stage : size_t
// {
//     DATA_PREP = 0,
//     RESET_GRAD,
//     FWD_RECORD,
//     FWD_LOSS_EXEC,
//     FWD_LOSS_GPU_SUBMIT,
//     FWD_LOSS_FENCE_WAIT,
//     LOSS_READBACK,
//     MUL_SCALAR,
//     BWD_RECORD,
//     BWD_EXEC,
//     BWD_GPU_SUBMIT,
//     BWD_FENCE_WAIT,
//     OPT_RECORD,
//     OPT_EXEC,
//     OPT_GPU_SUBMIT,
//     OPT_FENCE_WAIT,
//     BWD_WAITED_IN_OPT,
//     OPT_WAITED_IN_FWD,
//     SCHED_STEP,
//     TOTAL_BATCH
// };

// enum class Node_Stage : size_t
// {
//     FWD_LOSS_GPU_NODES = 0,
//     BWD_GPU_NODES,
//     OPT_GPU_NODES
// };

// class Step_Timings
// {
// public:
//     static constexpr size_t NUM_TIMING_STAGES = magic_enum::enum_count<Timing_Stage>();
//     static constexpr size_t NUM_NODE_STAGES = magic_enum::enum_count<Node_Stage>();

//     std::array<double, NUM_TIMING_STAGES> timings{};
//     std::array<size_t, NUM_NODE_STAGES> node_counts{};
//     bool is_chained_bwd_opt = false;

//     static Step_Timings& getInstance()
//     {
//         static Step_Timings instance;
//         return instance;
//     }

//     Step_Timings(const Step_Timings&) = delete;
//     Step_Timings& operator=(const Step_Timings&) = delete;
//     Step_Timings(Step_Timings&&) = delete;
//     Step_Timings& operator=(Step_Timings&&) = delete;

//     double& operator[](Timing_Stage stage) noexcept
//     {
//         return timings[std::to_underlying(stage)];
//     }

//     double operator[](Timing_Stage stage) const noexcept
//     {
//         return timings[std::to_underlying(stage)];
//     }

//     size_t& operator[](Node_Stage stage) noexcept
//     {
//         return node_counts[std::to_underlying(stage)];
//     }

//     size_t operator[](Node_Stage stage) const noexcept
//     {
//         return node_counts[std::to_underlying(stage)];
//     }

//     double& fwdGpuMs() noexcept { return (*this)[Timing_Stage::FWD_LOSS_EXEC]; }
//     double& bwdGpuMs() noexcept { return (*this)[Timing_Stage::BWD_EXEC]; }
//     double& optGpuMs() noexcept { return (*this)[Timing_Stage::OPT_EXEC]; }
//     double& totalMs() noexcept { return (*this)[Timing_Stage::TOTAL_BATCH]; }
//     double& lossReadMs() noexcept { return (*this)[Timing_Stage::LOSS_READBACK]; }

//     void reset() noexcept
//     {
//         timings.fill(0.0);
//         node_counts.fill(0);
//         is_chained_bwd_opt = false;
//     }

//     void add(const Step_Timings& other) noexcept
//     {
//         for (size_t i = 0; i < NUM_TIMING_STAGES; ++i)
//         {
//             timings[i] += other.timings[i];
//         }
//         for (size_t i = 0; i < NUM_NODE_STAGES; ++i)
//         {
//             node_counts[i] += other.node_counts[i];
//         }
//     }

//     void scale(double factor) noexcept
//     {
//         for (double& val : timings)
//         {
//             val *= factor;
//         }
//     }

//     void printTable(size_t batch_idx, size_t total_batches, double tok_s, size_t count = 1) const
//     {
//         double divisor = (count > 0) ? static_cast<double>(count) : 1.0;
//         double total_time = (*this)[Timing_Stage::TOTAL_BATCH] / divisor;

//         std::cout << std::format("\n┌{0:─^54}┐\n", "");
//         std::cout << std::format("│ Batch [{}/{}] | Throughput: {:.2f} tok/s{:>17}│\n",
//             batch_idx, total_batches, tok_s, "");
//         std::cout << std::format("├{0:─^32}┬{0:─^12}┬{0:─^8}┤\n", "");
//         std::cout << std::format("│ {:<30} │ {:>10} │ {:>6} │\n", "Stage Name", "Time (ms)", "%");
//         std::cout << std::format("├{0:─^32}┼{0:─^12}┼{0:─^8}┤\n", "");

//         for (Timing_Stage stage : magic_enum::enum_values<Timing_Stage>())
//         {
//             std::string_view name = magic_enum::enum_name(stage);
//             double time_ms = (*this)[stage] / divisor;
//             double pct = (total_time > 0.0) ? (time_ms / total_time) * 100.0 : 0.0;

//             if (stage == Timing_Stage::TOTAL_BATCH)
//             {
//                 std::cout << std::format("├{0:─^32}┼{0:─^12}┼{0:─^8}┤\n", "");
//                 std::cout << std::format("│ {:<30} │ {:>10.3f} │ {:>5.1f}% │\n", name, time_ms, pct);
//             }
//             else if (time_ms > 0.0001)
//             {
//                 std::cout << std::format("│ {:<30} │ {:>10.3f} │ {:>5.1f}% │\n", name, time_ms, pct);
//             }
//         }

//         std::cout << std::format("├{0:─^32}┴{0:─^12}┴{0:─^8}┤\n", "");
//         std::cout << std::format("│ GPU Dispatch Node Counts:{:<28}│\n", "");
//         for (Node_Stage node_stage : magic_enum::enum_values<Node_Stage>())
//         {
//             std::string_view name = magic_enum::enum_name(node_stage);
//             size_t nodes = (*this)[node_stage] / count;
//             std::cout << std::format("│   - {:<26} : {:<20}│\n", name, nodes);
//         }
//         std::cout << std::format("└{0:─^54}┘\n", "");
//     }

// private:
//     Step_Timings() = default;
//     ~Step_Timings() = default;
// };