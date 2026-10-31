// #pragma once

// #include <cstddef>
// #include <cstdint>

// class Step_Timings
// {
// public:
//     static Step_Timings& getInstance();

//     Step_Timings(const Step_Timings&) = delete;
//     Step_Timings& operator=(const Step_Timings&) = delete;
//     Step_Timings(Step_Timings&&) = delete;
//     Step_Timings& operator=(Step_Timings&&) = delete;

//     // Major Top-Level Stages (per batch or accumulated)
//     double data_prep_ms = 0.0;
//     double reset_grad_ms = 0.0;
//     double fwd_loss_bwd_ms = 0.0;
//     double opt_step_ms = 0.0;
//     double sched_step_ms = 0.0;
//     double total_batch_ms = 0.0;
//     double total_ms = 0.0; // Alias of total_batch_ms

//     // Forward Pass & Loss Sub-operations
//     double fwd_record_ms = 0.0;
//     double fwd_loss_exec_ms = 0.0;
//     double fwd_gpu_ms = 0.0; // Alias of fwd_loss_exec_ms
//     double fwd_loss_gpu_submit_ms = 0.0;
//     double fwd_loss_fence_wait_ms = 0.0;
//     size_t fwd_loss_gpu_nodes = 0;
//     size_t fwd_gpu_nodes = 0; // Alias of fwd_loss_gpu_nodes

//     double loss_readback_ms = 0.0;
//     double loss_read_ms = 0.0; // Alias of loss_readback_ms
//     double mul_scalar_ms = 0.0;

//     // Backward Pass Sub-operations
//     double bwd_record_ms = 0.0;
//     double bwd_exec_ms = 0.0;
//     double bwd_gpu_ms = 0.0; // Alias of bwd_exec_ms
//     double bwd_gpu_submit_ms = 0.0;
//     double bwd_fence_wait_ms = 0.0;
//     size_t bwd_gpu_nodes = 0;

//     // Optimizer Sub-operations
//     double opt_record_ms = 0.0;
//     double opt_exec_ms = 0.0;
//     double opt_gpu_ms = 0.0; // Alias of opt_exec_ms
//     double opt_gpu_submit_ms = 0.0;
//     double opt_fence_wait_ms = 0.0;
//     size_t opt_gpu_nodes = 0;

//     // Inter-stage pipeline fence shifts (caused by double-buffering MAX_FRAMES_IN_FLIGHT = 2)
//     double bwd_waited_in_opt_ms = 0.0;
//     double opt_waited_in_fwd_ms = 0.0;
//     bool is_chained_bwd_opt = false;

//     void syncAliases();
//     void reset();
//     void add(const Step_Timings& other);
//     void scale(double factor);
//     void printTable(size_t batch_idx, size_t total_batches, double tok_s, size_t count = 1);

// private:
//     Step_Timings() = default;
//     ~Step_Timings() = default;
// };

#pragma once

#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <string_view>
#include <utility>

#include "helper/magic_enum.hpp"

enum class Timing_Stage : size_t
{
    DATA_PREP = 0,
    RESET_GRAD,
    FWD_RECORD,
    FWD_LOSS_EXEC,
    FWD_LOSS_GPU_SUBMIT,
    FWD_LOSS_FENCE_WAIT,
    LOSS_READBACK,
    MUL_SCALAR,
    BWD_RECORD,
    BWD_EXEC,
    BWD_GPU_SUBMIT,
    BWD_FENCE_WAIT,
    OPT_RECORD,
    OPT_EXEC,
    OPT_GPU_SUBMIT,
    OPT_FENCE_WAIT,
    BWD_WAITED_IN_OPT,
    OPT_WAITED_IN_FWD,
    SCHED_STEP,
    TOTAL_BATCH
};

enum class Node_Stage : size_t
{
    FWD_LOSS_GPU_NODES = 0,
    BWD_GPU_NODES,
    OPT_GPU_NODES
};

class Step_Timings
{
public:
    static constexpr size_t NUM_TIMING_STAGES = magic_enum::enum_count<Timing_Stage>();
    static constexpr size_t NUM_NODE_STAGES = magic_enum::enum_count<Node_Stage>();

    std::array<double, NUM_TIMING_STAGES> timings{};
    std::array<size_t, NUM_NODE_STAGES> node_counts{};
    bool is_chained_bwd_opt = false;

    static Step_Timings& getInstance()
    {
        static Step_Timings instance;
        return instance;
    }

    Step_Timings(const Step_Timings&) = delete;
    Step_Timings& operator=(const Step_Timings&) = delete;
    Step_Timings(Step_Timings&&) = delete;
    Step_Timings& operator=(Step_Timings&&) = delete;

    double& operator[](Timing_Stage stage) noexcept
    {
        return timings[std::to_underlying(stage)];
    }

    double operator[](Timing_Stage stage) const noexcept
    {
        return timings[std::to_underlying(stage)];
    }

    size_t& operator[](Node_Stage stage) noexcept
    {
        return node_counts[std::to_underlying(stage)];
    }

    size_t operator[](Node_Stage stage) const noexcept
    {
        return node_counts[std::to_underlying(stage)];
    }

    double& fwdGpuMs() noexcept { return (*this)[Timing_Stage::FWD_LOSS_EXEC]; }
    double& bwdGpuMs() noexcept { return (*this)[Timing_Stage::BWD_EXEC]; }
    double& optGpuMs() noexcept { return (*this)[Timing_Stage::OPT_EXEC]; }
    double& totalMs() noexcept { return (*this)[Timing_Stage::TOTAL_BATCH]; }
    double& lossReadMs() noexcept { return (*this)[Timing_Stage::LOSS_READBACK]; }

    void reset() noexcept
    {
        timings.fill(0.0);
        node_counts.fill(0);
        is_chained_bwd_opt = false;
    }

    void add(const Step_Timings& other) noexcept
    {
        for (size_t i = 0; i < NUM_TIMING_STAGES; ++i)
        {
            timings[i] += other.timings[i];
        }
        for (size_t i = 0; i < NUM_NODE_STAGES; ++i)
        {
            node_counts[i] += other.node_counts[i];
        }
    }

    void scale(double factor) noexcept
    {
        for (double& val : timings)
        {
            val *= factor;
        }
    }

    void printTable(size_t batch_idx, size_t total_batches, double throughput, size_t count = 1, std::string_view unit = "tok/s") const
    {
        double divisor = (count > 0) ? static_cast<double>(count) : 1.0;
        double total_time = (*this)[Timing_Stage::TOTAL_BATCH] / divisor;
        if (total_time <= 0.0) total_time = 1e-6;

        auto get = [&](Timing_Stage s) { return (*this)[s] / divisor; };

        double v_data_prep    = get(Timing_Stage::DATA_PREP);
        double v_reset_grad   = get(Timing_Stage::RESET_GRAD);
        double v_fwd_record   = get(Timing_Stage::FWD_RECORD);
        double v_fwd_submit   = get(Timing_Stage::FWD_LOSS_GPU_SUBMIT);
        double v_fwd_fence    = get(Timing_Stage::FWD_LOSS_FENCE_WAIT);
        double v_fwd_exec     = get(Timing_Stage::FWD_LOSS_EXEC);
        if (v_fwd_exec <= 0.0 && (v_fwd_submit > 0.0 || v_fwd_fence > 0.0))
        {
            v_fwd_exec = v_fwd_submit + v_fwd_fence;
        }
        double v_loss_read    = get(Timing_Stage::LOSS_READBACK);
        double v_mul_scalar   = get(Timing_Stage::MUL_SCALAR);

        double v_bwd_record   = get(Timing_Stage::BWD_RECORD);
        double v_bwd_submit   = get(Timing_Stage::BWD_GPU_SUBMIT);
        double v_bwd_fence    = get(Timing_Stage::BWD_FENCE_WAIT);
        double v_bwd_exec     = get(Timing_Stage::BWD_EXEC);
        if (v_bwd_exec <= 0.0 && (v_bwd_submit > 0.0 || v_bwd_fence > 0.0))
        {
            v_bwd_exec = v_bwd_submit + v_bwd_fence;
        }

        double v_opt_record   = get(Timing_Stage::OPT_RECORD);
        double v_opt_submit   = get(Timing_Stage::OPT_GPU_SUBMIT);
        double v_opt_fence    = get(Timing_Stage::OPT_FENCE_WAIT);
        double v_opt_exec     = get(Timing_Stage::OPT_EXEC);
        double v_sched_step   = get(Timing_Stage::SCHED_STEP);

        auto print_major = [&](std::string_view name, double ms) {
            double pct = (total_time > 0.0) ? (ms / total_time) * 100.0 : 0.0;
            std::cout << std::format("│ {:<35} │ {:>10.3f} │ {:>7.1f}% │\n", name, ms, pct);
        };

        auto print_sub = [&](std::string_view name, double ms) {
            double pct = (total_time > 0.0) ? (ms / total_time) * 100.0 : 0.0;
            std::cout << std::format("│ {:<35} │ {:>10.3f} │ {:>8} │\n", name, ms, std::format("({:.1f}%)", pct));
        };

        std::cout << std::format("\n┌{0:─^61}┐\n", "");
        std::string header_str = std::format("Batch [{}/{}] | Throughput: {:.2f} {}", batch_idx, total_batches, throughput, unit);
        std::cout << std::format("│ {:<59} │\n", header_str);
        std::cout << std::format("├{0:─^37}┬{0:─^12}┬{0:─^10}┤\n", "");
        std::cout << std::format("│ {:<35} │ {:>10} │ {:>8} │\n", "Stage / Sub-operation", "Time (ms)", "%");
        std::cout << std::format("├{0:─^37}┼{0:─^12}┼{0:─^10}┤\n", "");

        size_t stage_num = 1;
        double sum_major = 0.0;

        if (v_data_prep > 0.0001)
        {
            print_major(std::format("[{}] DATA_PREP", stage_num++), v_data_prep);
            sum_major += v_data_prep;
        }

        if (v_reset_grad > 0.0001)
        {
            print_major(std::format("[{}] RESET_GRAD", stage_num++), v_reset_grad);
            sum_major += v_reset_grad;
        }

        bool has_fwd_loss = (v_fwd_record > 0.0001 || v_fwd_exec > 0.0001 || v_loss_read > 0.0001);
        if (has_fwd_loss)
        {
            double fwd_stage_ms = v_fwd_record + v_fwd_exec + v_loss_read + v_mul_scalar;
            print_major(std::format("[{}] FORWARD & LOSS", stage_num++), fwd_stage_ms);
            sum_major += fwd_stage_ms;

            if (v_fwd_record > 0.0001)
            {
                print_sub("  ├─ FWD_RECORD (CPU)", v_fwd_record);
            }
            if (v_fwd_exec > 0.0001)
            {
                bool has_sub = (v_fwd_submit > 0.0001 || v_fwd_fence > 0.0001);
                std::string_view prefix = (v_loss_read > 0.0001 || v_mul_scalar > 0.0001) ? "  ├─ " : "  └─ ";
                print_sub(std::format("{}FWD_LOSS_EXEC (GPU)", prefix), v_fwd_exec);
                if (has_sub)
                {
                    std::string_view sub_p = (v_loss_read > 0.0001 || v_mul_scalar > 0.0001) ? "  │   ↳ " : "      ↳ ";
                    if (v_fwd_submit > 0.0001) print_sub(std::format("{}GPU Submit", sub_p), v_fwd_submit);
                    if (v_fwd_fence > 0.0001)  print_sub(std::format("{}Fence Wait", sub_p), v_fwd_fence);
                }
            }
            if (v_loss_read > 0.0001)
            {
                std::string_view prefix = (v_mul_scalar > 0.0001) ? "  ├─ " : "  └─ ";
                print_sub(std::format("{}LOSS_READBACK", prefix), v_loss_read);
            }
            if (v_mul_scalar > 0.0001)
            {
                print_sub("  └─ MUL_SCALAR", v_mul_scalar);
            }
        }

        if (is_chained_bwd_opt)
        {
            double chained_gpu_ms = v_bwd_submit + v_bwd_fence;
            if (chained_gpu_ms <= 0.0) chained_gpu_ms = v_bwd_exec;

            double opt_host = (v_opt_exec > chained_gpu_ms) ? (v_opt_exec - chained_gpu_ms) : v_opt_exec;
            double host_overhead = (opt_host > v_opt_record) ? (opt_host - v_opt_record) : opt_host;

            double chained_stage_ms = v_bwd_record + v_opt_record + chained_gpu_ms + host_overhead;
            print_major(std::format("[{}] CHAINED BWD + OPT", stage_num++), chained_stage_ms);
            sum_major += chained_stage_ms;

            if (v_bwd_record > 0.0001)
            {
                print_sub("  ├─ BWD_RECORD (CPU)", v_bwd_record);
            }
            if (v_opt_record > 0.0001)
            {
                print_sub("  ├─ OPT_RECORD (CPU)", v_opt_record);
            }
            if (chained_gpu_ms > 0.0001)
            {
                bool has_sub = (v_bwd_submit > 0.0001 || v_bwd_fence > 0.0001);
                std::string_view prefix = (host_overhead > 0.0001) ? "  ├─ " : "  └─ ";
                print_sub(std::format("{}BWD+OPT GPU EXEC", prefix), chained_gpu_ms);
                if (has_sub)
                {
                    std::string_view sub_p = (host_overhead > 0.0001) ? "  │   ↳ " : "      ↳ ";
                    if (v_bwd_submit > 0.0001) print_sub(std::format("{}GPU Submit", sub_p), v_bwd_submit);
                    if (v_bwd_fence > 0.0001)  print_sub(std::format("{}Fence Wait", sub_p), v_bwd_fence);
                }
            }
            if (host_overhead > 0.0001)
            {
                print_sub("  └─ Host Overhead (Sync/Clip)", host_overhead);
            }
        }
        else
        {
            bool has_bwd = (v_bwd_record > 0.0001 || v_bwd_exec > 0.0001);
            if (has_bwd)
            {
                double bwd_stage_ms = v_bwd_record + v_bwd_exec;
                print_major(std::format("[{}] BACKWARD PASS", stage_num++), bwd_stage_ms);
                sum_major += bwd_stage_ms;

                if (v_bwd_record > 0.0001)
                {
                    std::string_view prefix = (v_bwd_exec > 0.0001) ? "  ├─ " : "  └─ ";
                    print_sub(std::format("{}BWD_RECORD (CPU)", prefix), v_bwd_record);
                }
                if (v_bwd_exec > 0.0001)
                {
                    print_sub("  └─ BWD GPU EXEC", v_bwd_exec);
                    if (v_bwd_submit > 0.0001) print_sub("      ↳ GPU Submit", v_bwd_submit);
                    if (v_bwd_fence > 0.0001)  print_sub("      ↳ Fence Wait", v_bwd_fence);
                }
            }

            double opt_gpu = (v_opt_submit + v_opt_fence > 0.0) ? (v_opt_submit + v_opt_fence) : 0.0;
            double opt_host = (v_opt_exec > opt_gpu) ? (v_opt_exec - opt_gpu) : 0.0;
            if (opt_host >= v_opt_record) opt_host -= v_opt_record;

            double opt_stage_ms = v_opt_record + opt_gpu + opt_host;
            if (opt_stage_ms <= 0.0 && v_opt_exec > 0.0) opt_stage_ms = v_opt_exec;

            if (opt_stage_ms > 0.0001)
            {
                print_major(std::format("[{}] OPTIMIZER STEP", stage_num++), opt_stage_ms);
                sum_major += opt_stage_ms;

                if (v_opt_record > 0.0001)
                {
                    std::string_view prefix = (opt_gpu > 0.0001 || opt_host > 0.0001) ? "  ├─ " : "  └─ ";
                    print_sub(std::format("{}OPT_RECORD (CPU)", prefix), v_opt_record);
                }
                if (opt_gpu > 0.0001)
                {
                    std::string_view prefix = (opt_host > 0.0001) ? "  ├─ " : "  └─ ";
                    print_sub(std::format("{}OPT GPU EXEC", prefix), opt_gpu);
                    std::string_view sub_p = (opt_host > 0.0001) ? "  │   ↳ " : "      ↳ ";
                    if (v_opt_submit > 0.0001) print_sub(std::format("{}GPU Submit", sub_p), v_opt_submit);
                    if (v_opt_fence > 0.0001)  print_sub(std::format("{}Fence Wait", sub_p), v_opt_fence);
                }
                if (opt_host > 0.0001)
                {
                    print_sub("  └─ Host Overhead (Sync/Clip)", opt_host);
                }
            }
        }

        if (!has_fwd_loss && !is_chained_bwd_opt && (total_time > sum_major + v_sched_step + 0.001))
        {
            double step_exec = total_time - (sum_major + v_sched_step);
            print_major(std::format("[{}] MODEL_TRAIN_STEP", stage_num++), step_exec);
            sum_major += step_exec;
        }

        if (v_sched_step > 0.0001)
        {
            print_major(std::format("[{}] SCHED_STEP", stage_num++), v_sched_step);
            sum_major += v_sched_step;
        }

        if (total_time > sum_major + 0.001)
        {
            double overhead = total_time - sum_major;
            print_major(std::format("[{}] LOOP_OVERHEAD", stage_num++), overhead);
        }

        std::cout << std::format("├{0:─^37}┼{0:─^12}┼{0:─^10}┤\n", "");
        print_major("TOTAL_BATCH", total_time);

        std::cout << std::format("├{0:─^61}┤\n", "");
        std::cout << std::format("│ {:<59} │\n", "GPU Dispatch Node Counts:");
        for (Node_Stage node_stage : magic_enum::enum_values<Node_Stage>())
        {
            std::string_view name = magic_enum::enum_name(node_stage);
            size_t nodes = (*this)[node_stage] / count;
            std::string node_line = std::format("  - {:<28} : {}", name, nodes);
            std::cout << std::format("│ {:<59} │\n", node_line);
        }
        std::cout << std::format("└{0:─^61}┘\n", "");
    }

private:
    Step_Timings() = default;
    ~Step_Timings() = default;
};