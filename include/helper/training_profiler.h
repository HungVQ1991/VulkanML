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