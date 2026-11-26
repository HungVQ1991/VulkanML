#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "math/tensor.h"

namespace cost_internal
{

inline void syncScratchTarget(Tensor &scratch, Execution_Target target)
{
    if (scratch.getExecutionTarget() != target)
    {
        scratch.setExecutionTarget(target);
    }
}

inline bool validateDenseInputs(
    const Tensor &pred,
    const Tensor &target,
    const char *ctx,
    bool log_empty_warning,
    bool log_dims,
    bool detailed_shape_err = false)
{
    if (pred.getShape() != target.getShape())
    {
        if (detailed_shape_err)
        {
            Logger::logMessage(Input_Format{"{}: Dimensions mismatch! Prediction: ({}x{}), Target: ({}x{})",
                                            ctx, pred.getRows(), pred.getColumns(),
                                            target.getRows(), target.getColumns()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument(std::string(ctx) + ": Dimensions mismatch");
        }
        Logger::logMessage(std::string(ctx) + ": Dimensions mismatch",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::LOSS_COMPUTE);
        throw std::invalid_argument("Dimensions mismatch");
    }

    if (pred.getExecutionTarget() != target.getExecutionTarget())
    {
        std::string msg = std::string(ctx) + ": Execution target mismatch between prediction and target matrix";
        Logger::logMessage(msg, Log_Level::LOG_ERROR, true, 0, Log_Feature::LOSS_COMPUTE);
        throw std::invalid_argument(msg);
    }

    if (pred.getDataType() == Data_Type::TERNARY_INT2 ||
        target.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(ctx) + ": TERNARY_INT2 is not supported");
    }

    if (pred.getTotalElements() == 0)
    {
        if (log_empty_warning)
        {
            Logger::logMessage(std::string(ctx) + ": Empty input matrix encountered",
                               Log_Level::LOG_WARNING, false, 0, Log_Feature::LOSS_COMPUTE);
        }
        return false;
    }

    if (log_dims)
    {
        Logger::logMessage(Input_Format{"{}: rows={}, columns={}", ctx, pred.getRows(), pred.getColumns()},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::LOSS_COMPUTE);
    }
    return true;
}

inline void computeScaledDiffGradient(
    const Tensor &pred,
    const Tensor &target,
    float scale,
    Tensor &synced_target_matrix,
    Tensor &difference_matrix,
    Tensor &gradient_matrix)
{
    const Execution_Target target_dev = pred.getExecutionTarget();
    const Tensor *effective_target = &target;
    if (target.getDataType() != pred.getDataType())
    {
        syncScratchTarget(synced_target_matrix, target_dev);
        target.to(pred.getDataType(), synced_target_matrix);
        effective_target = &synced_target_matrix;
    }
    else if (synced_target_matrix.getTotalElements() > 0)
    {
        synced_target_matrix = Tensor(0, 0, target_dev);
    }

    syncScratchTarget(difference_matrix, target_dev);
    syncScratchTarget(gradient_matrix, target_dev);
    pred.sub(*effective_target, difference_matrix);
    difference_matrix.mulScalar(scale, gradient_matrix);
}

inline void computeScaledDiffGradientOut(
    const Tensor &pred,
    const Tensor &target,
    float scale,
    Tensor &synced_target_matrix,
    Tensor &difference_matrix,
    Tensor &gradient_out)
{
    const Execution_Target target_dev = pred.getExecutionTarget();
    const Tensor *effective_target = &target;
    Tensor local_cast_target(target_dev);
    if (target.getDataType() != pred.getDataType())
    {
        if (target_dev == Execution_Target::CPU)
        {
            target.to(pred.getDataType(), local_cast_target);
            effective_target = &local_cast_target;
        }
        else
        {
            syncScratchTarget(synced_target_matrix, target_dev);
            target.to(pred.getDataType(), synced_target_matrix);
            effective_target = &synced_target_matrix;
        }
    }

    if (target_dev == Execution_Target::CPU)
    {
        Tensor local_diff(Execution_Target::CPU);
        pred.sub(*effective_target, local_diff);
        local_diff.mulScalar(scale, gradient_out);
    }
    else
    {
        syncScratchTarget(difference_matrix, target_dev);
        pred.sub(*effective_target, difference_matrix);
        difference_matrix.mulScalar(scale, gradient_out);
    }
}

}
