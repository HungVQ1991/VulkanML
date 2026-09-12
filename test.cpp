#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "helper/cost_function.h"
#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "engine/graph_optimizer.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_sub_allocator.h"
#include "helper/layer.h"
#include "helper/learning_rate.h"
#include "math/matrix.h"
#include "neural_network.h"
#include "helper/optimizer.h"
#include "rl/dqn_agent.h"
#include "rl/replay_buffer.h"
#include "rl/transition.h"

bool nearlyEqual(float a, float b, float eps = 1e-3f)
{
    return std::fabs(a - b) < eps;
}

bool verifyMatrix(const Matrix &mat, const std::vector<float> &expected_data, float eps = 1e-3f)
{
    if (mat.getTarget() == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::vector<float> actual_data = mat.getData();
    if (actual_data.size() != expected_data.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < actual_data.size(); ++i)
    {
        if (!nearlyEqual(actual_data[i], expected_data[i], eps))
        {
            return false;
        }
    }
    return true;
}

bool testMatrixAddition(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {5.0f, 6.0f, 7.0f, 8.0f}, exec_target);
    Matrix res = mat_a + mat_b;
    bool standard_ok = verifyMatrix(res, {6.0f, 8.0f, 10.0f, 12.0f});

    Matrix mat_broadcast(1, 2, {10.0f, 20.0f}, exec_target);
    Matrix broadcast_res = mat_a + mat_broadcast;
    bool broadcast_ok = verifyMatrix(broadcast_res, {11.0f, 22.0f, 13.0f, 24.0f});

    return standard_ok && broadcast_ok;
}

bool testMatrixSubtraction(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {5.0f, 6.0f, 7.0f, 8.0f}, exec_target);
    Matrix mat_b(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix res = mat_a - mat_b;
    bool standard_ok = verifyMatrix(res, {4.0f, 4.0f, 4.0f, 4.0f});

    Matrix mat_broadcast(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix broadcast_res = mat_a - mat_broadcast;
    bool broadcast_ok = verifyMatrix(broadcast_res, {4.0f, 4.0f, 6.0f, 6.0f});

    return standard_ok && broadcast_ok;
}

bool testMatrixMultiplication(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {2.0f, 0.0f, 1.0f, 2.0f}, exec_target);
    Matrix res = mat_a * mat_b;
    return verifyMatrix(res, {4.0f, 4.0f, 10.0f, 8.0f});
}

bool testScalarOperations(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mul_res = mat_a * 2.0f;
    Matrix div_res = mat_a / 2.0f;

    bool mul_ok = verifyMatrix(mul_res, {2.0f, 4.0f, 6.0f, 8.0f});
    bool div_ok = verifyMatrix(div_res, {0.5f, 1.0f, 1.5f, 2.0f});

    return mul_ok && div_ok;
}

bool testHadamardOperations(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {2.0f, 0.0f, 1.0f, 2.0f}, exec_target);

    Matrix mul_res = mat_a.hadamardMul(mat_b);
    Matrix div_res = mat_a.hadamardDiv(mat_a);

    bool mul_ok = verifyMatrix(mul_res, {2.0f, 0.0f, 3.0f, 8.0f});
    bool div_ok = verifyMatrix(div_res, {1.0f, 1.0f, 1.0f, 1.0f});

    return mul_ok && div_ok;
}

bool testTransposeAndInverse(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix trans_res = mat_a.transpose();
    bool trans_ok = verifyMatrix(trans_res, {1.0f, 3.0f, 2.0f, 4.0f});

    Matrix mat_inv_target(2, 2, {4.0f, 7.0f, 2.0f, 6.0f}, exec_target);
    Matrix inv_res = mat_inv_target.inverse();
    bool inv_ok = verifyMatrix(inv_res, {0.6f, -0.7f, -0.2f, 0.4f});

    return trans_ok && inv_ok;
}

bool testNormalize(Execution_Target exec_target)
{
    Matrix vec_mat(1, 2, {3.0f, 4.0f}, exec_target);
    Matrix norm_res = vec_mat.normalize();
    return verifyMatrix(norm_res, {0.6f, 0.8f});
}

bool testMatmulAdd(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {2.0f, 0.0f, 1.0f, 2.0f}, exec_target);
    Matrix mat_bias(1, 2, {5.0f, 6.0f}, exec_target);

    Matrix res = mat_a.matmulAdd(mat_b, mat_bias);
    return verifyMatrix(res, {9.0f, 10.0f, 15.0f, 14.0f});
}

bool testRelu(Execution_Target exec_target)
{
    Matrix act_mat(2, 2, {-1.0f, 2.0f, 0.0f, -3.0f}, exec_target);
    Matrix relu_res = act_mat.relu();
    bool fwd_ok = verifyMatrix(relu_res, {0.0f, 2.0f, 0.0f, 0.0f});

    Matrix grad_out(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad_in = act_mat.reluBackward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, {0.0f, 2.0f, 0.0f, 0.0f});

    return fwd_ok && bwd_ok;
}

bool testGelu(Execution_Target exec_target)
{
    Matrix input_mat(1, 4, {0.0f, 1.0f, -1.0f, 2.0f}, exec_target);
    Matrix gelu_res = input_mat.gelu();
    bool fwd_ok = verifyMatrix(gelu_res, {0.0f, 0.8412316f, -0.1587684f, 1.9546059f});

    Matrix grad_in = input_mat.geluBackward(Matrix(1, 4, {1.0f, 1.0f, 1.0f, 1.0f}, exec_target));
    bool bwd_ok = verifyMatrix(grad_in, {0.5f, 1.0829548f, -0.0829548f, 1.085999f});

    return fwd_ok && bwd_ok;
}

bool testSoftmax(Execution_Target exec_target)
{
    Matrix softmax_in(1, 2, {0.0f, 0.0f}, exec_target);
    Matrix softmax_res = softmax_in.softmax();
    bool fwd_ok = verifyMatrix(softmax_res, {0.5f, 0.5f});

    Matrix grad_out(1, 2, {1.0f, -1.0f}, exec_target);
    Matrix grad_in = softmax_res.softmaxBackward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, {0.5f, -0.5f});

    return fwd_ok && bwd_ok;
}

bool testMseLoss(Execution_Target exec_target)
{
    Mse_Cost cost_func(exec_target);
    Matrix pred(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix target(1, 2, {2.0f, 4.0f}, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 2.5f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, {-1.0f, -2.0f});

    return loss_ok && grad_ok;
}

bool testMaeLoss(Execution_Target exec_target)
{
    Mae_Cost cost_func(exec_target);
    Matrix pred(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix target(1, 2, {2.0f, 4.0f}, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 1.5f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, {-0.5f, -0.5f});

    return loss_ok && grad_ok;
}

bool testBceLoss(Execution_Target exec_target)
{
    Bce_Cost cost_func(1e-7f, exec_target);
    Matrix pred(1, 2, {0.8f, 0.2f}, exec_target);
    Matrix target(1, 2, {1.0f, 0.0f}, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 0.22314355f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, {-0.625f, 0.625f});

    return loss_ok && grad_ok;
}

bool testCceLoss(Execution_Target exec_target)
{
    Cce_Cost cost_func(1e-7f, exec_target);
    Matrix pred(1, 3, {0.7f, 0.2f, 0.1f}, exec_target);
    Matrix target(1, 3, {1.0f, 0.0f, 0.0f}, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 0.356675f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, {-0.3f, 0.2f, 0.1f});

    return loss_ok && grad_ok;
}

bool testHuberLoss(Execution_Target exec_target)
{
    Huber_Cost cost_func(1.0f, exec_target);
    Matrix pred(1, 3, {1.0f, 2.0f, 5.0f}, exec_target);
    Matrix target(1, 3, {1.5f, 4.0f, 2.0f}, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 1.375f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, {-0.166667f, -0.333333f, 0.333333f});

    return loss_ok && grad_ok;
}

bool testLinearLayer(Execution_Target exec_target)
{
    Linear_Layer layer(2, 3, exec_target);
    layer.setWeights(Matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, exec_target));
    layer.setBiases(Matrix(1, 3, {0.1f, 0.2f, 0.3f}, exec_target));

    Matrix input_x(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix output_y = layer.forward(input_x);
    bool fwd_ok = verifyMatrix(output_y, {9.1f, 12.2f, 15.3f});

    Matrix grad_out(1, 3, {1.0f, 1.0f, 1.0f}, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_input_ok = verifyMatrix(grad_in, {6.0f, 15.0f});

    bool bwd_w_ok = verifyMatrix(layer.getWeightsGradient(), {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f});
    bool bwd_b_ok = verifyMatrix(layer.getBiasesGradient(), {1.0f, 1.0f, 1.0f});

    return fwd_ok && bwd_input_ok && bwd_w_ok && bwd_b_ok;
}

bool testConv2dLayer(Execution_Target exec_target)
{
    Conv2d_Layer layer(3, 3, 1, 1, 2, 1, 0, exec_target);
    auto params = layer.getParametersAndGradients();
    params[0].first->uploadData({1.0f, 0.0f, 0.0f, 1.0f});
    params[1].first->uploadData({0.0f});

    Matrix input_mat(1, 9, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, {6.0f, 8.0f, 12.0f, 14.0f});

    Matrix grad_out(1, 4, {1.0f, 1.0f, 1.0f, 1.0f}, exec_target);
    Matrix grad_in = layer.backward(grad_out);

    bool grad_in_ok = verifyMatrix(grad_in, {1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f});
    bool grad_w_ok = verifyMatrix(layer.getWeightsGradient(), {12.0f, 16.0f, 24.0f, 28.0f});

    return fwd_ok && grad_in_ok && grad_w_ok;
}

bool testMaxPool2dLayer(Execution_Target exec_target)
{
    Max_Pool_2d_Layer layer(4, 4, 1, 2, 2, 0, exec_target);
    Matrix input_mat(1, 16, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}, exec_target);

    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, {6.0f, 8.0f, 14.0f, 16.0f});

    Matrix grad_out(1, 4, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 2.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 3.0f, 0.0f, 4.0f});

    return fwd_ok && bwd_ok;
}

bool testGlobalAvgPool2dLayer(Execution_Target exec_target)
{
    Global_Avg_Pool_2d_Layer layer(2, 2, 2, exec_target);
    Matrix input_mat(1, 8, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, exec_target);

    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, {4.0f, 5.0f});

    Matrix grad_out(1, 2, {4.0f, 8.0f}, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f});

    return fwd_ok && bwd_ok;
}

bool testBatchNormLayer(Execution_Target exec_target)
{
    Batch_Norm_Layer layer(1, 1e-5f, 0.1f, exec_target);
    layer.setTrainingMode(true);

    Matrix input_mat(3, 1, {1.0f, 2.0f, 3.0f}, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, {-1.2247f, 0.0f, 1.2247f});

    Matrix grad_out(3, 1, {1.0f, 2.0f, 1.0f}, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, {-0.4082f, 0.8165f, -0.4082f});

    return fwd_ok && bwd_ok;
}

bool testBatchNorm2dLayer(Execution_Target exec_target)
{
    Batch_Norm_2d_Layer layer(1, 2, 1, 1e-5f, 0.1f, exec_target);
    layer.setTrainingMode(true);

    Matrix input_mat(2, 2, {1.0f, 3.0f, 5.0f, 7.0f}, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, {-1.34164f, -0.44721f, 0.44721f, 1.34164f});

    Matrix grad_out(2, 2, {1.0f, 0.0f, 0.0f, 0.0f}, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, {0.13416f, -0.17889f, -0.04472f, 0.08944f});

    return fwd_ok && bwd_ok;
}

bool testResNetBlock2dLayer(Execution_Target exec_target)
{
    auto validateTensor = [](const Matrix &tensor, std::size_t expected_rows, std::size_t expected_columns, bool require_nonzero = false) -> bool
    {
        if (tensor.getTarget() == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }

        if (tensor.getRows() != expected_rows || tensor.getColumns() != expected_columns)
        {
            return false;
        }

        std::vector<float> tensor_data = tensor.getData();
        if (tensor_data.size() != expected_rows * expected_columns)
        {
            return false;
        }

        bool has_nonzero_value = false;
        for (float value : tensor_data)
        {
            if (std::isnan(value) || std::isinf(value))
            {
                return false;
            }
            if (std::abs(value) > 1e-7f)
            {
                has_nonzero_value = true;
            }
        }
        return require_nonzero ? has_nonzero_value : true;
    };

    auto validateParameters = [&validateTensor](const std::vector<std::pair<Matrix *, Matrix *>> &parameters, std::size_t expected_count) -> bool
    {
        if (parameters.size() != expected_count)
        {
            return false;
        }

        for (const auto &[param, grad] : parameters)
        {
            if (!param || !grad)
            {
                return false;
            }
            if (!validateTensor(*param, param->getRows(), param->getColumns(), false))
            {
                return false;
            }
            if (!validateTensor(*grad, grad->getRows(), grad->getColumns(), false))
            {
                return false;
            }
        }
        return true;
    };

    constexpr std::size_t batch_size = 1;
    constexpr std::size_t input_height = 4;
    constexpr std::size_t input_width = 4;
    constexpr std::size_t input_channels = 16;
    constexpr std::size_t input_features = input_height * input_width * input_channels;

    std::vector<float> input_data(input_features);
    for (std::size_t i = 0; i < input_features; ++i)
    {
        input_data[i] = 0.01f * static_cast<float>((i % 17) + 1);
    }

    std::vector<float> grad_identity_data(input_features);
    for (std::size_t i = 0; i < input_features; ++i)
    {
        grad_identity_data[i] = 0.02f * static_cast<float>((i % 13) + 1);
    }

    Res_Net_Block_2d_Layer block_identity(input_height, input_width, input_channels, input_channels, 1, exec_target);
    Matrix input_identity(batch_size, input_features, input_data, exec_target);
    Matrix out_identity = block_identity.forward(input_identity);
    bool identity_fwd_ok = validateTensor(out_identity, batch_size, input_features, true);

    Matrix grad_output_identity(batch_size, input_features, grad_identity_data, exec_target);
    Matrix grad_in_identity = block_identity.backward(grad_output_identity);
    bool identity_bwd_ok = validateTensor(grad_in_identity, batch_size, input_features, true);
    bool identity_params_ok = validateParameters(block_identity.getParametersAndGradients(), 8);
    block_identity.resetGradient();

    constexpr std::size_t proj_out_channels = 32;
    constexpr std::size_t proj_stride = 2;
    constexpr std::size_t proj_out_height = (input_height + proj_stride - 1) / proj_stride;
    constexpr std::size_t proj_out_width = (input_width + proj_stride - 1) / proj_stride;
    constexpr std::size_t proj_out_features = proj_out_height * proj_out_width * proj_out_channels;

    std::vector<float> grad_proj_data(proj_out_features);
    for (std::size_t i = 0; i < proj_out_features; ++i)
    {
        grad_proj_data[i] = 0.02f * static_cast<float>((i % 11) + 1);
    }

    Res_Net_Block_2d_Layer block_proj(input_height, input_width, input_channels, proj_out_channels, proj_stride, exec_target);
    Matrix input_proj(batch_size, input_features, input_data, exec_target);
    Matrix out_proj = block_proj.forward(input_proj);
    bool proj_fwd_ok = validateTensor(out_proj, batch_size, proj_out_features, true);

    Matrix grad_output_proj(batch_size, proj_out_features, grad_proj_data, exec_target);
    Matrix grad_in_proj = block_proj.backward(grad_output_proj);
    bool proj_bwd_ok = validateTensor(grad_in_proj, batch_size, input_features, true);
    bool proj_params_ok = validateParameters(block_proj.getParametersAndGradients(), 12);
    block_proj.resetGradient();

    return identity_fwd_ok && identity_bwd_ok && identity_params_ok &&
           proj_fwd_ok && proj_bwd_ok && proj_params_ok;
}

bool testSgdOptimizer(Execution_Target exec_target)
{
    Matrix param_mat(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad(2, 2, {0.5f, -1.0f, 2.0f, -3.0f}, exec_target);

    Sgd_Optimizer optimizer(0.1f, 100.0f);
    optimizer.step({{&param_mat, &grad}});

    return verifyMatrix(param_mat, {0.95f, 2.1f, 2.8f, 4.3f});
}

bool testAdamOptimizer(Execution_Target exec_target)
{
    Matrix param_mat(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix grad_mat(1, 2, {0.1f, -0.2f}, exec_target);

    Adam_Optimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f);
    optimizer.step({{&param_mat, &grad_mat}});

    return verifyMatrix(param_mat, {0.999f, 2.001f});
}

bool testLearningRateSchedulers()
{
    No_Decay no_decay(0.01f);
    no_decay.step();
    bool no_decay_ok = nearlyEqual(no_decay.getCurrentRate(), 0.01f);

    Step_Decay step_decay(0.1f, 1e-6f, 0.5f, 2);
    step_decay.step();
    bool step1_ok = nearlyEqual(step_decay.getCurrentRate(), 0.1f);
    step_decay.step();
    bool step2_ok = nearlyEqual(step_decay.getCurrentRate(), 0.05f);

    Multi_Step_Decay multi_step(0.1f, 1e-6f, 0.1f, {2.0f, 4.0f});
    multi_step.step();
    multi_step.step();
    bool multi_step_ok = nearlyEqual(multi_step.getCurrentRate(), 0.01f);

    Exponential_Decay exp_decay(0.1f, 1e-6f, 0.9f);
    exp_decay.step();
    bool exp_ok = nearlyEqual(exp_decay.getCurrentRate(), 0.09f);

    Cosine_Annealing cosine_decay(0.1f, 0.0f, 4);
    cosine_decay.step();
    cosine_decay.step();
    bool cosine_ok = nearlyEqual(cosine_decay.getCurrentRate(), 0.05f);

    Polynomial_Decay poly_decay(0.1f, 0.0f, 4);
    poly_decay.step();
    poly_decay.step();
    bool poly_ok = nearlyEqual(poly_decay.getCurrentRate(), 0.05f);

    Reduce_On_Plateau plateau_decay(0.1f, 1e-6f, 0.5f, 2, false);
    plateau_decay.step(1.0f);
    plateau_decay.step(1.2f);
    plateau_decay.step(1.3f);
    bool plateau_ok = nearlyEqual(plateau_decay.getCurrentRate(), 0.05f);

    return no_decay_ok && step1_ok && step2_ok && multi_step_ok && exp_ok && cosine_ok && poly_ok && plateau_ok;
}

bool testMatrixSerialization(Execution_Target exec_target)
{
    std::string temp_file = "temp_matrix_serialization.bin";
    Matrix original(2, 3, {1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f}, exec_target);

    std::ofstream out_file(temp_file, std::ios::binary);
    if (!out_file.is_open())
    {
        return false;
    }
    original.saveMatrix(out_file);
    out_file.close();

    std::ifstream in_file(temp_file, std::ios::binary);
    if (!in_file.is_open())
    {
        return false;
    }
    Matrix loaded = Matrix::loadMatrix(in_file, exec_target);
    in_file.close();
    std::remove(temp_file.c_str());

    return verifyMatrix(loaded, {1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f});
}

bool testModelInferenceSerialization(Execution_Target exec_target)
{
    std::string temp_file = "temp_model_inference.bin";

    Neural_Network network(exec_target);
    network.addLayer<Linear_Layer>(2, 3, exec_target);
    network.addLayer<Relu_Layer>(exec_target);
    network.addLayer<Linear_Layer>(3, 1, exec_target);

    Matrix input_data(1, 2, {1.5f, -0.5f}, exec_target);
    Matrix pred_before = network.forward(input_data);

    network.saveInference(temp_file);

    Neural_Network loaded_network(exec_target);
    loaded_network.loadInference(temp_file, exec_target);
    std::remove(temp_file.c_str());

    Matrix pred_after = loaded_network.forward(input_data);

    return verifyMatrix(pred_after, pred_before.getData());
}

bool testGpuVectorLifecycle()
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    const Vulkan_Context &context = engine.getContext();

    std::size_t initial_count = 1024;
    auto vec = std::make_unique<gpu::vector>(context, initial_count);
    if (vec->getElementCount() != initial_count || vec->getBuffer() == VK_NULL_HANDLE)
    {
        return false;
    }

    std::vector<float> upload_sample(initial_count, 3.1415f);
    vec->uploadData(upload_sample);

    std::vector<float> download_sample;
    vec->downloadData(download_sample);
    if (download_sample.size() != initial_count || !nearlyEqual(download_sample[0], 3.1415f))
    {
        return false;
    }

    std::size_t expanded_count = 2048;
    vec->allocateMemory(expanded_count);
    if (vec->getElementCount() != expanded_count)
    {
        return false;
    }

    gpu::vector moved_vec = std::move(*vec);
    if (moved_vec.getElementCount() != expanded_count || vec->getBuffer() != VK_NULL_HANDLE)
    {
        return false;
    }

    return true;
}

bool testVulkanSubAllocatorAndGarbageCollection()
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    const Vulkan_Context &context = engine.getContext();
    Vulkan_Sub_Allocator &allocator = context.getAllocator();

    VkMemoryRequirements mem_req{
        .size = 1024 * 1024,
        .alignment = 256,
        .memoryTypeBits = 0xFFFFFFFF};

    Memory_Allocation alloc_a = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Memory_Allocation alloc_b = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (alloc_a.memory == VK_NULL_HANDLE || alloc_b.memory == VK_NULL_HANDLE)
    {
        return false;
    }

    VkDeviceSize offset_a = alloc_a.offset;
    allocator.free(alloc_a);

    Memory_Allocation alloc_c = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bool reuse_ok = (alloc_c.offset == offset_a);

    allocator.free(alloc_b);
    allocator.free(alloc_c);

    std::uint32_t current_frame = context.getCurrentFrame();
    Memory_Allocation garbage_alloc = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    context.deferDestruction(current_frame, VK_NULL_HANDLE, garbage_alloc);

    context.cleanGarbage(current_frame);

    return reuse_ok;
}

bool testOperatorFusionAndGraphExecution()
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.getCurrentGraph().clear();

    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, Execution_Target::VULKAN_GPU);
    Matrix mat_b(2, 2, {2.0f, 3.0f, 4.0f, 5.0f}, Execution_Target::VULKAN_GPU);

    Matrix mat_add = mat_a + mat_b;
    Matrix mat_relu = mat_add.relu();

    std::size_t raw_node_count = engine.getCurrentGraph().getNodeCount();
    if (raw_node_count < 2)
    {
        return false;
    }

    engine.executeGraph();

    return verifyMatrix(mat_relu, {3.0f, 5.0f, 7.0f, 9.0f});
}

class Dummy_Data_Pipeline : public Async_Data_Pipeline
{
protected:
    void prepareBatchHost(std::size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets) override
    {
        std::fill(output_inputs.begin(), output_inputs.end(), 1.0f);
        std::fill(output_targets.begin(), output_targets.end(), 2.0f);
    }

public:
    Dummy_Data_Pipeline() : Async_Data_Pipeline() {}
    std::size_t getBatchSize() const override { return 2; }
};

bool testAsyncDataPipeline()
{
    Dummy_Data_Pipeline pipeline;
    pipeline.initializeBuffers(2, 2, 2, Execution_Target::CPU);
    pipeline.start();

    Batch_Data batch1 = pipeline.nextBatch(2, 2, 2);
    bool b1_ok = (batch1.input_matrix != nullptr && batch1.input_matrix->getData().size() == 4);

    Batch_Data batch2 = pipeline.nextBatch(2, 2, 2);
    bool b2_ok = (batch2.target_matrix != nullptr && batch2.target_matrix->getData().size() == 4);

    pipeline.stop();
    return b1_ok && b2_ok;
}

bool testReplayBuffer()
{
    constexpr std::size_t capacity = 4;
    Replay_Buffer buffer(capacity, 1337);

    if (buffer.getSize() != 0 || buffer.getCapacity() != capacity || buffer.isReady(1))
    {
        return false;
    }

    for (std::size_t i = 0; i < 5; ++i)
    {
        buffer.push(Transition{
            .state = {static_cast<float>(i), static_cast<float>(i + 1)},
            .action = i % 2,
            .reward = static_cast<float>(i) * 0.5f,
            .next_state = {static_cast<float>(i + 1), static_cast<float>(i + 2)},
            .is_terminal = (i == 4)});
    }

    if (buffer.getSize() != capacity || !buffer.isReady(capacity) || buffer.isReady(capacity + 1))
    {
        return false;
    }

    Transition_Batch batch = buffer.sample(2);
    bool batch_dim_ok = (batch.batch_size == 2 && batch.state_dimension == 2 &&
                         batch.states.size() == 4 && batch.next_states.size() == 4 &&
                         batch.actions.size() == 2 && batch.rewards.size() == 2 &&
                         batch.terminals.size() == 2);

    buffer.clear();
    bool clear_ok = (buffer.getSize() == 0 && !buffer.isReady(1));

    return batch_dim_ok && clear_ok;
}

bool testDqnAgent(Execution_Target exec_target)
{
    constexpr std::size_t state_dim = 2;
    constexpr std::size_t action_dim = 2;

    Neural_Network q_net(exec_target);
    q_net.addLayer<Linear_Layer>(state_dim, action_dim, exec_target);
    q_net.setCostFunction<Huber_Cost>(1.0f, exec_target);
    q_net.setOptimizer<Adam_Optimizer>(0.01f);

    Neural_Network target_net(exec_target);
    target_net.addLayer<Linear_Layer>(state_dim, action_dim, exec_target);
    target_net.setCostFunction<Huber_Cost>(1.0f, exec_target);
    target_net.setOptimizer<Adam_Optimizer>(0.01f);

    Dqn_Agent agent(std::move(q_net), state_dim, action_dim, 100, 0.99f, 0.0f, 0.01f, 0.95f, exec_target, 1234);
    agent.initializeTargetNetworkFromPrototype(std::move(target_net));
    agent.setTargetUpdateParameters(2, false);

    std::size_t deterministic_action = agent.selectAction({1.0f, 0.5f}, false);
    bool action_ok = (deterministic_action < action_dim);

    for (std::size_t i = 0; i < 6; ++i)
    {
        agent.storeTransition(Transition{
            .state = {static_cast<float>(i), 1.0f},
            .action = i % action_dim,
            .reward = 1.0f,
            .next_state = {static_cast<float>(i + 1), 1.0f},
            .is_terminal = (i % 3 == 0)});
    }

    agent.trainStep(4);
    agent.trainStep(4);

    agent.setEpsilon(1.0f);
    agent.decayEpsilon();
    bool epsilon_ok = nearlyEqual(agent.getEpsilon(), 0.95f);

    agent.synchronizeTargetNetworkSoft(0.1f);
    agent.synchronizeTargetNetworkHard();

    return action_ok && epsilon_ok;
}

bool testMatrixConcatAndSplit(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 1, {5.0f, 6.0f}, exec_target);

    Matrix concat_cols_res = mat_a.concatenateCollumns(mat_b);
    bool concat_cols_ok = verifyMatrix(concat_cols_res, {1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f});

    auto [split_left, split_right] = concat_cols_res.splitCollumns(2);
    bool split_cols_ok = verifyMatrix(split_left, {1.0f, 2.0f, 3.0f, 4.0f}) &&
                         verifyMatrix(split_right, {5.0f, 6.0f});

    Matrix mat_row_a(1, 2, {10.0f, 20.0f}, exec_target);
    Matrix mat_row_b(2, 2, {30.0f, 40.0f, 50.0f, 60.0f}, exec_target);

    Matrix concat_rows_res = mat_row_a.concatenateRows(mat_row_b);
    bool concat_rows_ok = verifyMatrix(concat_rows_res, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

    auto [split_up, split_down] = concat_rows_res.splitRows(1);
    bool split_rows_ok = verifyMatrix(split_up, {10.0f, 20.0f}) &&
                         verifyMatrix(split_down, {30.0f, 40.0f, 50.0f, 60.0f});

    return concat_cols_ok && split_cols_ok && concat_rows_ok && split_rows_ok;
}

bool testPpoActorCriticForward(Execution_Target exec_target)
{
    constexpr std::uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto &actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Matrix(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}, exec_target));
    actor_linear.setBiases(Matrix(1, 2, {0.5f, -0.5f}, exec_target));

    auto &critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Matrix(2, 1, {1.0f, 2.0f}, exec_target));
    critic_linear.setBiases(Matrix(1, 1, {1.0f}, exec_target));

    Matrix input_matrix(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix output_matrix = ppo_layer.forward(input_matrix);

    bool output_ok = verifyMatrix(output_matrix, {1.5f, 1.5f, 6.0f, 3.5f, 3.5f, 12.0f});
    bool actor_sub_ok = verifyMatrix(ppo_layer.getActorOutput(), {1.5f, 1.5f, 3.5f, 3.5f});
    bool critic_sub_ok = verifyMatrix(ppo_layer.getCriticOutput(), {6.0f, 12.0f});

    return output_ok && actor_sub_ok && critic_sub_ok;
}

bool testPpoActorCriticBackward(Execution_Target exec_target)
{
    constexpr std::uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto &actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Matrix(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}, exec_target));
    actor_linear.setBiases(Matrix(1, 2, {0.0f, 0.0f}, exec_target));

    auto &critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Matrix(2, 1, {1.0f, 1.0f}, exec_target));
    critic_linear.setBiases(Matrix(1, 1, {0.0f}, exec_target));

    Matrix input_matrix(1, 2, {2.0f, 3.0f}, exec_target);
    ppo_layer.forward(input_matrix);

    Matrix output_gradient(1, 3, {1.0f, 2.0f, 3.0f}, exec_target);
    Matrix input_gradient = ppo_layer.backward(output_gradient);

    bool input_grad_ok = verifyMatrix(input_gradient, {4.0f, 5.0f});

    auto params = ppo_layer.getParametersAndGradients();
    bool params_count_ok = (params.size() == 4);

    bool actor_weight_grad_ok = verifyMatrix(*params[0].second, {2.0f, 4.0f, 3.0f, 6.0f});
    bool actor_bias_grad_ok = verifyMatrix(*params[1].second, {1.0f, 2.0f});
    bool critic_weight_grad_ok = verifyMatrix(*params[2].second, {6.0f, 9.0f});
    bool critic_bias_grad_ok = verifyMatrix(*params[3].second, {3.0f});

    return input_grad_ok && params_count_ok && actor_weight_grad_ok &&
           actor_bias_grad_ok && critic_weight_grad_ok && critic_bias_grad_ok;
}

bool testPpoActorCriticSerialization(Execution_Target exec_target)
{
    std::string temp_file = "temp_ppo_layer_serialization.bin";
    constexpr std::uint64_t action_dim = 2;

    PPO_Actor_Critic_Layer ppo_source(action_dim, exec_target);
    auto &actor_linear = ppo_source.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Matrix(2, 2, {1.5f, -0.5f, 0.5f, 2.0f}, exec_target));
    actor_linear.setBiases(Matrix(1, 2, {0.1f, -0.2f}, exec_target));

    auto &critic_linear = ppo_source.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Matrix(2, 1, {0.8f, -1.2f}, exec_target));
    critic_linear.setBiases(Matrix(1, 1, {0.5f}, exec_target));

    Matrix input_mat(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix pred_before = ppo_source.forward(input_mat);

    std::ofstream out_stream(temp_file, std::ios::binary);
    if (!out_stream.is_open())
    {
        return false;
    }
    ppo_source.saveInference(out_stream);
    out_stream.close();

    PPO_Actor_Critic_Layer ppo_loaded(action_dim, exec_target);
    ppo_loaded.addActorLayer<Linear_Layer>(2, 2, exec_target);
    ppo_loaded.addCriticLayer<Linear_Layer>(2, 1, exec_target);

    std::ifstream in_stream(temp_file, std::ios::binary);
    if (!in_stream.is_open())
    {
        return false;
    }
    ppo_loaded.loadInference(in_stream);
    in_stream.close();
    std::remove(temp_file.c_str());

    Matrix pred_after = ppo_loaded.forward(input_mat);

    return verifyMatrix(pred_after, pred_before.getData());
}

void runTestSuite(Execution_Target exec_target, const std::string &target_name)
{
    std::cout << "========================================\n";
    std::cout << "   RUNNING TEST SUITE ON " << target_name << "\n";
    std::cout << "========================================\n";

    // std::cout << "\n[1. Basic Matrix Arithmetics]\n";
    // std::cout << "  Matrix Addition (with Broadcast):  " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Matrix Subtraction (with Broadcast): " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Matrix Multiplication (GEMM):      " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Scalar Multiplication & Division:   " << (testScalarOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Hadamard Multiplication & Division: " << (testHadamardOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Matrix Concat & Split (Row/Col):   " << (testMatrixConcatAndSplit(exec_target) ? "PASS" : "FAIL") << "\n";

    // std::cout << "\n[2. Transformations & Advanced Operations]\n";
    // std::cout << "  Transpose & Matrix Inversion:       " << (testTransposeAndInverse(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Euclidean L2 Normalization:        " << (testNormalize(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Fused Linear Bias Add (MatmulAdd): " << (testMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";

    // std::cout << "\n[3. Activation Functions]\n";
    // std::cout << "  ReLU Forward & Backward:           " << (testRelu(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  GELU Forward & Backward:           " << (testGelu(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Softmax Forward & Backward:        " << (testSoftmax(exec_target) ? "PASS" : "FAIL") << "\n";

    // std::cout << "\n[4. Cost & Loss Functions]\n";
    // std::cout << "  MSE Cost & Gradient:               " << (testMseLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  MAE Cost & Gradient:               " << (testMaeLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  BCE Cost & Gradient:               " << (testBceLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  CCE Cost & Gradient:               " << (testCceLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Huber Cost & Gradient:             " << (testHuberLoss(exec_target) ? "PASS" : "FAIL") << "\n";

    // std::cout << "\n[5. Neural Network Layers]\n";
    // std::cout << "  Linear Layer (Forward & Backward): " << (testLinearLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Conv2D Layer (Forward & Backward): " << (testConv2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  MaxPool2D Layer (with Mask):       " << (testMaxPool2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  GlobalAvgPool2D Layer:             " << (testGlobalAvgPool2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  BatchNorm 1D Layer:                " << (testBatchNormLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  BatchNorm 2D Layer:                " << (testBatchNorm2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  ResNet Block 2D (Id & Proj):       " << (testResNetBlock2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  PPO Layer Forward:                 " << (testPpoActorCriticForward(exec_target) ? "PASS" : "FAIL") << "\n";       
    // std::cout << "  PPO Layer Backward & Accumulation: " << (testPpoActorCriticBackward(exec_target) ? "PASS" : "FAIL") << "\n";      
    // std::cout << "  PPO Layer Serialization I/O:       " << (testPpoActorCriticSerialization(exec_target) ? "PASS" : "FAIL") << "\n"; 

    // std::cout << "\n[6. Optimizers]\n";
    // std::cout << "  SGD Optimizer Step:                " << (testSgdOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Adam Optimizer Step:               " << (testAdamOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";

    // std::cout << "\n[7. Reinforcement Learning]\n";
    // std::cout << "  DQN Agent (Train Step & Target Sync): " << (testDqnAgent(exec_target) ? "PASS" : "FAIL") << "\n";

    // std::cout << "\n[8. Serialization & I/O]\n";
    // std::cout << "  Matrix Binary I/O:                 " << (testMatrixSerialization(exec_target) ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Model Inference I/O (NNI1):        " << (testModelInferenceSerialization(exec_target) ? "PASS" : "FAIL") << "\n\n";
}

int main()
{
    Logger::setFileLogging(true);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);
    Logger::logMessage("==================================Test log==================================", Log_Level::LOG_INFO, true);
    runTestSuite(Execution_Target::CPU, "CPU BACKEND");
    runTestSuite(Execution_Target::VULKAN_GPU, "VULKAN GPU BACKEND");

    // std::cout << "========================================\n";
    // std::cout << "   SYSTEM & LIFECYCLE MANAGEMENT TESTS  \n";
    // std::cout << "========================================\n";

    // std::cout << "  Learning Rate Schedulers Suite:    " << (testLearningRateSchedulers() ? "PASS" : "FAIL") << "\n";
    // std::cout << "  GPU Vector Lifecycle & Resizing:   " << (testGpuVectorLifecycle() ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Sub-Allocator & Garbage Collector: " << (testVulkanSubAllocatorAndGarbageCollection() ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Operator Fusion & Graph Dispatch:  " << (testOperatorFusionAndGraphExecution() ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Async Data Pipeline Double-Buffer: " << (testAsyncDataPipeline() ? "PASS" : "FAIL") << "\n";
    // std::cout << "  Replay Buffer Capacity & Sampling: " << (testReplayBuffer() ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n";

    return 0;
}