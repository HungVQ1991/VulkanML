#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
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
#include "math/tensor.h"
#include "neural_network.h"
#include "training_context.h"
#include "helper/optimizer.h"
#include "rl/dqn_agent.h"
#include "rl/replay_buffer.h"
#include "rl/transition.h"
#include "population.h"

bool nearlyEqual(float a, float b, float eps = 1e-3f)
{
    return std::fabs(a - b) < eps;
}

bool verifyMatrix(const Matrix& mat, const std::vector<float>& expected_data, float eps = 1e-3f)
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
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix mat_b(2, 2, { 5.0f, 6.0f, 7.0f, 8.0f }, exec_target);
    Matrix res = mat_a + mat_b;
    bool standard_ok = verifyMatrix(res, { 6.0f, 8.0f, 10.0f, 12.0f });

    Matrix mat_broadcast(1, 2, { 10.0f, 20.0f }, exec_target);
    Matrix broadcast_res = mat_a + mat_broadcast;
    bool broadcast_ok = verifyMatrix(broadcast_res, { 11.0f, 22.0f, 13.0f, 24.0f });

    return standard_ok && broadcast_ok;
}

bool testMatrixSubtraction(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 5.0f, 6.0f, 7.0f, 8.0f }, exec_target);
    Matrix mat_b(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix res = mat_a - mat_b;
    bool standard_ok = verifyMatrix(res, { 4.0f, 4.0f, 4.0f, 4.0f });

    Matrix mat_broadcast(1, 2, { 1.0f, 2.0f }, exec_target);
    Matrix broadcast_res = mat_a - mat_broadcast;
    bool broadcast_ok = verifyMatrix(broadcast_res, { 4.0f, 4.0f, 6.0f, 6.0f });

    return standard_ok && broadcast_ok;
}

bool testMatrixMultiplication(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix mat_b(2, 2, { 2.0f, 0.0f, 1.0f, 2.0f }, exec_target);
    Matrix res = mat_a * mat_b;
    return verifyMatrix(res, { 4.0f, 4.0f, 10.0f, 8.0f });
}

bool testScalarOperations(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix mul_res = mat_a * 2.0f;
    Matrix div_res = mat_a / 2.0f;

    bool mul_ok = verifyMatrix(mul_res, { 2.0f, 4.0f, 6.0f, 8.0f });
    bool div_ok = verifyMatrix(div_res, { 0.5f, 1.0f, 1.5f, 2.0f });

    return mul_ok && div_ok;
}

bool testHadamardOperations(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix mat_b(2, 2, { 2.0f, 0.0f, 1.0f, 2.0f }, exec_target);

    Matrix mul_res = mat_a.hadamardMul(mat_b);
    Matrix div_res = mat_a.hadamardDiv(mat_a);

    bool mul_ok = verifyMatrix(mul_res, { 2.0f, 0.0f, 3.0f, 8.0f });
    bool div_ok = verifyMatrix(div_res, { 1.0f, 1.0f, 1.0f, 1.0f });

    Matrix mat_broadcast(1, 2, { 2.0f, 4.0f }, exec_target);
    Matrix mul_bcast = mat_a.hadamardMul(mat_broadcast);
    Matrix div_bcast = mat_a.hadamardDiv(mat_broadcast);

    bool mul_bcast_ok = verifyMatrix(mul_bcast, { 2.0f, 8.0f, 6.0f, 16.0f });
    bool div_bcast_ok = verifyMatrix(div_bcast, { 0.5f, 0.5f, 1.5f, 1.0f });

    return mul_ok && div_ok && mul_bcast_ok && div_bcast_ok;
}

bool testTransposeAndInverse(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix trans_res = mat_a.transpose();
    bool trans_ok = verifyMatrix(trans_res, { 1.0f, 3.0f, 2.0f, 4.0f });

    Matrix mat_inv_target(2, 2, { 4.0f, 7.0f, 2.0f, 6.0f }, exec_target);
    Matrix inv_res = mat_inv_target.inverse();
    bool inv_ok = verifyMatrix(inv_res, { 0.6f, -0.7f, -0.2f, 0.4f });

    return trans_ok && inv_ok;
}

bool testNormalize(Execution_Target exec_target)
{
    Matrix vec_mat(1, 2, { 3.0f, 4.0f }, exec_target);
    Matrix norm_res = vec_mat.normalize();
    return verifyMatrix(norm_res, { 0.6f, 0.8f });
}

bool testMatmulAdd(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix mat_b(2, 2, { 2.0f, 0.0f, 1.0f, 2.0f }, exec_target);
    Matrix mat_bias(1, 2, { 5.0f, 6.0f }, exec_target);

    Matrix res = mat_a.matmulAdd(mat_b, mat_bias);
    return verifyMatrix(res, { 9.0f, 10.0f, 15.0f, 14.0f });
}

bool testBatchedTensorMatmul(Execution_Target exec_target)
{
    Tensor t_a(Shape{ 2, 2, 3 }, {
        1.0f, 2.0f, 1.0f,
        0.0f, 1.0f, 2.0f,
        2.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f
        }, exec_target);

    Tensor t_b(Shape{ 2, 3, 2 }, {
        1.0f, 0.0f,
        2.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 2.0f,
        1.0f, 0.0f,
        2.0f, 1.0f
        }, exec_target);

    Tensor res = t_a * t_b;
    bool ok_3d = verifyMatrix(res, { 6.0f, 3.0f, 4.0f, 3.0f, 2.0f, 5.0f, 1.0f, 2.0f });
    bool shape_3d_ok = (res.getShape() == Shape{ 2, 2, 2 });

    Tensor t_b_bcast(Shape{ 3, 2 }, {
        1.0f, 2.0f,
        0.0f, 1.0f,
        1.0f, 0.0f
        }, exec_target);

    Tensor res_bcast = t_a * t_b_bcast;
    bool ok_bcast = verifyMatrix(res_bcast, { 2.0f, 4.0f, 2.0f, 1.0f, 3.0f, 4.0f, 1.0f, 3.0f });
    bool shape_bcast_ok = (res_bcast.getShape() == Shape{ 2, 2, 2 });

    return ok_3d && shape_3d_ok && ok_bcast && shape_bcast_ok;
}

bool testBatchedTensorMatmulAdd(Execution_Target exec_target)
{
    Tensor t_a(Shape{ 2, 2, 2 }, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
        }, exec_target);

    Tensor t_w(Shape{ 2, 2 }, {
        1.0f, 0.0f,
        0.0f, 2.0f
        }, exec_target);

    Tensor t_b_broadcast(Shape{ 1, 2 }, { 10.0f, 20.0f }, exec_target);

    Tensor res_bcast = t_a.matmulAdd(t_w, t_b_broadcast);
    bool ok_bcast = verifyMatrix(res_bcast, { 11.0f, 24.0f, 13.0f, 28.0f, 15.0f, 32.0f, 17.0f, 36.0f });

    Tensor t_b_full(Shape{ 2, 2, 2 }, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
        }, exec_target);

    Tensor res_full = t_a.matmulAdd(t_w, t_b_full);
    bool ok_full = verifyMatrix(res_full, { 2.0f, 6.0f, 6.0f, 12.0f, 10.0f, 18.0f, 14.0f, 24.0f });

    return ok_bcast && ok_full;
}

bool testRelu(Execution_Target exec_target)
{
    Matrix act_mat(2, 2, { -1.0f, 2.0f, 0.0f, -3.0f }, exec_target);
    Matrix relu_res = act_mat.relu();
    bool fwd_ok = verifyMatrix(relu_res, { 0.0f, 2.0f, 0.0f, 0.0f });

    Matrix grad_out(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix grad_in = act_mat.reluBackward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.0f, 2.0f, 0.0f, 0.0f });

    return fwd_ok && bwd_ok;
}

bool testGelu(Execution_Target exec_target)
{
    Matrix input_mat(1, 4, { 0.0f, 1.0f, -1.0f, 2.0f }, exec_target);
    Matrix gelu_res = input_mat.gelu();
    bool fwd_ok = verifyMatrix(gelu_res, { 0.0f, 0.8412316f, -0.1587684f, 1.9546059f });

    Matrix grad_in = input_mat.geluBackward(Matrix(1, 4, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target));
    bool bwd_ok = verifyMatrix(grad_in, { 0.5f, 1.0829548f, -0.0829548f, 1.085999f });

    return fwd_ok && bwd_ok;
}

bool testSoftmax(Execution_Target exec_target)
{
    Matrix softmax_in(1, 2, { 0.0f, 0.0f }, exec_target);
    Matrix softmax_res = softmax_in.softmax();
    bool fwd_ok = verifyMatrix(softmax_res, { 0.5f, 0.5f });

    Matrix grad_out(1, 2, { 1.0f, -1.0f }, exec_target);
    Matrix grad_in = softmax_res.softmaxBackward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.5f, -0.5f });

    return fwd_ok && bwd_ok;
}

bool testMseLoss(Execution_Target exec_target)
{
    Mse_Cost cost_func(exec_target);
    Matrix pred(1, 2, { 1.0f, 2.0f }, exec_target);
    Matrix target(1, 2, { 2.0f, 4.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 2.5f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -1.0f, -2.0f });

    return loss_ok && grad_ok;
}

bool testMaeLoss(Execution_Target exec_target)
{
    Mae_Cost cost_func(exec_target);
    Matrix pred(1, 2, { 1.0f, 2.0f }, exec_target);
    Matrix target(1, 2, { 2.0f, 4.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 1.5f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.5f, -0.5f });

    return loss_ok && grad_ok;
}

bool testBceLoss(Execution_Target exec_target)
{
    Bce_Cost cost_func(1e-7f, exec_target);
    Matrix pred(1, 2, { 0.8f, 0.2f }, exec_target);
    Matrix target(1, 2, { 1.0f, 0.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 0.22314355f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.625f, 0.625f });

    return loss_ok && grad_ok;
}

bool testCceLoss(Execution_Target exec_target)
{
    Cce_Cost cost_func(1e-7f, exec_target);
    Matrix pred(1, 3, { 0.7f, 0.2f, 0.1f }, exec_target);
    Matrix target(1, 3, { 1.0f, 0.0f, 0.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 0.356675f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.3f, 0.2f, 0.1f });

    return loss_ok && grad_ok;
}

bool testHuberLoss(Execution_Target exec_target)
{
    Huber_Cost cost_func(1.0f, exec_target);
    Matrix pred(1, 3, { 1.0f, 2.0f, 5.0f }, exec_target);
    Matrix target(1, 3, { 1.5f, 4.0f, 2.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 1.375f);

    Matrix grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.166667f, -0.333333f, 0.333333f });

    return loss_ok && grad_ok;
}

bool testLinearLayer(Execution_Target exec_target)
{
    Linear_Layer layer(2, 3, exec_target);
    layer.setWeights(Matrix(2, 3, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, exec_target));
    layer.setBiases(Matrix(1, 3, { 0.1f, 0.2f, 0.3f }, exec_target));

    Matrix input_x(1, 2, { 1.0f, 2.0f }, exec_target);
    Matrix output_y = layer.forward(input_x);
    bool fwd_ok = verifyMatrix(output_y, { 9.1f, 12.2f, 15.3f });

    Matrix grad_out(1, 3, { 1.0f, 1.0f, 1.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_input_ok = verifyMatrix(grad_in, { 6.0f, 15.0f });

    bool bwd_w_ok = verifyMatrix(layer.getWeightsGradient(), { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f });
    bool bwd_b_ok = verifyMatrix(layer.getBiasesGradient(), { 1.0f, 1.0f, 1.0f });

    return fwd_ok && bwd_input_ok && bwd_w_ok && bwd_b_ok;
}

bool testConv2dLayer(Execution_Target exec_target)
{
    Conv2d_Layer layer(3, 3, 1, 1, 2, 1, 0, exec_target);
    auto params = layer.getParametersAndGradients();
    params[0].first->uploadData({ 1.0f, 0.0f, 0.0f, 1.0f });
    params[1].first->uploadData({ 0.0f });

    Matrix input_mat(1, 9, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 6.0f, 8.0f, 12.0f, 14.0f });

    Matrix grad_out(1, 4, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);

    bool grad_in_ok = verifyMatrix(grad_in, { 1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f });
    bool grad_w_ok = verifyMatrix(layer.getWeightsGradient(), { 12.0f, 16.0f, 24.0f, 28.0f });

    return fwd_ok && grad_in_ok && grad_w_ok;
}

bool testConv2dLayerFp16(Execution_Target exec_target)
{
    if (exec_target != Execution_Target::VULKAN_GPU)
    {
        return true;
    }
    const auto &context = Execution_Engine::getInstance().getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        return true;
    }

    Conv2d_Layer layer(3, 3, 1, 1, 2, 1, 0, exec_target);
    layer.setMixedPrecision(true);
    auto params = layer.getParametersAndGradients();
    params[0].first->uploadData({ 1.0f, 0.0f, 0.0f, 1.0f });
    params[1].first->uploadData({ 0.0f });

    Matrix input_mat(1, 9, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 6.0f, 8.0f, 12.0f, 14.0f }, 1e-2f);

    Matrix grad_out(1, 4, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);

    bool grad_in_ok = verifyMatrix(grad_in, { 1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f }, 1e-2f);
    bool grad_w_ok = verifyMatrix(layer.getWeightsGradient(), { 12.0f, 16.0f, 24.0f, 28.0f }, 1e-2f);

    return fwd_ok && grad_in_ok && grad_w_ok;
}

bool testNativeFp16ZeroCastPipeline(Execution_Target exec_target)
{
    if (exec_target != Execution_Target::VULKAN_GPU)
    {
        return true;
    }
    const auto &context = Execution_Engine::getInstance().getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        return true;
    }

    Neural_Network nn(exec_target);
    nn.addLayer<Conv2d_Layer>(4, 4, 1, 2, 2, 1, 0, exec_target);
    nn.addLayer<Batch_Norm_2d_Layer>(3, 3, 2, 1e-5f, 0.1f, exec_target);
    nn.addLayer<Gelu_Layer>(exec_target);
    nn.addLayer<Max_Pool_2d_Layer>(3, 3, 2, 2, 1, 0, exec_target);
    nn.addLayer<Linear_Layer>(2 * 2 * 2, 2, exec_target);
    nn.setOptimizer<Adam_Optimizer>(0.01f);
    nn.enableMixedPrecision(true);

    std::vector<float> in_data(16, 1.0f);
    for (std::size_t i = 0; i < 16; ++i) in_data[i] = static_cast<float>(i + 1) * 0.1f;
    Tensor input(1, 16, in_data, exec_target);
    Tensor target(1, 2, std::vector<float>{1.0f, 0.0f}, exec_target);

    Tensor out = nn.forward(input);
    const auto &out_data = out.getData();
    if (out_data.size() != 2) return false;
    for (float v : out_data)
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    nn.trainStep(input, target);

    Tensor out_after = nn.forward(input);
    const auto &out_after_data = out_after.getData();
    if (out_after_data.size() != 2) return false;
    for (float v : out_after_data)
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    return true;
}

bool testPopulationFp16(Execution_Target exec_target)
{
    if (exec_target != Execution_Target::VULKAN_GPU)
    {
        return true;
    }
    const auto &context = Execution_Engine::getInstance().getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        return true;
    }

    std::size_t state_dim = 16;
    std::size_t hidden_dim = 32;
    std::size_t action_dim = 4;
    std::size_t pop_size = 8;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, hidden_dim, exec_target);
    template_net.addLayer<Gelu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(hidden_dim, action_dim, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target);
    pop.enableMixedPrecision(true);

    std::vector<float> states(pop_size * state_dim, 0.5f);
    std::vector<std::size_t> actions(pop_size, 0);

    pop.selectBatchActions(states.data(), nullptr, pop_size, actions.data());

    for (std::size_t a : actions)
    {
        if (a >= action_dim) return false;
    }

    return true;
}

bool testMaxPool2dLayer(Execution_Target exec_target)
{
    Max_Pool_2d_Layer layer(4, 4, 1, 2, 2, 0, exec_target);
    Matrix input_mat(1, 16, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f }, exec_target);

    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 6.0f, 8.0f, 14.0f, 16.0f });

    Matrix grad_out(1, 4, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 2.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 3.0f, 0.0f, 4.0f });

    return fwd_ok && bwd_ok;
}

bool testGlobalAvgPool2dLayer(Execution_Target exec_target)
{
    Global_Avg_Pool_2d_Layer layer(2, 2, 2, exec_target);
    Matrix input_mat(1, 8, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, exec_target);

    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 4.0f, 5.0f });

    Matrix grad_out(1, 2, { 4.0f, 8.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

    return fwd_ok && bwd_ok;
}

bool testBatchNormLayer(Execution_Target exec_target)
{
    Batch_Norm_Layer layer(1, 1e-5f, 0.1f, exec_target);
    layer.setTrainingMode(true);

    Matrix input_mat(3, 1, { 1.0f, 2.0f, 3.0f }, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { -1.2247f, 0.0f, 1.2247f });

    Matrix grad_out(3, 1, { 1.0f, 2.0f, 1.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { -0.4082f, 0.8165f, -0.4082f });

    return fwd_ok && bwd_ok;
}

bool testBatchNorm2dLayer(Execution_Target exec_target)
{
    Batch_Norm_2d_Layer layer(1, 2, 1, 1e-5f, 0.1f, exec_target);
    layer.setTrainingMode(true);

    Matrix input_mat(2, 2, { 1.0f, 3.0f, 5.0f, 7.0f }, exec_target);
    Matrix out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { -1.34164f, -0.44721f, 0.44721f, 1.34164f });

    Matrix grad_out(2, 2, { 1.0f, 0.0f, 0.0f, 0.0f }, exec_target);
    Matrix grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.13416f, -0.17889f, -0.04472f, 0.08944f });

    return fwd_ok && bwd_ok;
}

bool testResNetBlock2dLayer(Execution_Target exec_target)
{
    auto validateTensor = [](const Matrix& tensor, std::size_t expected_rows, std::size_t expected_columns, bool require_nonzero = false) -> bool
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

    auto validateParameters = [&validateTensor](const std::vector<std::pair<Matrix*, Matrix*>>& parameters, std::size_t expected_count) -> bool
        {
            if (parameters.size() != expected_count)
            {
                return false;
            }

            for (const auto& [param, grad] : parameters)
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
    Matrix param_mat(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix grad(2, 2, { 0.5f, -1.0f, 2.0f, -3.0f }, exec_target);

    Sgd_Optimizer optimizer(0.1f, 100.0f);
    optimizer.step({ { &param_mat, &grad } });

    return verifyMatrix(param_mat, { 0.95f, 2.1f, 2.8f, 4.3f });
}

bool testAdamOptimizer(Execution_Target exec_target)
{
    Matrix param_mat(1, 2, { 1.0f, 2.0f }, exec_target);
    Matrix grad_mat(1, 2, { 0.1f, -0.2f }, exec_target);

    Adam_Optimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f);
    optimizer.step({ { &param_mat, &grad_mat } });

    return verifyMatrix(param_mat, { 0.999f, 2.001f });
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

    Multi_Step_Decay multi_step(0.1f, 1e-6f, 0.1f, { 2.0f, 4.0f });
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
    Matrix original(2, 3, { 1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f }, exec_target);

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

    return verifyMatrix(loaded, { 1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f });
}

bool testModelInferenceSerialization(Execution_Target exec_target)
{
    std::string temp_file = "temp_model_inference.bin";

    Neural_Network network(exec_target);
    network.addLayer<Linear_Layer>(2, 3, exec_target);
    network.addLayer<Relu_Layer>(exec_target);
    network.addLayer<Linear_Layer>(3, 1, exec_target);

    Matrix input_data(1, 2, { 1.5f, -0.5f }, exec_target);
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
    Execution_Engine& engine = Execution_Engine::getInstance();
    const Vulkan_Context& context = engine.getContext();

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
    Execution_Engine& engine = Execution_Engine::getInstance();
    const Vulkan_Context& context = engine.getContext();
    Vulkan_Sub_Allocator& allocator = context.getAllocator();

    VkMemoryRequirements mem_req{
        .size = 1024 * 1024,
        .alignment = 256,
        .memoryTypeBits = 0xFFFFFFFF };

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
    Execution_Engine& engine = Execution_Engine::getInstance();
    engine.getCurrentGraph().clear();

    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, Execution_Target::VULKAN_GPU);
    Matrix mat_b(2, 2, { 2.0f, 3.0f, 4.0f, 5.0f }, Execution_Target::VULKAN_GPU);

    Matrix mat_add = mat_a + mat_b;
    Matrix mat_relu = mat_add.relu();

    std::size_t raw_node_count = engine.getCurrentGraph().getNodeCount();
    if (raw_node_count < 2)
    {
        return false;
    }

    engine.executeGraph();
    if (!verifyMatrix(mat_relu, { 3.0f, 5.0f, 7.0f, 9.0f }))
    {
        return false;
    }

    Matrix mat_c(2, 2, { -10.0f, 5.0f, 0.0f, 2.0f }, Execution_Target::VULKAN_GPU);
    Matrix mat_d(2, 2, { 3.0f, 2.0f, 1.0f, -5.0f }, Execution_Target::VULKAN_GPU);
    Matrix mat_add2 = mat_c + mat_d;
    Matrix mat_relu2 = mat_add2.relu();

    engine.executeGraph();
    if (!verifyMatrix(mat_relu2, { 0.0f, 7.0f, 1.0f, 0.0f }))
    {
        return false;
    }

    Matrix mat_e(2, 2, { 10.0f, -20.0f, 30.0f, -40.0f }, Execution_Target::VULKAN_GPU);
    Matrix mat_f(2, 2, { 5.0f, 5.0f, -5.0f, -5.0f }, Execution_Target::VULKAN_GPU);
    Matrix mat_add3 = mat_e + mat_f;
    Matrix mat_relu3 = mat_add3.relu();

    engine.executeGraph();
    return verifyMatrix(mat_relu3, { 15.0f, 0.0f, 25.0f, 0.0f });
}

bool testBatchedTensorMatmulFusion()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    engine.getCurrentGraph().clear();

    Tensor t_a(Shape{ 2, 2, 2 }, {
        -5.0f, 2.0f,
        3.0f, -4.0f,
        1.0f, -2.0f,
        -3.0f, 4.0f
        }, Execution_Target::VULKAN_GPU);

    Tensor t_w(Shape{ 2, 2 }, {
        1.0f, 0.0f,
        0.0f, 1.0f
        }, Execution_Target::VULKAN_GPU);

    Tensor t_b(Shape{ 1, 2 }, { 1.0f, 0.0f }, Execution_Target::VULKAN_GPU);

    Tensor t_mmadd = t_a.matmulAdd(t_w, t_b);
    Tensor t_fused = t_mmadd.relu();

    engine.executeGraph();

    return verifyMatrix(t_fused, { 0.0f, 2.0f, 4.0f, 0.0f, 2.0f, 0.0f, 0.0f, 4.0f });
}

class Dummy_Data_Pipeline : public Async_Data_Pipeline
{
protected:
    void prepareBatchHost(std::size_t batch_step, std::vector<float>& output_inputs, std::vector<float>& output_targets) override
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
            .state = { static_cast<float>(i), static_cast<float>(i + 1) },
            .action = i % 2,
            .reward = static_cast<float>(i) * 0.5f,
            .next_state = { static_cast<float>(i + 1), static_cast<float>(i + 2) },
            .is_terminal = (i == 4) });
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

    std::size_t deterministic_action = agent.selectAction({ 1.0f, 0.5f }, false);
    bool action_ok = (deterministic_action < action_dim);

    for (std::size_t i = 0; i < 6; ++i)
    {
        agent.storeTransition(Transition{
            .state = { static_cast<float>(i), 1.0f },
            .action = i % action_dim,
            .reward = 1.0f,
            .next_state = { static_cast<float>(i + 1), 1.0f },
            .is_terminal = (i % 3 == 0) });
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

bool testPopulation(Execution_Target exec_target)
{
    constexpr std::size_t pop_size = 8;
    constexpr std::size_t state_dim = 16;
    constexpr std::size_t action_dim = 4;
    constexpr std::size_t hidden_dim = 32;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, hidden_dim, exec_target);
    template_net.addLayer<Gelu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(hidden_dim, action_dim, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target, 42);
    if (pop.getPopulationSize() != pop_size || pop.getStateDimension() != state_dim || pop.getActionSpaceSize() != action_dim)
    {
        return false;
    }

    Population pop_tmpl(pop_size, template_net, state_dim, action_dim, exec_target, 123);
    if (pop_tmpl.getPopulationSize() != pop_size)
    {
        return false;
    }

    std::vector<float> test_state(state_dim);
    for (std::size_t i = 0; i < state_dim; ++i)
    {
        test_state[i] = std::sin(static_cast<float>(i) * 0.5f);
    }

    for (std::size_t i = 0; i < pop_size; ++i)
    {
        std::size_t act1 = pop.selectAction(i, test_state);
        std::size_t act2 = pop.selectAction(i, test_state);
        if (act1 != act2 || act1 >= action_dim)
        {
            return false;
        }
    }

    Neural_Network ind0 = pop.getIndividual(0);
    Matrix ind0_input(1, state_dim, test_state, exec_target);
    Matrix ind0_output = ind0.forward(ind0_input);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    std::vector<float> q_vals = ind0_output.getData();
    std::size_t expected_act0 = 0;
    float max_q = q_vals[0];
    for (std::size_t a = 1; a < q_vals.size(); ++a)
    {
        if (q_vals[a] > max_q)
        {
            max_q = q_vals[a];
            expected_act0 = a;
        }
    }
    if (pop.selectAction(0, test_state) != expected_act0)
    {
        return false;
    }

    std::vector<float> flat_states(pop_size * state_dim);
    std::vector<std::size_t> expected_batch(pop_size);
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        for (std::size_t d = 0; d < state_dim; ++d)
        {
            flat_states[i * state_dim + d] = std::cos(static_cast<float>(i * state_dim + d) * 0.2f);
        }
        expected_batch[i] = pop.selectAction(i, flat_states.data() + (i * state_dim));
    }

    std::vector<std::size_t> batch_results(pop_size);
    pop.selectBatchActions(flat_states.data(), nullptr, pop_size, batch_results.data());
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        if (batch_results[i] != expected_batch[i])
        {
            return false;
        }
    }

    std::vector<std::size_t> active_indices = { 1, 3, 6 };
    std::vector<float> subset_states(active_indices.size() * state_dim);
    for (std::size_t k = 0; k < active_indices.size(); ++k)
    {
        std::size_t idx = active_indices[k];
        std::copy(flat_states.begin() + idx * state_dim,
            flat_states.begin() + (idx + 1) * state_dim,
            subset_states.begin() + k * state_dim);
    }
    std::vector<std::size_t> subset_results(active_indices.size());
    pop.selectBatchActions(subset_states.data(), active_indices.data(), active_indices.size(), subset_results.data());
    for (std::size_t k = 0; k < active_indices.size(); ++k)
    {
        if (subset_results[k] != expected_batch[active_indices[k]])
        {
            return false;
        }
    }

    const std::string temp_file = "temp_population_individual.bin";
    if (!pop.saveIndividual(2, temp_file))
    {
        return false;
    }

    Population pop_loader(pop_size, template_net, state_dim, action_dim, exec_target, 9999);
    if (!pop_loader.loadIndividual(5, temp_file))
    {
        std::remove(temp_file.c_str());
        return false;
    }
    std::remove(temp_file.c_str());

    for (std::size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop_loader.selectAction(5, st) != pop.selectAction(2, st))
        {
            return false;
        }
    }

    constexpr std::size_t elite_candidate = 4;
    std::vector<std::size_t> ind4_actions(pop_size);
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        ind4_actions[i] = pop.selectAction(elite_candidate, flat_states.data() + i * state_dim);
    }

    std::vector<float> fitness(pop_size, 0.0f);
    fitness[elite_candidate] = 1000.0f;

    Neural_Network best_ind_before = pop.getBestIndividual(fitness.data());
    Matrix in_matrix_best(1, state_dim, std::vector<float>(flat_states.begin(), flat_states.begin() + state_dim), exec_target);
    Matrix out_matrix_best = best_ind_before.forward(in_matrix_best);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    const auto& best_raw_data = out_matrix_best.getData();
    std::size_t best_initial_act = static_cast<std::size_t>(std::distance(best_raw_data.begin(), std::max_element(best_raw_data.begin(), best_raw_data.end())));
    if (best_initial_act != ind4_actions[0])
    {
        return false;
    }

    const std::string best_temp = "temp_best_champ.bin";
    if (!pop.saveBestIndividual(best_temp, fitness.data()))
    {
        return false;
    }
    if (!pop_loader.loadBestIndividual(best_temp))
    {
        std::remove(best_temp.c_str());
        return false;
    }
    std::remove(best_temp.c_str());

    for (std::size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop_loader.selectAction(0, st) != ind4_actions[i])
        {
            return false;
        }
    }

    std::string checkpoint_temp = "temp_population_checkpoint.bin";
    std::uint64_t saved_gen = 105;
    if (!pop.saveCheckpoint(checkpoint_temp, saved_gen, fitness.data()))
    {
        return false;
    }

    std::uint64_t loaded_gen = 0;
    std::vector<float> loaded_fitness(pop_size, 0.0f);
    if (!pop_loader.loadCheckpoint(checkpoint_temp, loaded_gen, loaded_fitness.data()))
    {
        std::remove(checkpoint_temp.c_str());
        return false;
    }
    if (loaded_gen != saved_gen)
    {
        std::remove(checkpoint_temp.c_str());
        return false;
    }
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        if (!nearlyEqual(loaded_fitness[i], fitness[i]))
        {
            std::remove(checkpoint_temp.c_str());
            return false;
        }
    }

    std::uint64_t loaded_gen_null = 0;
    if (!pop_loader.loadCheckpoint(checkpoint_temp, loaded_gen_null, nullptr))
    {
        std::remove(checkpoint_temp.c_str());
        return false;
    }
    std::remove(checkpoint_temp.c_str());
    if (loaded_gen_null != saved_gen)
    {
        return false;
    }

    pop.evolve(fitness.data(), 0.15f, 0.0f, 0.0f, 0.0f, 3, 1);
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop.selectAction(0, st) != ind4_actions[i])
        {
            return false;
        }
    }

    pop.evolve(fitness.data(), 0.15f, 0.8f, 0.3f, 0.5f, 3, 1);

    return true;
}

bool testLayerInterfaceContracts(Execution_Target exec_target)
{
    std::mt19937 random_engine(1337);

    Linear_Layer lin(2, 3, exec_target, 2.0f);
    auto lin_clone = lin.clone();
    if (!lin_clone || lin_clone->getLayerType() != Layer_Type::LINEAR || lin_clone->getExecutionTarget() != exec_target)
    {
        return false;
    }
    auto lin_dims = lin.getPopulationParameterDims();
    if (lin_dims.size() != 2 || lin_dims[0] != Shape{ 2, 3 } || lin_dims[1] != Shape{ 1, 3 })
    {
        return false;
    }
    auto lin_evolvable = lin.getPopulationParameterIsEvolvable();
    if (lin_evolvable.size() != 2 || !lin_evolvable[0] || !lin_evolvable[1])
    {
        return false;
    }
    auto lin_init_w = lin.getPopulationParameterInitializer(0);
    auto lin_init_b = lin.getPopulationParameterInitializer(1);
    if (!lin_init_w || !lin_init_b || lin_init_b(random_engine) != 0.0f)
    {
        return false;
    }
    lin_clone->setPopulationParameter(0, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    lin_clone->setPopulationParameter(1, { 0.1f, 0.2f, 0.3f });
    Matrix lin_in(1, 2, { 1.0f, 2.0f }, exec_target);
    Matrix lin_out = lin_clone->forward(lin_in);
    if (!verifyMatrix(lin_out, { 9.1f, 12.2f, 15.3f }))
    {
        return false;
    }
    bool lin_bounds_ok = false;
    try
    {
        lin.setPopulationParameter(2, {});
    }
    catch (const std::out_of_range&)
    {
        lin_bounds_ok = true;
    }
    if (!lin_bounds_ok)
    {
        return false;
    }

    Relu_Layer relu(exec_target);
    auto relu_clone = relu.clone();
    if (!relu_clone || relu_clone->getLayerType() != Layer_Type::RELU || !relu.getPopulationParameterDims().empty() || !relu.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }
    bool relu_throw_ok = false;
    try
    {
        relu.setPopulationParameter(0, { 1.0f });
    }
    catch (const std::out_of_range&)
    {
        relu_throw_ok = true;
    }
    if (!relu_throw_ok)
    {
        return false;
    }

    Gelu_Layer gelu(exec_target);
    auto gelu_clone = gelu.clone();
    if (!gelu_clone || gelu_clone->getLayerType() != Layer_Type::GELU || !gelu.getPopulationParameterDims().empty() || !gelu.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    Softmax_Layer softmax(false, exec_target);
    auto softmax_clone = softmax.clone();
    if (!softmax_clone || softmax_clone->getLayerType() != Layer_Type::SOFTMAX || !softmax.getPopulationParameterDims().empty() || !softmax.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    Batch_Norm_Layer bn1d(4, 1e-5f, 0.1f, exec_target);
    auto bn1d_clone = bn1d.clone();
    if (!bn1d_clone || bn1d.getPopulationParameterDims().size() != 4 || bn1d.getPopulationParameterIsEvolvable() != std::vector<bool>{true, true, false, false})
    {
        return false;
    }
    auto bn1d_init_gamma = bn1d.getPopulationParameterInitializer(0);
    auto bn1d_init_beta = bn1d.getPopulationParameterInitializer(1);
    auto bn1d_init_mean = bn1d.getPopulationParameterInitializer(2);
    auto bn1d_init_var = bn1d.getPopulationParameterInitializer(3);
    if (bn1d_init_gamma(random_engine) != 1.0f || bn1d_init_beta(random_engine) != 0.0f || bn1d_init_mean(random_engine) != 0.0f || bn1d_init_var(random_engine) != 1.0f)
    {
        return false;
    }

    Batch_Norm_2d_Layer bn2d(2, 2, 3, 1e-5f, 0.1f, exec_target);
    auto bn2d_clone = bn2d.clone();
    if (!bn2d_clone || bn2d.getPopulationParameterDims().size() != 4 || bn2d.getPopulationParameterIsEvolvable() != std::vector<bool>{true, true, false, false})
    {
        return false;
    }

    Conv2d_Layer conv(4, 4, 1, 2, 3, 1, 1, exec_target);
    auto conv_clone = conv.clone();
    if (!conv_clone || conv.getPopulationParameterDims().size() != 2 || conv.getPopulationParameterIsEvolvable() != std::vector<bool>{true, true})
    {
        return false;
    }
    if (conv.getPopulationParameterDims()[0] != Shape{ 2, 1, 3, 3 } || conv.getPopulationParameterDims()[1] != Shape{ 1, 2 })
    {
        return false;
    }

    Max_Pool_2d_Layer maxpool(4, 4, 2, 2, 2, 0, exec_target);
    auto maxpool_clone = maxpool.clone();
    if (!maxpool_clone || !maxpool.getPopulationParameterDims().empty() || !maxpool.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    Global_Avg_Pool_2d_Layer gap(4, 4, 2, exec_target);
    auto gap_clone = gap.clone();
    if (!gap_clone || !gap.getPopulationParameterDims().empty() || !gap.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    PPO_Actor_Critic_Layer ppo(2, exec_target);
    ppo.addActorLayer<Linear_Layer>(2, 2, exec_target);
    ppo.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    auto ppo_clone = ppo.clone();
    if (!ppo_clone || ppo.getPopulationParameterDims().size() != 4 || ppo.getPopulationParameterIsEvolvable().size() != 4)
    {
        return false;
    }

    Res_Net_Block_2d_Layer res_block(4, 4, 8, 8, 1, exec_target);
    auto res_clone = res_block.clone();
    if (!res_clone || res_block.getPopulationParameterDims().empty() || res_block.getPopulationParameterIsEvolvable().empty())
    {
        return false;
    }

    return true;
}

bool testPopulationDeepNetwork(Execution_Target exec_target)
{
    constexpr std::size_t pop_size = 6;
    constexpr std::size_t state_dim = 8;
    constexpr std::size_t action_dim = 3;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, 16, exec_target);
    template_net.addLayer<Relu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(16, 8, exec_target);
    template_net.addLayer<Gelu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(8, action_dim, exec_target);
    template_net.addLayer<Softmax_Layer>(false, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target, 888);
    if (pop.getPopulationSize() != pop_size || pop.getStateDimension() != state_dim || pop.getActionSpaceSize() != action_dim)
    {
        return false;
    }

    std::vector<float> input_sample(state_dim);
    for (std::size_t i = 0; i < state_dim; ++i)
    {
        input_sample[i] = 0.1f * static_cast<float>(i + 1);
    }

    for (std::size_t i = 0; i < pop_size; ++i)
    {
        std::size_t act = pop.selectAction(i, input_sample);
        if (act >= action_dim)
        {
            return false;
        }

        Neural_Network ind = pop.getIndividual(i);
        Matrix in_mat(1, state_dim, input_sample, exec_target);
        Matrix out_mat = ind.forward(in_mat);
        if (exec_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }
        const auto& out_data = out_mat.getData();
        std::size_t expected_act = static_cast<std::size_t>(std::distance(out_data.begin(), std::max_element(out_data.begin(), out_data.end())));
        if (act != expected_act)
        {
            return false;
        }
    }

    std::vector<float> batch_input(pop_size * state_dim);
    for (std::size_t i = 0; i < pop_size * state_dim; ++i)
    {
        batch_input[i] = std::sin(static_cast<float>(i) * 0.3f);
    }

    std::vector<std::size_t> batch_actions(pop_size);
    pop.selectBatchActions(batch_input.data(), nullptr, pop_size, batch_actions.data());

    for (std::size_t i = 0; i < pop_size; ++i)
    {
        std::size_t single_act = pop.selectAction(i, batch_input.data() + i * state_dim);
        if (batch_actions[i] != single_act)
        {
            return false;
        }
    }

    std::vector<float> fitness(pop_size);
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        fitness[i] = static_cast<float>(i * 10);
    }

    Neural_Network best_ind = pop.getBestIndividual(fitness.data());
    Matrix in_mat_best(1, state_dim, input_sample, exec_target);
    Matrix best_out = best_ind.forward(in_mat_best);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    const auto& best_data = best_out.getData();
    std::size_t best_expected = static_cast<std::size_t>(std::distance(best_data.begin(), std::max_element(best_data.begin(), best_data.end())));
    if (pop.selectAction(pop_size - 1, input_sample) != best_expected)
    {
        return false;
    }

    std::string ckpt_path = "temp_deep_pop.bin";
    std::uint64_t gen_in = 50;
    if (!pop.saveCheckpoint(ckpt_path, gen_in, fitness.data()))
    {
        return false;
    }

    Population pop_loader(pop_size, template_net, state_dim, action_dim, exec_target, 111);
    std::uint64_t gen_out = 0;
    std::vector<float> fit_out(pop_size);
    if (!pop_loader.loadCheckpoint(ckpt_path, gen_out, fit_out.data()))
    {
        std::remove(ckpt_path.c_str());
        return false;
    }
    std::remove(ckpt_path.c_str());

    if (gen_out != gen_in)
    {
        return false;
    }
    for (std::size_t i = 0; i < pop_size; ++i)
    {
        if (!nearlyEqual(fit_out[i], fitness[i]))
        {
            return false;
        }
        if (pop_loader.selectAction(i, input_sample) != pop.selectAction(i, input_sample))
        {
            return false;
        }
    }

    return true;
}

bool testMatrixConcatAndSplit(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix mat_b(2, 1, { 5.0f, 6.0f }, exec_target);

    Matrix concat_cols_res = mat_a.concatenateCollumns(mat_b);
    bool concat_cols_ok = verifyMatrix(concat_cols_res, { 1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f });

    auto [split_left, split_right] = concat_cols_res.splitCollumns(2);
    bool split_cols_ok = verifyMatrix(split_left, { 1.0f, 2.0f, 3.0f, 4.0f }) &&
        verifyMatrix(split_right, { 5.0f, 6.0f });

    Matrix mat_row_a(1, 2, { 10.0f, 20.0f }, exec_target);
    Matrix mat_row_b(2, 2, { 30.0f, 40.0f, 50.0f, 60.0f }, exec_target);

    Matrix concat_rows_res = mat_row_a.concatenateRows(mat_row_b);
    bool concat_rows_ok = verifyMatrix(concat_rows_res, { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f });

    auto [split_up, split_down] = concat_rows_res.splitRows(1);
    bool split_rows_ok = verifyMatrix(split_up, { 10.0f, 20.0f }) &&
        verifyMatrix(split_down, { 30.0f, 40.0f, 50.0f, 60.0f });

    return concat_cols_ok && split_cols_ok && concat_rows_ok && split_rows_ok;
}

bool testPpoActorCriticForward(Execution_Target exec_target)
{
    constexpr std::uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto& actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Matrix(2, 2, { 1.0f, 0.0f, 0.0f, 1.0f }, exec_target));
    actor_linear.setBiases(Matrix(1, 2, { 0.5f, -0.5f }, exec_target));

    auto& critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Matrix(2, 1, { 1.0f, 2.0f }, exec_target));
    critic_linear.setBiases(Matrix(1, 1, { 1.0f }, exec_target));

    Matrix input_matrix(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Matrix output_matrix = ppo_layer.forward(input_matrix);

    bool output_ok = verifyMatrix(output_matrix, { 1.5f, 1.5f, 6.0f, 3.5f, 3.5f, 12.0f });
    bool actor_sub_ok = verifyMatrix(ppo_layer.getActorOutput(), { 1.5f, 1.5f, 3.5f, 3.5f });
    bool critic_sub_ok = verifyMatrix(ppo_layer.getCriticOutput(), { 6.0f, 12.0f });

    return output_ok && actor_sub_ok && critic_sub_ok;
}

bool testPpoActorCriticBackward(Execution_Target exec_target)
{
    constexpr std::uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto& actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Matrix(2, 2, { 1.0f, 0.0f, 0.0f, 1.0f }, exec_target));
    actor_linear.setBiases(Matrix(1, 2, { 0.0f, 0.0f }, exec_target));

    auto& critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Matrix(2, 1, { 1.0f, 1.0f }, exec_target));
    critic_linear.setBiases(Matrix(1, 1, { 0.0f }, exec_target));

    Matrix input_matrix(1, 2, { 2.0f, 3.0f }, exec_target);
    ppo_layer.forward(input_matrix);

    Matrix output_gradient(1, 3, { 1.0f, 2.0f, 3.0f }, exec_target);
    Matrix input_gradient = ppo_layer.backward(output_gradient);

    bool input_grad_ok = verifyMatrix(input_gradient, { 4.0f, 5.0f });

    auto params = ppo_layer.getParametersAndGradients();
    bool params_count_ok = (params.size() == 4);

    bool actor_weight_grad_ok = verifyMatrix(*params[0].second, { 2.0f, 4.0f, 3.0f, 6.0f });
    bool actor_bias_grad_ok = verifyMatrix(*params[1].second, { 1.0f, 2.0f });
    bool critic_weight_grad_ok = verifyMatrix(*params[2].second, { 6.0f, 9.0f });
    bool critic_bias_grad_ok = verifyMatrix(*params[3].second, { 3.0f });

    return input_grad_ok && params_count_ok && actor_weight_grad_ok &&
        actor_bias_grad_ok && critic_weight_grad_ok && critic_bias_grad_ok;
}

bool testPpoActorCriticSerialization(Execution_Target exec_target)
{
    std::string temp_file = "temp_ppo_layer_serialization.bin";
    constexpr std::uint64_t action_dim = 2;

    PPO_Actor_Critic_Layer ppo_source(action_dim, exec_target);
    auto& actor_linear = ppo_source.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Matrix(2, 2, { 1.5f, -0.5f, 0.5f, 2.0f }, exec_target));
    actor_linear.setBiases(Matrix(1, 2, { 0.1f, -0.2f }, exec_target));

    auto& critic_linear = ppo_source.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Matrix(2, 1, { 0.8f, -1.2f }, exec_target));
    critic_linear.setBiases(Matrix(1, 1, { 0.5f }, exec_target));

    Matrix input_mat(1, 2, { 1.0f, 2.0f }, exec_target);
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

bool testLinearLayerInitializer(Execution_Target exec_target)
{
    Linear_Layer layer(64, 64, exec_target);
    auto init_fn = layer.getPopulationParameterInitializer(0);
    std::mt19937 rng(42);

    constexpr std::size_t sample_count = 1000;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (std::size_t i = 0; i < sample_count; ++i)
    {
        float val = init_fn(rng);
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / static_cast<float>(sample_count);
    float variance = (sum_sq / static_cast<float>(sample_count)) - (mean * mean);

    bool mean_ok = std::abs(mean) < 0.15f;
    bool variance_ok = variance > 0.0005f;

    auto bias_init_fn = layer.getPopulationParameterInitializer(1);
    bool bias_zero = (bias_init_fn(rng) == 0.0f);

    return mean_ok && variance_ok && bias_zero;
}

bool testReluBatchedForward(Execution_Target exec_target)
{
    Relu_Layer relu(exec_target);
    std::vector<float> input_data = { -2.0f, -0.5f, 0.0f, 1.5f, 3.0f };
    Tensor batched_input(Shape{ 1, 1, 5 }, input_data, exec_target);
    std::vector<Tensor> batched_params;

    Tensor output = relu.forward(batched_input, batched_params);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::vector<float> expected = { 0.0f, 0.0f, 0.0f, 1.5f, 3.0f };
    std::vector<float> actual = output.getData();
    if (actual.size() != expected.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        if (!nearlyEqual(actual[i], expected[i], 1e-4f))
        {
            return false;
        }
    }
    return true;
}

bool testGradientAccumulation(Execution_Target exec_target)
{
    Neural_Network net(exec_target);
    net.addLayer<Linear_Layer>(4, 8, exec_target);
    net.addLayer<Relu_Layer>(exec_target);
    net.addLayer<Linear_Layer>(8, 2, exec_target);

    std::vector<float> mb1_in_data = { 1.0f, 0.5f, -0.5f, 2.0f,
                                       -1.0f, 2.0f, 0.0f, 1.0f };
    std::vector<float> mb2_in_data = { 0.5f, -1.0f, 1.5f, 0.0f,
                                       2.0f, 1.0f, -1.0f, -0.5f };
    std::vector<float> full_in_data = mb1_in_data;
    full_in_data.insert(full_in_data.end(), mb2_in_data.begin(), mb2_in_data.end());

    std::vector<float> mb1_target_data = { 1.0f, 0.0f,
                                           0.0f, 1.0f };
    std::vector<float> mb2_target_data = { 0.5f, 0.5f,
                                           1.0f, 0.0f };
    std::vector<float> full_target_data = mb1_target_data;
    full_target_data.insert(full_target_data.end(), mb2_target_data.begin(), mb2_target_data.end());

    Tensor full_in(4, 4, full_in_data, exec_target);
    Tensor full_target(4, 2, full_target_data, exec_target);

    Tensor mb1_in(2, 4, mb1_in_data, exec_target);
    Tensor mb1_target(2, 2, mb1_target_data, exec_target);

    Tensor mb2_in(2, 4, mb2_in_data, exec_target);
    Tensor mb2_target(2, 2, mb2_target_data, exec_target);

    net.setGradientAccumulation(false);
    net.zeroGradients();
    Tensor full_out = net.forward(full_in);
    Tensor full_grad = full_out - full_target;
    net.backward(full_grad);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    auto full_params_grads = net.getParametersAndGradients();
    std::vector<std::vector<float>> full_grads_data;
    for (const auto& pg : full_params_grads)
    {
        full_grads_data.push_back(pg.second->getData());
    }

    net.setGradientAccumulation(true);
    net.zeroGradients();
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    Tensor mb1_out = net.forward(mb1_in);
    Tensor mb1_grad = mb1_out - mb1_target;
    net.backward(mb1_grad);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    Tensor mb2_out = net.forward(mb2_in);
    Tensor mb2_grad = mb2_out - mb2_target;
    net.backward(mb2_grad);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    auto accum_params_grads = net.getParametersAndGradients();
    for (std::size_t i = 0; i < accum_params_grads.size(); ++i)
    {
        std::vector<float> accum_grad = accum_params_grads[i].second->getData();
        const auto& expected_grad = full_grads_data[i];
        if (accum_grad.size() != expected_grad.size())
        {
            return false;
        }
        for (std::size_t j = 0; j < accum_grad.size(); ++j)
        {
            if (!nearlyEqual(accum_grad[j], expected_grad[j], 1e-3f))
            {
                return false;
            }
        }
    }

    net.zeroGradients();
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    auto zeroed_grads = net.getParametersAndGradients();
    for (const auto& zg : zeroed_grads)
    {
        auto data = zg.second->getData();
        for (float v : data)
        {
            if (!nearlyEqual(v, 0.0f, 1e-5f))
            {
                return false;
            }
        }
    }

    net.setGradientAccumulation(false);
    return true;
}

bool testCompositeLayerSerialization(Execution_Target exec_target)
{
    std::string resnet_file = "temp_resnet_block_cfg.bin";
    {
        Res_Net_Block_2d_Layer res_block(4, 4, 8, 8, 1, exec_target);
        std::ofstream out(resnet_file, std::ios::binary);
        if (!out.is_open()) return false;
        res_block.saveConfiguration(out);
    }
    {
        std::ifstream in(resnet_file, std::ios::binary);
        if (!in.is_open()) return false;
        auto loaded_layer = Training_Context::constructLayerFromConfig(in, Layer_Type::RES_NET_BLOCK_2D, exec_target);
        if (!loaded_layer || loaded_layer->getLayerType() != Layer_Type::RES_NET_BLOCK_2D)
        {
            std::remove(resnet_file.c_str());
            return false;
        }
    }
    std::remove(resnet_file.c_str());

    std::string ppo_file = "temp_ppo_block_cfg.bin";
    {
        PPO_Actor_Critic_Layer ppo(2, exec_target);
        ppo.addActorLayer<Linear_Layer>(4, 4, exec_target);
        ppo.addCriticLayer<Linear_Layer>(4, 1, exec_target);
        std::ofstream out(ppo_file, std::ios::binary);
        if (!out.is_open()) return false;
        ppo.saveConfiguration(out);
    }
    {
        std::ifstream in(ppo_file, std::ios::binary);
        if (!in.is_open()) return false;
        auto loaded_layer = Training_Context::constructLayerFromConfig(in, Layer_Type::PPO_ACTOR_CRITIC, exec_target);
        if (!loaded_layer || loaded_layer->getLayerType() != Layer_Type::PPO_ACTOR_CRITIC)
        {
            std::remove(ppo_file.c_str());
            return false;
        }
    }
    std::remove(ppo_file.c_str());

    return true;
}

bool testCrossCompatibilityPopulationToNetwork(Execution_Target exec_target)
{
    constexpr std::size_t pop_size = 4;
    constexpr std::size_t state_dim = 4;
    constexpr std::size_t action_dim = 2;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, 8, exec_target);
    template_net.addLayer<Relu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(8, action_dim, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target, 777);

    std::string export_file = "temp_individual_inference.bin";
    if (!pop.saveIndividualAsInference(0, export_file))
    {
        return false;
    }

    Neural_Network standalone_net(exec_target);
    try
    {
        standalone_net.loadInference(export_file, exec_target);
    }
    catch (const std::exception&)
    {
        std::remove(export_file.c_str());
        return false;
    }
    std::remove(export_file.c_str());

    std::vector<float> sample_input = { 0.5f, -0.2f, 1.0f, -0.8f };
    Tensor in_tensor(1, state_dim, sample_input, exec_target);
    Tensor net_out = standalone_net.forward(in_tensor);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::size_t pop_action = pop.selectAction(0, sample_input);
    const auto& net_data = net_out.getData();
    std::size_t net_action = static_cast<std::size_t>(std::distance(
        net_data.begin(), std::max_element(net_data.begin(), net_data.end())));

    if (pop_action != net_action)
    {
        return false;
    }

    pop.setIndividual(1, standalone_net);
    std::size_t pop_action_ind1 = pop.selectAction(1, sample_input);
    return (pop_action_ind1 == net_action);
}

bool testTensorInterfaceUnification(Execution_Target exec_target)
{
    std::string temp_tensor_file = "temp_tensor_io.bin";
    Tensor original(Shape{ 2, 3 }, { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f }, exec_target);
    {
        std::ofstream out(temp_tensor_file, std::ios::binary);
        if (!out.is_open()) return false;
        original.saveTensor(out);
    }
    Tensor loaded;
    {
        std::ifstream in(temp_tensor_file, std::ios::binary);
        if (!in.is_open()) return false;
        loaded = Tensor::loadTensor(in, exec_target);
    }
    std::remove(temp_tensor_file.c_str());

    if (loaded.getShape() != original.getShape())
    {
        return false;
    }
    if (!verifyMatrix(loaded, original.getData()))
    {
        return false;
    }

    Linear_Layer linear(3, 2, exec_target);
    Tensor input(1, 3, { 1.0f, 0.5f, -1.0f }, exec_target);
    Tensor fwd_out = linear.forward(input);
    Tensor grad_in = linear.backward(Tensor(1, 2, { 0.1f, -0.1f }, exec_target));

    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    return (fwd_out.getColumns() == 2 && grad_in.getColumns() == 3);
}

bool testPipelineCachePersistence()
{
    auto &engine = Execution_Engine::getInstance();
    auto &cache_mgr = engine.getPipelineCacheManager();
    if (cache_mgr.getPipelineCache() == VK_NULL_HANDLE)
    {
        return false;
    }
    cache_mgr.savePipelineCache();
    std::string path = cache_mgr.getCacheFilePath();
    return std::filesystem::exists(path);
}

bool testFp16SupportAndCastingCpu()
{
    Shape shape{2, 3};
    std::vector<float> input_data = {1.0f, -2.5f, 0.125f, 500.0f, -0.05f, 0.0f};
    Tensor fp32_tensor(shape, input_data, Execution_Target::CPU);

    Tensor fp16_tensor = fp32_tensor.toFp16();
    if (fp16_tensor.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }

    Tensor recovered_fp32 = fp16_tensor.toFp32();
    if (recovered_fp32.getDataType() != Data_Type::FLOAT32)
    {
        return false;
    }

    const auto &rec_data = recovered_fp32.getData();
    for (std::size_t i = 0; i < input_data.size(); ++i)
    {
        if (std::abs(rec_data[i] - input_data[i]) > 1e-2f * (std::abs(input_data[i]) + 1.0f))
        {
            return false;
        }
    }
    return true;
}

bool testFp16SupportAndCastingGpu()
{
    auto &engine = Execution_Engine::getInstance();
    const auto &context = engine.getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        std::cout << "[INFO: GPU does not enable/support FP16, skipping GPU kernel test] ";
        return true;
    }

    Shape shape{2, 3};
    std::vector<float> input_data = {1.0f, -2.5f, 0.125f, 42.0f, -100.0f, 0.5f};
    Tensor gpu_fp32(shape, input_data, Execution_Target::VULKAN_GPU);

    Tensor gpu_fp16 = gpu_fp32.toFp16();
    if (gpu_fp16.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }

    Tensor gpu_recovered_fp32 = gpu_fp16.toFp32();
    engine.waitIdle();

    const auto &rec_data = gpu_recovered_fp32.getData();
    for (std::size_t i = 0; i < input_data.size(); ++i)
    {
        if (std::abs(rec_data[i] - input_data[i]) > 1e-2f * (std::abs(input_data[i]) + 1.0f))
        {
            return false;
        }
    }
    return true;
}

bool testLossScalerAndAmp()
{
    Loss_Scaler scaler(1024.0f, 2.0f, 0.5f, 2, true);

    float loss = 0.05f;
    float scaled_loss = scaler.scaleLoss(loss);
    if (std::abs(scaled_loss - 51.2f) > 1e-3f)
    {
        return false;
    }

    Shape shape{2, 2};
    std::vector<float> normal_grad = {0.1f, 0.2f, -0.1f, 0.05f};
    Tensor grad_tensor(shape, normal_grad, Execution_Target::CPU);

    scaler.scaleGradient(grad_tensor);
    if (std::abs(grad_tensor.getData()[0] - 102.4f) > 1e-2f)
    {
        return false;
    }

    scaler.unscaleGradient(grad_tensor);
    if (std::abs(grad_tensor.getData()[0] - 0.1f) > 1e-3f)
    {
        return false;
    }

    if (scaler.hasOverflow(grad_tensor))
    {
        return false;
    }

    std::vector<float> overflow_grad = {0.1f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f};
    Tensor overflow_tensor(shape, overflow_grad, Execution_Target::CPU);
    if (!scaler.hasOverflow(overflow_tensor))
    {
        return false;
    }

    bool accepted = scaler.step(true);
    if (accepted || std::abs(scaler.getScaleFactor() - 512.0f) > 1e-3f)
    {
        return false;
    }

    scaler.step(false);
    scaler.step(false);
    if (std::abs(scaler.getScaleFactor() - 1024.0f) > 1e-3f)
    {
        return false;
    }

    Neural_Network nn(Execution_Target::CPU);
    nn.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
    nn.setOptimizer<Sgd_Optimizer>(0.01f);
    nn.enableMixedPrecision(true);

    if (!nn.isMixedPrecisionEnabled())
    {
        return false;
    }

    Tensor input(1, 4, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Execution_Target::CPU);
    Tensor target(1, 2, std::vector<float>{0.5f, -0.5f}, Execution_Target::CPU);
    nn.trainStep(input, target);

    return true;
}

bool testStaticCommandBuffer()
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.waitIdle();

    Neural_Network nn(Execution_Target::VULKAN_GPU);
    nn.enableStaticGraph(true);
    if (!nn.isStaticGraphEnabled())
    {
        return false;
    }

    nn.addLayer<Linear_Layer>(4, 8, Execution_Target::VULKAN_GPU);
    nn.addLayer<Gelu_Layer>(Execution_Target::VULKAN_GPU);
    nn.addLayer<Linear_Layer>(8, 2, Execution_Target::VULKAN_GPU);

    nn.setOptimizer<Adam_Optimizer>(0.01f);
    nn.setCostFunction<Mse_Cost>();
    nn.setTrainingMode(true);

    // Initial check: static graph not baked yet
    if (engine.getGraphExecutor().isStaticBaked(0) || engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    // Step 1: Batch 1 -> should bake first frame
    std::uint32_t f0 = engine.getContext().getCurrentFrame();
    Tensor in1(2, 4, std::vector<float>{1.0f, 0.5f, -0.5f, 2.0f, -1.0f, 0.0f, 1.5f, -2.0f}, Execution_Target::VULKAN_GPU);
    Tensor tgt1(2, 2, std::vector<float>{0.5f, -0.5f, 1.0f, 0.0f}, Execution_Target::VULKAN_GPU);
    nn.trainStep(in1, tgt1);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(f0))
    {
        return false;
    }

    // Step 2: Batch 2 -> should bake second frame
    std::uint32_t f1 = engine.getContext().getCurrentFrame();
    Tensor in2(2, 4, std::vector<float>{0.2f, -0.3f, 0.8f, 1.1f, -0.5f, 0.4f, -1.2f, 0.7f}, Execution_Target::VULKAN_GPU);
    Tensor tgt2(2, 2, std::vector<float>{0.1f, 0.9f, -0.3f, 0.4f}, Execution_Target::VULKAN_GPU);
    nn.trainStep(in2, tgt2);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(f1))
    {
        return false;
    }

    // Both frames are now baked
    if (!engine.getGraphExecutor().isStaticBaked(0) || !engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    // Step 3: Batch 3 -> should REPLAY baked frame without re-baking
    nn.trainStep(in1, tgt1);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(0) || !engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    // Switch to evaluation mode: should invalidate static graph
    nn.setTrainingMode(false);
    if (engine.getGraphExecutor().isStaticBaked(0) || engine.getGraphExecutor().isStaticBaked(1))
    {
        return false;
    }

    // Inference forward:
    std::uint32_t f_inf = engine.getContext().getCurrentFrame();
    Matrix test_in(2, 4, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, Execution_Target::VULKAN_GPU);
    Matrix pred = nn.forward(test_in);
    engine.executeGraph();
    engine.waitIdle();

    // Now current frame of inference should be baked
    if (!engine.getGraphExecutor().isStaticBaked(f_inf))
    {
        return false;
    }

    // Replay inference on second batch
    Matrix pred2 = nn.forward(test_in);
    engine.executeGraph();
    engine.waitIdle();

    auto pred_data = pred.getData();
    if (pred_data.size() != 4)
    {
        return false;
    }

    // Clean up
    nn.enableStaticGraph(false);
    engine.invalidateStaticGraph();
    engine.waitIdle();

    return true;
}

void runTestSuite(Execution_Target exec_target, const std::string& target_name)
{
    std::cout << "   RUNNING TEST SUITE ON " << target_name << "\n";

    std::cout << "\n[1. Basic Matrix Arithmetics]\n";
    std::cout << "  Matrix Addition (with Broadcast):  " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Subtraction (with Broadcast): " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Multiplication (GEMM):      " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Scalar Multiplication & Division:   " << (testScalarOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Hadamard Multiplication & Division: " << (testHadamardOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Concat & Split (Row/Col):   " << (testMatrixConcatAndSplit(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[2. Transformations & Advanced Operations]\n";
    std::cout << "  Transpose & Matrix Inversion:       " << (testTransposeAndInverse(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Euclidean L2 Normalization:        " << (testNormalize(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Fused Linear Bias Add (MatmulAdd): " << (testMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Batched Tensor GEMM (3D):          " << (testBatchedTensorMatmul(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Batched Tensor GEMM Add (3D):      " << (testBatchedTensorMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[3. Activation Functions]\n";
    std::cout << "  ReLU Forward & Backward:           " << (testRelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  GELU Forward & Backward:           " << (testGelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Softmax Forward & Backward:        " << (testSoftmax(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[4. Cost & Loss Functions]\n";
    std::cout << "  MSE Cost & Gradient:               " << (testMseLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  MAE Cost & Gradient:               " << (testMaeLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BCE Cost & Gradient:               " << (testBceLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  CCE Cost & Gradient:               " << (testCceLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Huber Cost & Gradient:             " << (testHuberLoss(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[5. Neural Network Layers]\n";
    std::cout << "  Linear Layer (Forward & Backward): " << (testLinearLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Conv2D Layer (Forward & Backward): " << (testConv2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Conv2D Native FP16 (Fwd & Bwd):    " << (testConv2dLayerFp16(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Zero-Cast Pipeline (Native FP16):  " << (testNativeFp16ZeroCastPipeline(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  MaxPool2D Layer (with Mask):       " << (testMaxPool2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  GlobalAvgPool2D Layer:             " << (testGlobalAvgPool2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm 1D Layer:                " << (testBatchNormLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm 2D Layer:                " << (testBatchNorm2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  ResNet Block 2D (Id & Proj):       " << (testResNetBlock2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  PPO Layer Forward:                 " << (testPpoActorCriticForward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  PPO Layer Backward & Accumulation: " << (testPpoActorCriticBackward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  PPO Layer Serialization I/O:       " << (testPpoActorCriticSerialization(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[6. Optimizers]\n";
    std::cout << "  SGD Optimizer Step:                " << (testSgdOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Adam Optimizer Step:               " << (testAdamOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[7. Reinforcement Learning & Neuroevolution]\n";
    std::cout << "  DQN Agent (Train Step & Target Sync): " << (testDqnAgent(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Population Suite (GEMM/Evolve/IO): " << (testPopulation(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Population Suite (Native FP16):    " << (testPopulationFp16(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Population Deep Architecture:      " << (testPopulationDeepNetwork(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Layer Population Interface & Clone:" << (testLayerInterfaceContracts(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[8. Serialization & I/O]\n";
    std::cout << "  Matrix Binary I/O:                 " << (testMatrixSerialization(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Model Inference I/O (NNI1):        " << (testModelInferenceSerialization(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[9. Unified Architecture & New Features]\n";
    std::cout << "  Linear Layer Init (He/Xavier):      " << (testLinearLayerInitializer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  ReLU Batched Forward (Population):  " << (testReluBatchedForward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Gradient Accumulation (Micro-batch):" << (testGradientAccumulation(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Composite Layer Deserialization:    " << (testCompositeLayerSerialization(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Pop to NN Inference Compatibility:  " << (testCrossCompatibilityPopulationToNetwork(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Tensor Interface Unification:       " << (testTensorInterfaceUnification(exec_target) ? "PASS" : "FAIL") << "\n\n";
}

int main()
{
    Logger::setFileLogging(true);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);
    runTestSuite(Execution_Target::CPU, "CPU BACKEND");
    runTestSuite(Execution_Target::VULKAN_GPU, "VULKAN GPU BACKEND");

    std::cout << "  Learning Rate Schedulers Suite:    " << (testLearningRateSchedulers() ? "PASS" : "FAIL") << "\n";
    std::cout << "  GPU Vector Lifecycle & Resizing:   " << (testGpuVectorLifecycle() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Sub-Allocator & Garbage Collector: " << (testVulkanSubAllocatorAndGarbageCollection() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Operator Fusion & Graph Dispatch:  " << (testOperatorFusionAndGraphExecution() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Batched GEMM + ReLU Fusion:        " << (testBatchedTensorMatmulFusion() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Async Data Pipeline Double-Buffer: " << (testAsyncDataPipeline() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Replay Buffer Capacity & Sampling: " << (testReplayBuffer() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Pipeline Cache Persistence:        " << (testPipelineCachePersistence() ? "PASS" : "FAIL") << "\n";
    std::cout << "  FP16 CPU Support & Precision Cast: " << (testFp16SupportAndCastingCpu() ? "PASS" : "FAIL") << "\n";
    std::cout << "  FP16 GPU Support & Precision Cast: " << (testFp16SupportAndCastingGpu() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Loss Scaler & Mixed Precision AMP: " << (testLossScalerAndAmp() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Static Command Buffer (Replay):    " << (testStaticCommandBuffer() ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n";

    return 0;
}