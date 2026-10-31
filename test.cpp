#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
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
#include "tokenizer/bpe_tokenizer.h"
#include "helper/llm.h"
#include "llm/binary_dataset.h"
#include "cost_function/dice_bce_cost.h"
#include "cost_function/dice_cost.h"
#include "layer/instance_norm2d_layer.h"
#include "layer/leaky_relu_layer.h"
#include "layer/upsample2d_layer.h"
#include "layer/nn_unet_layer.h"
#include "layer/u_net_layer.h"
#include "ops/loss/dice_bce.h"
#include "ops/normalization/instancenorm.h"
#include "ops/activation/activation.h"
#include "ops/vision/pooling.h"
#include "chexlocalize/chest_seg_dataset.h"
#include <unordered_set>

bool nearlyEqual(float a, float b, float eps = 1e-3f)
{
    return std::fabs(a - b) < eps;
}

bool verifyMatrix(const Tensor& mat, const std::vector<float>& expected_data, float eps = 1e-3f)
{
    if (mat.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty() || !engine.getContext().getTransferTasks().empty())
        {
            engine.executeGraph();
        }
    }

    std::vector<float> actual_data = mat.getData();
    if (actual_data.size() != expected_data.size())
    {
        return false;
    }
    for (size_t i = 0; i < actual_data.size(); ++i)
    {
        if (!nearlyEqual(actual_data[i], expected_data[i], eps))
        {
            return false;
        }
    }
    return true;
}

bool testEmbeddingLayer(Execution_Target exec_target)
{
    constexpr size_t vocab_size = 8;
    constexpr size_t embedding_dim = 4;

    Embedding_Layer layer(vocab_size, embedding_dim, exec_target);

    std::vector<int32_t> tokens_1d = { 1, 3, 5 };
    Tensor out_1d = layer.forward(tokens_1d);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (out_1d.getShape() != Shape{ 3, embedding_dim })
    {
        return false;
    }
    for (float val : out_1d.getData())
    {
        if (std::isnan(val) || std::isinf(val))
        {
            return false;
        }
    }

    std::vector<std::vector<int32_t>> tokens_2d = {
        { 0, 2 },
        { 4, 7 }
    };
    Tensor out_2d = layer.forward(tokens_2d);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (out_2d.getShape() != Shape{ 2, 2, embedding_dim })
    {
        return false;
    }
    for (float val : out_2d.getData())
    {
        if (std::isnan(val) || std::isinf(val))
        {
            return false;
        }
    }

    Tensor input_tensor(Shape{ 2, 2 }, std::vector<float>{ 1.0f, 3.0f, 5.0f, 6.0f }, exec_target);
    Tensor out_tensor = layer.forward(input_tensor);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (out_tensor.getShape() != Shape{ 2, 2, embedding_dim })
    {
        return false;
    }

    Tensor grad_out(Shape{ 2, 2, embedding_dim }, std::vector<float>(2 * 2 * embedding_dim, 0.5f), exec_target);
    Tensor grad_in = layer.backward(grad_out);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (grad_in.getTotalElements() != 0)
    {
        return false;
    }

    auto params_and_grads = layer.getParametersAndGradients();
    if (params_and_grads.size() != 1)
    {
        return false;
    }
    Tensor* weight_grad = params_and_grads[0].second;
    if (!weight_grad || weight_grad->getTotalElements() != vocab_size * embedding_dim)
    {
        return false;
    }
    bool has_nonzero_grad = false;
    for (float val : weight_grad->getData())
    {
        if (std::abs(val) > 1e-6f)
        {
            has_nonzero_grad = true;
            break;
        }
    }
    if (!has_nonzero_grad)
    {
        return false;
    }

    layer.resetGradients();
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    for (float val : weight_grad->getData())
    {
        if (std::abs(val) > 1e-6f)
        {
            return false;
        }
    }

    auto layer_clone = layer.clone();
    if (!layer_clone || layer_clone->getLayerType() != Layer_Type::EMBEDDING)
    {
        return false;
    }

    std::string temp_file = "temp_embedding_io.bin";
    {
        std::ofstream out(temp_file, std::ios::binary);
        if (!out.is_open())
        {
            return false;
        }
        layer.saveInference(out);
    }
    Embedding_Layer loaded_layer(vocab_size, embedding_dim, exec_target);
    {
        std::ifstream in(temp_file, std::ios::binary);
        if (!in.is_open())
        {
            return false;
        }
        loaded_layer.loadInference(in);
    }
    std::filesystem::remove(temp_file);

    Tensor orig_eval = layer.forward(tokens_1d);
    Tensor loaded_eval = loaded_layer.forward(tokens_1d);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    return verifyMatrix(loaded_eval, orig_eval.getData());
}

bool testFusedCrossEntropy(Execution_Target exec_target)
{
    Fused_Cross_Entropy cce(-1);
    if (!cce.isFused() || !cce.supportsIntegerTargets() || cce.getType() != Cost_Type::FUSED_CCE)
    {
        return false;
    }

    constexpr size_t batch_size = 2;
    constexpr size_t seq_len = 3;
    constexpr size_t vocab_size = 4;
    constexpr size_t total_tokens = batch_size * seq_len;

    std::vector<float> logits_data = {
        2.0f,  1.0f,  0.1f, -1.0f,
       -0.5f,  3.0f,  1.2f,  0.0f,
        0.0f,  0.0f,  2.5f,  0.1f,

        1.5f,  0.2f, -0.1f,  2.0f,
        0.1f,  0.5f,  0.2f,  0.3f,
       -1.0f,  2.0f, -0.5f,  0.1f
    };

    Tensor logits(Shape{ batch_size, seq_len, vocab_size }, logits_data, exec_target);
    std::vector<int32_t> targets = { 0, 1, 2, 3, 1, 1 };

    Tensor grad_indices;
    float loss_indices = cce.computeLossAndGradient(logits, targets, grad_indices);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    if (std::isnan(loss_indices) || std::isinf(loss_indices) || loss_indices <= 0.0f)
    {
        return false;
    }
    if (grad_indices.getShape() != logits.getShape())
    {
        return false;
    }

    std::vector<float> grad_data = grad_indices.getData();
    for (size_t t = 0; t < total_tokens; ++t)
    {
        float row_sum = 0.0f;
        for (size_t v = 0; v < vocab_size; ++v)
        {
            row_sum += grad_data[t * vocab_size + v];
        }
        if (std::abs(row_sum) > 1e-4f)
        {
            return false;
        }
    }

    std::vector<float> target_tensor_data(total_tokens);
    for (size_t i = 0; i < total_tokens; ++i)
    {
        target_tensor_data[i] = static_cast<float>(targets[i]);
    }
    Tensor target_tensor(Shape{ batch_size, seq_len }, target_tensor_data, exec_target);

    Tensor grad_tensor;
    float loss_tensor = cce.computeLossAndGradient(logits, target_tensor, grad_tensor);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    if (!nearlyEqual(loss_indices, loss_tensor, 1e-3f))
    {
        return false;
    }
    if (!verifyMatrix(grad_tensor, grad_data, 1e-3f))
    {
        return false;
    }

    Fused_Cross_Entropy cce_ignore(2);
    if (cce_ignore.getIgnoreIndex() != 2)
    {
        return false;
    }
    Tensor grad_ignored;
    float loss_ignored = cce_ignore.computeLossAndGradient(logits, targets, grad_ignored);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (std::isnan(loss_ignored) || std::isinf(loss_ignored) || loss_ignored <= 0.0f)
    {
        return false;
    }

    std::vector<int32_t> all_ignored = { 2, 2, 2, 2, 2, 2 };
    Tensor grad_all_ignored;
    float loss_all_ignored = cce_ignore.computeLossAndGradient(logits, all_ignored, grad_all_ignored);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (loss_all_ignored != 0.0f)
    {
        return false;
    }
    for (float v : grad_all_ignored.getData())
    {
        if (v != 0.0f)
        {
            return false;
        }
    }

    return true;
}

bool testMatrixAddition(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mat_b(2, 2, { 5.0f, 6.0f, 7.0f, 8.0f }, exec_target);
    Tensor res = mat_a + mat_b;
    bool standard_ok = verifyMatrix(res, { 6.0f, 8.0f, 10.0f, 12.0f });

    Tensor mat_broadcast(1, 2, { 10.0f, 20.0f }, exec_target);
    Tensor broadcast_res = mat_a + mat_broadcast;
    bool broadcast_ok = verifyMatrix(broadcast_res, { 11.0f, 22.0f, 13.0f, 24.0f });

    return standard_ok && broadcast_ok;
}

bool testMatrixSubtraction(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 5.0f, 6.0f, 7.0f, 8.0f }, exec_target);
    Tensor mat_b(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor res = mat_a - mat_b;
    bool standard_ok = verifyMatrix(res, { 4.0f, 4.0f, 4.0f, 4.0f });

    Tensor mat_broadcast(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor broadcast_res = mat_a - mat_broadcast;
    bool broadcast_ok = verifyMatrix(broadcast_res, { 4.0f, 4.0f, 6.0f, 6.0f });

    return standard_ok && broadcast_ok;
}

bool testMatrixMultiplication(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mat_b(2, 2, { 2.0f, 0.0f, 1.0f, 2.0f }, exec_target);
    Tensor res = mat_a * mat_b;
    return verifyMatrix(res, { 4.0f, 4.0f, 10.0f, 8.0f });
}

bool testScalarOperations(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mul_res = mat_a * 2.0f;
    Tensor div_res = mat_a / 2.0f;

    bool mul_ok = verifyMatrix(mul_res, { 2.0f, 4.0f, 6.0f, 8.0f });
    bool div_ok = verifyMatrix(div_res, { 0.5f, 1.0f, 1.5f, 2.0f });

    return mul_ok && div_ok;
}

bool testHadamardOperations(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mat_b(2, 2, { 2.0f, 0.0f, 1.0f, 2.0f }, exec_target);

    Tensor mul_res = mat_a.hadamardMul(mat_b);
    Tensor div_res = mat_a.hadamardDiv(mat_a);

    bool mul_ok = verifyMatrix(mul_res, { 2.0f, 0.0f, 3.0f, 8.0f });
    bool div_ok = verifyMatrix(div_res, { 1.0f, 1.0f, 1.0f, 1.0f });

    Tensor mat_broadcast(1, 2, { 2.0f, 4.0f }, exec_target);
    Tensor mul_bcast = mat_a.hadamardMul(mat_broadcast);
    Tensor div_bcast = mat_a.hadamardDiv(mat_broadcast);

    bool mul_bcast_ok = verifyMatrix(mul_bcast, { 2.0f, 8.0f, 6.0f, 16.0f });
    bool div_bcast_ok = verifyMatrix(div_bcast, { 0.5f, 0.5f, 1.5f, 1.0f });

    return mul_ok && div_ok && mul_bcast_ok && div_bcast_ok;
}

bool testTransposeAndInverse(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor trans_res = mat_a.transpose();
    bool trans_ok = verifyMatrix(trans_res, { 1.0f, 3.0f, 2.0f, 4.0f });

    Tensor mat_inv_target(2, 2, { 4.0f, 7.0f, 2.0f, 6.0f }, exec_target);
    Tensor inv_res = mat_inv_target.inverse();
    bool inv_ok = verifyMatrix(inv_res, { 0.6f, -0.7f, -0.2f, 0.4f });

    return trans_ok && inv_ok;
}

bool testNormalize(Execution_Target exec_target)
{
    Tensor vec_mat(1, 2, { 3.0f, 4.0f }, exec_target);
    Tensor norm_res = vec_mat.normalize();
    return verifyMatrix(norm_res, { 0.6f, 0.8f });
}

bool testMatmulAdd(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mat_b(2, 2, { 2.0f, 0.0f, 1.0f, 2.0f }, exec_target);
    Tensor mat_bias(1, 2, { 5.0f, 6.0f }, exec_target);

    Tensor res = mat_a.matmulAdd(mat_b, mat_bias);
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
    Tensor act_mat(2, 2, { -1.0f, 2.0f, 0.0f, -3.0f }, exec_target);
    Tensor relu_res = act_mat.relu();
    bool fwd_ok = verifyMatrix(relu_res, { 0.0f, 2.0f, 0.0f, 0.0f });

    Tensor grad_out(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor grad_in = act_mat.reluBackward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.0f, 2.0f, 0.0f, 0.0f });

    return fwd_ok && bwd_ok;
}

bool testGelu(Execution_Target exec_target)
{
    Tensor input_mat(1, 4, { 0.0f, 1.0f, -1.0f, 2.0f }, exec_target);
    Tensor gelu_res = input_mat.gelu();
    bool fwd_ok = verifyMatrix(gelu_res, { 0.0f, 0.8412316f, -0.1587684f, 1.9546059f });

    Tensor grad_in = input_mat.geluBackward(Tensor(1, 4, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target));
    bool bwd_ok = verifyMatrix(grad_in, { 0.5f, 1.0829548f, -0.0829548f, 1.085999f });

    return fwd_ok && bwd_ok;
}

bool testSoftmax(Execution_Target exec_target)
{
    Tensor softmax_in(1, 2, { 0.0f, 0.0f }, exec_target);
    Tensor softmax_res = softmax_in.softmax();
    bool fwd_ok = verifyMatrix(softmax_res, { 0.5f, 0.5f });

    Tensor grad_out(1, 2, { 1.0f, -1.0f }, exec_target);
    Tensor grad_in = softmax_res.softmaxBackward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.5f, -0.5f });

    return fwd_ok && bwd_ok;
}

bool testMseLoss(Execution_Target exec_target)
{
    Mse_Cost cost_func(exec_target);
    Tensor pred(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor target(1, 2, { 2.0f, 4.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 2.5f);

    Tensor grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -1.0f, -2.0f });

    return loss_ok && grad_ok;
}

bool testMaeLoss(Execution_Target exec_target)
{
    Mae_Cost cost_func(exec_target);
    Tensor pred(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor target(1, 2, { 2.0f, 4.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 1.5f);

    Tensor grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.5f, -0.5f });

    return loss_ok && grad_ok;
}

bool testBceLoss(Execution_Target exec_target)
{
    Bce_Cost cost_func(1e-7f, exec_target);
    Tensor pred(1, 2, { 0.8f, 0.2f }, exec_target);
    Tensor target(1, 2, { 1.0f, 0.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 0.22314355f);

    Tensor grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.625f, 0.625f });

    return loss_ok && grad_ok;
}

bool testCceLoss(Execution_Target exec_target)
{
    Cce_Cost cost_func(1e-7f, exec_target);
    Tensor pred(1, 3, { 0.7f, 0.2f, 0.1f }, exec_target);
    Tensor target(1, 3, { 1.0f, 0.0f, 0.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 0.356675f);

    Tensor grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.3f, 0.2f, 0.1f });

    return loss_ok && grad_ok;
}

bool testHuberLoss(Execution_Target exec_target)
{
    Huber_Cost cost_func(1.0f, exec_target);
    Tensor pred(1, 3, { 1.0f, 2.0f, 5.0f }, exec_target);
    Tensor target(1, 3, { 1.5f, 4.0f, 2.0f }, exec_target);

    float loss_val = cost_func.computeLoss(pred, target);
    bool loss_ok = nearlyEqual(loss_val, 1.375f);

    Tensor grad_matrix = cost_func.computeGradient(pred, target);
    bool grad_ok = verifyMatrix(grad_matrix, { -0.166667f, -0.333333f, 0.333333f });

    return loss_ok && grad_ok;
}

bool testLinearLayer(Execution_Target exec_target)
{
    Linear_Layer layer(2, 3, exec_target);
    layer.setWeights(Tensor(2, 3, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, exec_target));
    layer.setBiases(Tensor(1, 3, { 0.1f, 0.2f, 0.3f }, exec_target));

    Tensor input_x(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor output_y = layer.forward(input_x);
    bool fwd_ok = verifyMatrix(output_y, { 9.1f, 12.2f, 15.3f }, 0.02f);
    output_y.print();

    Tensor grad_out(1, 3, { 1.0f, 1.0f, 1.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    bool bwd_input_ok = verifyMatrix(grad_in, { 6.0f, 15.0f }, 0.02f);
    grad_in.print();

    bool bwd_w_ok = verifyMatrix(layer.getWeightsGradient(), { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f }, 0.02f);
    layer.getWeightsGradient().print();
    bool bwd_b_ok = verifyMatrix(layer.getBiasesGradient(), { 1.0f, 1.0f, 1.0f }, 0.02f);
    layer.getBiasesGradient().print();
    return fwd_ok && bwd_input_ok && bwd_w_ok && bwd_b_ok;
}

bool testConv2dLayer(Execution_Target exec_target)
{
    Conv2d_Layer layer(3, 3, 1, 1, 2, 1, 0, exec_target);
    auto params = layer.getParametersAndGradients();
    params[0].first->uploadData({ 1.0f, 0.0f, 0.0f, 1.0f });
    params[1].first->uploadData({ 0.0f });

    Tensor input_mat(1, 9, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 6.0f, 8.0f, 12.0f, 14.0f });

    Tensor grad_out(1, 4, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);

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

    Tensor input_mat(1, 9, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 6.0f, 8.0f, 12.0f, 14.0f }, 1e-2f);

    Tensor grad_out(1, 4, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);

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
    for (size_t i = 0; i < 16; ++i) in_data[i] = static_cast<float>(i + 1) * 0.1f;
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

    size_t state_dim = 16;
    size_t hidden_dim = 32;
    size_t action_dim = 4;
    size_t pop_size = 8;

    Neural_Network template_net(exec_target);
    template_net.addLayer<Linear_Layer>(state_dim, hidden_dim, exec_target);
    template_net.addLayer<Gelu_Layer>(exec_target);
    template_net.addLayer<Linear_Layer>(hidden_dim, action_dim, exec_target);

    Population pop(pop_size, template_net, state_dim, action_dim, exec_target);
    pop.enableMixedPrecision(true);

    std::vector<float> states(pop_size * state_dim, 0.5f);
    std::vector<size_t> actions(pop_size, 0);

    pop.selectBatchActions(states.data(), nullptr, pop_size, actions.data());

    for (size_t a : actions)
    {
        if (a >= action_dim) return false;
    }

    return true;
}

bool testMaxPool2dLayer(Execution_Target exec_target)
{
    Max_Pool_2d_Layer layer(4, 4, 1, 2, 2, 0, exec_target);
    Tensor input_mat(1, 16, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f }, exec_target);

    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 6.0f, 8.0f, 14.0f, 16.0f });

    Tensor grad_out(1, 4, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 2.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 3.0f, 0.0f, 4.0f });

    return fwd_ok && bwd_ok;
}

bool testGlobalAvgPool2dLayer(Execution_Target exec_target)
{
    Global_Avg_Pool_2d_Layer layer(2, 2, 2, exec_target);
    Tensor input_mat(1, 8, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, exec_target);

    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 4.0f, 5.0f });

    Tensor grad_out(1, 2, { 4.0f, 8.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

    return fwd_ok && bwd_ok;
}

bool testBatchNormLayer(Execution_Target exec_target)
{
    Batch_Norm_Layer layer(1, 1e-5f, 0.1f, exec_target);
    layer.setTrainingMode(true);

    Tensor input_mat(3, 1, { 1.0f, 2.0f, 3.0f }, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { -1.2247f, 0.0f, 1.2247f });

    Tensor grad_out(3, 1, { 1.0f, 2.0f, 1.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { -0.4082f, 0.8165f, -0.4082f });

    return fwd_ok && bwd_ok;
}

bool testBatchNorm2dLayer(Execution_Target exec_target)
{
    Batch_Norm_2d_Layer layer(1, 2, 1, 1e-5f, 0.1f, exec_target);
    layer.setTrainingMode(true);

    Tensor input_mat(2, 2, { 1.0f, 3.0f, 5.0f, 7.0f }, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { -1.34164f, -0.44721f, 0.44721f, 1.34164f });

    Tensor grad_out(2, 2, { 1.0f, 0.0f, 0.0f, 0.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    bool bwd_ok = verifyMatrix(grad_in, { 0.13416f, -0.17889f, -0.04472f, 0.08944f });

    return fwd_ok && bwd_ok;
}

bool testRmsNormLayer(Execution_Target exec_target)
{
    RMSNorm_Layer layer(4, 1e-5f, exec_target);

    Tensor input_mat(2, 4, { 1.0f, 2.0f, 3.0f, 4.0f,
                             2.0f, 2.0f, 2.0f, 2.0f }, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    bool fwd_ok = verifyMatrix(out_mat, { 0.36515f, 0.73030f, 1.09545f, 1.46059f,
                                          1.0f,     1.0f,     1.0f,     1.0f });

    Tensor grad_out(2, 4, { 1.0f, 1.0f, 1.0f, 1.0f,
                            0.5f, 0.5f, 0.5f, 0.5f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    if (grad_in.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    bool bwd_ok = (grad_in.getRows() == 2 && grad_in.getColumns() == 4);

    bool fp16_ok = true;
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        RMSNorm_Layer layer_fp16(4, 1e-5f, exec_target);
        layer_fp16.setMixedPrecision(true);
        Tensor out_fp16 = layer_fp16.forward(input_mat);
        fp16_ok = verifyMatrix(out_fp16, { 0.36515f, 0.73030f, 1.09545f, 1.46059f,
                                           1.0f,     1.0f,     1.0f,     1.0f }, 2e-3f);
    }

    return fwd_ok && bwd_ok && fp16_ok;
}

bool testRoPE(Execution_Target exec_target)
{
    Tensor q(Shape{1, 1, 2, 4}, { 1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f }, exec_target);
    Tensor q_rot(exec_target);
    Tensor q_back(exec_target);

    q.applyRoPE(q_rot, 2, 4, 1, 10000.0f);
    if (q_rot.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::vector<float> rot_data = q_rot.getData();
    bool pos0_ok = nearlyEqual(rot_data[0], 1.0f, 1e-3f) &&
                   nearlyEqual(rot_data[1], 2.0f, 1e-3f) &&
                   nearlyEqual(rot_data[2], 3.0f, 1e-3f) &&
                   nearlyEqual(rot_data[3], 4.0f, 1e-3f);

    q_rot.applyRoPE(q_back, 2, 4, -1, 10000.0f);
    bool rev_ok = verifyMatrix(q_back, { 1.0f, 2.0f, 3.0f, 4.0f,
                                         5.0f, 6.0f, 7.0f, 8.0f });

    bool fp16_ok = true;
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Tensor q_fp16(exec_target);
        q.to(Data_Type::FLOAT16, q_fp16);
        Tensor q_rot_fp16(exec_target);
        Tensor q_back_fp16(exec_target);
        q_fp16.applyRoPE(q_rot_fp16, 2, 4, 1, 10000.0f);
        q_rot_fp16.applyRoPE(q_back_fp16, 2, 4, -1, 10000.0f);
        fp16_ok = verifyMatrix(q_back_fp16, { 1.0f, 2.0f, 3.0f, 4.0f,
                                              5.0f, 6.0f, 7.0f, 8.0f }, 2e-3f);
    }

    return pos0_ok && rev_ok && fp16_ok;
}

bool testSwiGluLayer(Execution_Target exec_target)
{
    SwiGLU_Layer layer(exec_target);

    Tensor input_mat(2, 4, { 0.0f, 1.0f, 2.0f, 3.0f,
                            -1.0f, 2.0f, 4.0f, 5.0f }, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    bool fwd_ok = verifyMatrix(out_mat, { 0.0f, 2.19318f, -1.07577f, 8.80797f }, 5e-3f);
    out_mat.print();

    Tensor grad_out(2, 2, { 1.0f, 1.0f, 1.0f, 1.0f }, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    if (grad_in.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    bool bwd_ok = (grad_in.getRows() == 2 && grad_in.getColumns() == 4);

    bool fp16_ok = true;
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        SwiGLU_Layer layer_fp16(exec_target);
        layer_fp16.setMixedPrecision(true);
        Tensor out_fp16 = layer_fp16.forward(input_mat);
        Execution_Engine::getInstance().executeGraph();
        fp16_ok = verifyMatrix(out_fp16, { 0.0f, 2.19318f, -1.07577f, 8.80797f }, 5e-3f);
        out_fp16.print();
    }

    return fwd_ok && bwd_ok && fp16_ok;
}

bool testFlashAttention(Execution_Target exec_target)
{
    std::vector<float> q_data(256, 0.1f);
    std::vector<float> k_data(256, 0.1f);
    std::vector<float> v_data(256, 0.2f);
    std::vector<float> do_data(256, 0.05f);
    for (size_t i = 0; i < 256; ++i)
    {
        q_data[i] = 0.05f * static_cast<float>((i % 7) + 1);
        k_data[i] = 0.05f * static_cast<float>((i % 5) + 1);
        v_data[i] = 0.1f * static_cast<float>((i % 11) + 1);
        do_data[i] = 0.02f * static_cast<float>((i % 9) + 1);
    }

    Tensor q_cpu(Shape{1, 1, 16, 16}, q_data, Execution_Target::CPU);
    Tensor k_cpu(Shape{1, 1, 16, 16}, k_data, Execution_Target::CPU);
    Tensor v_cpu(Shape{1, 1, 16, 16}, v_data, Execution_Target::CPU);
    Tensor do_cpu(Shape{1, 1, 16, 16}, do_data, Execution_Target::CPU);

    Tensor out_ref_causal(Execution_Target::CPU);
    Tensor out_ref_noncausal(Execution_Target::CPU);

    Tensor l_ref(Execution_Target::CPU);
    q_cpu.flashAttentionForward(k_cpu, v_cpu, out_ref_causal, 1, 16, 16, true, 0.0f, &l_ref);
    q_cpu.flashAttentionForward(k_cpu, v_cpu, out_ref_noncausal, 1, 16, 16, false);

    Tensor dq_ref(Execution_Target::CPU);
    Tensor dk_ref(Execution_Target::CPU);
    Tensor dv_ref(Execution_Target::CPU);
    q_cpu.flashAttentionBackward(k_cpu, v_cpu, out_ref_causal, do_cpu, dq_ref, dk_ref, dv_ref, 1, 16, 16, true, 0.0f, &l_ref);

    if (exec_target == Execution_Target::CPU)
    {
        return !out_ref_causal.getData().empty() && !out_ref_noncausal.getData().empty()
            && !dq_ref.getData().empty() && !dk_ref.getData().empty() && !dv_ref.getData().empty();
    }

    Tensor q_gpu(Shape{1, 1, 16, 16}, q_data, Execution_Target::VULKAN_GPU);
    Tensor k_gpu(Shape{1, 1, 16, 16}, k_data, Execution_Target::VULKAN_GPU);
    Tensor v_gpu(Shape{1, 1, 16, 16}, v_data, Execution_Target::VULKAN_GPU);
    Tensor do_gpu(Shape{1, 1, 16, 16}, do_data, Execution_Target::VULKAN_GPU);

    Tensor q_gpu_fp16(Execution_Target::VULKAN_GPU);
    Tensor k_gpu_fp16(Execution_Target::VULKAN_GPU);
    Tensor v_gpu_fp16(Execution_Target::VULKAN_GPU);
    Tensor do_gpu_fp16(Execution_Target::VULKAN_GPU);
    q_gpu.to(Data_Type::FLOAT16, q_gpu_fp16);
    k_gpu.to(Data_Type::FLOAT16, k_gpu_fp16);
    v_gpu.to(Data_Type::FLOAT16, v_gpu_fp16);
    do_gpu.to(Data_Type::FLOAT16, do_gpu_fp16);

    Tensor out_gpu_causal(Execution_Target::VULKAN_GPU);
    Tensor out_gpu_noncausal(Execution_Target::VULKAN_GPU);
    Tensor l_gpu(Execution_Target::VULKAN_GPU);

    q_gpu_fp16.flashAttentionForward(k_gpu_fp16, v_gpu_fp16, out_gpu_causal, 1, 16, 16, true, 0.0f, &l_gpu);
    q_gpu_fp16.flashAttentionForward(k_gpu_fp16, v_gpu_fp16, out_gpu_noncausal, 1, 16, 16, false);

    Tensor dq_gpu(Execution_Target::VULKAN_GPU);
    Tensor dk_gpu(Execution_Target::VULKAN_GPU);
    Tensor dv_gpu(Execution_Target::VULKAN_GPU);
    q_gpu_fp16.flashAttentionBackward(k_gpu_fp16, v_gpu_fp16, out_gpu_causal, do_gpu_fp16,
                                      dq_gpu, dk_gpu, dv_gpu, 1, 16, 16, true, 0.0f, &l_gpu);

    bool causal_ok = verifyMatrix(out_gpu_causal, out_ref_causal.getData(), 2e-3f);
    bool noncausal_ok = verifyMatrix(out_gpu_noncausal, out_ref_noncausal.getData(), 2e-3f);
    bool bwd_dq_ok = verifyMatrix(dq_gpu, dq_ref.getData(), 5e-3f);
    bool bwd_dk_ok = verifyMatrix(dk_gpu, dk_ref.getData(), 5e-3f);
    bool bwd_dv_ok = verifyMatrix(dv_gpu, dv_ref.getData(), 5e-3f);

    if (!causal_ok || !noncausal_ok || !bwd_dq_ok || !bwd_dk_ok || !bwd_dv_ok)
    {
        std::cout << " [FlashAttn DBG: causal=" << causal_ok << " noncausal=" << noncausal_ok 
                  << " dq=" << bwd_dq_ok << " dk=" << bwd_dk_ok << " dv=" << bwd_dv_ok << "] ";
    }

    return causal_ok && noncausal_ok && bwd_dq_ok && bwd_dk_ok && bwd_dv_ok;
}

bool testBpeTokenizer()
{
    Bpe_Tokenizer tokenizer;
    if (!tokenizer.load("tokenizer/tokenizer.json"))
    {
        std::cerr << "[FAIL: could not load tokenizer/tokenizer.json]\n";
        return false;
    }

    if (tokenizer.getVocabSize() != 32768)
    {
        std::cerr << "[FAIL: vocab size expected 32768, got " << tokenizer.getVocabSize() << "]\n";
        return false;
    }

    // Test 1: Tiếng Việt UTF-8
    {
        std::string text = "Xin chào thế giới!";
        std::vector<int32_t> expected_ids = {1, 2268, 1030, 1252, 29607, 29477, 1074, 30390, 5988, 30793, 29478, 29576};
        auto ids = tokenizer.encode(text, true);
        if (ids != expected_ids)
        {
            std::cout << "\n[DEBUG: Test 1 encode failed. Expected size " << expected_ids.size() << ", got " << ids.size() << "]\n";
            std::cout << "Got ids: ";
            for (auto id : ids) std::cout << id << " ";
            std::cout << "\n";
            return false;
        }
        std::string decoded = tokenizer.decode(ids, true);
        if (decoded != text)
        {
            std::cout << "\n[DEBUG: Test 1 decode failed. Expected '" << text << "', got '" << decoded << "']\n";
            return false;
        }
    }

    // Test 2: Tiếng Anh + số + toán tử
    {
        std::string text = "Hello world! 123 + 456 = 579.";
        std::vector<int32_t> expected_ids = {1, 23325, 2294, 29576, 29473, 29508, 29518, 29538, 1416, 29473, 29549, 29550, 29552, 1095, 29473, 29550, 29555, 29542, 29491};
        auto ids = tokenizer.encode(text, true);
        if (ids != expected_ids)
        {
            std::cout << "\n[DEBUG: Test 2 encode failed]\n";
            return false;
        }
        std::string decoded = tokenizer.decode(ids, true);
        if (decoded != text)
        {
            std::cout << "\n[DEBUG: Test 2 decode failed]\n";
            return false;
        }
    }

    // Test 3: Khoảng trắng liên tiếp, xuống dòng, tab
    {
        std::string text = "  nhiều   khoảng   trắng  \n dòng mới \t tab";
        std::vector<int32_t> expected_ids = {1, 29473, 1075, 6133, 30907, 29486, 1027, 1214, 2892, 30276, 1585, 1027, 1235, 31197, 1585, 1027, 781, 1049, 29666, 1585, 1058, 30793, 29478, 29473, 780, 8451};
        auto ids = tokenizer.encode(text, true);
        if (ids != expected_ids)
        {
            std::cout << "\n[DEBUG: Test 3 encode failed]\n";
            return false;
        }
        std::string decoded = tokenizer.decode(ids, true);
        // Lưu ý: Hugging Face Metaspace Decoder cấu hình Strip 1 space ở đầu chuỗi (do prepend_scheme = first)
        std::string expected_decoded = " nhiều   khoảng   trắng  \n dòng mới \t tab";
        if (decoded != expected_decoded)
        {
            std::cout << "\n[DEBUG: Test 3 decode failed: expected '" << expected_decoded << "', got '" << decoded << "']\n";
            return false;
        }
    }

    // Test 4: Byte Fallback với ký tự ngoài từ điển (𠮷 UTF-8: F0 A0 AE B7)
    {
        std::string text = "Fallback: \x01\x02\x7f 你好 𠮷";
        std::vector<int32_t> expected_ids = {1, 12936, 2203, 29515, 29473, 30302, 31319, 31750, 29473, 30151, 30298, 29473, 1011, 931, 945, 954};
        auto ids = tokenizer.encode(text, true);
        if (ids != expected_ids)
        {
            std::cout << "\n[DEBUG: Test 4 encode failed]\n";
            return false;
        }
        std::string decoded = tokenizer.decode(ids, true);
        if (decoded != text)
        {
            std::cout << "\n[DEBUG: Test 4 decode failed]\n";
            return false;
        }
    }

    // Test 5: Special control tokens không decode khi skip_special = true
    {
        std::vector<int32_t> tokens = {1, 23325, 2}; // <s> Hello </s>
        std::string decoded = tokenizer.decode(tokens, true);
        if (decoded != "Hello")
        {
            return false;
        }
    }

    return true;
}

bool testKvCacheManager(Execution_Target exec_target)
{
    constexpr size_t batch_size = 1;
    constexpr size_t num_heads = 2;
    constexpr size_t max_seq_len = 16;
    constexpr size_t head_dim = 8;

    // 1. Static Mode Test
    {
        KV_Cache_Manager cache(batch_size, num_heads, max_seq_len, head_dim, exec_target, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);
        if (cache.getCurrentSeqLen() != 0 || cache.getMaxSeqLen() != max_seq_len)
        {
            return false;
        }

        // Prefill 4 tokens: shape {1, 2, 4, 8}
        std::vector<float> k_data_prefill(1 * 2 * 4 * 8, 1.5f);
        std::vector<float> v_data_prefill(1 * 2 * 4 * 8, 2.5f);
        Tensor k_prefill(Shape{1, 2, 4, 8}, k_data_prefill, exec_target);
        Tensor v_prefill(Shape{1, 2, 4, 8}, v_data_prefill, exec_target);

        cache.append(k_prefill, v_prefill);
        if (exec_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }

        if (cache.getCurrentSeqLen() != 4)
        {
            return false;
        }

        Tensor k_slice = cache.getK();
        Tensor v_slice = cache.getV();
        if (k_slice.getShape() != Shape{1, 2, 4, 8} || v_slice.getShape() != Shape{1, 2, 4, 8})
        {
            return false;
        }

        if (!verifyMatrix(k_slice, k_data_prefill) || !verifyMatrix(v_slice, v_data_prefill))
        {
            return false;
        }

        // Autoregressive decoding: append 2 tokens one by one
        for (size_t t = 0; t < 2; ++t)
        {
            std::vector<float> k_tok(1 * 2 * 1 * 8, static_cast<float>(t + 10));
            std::vector<float> v_tok(1 * 2 * 1 * 8, static_cast<float>(t + 20));
            Tensor k_new(Shape{1, 2, 1, 8}, k_tok, exec_target);
            Tensor v_new(Shape{1, 2, 1, 8}, v_tok, exec_target);
            cache.append(k_new, v_new);
        }
        if (exec_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }

        if (cache.getCurrentSeqLen() != 6)
        {
            return false;
        }

        Tensor k_6 = cache.getK();
        if (k_6.getShape() != Shape{1, 2, 6, 8})
        {
            return false;
        }

        // Test Reset without reallocation
        cache.reset();
        if (cache.getCurrentSeqLen() != 0)
        {
            return false;
        }
    }

    // 2. Ring Buffer Mode Test (Rolling Window)
    {
        constexpr size_t ring_max_seq = 4;
        KV_Cache_Manager ring_cache(1, 1, ring_max_seq, 4, exec_target, Data_Type::FLOAT32, KV_Cache_Mode::RING_BUFFER);

        // Append 6 tokens sequentially into capacity-4 ring buffer
        for (size_t i = 1; i <= 6; ++i)
        {
            std::vector<float> tok_val(4, static_cast<float>(i));
            Tensor k_tok(Shape{1, 1, 1, 4}, tok_val, exec_target);
            Tensor v_tok(Shape{1, 1, 1, 4}, tok_val, exec_target);
            ring_cache.append(k_tok, v_tok);
        }
        if (exec_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }

        if (ring_cache.getCurrentSeqLen() != ring_max_seq)
        {
            return false;
        }

        Tensor k_res = ring_cache.getK();
        // Since ring buffer holds the latest 4 tokens, tokens should be [3, 4, 5, 6]
        std::vector<float> expected_ring = {
            3.0f, 3.0f, 3.0f, 3.0f,
            4.0f, 4.0f, 4.0f, 4.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            6.0f, 6.0f, 6.0f, 6.0f
        };
        if (!verifyMatrix(k_res, expected_ring))
        {
            return false;
        }
    }

    return true;
}

bool testTransformerBlock(Execution_Target exec_target)
{
    constexpr size_t hidden_dim = 64;
    constexpr size_t num_heads = 4;
    constexpr size_t head_dim = 16;
    constexpr size_t intermediate_dim = 128;
    constexpr size_t S = 4;

    Transformer_Block block(hidden_dim, num_heads, intermediate_dim, 1e-5f, 10000.0f, exec_target);

    // 1. Prefill Forward without KV Cache
    std::vector<float> in_data(S * hidden_dim, 0.1f);
    for (size_t i = 0; i < in_data.size(); ++i)
    {
        in_data[i] = 0.05f * static_cast<float>((i % 9) + 1);
    }
    Tensor in_tensor(Shape{ 1, S, hidden_dim }, in_data, exec_target);
    Tensor out_tensor = block.forward(in_tensor);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    if (out_tensor.getTotalElements() != S * hidden_dim)
    {
        return false;
    }
    std::vector<float> out_data = out_tensor.getData();
    for (float val : out_data)
    {
        if (std::isnan(val) || std::isinf(val))
        {
            return false;
        }
    }

    // 2. Incremental Forward with KV Cache
    KV_Cache_Manager cache(1, num_heads, 16, head_dim, exec_target, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);

    // Step A: Prefill 3 tokens
    Tensor prefill_in(Shape{ 1, 3, hidden_dim }, std::vector<float>(3 * hidden_dim, 0.1f), exec_target);
    Tensor prefill_out = block.forward(prefill_in, &cache);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (cache.getCurrentSeqLen() != 3)
    {
        return false;
    }

    // Step B: Decode 1 token
    Tensor decode_in(Shape{ 1, 1, hidden_dim }, std::vector<float>(1 * hidden_dim, 0.2f), exec_target);
    Tensor decode_out = block.forward(decode_in, &cache);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (cache.getCurrentSeqLen() != 4)
    {
        return false;
    }
    if (decode_out.getTotalElements() != hidden_dim)
    {
        return false;
    }

    block.setKVCache(nullptr);
    Tensor fwd_res = block.forward(in_tensor);
    Tensor d_out(Shape{ S, hidden_dim }, std::vector<float>(S * hidden_dim, 0.01f), exec_target);
    Tensor d_in = block.backward(d_out);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (d_in.getTotalElements() != S * hidden_dim)
    {
        return false;
    }
    std::vector<float> d_in_data = d_in.getData();
    for (float val : d_in_data)
    {
        if (std::isnan(val) || std::isinf(val))
        {
            return false;
        }
    }

    // 4. Test 3D Batching Forward & Backward: [B, S, hidden_dim]
    constexpr size_t B_test = 2;
    std::vector<float> batch_in_data(B_test * S * hidden_dim, 0.1f);
    for (size_t i = 0; i < batch_in_data.size(); ++i)
    {
        batch_in_data[i] = 0.05f * static_cast<float>((i % 9) + 1);
    }
    Tensor batch_in(Shape{ B_test, S, hidden_dim }, batch_in_data, exec_target);
    Tensor batch_out = block.forward(batch_in);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (batch_out.getShape() != Shape{ B_test, S, hidden_dim })
    {
        std::cout << " [TB DBG: Step 4 b_out shape failed, rank=" << batch_out.getShape().getRank() << " dims: ";
        for (size_t d = 0; d < batch_out.getShape().getRank(); ++d) std::cout << batch_out.getShape()[d] << " ";
        std::cout << "] ";
        return false;
    }
    std::vector<float> b_out_data = batch_out.getData();
    for (float val : b_out_data)
    {
        if (std::isnan(val) || std::isinf(val))
        {
            std::cout << " [TB DBG: Step 4 b_out nan/inf] ";
            return false;
        }
    }

    Tensor d_batch_out(Shape{ B_test, S, hidden_dim }, std::vector<float>(B_test * S * hidden_dim, 0.01f), exec_target);
    Tensor d_batch_in = block.backward(d_batch_out);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (d_batch_in.getShape() != Shape{ B_test, S, hidden_dim })
    {
        std::cout << " [TB DBG: Step 4 d_b_in shape failed] ";
        return false;
    }
    std::vector<float> d_b_in_data = d_batch_in.getData();
    for (float val : d_b_in_data)
    {
        if (std::isnan(val) || std::isinf(val))
        {
            std::cout << " [TB DBG: Step 4 d_b_in nan/inf] ";
            return false;
        }
    }

    return true;
}

bool testCausalLMModel(Execution_Target exec_target)
{
    Causal_LM_Config cfg;
    cfg.vocab_size = 512;
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.intermediate_dim = 128;
    cfg.num_layers = 2;
    cfg.max_seq_len = 64;
    cfg.execution_target = exec_target;
    cfg.data_type = Data_Type::FLOAT32;
    cfg.tokenizer_path = "tokenizer/tokenizer.json";

    Causal_LM model(cfg);

    // 1. Forward test with token IDs
    std::vector<int32_t> test_tokens = { 1, 15, 42, 100 };
    Tensor logits = model.forward(test_tokens, false);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    if (logits.getTotalElements() != test_tokens.size() * cfg.vocab_size)
    {
        return false;
    }

    std::vector<float> logits_data = logits.getData();
    for (size_t i = 0; i < std::min<size_t>(100, logits_data.size()); ++i)
    {
        if (std::isnan(logits_data[i]) || std::isinf(logits_data[i]))
        {
            return false;
        }
    }

    // 2. Autoregressive Text Generation test
    std::string prompt = "Hello";
    std::string generated = model.generate(prompt, 6, 0.7f, 0.9f);
    if (generated.empty())
    {
        return false;
    }

    // 3. Neural_Network Unified Pipeline with Embedding & Transformer_Block
    Neural_Network nn(exec_target);
    nn.addLayer<Embedding_Layer>(cfg.vocab_size, cfg.hidden_dim, exec_target);
    nn.addLayer<Transformer_Block>(cfg.hidden_dim, cfg.num_heads, cfg.intermediate_dim, 1e-5f, 10000.0f, exec_target);
    nn.addLayer<RMSNorm_Layer>(cfg.hidden_dim, 1e-5f, exec_target);
    nn.addLayer<Linear_Layer>(cfg.hidden_dim, cfg.vocab_size, exec_target);

    Tensor input_tokens(1, 4, std::vector<float>{ 1.0f, 15.0f, 42.0f, 100.0f }, exec_target);
    Tensor nn_out = nn.forward(input_tokens);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    if (nn_out.getTotalElements() != 4 * cfg.vocab_size)
    {
        return false;
    }

    return true;
}

bool testCausalLMTraining(Execution_Target exec_target)
{
    Causal_LM_Config cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.num_layers = 1;
    cfg.max_seq_len = 32;
    cfg.execution_target = exec_target;
    cfg.data_type = Data_Type::FLOAT32;
    cfg.tokenizer_path = "tokenizer/tokenizer.json";

    Causal_LM model(cfg);

    // 1. Test Loss and Gradient Computation
    std::vector<int32_t> input_tokens = { 5, 12, 23, 40 };
    std::vector<int32_t> target_tokens = { 12, 23, 40, 55 };

    Tensor logits = model.forward(input_tokens, false);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    Tensor d_logits;
    float initial_loss = model.computeLossAndGradient(logits, target_tokens, d_logits);
    if (std::isnan(initial_loss) || std::isinf(initial_loss) || initial_loss <= 0.0f)
    {
        return false;
    }

    if (d_logits.getTotalElements() != input_tokens.size() * cfg.vocab_size)
    {
        return false;
    }

    std::vector<float> d_logits_data = d_logits.getData();
    for (size_t s = 0; s < input_tokens.size(); ++s)
    {
        float row_sum = 0.0f;
        for (size_t v = 0; v < cfg.vocab_size; ++v)
        {
            row_sum += d_logits_data[s * cfg.vocab_size + v];
        }
        if (std::fabs(row_sum) > 1e-4f)
        {
            return false;
        }
    }

    // 2. Test Training Step & Convergence with Adam Optimizer
    Adam_Optimizer optimizer(0.02f);
    float current_loss = initial_loss;

    for (int step = 0; step < 8; ++step)
    {
        current_loss = model.trainStep(input_tokens, target_tokens, optimizer);
        if (std::isnan(current_loss) || std::isinf(current_loss))
        {
            return false;
        }
    }

    if (current_loss >= initial_loss)
    {
        return false;
    }

    // 3. Test Checkpoint Serialization (Save & Load)
    std::string ckpt_path = (exec_target == Execution_Target::CPU) ? "causal_lm_cpu.ckpt" : "causal_lm_gpu.ckpt";
    model.saveCheckpoint(ckpt_path);

    Causal_LM loaded_model(cfg);
    loaded_model.loadCheckpoint(ckpt_path);
    std::filesystem::remove(ckpt_path);

    Tensor logits_orig = model.forward(input_tokens, false);
    Tensor logits_loaded = loaded_model.forward(input_tokens, false);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::vector<float> orig_data = logits_orig.getData();
    std::vector<float> loaded_data = logits_loaded.getData();
    if (orig_data.size() != loaded_data.size())
    {
        return false;
    }
    for (size_t i = 0; i < orig_data.size(); ++i)
    {
        if (std::fabs(orig_data[i] - loaded_data[i]) > 1e-4f)
        {
            return false;
        }
    }

    // 4. Test Inference Serialization (Save & Load)
    std::string inf_path = (exec_target == Execution_Target::CPU) ? "causal_lm_cpu.bin" : "causal_lm_gpu.bin";
    model.saveInference(inf_path);

    Causal_LM inf_model(cfg);
    inf_model.loadInference(inf_path);
    std::filesystem::remove(inf_path);

    Tensor logits_inf = inf_model.forward(input_tokens, false);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    std::vector<float> inf_data = logits_inf.getData();
    for (size_t i = 0; i < orig_data.size(); ++i)
    {
        if (std::fabs(orig_data[i] - inf_data[i]) > 1e-4f)
        {
            return false;
        }
    }

    // 5. Test Text Training Step with Tokenizer
    if (model.getTokenizer().isLoaded())
    {
        Causal_LM_Config text_cfg = cfg;
        text_cfg.vocab_size = model.getTokenizer().getVocabSize();
        Causal_LM text_model(text_cfg);
        Adam_Optimizer text_opt(0.01f);
        float text_loss = text_model.trainStep("Antigravity sLLM training engine", text_opt);
        if (std::isnan(text_loss) || std::isinf(text_loss) || text_loss <= 0.0f)
        {
            return false;
        }
    }

    // 6. Test Gradient Clipping
    model.forwardLossAndBackward(input_tokens, target_tokens, 1.0f);
    float norm_clipped = model.clipGradients(0.5f);
    if (std::isnan(norm_clipped) || std::isinf(norm_clipped))
    {
        return false;
    }
    model.resetGradients();

    // 7. Test Gradient Accumulation (micro-batching)
    model.setAccumulated(false);
    model.resetGradients();
    float mb1_loss = model.forwardLossAndBackward(input_tokens, target_tokens, 0.5f);
    model.setAccumulated(true);
    float mb2_loss = model.forwardLossAndBackward(input_tokens, target_tokens, 0.5f);
    model.stepOptimizer(optimizer, 1.0f);
    if (std::isnan(mb1_loss) || std::isnan(mb2_loss))
    {
        return false;
    }

    // 8. Test Binary Token Dataset I/O
    std::string bin_path = "temp_test_dataset.bin";
    std::vector<int32_t> sample_tokens = { 1, 2, 42, 1337, 99999 };
    if (!Binary_Token_Dataset::save(bin_path, sample_tokens))
    {
        return false;
    }
    std::vector<int32_t> loaded_tokens;
    if (!Binary_Token_Dataset::load(bin_path, loaded_tokens) || loaded_tokens != sample_tokens)
    {
        std::filesystem::remove(bin_path);
        return false;
    }
    std::filesystem::remove(bin_path);

    // 9. Test Top-k and Stop Sequences Generation
    std::string topk_gen = model.generate("Hello", 4, 0.7f, 0.9f, 2, true, nullptr, 1.2f, 5, { "el" });
    if (topk_gen.empty())
    {
        return false;
    }

    // 10. Test 3D Batching Train Step: B = 2, S = 4
    std::vector<std::vector<int32_t>> batch_inputs = {
        { 5, 12, 23, 40 },
        { 8, 14, 25, 42 }
    };
    std::vector<std::vector<int32_t>> batch_targets = {
        { 12, 23, 40, 55 },
        { 14, 25, 42, 60 }
    };
    float batch_loss = model.trainStep(batch_inputs, batch_targets, optimizer);
    if (std::isnan(batch_loss) || std::isinf(batch_loss) || batch_loss <= 0.0f)
    {
        return false;
    }

    return true;
}

bool testCausalLMFP16Training(Execution_Target exec_target)
{
    if (exec_target != Execution_Target::VULKAN_GPU)
    {
        return true;
    }

    Causal_LM_Config cfg;
    cfg.vocab_size = 64;
    cfg.hidden_dim = 32;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.num_layers = 1;
    cfg.max_seq_len = 32;
    cfg.execution_target = exec_target;
    cfg.data_type = Data_Type::FLOAT16;
    cfg.use_loss_scaler = true;
    cfg.initial_loss_scale = 128.0f;
    cfg.tokenizer_path = "tokenizer/tokenizer.json";

    Causal_LM model(cfg);

    std::vector<int32_t> input_tokens = { 5, 12, 23, 40 };
    std::vector<int32_t> target_tokens = { 12, 23, 40, 55 };

    // 1. Forward pass pure FP16
    Tensor logits = model.forward(input_tokens, false);
    if (logits.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }
    Execution_Engine::getInstance().executeGraph();

    // 2. Compute loss and FP16 d_logits
    Tensor d_logits;
    float initial_loss = model.computeLossAndGradient(logits, target_tokens, d_logits);
    if (std::isnan(initial_loss) || std::isinf(initial_loss) || initial_loss <= 0.0f)
    {
        return false;
    }
    if (d_logits.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }

    // 3. Train steps with FP32 Master Weights + FP16 Graph + Loss Scaler
    Adam_Optimizer optimizer(0.02f);
    float current_loss = initial_loss;

    for (int step = 0; step < 8; ++step)
    {
        current_loss = model.trainStep(input_tokens, target_tokens, optimizer);
        if (std::isnan(current_loss) || std::isinf(current_loss))
        {
            return false;
        }
    }

    if (current_loss >= initial_loss)
    {
        return false;
    }

    // 4. Batching FP16 train step: B = 2, S = 4
    std::vector<std::vector<int32_t>> batch_inputs = {
        { 5, 12, 23, 40 },
        { 8, 14, 25, 42 }
    };
    std::vector<std::vector<int32_t>> batch_targets = {
        { 12, 23, 40, 55 },
        { 14, 25, 42, 60 }
    };
    float batch_loss = model.trainStep(batch_inputs, batch_targets, optimizer);
    if (std::isnan(batch_loss) || std::isinf(batch_loss) || batch_loss <= 0.0f)
    {
        return false;
    }

    return true;
}

bool testResNetBlock2dLayer(Execution_Target exec_target)
{
    auto validateTensor = [](const Tensor& tensor, size_t expected_rows, size_t expected_columns, bool require_nonzero = false) -> bool
        {
            if (tensor.getExecutionTarget() == Execution_Target::VULKAN_GPU)
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

    auto validateParameters = [&validateTensor](const std::vector<std::pair<Tensor*, Tensor*>>& parameters, size_t expected_count) -> bool
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

    constexpr size_t batch_size = 1;
    constexpr size_t input_height = 4;
    constexpr size_t input_width = 4;
    constexpr size_t input_channels = 16;
    constexpr size_t input_features = input_height * input_width * input_channels;

    std::vector<float> input_data(input_features);
    for (size_t i = 0; i < input_features; ++i)
    {
        input_data[i] = 0.01f * static_cast<float>((i % 17) + 1);
    }

    std::vector<float> grad_identity_data(input_features);
    for (size_t i = 0; i < input_features; ++i)
    {
        grad_identity_data[i] = 0.02f * static_cast<float>((i % 13) + 1);
    }

    Res_Net_Block_2d_Layer block_identity(input_height, input_width, input_channels, input_channels, 1, exec_target);
    Tensor input_identity(batch_size, input_features, input_data, exec_target);
    Tensor out_identity = block_identity.forward(input_identity);
    bool identity_fwd_ok = validateTensor(out_identity, batch_size, input_features, true);

    Tensor grad_output_identity(batch_size, input_features, grad_identity_data, exec_target);
    Tensor grad_in_identity = block_identity.backward(grad_output_identity);
    bool identity_bwd_ok = validateTensor(grad_in_identity, batch_size, input_features, true);
    bool identity_params_ok = validateParameters(block_identity.getParametersAndGradients(), 8);
    block_identity.resetGradient();

    constexpr size_t proj_out_channels = 32;
    constexpr size_t proj_stride = 2;
    constexpr size_t proj_out_height = (input_height + proj_stride - 1) / proj_stride;
    constexpr size_t proj_out_width = (input_width + proj_stride - 1) / proj_stride;
    constexpr size_t proj_out_features = proj_out_height * proj_out_width * proj_out_channels;

    std::vector<float> grad_proj_data(proj_out_features);
    for (size_t i = 0; i < proj_out_features; ++i)
    {
        grad_proj_data[i] = 0.02f * static_cast<float>((i % 11) + 1);
    }

    Res_Net_Block_2d_Layer block_proj(input_height, input_width, input_channels, proj_out_channels, proj_stride, exec_target);
    Tensor input_proj(batch_size, input_features, input_data, exec_target);
    Tensor out_proj = block_proj.forward(input_proj);
    bool proj_fwd_ok = validateTensor(out_proj, batch_size, proj_out_features, true);

    Tensor grad_output_proj(batch_size, proj_out_features, grad_proj_data, exec_target);
    Tensor grad_in_proj = block_proj.backward(grad_output_proj);
    bool proj_bwd_ok = validateTensor(grad_in_proj, batch_size, input_features, true);
    bool proj_params_ok = validateParameters(block_proj.getParametersAndGradients(), 12);
    block_proj.resetGradient();

    return identity_fwd_ok && identity_bwd_ok && identity_params_ok &&
        proj_fwd_ok && proj_bwd_ok && proj_params_ok;
}

bool testSgdOptimizer(Execution_Target exec_target)
{
    Tensor param_mat(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor grad(2, 2, { 0.5f, -1.0f, 2.0f, -3.0f }, exec_target);

    Sgd_Optimizer optimizer(0.1f, 100.0f);
    optimizer.step({ { &param_mat, &grad } });

    return verifyMatrix(param_mat, { 0.95f, 2.1f, 2.8f, 4.3f });
}

bool testAdamOptimizer(Execution_Target exec_target)
{
    Tensor param_mat(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor grad_mat(1, 2, { 0.1f, -0.2f }, exec_target);

    Adam_Optimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f);
    optimizer.step({ { &param_mat, &grad_mat } });

    if (!verifyMatrix(param_mat, { 0.999f, 2.001f }))
    {
        return false;
    }

    Tensor param_wd(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor grad_wd(1, 2, { 0.1f, -0.2f }, exec_target);

    Adam_Optimizer optimizer_wd(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f, 0.1f);
    optimizer_wd.step({ { &param_wd, &grad_wd } });

    return verifyMatrix(param_wd, { 0.9989f, 2.0008f });
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

    Cosine_Annealing warmup_cosine(0.1f, 0.0f, 10, 2);
    warmup_cosine.step();
    bool warmup1_ok = nearlyEqual(warmup_cosine.getCurrentRate(), 0.05f);
    warmup_cosine.step();
    bool warmup2_ok = nearlyEqual(warmup_cosine.getCurrentRate(), 0.1f);
    warmup_cosine.step();
    bool warmup3_ok = (warmup_cosine.getCurrentRate() < 0.1f) && (warmup_cosine.getCurrentRate() > 0.0f);

    return no_decay_ok && step1_ok && step2_ok && multi_step_ok && exp_ok && cosine_ok && poly_ok && plateau_ok && warmup1_ok && warmup2_ok && warmup3_ok;
}

bool testMatrixSerialization(Execution_Target exec_target)
{
    std::string temp_file = "temp_matrix_serialization.bin";
    Tensor original(2, 3, { 1.0f, -2.5f, 3.2f, 4.8f, 5.0f, -6.1f }, exec_target);

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
    Tensor loaded = Tensor::loadMatrix(in_file, exec_target);
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

    Tensor input_data(1, 2, { 1.5f, -0.5f }, exec_target);
    Tensor pred_before = network.forward(input_data);

    network.saveInference(temp_file);

    Neural_Network loaded_network(exec_target);
    loaded_network.loadInference(temp_file, exec_target);
    std::remove(temp_file.c_str());

    Tensor pred_after = loaded_network.forward(input_data);

    return verifyMatrix(pred_after, pred_before.getData());
}

bool testGpuVectorLifecycle()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    const Vulkan_Context& context = engine.getContext();

    size_t initial_count = 1024;
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

    size_t expanded_count = 2048;
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

    uint32_t current_frame = context.getCurrentFrame();
    Memory_Allocation garbage_alloc = allocator.allocate(mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    context.deferDestruction(current_frame, VK_NULL_HANDLE, garbage_alloc);

    context.cleanGarbage(current_frame);

    return reuse_ok;
}

bool testOperatorFusionAndGraphExecution()
{
    Execution_Engine& engine = Execution_Engine::getInstance();
    engine.getCurrentGraph().clear();

    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_b(2, 2, { 2.0f, 3.0f, 4.0f, 5.0f }, Execution_Target::VULKAN_GPU);

    Tensor mat_add = mat_a + mat_b;
    Tensor mat_relu = mat_add.relu();

    size_t raw_node_count = engine.getCurrentGraph().getNodeCount();
    if (raw_node_count < 2)
    {
        return false;
    }

    engine.executeGraph();
    if (!verifyMatrix(mat_relu, { 3.0f, 5.0f, 7.0f, 9.0f }))
    {
        return false;
    }

    Tensor mat_c(2, 2, { -10.0f, 5.0f, 0.0f, 2.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_d(2, 2, { 3.0f, 2.0f, 1.0f, -5.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_add2 = mat_c + mat_d;
    Tensor mat_relu2 = mat_add2.relu();

    engine.executeGraph();
    if (!verifyMatrix(mat_relu2, { 0.0f, 7.0f, 1.0f, 0.0f }))
    {
        return false;
    }

    Tensor mat_e(2, 2, { 10.0f, -20.0f, 30.0f, -40.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_f(2, 2, { 5.0f, 5.0f, -5.0f, -5.0f }, Execution_Target::VULKAN_GPU);
    Tensor mat_add3 = mat_e + mat_f;
    Tensor mat_relu3 = mat_add3.relu();

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
    void prepareBatchHost(size_t batch_step, std::vector<float>& output_inputs, std::vector<float>& output_targets) override
    {
        std::fill(output_inputs.begin(), output_inputs.end(), 1.0f);
        std::fill(output_targets.begin(), output_targets.end(), 2.0f);
    }

public:
    Dummy_Data_Pipeline() : Async_Data_Pipeline() {}
    size_t getBatchSize() const override { return 2; }
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
    constexpr size_t capacity = 4;
    Replay_Buffer buffer(capacity, 1337);

    if (buffer.getSize() != 0 || buffer.getCapacity() != capacity || buffer.isReady(1))
    {
        return false;
    }

    for (size_t i = 0; i < 5; ++i)
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
    constexpr size_t state_dim = 2;
    constexpr size_t action_dim = 2;

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

    size_t deterministic_action = agent.selectAction({ 1.0f, 0.5f }, false);
    bool action_ok = (deterministic_action < action_dim);

    for (size_t i = 0; i < 6; ++i)
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
    constexpr size_t pop_size = 8;
    constexpr size_t state_dim = 16;
    constexpr size_t action_dim = 4;
    constexpr size_t hidden_dim = 32;

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
    for (size_t i = 0; i < state_dim; ++i)
    {
        test_state[i] = std::sin(static_cast<float>(i) * 0.5f);
    }

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t act1 = pop.selectAction(i, test_state);
        size_t act2 = pop.selectAction(i, test_state);
        if (act1 != act2 || act1 >= action_dim)
        {
            return false;
        }
    }

    Neural_Network ind0 = pop.getIndividual(0);
    Tensor ind0_input(1, state_dim, test_state, exec_target);
    Tensor ind0_output = ind0.forward(ind0_input);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    std::vector<float> q_vals = ind0_output.getData();
    size_t expected_act0 = 0;
    float max_q = q_vals[0];
    for (size_t a = 1; a < q_vals.size(); ++a)
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
    std::vector<size_t> expected_batch(pop_size);
    for (size_t i = 0; i < pop_size; ++i)
    {
        for (size_t d = 0; d < state_dim; ++d)
        {
            flat_states[i * state_dim + d] = std::cos(static_cast<float>(i * state_dim + d) * 0.2f);
        }
        expected_batch[i] = pop.selectAction(i, flat_states.data() + (i * state_dim));
    }

    std::vector<size_t> batch_results(pop_size);
    pop.selectBatchActions(flat_states.data(), nullptr, pop_size, batch_results.data());
    for (size_t i = 0; i < pop_size; ++i)
    {
        if (batch_results[i] != expected_batch[i])
        {
            return false;
        }
    }

    std::vector<size_t> active_indices = { 1, 3, 6 };
    std::vector<float> subset_states(active_indices.size() * state_dim);
    for (size_t k = 0; k < active_indices.size(); ++k)
    {
        size_t idx = active_indices[k];
        std::copy(flat_states.begin() + idx * state_dim,
            flat_states.begin() + (idx + 1) * state_dim,
            subset_states.begin() + k * state_dim);
    }
    std::vector<size_t> subset_results(active_indices.size());
    pop.selectBatchActions(subset_states.data(), active_indices.data(), active_indices.size(), subset_results.data());
    for (size_t k = 0; k < active_indices.size(); ++k)
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

    for (size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop_loader.selectAction(5, st) != pop.selectAction(2, st))
        {
            return false;
        }
    }

    constexpr size_t elite_candidate = 4;
    std::vector<size_t> ind4_actions(pop_size);
    for (size_t i = 0; i < pop_size; ++i)
    {
        ind4_actions[i] = pop.selectAction(elite_candidate, flat_states.data() + i * state_dim);
    }

    std::vector<float> fitness(pop_size, 0.0f);
    fitness[elite_candidate] = 1000.0f;

    Neural_Network best_ind_before = pop.getBestIndividual(fitness.data());
    Tensor in_matrix_best(1, state_dim, std::vector<float>(flat_states.begin(), flat_states.begin() + state_dim), exec_target);
    Tensor out_matrix_best = best_ind_before.forward(in_matrix_best);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    const auto& best_raw_data = out_matrix_best.getData();
    size_t best_initial_act = static_cast<size_t>(std::distance(best_raw_data.begin(), std::max_element(best_raw_data.begin(), best_raw_data.end())));
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

    for (size_t i = 0; i < pop_size; ++i)
    {
        const float* st = flat_states.data() + i * state_dim;
        if (pop_loader.selectAction(0, st) != ind4_actions[i])
        {
            return false;
        }
    }

    std::string checkpoint_temp = "temp_population_checkpoint.bin";
    uint64_t saved_gen = 105;
    if (!pop.saveCheckpoint(checkpoint_temp, saved_gen, fitness.data()))
    {
        return false;
    }

    uint64_t loaded_gen = 0;
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
    for (size_t i = 0; i < pop_size; ++i)
    {
        if (!nearlyEqual(loaded_fitness[i], fitness[i]))
        {
            std::remove(checkpoint_temp.c_str());
            return false;
        }
    }

    uint64_t loaded_gen_null = 0;
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
    for (size_t i = 0; i < pop_size; ++i)
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
    Tensor lin_in(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor lin_out = lin_clone->forward(lin_in);
    if (!verifyMatrix(lin_out, { 9.1f, 12.2f, 15.3f }, (exec_target == Execution_Target::VULKAN_GPU) ? 0.02f : 1e-3f))
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
    constexpr size_t pop_size = 6;
    constexpr size_t state_dim = 8;
    constexpr size_t action_dim = 3;

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
        std::cout << "Place 1";
        return false;
    }

    std::vector<float> input_sample(state_dim);
    for (size_t i = 0; i < state_dim; ++i)
    {
        input_sample[i] = 0.1f * static_cast<float>(i + 1);
    }

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t act = pop.selectAction(i, input_sample);
        if (act >= action_dim)
        {
            std::cout << "Place 2";
            return false;
        }

        Neural_Network ind = pop.getIndividual(i);
        Tensor in_mat(1, state_dim, input_sample, exec_target);
        Tensor out_mat = ind.forward(in_mat);
        if (exec_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }
        const auto& out_data = out_mat.getData();
        size_t expected_act = static_cast<size_t>(std::distance(out_data.begin(), std::max_element(out_data.begin(), out_data.end())));
        if (act != expected_act)
        {
            std::cout << "Place 3";
            return false;
        }
    }

    std::vector<float> batch_input(pop_size * state_dim);
    for (size_t i = 0; i < pop_size * state_dim; ++i)
    {
        batch_input[i] = std::sin(static_cast<float>(i) * 0.3f);
    }

    std::vector<size_t> batch_actions(pop_size);
    pop.selectBatchActions(batch_input.data(), nullptr, pop_size, batch_actions.data());

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t single_act = pop.selectAction(i, batch_input.data() + i * state_dim);
        if (batch_actions[i] != single_act)
        {
            std::cout << "Place 4";
            return false;
        }
    }

    std::vector<float> fitness(pop_size);
    for (size_t i = 0; i < pop_size; ++i)
    {
        fitness[i] = static_cast<float>(i * 10);
    }

    Neural_Network best_ind = pop.getBestIndividual(fitness.data());
    Tensor in_mat_best(1, state_dim, input_sample, exec_target);
    Tensor best_out = best_ind.forward(in_mat_best);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    const auto& best_data = best_out.getData();
    size_t best_expected = static_cast<size_t>(std::distance(best_data.begin(), std::max_element(best_data.begin(), best_data.end())));
    if (pop.selectAction(pop_size - 1, input_sample) != best_expected)
    {
        std::cout << "Place 5";
        return false;
    }

    std::string ckpt_path = "temp_deep_pop.bin";
    uint64_t gen_in = 50;
    if (!pop.saveCheckpoint(ckpt_path, gen_in, fitness.data()))
    {
        std::cout << "Place 6";
        return false;
    }

    Population pop_loader(pop_size, template_net, state_dim, action_dim, exec_target, 111);
    uint64_t gen_out = 0;
    std::vector<float> fit_out(pop_size);
    if (!pop_loader.loadCheckpoint(ckpt_path, gen_out, fit_out.data()))
    {
        std::remove(ckpt_path.c_str());
        std::cout << "Place 7";
        return false;
    }
    std::remove(ckpt_path.c_str());

    if (gen_out != gen_in)
    {
        std::cout << "Place 8";
        return false;
    }
    for (size_t i = 0; i < pop_size; ++i)
    {
        if (!nearlyEqual(fit_out[i], fitness[i]))
        {
            std::cout << "Place 9";
            return false;
        }
        if (pop_loader.selectAction(i, input_sample) != pop.selectAction(i, input_sample))
        {
            std::cout << "Place 9";
            return false;
        }
    }
    return true;
}

bool testMatrixConcatAndSplit(Execution_Target exec_target)
{
    Tensor mat_a(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor mat_b(2, 1, { 5.0f, 6.0f }, exec_target);

    Tensor concat_cols_res = mat_a.concatenateColumns(mat_b);
    bool concat_cols_ok = verifyMatrix(concat_cols_res, { 1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f });

    auto [split_left, split_right] = concat_cols_res.splitColumns(2);
    bool split_cols_ok = verifyMatrix(split_left, { 1.0f, 2.0f, 3.0f, 4.0f }) &&
        verifyMatrix(split_right, { 5.0f, 6.0f });

    Tensor mat_row_a(1, 2, { 10.0f, 20.0f }, exec_target);
    Tensor mat_row_b(2, 2, { 30.0f, 40.0f, 50.0f, 60.0f }, exec_target);

    Tensor concat_rows_res = mat_row_a.concatenateRows(mat_row_b);
    bool concat_rows_ok = verifyMatrix(concat_rows_res, { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f });

    auto [split_up, split_down] = concat_rows_res.splitRows(1);
    bool split_rows_ok = verifyMatrix(split_up, { 10.0f, 20.0f }) &&
        verifyMatrix(split_down, { 30.0f, 40.0f, 50.0f, 60.0f });

    return concat_cols_ok && split_cols_ok && concat_rows_ok && split_rows_ok;
}

bool testPpoActorCriticForward(Execution_Target exec_target)
{
    constexpr uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto& actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Tensor(2, 2, { 1.0f, 0.0f, 0.0f, 1.0f }, exec_target));
    actor_linear.setBiases(Tensor(1, 2, { 0.5f, -0.5f }, exec_target));

    auto& critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Tensor(2, 1, { 1.0f, 2.0f }, exec_target));
    critic_linear.setBiases(Tensor(1, 1, { 1.0f }, exec_target));

    Tensor input_matrix(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f }, exec_target);
    Tensor output_matrix = ppo_layer.forward(input_matrix);

    bool output_ok = verifyMatrix(output_matrix, { 1.5f, 1.5f, 6.0f, 3.5f, 3.5f, 12.0f });
    output_matrix.print();
    bool actor_sub_ok = verifyMatrix(ppo_layer.getActorOutput(), { 1.5f, 1.5f, 3.5f, 3.5f });
    ppo_layer.getActorOutput().print();
    bool critic_sub_ok = verifyMatrix(ppo_layer.getCriticOutput(), { 6.0f, 12.0f });
    ppo_layer.getCriticOutput().print();
    return output_ok && actor_sub_ok && critic_sub_ok;
}

bool testPpoActorCriticBackward(Execution_Target exec_target)
{
    constexpr uint64_t action_dim = 2;
    PPO_Actor_Critic_Layer ppo_layer(action_dim, exec_target);

    auto& actor_linear = ppo_layer.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Tensor(2, 2, { 1.0f, 0.0f, 0.0f, 1.0f }, exec_target));
    actor_linear.setBiases(Tensor(1, 2, { 0.0f, 0.0f }, exec_target));

    auto& critic_linear = ppo_layer.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Tensor(2, 1, { 1.0f, 1.0f }, exec_target));
    critic_linear.setBiases(Tensor(1, 1, { 0.0f }, exec_target));

    Tensor input_matrix(1, 2, { 2.0f, 3.0f }, exec_target);
    ppo_layer.forward(input_matrix);

    Tensor output_gradient(1, 3, { 1.0f, 2.0f, 3.0f }, exec_target);
    Tensor input_gradient = ppo_layer.backward(output_gradient);

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
    constexpr uint64_t action_dim = 2;

    PPO_Actor_Critic_Layer ppo_source(action_dim, exec_target);
    auto& actor_linear = ppo_source.addActorLayer<Linear_Layer>(2, 2, exec_target);
    actor_linear.setWeights(Tensor(2, 2, { 1.5f, -0.5f, 0.5f, 2.0f }, exec_target));
    actor_linear.setBiases(Tensor(1, 2, { 0.1f, -0.2f }, exec_target));

    auto& critic_linear = ppo_source.addCriticLayer<Linear_Layer>(2, 1, exec_target);
    critic_linear.setWeights(Tensor(2, 1, { 0.8f, -1.2f }, exec_target));
    critic_linear.setBiases(Tensor(1, 1, { 0.5f }, exec_target));

    Tensor input_mat(1, 2, { 1.0f, 2.0f }, exec_target);
    Tensor pred_before = ppo_source.forward(input_mat);

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

    Tensor pred_after = ppo_loaded.forward(input_mat);

    return verifyMatrix(pred_after, pred_before.getData());
}

bool testLinearLayerInitializer(Execution_Target exec_target)
{
    Linear_Layer layer(64, 64, exec_target);
    auto init_fn = layer.getPopulationParameterInitializer(0);
    std::mt19937 rng(42);

    constexpr size_t sample_count = 1000;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < sample_count; ++i)
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
    for (size_t i = 0; i < expected.size(); ++i)
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

    std::cout << "\n--- [DEBUG: Gradient Accumulation Inspection] ---\n";
    auto mb1_params_grads = net.getParametersAndGradients();
    for (size_t i = 0; i < mb1_params_grads.size(); ++i)
    {
        std::cout << "Param [" << i << "] Gradient after Micro-batch 1 ("
            << mb1_params_grads[i].second->getRows() << "x"
            << mb1_params_grads[i].second->getColumns() << "):\n";
        mb1_params_grads[i].second->print();
    }

    Tensor mb2_out = net.forward(mb2_in);
    Tensor mb2_grad = mb2_out - mb2_target;
    net.backward(mb2_grad);
    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    auto accum_params_grads = net.getParametersAndGradients();
    float tol = (exec_target == Execution_Target::VULKAN_GPU) ? 1e-2f : 1e-3f;
    bool all_match = true;

    for (size_t i = 0; i < accum_params_grads.size(); ++i)
    {
        std::vector<float> accum_grad = accum_params_grads[i].second->getData();
        const auto& expected_grad = full_grads_data[i];

        std::cout << "Param [" << i << "] Accumulated Grad (MB1 + MB2):\n";
        accum_params_grads[i].second->print();

        std::cout << "Param [" << i << "] Expected Full-batch Grad:\n";
        Tensor expected_tensor(accum_params_grads[i].second->getShape(), expected_grad, exec_target);
        expected_tensor.print();

        if (accum_grad.size() != expected_grad.size())
        {
            std::cout << "[FAIL] Param [" << i << "] Size mismatch: "
                << accum_grad.size() << " != " << expected_grad.size() << "\n";
            return false;
        }

        float max_diff = 0.0f;
        for (size_t j = 0; j < accum_grad.size(); ++j)
        {
            float diff = std::abs(accum_grad[j] - expected_grad[j]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            if (!nearlyEqual(accum_grad[j], expected_grad[j], tol))
            {
                all_match = false;
            }
        }
        std::cout << "Param [" << i << "] Max Abs Diff = " << max_diff << "\n\n";
    }

    if (!all_match)
    {
        return false;
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
    constexpr size_t pop_size = 4;
    constexpr size_t state_dim = 4;
    constexpr size_t action_dim = 2;

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

    size_t pop_action = pop.selectAction(0, sample_input);
    const auto& net_data = net_out.getData();
    size_t net_action = static_cast<size_t>(std::distance(
        net_data.begin(), std::max_element(net_data.begin(), net_data.end())));

    if (pop_action != net_action)
    {
        return false;
    }

    pop.setIndividual(1, standalone_net);
    size_t pop_action_ind1 = pop.selectAction(1, sample_input);
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
    for (size_t i = 0; i < input_data.size(); ++i)
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
    for (size_t i = 0; i < input_data.size(); ++i)
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
    uint32_t f0 = engine.getContext().getCurrentFrame();
    Tensor in1(2, 4, std::vector<float>{1.0f, 0.5f, -0.5f, 2.0f, -1.0f, 0.0f, 1.5f, -2.0f}, Execution_Target::VULKAN_GPU);
    Tensor tgt1(2, 2, std::vector<float>{0.5f, -0.5f, 1.0f, 0.0f}, Execution_Target::VULKAN_GPU);
    nn.trainStep(in1, tgt1);
    engine.waitIdle();

    if (!engine.getGraphExecutor().isStaticBaked(f0))
    {
        return false;
    }

    // Step 2: Batch 2 -> should bake second frame
    uint32_t f1 = engine.getContext().getCurrentFrame();
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
    uint32_t f_inf = engine.getContext().getCurrentFrame();
    Tensor test_in(2, 4, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, Execution_Target::VULKAN_GPU);
    Tensor pred = nn.forward(test_in);
    engine.executeGraph();
    engine.waitIdle();

    // Now current frame of inference should be baked
    if (!engine.getGraphExecutor().isStaticBaked(f_inf))
    {
        return false;
    }

    // Replay inference on second batch
    Tensor pred2 = nn.forward(test_in);
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

    std::cout << "\n[1. Basic Tensor Arithmetics]\n";
    std::cout << "  Tensor Addition (with Broadcast):  " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Tensor Subtraction (with Broadcast): " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Tensor Multiplication (GEMM):      " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Scalar Multiplication & Division:   " << (testScalarOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Hadamard Multiplication & Division: " << (testHadamardOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Tensor Concat & Split (Row/Col):   " << (testMatrixConcatAndSplit(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[2. Transformations & Advanced Operations]\n";
    std::cout << "  Transpose & Tensor Inversion:       " << (testTransposeAndInverse(exec_target) ? "PASS" : "FAIL") << "\n";
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
    std::cout << "  Fused CCE Cost & Gradient:         " << (testFusedCrossEntropy(exec_target) ? "PASS" : "FAIL") << "\n";

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
    std::cout << "  RMSNorm Layer (Fwd & Bwd):         " << (testRmsNormLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  RoPE Rotary Embedding (Fwd & Bwd): " << (testRoPE(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Embedding Layer (Fwd/Bwd/Clone):   " << (testEmbeddingLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  SwiGLU FFN Layer (Fwd & Bwd):       " << (testSwiGluLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    auto safeRun = [](auto&& fn, Execution_Target target) -> std::string {
        try {
            return fn(target) ? "PASS" : "FAIL";
        } catch (const std::exception& e) {
            return std::string("[EXCEPTION: ") + e.what() + "] FAIL";
        } catch (...) {
            return "[UNKNOWN EXCEPTION] FAIL";
        }
    };

    std::cout << "  FlashAttention-2 FP16 (CoopMat):   " << safeRun(testFlashAttention, exec_target) << std::endl;
    std::cout << "  KV Cache Manager (Paged / Ring):   " << safeRun(testKvCacheManager, exec_target) << std::endl;
    std::cout << "  Transformer Block (Prefill & Decode): " << safeRun(testTransformerBlock, exec_target) << std::endl;
    std::cout << "  Causal LM Model (Forward & Generate): " << safeRun(testCausalLMModel, exec_target) << std::endl;
    std::cout << "  Causal LM Pre-training Engine Suite:  " << safeRun(testCausalLMTraining, exec_target) << std::endl;
    std::cout << "  Causal LM Pure FP16 Engine Suite:     " << safeRun(testCausalLMFP16Training, exec_target) << std::endl;

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
    std::cout << "  Tensor Binary I/O:                 " << (testMatrixSerialization(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Model Inference I/O (NNI1):        " << (testModelInferenceSerialization(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[9. Unified Architecture & New Features]\n";
    std::cout << "  Linear Layer Init (He/Xavier):      " << (testLinearLayerInitializer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  ReLU Batched Forward (Population):  " << (testReluBatchedForward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Gradient Accumulation (Micro-batch):" << (testGradientAccumulation(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Composite Layer Deserialization:    " << (testCompositeLayerSerialization(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Pop to NN Inference Compatibility:  " << (testCrossCompatibilityPopulationToNetwork(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Tensor Interface Unification:       " << (testTensorInterfaceUnification(exec_target) ? "PASS" : "FAIL") << "\n\n";
}


bool testTernaryComputingPhase1()
{
    // 1. Test sizing: getTensorStorageBytes with row-padding K
    size_t N = 3;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 1})) != 12) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 15})) != 12) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 16})) != 12) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 17})) != 24) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 33})) != 36) return false;

    // 2. Test bit-exact pack & unpack
    size_t test_rows = 2;
    size_t test_cols = 33;
    std::vector<float> original(test_rows * test_cols);
    for (size_t i = 0; i < original.size(); ++i)
    {
        int mod = static_cast<int>(i % 3);
        original[i] = (mod == 0) ? -1.0f : ((mod == 1) ? 0.0f : 1.0f);
    }
    size_t words_per_row = (test_cols + 15) / 16;
    std::vector<uint32_t> packed(test_rows * words_per_row, 0);
    Cpu_Tensor_Impl::packTernary(original.data(), packed.data(), test_rows, test_cols);

    std::vector<float> unpacked(test_rows * test_cols, 0.0f);
    float gamma = 0.75f;
    Cpu_Tensor_Impl::unpackTernary(packed.data(), unpacked.data(), test_rows, test_cols, gamma);

    for (size_t i = 0; i < original.size(); ++i)
    {
        float expected = original[i] * gamma;
        if (std::fabs(unpacked[i] - expected) > 1e-6f)
        {
            return false;
        }
    }

    // 3. Test math: ternaryMatmul vs FP32 GEMM
    size_t M = 2, K = 32, N_out = 4;
    std::vector<float> A(M * K);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i % 7) - 3.0f;

    std::vector<float> W_raw(N_out * K);
    for (size_t i = 0; i < W_raw.size(); ++i)
    {
        int mod = static_cast<int>(i % 3);
        W_raw[i] = (mod == 0) ? -1.0f : ((mod == 1) ? 0.0f : 1.0f);
    }
    std::vector<float> biases = {0.5f, -0.5f, 1.0f, -1.0f};

    size_t w_words_per_row = (K + 15) / 16;
    std::vector<uint32_t> B_packed(N_out * w_words_per_row);
    Cpu_Tensor_Impl::packTernary(W_raw.data(), B_packed.data(), N_out, K);

    std::vector<float> C_ternary(M * N_out, 0.0f);
    float test_gamma = 0.4f;
    Cpu_Tensor_Impl::ternaryMatmul(A.data(), B_packed.data(), C_ternary.data(), M, K, N_out, test_gamma, biases.data());

    std::vector<float> C_ref(M * N_out, 0.0f);
    for (size_t m = 0; m < M; ++m)
    {
        for (size_t n = 0; n < N_out; ++n)
        {
            float sum = biases[n];
            for (size_t k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * (W_raw[n * K + k] * test_gamma);
            }
            C_ref[m * N_out + n] = sum;
        }
    }

    for (size_t i = 0; i < C_ternary.size(); ++i)
    {
        if (std::fabs(C_ternary[i] - C_ref[i]) > 1e-4f)
        {
            return false;
        }
    }

    // 4. Negative tests on Tensor with TERNARY_INT2
    Tensor t_ternary(Shape({N_out, K}), Data_Type::TERNARY_INT2);
    bool caught_add = false;
    try { auto res = t_ternary + t_ternary; (void)res; } catch (const std::invalid_argument&) { caught_add = true; }
    if (!caught_add) return false;

    bool caught_relu = false;
    try { auto res = t_ternary.relu(); (void)res; } catch (const std::invalid_argument&) { caught_relu = true; }
    if (!caught_relu) return false;

    bool caught_transpose = false;
    try { auto res = t_ternary.transpose(); (void)res; } catch (const std::invalid_argument&) { caught_transpose = true; }
    if (!caught_transpose) return false;

    return true;
}


bool testTernaryComputingPhase2()
{
    auto &engine = Execution_Engine::getInstance();
    auto &context = engine.getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        std::cout << "[INFO: GPU does not enable/support FP16, skipping Ternary Phase 2 test] ";
        return true;
    }

    // 1. Pipeline verification
    auto pipe_p1 = engine.getNetwork().getPipeline(Compute_Pipeline::REDUCE_ABSMEAN_PASS1);
    auto pipe_p2 = engine.getNetwork().getPipeline(Compute_Pipeline::REDUCE_ABSMEAN_PASS2);
    auto pipe_fq = engine.getNetwork().getPipeline(Compute_Pipeline::FAKEQUANT_TERNARY);
    auto pipe_qp = engine.getNetwork().getPipeline(Compute_Pipeline::QUANTIZE_PACK_TERNARY);
    auto pipe_tg = engine.getNetwork().getPipeline(Compute_Pipeline::MATMUL_TERNARY_FP16);

    if (pipe_p1 == VK_NULL_HANDLE || pipe_p2 == VK_NULL_HANDLE || pipe_fq == VK_NULL_HANDLE ||
        pipe_qp == VK_NULL_HANDLE || pipe_tg == VK_NULL_HANDLE)
    {
        std::cout << "[ERROR: One or more Ternary pipelines failed to compile] ";
        return false;
    }

    // 2. Test Reduction: W_master [K=64, N=128]
    uint32_t K = 64;
    uint32_t N = 128;
    uint32_t total_elements = K * N;

    std::vector<float> host_w(total_elements);
    double cpu_sum_abs = 0.0;
    for (size_t i = 0; i < total_elements; ++i)
    {
        float val = std::sin(static_cast<float>(i + 1) * 0.123f);
        host_w[i] = val;
        cpu_sum_abs += std::fabs(val);
    }
    float cpu_gamma = std::max(static_cast<float>(cpu_sum_abs / static_cast<double>(total_elements)), 1e-5f);
    float cpu_inv_gamma = 1.0f / cpu_gamma;

    auto w_buf = std::make_shared<gpu::vector>(context, total_elements, Data_Type::FLOAT32);
    w_buf->uploadData(host_w);

    uint32_t num_workgroups = std::min((total_elements + 255u) / 256u, 1024u);
    auto partials_buf = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);

    auto gamma_buf = std::make_shared<gpu::vector>(context);
    gamma_buf->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);

    // Pass 1: Reduce absmean pass 1
    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS1;
        node.buffers = { w_buf, partials_buf };
        struct PushConsts { uint32_t total; } pcs{ total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = num_workgroups;
        engine.getCurrentGraph().addNode(std::move(node));
    }
    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    // Pass 2: Reduce absmean pass 2
    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS2;
        node.buffers = { partials_buf, gamma_buf };
        struct PushConsts { uint32_t num_partials; uint32_t total; } pcs{ num_workgroups, total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = 1;
        engine.getCurrentGraph().addNode(std::move(node));
    }
    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<float> host_gamma(2, 0.0f);
    gamma_buf->downloadData(host_gamma);
    float gpu_gamma = host_gamma[0];
    float gpu_inv_gamma = host_gamma[1];

    if (std::fabs(gpu_gamma - cpu_gamma) / cpu_gamma > 1e-4f)
    {
        std::cout << "[ERROR: Gamma mismatch: GPU=" << gpu_gamma << ", CPU=" << cpu_gamma << "] ";
        return false;
    }
    if (std::fabs(gpu_inv_gamma - cpu_inv_gamma) / cpu_inv_gamma > 1e-4f)
    {
        std::cout << "[ERROR: InvGamma mismatch: GPU=" << gpu_inv_gamma << ", CPU=" << cpu_inv_gamma << "] ";
        return false;
    }

    // 3. Test FakeQuant: FP32 [K][N] -> FP16 [K][N]
    auto w_eff_buf = std::make_shared<gpu::vector>(context, total_elements, Data_Type::FLOAT16);
    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::FAKEQUANT_TERNARY;
        node.buffers = { w_buf, gamma_buf, w_eff_buf };
        struct PushConsts { uint32_t total; } pcs{ total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (total_elements + 255u) / 256u;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<float16_t> host_w_eff(total_elements);
    w_eff_buf->downloadData(host_w_eff);

    for (size_t i = 0; i < total_elements; ++i)
    {
        float w = host_w[i];
        float q = std::clamp(std::floor(std::fabs(w) * cpu_inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
        float expected = q * cpu_gamma;
        float actual = static_cast<float>(host_w_eff[i]);
        if (std::fabs(actual - expected) > 1e-3f * (std::fabs(expected) + 1.0f))
        {
            std::cout << "[ERROR: FakeQuant mismatch at " << i << ": expected " << expected << ", got " << actual << "] ";
            return false;
        }
    }

    // 4. Test Quantize Pack: FP32 [K][N] -> 2-bit [N][ceil(K/16)]
    uint32_t words_per_row = (K + 15u) / 16u;
    uint32_t total_words = N * words_per_row;
    auto packed_buf = std::make_shared<gpu::vector>(context);
    packed_buf->allocateMemoryByBytes(total_words * sizeof(uint32_t), Data_Type::TERNARY_INT2, total_words);

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::QUANTIZE_PACK_TERNARY;
        node.buffers = { w_buf, gamma_buf, packed_buf };
        struct PushConsts { uint32_t K; uint32_t N; uint32_t words_per_row; } pcs{ K, N, words_per_row };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (words_per_row + 15u) / 16u;
        node.workgroup_count_y = (N + 15u) / 16u;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<uint32_t> gpu_packed(total_words, 0);
    packed_buf->downloadData(gpu_packed);

    // Compute CPU golden reference packing
    std::vector<uint32_t> cpu_packed(total_words, 0);
    for (uint32_t n = 0; n < N; ++n)
    {
        for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
        {
            uint32_t word = 0u;
            for (uint32_t b = 0; b < 16; ++b)
            {
                uint32_t k = w_idx * 16 + b;
                if (k < K)
                {
                    float w = host_w[k * N + n];
                    float q = std::clamp(std::floor(std::fabs(w) * cpu_inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
                    uint32_t code = static_cast<uint32_t>(static_cast<int>(q)) & 3u;
                    word |= (code << (b * 2u));
                }
            }
            cpu_packed[n * words_per_row + w_idx] = word;
        }
    }

    for (size_t i = 0; i < total_words; ++i)
    {
        if (gpu_packed[i] != cpu_packed[i])
        {
            std::cout << "[ERROR: QuantizePack bit-exact mismatch at word " << i << ": GPU=" << gpu_packed[i] << ", CPU=" << cpu_packed[i] << "] ";
            return false;
        }
    }

    // 5. Test Inference T-GEMM: Y = X * W_p^T * gamma + bias
    uint32_t M = 4;
    std::vector<float16_t> host_x(M * K);
    for (size_t i = 0; i < host_x.size(); ++i)
    {
        host_x[i] = static_cast<float16_t>(std::cos(static_cast<float>(i) * 0.2f) * 2.0f);
    }
    std::vector<float16_t> host_bias(N);
    for (size_t i = 0; i < N; ++i)
    {
        host_bias[i] = static_cast<float16_t>(0.1f * static_cast<float>(i % 5));
    }

    auto x_buf = std::make_shared<gpu::vector>(context, M * K, Data_Type::FLOAT16);
    x_buf->uploadData(host_x);

    auto bias_buf = std::make_shared<gpu::vector>(context, N, Data_Type::FLOAT16);
    bias_buf->uploadData(host_bias);

    auto y_buf = std::make_shared<gpu::vector>(context, M * N, Data_Type::FLOAT16);

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::MATMUL_TERNARY_FP16;
        node.buffers = { x_buf, packed_buf, bias_buf, gamma_buf, y_buf };
        struct PushConsts { uint32_t M; uint32_t K; uint32_t N; uint32_t words_per_row; uint32_t has_bias; }
            pcs{ M, K, N, words_per_row, 1u };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (N + 15u) / 16u;
        node.workgroup_count_y = (M + 15u) / 16u;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<float16_t> gpu_y(M * N);
    y_buf->downloadData(gpu_y);

    // Compute CPU golden reference T-GEMM
    for (uint32_t m = 0; m < M; ++m)
    {
        for (uint32_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
            {
                uint32_t pval = cpu_packed[n * words_per_row + w_idx];
                for (uint32_t b = 0; b < 16; ++b)
                {
                    uint32_t k = w_idx * 16 + b;
                    if (k < K)
                    {
                        uint32_t code = (pval >> (b * 2u)) & 3u;
                        int weight = static_cast<int>(code & 1u) * (1 - static_cast<int>(code & 2u));
                        acc += static_cast<float>(host_x[m * K + k]) * static_cast<float>(weight);
                    }
                }
            }
            acc = acc * cpu_gamma + static_cast<float>(host_bias[n]);
            float actual = static_cast<float>(gpu_y[m * N + n]);
            if (std::fabs(actual - acc) > 5e-2f * (std::fabs(acc) + 1.0f))
            {
                std::cout << "[ERROR: TGEMM mismatch at (" << m << ", " << n << "): GPU=" << actual << ", CPU=" << acc << "] ";
                return false;
            }
        }
    }

    return true;
}

bool testTernaryComputingPhase4()
{
    // 1. CPU Layer Forward, Backward & State Machine
    {
        size_t K = 32;
        size_t N = 16;
        size_t M = 4;
        Ternary_Linear_Layer layer(K, N, Execution_Target::CPU);

        auto params_and_grads = layer.getParametersAndGradients();
        if (params_and_grads.size() != 2) return false;
        if (params_and_grads[0].first->getShape() != Shape{ K, N }) return false;
        if (params_and_grads[1].first->getShape() != Shape{ 1, N }) return false;

        std::vector<float> input_data(M * K, 1.0f);
        Tensor in(M, K, input_data, Execution_Target::CPU);
        Tensor out = layer.forward(in);
        if (out.getShape() != Shape{ M, N }) return false;

        std::vector<float> grad_data(M * N, 0.5f);
        Tensor grad_out(M, N, grad_data, Execution_Target::CPU);
        Tensor grad_in = layer.backward(grad_out);
        if (grad_in.getShape() != Shape{ M, K }) return false;

        const auto &w_grad = layer.getWeightsGradient().getData();
        bool has_grad = false;
        for (float g : w_grad)
        {
            if (std::abs(g) > 1e-6f)
            {
                has_grad = true;
                break;
            }
        }
        if (!has_grad) return false;

        // Inference Baking
        layer.prepareInference();
        if (!layer.hasPackedInference()) return false;
        Tensor out_infer = layer.forward(in);
        if (out_infer.getShape() != Shape{ M, N }) return false;

        // State Machine transition: setTrainingMode(true) should reactivate training path
        layer.setTrainingMode(true);
        if (layer.hasPackedInference()) return false;
        Tensor out_train = layer.forward(in);
        if (out_train.getShape() != Shape{ M, N }) return false;

        // Serialization: Checkpoint save & load
        std::string ckpt_path = "test_ternary_ckpt.bin";
        {
            std::ofstream ofs(ckpt_path, std::ios::binary);
            layer.saveCheckpoint(ofs);
        }
        Ternary_Linear_Layer loaded_ckpt(K, N, Execution_Target::CPU);
        {
            std::ifstream ifs(ckpt_path, std::ios::binary);
            loaded_ckpt.loadCheckpoint(ifs);
        }
        std::remove(ckpt_path.c_str());
        if (loaded_ckpt.getWeights().getShape() != Shape{ K, N }) return false;

        // Serialization: Inference save & load
        layer.prepareInference();
        std::string infer_path = "test_ternary_infer.bin";
        {
            std::ofstream ofs(infer_path, std::ios::binary);
            layer.saveInference(ofs);
        }
        Ternary_Linear_Layer loaded_infer(K, N, Execution_Target::CPU);
        {
            std::ifstream ifs(infer_path, std::ios::binary);
            loaded_infer.loadInference(ifs);
        }
        std::remove(infer_path.c_str());
        if (!loaded_infer.hasPackedInference()) return false;
        Tensor out_loaded = loaded_infer.forward(in);
        if (out_loaded.getShape() != Shape{ M, N }) return false;

        // Strict inference mode: backward and setTrainingMode(true) must throw
        bool threw_on_backward = false;
        try
        {
            loaded_infer.backward(grad_out);
        }
        catch (const std::runtime_error &)
        {
            threw_on_backward = true;
        }
        if (!threw_on_backward) return false;

        bool threw_on_train = false;
        try
        {
            loaded_infer.setTrainingMode(true);
        }
        catch (const std::runtime_error &)
        {
            threw_on_train = true;
        }
        if (!threw_on_train) return false;
    }

    // 2. GPU Layer Forward, Backward & State Machine
    {
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            size_t K = 64;
            size_t N = 32;
            size_t M = 4;
            Ternary_Linear_Layer gpu_layer(K, N, Execution_Target::VULKAN_GPU);

            std::vector<float> input_data(M * K, 0.5f);
            Tensor in(M, K, input_data, Execution_Target::VULKAN_GPU);
            Tensor out = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out.getShape() != Shape{ M, N }) return false;

            std::vector<float> grad_data(M * N, 0.2f);
            Tensor grad_out(M, N, grad_data, Execution_Target::VULKAN_GPU);
            Tensor grad_in = gpu_layer.backward(grad_out);
            engine.executeGraph();
            engine.waitIdle();
            if (grad_in.getShape() != Shape{ M, K }) return false;

            gpu_layer.prepareInference();
            engine.executeGraph();
            engine.waitIdle();
            if (!gpu_layer.hasPackedInference()) return false;

            Tensor out_infer = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_infer.getShape() != Shape{ M, N }) return false;

            // GPU Inference Serialization
            std::string gpu_infer_path = "test_ternary_gpu_infer.bin";
            {
                std::ofstream ofs(gpu_infer_path, std::ios::binary);
                gpu_layer.saveInference(ofs);
            }
            Ternary_Linear_Layer loaded_gpu(K, N, Execution_Target::VULKAN_GPU);
            {
                std::ifstream ifs(gpu_infer_path, std::ios::binary);
                loaded_gpu.loadInference(ifs);
            }
            std::remove(gpu_infer_path.c_str());
            if (!loaded_gpu.hasPackedInference()) return false;

            Tensor out_loaded_gpu = loaded_gpu.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_loaded_gpu.getShape() != Shape{ M, N }) return false;
        }
    }

    return true;
}

bool testTernaryComputingPhase5()
{
    // 1. Train -> Pack -> Infer Drift Verification
    {
        size_t K = 128;
        size_t N = 64;
        size_t M = 8;

        // CPU Drift Check
        {
            Ternary_Linear_Layer cpu_layer(K, N, Execution_Target::CPU);
            std::vector<float> in_data(M * K);
            for (size_t i = 0; i < in_data.size(); ++i)
            {
                in_data[i] = std::sin(static_cast<float>(i) * 0.1f);
            }
            Tensor in(M, K, in_data, Execution_Target::CPU);

            Tensor out_fake = cpu_layer.forward(in);
            auto data_fake = out_fake.getData();

            cpu_layer.prepareInference();
            Tensor out_infer = cpu_layer.forward(in);
            auto data_infer = out_infer.getData();

            double diff_sq = 0.0;
            double norm_sq = 0.0;
            for (size_t i = 0; i < data_fake.size(); ++i)
            {
                double d = data_infer[i] - data_fake[i];
                diff_sq += d * d;
                norm_sq += data_fake[i] * data_fake[i];
            }
            double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
            if (rel_err > 1e-3)
            {
                std::cout << "[ERROR: CPU Train -> Infer drift error: " << rel_err << "] ";
                return false;
            }
        }

        // GPU Drift Check
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            Ternary_Linear_Layer gpu_layer(K, N, Execution_Target::VULKAN_GPU);
            std::vector<float> in_data(M * K);
            for (size_t i = 0; i < in_data.size(); ++i)
            {
                in_data[i] = std::sin(static_cast<float>(i) * 0.1f);
            }
            Tensor in(M, K, in_data, Execution_Target::VULKAN_GPU);

            Tensor out_fake = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            auto data_fake = out_fake.getData();

            gpu_layer.prepareInference();
            engine.executeGraph();
            engine.waitIdle();

            Tensor out_infer = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            auto data_infer = out_infer.getData();

            double diff_sq = 0.0;
            double norm_sq = 0.0;
            for (size_t i = 0; i < data_fake.size(); ++i)
            {
                double d = data_infer[i] - data_fake[i];
                diff_sq += d * d;
                norm_sq += data_fake[i] * data_fake[i];
            }
            double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
            if (rel_err > 0.01) // Relative error <= 1%
            {
                std::cout << "[ERROR: GPU Train -> Infer drift error: " << rel_err << "] ";
                return false;
            }
        }
    }

    // 2. Non-multiple of 16 Edge Cases: K in {1, 15, 17, 33}
    {
        std::vector<size_t> test_k = { 1, 15, 17, 33 };
        size_t N = 8;
        size_t M = 2;

        for (size_t K : test_k)
        {
            // Verify byte allocation formula
            size_t expected_words_per_row = (K + 15) / 16;
            size_t expected_bytes = N * expected_words_per_row * sizeof(uint32_t);
            size_t actual_bytes = getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape{ N, K });
            if (actual_bytes != expected_bytes)
            {
                std::cout << "[ERROR: getTensorStorageBytes mismatch for K=" << K << "] ";
                return false;
            }

            // CPU test
            {
                Ternary_Linear_Layer layer(K, N, Execution_Target::CPU);
                std::vector<float> in_data(M * K, 0.3f);
                Tensor in(M, K, in_data, Execution_Target::CPU);
                Tensor out = layer.forward(in);
                if (out.getShape() != Shape{ M, N }) return false;

                std::vector<float> g_data(M * N, 0.1f);
                Tensor g_out(M, N, g_data, Execution_Target::CPU);
                Tensor g_in = layer.backward(g_out);
                if (g_in.getShape() != Shape{ M, K }) return false;

                layer.prepareInference();
                Tensor out_inf = layer.forward(in);
                if (out_inf.getShape() != Shape{ M, N }) return false;
            }

            // GPU test
            auto &engine = Execution_Engine::getInstance();
            auto &context = engine.getContext();
            if (context.isFloat16Supported() && context.isFloat16Enabled())
            {
                Ternary_Linear_Layer layer(K, N, Execution_Target::VULKAN_GPU);
                std::vector<float> in_data(M * K, 0.3f);
                Tensor in(M, K, in_data, Execution_Target::VULKAN_GPU);
                Tensor out = layer.forward(in);
                engine.executeGraph();
                engine.waitIdle();
                if (out.getShape() != Shape{ M, N }) return false;

                std::vector<float> g_data(M * N, 0.1f);
                Tensor g_out(M, N, g_data, Execution_Target::VULKAN_GPU);
                Tensor g_in = layer.backward(g_out);
                engine.executeGraph();
                engine.waitIdle();
                if (g_in.getShape() != Shape{ M, K }) return false;

                layer.prepareInference();
                engine.executeGraph();
                engine.waitIdle();
                Tensor out_inf = layer.forward(in);
                engine.executeGraph();
                engine.waitIdle();
                if (out_inf.getShape() != Shape{ M, N }) return false;
            }
        }
    }

    return true;
}

// ==============================================================================
// TEST SUITE: Dice_BCE_Cost Comprehensive Tests
// ==============================================================================
namespace TestDiceBceCost
{
double computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) return 1e9;
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isnan(a[i]) || std::isnan(b[i])) return 1e9;
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

std::vector<float> generateRandomProbabilities(size_t count, unsigned int seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.05f, 0.95f);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = dist(rng);
    }
    return data;
}

std::vector<float> generateRandomBinaryTargets(size_t count, float pos_ratio = 0.3f, unsigned int seed = 123)
{
    std::mt19937 rng(seed);
    std::bernoulli_distribution dist(pos_ratio);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = dist(rng) ? 1.0f : 0.0f;
    }
    return data;
}

bool testKnownValuesCpu()
{
    std::vector<float> pred_raw = {0.8f, 0.2f, 0.7f, 0.1f};
    std::vector<float> target_raw = {1.0f, 0.0f, 1.0f, 0.0f};

    Tensor pred(1, 4, pred_raw, Execution_Target::CPU);
    Tensor target(1, 4, target_raw, Execution_Target::CPU);

    Dice_BCE_Cost cost_fn(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    float computed_loss = cost_fn.computeLoss(pred, target);

    float bce_0 = -std::log(0.8f);
    float bce_1 = -std::log(0.8f);
    float bce_2 = -std::log(0.7f);
    float bce_3 = -std::log(0.9f);
    float expected_bce = (bce_0 + bce_1 + bce_2 + bce_3) / 4.0f;

    float inter = 0.8f * 1.0f + 0.2f * 0.0f + 0.7f * 1.0f + 0.1f * 0.0f;
    float pred_sum = 0.8f + 0.2f + 0.7f + 0.1f;
    float target_sum = 1.0f + 0.0f + 1.0f + 0.0f;
    float expected_dice = 1.0f - (2.0f * inter + 1e-7f) / (pred_sum + target_sum + 1e-7f);
    float expected_total = 1.0f * expected_dice + 1.0f * expected_bce;

    double diff = std::fabs(computed_loss - expected_total);
    return diff <= 1e-5;
}

bool testNumericalGradientCpu()
{
    size_t count = 16;
    auto pred_raw = generateRandomProbabilities(count, 101);
    auto target_raw = generateRandomBinaryTargets(count, 0.4f, 202);

    Tensor pred(1, count, pred_raw, Execution_Target::CPU);
    Tensor target(1, count, target_raw, Execution_Target::CPU);

    Dice_BCE_Cost cost_fn(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    Tensor analytical_grad = cost_fn.computeGradient(pred, target);
    auto grad_data = analytical_grad.getData();

    float delta = 1e-3f;
    double max_rel_err = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        auto pred_plus = pred_raw;
        auto pred_minus = pred_raw;
        pred_plus[i] += delta;
        pred_minus[i] -= delta;

        Tensor t_plus(1, count, pred_plus, Execution_Target::CPU);
        Tensor t_minus(1, count, pred_minus, Execution_Target::CPU);

        float loss_plus = cost_fn.computeLoss(t_plus, target);
        float loss_minus = cost_fn.computeLoss(t_minus, target);

        float num_grad = (loss_plus - loss_minus) / (2.0f * delta);
        float ana_grad = grad_data[i];

        double abs_err = std::fabs(num_grad - ana_grad);
        double denom = 1.0 + std::max(std::fabs(num_grad), std::fabs(ana_grad));
        double rel_err = abs_err / denom;
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    return max_rel_err <= 1e-2;
}

bool testForwardGpuParity()
{
    size_t count = 512;
    auto pred_raw = generateRandomProbabilities(count, 303);
    auto target_raw = generateRandomBinaryTargets(count, 0.35f, 404);

    Tensor pred_cpu(1, count, pred_raw, Execution_Target::CPU);
    Tensor target_cpu(1, count, target_raw, Execution_Target::CPU);
    Dice_BCE_Cost cost_cpu(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    float loss_cpu = cost_cpu.computeLoss(pred_cpu, target_cpu);

    Tensor pred_gpu(1, count, pred_raw, Execution_Target::VULKAN_GPU);
    Tensor target_gpu(1, count, target_raw, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    float loss_gpu = cost_gpu.computeLoss(pred_gpu, target_gpu);

    double diff = std::fabs(loss_cpu - loss_gpu);
    return diff <= 1e-4;
}

bool testBackwardGpuParity()
{
    size_t count = 512;
    auto pred_raw = generateRandomProbabilities(count, 505);
    auto target_raw = generateRandomBinaryTargets(count, 0.45f, 606);

    Tensor pred_cpu(1, count, pred_raw, Execution_Target::CPU);
    Tensor target_cpu(1, count, target_raw, Execution_Target::CPU);
    Dice_BCE_Cost cost_cpu(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    Tensor grad_cpu = cost_cpu.computeGradient(pred_cpu, target_cpu);

    Tensor pred_gpu(1, count, pred_raw, Execution_Target::VULKAN_GPU);
    Tensor target_gpu(1, count, target_raw, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    Tensor grad_gpu = cost_gpu.computeGradient(pred_gpu, target_gpu);

    auto cpu_data = grad_cpu.getData();
    auto gpu_data = grad_gpu.getData();

    double max_diff = computeMaxAbsDiff(cpu_data, gpu_data);
    return max_diff <= 1e-4;
}

bool testFp16AndCoopmatParity()
{
    size_t count = 1024;
    auto pred_raw = generateRandomProbabilities(count, 707);
    auto target_raw = generateRandomBinaryTargets(count, 0.3f, 808);

    Tensor pred_cpu(1, count, pred_raw, Execution_Target::CPU);
    Tensor target_cpu(1, count, target_raw, Execution_Target::CPU);
    Dice_BCE_Cost cost_cpu(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    float loss_cpu = cost_cpu.computeLoss(pred_cpu, target_cpu);
    Tensor grad_cpu = cost_cpu.computeGradient(pred_cpu, target_cpu);

    Tensor pred_fp16(Shape{1, count}, pred_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor target_fp16(Shape{1, count}, target_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);

    float loss_fp16 = cost_gpu.computeLoss(pred_fp16, target_fp16);
    Tensor grad_fp16 = cost_gpu.computeGradient(pred_fp16, target_fp16);

    double loss_diff = std::fabs(loss_cpu - loss_fp16);
    double grad_diff = computeMaxAbsDiff(grad_cpu.getData(), grad_fp16.getData());

    return (loss_diff <= 2e-3 && grad_diff <= 2e-3);
}

bool testNumericalGradientGpu()
{
    size_t count = 32;
    auto pred_raw = generateRandomProbabilities(count, 909);
    auto target_raw = generateRandomBinaryTargets(count, 0.5f, 1010);

    Tensor pred_gpu(1, count, pred_raw, Execution_Target::VULKAN_GPU);
    Tensor target_gpu(1, count, target_raw, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);

    Tensor ana_grad_gpu = cost_gpu.computeGradient(pred_gpu, target_gpu);
    auto ana_data = ana_grad_gpu.getData();

    float delta = 1e-3f;
    double max_rel_err = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        auto pred_plus = pred_raw;
        auto pred_minus = pred_raw;
        pred_plus[i] += delta;
        pred_minus[i] -= delta;

        Tensor t_plus(1, count, pred_plus, Execution_Target::VULKAN_GPU);
        Tensor t_minus(1, count, pred_minus, Execution_Target::VULKAN_GPU);

        float loss_plus = cost_gpu.computeLoss(t_plus, target_gpu);
        float loss_minus = cost_gpu.computeLoss(t_minus, target_gpu);

        float num_grad = (loss_plus - loss_minus) / (2.0f * delta);
        float ana_grad = ana_data[i];

        double abs_err = std::fabs(num_grad - ana_grad);
        double denom = 1.0 + std::max(std::fabs(num_grad), std::fabs(ana_grad));
        double rel_err = abs_err / denom;
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    return max_rel_err <= 1e-2;
}

bool testWeightsAndEdgeCases()
{
    size_t count = 16;
    std::vector<float> extreme_preds = {0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.0f, 1.0f, 0.5f,
                                        0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.0f, 1.0f, 0.5f};
    std::vector<float> targets =       {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    Tensor p_cpu(1, count, extreme_preds, Execution_Target::CPU);
    Tensor t_cpu(1, count, targets, Execution_Target::CPU);

    Dice_BCE_Cost cost_weighted(0.5f, 2.0f, 1e-7f, Execution_Target::CPU);
    float loss_cpu = cost_weighted.computeLoss(p_cpu, t_cpu);
    Tensor grad_cpu = cost_weighted.computeGradient(p_cpu, t_cpu);

    if (std::isnan(loss_cpu) || std::isinf(loss_cpu)) return false;
    for (float g : grad_cpu.getData())
    {
        if (std::isnan(g) || std::isinf(g)) return false;
    }

    Tensor p_gpu(1, count, extreme_preds, Execution_Target::VULKAN_GPU);
    Tensor t_gpu(1, count, targets, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(0.5f, 2.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    float loss_gpu = cost_gpu.computeLoss(p_gpu, t_gpu);
    Tensor grad_gpu = cost_gpu.computeGradient(p_gpu, t_gpu);

    if (std::isnan(loss_gpu) || std::isinf(loss_gpu)) return false;
    for (float g : grad_gpu.getData())
    {
        if (std::isnan(g) || std::isinf(g)) return false;
    }

    return true;
}

bool testPolymorphicInterface()
{
    std::unique_ptr<ICost_Function> cost = std::make_unique<Dice_BCE_Cost>(0.75f, 1.25f, 1e-5f, Execution_Target::CPU);
    if (cost->getType() != Cost_Type::DICE_BCE) return false;

    size_t count = 32;
    auto pred_raw = generateRandomProbabilities(count, 1111);
    auto target_raw = generateRandomBinaryTargets(count, 0.4f, 2222);

    Tensor p(1, count, pred_raw, Execution_Target::CPU);
    Tensor t(1, count, target_raw, Execution_Target::CPU);
    Tensor g(Execution_Target::CPU);

    float combined_loss = cost->computeLossAndGradient(p, t, g);
    float separate_loss = cost->computeLoss(p, t);
    Tensor separate_g = cost->computeGradient(p, t);

    if (std::fabs(combined_loss - separate_loss) > 1e-6) return false;
    double g_diff = computeMaxAbsDiff(g.getData(), separate_g.getData());
    if (g_diff > 1e-6) return false;

    std::string checkpoint_file = "test_dice_bce_ckpt.bin";
    {
        std::ofstream out(checkpoint_file, std::ios::binary);
        cost->saveCheckpoint(out);
    }
    Dice_BCE_Cost loaded_cost(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    {
        std::ifstream in(checkpoint_file, std::ios::binary);
        loaded_cost.loadCheckpoint(in);
    }
    std::remove(checkpoint_file.c_str());

    if (std::fabs(loaded_cost.getDiceWeight() - 0.75f) > 1e-6 ||
        std::fabs(loaded_cost.getBceWeight() - 1.25f) > 1e-6 ||
        std::fabs(loaded_cost.getEpsilon() - 1e-5f) > 1e-8)
    {
        return false;
    }

    return true;
}

bool runSuite()
{
    bool ok = true;
    ok &= testKnownValuesCpu();
    ok &= testNumericalGradientCpu();
    ok &= testForwardGpuParity();
    ok &= testBackwardGpuParity();
    ok &= testFp16AndCoopmatParity();
    ok &= testNumericalGradientGpu();
    ok &= testWeightsAndEdgeCases();
    ok &= testPolymorphicInterface();
    return ok;
}
} // namespace TestDiceBceCost

// ==============================================================================
// TEST SUITE: Instance_Norm_2d_Layer Comprehensive Tests
// ==============================================================================
namespace TestInstanceNorm
{
double computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) return 1e9;
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isnan(a[i]) || std::isnan(b[i])) return 1e9;
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

std::vector<float> generateRandomData(size_t count, float mean = 0.0f, float stddev = 1.0f)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = dist(rng);
    }
    return data;
}

bool testForwardKnownValues()
{
    uint32_t b_size = 2, h = 2, w = 2, c = 2;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = {
        1.0f, 10.0f,  2.0f, 20.0f,  3.0f, 30.0f,  4.0f, 40.0f,
        5.0f, 50.0f,  6.0f, 60.0f,  7.0f, 70.0f,  8.0f, 80.0f
    };
    std::vector<float> gamma_raw = {1.5f, 0.5f};
    std::vector<float> beta_raw = {0.1f, -0.2f};

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor gamma_cpu(1, c, gamma_raw, Execution_Target::CPU);
    Tensor beta_cpu(1, c, beta_raw, Execution_Target::CPU);
    Tensor saved_mean(Execution_Target::CPU), saved_inv_std(Execution_Target::CPU), norm_in(Execution_Target::CPU), out_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dForward(
        in_cpu, gamma_cpu, beta_cpu,
        saved_mean, saved_inv_std, norm_in, out_cpu,
        h, w, c, epsilon);

    auto mean_data = saved_mean.getData();
    float exp_mean_b0_c0 = (1.0f + 2.0f + 3.0f + 4.0f) / 4.0f;
    float exp_mean_b0_c1 = (10.0f + 20.0f + 30.0f + 40.0f) / 4.0f;
    float exp_mean_b1_c0 = (5.0f + 6.0f + 7.0f + 8.0f) / 4.0f;
    float exp_mean_b1_c1 = (50.0f + 60.0f + 70.0f + 80.0f) / 4.0f;

    if (std::fabs(mean_data[0] - exp_mean_b0_c0) > 1e-4f ||
        std::fabs(mean_data[1] - exp_mean_b0_c1) > 1e-4f ||
        std::fabs(mean_data[2] - exp_mean_b1_c0) > 1e-4f ||
        std::fabs(mean_data[3] - exp_mean_b1_c1) > 1e-4f)
    {
        return false;
    }
    return true;
}

bool testForwardGpuParity()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 2, h = 16, w = 16, c = 32;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomData(b_size * h * w * c, 2.0f, 3.0f);
    std::vector<float> gamma_raw = generateRandomData(c, 1.0f, 0.2f);
    std::vector<float> beta_raw = generateRandomData(c, 0.0f, 0.1f);

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor gamma_cpu(1, c, gamma_raw, Execution_Target::CPU);
    Tensor beta_cpu(1, c, beta_raw, Execution_Target::CPU);
    Tensor mean_cpu(Execution_Target::CPU), inv_std_cpu(Execution_Target::CPU), norm_cpu(Execution_Target::CPU), out_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dForward(in_cpu, gamma_cpu, beta_cpu, mean_cpu, inv_std_cpu, norm_cpu, out_cpu, h, w, c, epsilon);

    Tensor in_gpu(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    Tensor gamma_gpu(1, c, gamma_raw, Execution_Target::VULKAN_GPU);
    Tensor beta_gpu(1, c, beta_raw, Execution_Target::VULKAN_GPU);
    Tensor mean_gpu(Execution_Target::VULKAN_GPU), inv_std_gpu(Execution_Target::VULKAN_GPU), norm_gpu(Execution_Target::VULKAN_GPU), out_gpu(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_gpu, gamma_gpu, beta_gpu, mean_gpu, inv_std_gpu, norm_gpu, out_gpu, h, w, c, epsilon);
    engine.executeGraph();

    double out_err = computeMaxAbsDiff(out_gpu.getData(), out_cpu.getData());
    double mean_err = computeMaxAbsDiff(mean_gpu.getData(), mean_cpu.getData());
    double inv_std_err = computeMaxAbsDiff(inv_std_gpu.getData(), inv_std_cpu.getData());

    return (out_err <= 1e-4 && mean_err <= 1e-4 && inv_std_err <= 1e-4);
}

bool testForwardFp16AndCoopmatParity()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 2, h = 16, w = 16, c = 32;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomData(b_size * h * w * c, 1.0f, 1.5f);
    std::vector<float> gamma_raw = generateRandomData(c, 1.0f, 0.1f);
    std::vector<float> beta_raw = generateRandomData(c, 0.0f, 0.1f);

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor gamma_cpu(1, c, gamma_raw, Execution_Target::CPU);
    Tensor beta_cpu(1, c, beta_raw, Execution_Target::CPU);
    Tensor mean_cpu(Execution_Target::CPU), inv_std_cpu(Execution_Target::CPU), norm_cpu(Execution_Target::CPU), out_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dForward(in_cpu, gamma_cpu, beta_cpu, mean_cpu, inv_std_cpu, norm_cpu, out_cpu, h, w, c, epsilon);

    Tensor in_gpu_fp16(Shape{b_size, h * w * c}, input_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor gamma_gpu(1, c, gamma_raw, Execution_Target::VULKAN_GPU);
    Tensor beta_gpu(1, c, beta_raw, Execution_Target::VULKAN_GPU);
    Tensor mean_gpu(Execution_Target::VULKAN_GPU), inv_std_gpu(Execution_Target::VULKAN_GPU), norm_gpu(Execution_Target::VULKAN_GPU), out_gpu_fp16(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_gpu_fp16, gamma_gpu, beta_gpu, mean_gpu, inv_std_gpu, norm_gpu, out_gpu_fp16, h, w, c, epsilon);
    engine.executeGraph();

    double out_err = computeMaxAbsDiff(out_gpu_fp16.getData(), out_cpu.getData());
    return out_err <= 5e-2;
}

bool testBackwardNumericalGradient()
{
    uint32_t b_size = 2, h = 4, w = 4, c = 3;
    float epsilon = 1e-5f, eps_h = 1e-3f;

    std::vector<float> input_raw = generateRandomData(b_size * h * w * c, 0.5f, 1.0f);
    std::vector<float> gamma_raw = generateRandomData(c, 1.2f, 0.3f);
    std::vector<float> beta_raw = generateRandomData(c, 0.1f, 0.2f);
    std::vector<float> grad_out_raw = generateRandomData(b_size * h * w * c, 0.0f, 0.5f);

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor gamma_cpu(1, c, gamma_raw, Execution_Target::CPU);
    Tensor beta_cpu(1, c, beta_raw, Execution_Target::CPU);
    Tensor mean_cpu(Execution_Target::CPU), inv_std_cpu(Execution_Target::CPU), norm_cpu(Execution_Target::CPU), out_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dForward(in_cpu, gamma_cpu, beta_cpu, mean_cpu, inv_std_cpu, norm_cpu, out_cpu, h, w, c, epsilon);

    Tensor dY_cpu(b_size, h * w * c, grad_out_raw, Execution_Target::CPU);
    Tensor dgamma_cpu(Execution_Target::CPU), dbeta_cpu(Execution_Target::CPU), dX_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dBackward(dY_cpu, norm_cpu, gamma_cpu, inv_std_cpu, dgamma_cpu, dbeta_cpu, dX_cpu, h, w, c, epsilon);

    auto computeLoss = [&](const std::vector<float> &x, const std::vector<float> &g, const std::vector<float> &bt) -> float
    {
        Tensor cur_x(b_size, h * w * c, x, Execution_Target::CPU);
        Tensor cur_g(1, c, g, Execution_Target::CPU);
        Tensor cur_b(1, c, bt, Execution_Target::CPU);
        Tensor m_t(Execution_Target::CPU), inv_t(Execution_Target::CPU), n_t(Execution_Target::CPU), o_t(Execution_Target::CPU);

        ops::norm::instanceNorm2dForward(cur_x, cur_g, cur_b, m_t, inv_t, n_t, o_t, h, w, c, epsilon);
        const auto &out_v = o_t.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < out_v.size(); ++i) loss += out_v[i] * grad_out_raw[i];
        return loss;
    };

    auto ana_dgamma = dgamma_cpu.getData();
    auto ana_dbeta = dbeta_cpu.getData();
    auto ana_dX = dX_cpu.getData();

    double max_rel_gamma_err = 0.0;
    for (uint32_t ch = 0; ch < c; ++ch)
    {
        auto g_plus = gamma_raw, g_minus = gamma_raw;
        g_plus[ch] += eps_h; g_minus[ch] -= eps_h;
        float num_g = (computeLoss(input_raw, g_plus, beta_raw) - computeLoss(input_raw, g_minus, beta_raw)) / (2.0f * eps_h);
        double diff = std::fabs(ana_dgamma[ch] - num_g);
        max_rel_gamma_err = std::max(max_rel_gamma_err, diff / (1.0 + std::max(std::fabs(ana_dgamma[ch]), std::fabs(num_g))));
    }

    double max_rel_beta_err = 0.0;
    for (uint32_t ch = 0; ch < c; ++ch)
    {
        auto b_plus = beta_raw, b_minus = beta_raw;
        b_plus[ch] += eps_h; b_minus[ch] -= eps_h;
        float num_b = (computeLoss(input_raw, gamma_raw, b_plus) - computeLoss(input_raw, gamma_raw, b_minus)) / (2.0f * eps_h);
        double diff = std::fabs(ana_dbeta[ch] - num_b);
        max_rel_beta_err = std::max(max_rel_beta_err, diff / (1.0 + std::max(std::fabs(ana_dbeta[ch]), std::fabs(num_b))));
    }

    double max_rel_x_err = 0.0;
    for (size_t idx = 0; idx < input_raw.size(); ++idx)
    {
        auto x_plus = input_raw, x_minus = input_raw;
        x_plus[idx] += eps_h; x_minus[idx] -= eps_h;
        float num_x = (computeLoss(x_plus, gamma_raw, beta_raw) - computeLoss(x_minus, gamma_raw, beta_raw)) / (2.0f * eps_h);
        double diff = std::fabs(ana_dX[idx] - num_x);
        max_rel_x_err = std::max(max_rel_x_err, diff / (1.0 + std::max(std::fabs(ana_dX[idx]), std::fabs(num_x))));
    }

    return (max_rel_gamma_err <= 1e-2 && max_rel_beta_err <= 1e-2 && max_rel_x_err <= 1e-2);
}

bool testBackwardGpuParity()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 2, h = 8, w = 8, c = 16;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomData(b_size * h * w * c, 1.0f, 2.0f);
    std::vector<float> gamma_raw = generateRandomData(c, 1.0f, 0.2f);
    std::vector<float> beta_raw = generateRandomData(c, 0.0f, 0.1f);
    std::vector<float> grad_out_raw = generateRandomData(b_size * h * w * c, 0.0f, 1.0f);

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor gamma_cpu(1, c, gamma_raw, Execution_Target::CPU);
    Tensor beta_cpu(1, c, beta_raw, Execution_Target::CPU);
    Tensor mean_cpu(Execution_Target::CPU), inv_std_cpu(Execution_Target::CPU), norm_cpu(Execution_Target::CPU), out_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dForward(in_cpu, gamma_cpu, beta_cpu, mean_cpu, inv_std_cpu, norm_cpu, out_cpu, h, w, c, epsilon);

    Tensor dY_cpu(b_size, h * w * c, grad_out_raw, Execution_Target::CPU);
    Tensor dgamma_cpu(Execution_Target::CPU), dbeta_cpu(Execution_Target::CPU), dX_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dBackward(dY_cpu, norm_cpu, gamma_cpu, inv_std_cpu, dgamma_cpu, dbeta_cpu, dX_cpu, h, w, c, epsilon);

    Tensor in_gpu(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    Tensor gamma_gpu(1, c, gamma_raw, Execution_Target::VULKAN_GPU);
    Tensor beta_gpu(1, c, beta_raw, Execution_Target::VULKAN_GPU);
    Tensor mean_gpu(Execution_Target::VULKAN_GPU), inv_std_gpu(Execution_Target::VULKAN_GPU), norm_gpu(Execution_Target::VULKAN_GPU), out_gpu(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_gpu, gamma_gpu, beta_gpu, mean_gpu, inv_std_gpu, norm_gpu, out_gpu, h, w, c, epsilon);

    Tensor dY_gpu(b_size, h * w * c, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor dgamma_gpu(Execution_Target::VULKAN_GPU), dbeta_gpu(Execution_Target::VULKAN_GPU), dX_gpu(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dBackward(dY_gpu, norm_gpu, gamma_gpu, inv_std_gpu, dgamma_gpu, dbeta_gpu, dX_gpu, h, w, c, epsilon);
    engine.executeGraph();

    double dx_err = computeMaxAbsDiff(dX_gpu.getData(), dX_cpu.getData());
    double dg_err = computeMaxAbsDiff(dgamma_gpu.getData(), dgamma_cpu.getData());
    double db_err = computeMaxAbsDiff(dbeta_gpu.getData(), dbeta_cpu.getData());

    return (dx_err <= 1e-4 && dg_err <= 1e-4 && db_err <= 1e-4);
}

bool testBackwardGpuFp16AndCoopmatParity()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 2, h = 8, w = 8, c = 32;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomData(b_size * h * w * c, 1.0f, 1.0f);
    std::vector<float> gamma_raw = generateRandomData(c, 1.0f, 0.2f);
    std::vector<float> beta_raw = generateRandomData(c, 0.0f, 0.1f);
    std::vector<float> grad_out_raw = generateRandomData(b_size * h * w * c, 0.0f, 1.0f);

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor gamma_cpu(1, c, gamma_raw, Execution_Target::CPU);
    Tensor beta_cpu(1, c, beta_raw, Execution_Target::CPU);
    Tensor mean_cpu(Execution_Target::CPU), inv_std_cpu(Execution_Target::CPU), norm_cpu(Execution_Target::CPU), out_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dForward(in_cpu, gamma_cpu, beta_cpu, mean_cpu, inv_std_cpu, norm_cpu, out_cpu, h, w, c, epsilon);

    Tensor dY_cpu(b_size, h * w * c, grad_out_raw, Execution_Target::CPU);
    Tensor dgamma_cpu(Execution_Target::CPU), dbeta_cpu(Execution_Target::CPU), dX_cpu(Execution_Target::CPU);

    ops::norm::instanceNorm2dBackward(dY_cpu, norm_cpu, gamma_cpu, inv_std_cpu, dgamma_cpu, dbeta_cpu, dX_cpu, h, w, c, epsilon);

    Tensor in_gpu(Shape{b_size, h * w * c}, input_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor gamma_gpu(1, c, gamma_raw, Execution_Target::VULKAN_GPU);
    Tensor beta_gpu(1, c, beta_raw, Execution_Target::VULKAN_GPU);
    Tensor mean_gpu(Execution_Target::VULKAN_GPU), inv_std_gpu(Execution_Target::VULKAN_GPU), norm_gpu(Execution_Target::VULKAN_GPU), out_gpu(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_gpu, gamma_gpu, beta_gpu, mean_gpu, inv_std_gpu, norm_gpu, out_gpu, h, w, c, epsilon);

    Tensor dY_gpu(Shape{b_size, h * w * c}, grad_out_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor dgamma_gpu(Execution_Target::VULKAN_GPU), dbeta_gpu(Execution_Target::VULKAN_GPU), dX_gpu(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dBackward(dY_gpu, norm_gpu, gamma_gpu, inv_std_gpu, dgamma_gpu, dbeta_gpu, dX_gpu, h, w, c, epsilon);
    engine.executeGraph();

    double dx_err = computeMaxAbsDiff(dX_gpu.getData(), dX_cpu.getData());
    double dg_err = computeMaxAbsDiff(dgamma_gpu.getData(), dgamma_cpu.getData());
    double db_err = computeMaxAbsDiff(dbeta_gpu.getData(), dbeta_cpu.getData());

    return (dx_err <= 5e-2 && dg_err <= 5e-2 && db_err <= 5e-2);
}

bool testBatchInvariance()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t h = 8, w = 8, c = 16;
    float epsilon = 1e-5f;
    uint32_t spatial_feat = h * w * c;

    std::vector<float> sample_a = generateRandomData(spatial_feat, 1.0f, 2.0f);
    std::vector<float> sample_b = generateRandomData(spatial_feat, 5.0f, 10.0f);

    std::vector<float> batch_ab = sample_a;
    batch_ab.insert(batch_ab.end(), sample_b.begin(), sample_b.end());

    std::vector<float> gamma_raw = generateRandomData(c, 1.0f, 0.1f);
    std::vector<float> beta_raw = generateRandomData(c, 0.0f, 0.1f);

    Tensor in_a_gpu(1, spatial_feat, sample_a, Execution_Target::VULKAN_GPU);
    Tensor gamma_gpu(1, c, gamma_raw, Execution_Target::VULKAN_GPU);
    Tensor beta_gpu(1, c, beta_raw, Execution_Target::VULKAN_GPU);
    Tensor mean_a(Execution_Target::VULKAN_GPU), inv_a(Execution_Target::VULKAN_GPU), norm_a(Execution_Target::VULKAN_GPU), out_a(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_a_gpu, gamma_gpu, beta_gpu, mean_a, inv_a, norm_a, out_a, h, w, c, epsilon);

    Tensor in_ab_gpu(2, spatial_feat, batch_ab, Execution_Target::VULKAN_GPU);
    Tensor mean_ab(Execution_Target::VULKAN_GPU), inv_ab(Execution_Target::VULKAN_GPU), norm_ab(Execution_Target::VULKAN_GPU), out_ab(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_ab_gpu, gamma_gpu, beta_gpu, mean_ab, inv_ab, norm_ab, out_ab, h, w, c, epsilon);
    engine.executeGraph();

    auto out_a_data = out_a.getData();
    auto out_ab_data = out_ab.getData();

    std::vector<float> out_ab_slice_a(out_ab_data.begin(), out_ab_data.begin() + spatial_feat);
    double diff = computeMaxAbsDiff(out_a_data, out_ab_slice_a);

    return diff <= 1e-5;
}

bool testLayerTrainEvalConsistency()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 2, h = 8, w = 8, c = 16;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomData(b_size * h * w * c, 2.0f, 1.0f);

    Instance_Norm_2d_Layer layer(h, w, c, epsilon, Execution_Target::VULKAN_GPU);

    Tensor in_train(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    layer.setTrainingMode(true);
    Tensor out_train = layer.forward(in_train);
    engine.executeGraph();
    auto train_data = out_train.getData();

    Tensor in_eval(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    layer.setTrainingMode(false);
    Tensor out_eval = layer.forward(in_eval);
    engine.executeGraph();
    auto eval_data = out_eval.getData();

    double diff = computeMaxAbsDiff(train_data, eval_data);
    return diff <= 1e-5;
}

bool runSuite()
{
    bool ok = true;
    ok &= testForwardKnownValues();
    ok &= testForwardGpuParity();
    ok &= testForwardFp16AndCoopmatParity();
    ok &= testBackwardNumericalGradient();
    ok &= testBackwardGpuParity();
    ok &= testBackwardGpuFp16AndCoopmatParity();
    ok &= testBatchInvariance();
    ok &= testLayerTrainEvalConsistency();
    return ok;
}
} // namespace TestInstanceNorm

// ==============================================================================
// TEST SUITE: Leaky_Relu_Layer Comprehensive Tests
// ==============================================================================
namespace TestLeakyRelu
{
double computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) return 1e9;
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isnan(a[i]) || std::isnan(b[i])) return 1e9;
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

std::vector<float> generateRandomData(size_t count, float mean = 0.0f, float stddev = 1.0f, unsigned int seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        float val = dist(rng);
        if (std::fabs(val) < 1e-3f) val = (val >= 0.0f) ? 1e-3f : -1e-3f;
        data[i] = val;
    }
    return data;
}

bool testForwardKnownValues()
{
    float slope = 0.01f;
    std::vector<float> input_raw = {-2.0f, -0.5f, 0.0f, 1.5f, 3.0f};
    std::vector<float> expected_raw = {-0.02f, -0.005f, 0.0f, 1.5f, 3.0f};

    Tensor in_cpu(1, 5, input_raw, Execution_Target::CPU);
    Tensor out_cpu(Execution_Target::CPU);

    ops::activation::leakyRelu(in_cpu, out_cpu, slope);
    auto out_data = out_cpu.getData();

    double diff = computeMaxAbsDiff(out_data, expected_raw);
    return diff <= 1e-6;
}

bool testBackwardNumericalGradientCpu()
{
    size_t count = 32;
    float slope = 0.01f;
    float delta = 1e-3f;

    auto input_raw = generateRandomData(count, 0.0f, 1.5f, 101);
    auto grad_out_raw = generateRandomData(count, 0.0f, 1.0f, 202);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor grad_out_cpu(1, count, grad_out_raw, Execution_Target::CPU);
    Tensor grad_in_cpu(Execution_Target::CPU);

    ops::activation::leakyReluBackward(in_cpu, grad_out_cpu, grad_in_cpu, slope);
    auto ana_grad = grad_in_cpu.getData();

    auto computeScalarLoss = [&](const std::vector<float> &x) -> float
    {
        Tensor cur_x(1, count, x, Execution_Target::CPU);
        Tensor cur_y(Execution_Target::CPU);
        ops::activation::leakyRelu(cur_x, cur_y, slope);
        const auto &y_data = cur_y.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < count; ++i) loss += y_data[i] * grad_out_raw[i];
        return loss;
    };

    double max_rel_err = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        auto x_plus = input_raw, x_minus = input_raw;
        x_plus[i] += delta; x_minus[i] -= delta;
        float num_grad = (computeScalarLoss(x_plus) - computeScalarLoss(x_minus)) / (2.0f * delta);
        float ana_val = ana_grad[i];
        double abs_err = std::fabs(num_grad - ana_val);
        double denom = 1.0 + std::max(std::fabs(num_grad), std::fabs(ana_val));
        max_rel_err = std::max(max_rel_err, abs_err / denom);
    }

    return max_rel_err <= 1e-2;
}

bool testForwardGpuParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomData(count, 0.0f, 2.0f, 303);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor out_cpu(Execution_Target::CPU);
    ops::activation::leakyRelu(in_cpu, out_cpu, slope);

    Tensor in_gpu(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor out_gpu(Execution_Target::VULKAN_GPU);
    ops::activation::leakyRelu(in_gpu, out_gpu, slope);

    Execution_Engine::getInstance().executeGraph();

    double diff = computeMaxAbsDiff(out_cpu.getData(), out_gpu.getData());
    return diff <= 1e-5;
}

bool testBackwardGpuParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomData(count, 0.0f, 2.0f, 404);
    auto grad_out_raw = generateRandomData(count, 0.0f, 1.0f, 505);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor grad_out_cpu(1, count, grad_out_raw, Execution_Target::CPU);
    Tensor grad_in_cpu(Execution_Target::CPU);
    ops::activation::leakyReluBackward(in_cpu, grad_out_cpu, grad_in_cpu, slope);

    Tensor in_gpu(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_out_gpu(1, count, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_in_gpu(Execution_Target::VULKAN_GPU);
    ops::activation::leakyReluBackward(in_gpu, grad_out_gpu, grad_in_gpu, slope);

    Execution_Engine::getInstance().executeGraph();

    double diff = computeMaxAbsDiff(grad_in_cpu.getData(), grad_in_gpu.getData());
    return diff <= 1e-5;
}

bool testForwardFp16AndCoopmatParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomData(count, 0.0f, 2.0f, 606);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor out_cpu(Execution_Target::CPU);
    ops::activation::leakyRelu(in_cpu, out_cpu, slope);

    Tensor in_gpu_fp16(Shape{1, count}, input_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor out_gpu_fp16(Execution_Target::VULKAN_GPU);
    ops::activation::leakyRelu(in_gpu_fp16, out_gpu_fp16, slope);

    Execution_Engine::getInstance().executeGraph();

    double diff = computeMaxAbsDiff(out_cpu.getData(), out_gpu_fp16.getData());
    return diff <= 2e-3;
}

bool testBackwardFp16AndCoopmatParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomData(count, 0.0f, 2.0f, 707);
    auto grad_out_raw = generateRandomData(count, 0.0f, 1.0f, 808);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor grad_out_cpu(1, count, grad_out_raw, Execution_Target::CPU);
    Tensor grad_in_cpu(Execution_Target::CPU);
    ops::activation::leakyReluBackward(in_cpu, grad_out_cpu, grad_in_cpu, slope);

    Tensor in_gpu_fp16(Shape{1, count}, input_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor grad_out_gpu_fp16(Shape{1, count}, grad_out_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor grad_in_gpu_fp16(Execution_Target::VULKAN_GPU);
    ops::activation::leakyReluBackward(in_gpu_fp16, grad_out_gpu_fp16, grad_in_gpu_fp16, slope);

    Execution_Engine::getInstance().executeGraph();

    double diff = computeMaxAbsDiff(grad_in_cpu.getData(), grad_in_gpu_fp16.getData());
    return diff <= 2e-3;
}

bool testNumericalGradientGpu()
{
    size_t count = 32;
    float slope = 0.01f;
    float delta = 1e-3f;

    auto input_raw = generateRandomData(count, 0.0f, 1.5f, 909);
    auto grad_out_raw = generateRandomData(count, 0.0f, 1.0f, 1010);

    Tensor in_gpu(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_out_gpu(1, count, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_in_gpu(Execution_Target::VULKAN_GPU);

    ops::activation::leakyReluBackward(in_gpu, grad_out_gpu, grad_in_gpu, slope);
    Execution_Engine::getInstance().executeGraph();
    auto ana_grad = grad_in_gpu.getData();

    auto computeGpuLoss = [&](const std::vector<float> &x) -> float
    {
        Tensor cur_x(1, count, x, Execution_Target::VULKAN_GPU);
        Tensor cur_y(Execution_Target::VULKAN_GPU);
        ops::activation::leakyRelu(cur_x, cur_y, slope);
        Execution_Engine::getInstance().executeGraph();
        const auto &y_data = cur_y.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < count; ++i) loss += y_data[i] * grad_out_raw[i];
        return loss;
    };

    double max_rel_err = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        auto x_plus = input_raw, x_minus = input_raw;
        x_plus[i] += delta; x_minus[i] -= delta;
        float num_grad = (computeGpuLoss(x_plus) - computeGpuLoss(x_minus)) / (2.0f * delta);
        float ana_val = ana_grad[i];
        double abs_err = std::fabs(num_grad - ana_val);
        double denom = 1.0 + std::max(std::fabs(num_grad), std::fabs(ana_val));
        max_rel_err = std::max(max_rel_err, abs_err / denom);
    }

    return max_rel_err <= 1e-2;
}

bool testLayerIntegrationAndSerialization()
{
    float slope = 0.05f;
    size_t count = 64;
    auto input_raw = generateRandomData(count, 0.0f, 1.0f, 1111);
    auto grad_out_raw = generateRandomData(count, 0.0f, 1.0f, 2222);

    Leaky_Relu_Layer layer(slope, Execution_Target::VULKAN_GPU);
    if (layer.getLayerType() != Layer_Type::LEAKY_RELU) return false;

    Tensor in_t(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor out_t = layer.forward(in_t);
    Execution_Engine::getInstance().executeGraph();

    Tensor dy_t(1, count, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor dx_t = layer.backward(dy_t);
    Execution_Engine::getInstance().executeGraph();

    if (out_t.getRows() != 1 || out_t.getColumns() != count || dx_t.getColumns() != count) return false;

    auto cloned = layer.clone();
    auto cloned_leaky = dynamic_cast<Leaky_Relu_Layer *>(cloned.get());
    if (!cloned_leaky || std::fabs(cloned_leaky->getNegativeSlope() - slope) > 1e-6) return false;

    std::string ckpt_name = "test_leaky_relu_ckpt.bin";
    {
        std::ofstream out(ckpt_name, std::ios::binary);
        layer.saveConfiguration(out);
    }

    Leaky_Relu_Layer loaded_layer(0.0f, Execution_Target::CPU);
    {
        std::ifstream in(ckpt_name, std::ios::binary);
        loaded_layer.loadInference(in);
    }
    std::remove(ckpt_name.c_str());

    if (std::fabs(loaded_layer.getNegativeSlope() - slope) > 1e-6) return false;

    return true;
}

bool runSuite()
{
    bool ok = true;
    ok &= testForwardKnownValues();
    ok &= testBackwardNumericalGradientCpu();
    ok &= testForwardGpuParity();
    ok &= testBackwardGpuParity();
    ok &= testForwardFp16AndCoopmatParity();
    ok &= testBackwardFp16AndCoopmatParity();
    ok &= testNumericalGradientGpu();
    ok &= testLayerIntegrationAndSerialization();
    return ok;
}
} // namespace TestLeakyRelu

// ==============================================================================
// TEST SUITE: Up_Sample_2d_Layer Comprehensive Tests
// ==============================================================================
namespace TestUpsample
{
double computeMaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return 1e9;
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isnan(a[i]) || std::isnan(b[i])) return 1e9;
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

bool testForwardSmallGrid()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 1, h = 2, w = 2, c = 2, scale = 2;
    std::vector<float> input_raw = {
        1.0f, 2.0f,   3.0f, 4.0f,
        5.0f, 6.0f,   7.0f, 8.0f
    };

    std::vector<float> expected(1 * 4 * 4 * 2, 0.0f);
    for (uint32_t ih = 0; ih < h; ++ih)
    {
        for (uint32_t iw = 0; iw < w; ++iw)
        {
            for (uint32_t ch = 0; ch < c; ++ch)
            {
                float val = input_raw[ih * w * c + iw * c + ch];
                for (uint32_t sy = 0; sy < scale; ++sy)
                {
                    uint32_t oh = ih * scale + sy;
                    for (uint32_t sx = 0; sx < scale; ++sx)
                    {
                        uint32_t ow = iw * scale + sx;
                        expected[oh * 4 * c + ow * c + ch] = val;
                    }
                }
            }
        }
    }

    Tensor in_cpu(b_size, h * w * c, input_raw, Execution_Target::CPU);
    Tensor out_cpu(Execution_Target::CPU);
    ops::vision::upsample2d(in_cpu, out_cpu, h, w, c, scale);

    double cpu_err = computeMaxAbsDiff(out_cpu.getData(), expected);
    if (cpu_err > 1e-5) return false;

    Tensor in_gpu_fp32(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    Tensor out_gpu_fp32(Execution_Target::VULKAN_GPU);
    ops::vision::upsample2d(in_gpu_fp32, out_gpu_fp32, h, w, c, scale);
    engine.executeGraph();

    double gpu_fp32_err = computeMaxAbsDiff(out_gpu_fp32.getData(), expected);
    if (gpu_fp32_err > 1e-4) return false;

    Tensor in_gpu_fp16(Shape{b_size, h * w * c}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    in_gpu_fp32.to(Data_Type::FLOAT16, in_gpu_fp16);
    engine.executeGraph();

    Tensor out_gpu_fp16(Execution_Target::VULKAN_GPU);
    ops::vision::upsample2d(in_gpu_fp16, out_gpu_fp16, h, w, c, scale);
    engine.executeGraph();

    double gpu_fp16_err = computeMaxAbsDiff(out_gpu_fp16.getData(), expected);
    return gpu_fp16_err <= 1e-2;
}

bool testBackwardSmallGrid()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t b_size = 1, h = 2, w = 2, c = 2, scale = 2;
    uint32_t out_h = h * scale;
    uint32_t out_w = w * scale;

    std::vector<float> grad_out_raw(b_size * out_h * out_w * c);
    for (size_t i = 0; i < grad_out_raw.size(); ++i)
    {
        grad_out_raw[i] = static_cast<float>(i + 1);
    }

    std::vector<float> expected_in_grad(b_size * h * w * c, 0.0f);
    for (uint32_t ih = 0; ih < h; ++ih)
    {
        for (uint32_t iw = 0; iw < w; ++iw)
        {
            for (uint32_t ch = 0; ch < c; ++ch)
            {
                float sum = 0.0f;
                for (uint32_t sy = 0; sy < scale; ++sy)
                {
                    uint32_t oh = ih * scale + sy;
                    for (uint32_t sx = 0; sx < scale; ++sx)
                    {
                        uint32_t ow = iw * scale + sx;
                        sum += grad_out_raw[oh * out_w * c + ow * c + ch];
                    }
                }
                expected_in_grad[ih * w * c + iw * c + ch] = sum;
            }
        }
    }

    Tensor grad_out_cpu(b_size, out_h * out_w * c, grad_out_raw, Execution_Target::CPU);
    Tensor in_grad_cpu(Execution_Target::CPU);
    ops::vision::upsample2dBackward(grad_out_cpu, in_grad_cpu, h, w, c, scale);

    double cpu_err = computeMaxAbsDiff(in_grad_cpu.getData(), expected_in_grad);
    if (cpu_err > 1e-5) return false;

    Tensor grad_out_gpu_fp32(b_size, out_h * out_w * c, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor in_grad_gpu_fp32(Execution_Target::VULKAN_GPU);
    ops::vision::upsample2dBackward(grad_out_gpu_fp32, in_grad_gpu_fp32, h, w, c, scale);
    engine.executeGraph();

    double gpu_fp32_err = computeMaxAbsDiff(in_grad_gpu_fp32.getData(), expected_in_grad);
    if (gpu_fp32_err > 1e-4) return false;

    Tensor grad_out_gpu_fp16(Shape{b_size, out_h * out_w * c}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    grad_out_gpu_fp32.to(Data_Type::FLOAT16, grad_out_gpu_fp16);
    engine.executeGraph();

    Tensor in_grad_gpu_fp16(Execution_Target::VULKAN_GPU);
    ops::vision::upsample2dBackward(grad_out_gpu_fp16, in_grad_gpu_fp16, h, w, c, scale);
    engine.executeGraph();

    double gpu_fp16_err = computeMaxAbsDiff(in_grad_gpu_fp16.getData(), expected_in_grad);
    return gpu_fp16_err <= 1e-2;
}

bool testNumericalGradientCheck()
{
    uint32_t b_size = 2, h = 3, w = 3, c = 4, scale = 2;
    uint32_t out_h = h * scale, out_w = w * scale;
    size_t in_features = h * w * c, out_features = out_h * out_w * c;
    size_t in_elems = b_size * in_features, out_elems = b_size * out_features;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> x_raw(in_elems);
    for (float &v : x_raw) v = dist(rng);

    std::vector<float> weight_raw(out_elems);
    for (float &v : weight_raw) v = dist(rng);

    Tensor grad_out_tensor(b_size, out_features, weight_raw, Execution_Target::CPU);
    Tensor analytical_grad(Execution_Target::CPU);
    ops::vision::upsample2dBackward(grad_out_tensor, analytical_grad, h, w, c, scale);
    auto analytical_data = analytical_grad.getData();

    const float eps = 1e-3f;
    std::vector<float> numerical_grad(in_elems, 0.0f);

    auto compute_loss = [&](const std::vector<float>& x_vec) -> float
    {
        Tensor x_t(b_size, in_features, x_vec, Execution_Target::CPU);
        Tensor y_t(Execution_Target::CPU);
        ops::vision::upsample2d(x_t, y_t, h, w, c, scale);
        auto y_data = y_t.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < out_elems; ++i) loss += y_data[i] * weight_raw[i];
        return loss;
    };

    double grad_norm_diff = 0.0, grad_norm_sum = 0.0;
    for (size_t i = 0; i < in_elems; ++i)
    {
        std::vector<float> x_pos = x_raw, x_neg = x_raw;
        x_pos[i] += eps; x_neg[i] -= eps;

        float loss_pos = compute_loss(x_pos);
        float loss_neg = compute_loss(x_neg);

        float num_g = (loss_pos - loss_neg) / (2.0f * eps);
        numerical_grad[i] = num_g;

        double diff = static_cast<double>(analytical_data[i] - num_g);
        grad_norm_diff += diff * diff;
        grad_norm_sum += static_cast<double>(analytical_data[i] * analytical_data[i] + num_g * num_g);
    }

    double relative_error = std::sqrt(grad_norm_diff) / (std::sqrt(grad_norm_sum) + 1e-8);
    return relative_error <= 1e-3;
}

bool testUnetScales(Execution_Target target, bool mixed_precision)
{
    auto &engine = Execution_Engine::getInstance();
    struct ScaleConfig {
        std::string name;
        uint32_t h, w, c;
    };

    std::vector<ScaleConfig> configs = {
        {"Level 3 (Bottleneck -> Dec3)", 40, 40, 64},
        {"Level 2 (Dec3 -> Dec2)",       80, 80, 32},
        {"Level 1 (Dec2 -> Dec1)",      160, 160, 16}
    };

    uint32_t b_size = 2;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto &cfg : configs)
    {
        uint32_t out_h = cfg.h * 2, out_w = cfg.w * 2;
        size_t in_features = cfg.h * cfg.w * cfg.c, out_features = out_h * out_w * cfg.c;
        size_t in_size = b_size * in_features, out_size = b_size * out_features;

        std::vector<float> in_data(in_size);
        for (float &v : in_data) v = dist(rng);

        std::vector<float> grad_out_data(out_size);
        for (float &v : grad_out_data) v = dist(rng);

        Tensor in_cpu(b_size, in_features, in_data, Execution_Target::CPU);
        Tensor out_cpu(Execution_Target::CPU);
        ops::vision::upsample2d(in_cpu, out_cpu, cfg.h, cfg.w, cfg.c, 2);
        auto ref_out = out_cpu.getData();

        Tensor grad_out_cpu(b_size, out_features, grad_out_data, Execution_Target::CPU);
        Tensor in_grad_cpu(Execution_Target::CPU);
        ops::vision::upsample2dBackward(grad_out_cpu, in_grad_cpu, cfg.h, cfg.w, cfg.c, 2);
        auto ref_in_grad = in_grad_cpu.getData();

        Up_Sample_2d_Layer layer(cfg.h, cfg.w, cfg.c, 2, target);
        if (mixed_precision) layer.setMixedPrecision(true);

        Tensor input_tensor;
        if (target == Execution_Target::VULKAN_GPU && mixed_precision)
        {
            Tensor in_fp32(b_size, in_features, in_data, Execution_Target::VULKAN_GPU);
            input_tensor = Tensor(Shape{b_size, in_features}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
            in_fp32.to(Data_Type::FLOAT16, input_tensor);
        }
        else
        {
            input_tensor = Tensor(b_size, in_features, in_data, target);
        }

        Tensor out_tensor = layer.forward(input_tensor);
        if (target == Execution_Target::VULKAN_GPU) engine.executeGraph();

        auto test_out = out_tensor.getData();
        double fwd_err = computeMaxAbsDiff(test_out, ref_out);

        int fwd_nans = 0;
        for (float v : test_out) if (std::isnan(v)) fwd_nans++;

        Tensor grad_out_tensor;
        if (target == Execution_Target::VULKAN_GPU && mixed_precision)
        {
            Tensor grad_fp32(b_size, out_features, grad_out_data, Execution_Target::VULKAN_GPU);
            grad_out_tensor = Tensor(Shape{b_size, out_features}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
            grad_fp32.to(Data_Type::FLOAT16, grad_out_tensor);
        }
        else
        {
            grad_out_tensor = Tensor(b_size, out_features, grad_out_data, target);
        }

        Tensor grad_in_tensor = layer.backward(grad_out_tensor);
        if (target == Execution_Target::VULKAN_GPU) engine.executeGraph();

        auto test_in_grad = grad_in_tensor.getData();
        double bwd_err = computeMaxAbsDiff(test_in_grad, ref_in_grad);

        int bwd_nans = 0;
        for (float v : test_in_grad) if (std::isnan(v)) bwd_nans++;

        double tol = mixed_precision ? 1e-2 : 1e-4;
        bool pass = (fwd_err <= tol && bwd_err <= tol && fwd_nans == 0 && bwd_nans == 0);
        if (!pass) return false;
    }
    return true;
}

bool testLayerContracts()
{
    Up_Sample_2d_Layer layer(16, 16, 8, 2, Execution_Target::VULKAN_GPU);
    layer.setMixedPrecision(true);

    auto clone_ptr = layer.clone();
    if (!clone_ptr) return false;

    auto *cloned_layer = dynamic_cast<Up_Sample_2d_Layer*>(clone_ptr.get());
    if (!cloned_layer) return false;

    if (cloned_layer->getScaleFactor() != 2 ||
        cloned_layer->getInputHeight() != 16 ||
        cloned_layer->getInputWidth() != 16 ||
        cloned_layer->getChannels() != 8)
    {
        return false;
    }

    layer.resetGradient();
    return true;
}

bool runSuite()
{
    bool ok = true;
    ok &= testForwardSmallGrid();
    ok &= testBackwardSmallGrid();
    ok &= testNumericalGradientCheck();
    ok &= testUnetScales(Execution_Target::CPU, false);
    ok &= testUnetScales(Execution_Target::VULKAN_GPU, false);
    ok &= testUnetScales(Execution_Target::VULKAN_GPU, true);
    ok &= testLayerContracts();
    return ok;
}
} // namespace TestUpsample

// ==============================================================================
// TEST SUITE: NN_UNet_Layer Comprehensive Tests
// ==============================================================================
namespace TestNnUnet
{
double computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) return 1e9;
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isnan(a[i]) || std::isnan(b[i])) return 1e9;
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

std::vector<float> generateRandomData(size_t count, float mean = 0.0f, float stddev = 1.0f, unsigned int seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = dist(rng);
    }
    return data;
}

bool testForwardShapeAndOutputRange()
{
    uint32_t h = 32, w = 32, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 101);
    Tensor in_tensor(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_tensor.uploadData(in_data);

    Tensor out_tensor = layer.forward(in_tensor);
    if (out_tensor.getTotalElements() != b * out_c * h * w) return false;

    auto out_data = out_tensor.getData();
    float min_val = 1e9f, max_val = -1e9f;
    for (float v : out_data)
    {
        if (std::isnan(v) || std::isinf(v)) return false;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    if (min_val < 0.0f || max_val > 1.0f) return false;
    return true;
}

bool testTrainEvalConsistency()
{
    uint32_t h = 32, w = 32, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 202);
    Tensor in_tensor(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_tensor.uploadData(in_data);

    layer.setTrainingMode(true);
    Tensor out_train = layer.forward(in_tensor);
    auto train_data = out_train.getData();

    layer.setTrainingMode(false);
    Tensor out_eval = layer.forward(in_tensor);
    auto eval_data = out_eval.getData();

    double diff = computeMaxAbsDiff(train_data, eval_data);
    return diff <= 1e-6;
}

bool testBatchInvarianceParity()
{
    uint32_t h = 32, w = 32, in_c = 1, out_c = 1, base_c = 4;
    size_t sample_size = in_c * h * w;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    std::vector<float> sample0 = generateRandomData(sample_size, 0.5f, 1.0f, 303);
    std::vector<float> sample1 = generateRandomData(sample_size, -1.0f, 2.5f, 404);

    Tensor in_single(Shape{1, sample_size}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_single.uploadData(sample0);
    Tensor out_single = layer.forward(in_single);
    auto single_data = out_single.getData();

    std::vector<float> batched_input;
    batched_input.insert(batched_input.end(), sample0.begin(), sample0.end());
    batched_input.insert(batched_input.end(), sample1.begin(), sample1.end());

    Tensor in_batched(Shape{2, sample_size}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_batched.uploadData(batched_input);
    Tensor out_batched = layer.forward(in_batched);
    auto batched_data = out_batched.getData();

    size_t out_sample_size = out_c * h * w;
    std::vector<float> batched_sample0(batched_data.begin(), batched_data.begin() + out_sample_size);

    double diff = computeMaxAbsDiff(single_data, batched_sample0);
    return diff <= 1e-5;
}

bool testBackwardGradientFlowCpu()
{
    uint32_t h = 16, w = 16, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 505);
    Tensor in_tensor(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_tensor.uploadData(in_data);

    layer.forward(in_tensor);

    std::vector<float> grad_out_data = generateRandomData(b * out_c * h * w, 0.0f, 0.5f, 606);
    Tensor grad_out_tensor(Shape{b, out_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    grad_out_tensor.uploadData(grad_out_data);

    Tensor in_grad = layer.backward(grad_out_tensor);
    if (in_grad.getTotalElements() != b * in_c * h * w) return false;

    auto in_grad_data = in_grad.getData();
    float grad_abs_sum = 0.0f;
    for (float v : in_grad_data)
    {
        if (std::isnan(v) || std::isinf(v)) return false;
        grad_abs_sum += std::fabs(v);
    }
    if (grad_abs_sum == 0.0f) return false;

    auto params_grads = layer.getParametersAndGradients();
    for (auto &[param, grad] : params_grads)
    {
        if (!param || !grad) continue;
        auto gd = grad->getData();
        for (float v : gd)
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }
    }
    return true;
}

bool testGpuForwardAndBackwardParity()
{
    auto &engine = Execution_Engine::getInstance();
    uint32_t h = 16, w = 16, in_c = 1, out_c = 1, base_c = 4, b = 2;

    NN_UNet_Layer cpu_layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);
    NN_UNet_Layer gpu_layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    auto cpu_params = cpu_layer.getParametersAndGradients();
    auto gpu_params = gpu_layer.getParametersAndGradients();
    for (size_t i = 0; i < cpu_params.size(); ++i)
    {
        if (cpu_params[i].first && gpu_params[i].first)
        {
            gpu_params[i].first->uploadData(cpu_params[i].first->getData());
        }
    }
    gpu_layer.setExecutionTarget(Execution_Target::VULKAN_GPU);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 707);
    Tensor cpu_in(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    cpu_in.uploadData(in_data);
    Tensor gpu_in(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    gpu_in.uploadData(in_data);

    Tensor cpu_out = cpu_layer.forward(cpu_in);
    Tensor gpu_out = gpu_layer.forward(gpu_in);
    engine.executeGraph();

    double fwd_diff = computeMaxAbsDiff(cpu_out.getData(), gpu_out.getData());
    if (fwd_diff > 1e-4) return false;

    std::vector<float> grad_out_data = generateRandomData(b * out_c * h * w, 0.0f, 0.5f, 808);
    Tensor cpu_grad_out(Shape{b, out_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    cpu_grad_out.uploadData(grad_out_data);
    Tensor gpu_grad_out(Shape{b, out_c * h * w}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    gpu_grad_out.uploadData(grad_out_data);

    Tensor cpu_in_grad = cpu_layer.backward(cpu_grad_out);
    Tensor gpu_in_grad = gpu_layer.backward(gpu_grad_out);
    engine.executeGraph();

    double bwd_diff = computeMaxAbsDiff(cpu_in_grad.getData(), gpu_in_grad.getData());
    return bwd_diff <= 1e-3;
}

bool testCloneAndSerialization()
{
    uint32_t h = 16, w = 16, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer1(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    const std::string ckpt_path = "test_nnunet_tmp.nnck";
    {
        std::ofstream out_file(ckpt_path, std::ios::binary);
        if (!out_file.is_open()) return false;
        layer1.saveCheckpoint(out_file);
    }

    NN_UNet_Layer layer2(h, w, in_c, out_c, base_c, Execution_Target::CPU);
    {
        std::ifstream in_file(ckpt_path, std::ios::binary);
        if (!in_file.is_open()) return false;
        layer2.loadCheckpoint(in_file);
    }
    std::remove(ckpt_path.c_str());

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 909);
    Tensor in_tensor(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_tensor.uploadData(in_data);

    Tensor out1 = layer1.forward(in_tensor);
    Tensor out2 = layer2.forward(in_tensor);

    double diff = computeMaxAbsDiff(out1.getData(), out2.getData());
    if (diff > 1e-6) return false;

    auto cloned = layer1.clone();
    Tensor out_cloned = cloned->forward(in_tensor);
    double clone_diff = computeMaxAbsDiff(out1.getData(), out_cloned.getData());
    return clone_diff <= 1e-6;
}

bool testRank4InputForwardBackwardGpu()
{
    uint32_t h = 32, w = 32, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::VULKAN_GPU);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 999);
    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);

    Tensor out_tensor = layer.forward(in_tensor);
    if (out_tensor.getShape() != Shape{b, out_c, h, w}) return false;

    std::vector<float> grad_data = generateRandomData(b * out_c * h * w, 0.1f, 0.5f, 888);
    Tensor grad_out(Shape{b, out_c, h, w}, grad_data, Execution_Target::VULKAN_GPU);

    Tensor in_grad = layer.backward(grad_out);
    Execution_Engine::getInstance().executeGraph();

    if (in_grad.getShape() != Shape{b, in_c, h, w}) return false;

    for (float v : in_grad.getData())
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }
    return true;
}

bool testGradientClipping()
{
    // CPU
    Neural_Network net_cpu(Execution_Target::CPU);
    net_cpu.addLayer<NN_UNet_Layer>(16, 16, 1, 1, 4, Execution_Target::CPU);
    net_cpu.setTrainingMode(true);

    Tensor in_cpu(Shape{1, 1, 16, 16}, generateRandomData(256, 0.0f, 1.0f, 42), Execution_Target::CPU);
    net_cpu.forward(in_cpu);

    Tensor grad_cpu(Shape{1, 1, 16, 16}, std::vector<float>(256, 10.0f), Execution_Target::CPU);
    net_cpu.backward(grad_cpu, true);

    auto pairs_cpu = net_cpu.getParametersAndGradients();
    double sum_sq_cpu = 0.0;
    for (auto &[p, g] : pairs_cpu)
    {
        if (g && !g->isEmpty())
        {
            for (float v : g->getData()) sum_sq_cpu += static_cast<double>(v) * static_cast<double>(v);
        }
    }
    if (static_cast<float>(std::sqrt(sum_sq_cpu)) <= 1.0f) return false;

    net_cpu.clipGradients(1.0f);
    sum_sq_cpu = 0.0;
    for (auto &[p, g] : pairs_cpu)
    {
        if (g && !g->isEmpty())
        {
            for (float v : g->getData()) sum_sq_cpu += static_cast<double>(v) * static_cast<double>(v);
        }
    }
    float post_norm_cpu = static_cast<float>(std::sqrt(sum_sq_cpu));
    if (std::abs(post_norm_cpu - 1.0f) > 1e-2f) return false;

    // GPU
    Neural_Network net_gpu(Execution_Target::VULKAN_GPU);
    net_gpu.addLayer<NN_UNet_Layer>(16, 16, 1, 1, 4, Execution_Target::VULKAN_GPU);
    net_gpu.setTrainingMode(true);

    Tensor in_gpu(Shape{1, 1, 16, 16}, generateRandomData(256, 0.0f, 1.0f, 42), Execution_Target::VULKAN_GPU);
    net_gpu.forward(in_gpu);

    Tensor grad_gpu(Shape{1, 1, 16, 16}, std::vector<float>(256, 10.0f), Execution_Target::VULKAN_GPU);
    net_gpu.backward(grad_gpu, true);

    net_gpu.clipGradients(1.0f);
    auto pairs_gpu = net_gpu.getParametersAndGradients();
    double sum_sq_gpu = 0.0;
    for (auto &[p, g] : pairs_gpu)
    {
        if (g && !g->isEmpty())
        {
            for (float v : g->getData()) sum_sq_gpu += static_cast<double>(v) * static_cast<double>(v);
        }
    }
    float post_norm_gpu = static_cast<float>(std::sqrt(sum_sq_gpu));
    return (std::abs(post_norm_gpu - 1.0f) <= 1e-2f);
}

bool testGpuMixedPrecisionStability()
{
    uint32_t h = 64, w = 64, in_c = 1, out_c = 1, base_c = 8, b = 2;

    Neural_Network net(Execution_Target::VULKAN_GPU);
    net.addLayer<NN_UNet_Layer>(h, w, in_c, out_c, base_c, Execution_Target::VULKAN_GPU);
    net.setMixedPrecision(true);
    net.setTrainingMode(true);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.5f, 0.25f, 12345);
    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);

    Tensor out = net.forward(in_tensor);
    Execution_Engine::getInstance().executeGraph();

    for (float v : out.getData())
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    std::vector<float> grad_data = generateRandomData(b * out_c * h * w, 0.1f, 0.5f, 67890);
    Tensor grad_tensor(Shape{b, out_c, h, w}, grad_data, Execution_Target::VULKAN_GPU);
    net.backward(grad_tensor, true);
    Execution_Engine::getInstance().executeGraph();

    net.setGradientAccumulation(true);
    net.forward(in_tensor);
    net.backward(grad_tensor, true);
    Execution_Engine::getInstance().executeGraph();
    net.setGradientAccumulation(false);

    auto pairs = net.getParametersAndGradients();
    for (const auto &[param, grad] : pairs)
    {
        if (grad && !grad->isEmpty())
        {
            for (float v : grad->getData())
            {
                if (std::isnan(v) || std::isinf(v) || std::abs(v) > 1e5f) return false;
            }
        }
    }

    float clipped_norm = net.clipGradients(1.0f);
    return (!std::isnan(clipped_norm) && !std::isinf(clipped_norm));
}

bool testOptimizerWeightUpdateAndLossDecrease()
{
    uint32_t h = 64, w = 64, in_c = 1, out_c = 1, base_c = 8, b = 2;

    Neural_Network net(Execution_Target::VULKAN_GPU);
    net.addLayer<NN_UNet_Layer>(h, w, in_c, out_c, base_c, Execution_Target::VULKAN_GPU);
    net.setMixedPrecision(true);
    net.setTrainingMode(true);

    auto &scheduler = net.setLearningRate<Cosine_Annealing>(0.01f, 0.001f, 10);
    net.setOptimizer<Adam_Optimizer>(scheduler, 0.9f, 0.999f, 1e-8f, 1.0f, 0.0f);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.5f, 0.25f, 12345);
    std::vector<float> target_data = generateRandomData(b * out_c * h * w, 0.5f, 0.25f, 54321);
    for (float &v : target_data) v = (v > 0.5f) ? 1.0f : 0.0f;

    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);
    Tensor target_tensor(Shape{b, out_c, h, w}, target_data, Execution_Target::VULKAN_GPU);

    Tensor pred0 = net.forward(in_tensor);
    Execution_Engine::getInstance().executeGraph();

    auto pairs = net.getParametersAndGradients();
    auto p0_data = pairs[0].first->getData();

    Dice_BCE_Cost cost_fn(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    Tensor grad;
    cost_fn.computeLossAndGradient(pred0, target_tensor, grad);
    net.backward(grad, true);
    net.clipGradients(1.0f);

    net.getTrainingContext().getOptimizer().step(net.getParametersAndGradients());
    net.invalidateLayerWeightCaches();
    Execution_Engine::getInstance().executeGraph();
    net.zeroGradients();
    Execution_Engine::getInstance().executeGraph();

    auto p1_data = pairs[0].first->getData();
    float weight_diff = 0.0f;
    for (size_t i = 0; i < p0_data.size(); ++i)
    {
        weight_diff += std::abs(p1_data[i] - p0_data[i]);
    }

    return weight_diff > 0.0f;
}

bool testDataAugmentationIntegrity()
{
    std::string seg_path = "data/chest-segment/chest_seg_320x320.bin";
    if (!std::filesystem::exists(seg_path))
    {
        return true;
    }

    chexlocalize::Chest_Seg_Dataset dataset;
    dataset.open(seg_path);

    std::vector<size_t> indices = {10, 25};
    std::mt19937 rng1(42);

    chexlocalize::LungSegAugConfig aug_cfg;
    aug_cfg.enable_hflip = true;
    aug_cfg.hflip_prob = 1.0f;
    aug_cfg.enable_affine = true;
    aug_cfg.affine_prob = 1.0f;
    aug_cfg.enable_intensity = true;
    aug_cfg.intensity_prob = 1.0f;

    auto batch_aug = dataset.getBatch(indices, Execution_Target::CPU, true, &rng1, aug_cfg);
    auto img_data = batch_aug.images.getData();
    auto mask_data = batch_aug.masks.getData();

    int mask_positives = 0;
    for (float m : mask_data)
    {
        if (m != 0.0f && m != 1.0f) return false;
        if (m == 1.0f) mask_positives++;
    }
    if (mask_positives == 0) return false;

    for (float p : img_data)
    {
        if (std::isnan(p) || std::isinf(p)) return false;
    }

    auto b_clean1 = dataset.getBatch(indices, Execution_Target::CPU, false, nullptr);
    auto b_clean2 = dataset.getBatch(indices, Execution_Target::CPU, false, nullptr);
    double clean_diff = computeMaxAbsDiff(b_clean1.images.getData(), b_clean2.images.getData());
    if (clean_diff != 0.0) return false;

    double aug_diff = computeMaxAbsDiff(batch_aug.images.getData(), b_clean1.images.getData());
    return aug_diff >= 1e-3;
}

bool testStratifiedSamplingDistribution()
{
    constexpr size_t TOTAL_SAMPLES = 16068;
    constexpr size_t CHEST_SEG_BOUNDARY = 6810;
    size_t total_cs = CHEST_SEG_BOUNDARY;
    size_t total_qata = TOTAL_SAMPLES - CHEST_SEG_BOUNDARY;

    size_t val_target = 100;
    size_t val_cs_target = (val_target * 30 + 50) / 100;
    size_t val_qata_target = val_target - val_cs_target;

    std::mt19937 split_rng(42);
    std::vector<size_t> cs_all(total_cs);
    std::iota(cs_all.begin(), cs_all.end(), 0);
    std::shuffle(cs_all.begin(), cs_all.end(), split_rng);

    std::vector<size_t> qata_all(total_qata);
    std::iota(qata_all.begin(), qata_all.end(), CHEST_SEG_BOUNDARY);
    std::shuffle(qata_all.begin(), qata_all.end(), split_rng);

    std::unordered_set<size_t> val_set;
    for (size_t i = 0; i < val_cs_target; ++i) val_set.insert(cs_all[i]);
    for (size_t i = 0; i < val_qata_target; ++i) val_set.insert(qata_all[i]);

    std::vector<size_t> pool_train_cs(cs_all.begin() + val_cs_target, cs_all.end());
    std::vector<size_t> pool_train_qata(qata_all.begin() + val_qata_target, qata_all.end());

    size_t train_count = 100;
    size_t target_cs = (train_count * 30 + 50) / 100;
    size_t target_qata = train_count - target_cs;

    std::mt19937 rng(1337);
    for (size_t ep = 0; ep < 10; ++ep)
    {
        std::shuffle(pool_train_cs.begin(), pool_train_cs.end(), rng);
        std::shuffle(pool_train_qata.begin(), pool_train_qata.end(), rng);

        std::vector<size_t> ep_train;
        ep_train.insert(ep_train.end(), pool_train_cs.begin(), pool_train_cs.begin() + target_cs);
        ep_train.insert(ep_train.end(), pool_train_qata.begin(), pool_train_qata.begin() + target_qata);

        size_t cs_in_batch = 0, qata_in_batch = 0;
        for (size_t idx : ep_train)
        {
            if (val_set.count(idx) > 0) return false;
            if (idx < CHEST_SEG_BOUNDARY) cs_in_batch++;
            else qata_in_batch++;
        }

        if (cs_in_batch != target_cs || qata_in_batch != target_qata) return false;
    }
    return true;
}

bool testDeepSupervisionAndResidualParity()
{
    uint32_t h = 160, w = 160, in_c = 1, out_c = 1, base_c = 8, b = 2;

    NN_UNet_Layer layer_cpu(h, w, in_c, out_c, base_c, Execution_Target::CPU);
    layer_cpu.setTrainingMode(true);

    std::vector<float> in_data = generateRandomData(b * in_c * h * w, 0.0f, 1.0f, 777);
    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::CPU);

    Tensor out_main = layer_cpu.forward(in_tensor);
    const Tensor &ds2_out = layer_cpu.getDeepSupervisionOutput2();
    const Tensor &ds3_out = layer_cpu.getDeepSupervisionOutput3();

    if (out_main.getShape() != Shape{b, out_c, h, w} ||
        ds2_out.getShape() != Shape{b, out_c, h / 2, w / 2} ||
        ds3_out.getShape() != Shape{b, out_c, h / 4, w / 4})
    {
        return false;
    }

    Tensor grad_main(Shape{b, out_c, h, w}, generateRandomData(b * out_c * h * w, 0.1f, 0.5f, 111), Execution_Target::CPU);
    Tensor grad_ds2(Shape{b, out_c, h / 2, w / 2}, generateRandomData(b * out_c * (h / 2) * (w / 2), 0.1f, 0.5f, 222), Execution_Target::CPU);
    Tensor grad_ds3(Shape{b, out_c, h / 4, w / 4}, generateRandomData(b * out_c * (h / 4) * (w / 4), 0.1f, 0.5f, 333), Execution_Target::CPU);

    layer_cpu.setDeepSupervisionGradients(grad_ds2, grad_ds3);
    Tensor in_grad = layer_cpu.backward(grad_main);
    layer_cpu.clearDeepSupervisionGradients();

    if (in_grad.getShape() != Shape{b, in_c, h, w}) return false;
    for (float v : in_grad.getData())
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    layer_cpu.setTrainingMode(false);
    layer_cpu.forward(in_tensor);
    if (!layer_cpu.getDeepSupervisionOutput2().isEmpty() || !layer_cpu.getDeepSupervisionOutput3().isEmpty())
    {
        return false;
    }

    NN_UNet_Layer layer_gpu(h, w, in_c, out_c, base_c, Execution_Target::VULKAN_GPU);
    layer_gpu.setTrainingMode(true);
    Tensor gpu_in(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);
    Tensor gpu_out = layer_gpu.forward(gpu_in);
    Execution_Engine::getInstance().executeGraph();

    if (gpu_out.getShape() != Shape{b, out_c, h, w} ||
        layer_gpu.getDeepSupervisionOutput2().getShape() != Shape{b, out_c, h / 2, w / 2} ||
        layer_gpu.getDeepSupervisionOutput3().getShape() != Shape{b, out_c, h / 4, w / 4})
    {
        return false;
    }

    Tensor gpu_grad_main(Shape{b, out_c, h, w}, grad_main.getData(), Execution_Target::VULKAN_GPU);
    Tensor gpu_grad_ds2(Shape{b, out_c, h / 2, w / 2}, grad_ds2.getData(), Execution_Target::VULKAN_GPU);
    Tensor gpu_grad_ds3(Shape{b, out_c, h / 4, w / 4}, grad_ds3.getData(), Execution_Target::VULKAN_GPU);
    layer_gpu.setDeepSupervisionGradients(gpu_grad_ds2, gpu_grad_ds3);
    Tensor gpu_in_grad = layer_gpu.backward(gpu_grad_main);
    Execution_Engine::getInstance().executeGraph();

    for (float v : gpu_in_grad.getData())
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    return true;
}

bool runSuite()
{
    bool ok = true;
    ok &= testForwardShapeAndOutputRange();
    ok &= testTrainEvalConsistency();
    ok &= testBatchInvarianceParity();
    ok &= testBackwardGradientFlowCpu();
    ok &= testGpuForwardAndBackwardParity();
    ok &= testCloneAndSerialization();
    ok &= testRank4InputForwardBackwardGpu();
    ok &= testGradientClipping();
    ok &= testGpuMixedPrecisionStability();
    ok &= testOptimizerWeightUpdateAndLossDecrease();
    ok &= testDataAugmentationIntegrity();
    ok &= testStratifiedSamplingDistribution();
    ok &= testDeepSupervisionAndResidualParity();
    return ok;
}
} // namespace TestNnUnet

// ==============================================================================
// TEST SUITE: BatchNorm U-Net Verification
// ==============================================================================
namespace TestBatchNormUnet
{
bool runSuite()
{
    const std::string dataset_path = "data/chest-segment/chest_seg_320x320.bin";
    const std::string ckpt_path = "output/unet/unet_latest.nnck";
    if (!std::filesystem::exists(dataset_path) || !std::filesystem::exists(ckpt_path))
    {
        return true;
    }

    try
    {
        Execution_Target target = Execution_Target::VULKAN_GPU;
        auto &engine = Execution_Engine::getInstance();

        chexlocalize::Chest_Seg_Dataset dataset;
        dataset.open(dataset_path);

        Neural_Network net(target);
        net.addLayer(std::make_unique<U_Net_Layer>(320, 320, 1, 1, 16, target));
        net.enableMixedPrecision(true);

        net.loadTrainingCheckpoint(ckpt_path, 30);
        if (net.getLayers().empty() || !net.getLayers()[0]) return false;
        auto *unet = dynamic_cast<U_Net_Layer*>(net.getLayers()[0].get());
        if (unet)
        {
            unet->inspectAllBatchNorms();
        }

        std::vector<size_t> indices = { 100, 101 };
        auto [b_imgs, b_masks] = dataset.getBatch(indices, target, false, nullptr);

        net.setTrainingMode(true);
        Tensor pred_train = net.forward(b_imgs);
        engine.executeGraph();
        for (float v : pred_train.getData())
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        net.setTrainingMode(false);
        Tensor pred_eval = net.forward(b_imgs);
        engine.executeGraph();
        for (float v : pred_eval.getData())
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        std::vector<size_t> train_indices = { 0, 1 };
        auto [tr_imgs, tr_masks] = dataset.getBatch(train_indices, target, false, nullptr);
        Tensor pred_tr_eval = net.forward(tr_imgs);
        engine.executeGraph();
        for (float v : pred_tr_eval.getData())
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "  [FAIL] Exception in BatchNorm U-Net test: " << e.what() << "\n";
        return false;
    }
}
} // namespace TestBatchNormUnet

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
    std::cout << "  BPE Tokenizer (UTF-8 & Fallback):  " << (testBpeTokenizer() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Ternary Computing Phase 1 Suite:   " << (testTernaryComputingPhase1() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Ternary Computing Phase 2 Suite:   " << (testTernaryComputingPhase2() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Ternary Computing Phase 4 Suite:   " << (testTernaryComputingPhase4() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Ternary Computing Phase 5 Suite:   " << (testTernaryComputingPhase5() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Dice BCE Cost Function Suite:      " << (TestDiceBceCost::runSuite() ? "PASS" : "FAIL") << "\n";
    std::cout << "  InstanceNorm2D Operator Suite:     " << (TestInstanceNorm::runSuite() ? "PASS" : "FAIL") << "\n";
    std::cout << "  LeakyReLU Operator & Layer Suite:  " << (TestLeakyRelu::runSuite() ? "PASS" : "FAIL") << "\n";
    std::cout << "  UpSample2D Operator & Layer Suite: " << (TestUpsample::runSuite() ? "PASS" : "FAIL") << "\n";
    std::cout << "  NN-UNet Comprehensive Suite (x13): " << (TestNnUnet::runSuite() ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm U-Net Verification:      " << (TestBatchNormUnet::runSuite() ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n";

    return 0;
}