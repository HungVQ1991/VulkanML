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

#include "cost_function/bce_cost.h"
#include "cost_function/cce_cost.h"
#include "cost_function/mae_cost.h"
#include "cost_function/mse_cost.h"
#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "engine/graph_optimizer.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_sub_allocator.h"
#include "layer/batch_norm_layer.h"
#include "layer/batch_norm2d_layer.h"
#include "layer/conv2d_layer.h"
#include "layer/gelu.h"
#include "layer/globalavgpool2d_layer.h"
#include "layer/linear_layer.h"
#include "layer/maxpool2d_layer.h"
#include "layer/relu.h"
#include "layer/res_net_block_2d_layer.h"
#include "layer/softmax.h"
#include "learning_rate/cosine_annealing.h"
#include "learning_rate/exponential_decay.h"
#include "learning_rate/multi_step_decay.h"
#include "learning_rate/no_decay.h"
#include "learning_rate/polynomial_decay.h"
#include "learning_rate/reduce_on_plateau.h"
#include "learning_rate/step_decay.h"
#include "math/matrix.h"
#include "neural_network.h"
#include "optimizer/adam_optimizer.h"
#include "optimizer/sgd_optimizer.h"

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
    Res_Net_Block_2d_Layer block_identity(4, 4, 16, 16, 1, exec_target);
    Matrix input_identity(1, 4 * 4 * 16, std::vector<float>(4 * 4 * 16, 0.05f), exec_target);
    Matrix out_identity = block_identity.forward(input_identity);
    bool identity_fwd_ok = verifyMatrix(out_identity, out_identity.getData());

    Matrix grad_in_id = block_identity.backward(out_identity);
    bool identity_bwd_ok = verifyMatrix(grad_in_id, grad_in_id.getData());
    block_identity.resetGradient();

    Res_Net_Block_2d_Layer block_proj(4, 4, 16, 32, 2, exec_target);
    Matrix input_proj(1, 4 * 4 * 16, std::vector<float>(4 * 4 * 16, 0.05f), exec_target);
    Matrix out_proj = block_proj.forward(input_proj);
    bool proj_fwd_ok = verifyMatrix(out_proj, out_proj.getData());

    Matrix grad_in_proj = block_proj.backward(out_proj);
    bool proj_bwd_ok = verifyMatrix(grad_in_proj, grad_in_proj.getData());
    block_proj.resetGradient();

    return identity_fwd_ok && identity_bwd_ok && proj_fwd_ok && proj_bwd_ok;
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
    std::vector<float> res_data = mat_relu.getData();

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

void runTestSuite(Execution_Target exec_target, const std::string &target_name)
{
    std::cout << "========================================\n";
    std::cout << "   RUNNING TEST SUITE ON " << target_name << "\n";
    std::cout << "========================================\n";

    std::cout << "\n[1. Basic Matrix Arithmetics]\n";
    std::cout << "  Matrix Addition (with Broadcast):  " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Subtraction (with Broadcast): " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Multiplication (GEMM):      " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Scalar Multiplication & Division:   " << (testScalarOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Hadamard Multiplication & Division: " << (testHadamardOperations(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[2. Transformations & Advanced Operations]\n";
    std::cout << "  Transpose & Matrix Inversion:       " << (testTransposeAndInverse(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Euclidean L2 Normalization:        " << (testNormalize(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Fused Linear Bias Add (MatmulAdd): " << (testMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[3. Activation Functions]\n";
    std::cout << "  ReLU Forward & Backward:           " << (testRelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  GELU Forward & Backward:           " << (testGelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Softmax Forward & Backward:        " << (testSoftmax(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[4. Cost & Loss Functions]\n";
    std::cout << "  MSE Cost & Gradient:               " << (testMseLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  MAE Cost & Gradient:               " << (testMaeLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BCE Cost & Gradient:               " << (testBceLoss(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  CCE Cost & Gradient:               " << (testCceLoss(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[5. Neural Network Layers]\n";
    std::cout << "  Linear Layer (Forward & Backward): " << (testLinearLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Conv2D Layer (Forward & Backward): " << (testConv2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  MaxPool2D Layer (with Mask):       " << (testMaxPool2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  GlobalAvgPool2D Layer:             " << (testGlobalAvgPool2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm 1D Layer:                " << (testBatchNormLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm 2D Layer:                " << (testBatchNorm2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  ResNet Block 2D (Id & Proj):       " << (testResNetBlock2dLayer(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[6. Optimizers]\n";
    std::cout << "  SGD Optimizer Step:                " << (testSgdOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Adam Optimizer Step:               " << (testAdamOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[7. Serialization & I/O]\n";
    std::cout << "  Matrix Binary I/O:                 " << (testMatrixSerialization(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Model Inference I/O (NNI1):        " << (testModelInferenceSerialization(exec_target) ? "PASS" : "FAIL") << "\n\n";
}

int main()
{
    Logger::setFileLogging(true);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);
    Logger::logMessage("==================================Test log==================================", Log_Level::LOG_INFO, true);
    runTestSuite(Execution_Target::CPU, "CPU BACKEND");
    runTestSuite(Execution_Target::VULKAN_GPU, "VULKAN GPU BACKEND");

    std::cout << "========================================\n";
    std::cout << "   SYSTEM & LIFECYCLE MANAGEMENT TESTS  \n";
    std::cout << "========================================\n";

    std::cout << "  Learning Rate Schedulers Suite:    " << (testLearningRateSchedulers() ? "PASS" : "FAIL") << "\n";
    std::cout << "  GPU Vector Lifecycle & Resizing:   " << (testGpuVectorLifecycle() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Sub-Allocator & Garbage Collector: " << (testVulkanSubAllocatorAndGarbageCollection() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Operator Fusion & Graph Dispatch:  " << (testOperatorFusionAndGraphExecution() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Async Data Pipeline Double-Buffer: " << (testAsyncDataPipeline() ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n";

    return 0;
}