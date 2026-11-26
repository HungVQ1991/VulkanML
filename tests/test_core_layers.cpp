#include "test_common.h"

bool testEmbeddingLayer(Execution_Target exec_target)
{
    constexpr size_t vocab_size = 8;
    constexpr size_t embedding_dim = 4;

    Embedding_Layer layer(vocab_size, embedding_dim, exec_target);

    std::vector<int32_t> tokens_1d = { 1, 3, 5 };
    Tensor out_1d = layer.forward(tokens_1d);
    flushIfGpu(exec_target);
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
    flushIfGpu(exec_target);
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
    flushIfGpu(exec_target);
    if (out_tensor.getShape() != Shape{ 2, 2, embedding_dim })
    {
        return false;
    }

    Tensor grad_out(Shape{ 2, 2, embedding_dim }, std::vector<float>(2 * 2 * embedding_dim, 0.5f), exec_target);
    Tensor grad_in = layer.backward(grad_out);
    flushIfGpu(exec_target);
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
    flushIfGpu(exec_target);
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

    const std::string temp_file = makeUniqueTestTempPath("temp_embedding_io").string();
    Scoped_Temp_File_Guard temp_guard({temp_file});
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

    Tensor orig_eval = layer.forward(tokens_1d);
    Tensor loaded_eval = loaded_layer.forward(tokens_1d);
    flushIfGpu(exec_target);
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
    flushIfGpu(exec_target);

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
    flushIfGpu(exec_target);

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
    flushIfGpu(exec_target);
    if (std::isnan(loss_ignored) || std::isinf(loss_ignored) || loss_ignored <= 0.0f)
    {
        return false;
    }

    std::vector<int32_t> all_ignored = { 2, 2, 2, 2, 2, 2 };
    Tensor grad_all_ignored;
    float loss_all_ignored = cce_ignore.computeLossAndGradient(logits, all_ignored, grad_all_ignored);
    flushIfGpu(exec_target);
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

bool testConv2dFp16CoopMatMultiChannelParity(Execution_Target exec_target)
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

    Scoped_Engine_Flags_Guard flags_guard;
    if (Runtime::isCooperativeMatrixSupported())
    {
        Runtime::setCooperativeMatrixEnabled(true);
    }

    {
        Linear_Layer warmup_lin(16, 16, Execution_Target::VULKAN_GPU);
        warmup_lin.setMixedPrecision(true);
        Tensor warmup_in(Shape{2, 16}, generateRandomVector(32, -0.5f, 0.5f, 101), Execution_Target::VULKAN_GPU);
        (void)warmup_lin.forward(warmup_in);
        flushGpuEngine(true);
    }

    constexpr uint32_t H = 8;
    constexpr uint32_t W = 8;
    constexpr uint32_t C_IN = 16;
    constexpr uint32_t C_OUT = 32;
    constexpr uint32_t K = 3;
    constexpr uint32_t S = 1;
    constexpr uint32_t P = 1;
    constexpr size_t B = 2;

    const std::vector<float> w_vals = generateRandomVector(static_cast<size_t>(K) * K * C_IN * C_OUT, -0.25f, 0.25f, 202);
    const std::vector<float> b_vals = generateRandomVector(C_OUT, -0.2f, 0.2f, 303);
    const std::vector<float> in_vals = generateRandomVector(B * H * W * C_IN, -0.5f, 0.5f, 404);
    const std::vector<float> grad_vals = generateRandomVector(B * H * W * C_OUT, -0.25f, 0.25f, 505);

    Conv2d_Layer cpu_layer(H, W, C_IN, C_OUT, K, S, P, Execution_Target::CPU);
    auto cpu_params = cpu_layer.getParametersAndGradients();
    cpu_params[0].first->uploadData(w_vals);
    cpu_params[1].first->uploadData(b_vals);
    Tensor cpu_in(Shape{B, H * W * C_IN}, in_vals, Execution_Target::CPU);
    Tensor cpu_out = cpu_layer.forward(cpu_in);
    Tensor cpu_grad_out(Shape{B, H * W * C_OUT}, grad_vals, Execution_Target::CPU);
    Tensor cpu_grad_in = cpu_layer.backward(cpu_grad_out);

    Conv2d_Layer gpu_layer(H, W, C_IN, C_OUT, K, S, P, Execution_Target::VULKAN_GPU);
    auto gpu_params = gpu_layer.getParametersAndGradients();
    gpu_params[0].first->uploadData(w_vals);
    gpu_params[1].first->uploadData(b_vals);
    gpu_layer.setMixedPrecision(true);

    Tensor gpu_in(Shape{B, H * W * C_IN}, in_vals, Execution_Target::VULKAN_GPU);
    Tensor gpu_out = gpu_layer.forward(gpu_in);
    Tensor gpu_grad_out(Shape{B, H * W * C_OUT}, grad_vals, Execution_Target::VULKAN_GPU);
    Tensor gpu_grad_in = gpu_layer.backward(gpu_grad_out);
    flushGpuEngine(true);

    const auto &gpu_out_data = gpu_out.getData();
    if (gpu_out_data.size() != B * H * W * C_OUT)
    {
        return failTestStep("Conv2dFp16CoopMatMultiCh", "output size mismatch");
    }
    auto [min_it, max_it] = std::minmax_element(gpu_out_data.begin(), gpu_out_data.end());
    if ((*max_it - *min_it) <= 1e-2f)
    {
        return failTestStep("Conv2dFp16CoopMatMultiCh", "output collapsed to constant values");
    }

    if (!verifyMatrix(gpu_out, cpu_out.getData(), 5e-2f))
    {
        return failTestStep("Conv2dFp16CoopMatMultiCh", "forward CPU/GPU parity mismatch");
    }
    if (!verifyMatrix(gpu_grad_in, cpu_grad_in.getData(), 5e-2f))
    {
        return failTestStep("Conv2dFp16CoopMatMultiCh", "input gradient CPU/GPU parity mismatch");
    }
    if (!verifyMatrix(gpu_layer.getWeightsGradient(), cpu_layer.getWeightsGradient().getData(), 1e-1f))
    {
        return failTestStep("Conv2dFp16CoopMatMultiCh", "weight gradient CPU/GPU parity mismatch");
    }
    if (!verifyMatrix(*gpu_params[1].second, cpu_params[1].second->getData(), 1e-1f))
    {
        return failTestStep("Conv2dFp16CoopMatMultiCh", "bias gradient CPU/GPU parity mismatch");
    }

    return true;
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
        Tensor q_fp16 = castTensorToFp16(q);
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
    flushIfGpu(exec_target);
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

    Tensor q_gpu_fp16 = castTensorToFp16(Tensor(Shape{1, 1, 16, 16}, q_data, Execution_Target::VULKAN_GPU));
    Tensor k_gpu_fp16 = castTensorToFp16(Tensor(Shape{1, 1, 16, 16}, k_data, Execution_Target::VULKAN_GPU));
    Tensor v_gpu_fp16 = castTensorToFp16(Tensor(Shape{1, 1, 16, 16}, v_data, Execution_Target::VULKAN_GPU));
    Tensor do_gpu_fp16 = castTensorToFp16(Tensor(Shape{1, 1, 16, 16}, do_data, Execution_Target::VULKAN_GPU));

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
        std::string expected_decoded = " nhiều   khoảng   trắng  \n dòng mới \t tab";
        if (decoded != expected_decoded)
        {
            std::cout << "\n[DEBUG: Test 3 decode failed: expected '" << expected_decoded << "', got '" << decoded << "']\n";
            return false;
        }
    }

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

    {
        std::vector<int32_t> tokens = {1, 23325, 2};
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

    {
        KV_Cache_Manager cache(batch_size, num_heads, max_seq_len, head_dim, exec_target, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);
        if (cache.getCurrentSeqLen() != 0 || cache.getMaxSeqLen() != max_seq_len)
        {
            return false;
        }

        std::vector<float> k_data_prefill(1 * 2 * 4 * 8, 1.5f);
        std::vector<float> v_data_prefill(1 * 2 * 4 * 8, 2.5f);
        Tensor k_prefill(Shape{1, 2, 4, 8}, k_data_prefill, exec_target);
        Tensor v_prefill(Shape{1, 2, 4, 8}, v_data_prefill, exec_target);

        cache.append(k_prefill, v_prefill);
        flushIfGpu(exec_target);

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

        for (size_t t = 0; t < 2; ++t)
        {
            std::vector<float> k_tok(1 * 2 * 1 * 8, static_cast<float>(t + 10));
            std::vector<float> v_tok(1 * 2 * 1 * 8, static_cast<float>(t + 20));
            Tensor k_new(Shape{1, 2, 1, 8}, k_tok, exec_target);
            Tensor v_new(Shape{1, 2, 1, 8}, v_tok, exec_target);
            cache.append(k_new, v_new);
        }
        flushIfGpu(exec_target);

        if (cache.getCurrentSeqLen() != 6)
        {
            return false;
        }

        Tensor k_6 = cache.getK();
        if (k_6.getShape() != Shape{1, 2, 6, 8})
        {
            return false;
        }

        cache.reset();
        if (cache.getCurrentSeqLen() != 0)
        {
            return false;
        }
    }

    {
        constexpr size_t ring_max_seq = 4;
        KV_Cache_Manager ring_cache(1, 1, ring_max_seq, 4, exec_target, Data_Type::FLOAT32, KV_Cache_Mode::RING_BUFFER);

        for (size_t i = 1; i <= 6; ++i)
        {
            std::vector<float> tok_val(4, static_cast<float>(i));
            Tensor k_tok(Shape{1, 1, 1, 4}, tok_val, exec_target);
            Tensor v_tok(Shape{1, 1, 1, 4}, tok_val, exec_target);
            ring_cache.append(k_tok, v_tok);
        }
        flushIfGpu(exec_target);

        if (ring_cache.getCurrentSeqLen() != ring_max_seq)
        {
            return false;
        }

        Tensor k_res = ring_cache.getK();
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

    std::vector<float> in_data(S * hidden_dim, 0.1f);
    for (size_t i = 0; i < in_data.size(); ++i)
    {
        in_data[i] = 0.05f * static_cast<float>((i % 9) + 1);
    }
    Tensor in_tensor(Shape{ 1, S, hidden_dim }, in_data, exec_target);
    Tensor out_tensor = block.forward(in_tensor);
    flushIfGpu(exec_target);

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

    KV_Cache_Manager cache(1, num_heads, 16, head_dim, exec_target, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);

    Tensor prefill_in(Shape{ 1, 3, hidden_dim }, std::vector<float>(3 * hidden_dim, 0.1f), exec_target);
    Tensor prefill_out = block.forward(prefill_in, &cache);
    flushIfGpu(exec_target);
    if (cache.getCurrentSeqLen() != 3)
    {
        return false;
    }

    Tensor decode_in(Shape{ 1, 1, hidden_dim }, std::vector<float>(1 * hidden_dim, 0.2f), exec_target);
    Tensor decode_out = block.forward(decode_in, &cache);
    flushIfGpu(exec_target);
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
    flushIfGpu(exec_target);
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

    constexpr size_t B_test = 2;
    std::vector<float> batch_in_data(B_test * S * hidden_dim, 0.1f);
    for (size_t i = 0; i < batch_in_data.size(); ++i)
    {
        batch_in_data[i] = 0.05f * static_cast<float>((i % 9) + 1);
    }
    Tensor batch_in(Shape{ B_test, S, hidden_dim }, batch_in_data, exec_target);
    Tensor batch_out = block.forward(batch_in);
    flushIfGpu(exec_target);
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
    flushIfGpu(exec_target);
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

    std::vector<int32_t> test_tokens = { 1, 15, 42, 100 };
    Tensor logits = model.forward(test_tokens, false);
    flushIfGpu(exec_target);

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

    std::string prompt = "Hello";
    std::string generated = model.generate(prompt, 6, 0.0f, 0.9f);
    if (generated.empty())
    {
        return false;
    }

    Neural_Network nn(exec_target);
    nn.addLayer<Embedding_Layer>(cfg.vocab_size, cfg.hidden_dim, exec_target);
    nn.addLayer<Transformer_Block>(cfg.hidden_dim, cfg.num_heads, cfg.intermediate_dim, 1e-5f, 10000.0f, exec_target);
    nn.addLayer<RMSNorm_Layer>(cfg.hidden_dim, 1e-5f, exec_target);
    nn.addLayer<Linear_Layer>(cfg.hidden_dim, cfg.vocab_size, exec_target);

    Tensor input_tokens(1, 4, std::vector<float>{ 1.0f, 15.0f, 42.0f, 100.0f }, exec_target);
    Tensor nn_out = nn.forward(input_tokens);
    flushIfGpu(exec_target);
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

    std::vector<int32_t> input_tokens = { 5, 12, 23, 40 };
    std::vector<int32_t> target_tokens = { 12, 23, 40, 55 };

    Tensor logits = model.forward(input_tokens, false);
    flushIfGpu(exec_target);

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

    const std::string ckpt_path = makeUniqueTestTempPath(
        (exec_target == Execution_Target::CPU) ? "causal_lm_cpu" : "causal_lm_gpu", ".ckpt").string();
    Scoped_Temp_File_Guard ckpt_guard({ckpt_path});
    model.saveCheckpoint(ckpt_path);

    Causal_LM loaded_model(cfg);
    loaded_model.loadCheckpoint(ckpt_path);

    Tensor logits_orig = model.forward(input_tokens, false);
    Tensor logits_loaded = loaded_model.forward(input_tokens, false);
    flushIfGpu(exec_target);

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

    const std::string inf_path = makeUniqueTestTempPath(
        (exec_target == Execution_Target::CPU) ? "causal_lm_cpu" : "causal_lm_gpu").string();
    Scoped_Temp_File_Guard inf_guard({inf_path});
    model.saveInference(inf_path);

    Causal_LM inf_model(cfg);
    inf_model.loadInference(inf_path);

    Tensor logits_inf = inf_model.forward(input_tokens, false);
    flushIfGpu(exec_target);
    std::vector<float> inf_data = logits_inf.getData();
    for (size_t i = 0; i < orig_data.size(); ++i)
    {
        if (std::fabs(orig_data[i] - inf_data[i]) > 1e-4f)
        {
            return false;
        }
    }

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

    model.forwardLossAndBackward(input_tokens, target_tokens, 1.0f);
    float norm_clipped = model.clipGradients(0.5f);
    if (std::isnan(norm_clipped) || std::isinf(norm_clipped))
    {
        return false;
    }
    model.resetGradients();

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

    const std::string bin_path = makeUniqueTestTempPath("temp_test_dataset").string();
    Scoped_Temp_File_Guard bin_guard({bin_path});
    std::vector<int32_t> sample_tokens = { 1, 2, 42, 1337, 99999 };
    if (!Binary_Token_Dataset::save(bin_path, sample_tokens))
    {
        return false;
    }
    std::vector<int32_t> loaded_tokens;
    if (!Binary_Token_Dataset::load(bin_path, loaded_tokens) || loaded_tokens != sample_tokens)
    {
        return false;
    }

    std::string topk_gen = model.generate("Hello", 4, 0.0f, 0.9f, 2, true, nullptr, 1.2f, 5, { "el" });
    if (topk_gen.empty())
    {
        return false;
    }

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

    Tensor logits = model.forward(input_tokens, false);
    if (logits.getDataType() != Data_Type::FLOAT16)
    {
        return false;
    }
    Execution_Engine::getInstance().executeGraph();

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

bool testSigmoidLayer(Execution_Target exec_target)
{
    Sigmoid_Layer layer(exec_target);
    if (layer.getLayerType() != Layer_Type::SIGMOID ||
        !layer.supportsPopulationBatch() ||
        layer.hasParameters() ||
        layer.isForwardCompleted())
    {
        return false;
    }

    Tensor unready_grad(1, 3, {1.0f, 1.0f, 1.0f}, exec_target);
    if (!expectThrows<std::logic_error>([&]() { (void)layer.backward(unready_grad); }))
    {
        return false;
    }
    if (!expectThrows<std::out_of_range>([&]() { layer.setPopulationParameter(0, {}); }))
    {
        return false;
    }

    Tensor input_mat(1, 3, {-1.0f, 0.0f, 1.0f}, exec_target);
    Tensor out_mat = layer.forward(input_mat);
    flushIfGpu(exec_target);
    if (!layer.isForwardCompleted() || !verifyMatrix(out_mat, {0.26894142f, 0.5f, 0.73105858f}, 1e-3f))
    {
        return false;
    }

    Tensor grad_out(1, 3, {1.0f, -2.0f, 0.5f}, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    flushIfGpu(exec_target);
    if (!verifyMatrix(grad_in, {0.19661193f, -0.5f, 0.09830597f}, 1e-3f))
    {
        return false;
    }

    Tensor pop_in(Shape{2, 2}, std::vector<float>{-1.0f, 0.0f, 1.0f, 2.0f}, exec_target);
    Tensor pop_out = layer.forward(pop_in, {});
    flushIfGpu(exec_target);
    if (!verifyMatrix(pop_out, {0.26894142f, 0.5f, 0.73105858f, 0.88079708f}, 1e-3f))
    {
        return false;
    }

    auto cloned = layer.clone();
    if (!cloned || cloned->getLayerType() != Layer_Type::SIGMOID)
    {
        return false;
    }
    layer.resetGradient();
    return !layer.isForwardCompleted();
}

bool testAslLoss(Execution_Target exec_target)
{
    if (!expectThrows<std::invalid_argument>([&]() { Asl_Cost(-0.1f, 4.0f, 0.05f, 1e-7f, exec_target); }) ||
        !expectThrows<std::invalid_argument>([&]() { Asl_Cost(1.0f, -1.0f, 0.05f, 1e-7f, exec_target); }) ||
        !expectThrows<std::invalid_argument>([&]() { Asl_Cost(1.0f, 4.0f, 1.0f, 1e-7f, exec_target); }) ||
        !expectThrows<std::invalid_argument>([&]() { Asl_Cost(1.0f, 4.0f, 0.05f, 0.0f, exec_target); }))
    {
        return false;
    }

    Asl_Cost asl(1.0f, 4.0f, 0.05f, 1e-7f, exec_target);
    if (asl.getType() != Cost_Type::ASL)
    {
        return false;
    }

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    asl.setGammaPos(2.0f);
    asl.setGammaNeg(3.0f);
    asl.setClipMargin(0.10f);
    asl.saveCheckpoint(ss);
    Asl_Cost loaded(0.0f, 0.0f, 0.0f, 1e-5f, exec_target);
    ss.seekg(0);
    loaded.loadCheckpoint(ss);
    if (!nearlyEqual(loaded.getGammaPos(), 2.0f, 1e-6f) ||
        !nearlyEqual(loaded.getGammaNeg(), 3.0f, 1e-6f) ||
        !nearlyEqual(loaded.getClipMargin(), 0.10f, 1e-6f))
    {
        return false;
    }

    Asl_Cost cost_fn(1.0f, 4.0f, 0.05f, 1e-7f, exec_target);
    Tensor pred_clipped(1, 2, {0.02f, 0.05f}, exec_target);
    Tensor tgt_clipped(1, 2, {0.0f, 0.0f}, exec_target);
    float clipped_loss = cost_fn.computeLoss(pred_clipped, tgt_clipped);
    Tensor clipped_grad = cost_fn.computeGradient(pred_clipped, tgt_clipped);
    flushIfGpu(exec_target);
    if (!nearlyEqual(clipped_loss, 0.0f, 1e-6f) || !verifyMatrix(clipped_grad, {0.0f, 0.0f}, 1e-6f))
    {
        return false;
    }

    const std::vector<float> pred_vals = {0.80f, 0.25f, 0.45f, 0.70f};
    const std::vector<float> tgt_vals = {1.0f, 0.0f, 1.0f, 0.0f};
    Tensor pred(Shape{2, 2}, pred_vals, exec_target);
    Tensor target(Shape{2, 2}, tgt_vals, exec_target);

    float loss_separate = cost_fn.computeLoss(pred, target);
    Tensor grad_separate = cost_fn.computeGradient(pred, target);
    Tensor grad_fused(exec_target);
    float loss_fused = cost_fn.computeLossAndGradient(pred, target, grad_fused);
    flushIfGpu(exec_target);
    if (!std::isfinite(loss_separate) || loss_separate <= 0.0f ||
        !nearlyEqual(loss_separate, loss_fused, 1e-5f) ||
        !verifyTensorsClose(grad_separate, grad_fused, 1e-4f))
    {
        return false;
    }

    const auto grad_vec = grad_separate.getData();
    const double rel_err = computeNumericalGradientMaxRelErr(
        pred_vals,
        1e-3f,
        grad_vec,
        [&](const std::vector<float> &perturbed) {
            Tensor p(Shape{2, 2}, perturbed, exec_target);
            return cost_fn.computeLoss(p, target);
        });
    return rel_err < 1e-2;
}

bool testAvgPool2dLayer(Execution_Target exec_target)
{
    Avg_Pool_2d_Layer layer(4, 4, 1, 2, 2, 0, exec_target);
    Tensor input_mat(1, 16, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f}, exec_target);

    Tensor out_mat = layer.forward(input_mat);
    flushIfGpu(exec_target);
    if (!verifyMatrix(out_mat, {3.5f, 5.5f, 11.5f, 13.5f}, 1e-3f))
    {
        return false;
    }

    Tensor grad_out(1, 4, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Tensor grad_in = layer.backward(grad_out);
    flushIfGpu(exec_target);
    if (!verifyMatrix(grad_in, {
        0.25f, 0.25f, 0.5f, 0.5f,
        0.25f, 0.25f, 0.5f, 0.5f,
        0.75f, 0.75f, 1.0f, 1.0f,
        0.75f, 0.75f, 1.0f, 1.0f}, 1e-3f))
    {
        return false;
    }

    Tensor pad_in(Shape{1, 2, 2, 1}, std::vector<float>{4.0f, 8.0f, 12.0f, 16.0f}, exec_target);
    Tensor pad_out(exec_target);
    ops::vision::avgpool2d(pad_in, pad_out, 2, 2, 1, 2, 1, 1);
    Tensor pad_grad_out(Shape{1, 3, 3, 1}, std::vector<float>(9, 4.0f), exec_target);
    Tensor pad_grad_in(exec_target);
    ops::vision::avgpool2dBackward(pad_grad_out, pad_grad_in, 2, 2, 1, 3, 3, 2, 1, 1);
    flushIfGpu(exec_target);

    if (pad_out.getShape() != Shape{1, 3, 3, 1} ||
        !verifyMatrix(pad_out, {1.0f, 3.0f, 2.0f, 4.0f, 10.0f, 6.0f, 3.0f, 7.0f, 4.0f}, 1e-3f) ||
        !verifyMatrix(pad_grad_in, {4.0f, 4.0f, 4.0f, 4.0f}, 1e-3f))
    {
        return false;
    }

    auto cloned = layer.clone();
    layer.resetGradient();
    return cloned != nullptr && expectThrows<std::logic_error>([&]() { (void)layer.backward(grad_out); });
}

bool testGlobalMaxPool2d(Execution_Target exec_target)
{
    const std::vector<float> in_vals = {
        1.0f, 8.0f, 5.0f, 2.0f, 3.0f, 6.0f, 4.0f, 7.0f,
        -3.0f, 0.0f, -1.0f, -4.0f, -5.0f, -2.0f, -2.0f, -1.0f};
    Tensor in_r4(Shape{2, 2, 2, 2}, in_vals, exec_target);
    Tensor out_r4(exec_target);
    Tensor mask_r4(exec_target);
    ops::vision::globalMaxPool2d(in_r4, out_r4, mask_r4, 2, 2, 2);

    Tensor dy_r4(Shape{2, 1, 1, 2}, std::vector<float>{1.5f, -2.0f, 3.0f, 4.0f}, exec_target);
    Tensor dx_r4(exec_target);
    ops::vision::globalMaxPool2dBackward(dy_r4, mask_r4, dx_r4, 2, 2, 2);
    flushIfGpu(exec_target);

    const std::vector<float> expected_out = {5.0f, 8.0f, -1.0f, 0.0f};
    const std::vector<float> expected_dx = {
        0.0f, -2.0f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (out_r4.getShape() != Shape{2, 2} ||
        dx_r4.getShape() != Shape{2, 2, 2, 2} ||
        !verifyMatrix(out_r4, expected_out, 1e-4f) ||
        !verifyMatrix(dx_r4, expected_dx, 1e-4f))
    {
        return false;
    }

    Tensor in_r2(Shape{2, 8}, in_vals, exec_target);
    Tensor out_r2(exec_target);
    Tensor mask_r2(exec_target);
    ops::vision::globalMaxPool2d(in_r2, out_r2, mask_r2, 2, 2, 2);
    Tensor dy_r2(Shape{2, 2}, std::vector<float>{1.5f, -2.0f, 3.0f, 4.0f}, exec_target);
    Tensor dx_r2(exec_target);
    ops::vision::globalMaxPool2dBackward(dy_r2, mask_r2, dx_r2, 2, 2, 2);
    flushIfGpu(exec_target);
    if (out_r2.getShape() != Shape{2, 2} ||
        dx_r2.getShape() != Shape{2, 8} ||
        !verifyMatrix(out_r2, expected_out, 1e-4f) ||
        !verifyMatrix(dx_r2, expected_dx, 1e-4f))
    {
        return false;
    }

    Tensor empty_in(Shape{0, 2, 2, 2}, exec_target);
    Tensor empty_out(exec_target), empty_mask(exec_target);
    ops::vision::globalMaxPool2d(empty_in, empty_out, empty_mask, 2, 2, 2);
    flushIfGpu(exec_target);
    return empty_out.getShape() == Shape{0, 2};
}
