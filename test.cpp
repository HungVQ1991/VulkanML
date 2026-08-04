#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

#include "optimizer.h"
#include "batch_norm_layer.h"
#include "layer.h"
#include "relu.h"
#include "softmax.h"
#include "cost_function.h"
#include "neural_network.h"
#include "learning_rate.h"
#include "math/matrix.h"

bool nearlyEqual(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) < eps;
}

bool verifyMatrix(const Matrix &mat, const std::vector<float> &expected_data)
{
    std::vector<float> actual_data = mat.getData();
    if (actual_data.size() != expected_data.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < actual_data.size(); ++i)
    {
        if (!nearlyEqual(actual_data[i], expected_data[i]))
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
    return verifyMatrix(res, {6.0f, 8.0f, 10.0f, 12.0f});
}

bool testMatrixSubtraction(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {5.0f, 6.0f, 7.0f, 8.0f}, exec_target);
    Matrix mat_b(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix res = mat_a - mat_b;
    return verifyMatrix(res, {4.0f, 4.0f, 4.0f, 4.0f});
}

bool testMatrixMultiplication(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {2.0f, 0.0f, 1.0f, 2.0f}, exec_target);
    Matrix res = mat_a * mat_b;
    return verifyMatrix(res, {4.0f, 4.0f, 10.0f, 8.0f});
}

bool testMatmulAdd(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {2.0f, 0.0f, 1.0f, 2.0f}, exec_target);
    Matrix mat_bias(1, 2, {5.0f, 6.0f}, exec_target);

    Matrix res = mat_a.matmulAdd(mat_b, mat_bias);
    return verifyMatrix(res, {9.0f, 10.0f, 15.0f, 14.0f});
}

bool testSgdUpdate(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad(2, 2, {0.5f, -1.0f, 2.0f, -3.0f}, exec_target);

    mat_a.sgdUpdate(grad, 0.1f, 100.0f);
    return verifyMatrix(mat_a, {0.95f, 2.1f, 2.8f, 4.3f});
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

bool testReluBackward(Execution_Target exec_target)
{
    ReLU relu_layer(exec_target);
    Matrix input_mat(2, 2, {-1.5f, 0.0f, 2.0f, -0.5f}, exec_target);
    Matrix grad_out(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);

    relu_layer.forward(input_mat);
    Matrix grad_in = relu_layer.backward(grad_out);

    return verifyMatrix(grad_in, {0.0f, 0.0f, 3.0f, 0.0f});
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

bool testNormalizeAndActivations(Execution_Target exec_target)
{
    Matrix vec_mat(1, 2, {3.0f, 4.0f}, exec_target);
    Matrix norm_res = vec_mat.normalize();
    bool norm_ok = verifyMatrix(norm_res, {0.6f, 0.8f});

    Matrix act_mat(2, 2, {-1.0f, 2.0f, 0.0f, -3.0f}, exec_target);
    Matrix relu_res = act_mat.relu();
    bool relu_ok = verifyMatrix(relu_res, {0.0f, 2.0f, 0.0f, 0.0f});

    Matrix softmax_in(1, 2, {0.0f, 0.0f}, exec_target);
    Matrix softmax_res = softmax_in.softmax();
    bool softmax_ok = verifyMatrix(softmax_res, {0.5f, 0.5f});

    return norm_ok && relu_ok && softmax_ok;
}


bool testNeuralNetworkPipeline(Execution_Target exec_target)
{
    Neural_Network net_inst(exec_target);

    auto layer_1 = std::make_unique<Layer>(2, 2, exec_target);
    layer_1->setWeights(Matrix(2, 2, {0.5f, -0.5f, 0.5f, 0.5f}, exec_target));
    layer_1->setBiases(Matrix(1, 2, {0.1f, 0.1f}, exec_target));

    auto relu_layer = std::make_unique<ReLU>(exec_target);

    auto layer_2 = std::make_unique<Layer>(2, 1, exec_target);
    layer_2->setWeights(Matrix(2, 1, {1.0f, 2.0f}, exec_target));
    layer_2->setBiases(Matrix(1, 1, {0.0f}, exec_target));

    net_inst.addLayer(std::move(layer_1));
    net_inst.addLayer(std::move(relu_layer));
    net_inst.addLayer(std::move(layer_2));

    Matrix input_mat(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix target_mat(1, 1, {1.0f}, exec_target);
    MSE_Cost cost_fn;
    Learning_Rate lr(0.01, Decay_Mode::NO_DECAY);
    Adam_Optimizer optimizer(lr);

    net_inst.trainStep(input_mat, target_mat, cost_fn, lr.getCurrentRate(), optimizer);
    float initial_loss = net_inst.evaluate(input_mat, target_mat, cost_fn);
    net_inst.trainStep(input_mat, target_mat, cost_fn, lr.getCurrentRate(), optimizer);
    float second_loss = net_inst.evaluate(input_mat, target_mat, cost_fn);
    return !std::isnan(initial_loss) && !std::isnan(second_loss) && (second_loss <= initial_loss);
}

bool testMatmulTransA(Execution_Target exec_target)
{
    Matrix mat_a(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, exec_target);
    Matrix mat_b(3, 2, {1.0f, 0.0f, 2.0f, 1.0f, 0.0f, 1.0f}, exec_target);

    Matrix res = mat_a.matmulTransA(mat_b);
    return verifyMatrix(res, {7.0f, 8.0f, 10.0f, 10.0f});
}

bool testMatmulTransB(Execution_Target exec_target)
{
    Matrix mat_a(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, exec_target);
    Matrix mat_b(2, 3, {1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f}, exec_target);

    Matrix res = mat_a.matmulTransB(mat_b);
    return verifyMatrix(res, {4.0f, 4.0f, 10.0f, 13.0f});
}

bool testGelu(Execution_Target exec_target)
{
    Matrix input_mat(1, 4, {0.0f, 1.0f, -1.0f, 2.0f}, exec_target);
    Matrix gelu_res = input_mat.gelu();

    return verifyMatrix(gelu_res, {0.0f, 0.8412316f, -0.1587684f, 1.9546059f});
}

bool testGeluBackward(Execution_Target exec_target)
{
    Matrix input_mat(1, 2, {0.0f, 1.0f}, exec_target);
    Matrix grad_out(1, 2, {1.0f, 1.0f}, exec_target);

    Matrix grad_in = input_mat.geluBackward(grad_out);

    return verifyMatrix(grad_in, {0.5f, 1.0829548f});
}

bool testConv2d(Execution_Target exec_target)
{
    Matrix input_mat(1, 9, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, exec_target);

    Matrix weights(1, 4, {1.0f, 0.0f, 0.0f, 1.0f}, exec_target);

    Matrix bias(1, 1, {0.0f}, exec_target);

    Matrix out_mat = input_mat.conv2d(weights, bias, 3, 3, 1, 1, 2, 1, 0);

    return verifyMatrix(out_mat, {6.0f, 8.0f,
                                  12.0f, 14.0f});
}

bool testMaxPool2d(Execution_Target exec_target)
{
    Matrix input_mat(1, 16, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}, exec_target);

    auto [out_mat, mask_mat] = input_mat.maxpool2d(4, 4, 1, 2, 2, 0);

    return verifyMatrix(out_mat, {6.0f, 8.0f,
                                  14.0f, 16.0f});
}

bool testBatchNorm(Execution_Target exec_target)
{
    Batch_Norm_Layer bn_layer(1, 1e-5f, 0.1f, exec_target);
    Matrix input_mat(3, 1, {1.0f, 2.0f, 3.0f}, exec_target);

    Matrix out_mat = bn_layer.forward(input_mat);
    bool forward_ok = verifyMatrix(out_mat, {-1.2247f, 0.0f, 1.2247f});

    Matrix grad_out(3, 1, {1.0f, 2.0f, 1.0f}, exec_target);
    Matrix grad_in = bn_layer.backward(grad_out);
    bool backward_ok = verifyMatrix(grad_in, {-0.4082f, 0.8165f, -0.4082f});

    return forward_ok && backward_ok;
}

bool testGlobalAvgPool2d(Execution_Target exec_target)
{
    Matrix input_mat(1, 8, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, exec_target);

    Matrix out_mat = input_mat.globalAvgPool2d(2, 2, 2);

    return verifyMatrix(out_mat, {4.0f, 5.0f});
}

bool testConv2dBackward(Execution_Target exec_target)
{
    Matrix input_mat(1, 9, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, exec_target);

    Matrix weights(1, 4, {1.0f, 0.0f, 0.0f, 1.0f}, exec_target);

    Matrix grad_out(1, 4, {1.0f, 1.0f, 1.0f, 1.0f}, exec_target);

    Matrix grad_in = grad_out.conv2dBackwardInput(weights, 3, 3, 1, 2, 2, 1, 2, 1, 0);

    Matrix grad_weights(1, 4, exec_target);
    Matrix grad_biases(1, 1, exec_target);

    input_mat.conv2dBackwardWeight(grad_out, grad_weights, grad_biases, 3, 3, 1, 2, 2, 1, 2, 1, 0);

    bool grad_in_ok = verifyMatrix(grad_in, {1.0f, 1.0f, 0.0f,
                                             1.0f, 2.0f, 1.0f,
                                             0.0f, 1.0f, 1.0f});

    bool grad_w_ok = verifyMatrix(grad_weights, {12.0f, 16.0f,
                                                 24.0f, 28.0f});

    bool grad_b_ok = verifyMatrix(grad_biases, {4.0f});

    return grad_in_ok && grad_w_ok && grad_b_ok;
}

bool testMaxPool2dBackward(Execution_Target exec_target)
{
    Matrix input_mat(1, 16, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}, exec_target);

    auto [out_mat, mask_mat] = input_mat.maxpool2d(4, 4, 1, 2, 2, 0);

    Matrix grad_out(1, 4, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);

    Matrix grad_in = grad_out.maxpool2dBackward(mask_mat, 4, 4, 1, 2, 2, 2, 2, 0);

    return verifyMatrix(grad_in, {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f, 2.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 3.0f, 0.0f, 4.0f});
}

bool testGlobalAvgPool2dBackward(Execution_Target exec_target)
{
    Matrix grad_out(1, 2, {4.0f, 8.0f}, exec_target);

    Matrix grad_in = grad_out.globalAvgPool2dBackward(2, 2, 2);

    return verifyMatrix(grad_in, {1.0f, 2.0f,
                                  1.0f, 2.0f,
                                  1.0f, 2.0f,
                                  1.0f, 2.0f});
}

bool testLearningRate()
{
    Learning_Rate lr_no_decay(0.01f, Decay_Mode::NO_DECAY);
    for (int epoch = 0; epoch < 5; ++epoch)
    {
        if (!nearlyEqual(lr_no_decay.getCurrentRate(), 0.01f))
        {
            return false;
        }
        lr_no_decay.step();
    }

    Learning_Rate lr_step(0.01f, Decay_Mode::STEP_DECAY, 0.1f, 5);
    if (!nearlyEqual(lr_step.getCurrentRate(), 0.01f))
    {
        return false;
    }
    for (int epoch = 0; epoch < 5; ++epoch)
    {
        lr_step.step();
    }
    if (!nearlyEqual(lr_step.getCurrentRate(), 0.001f))
    {
        return false;
    }

    Learning_Rate lr_multi(0.01f, Decay_Mode::MULTI_STEP_DECAY, 0.1f, std::vector<float>{3.0f, 7.0f});
    if (!nearlyEqual(lr_multi.getCurrentRate(), 0.01f))
    {
        return false;
    }
    for (int epoch = 0; epoch < 3; ++epoch)
    {
        lr_multi.step();
    }
    if (!nearlyEqual(lr_multi.getCurrentRate(), 0.001f))
    {
        return false;
    }

    Learning_Rate lr_exp(0.1f, Decay_Mode::EXPONENTIAL_DECAY, 0.9f);
    if (!nearlyEqual(lr_exp.getCurrentRate(), 0.1f))
    {
        return false;
    }
    lr_exp.step();
    if (!nearlyEqual(lr_exp.getCurrentRate(), 0.09f))
    {
        return false;
    }

    Learning_Rate lr_cos(0.1f, Decay_Mode::COSINE_ANNEALING, 0.1f, 10, 10, 0.0f);
    if (!nearlyEqual(lr_cos.getCurrentRate(), 0.1f))
    {
        return false;
    }
    for (int epoch = 0; epoch < 5; ++epoch)
    {
        lr_cos.step();
    }
    if (!nearlyEqual(lr_cos.getCurrentRate(), 0.05f))
    {
        return false;
    }

    Learning_Rate lr_poly(0.1f, Decay_Mode::POLYNOMIAL_DECAY, 0.1f, 10, 10, 0.0f);
    if (!nearlyEqual(lr_poly.getCurrentRate(), 0.1f))
    {
        return false;
    }
    for (int epoch = 0; epoch < 5; ++epoch)
    {
        lr_poly.step();
    }
    if (!nearlyEqual(lr_poly.getCurrentRate(), 0.05f))
    {
        return false;
    }

    Learning_Rate lr_plateau(0.01f, Decay_Mode::REDUCE_ON_PLATEAU, 0.5f, 3, 1e-5f, true);
    lr_plateau.step(50.0f);
    lr_plateau.step(50.0f);
    lr_plateau.step(50.0f);
    if (!nearlyEqual(lr_plateau.getCurrentRate(), 0.01f))
    {
        return false;
    }
    lr_plateau.step(50.0f);
    if (!nearlyEqual(lr_plateau.getCurrentRate(), 0.005f))
    {
        return false;
    }
    return true;
}

bool testSgdOptimizer(Execution_Target exec_target)
{
    SGD_Optimizer optimizer(0.1f, 100.0f);

    Matrix param_mat(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad_mat(2, 2, {0.5f, -1.0f, 2.0f, -3.0f}, exec_target);

    std::vector<std::pair<Matrix *, Matrix *>> param_grad_pairs = {
        {&param_mat, &grad_mat}};

    optimizer.step(param_grad_pairs);

    return verifyMatrix(param_mat, {0.95f, 2.1f, 2.8f, 4.3f});
}

bool testAdamOptimizer(Execution_Target exec_target)
{
    Adam_Optimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 100.0f);

    Matrix param_mat(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix grad_mat(1, 2, {0.1f, -0.2f}, exec_target);

    std::vector<std::pair<Matrix *, Matrix *>> param_grad_pairs = {
        {&param_mat, &grad_mat}};

    optimizer.step(param_grad_pairs);

    bool first_step_ok = verifyMatrix(param_mat, {0.999f, 2.001f});

    optimizer.reset();

    Matrix param_reset(1, 2, {1.0f, 2.0f}, exec_target);
    std::vector<std::pair<Matrix *, Matrix *>> reset_pairs = {
        {&param_reset, &grad_mat}};

    optimizer.step(reset_pairs);

    bool reset_ok = verifyMatrix(param_reset, {0.999f, 2.001f});

    return first_step_ok && reset_ok;
}

void runTestSuite(Execution_Target exec_target, const std::string &target_name)
{
    std::cout << "--- Running Tests on " << target_name << " ---\n";

    std::cout << "Matrix Addition: " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Matrix Subtraction: " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Matrix Multiplication: " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MatmulAdd Fusion: " << (testMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MatmulTransA (A^T * B): " << (testMatmulTransA(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MatmulTransB (A * B^T): " << (testMatmulTransB(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Scalar Operations: " << (testScalarOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Hadamard Operations: " << (testHadamardOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Transpose & Inverse: " << (testTransposeAndInverse(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Normalize & Activations: " << (testNormalizeAndActivations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "ReLU Backward: " << (testReluBackward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "GELU Forward: " << (testGelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "GELU Backward: " << (testGeluBackward(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "Conv2D Forward: " << (testConv2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Conv2D Backward: " << (testConv2dBackward(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "MaxPool2D Forward: " << (testMaxPool2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MaxPool2D Backward: " << (testMaxPool2dBackward(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "GlobalAvgPool2D Forward: " << (testGlobalAvgPool2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "GlobalAvgPool2D Backward: " << (testGlobalAvgPool2dBackward(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "Neural Network Pipeline: " << (testNeuralNetworkPipeline(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "BatchNorm Forward & Backward: " << (testBatchNorm(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Learning Rate Scheduler: " << (testLearningRate() ? "PASS" : "FAIL") << "\n\n";

    std::cout << "SGD Optimizer Step: " << (testSgdOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Adam Optimizer Step & Reset: " << (testAdamOptimizer(exec_target) ? "PASS" : "FAIL") << "\n";
}

int main()
{
    runTestSuite(Execution_Target::VULKAN_GPU, "GPU");
    runTestSuite(Execution_Target::CPU, "CPU");
    return 0;
}