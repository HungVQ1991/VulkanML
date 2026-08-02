#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

#include "batch_norm_layer.h"
#include "layer.h"
#include "relu.h"
#include "softmax.h"
#include "cost_function.h"
#include "neural_network.h"
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

bool testLayerUpdate(Execution_Target exec_target)
{
    Layer layer_inst(2, 2, exec_target);

    layer_inst.setWeights(Matrix(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target));
    layer_inst.setBiases(Matrix(1, 2, {5.0f, 6.0f}, exec_target));

    layer_inst.weights_gradient = Matrix(2, 2, {0.5f, -1.0f, 2.0f, -3.0f}, exec_target);
    layer_inst.biases_gradient = Matrix(1, 2, {4.0f, -2.0f}, exec_target);

    layer_inst.update(0.1f, 100.0f);

    Matrix expected_w(2, 2, {0.95f, 2.1f, 2.8f, 4.3f}, exec_target);
    Matrix expected_b(1, 2, {4.6f, 6.2f}, exec_target);

    bool w_ok = verifyMatrix(layer_inst.getWeights(), expected_w.getData());
    bool b_ok = verifyMatrix(layer_inst.getBiases(), expected_b.getData());

    return w_ok && b_ok;
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

    float initial_loss = net_inst.trainStep(input_mat, target_mat, cost_fn, 0.01f, true);
    float second_loss = net_inst.trainStep(input_mat, target_mat, cost_fn, 0.01f, true);

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

void runTestSuite(Execution_Target exec_target, const std::string &target_name)
{
    std::cout << "--- Running Tests on " << target_name << " ---\n";

    std::cout << "Matrix Addition: " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Matrix Subtraction: " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Matrix Multiplication: " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MatmulAdd Fusion: " << (testMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MatmulTransA (A^T * B): " << (testMatmulTransA(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "MatmulTransB (A * B^T): " << (testMatmulTransB(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "SGD Update: " << (testSgdUpdate(exec_target) ? "PASS" : "FAIL") << "\n";
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

    std::cout << "Layer Gradient Update: " << (testLayerUpdate(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "Neural Network Pipeline: " << (testNeuralNetworkPipeline(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "BatchNorm Forward & Backward: " << (testBatchNorm(exec_target) ? "PASS" : "FAIL") << "\n\n";
}

int main()
{
    runTestSuite(Execution_Target::VULKAN_GPU, "GPU");
    runTestSuite(Execution_Target::CPU, "CPU");
    std::cout << "Hello";
    return 0;
}