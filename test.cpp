#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <cstdio>

#include "math/matrix.h"
#include "engine/execution_engine.h"

bool nearlyEqual(float a, float b, float eps = 1e-3f)
{
    return std::fabs(a - b) < eps;
}

bool verifyMatrix(const Matrix &mat, const std::vector<float> &expected_data)
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

bool testMatrixIO(Execution_Target exec_target)
{
    std::string temp_file = "temp_test_matrix.bin";
    Matrix mat_original(2, 3, {1.0f, -2.0f, 3.5f, 4.2f, 5.0f, -6.1f}, exec_target);

    std::ofstream out_file(temp_file, std::ios::binary);
    if (!out_file.is_open())
        return false;
    mat_original.saveMatrix(out_file);
    out_file.close();

    std::ifstream in_file(temp_file, std::ios::binary);
    if (!in_file.is_open())
        return false;
    Matrix mat_loaded = Matrix::loadMatrix(in_file, exec_target);
    in_file.close();
    std::remove(temp_file.c_str());

    return verifyMatrix(mat_loaded, {1.0f, -2.0f, 3.5f, 4.2f, 5.0f, -6.1f});
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

bool testLinearForward(Execution_Target exec_target)
{
    Matrix input_x(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix weights_w(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, exec_target);
    Matrix biases_b(1, 3, {0.1f, 0.2f, 0.3f}, exec_target);
    Matrix output_y(1, 3, exec_target);

    input_x.linearForward(weights_w, biases_b, output_y);
    return verifyMatrix(output_y, {9.1f, 12.2f, 15.3f});
}

bool testLinearBackward(Execution_Target exec_target)
{
    Matrix input_x(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix weights_w(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, exec_target);
    Matrix grad_y(1, 3, {1.0f, 1.0f, 1.0f}, exec_target);

    Matrix grad_x(1, 2, exec_target);
    grad_y.linearBackwardInput(weights_w, grad_x);
    bool bwd_input_ok = verifyMatrix(grad_x, {6.0f, 15.0f});

    Matrix grad_w(2, 3, exec_target);
    Matrix grad_b(1, 3, exec_target);
    input_x.linearBackwardWeightBias(grad_y, grad_w, grad_b);
    bool bwd_wb_ok = verifyMatrix(grad_w, {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}) &&
                     verifyMatrix(grad_b, {1.0f, 1.0f, 1.0f});

    return bwd_input_ok && bwd_wb_ok;
}

bool testConv2d(Execution_Target exec_target)
{
    Matrix input_mat(1, 9, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, exec_target);
    Matrix weights(1, 4, {1.0f, 0.0f, 0.0f, 1.0f}, exec_target);
    Matrix bias(1, 1, {0.0f}, exec_target);

    Matrix out_mat = input_mat.conv2d(weights, bias, 3, 3, 1, 1, 2, 1, 0);
    return verifyMatrix(out_mat, {6.0f, 8.0f, 12.0f, 14.0f});
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

    bool grad_in_ok = verifyMatrix(grad_in, {1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f});
    bool grad_w_ok = verifyMatrix(grad_weights, {12.0f, 16.0f, 24.0f, 28.0f});
    bool grad_b_ok = verifyMatrix(grad_biases, {4.0f});

    return grad_in_ok && grad_w_ok && grad_b_ok;
}

bool testMaxPool2d(Execution_Target exec_target)
{
    Matrix input_mat(1, 16, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}, exec_target);
    auto [out_mat, mask_mat] = input_mat.maxpool2d(4, 4, 1, 2, 2, 0);
    bool fwd_ok = verifyMatrix(out_mat, {6.0f, 8.0f, 14.0f, 16.0f});

    Matrix grad_out(1, 4, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad_in = grad_out.maxpool2dBackward(mask_mat, 4, 4, 1, 2, 2, 2, 2, 0);
    bool bwd_ok = verifyMatrix(grad_in, {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 2.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 3.0f, 0.0f, 4.0f});

    return fwd_ok && bwd_ok;
}

bool testGlobalAvgPool2d(Execution_Target exec_target)
{
    Matrix input_mat(1, 8, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, exec_target);
    Matrix out_mat = input_mat.globalAvgPool2d(2, 2, 2);
    bool fwd_ok = verifyMatrix(out_mat, {4.0f, 5.0f});

    Matrix grad_out(1, 2, {4.0f, 8.0f}, exec_target);
    Matrix grad_in = grad_out.globalAvgPool2dBackward(2, 2, 2);
    bool bwd_ok = verifyMatrix(grad_in, {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f});

    return fwd_ok && bwd_ok;
}

bool testBatchNorm(Execution_Target exec_target)
{
    Matrix input_mat(3, 1, {1.0f, 2.0f, 3.0f}, exec_target);
    Matrix gamma(1, 1, {1.0f}, exec_target);
    Matrix beta(1, 1, {0.0f}, exec_target);

    Matrix running_mean(1, 1, {0.0f}, exec_target);
    Matrix running_var(1, 1, {1.0f}, exec_target);
    Matrix batch_mean(1, 1, exec_target);
    Matrix batch_var(1, 1, exec_target);
    Matrix x_hat(3, 1, exec_target);

    Matrix out_mat = input_mat.batchNormForward(gamma, beta, running_mean, running_var, batch_mean, batch_var, x_hat, 1e-5f, 0.1f, true);
    bool forward_ok = verifyMatrix(out_mat, {-1.2247f, 0.0f, 1.2247f});

    Matrix grad_out(3, 1, {1.0f, 2.0f, 1.0f}, exec_target);
    Matrix grad_gamma(1, 1, exec_target);
    Matrix grad_beta(1, 1, exec_target);

    Matrix grad_in = input_mat.batchNormBackward(grad_out, gamma, batch_var, x_hat, grad_gamma, grad_beta, 1e-5f);
    bool backward_ok = verifyMatrix(grad_in, {-0.4082f, 0.8165f, -0.4082f});

    return forward_ok && backward_ok;
}

bool testSgdUpdate(Execution_Target exec_target)
{
    Matrix param_mat(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix grad(2, 2, {0.5f, -1.0f, 2.0f, -3.0f}, exec_target);

    param_mat.sgdUpdate(grad, 0.1f, 100.0f);
    return verifyMatrix(param_mat, {0.95f, 2.1f, 2.8f, 4.3f});
}

bool testAdamUpdate(Execution_Target exec_target)
{
    Matrix param_mat(1, 2, {1.0f, 2.0f}, exec_target);
    Matrix grad_mat(1, 2, {0.1f, -0.2f}, exec_target);
    Matrix m_mat(1, 2, {0.0f, 0.0f}, exec_target);
    Matrix v_mat(1, 2, {0.0f, 0.0f}, exec_target);

    param_mat.adamUpdate(grad_mat, m_mat, v_mat, 0.001f, 0.9f, 0.999f, 1e-8f, 1, 100.0f);

    bool adam_ok = verifyMatrix(param_mat, {0.999f, 2.001f});
    bool m_ok = verifyMatrix(m_mat, {0.01f, -0.02f});
    bool v_ok = verifyMatrix(v_mat, {0.00001f, 0.00004f});

    return adam_ok && m_ok && v_ok;
}

bool testMatmulAdd(Execution_Target exec_target)
{
    Matrix mat_a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, exec_target);
    Matrix mat_b(2, 2, {2.0f, 0.0f, 1.0f, 2.0f}, exec_target);
    Matrix mat_bias(1, 2, {5.0f, 6.0f}, exec_target);

    Matrix res = mat_a.matmulAdd(mat_b, mat_bias);
    return verifyMatrix(res, {9.0f, 10.0f, 15.0f, 14.0f});
}

bool testBatchNorm2d(Execution_Target exec_target)
{
    Matrix input_mat(2, 2, {1.0f, 3.0f, 5.0f, 7.0f}, exec_target);
    Matrix gamma(1, 1, {1.0f}, exec_target);
    Matrix beta(1, 1, {0.0f}, exec_target);

    Matrix running_mean(1, 1, {0.0f}, exec_target);
    Matrix running_var(1, 1, {1.0f}, exec_target);
    Matrix batch_mean(1, 1, exec_target);
    Matrix batch_var(1, 1, exec_target);
    Matrix x_hat(2, 2, exec_target);

    Matrix out_mat = input_mat.batchNorm2dForward(gamma, beta, running_mean, running_var, batch_mean, batch_var, x_hat, 1, 2, 1, 1e-5f, 0.1f, true);
    bool forward_ok = verifyMatrix(out_mat, {-1.34164f, -0.44721f, 0.44721f, 1.34164f});

    Matrix grad_out(2, 2, {1.0f, 0.0f, 0.0f, 0.0f}, exec_target);
    Matrix grad_gamma(1, 1, exec_target);
    Matrix grad_beta(1, 1, exec_target);

    Matrix grad_in = input_mat.batchNorm2dBackward(grad_out, gamma, batch_var, x_hat, grad_gamma, grad_beta, 1, 2, 1, 1e-5f);
    bool backward_ok = verifyMatrix(grad_in, {0.13416f, -0.17889f, -0.04472f, 0.08944f});

    return forward_ok && backward_ok;
}

void runTestSuite(Execution_Target exec_target, const std::string &target_name)
{
    std::cout << "--- Running Tests on " << target_name << " ---\n";

    std::cout << "[Basic Arithmetics]\n";
    std::cout << "  Matrix Addition: " << (testMatrixAddition(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Subtraction: " << (testMatrixSubtraction(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix Multiplication: " << (testMatrixMultiplication(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Scalar Operations: " << (testScalarOperations(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Hadamard Operations: " << (testHadamardOperations(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[Transformations & Utility]\n";
    std::cout << "  Transpose & Inverse: " << (testTransposeAndInverse(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Normalize: " << (testNormalize(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Matrix I/O (Save/Load): " << (testMatrixIO(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[Activations]\n";
    std::cout << "  ReLU Forward & Backward: " << (testRelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  GELU Forward & Backward: " << (testGelu(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Softmax Forward & Backward: " << (testSoftmax(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[Neural Network Layers]\n";
    std::cout << "  Linear Forward: " << (testLinearForward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Linear Backward (Input & Weights): " << (testLinearBackward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Conv2D Forward: " << (testConv2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Conv2D Backward: " << (testConv2dBackward(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  MaxPool2D Forward & Backward: " << (testMaxPool2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  GlobalAvgPool2D Forward & Backward: " << (testGlobalAvgPool2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm Forward & Backward: " << (testBatchNorm(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  BatchNorm2D Forward & Backward: " << (testBatchNorm2d(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  MatmulAdd (Fused FC): " << (testMatmulAdd(exec_target) ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[Optimizers]\n";
    std::cout << "  SGD Update: " << (testSgdUpdate(exec_target) ? "PASS" : "FAIL") << "\n";
    std::cout << "  Adam Update: " << (testAdamUpdate(exec_target) ? "PASS" : "FAIL") << "\n\n";
}

int main()
{
    runTestSuite(Execution_Target::VULKAN_GPU, "GPU");
    runTestSuite(Execution_Target::CPU, "CPU");
    return 0;
}