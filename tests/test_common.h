#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "chexlocalize/bitmap_font.h"
#include "chexlocalize/chest_seg_dataset.h"
#include "chexlocalize/chexpert_classifier.h"
#include "chexlocalize/chexpert_dataset.h"
#include "chexlocalize/evaluator.h"
#include "chexlocalize/gradcam_engine.h"
#include "chexlocalize/quaternion_blender.h"
#include "chexlocalize/rle_decoder.h"
#include "cost_function/dice_bce_cost.h"
#include "cost_function/dice_cost.h"
#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "engine/graph_optimizer.h"
#include "engine/loss_scaler.h"
#include "engine/shader_dictionary.h"
#include "engine/shader_generator.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_sub_allocator.h"
#include "helper/cost_function.h"
#include "helper/layer.h"
#include "helper/learning_rate.h"
#include "helper/llm.h"
#include "helper/model_quantizer.h"
#include "helper/optimizer.h"
#include "helper/runtime.h"
#include "helper/trainer.h"
#include "helper/training_profiler.h"
#include "helper/user_preferences.h"
#include "layer/instance_norm2d_layer.h"
#include "layer/leaky_relu_layer.h"
#include "layer/nn_unet_layer.h"
#include "layer/u_net_layer.h"
#include "layer/upsample2d_layer.h"
#include "llm/binary_dataset.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"
#include "neural_network.h"
#include "ops/activation/activation.h"
#include "ops/dropout/dropout.h"
#include "ops/linear/linear.h"
#include "ops/llm/attention.h"
#include "ops/llm/embedding.h"
#include "ops/llm/rope.h"
#include "ops/llm/swiglu.h"
#include "ops/loss/dice_bce.h"
#include "ops/loss/loss_ops.h"
#include "ops/normalization/batchnorm.h"
#include "ops/normalization/instancenorm.h"
#include "ops/normalization/rmsnorm.h"
#include "ops/optimizer/adam_step.h"
#include "ops/optimizer/sgd_step.h"
#include "ops/vision/conv2d.h"
#include "ops/vision/pooling.h"
#include "population.h"
#include "rl/c_api.h"
#include "rl/cartpole_env.h"
#include "rl/dqn_agent.h"
#include "rl/ppo_agent.h"
#include "rl/replay_buffer.h"
#include "rl/transition.h"
#include "tokenizer/bpe_tokenizer.h"
#include "training_context.h"

struct Test_Result_Counter
{
    int passed = 0;
    int failed = 0;

    void record(const char *label, bool ok);
};

bool nearlyEqual(float a, float b, float eps = 1e-3f);
bool verifyMatrix(const Tensor &mat, const std::vector<float> &expected_data, float eps = 1e-3f);
double computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b);
std::vector<float> generateRandomVector(size_t count, float low = -1.0f, float high = 1.0f, unsigned int seed = 42);
std::vector<float> generateRandomNormalVector(size_t count, float mean = 0.0f, float stddev = 1.0f, unsigned int seed = 42, float min_abs = 0.0f);
std::vector<float> generateRandomBinaryVector(size_t count, float pos_ratio = 0.5f, unsigned int seed = 123);
bool failTestStep(std::string_view suite_tag, std::string_view message);
void flushGpuEngine(bool wait_idle = false);
std::filesystem::path makeUniqueTestTempPath(std::string_view prefix, std::string_view ext = ".bin");
void flushIfGpu(Execution_Target target, bool wait_idle = false);
Tensor castTensorToFp16(const Tensor &src);
void convertTensorToFp16InPlace(Tensor &tensor);
bool verifyVectorsClose(const std::vector<float> &a, const std::vector<float> &b, float eps = 1e-3f);
bool verifyTensorsClose(const Tensor &a, const Tensor &b, float eps = 1e-3f);
bool hasVulkanDevice();
bool checkVectorsClose(const std::vector<float> &a, const std::vector<float> &b, float tol, const char *tag, const char *step);
double computeNumericalGradientMaxRelErr(
    const std::vector<float> &base_data,
    float delta,
    const std::vector<float> &analytical_grad,
    const std::function<float(const std::vector<float> &)> &eval_loss);

template <typename ExceptionT = std::exception, typename Fn>
bool expectThrows(Fn &&fn)
{
    try
    {
        fn();
    }
    catch (const ExceptionT &)
    {
        return true;
    }
    return false;
}

struct Scoped_Coop_Mat_Guard
{
    bool prev = Execution_Engine::getInstance().isCooperativeMatrixEnabled();
    ~Scoped_Coop_Mat_Guard() { Execution_Engine::getInstance().setCooperativeMatrixEnabled(prev); }
};

struct Scoped_Temp_File_Guard
{
    std::vector<std::string> files;
    std::vector<std::filesystem::path> dirs;
    explicit Scoped_Temp_File_Guard(
        std::vector<std::string> f = {},
        std::vector<std::filesystem::path> d = {})
        : files(std::move(f)), dirs(std::move(d))
    {
    }
    void clean() const;
    ~Scoped_Temp_File_Guard() { clean(); }
};

struct Scoped_Logger_Guard
{
    Logger_Config saved = Logger::getConfig();
    bool reset_counters_on_exit = false;
    explicit Scoped_Logger_Guard(bool reset_counters = false) : reset_counters_on_exit(reset_counters) {}
    ~Scoped_Logger_Guard();
};

struct Scoped_Engine_Flags_Guard
{
    bool fp16 = Runtime::isFloat16Enabled();
    bool coop = Runtime::isCooperativeMatrixEnabled();
    bool stat = Runtime::isStaticGraphEnabled();
    bool single = Runtime::isSingleSubmitEnabled();
    bool fused = Execution_Engine::getInstance().isFusedGemmAdamEnabled();
    bool profiling = Execution_Engine::getInstance().isGpuProfilingEnabled();
    bool graph_cache = Execution_Engine::getInstance().isGraphCacheEnabled();
    ~Scoped_Engine_Flags_Guard();
};

bool testEmbeddingLayer(Execution_Target exec_target);
bool testFusedCrossEntropy(Execution_Target exec_target);
bool testMatrixAddition(Execution_Target exec_target);
bool testMatrixSubtraction(Execution_Target exec_target);
bool testMatrixMultiplication(Execution_Target exec_target);
bool testScalarOperations(Execution_Target exec_target);
bool testHadamardOperations(Execution_Target exec_target);
bool testTransposeAndInverse(Execution_Target exec_target);
bool testNormalize(Execution_Target exec_target);
bool testMatmulAdd(Execution_Target exec_target);
bool testBatchedTensorMatmul(Execution_Target exec_target);
bool testBatchedTensorMatmulAdd(Execution_Target exec_target);
bool testRelu(Execution_Target exec_target);
bool testSigmoidLayer(Execution_Target exec_target);
bool testGelu(Execution_Target exec_target);
bool testSoftmax(Execution_Target exec_target);
bool testMseLoss(Execution_Target exec_target);
bool testMaeLoss(Execution_Target exec_target);
bool testBceLoss(Execution_Target exec_target);
bool testCceLoss(Execution_Target exec_target);
bool testHuberLoss(Execution_Target exec_target);
bool testAslLoss(Execution_Target exec_target);
bool testLinearLayer(Execution_Target exec_target);
bool testConv2dLayer(Execution_Target exec_target);
bool testConv2dLayerFp16(Execution_Target exec_target);
bool testConv2dFp16CoopMatMultiChannelParity(Execution_Target exec_target);
bool testNativeFp16ZeroCastPipeline(Execution_Target exec_target);
bool testPopulationFp16(Execution_Target exec_target);
bool testMaxPool2dLayer(Execution_Target exec_target);
bool testAvgPool2dLayer(Execution_Target exec_target);
bool testGlobalAvgPool2dLayer(Execution_Target exec_target);
bool testGlobalMaxPool2d(Execution_Target exec_target);
bool testBatchNormLayer(Execution_Target exec_target);
bool testBatchNorm2dLayer(Execution_Target exec_target);
bool testRmsNormLayer(Execution_Target exec_target);
bool testRoPE(Execution_Target exec_target);
bool testSwiGluLayer(Execution_Target exec_target);
bool testFlashAttention(Execution_Target exec_target);
bool testBpeTokenizer();
bool testKvCacheManager(Execution_Target exec_target);
bool testTransformerBlock(Execution_Target exec_target);
bool testCausalLMModel(Execution_Target exec_target);
bool testCausalLMTraining(Execution_Target exec_target);
bool testCausalLMFP16Training(Execution_Target exec_target);
bool testResNetBlock2dLayer(Execution_Target exec_target);

void runCoreTestSuite(Execution_Target exec_target, const std::string &target_name, Test_Result_Counter &counter);
void runEngineSystemsSuite(Test_Result_Counter &counter);
void runTernaryQuantSuite(Test_Result_Counter &counter);
void runVisionSegSuite(Test_Result_Counter &counter);
void runHeavyUnetSuite(Test_Result_Counter &counter);
void runRegressionsW2W5Suite(Test_Result_Counter &counter);
void runRegressionsW6W7Suite(Test_Result_Counter &counter);
void runPerfPhasesSuite(Test_Result_Counter &counter);
void runDxInfraSuite(Test_Result_Counter &counter);
void runLayerUpgradesSuite(Test_Result_Counter &counter);
void runSealedReadSuite(Test_Result_Counter &counter);
void runWeightLoaderSuite(Test_Result_Counter &counter);
