#include "test_common.h"

namespace TestDiceBceCost
{
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
    auto pred_raw = generateRandomVector(count, 0.05f, 0.95f, 101);
    auto target_raw = generateRandomBinaryVector(count, 0.4f, 202);

    Tensor pred(1, count, pred_raw, Execution_Target::CPU);
    Tensor target(1, count, target_raw, Execution_Target::CPU);

    Dice_BCE_Cost cost_fn(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    auto grad_data = cost_fn.computeGradient(pred, target).getData();

    double max_rel_err = computeNumericalGradientMaxRelErr(pred_raw, 1e-3f, grad_data, [&](const std::vector<float> &p) {
        Tensor t_p(1, count, p, Execution_Target::CPU);
        return cost_fn.computeLoss(t_p, target);
    });
    return max_rel_err <= 1e-2;
}

bool testForwardGpuParity()
{
    size_t count = 512;
    auto pred_raw = generateRandomVector(count, 0.05f, 0.95f, 303);
    auto target_raw = generateRandomBinaryVector(count, 0.35f, 404);

    Tensor pred_cpu(1, count, pred_raw, Execution_Target::CPU);
    Tensor target_cpu(1, count, target_raw, Execution_Target::CPU);
    Dice_BCE_Cost cost_cpu(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    float loss_cpu = cost_cpu.computeLoss(pred_cpu, target_cpu);

    Tensor pred_gpu(1, count, pred_raw, Execution_Target::VULKAN_GPU);
    Tensor target_gpu(1, count, target_raw, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    float loss_gpu = cost_gpu.computeLoss(pred_gpu, target_gpu);

    return std::fabs(loss_cpu - loss_gpu) <= 1e-4;
}

bool testBackwardGpuParity()
{
    size_t count = 512;
    auto pred_raw = generateRandomVector(count, 0.05f, 0.95f, 505);
    auto target_raw = generateRandomBinaryVector(count, 0.45f, 606);

    Tensor pred_cpu(1, count, pred_raw, Execution_Target::CPU);
    Tensor target_cpu(1, count, target_raw, Execution_Target::CPU);
    Dice_BCE_Cost cost_cpu(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    Tensor grad_cpu = cost_cpu.computeGradient(pred_cpu, target_cpu);

    Tensor pred_gpu(1, count, pred_raw, Execution_Target::VULKAN_GPU);
    Tensor target_gpu(1, count, target_raw, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    Tensor grad_gpu = cost_gpu.computeGradient(pred_gpu, target_gpu);

    return computeMaxAbsDiff(grad_cpu.getData(), grad_gpu.getData()) <= 1e-4;
}

bool testFp16AndCoopmatParity()
{
    size_t count = 1024;
    auto pred_raw = generateRandomVector(count, 0.05f, 0.95f, 707);
    auto target_raw = generateRandomBinaryVector(count, 0.3f, 808);

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
    auto pred_raw = generateRandomVector(count, 0.05f, 0.95f, 909);
    auto target_raw = generateRandomBinaryVector(count, 0.5f, 1010);

    Tensor pred_gpu(1, count, pred_raw, Execution_Target::VULKAN_GPU);
    Tensor target_gpu(1, count, target_raw, Execution_Target::VULKAN_GPU);
    Dice_BCE_Cost cost_gpu(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);

    auto ana_data = cost_gpu.computeGradient(pred_gpu, target_gpu).getData();
    double max_rel_err = computeNumericalGradientMaxRelErr(pred_raw, 1e-3f, ana_data, [&](const std::vector<float> &p) {
        Tensor t_p(1, count, p, Execution_Target::VULKAN_GPU);
        return cost_gpu.computeLoss(t_p, target_gpu);
    });
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
    auto pred_raw = generateRandomVector(count, 0.05f, 0.95f, 1111);
    auto target_raw = generateRandomBinaryVector(count, 0.4f, 2222);

    Tensor p(1, count, pred_raw, Execution_Target::CPU);
    Tensor t(1, count, target_raw, Execution_Target::CPU);
    Tensor g(Execution_Target::CPU);

    float combined_loss = cost->computeLossAndGradient(p, t, g);
    float separate_loss = cost->computeLoss(p, t);
    Tensor separate_g = cost->computeGradient(p, t);

    if (std::fabs(combined_loss - separate_loss) > 1e-6) return false;
    double g_diff = computeMaxAbsDiff(g.getData(), separate_g.getData());
    if (g_diff > 1e-6) return false;

    const std::string checkpoint_file = makeUniqueTestTempPath("test_dice_bce_ckpt").string();
    Scoped_Temp_File_Guard ckpt_guard({checkpoint_file});
    {
        std::ofstream out(checkpoint_file, std::ios::binary);
        cost->saveCheckpoint(out);
    }
    Dice_BCE_Cost loaded_cost(1.0f, 1.0f, 1e-7f, Execution_Target::CPU);
    {
        std::ifstream in(checkpoint_file, std::ios::binary);
        loaded_cost.loadCheckpoint(in);
    }

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
}

namespace TestInstanceNorm
{
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
    uint32_t b_size = 2, h = 16, w = 16, c = 32;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomNormalVector(b_size * h * w * c, 2.0f, 3.0f);
    std::vector<float> gamma_raw = generateRandomNormalVector(c, 1.0f, 0.2f);
    std::vector<float> beta_raw = generateRandomNormalVector(c, 0.0f, 0.1f);

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
    flushGpuEngine();

    double out_err = computeMaxAbsDiff(out_gpu.getData(), out_cpu.getData());
    double mean_err = computeMaxAbsDiff(mean_gpu.getData(), mean_cpu.getData());
    double inv_std_err = computeMaxAbsDiff(inv_std_gpu.getData(), inv_std_cpu.getData());

    return (out_err <= 1e-4 && mean_err <= 1e-4 && inv_std_err <= 1e-4);
}

bool testForwardFp16AndCoopmatParity()
{
    uint32_t b_size = 2, h = 16, w = 16, c = 32;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomNormalVector(b_size * h * w * c, 1.0f, 1.5f);
    std::vector<float> gamma_raw = generateRandomNormalVector(c, 1.0f, 0.1f);
    std::vector<float> beta_raw = generateRandomNormalVector(c, 0.0f, 0.1f);

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
    flushGpuEngine();

    return computeMaxAbsDiff(out_gpu_fp16.getData(), out_cpu.getData()) <= 5e-2;
}

bool testBackwardNumericalGradient()
{
    uint32_t b_size = 2, h = 4, w = 4, c = 3;
    float epsilon = 1e-5f, eps_h = 1e-3f;

    std::vector<float> input_raw = generateRandomNormalVector(b_size * h * w * c, 0.5f, 1.0f);
    std::vector<float> gamma_raw = generateRandomNormalVector(c, 1.2f, 0.3f);
    std::vector<float> beta_raw = generateRandomNormalVector(c, 0.1f, 0.2f);
    std::vector<float> grad_out_raw = generateRandomNormalVector(b_size * h * w * c, 0.0f, 0.5f);

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

    double max_rel_gamma_err = computeNumericalGradientMaxRelErr(gamma_raw, eps_h, dgamma_cpu.getData(), [&](const std::vector<float> &g) {
        return computeLoss(input_raw, g, beta_raw);
    });
    double max_rel_beta_err = computeNumericalGradientMaxRelErr(beta_raw, eps_h, dbeta_cpu.getData(), [&](const std::vector<float> &bt) {
        return computeLoss(input_raw, gamma_raw, bt);
    });
    double max_rel_x_err = computeNumericalGradientMaxRelErr(input_raw, eps_h, dX_cpu.getData(), [&](const std::vector<float> &x) {
        return computeLoss(x, gamma_raw, beta_raw);
    });

    return (max_rel_gamma_err <= 1e-2 && max_rel_beta_err <= 1e-2 && max_rel_x_err <= 1e-2);
}

bool testBackwardGpuParity()
{
    uint32_t b_size = 2, h = 8, w = 8, c = 16;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomNormalVector(b_size * h * w * c, 1.0f, 2.0f);
    std::vector<float> gamma_raw = generateRandomNormalVector(c, 1.0f, 0.2f);
    std::vector<float> beta_raw = generateRandomNormalVector(c, 0.0f, 0.1f);
    std::vector<float> grad_out_raw = generateRandomNormalVector(b_size * h * w * c, 0.0f, 1.0f);

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
    flushGpuEngine();

    double dx_err = computeMaxAbsDiff(dX_gpu.getData(), dX_cpu.getData());
    double dg_err = computeMaxAbsDiff(dgamma_gpu.getData(), dgamma_cpu.getData());
    double db_err = computeMaxAbsDiff(dbeta_gpu.getData(), dbeta_cpu.getData());

    return (dx_err <= 1e-4 && dg_err <= 1e-4 && db_err <= 1e-4);
}

bool testBackwardGpuFp16AndCoopmatParity()
{
    uint32_t b_size = 2, h = 8, w = 8, c = 32;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomNormalVector(b_size * h * w * c, 1.0f, 1.0f);
    std::vector<float> gamma_raw = generateRandomNormalVector(c, 1.0f, 0.2f);
    std::vector<float> beta_raw = generateRandomNormalVector(c, 0.0f, 0.1f);
    std::vector<float> grad_out_raw = generateRandomNormalVector(b_size * h * w * c, 0.0f, 1.0f);

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
    flushGpuEngine();

    double dx_err = computeMaxAbsDiff(dX_gpu.getData(), dX_cpu.getData());
    double dg_err = computeMaxAbsDiff(dgamma_gpu.getData(), dgamma_cpu.getData());
    double db_err = computeMaxAbsDiff(dbeta_gpu.getData(), dbeta_cpu.getData());

    return (dx_err <= 5e-2 && dg_err <= 5e-2 && db_err <= 5e-2);
}

bool testBatchInvariance()
{
    uint32_t h = 8, w = 8, c = 16;
    float epsilon = 1e-5f;
    uint32_t spatial_feat = h * w * c;

    std::vector<float> sample_a = generateRandomNormalVector(spatial_feat, 1.0f, 2.0f);
    std::vector<float> sample_b = generateRandomNormalVector(spatial_feat, 5.0f, 10.0f);

    std::vector<float> batch_ab = sample_a;
    batch_ab.insert(batch_ab.end(), sample_b.begin(), sample_b.end());

    std::vector<float> gamma_raw = generateRandomNormalVector(c, 1.0f, 0.1f);
    std::vector<float> beta_raw = generateRandomNormalVector(c, 0.0f, 0.1f);

    Tensor in_a_gpu(1, spatial_feat, sample_a, Execution_Target::VULKAN_GPU);
    Tensor gamma_gpu(1, c, gamma_raw, Execution_Target::VULKAN_GPU);
    Tensor beta_gpu(1, c, beta_raw, Execution_Target::VULKAN_GPU);
    Tensor mean_a(Execution_Target::VULKAN_GPU), inv_a(Execution_Target::VULKAN_GPU), norm_a(Execution_Target::VULKAN_GPU), out_a(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_a_gpu, gamma_gpu, beta_gpu, mean_a, inv_a, norm_a, out_a, h, w, c, epsilon);

    Tensor in_ab_gpu(2, spatial_feat, batch_ab, Execution_Target::VULKAN_GPU);
    Tensor mean_ab(Execution_Target::VULKAN_GPU), inv_ab(Execution_Target::VULKAN_GPU), norm_ab(Execution_Target::VULKAN_GPU), out_ab(Execution_Target::VULKAN_GPU);

    ops::norm::instanceNorm2dForward(in_ab_gpu, gamma_gpu, beta_gpu, mean_ab, inv_ab, norm_ab, out_ab, h, w, c, epsilon);
    flushGpuEngine();

    auto out_a_data = out_a.getData();
    auto out_ab_data = out_ab.getData();

    std::vector<float> out_ab_slice_a(out_ab_data.begin(), out_ab_data.begin() + spatial_feat);
    return computeMaxAbsDiff(out_a_data, out_ab_slice_a) <= 1e-5;
}

bool testLayerTrainEvalConsistency()
{
    uint32_t b_size = 2, h = 8, w = 8, c = 16;
    float epsilon = 1e-5f;

    std::vector<float> input_raw = generateRandomNormalVector(b_size * h * w * c, 2.0f, 1.0f);

    Instance_Norm_2d_Layer layer(h, w, c, epsilon, Execution_Target::VULKAN_GPU);

    Tensor in_train(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    layer.setTrainingMode(true);
    Tensor out_train = layer.forward(in_train);
    flushGpuEngine();
    auto train_data = out_train.getData();

    Tensor in_eval(b_size, h * w * c, input_raw, Execution_Target::VULKAN_GPU);
    layer.setTrainingMode(false);
    Tensor out_eval = layer.forward(in_eval);
    flushGpuEngine();
    auto eval_data = out_eval.getData();

    return computeMaxAbsDiff(train_data, eval_data) <= 1e-5;
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
}

namespace TestLeakyRelu
{
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

    auto input_raw = generateRandomNormalVector(count, 0.0f, 1.5f, 101, 1e-3f);
    auto grad_out_raw = generateRandomNormalVector(count, 0.0f, 1.0f, 202, 1e-3f);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor grad_out_cpu(1, count, grad_out_raw, Execution_Target::CPU);
    Tensor grad_in_cpu(Execution_Target::CPU);

    ops::activation::leakyReluBackward(in_cpu, grad_out_cpu, grad_in_cpu, slope);
    double max_rel_err = computeNumericalGradientMaxRelErr(input_raw, 1e-3f, grad_in_cpu.getData(), [&](const std::vector<float> &x) {
        Tensor cur_x(1, count, x, Execution_Target::CPU);
        Tensor cur_y(Execution_Target::CPU);
        ops::activation::leakyRelu(cur_x, cur_y, slope);
        const auto &y_data = cur_y.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < count; ++i) loss += y_data[i] * grad_out_raw[i];
        return loss;
    });
    return max_rel_err <= 1e-2;
}

bool testForwardGpuParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomNormalVector(count, 0.0f, 2.0f, 303, 1e-3f);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor out_cpu(Execution_Target::CPU);
    ops::activation::leakyRelu(in_cpu, out_cpu, slope);

    Tensor in_gpu(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor out_gpu(Execution_Target::VULKAN_GPU);
    ops::activation::leakyRelu(in_gpu, out_gpu, slope);
    flushGpuEngine();

    return computeMaxAbsDiff(out_cpu.getData(), out_gpu.getData()) <= 1e-5;
}

bool testBackwardGpuParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomNormalVector(count, 0.0f, 2.0f, 404, 1e-3f);
    auto grad_out_raw = generateRandomNormalVector(count, 0.0f, 1.0f, 505, 1e-3f);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor grad_out_cpu(1, count, grad_out_raw, Execution_Target::CPU);
    Tensor grad_in_cpu(Execution_Target::CPU);
    ops::activation::leakyReluBackward(in_cpu, grad_out_cpu, grad_in_cpu, slope);

    Tensor in_gpu(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_out_gpu(1, count, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_in_gpu(Execution_Target::VULKAN_GPU);
    ops::activation::leakyReluBackward(in_gpu, grad_out_gpu, grad_in_gpu, slope);
    flushGpuEngine();

    return computeMaxAbsDiff(grad_in_cpu.getData(), grad_in_gpu.getData()) <= 1e-5;
}

bool testForwardFp16AndCoopmatParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomNormalVector(count, 0.0f, 2.0f, 606, 1e-3f);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor out_cpu(Execution_Target::CPU);
    ops::activation::leakyRelu(in_cpu, out_cpu, slope);

    Tensor in_gpu_fp16(Shape{1, count}, input_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor out_gpu_fp16(Execution_Target::VULKAN_GPU);
    ops::activation::leakyRelu(in_gpu_fp16, out_gpu_fp16, slope);
    flushGpuEngine();

    return computeMaxAbsDiff(out_cpu.getData(), out_gpu_fp16.getData()) <= 2e-3;
}

bool testBackwardFp16AndCoopmatParity()
{
    size_t count = 1024;
    float slope = 0.01f;
    auto input_raw = generateRandomNormalVector(count, 0.0f, 2.0f, 707, 1e-3f);
    auto grad_out_raw = generateRandomNormalVector(count, 0.0f, 1.0f, 808, 1e-3f);

    Tensor in_cpu(1, count, input_raw, Execution_Target::CPU);
    Tensor grad_out_cpu(1, count, grad_out_raw, Execution_Target::CPU);
    Tensor grad_in_cpu(Execution_Target::CPU);
    ops::activation::leakyReluBackward(in_cpu, grad_out_cpu, grad_in_cpu, slope);

    Tensor in_gpu_fp16(Shape{1, count}, input_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor grad_out_gpu_fp16(Shape{1, count}, grad_out_raw, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    Tensor grad_in_gpu_fp16(Execution_Target::VULKAN_GPU);
    ops::activation::leakyReluBackward(in_gpu_fp16, grad_out_gpu_fp16, grad_in_gpu_fp16, slope);
    flushGpuEngine();

    return computeMaxAbsDiff(grad_in_cpu.getData(), grad_in_gpu_fp16.getData()) <= 2e-3;
}

bool testNumericalGradientGpu()
{
    size_t count = 32;
    float slope = 0.01f;

    auto input_raw = generateRandomNormalVector(count, 0.0f, 1.5f, 909, 1e-3f);
    auto grad_out_raw = generateRandomNormalVector(count, 0.0f, 1.0f, 1010, 1e-3f);

    Tensor in_gpu(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_out_gpu(1, count, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor grad_in_gpu(Execution_Target::VULKAN_GPU);

    ops::activation::leakyReluBackward(in_gpu, grad_out_gpu, grad_in_gpu, slope);
    flushGpuEngine();

    double max_rel_err = computeNumericalGradientMaxRelErr(input_raw, 1e-3f, grad_in_gpu.getData(), [&](const std::vector<float> &x) {
        Tensor cur_x(1, count, x, Execution_Target::VULKAN_GPU);
        Tensor cur_y(Execution_Target::VULKAN_GPU);
        ops::activation::leakyRelu(cur_x, cur_y, slope);
        flushGpuEngine();
        const auto &y_data = cur_y.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < count; ++i) loss += y_data[i] * grad_out_raw[i];
        return loss;
    });
    return max_rel_err <= 1e-2;
}

bool testLayerIntegrationAndSerialization()
{
    float slope = 0.05f;
    size_t count = 64;
    auto input_raw = generateRandomNormalVector(count, 0.0f, 1.0f, 1111, 1e-3f);
    auto grad_out_raw = generateRandomNormalVector(count, 0.0f, 1.0f, 2222, 1e-3f);

    Leaky_Relu_Layer layer(slope, Execution_Target::VULKAN_GPU);
    if (layer.getLayerType() != Layer_Type::LEAKY_RELU) return false;

    Tensor in_t(1, count, input_raw, Execution_Target::VULKAN_GPU);
    Tensor out_t = layer.forward(in_t);
    flushGpuEngine();

    Tensor dy_t(1, count, grad_out_raw, Execution_Target::VULKAN_GPU);
    Tensor dx_t = layer.backward(dy_t);
    flushGpuEngine();

    if (out_t.getRows() != 1 || out_t.getColumns() != count || dx_t.getColumns() != count) return false;

    auto cloned = layer.clone();
    auto cloned_leaky = dynamic_cast<Leaky_Relu_Layer *>(cloned.get());
    if (!cloned_leaky || std::fabs(cloned_leaky->getNegativeSlope() - slope) > 1e-6) return false;

    const std::string ckpt_name = makeUniqueTestTempPath("test_leaky_relu_ckpt").string();
    Scoped_Temp_File_Guard ckpt_guard({ckpt_name});
    {
        std::ofstream out(ckpt_name, std::ios::binary);
        layer.saveConfiguration(out);
    }

    Leaky_Relu_Layer loaded_layer(0.0f, Execution_Target::CPU);
    {
        std::ifstream in(ckpt_name, std::ios::binary);
        loaded_layer.loadInference(in);
    }

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
}

namespace TestUpsample
{
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
}

namespace TestNnUnet
{
bool testForwardShapeAndOutputRange()
{
    uint32_t h = 32, w = 32, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 101);
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

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 202);
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

    std::vector<float> sample0 = generateRandomNormalVector(sample_size, 0.5f, 1.0f, 303);
    std::vector<float> sample1 = generateRandomNormalVector(sample_size, -1.0f, 2.5f, 404);

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

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 505);
    Tensor in_tensor(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_tensor.uploadData(in_data);

    layer.forward(in_tensor);

    std::vector<float> grad_out_data = generateRandomNormalVector(b * out_c * h * w, 0.0f, 0.5f, 606);
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

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 707);
    Tensor cpu_in(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    cpu_in.uploadData(in_data);
    Tensor gpu_in(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    gpu_in.uploadData(in_data);

    Tensor cpu_out = cpu_layer.forward(cpu_in);
    Tensor gpu_out = gpu_layer.forward(gpu_in);
    flushGpuEngine();

    double fwd_diff = computeMaxAbsDiff(cpu_out.getData(), gpu_out.getData());
    if (fwd_diff > 1e-4) return false;

    std::vector<float> grad_out_data = generateRandomNormalVector(b * out_c * h * w, 0.0f, 0.5f, 808);
    Tensor cpu_grad_out(Shape{b, out_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    cpu_grad_out.uploadData(grad_out_data);
    Tensor gpu_grad_out(Shape{b, out_c * h * w}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    gpu_grad_out.uploadData(grad_out_data);

    Tensor cpu_in_grad = cpu_layer.backward(cpu_grad_out);
    Tensor gpu_in_grad = gpu_layer.backward(gpu_grad_out);
    flushGpuEngine();

    return computeMaxAbsDiff(cpu_in_grad.getData(), gpu_in_grad.getData()) <= 1e-3;
}

bool testCloneAndSerialization()
{
    uint32_t h = 16, w = 16, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer1(h, w, in_c, out_c, base_c, Execution_Target::CPU);

    const std::string ckpt_path = makeUniqueTestTempPath("test_nnunet_tmp", ".nnck").string();
    Scoped_Temp_File_Guard ckpt_guard({ckpt_path});
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

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 909);
    Tensor in_tensor(Shape{b, in_c * h * w}, Data_Type::FLOAT32, Execution_Target::CPU);
    in_tensor.uploadData(in_data);

    Tensor out1 = layer1.forward(in_tensor);
    Tensor out2 = layer2.forward(in_tensor);

    double diff = computeMaxAbsDiff(out1.getData(), out2.getData());
    if (diff > 1e-6) return false;

    auto cloned = layer1.clone();
    Tensor out_cloned = cloned->forward(in_tensor);
    return computeMaxAbsDiff(out1.getData(), out_cloned.getData()) <= 1e-6;
}

bool testRank4InputForwardBackwardGpu()
{
    uint32_t h = 32, w = 32, in_c = 1, out_c = 1, base_c = 4, b = 2;
    NN_UNet_Layer layer(h, w, in_c, out_c, base_c, Execution_Target::VULKAN_GPU);

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 999);
    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);

    Tensor out_tensor = layer.forward(in_tensor);
    if (out_tensor.getShape() != Shape{b, out_c, h, w}) return false;

    std::vector<float> grad_data = generateRandomNormalVector(b * out_c * h * w, 0.1f, 0.5f, 888);
    Tensor grad_out(Shape{b, out_c, h, w}, grad_data, Execution_Target::VULKAN_GPU);

    Tensor in_grad = layer.backward(grad_out);
    flushGpuEngine();

    if (in_grad.getShape() != Shape{b, in_c, h, w}) return false;

    for (float v : in_grad.getData())
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }
    return true;
}

bool testGradientClipping()
{
    Neural_Network net_cpu(Execution_Target::CPU);
    net_cpu.addLayer<NN_UNet_Layer>(16, 16, 1, 1, 4, Execution_Target::CPU);
    net_cpu.setTrainingMode(true);

    Tensor in_cpu(Shape{1, 1, 16, 16}, generateRandomNormalVector(256, 0.0f, 1.0f, 42), Execution_Target::CPU);
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

    Neural_Network net_gpu(Execution_Target::VULKAN_GPU);
    net_gpu.addLayer<NN_UNet_Layer>(16, 16, 1, 1, 4, Execution_Target::VULKAN_GPU);
    net_gpu.setTrainingMode(true);

    Tensor in_gpu(Shape{1, 1, 16, 16}, generateRandomNormalVector(256, 0.0f, 1.0f, 42), Execution_Target::VULKAN_GPU);
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
    net.enableMixedPrecision(true);
    net.setTrainingMode(true);

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.5f, 0.25f, 12345);
    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);

    Tensor out = net.forward(in_tensor);
    flushGpuEngine();

    for (float v : out.getData())
    {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    std::vector<float> grad_data = generateRandomNormalVector(b * out_c * h * w, 0.1f, 0.5f, 67890);
    Tensor grad_tensor(Shape{b, out_c, h, w}, grad_data, Execution_Target::VULKAN_GPU);
    net.backward(grad_tensor, true);
    flushGpuEngine();

    net.enableGradientAccumulation(true);
    net.forward(in_tensor);
    net.backward(grad_tensor, true);
    flushGpuEngine();
    net.enableGradientAccumulation(false);

    auto pairs = net.getParametersAndGradients();
    for (const auto &[param, grad] : pairs)
    {
        if (grad && !grad->isEmpty())
        {
            for (float v : grad->getData())
            {
                if (std::isnan(v) || std::isinf(v) || std::abs(v) > 1e8f) return false;
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
    net.enableMixedPrecision(true);
    net.setTrainingMode(true);

    auto &scheduler = net.setLearningRate<Cosine_Annealing>(0.01f, 0.001f, 10);
    net.setOptimizer<Adam_Optimizer>(scheduler, 0.9f, 0.999f, 1e-8f, 1.0f, 0.0f);

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.5f, 0.25f, 12345);
    std::vector<float> target_data = generateRandomNormalVector(b * out_c * h * w, 0.5f, 0.25f, 54321);
    for (float &v : target_data) v = (v > 0.5f) ? 1.0f : 0.0f;

    Tensor in_tensor(Shape{b, in_c, h, w}, in_data, Execution_Target::VULKAN_GPU);
    Tensor target_tensor(Shape{b, out_c, h, w}, target_data, Execution_Target::VULKAN_GPU);

    Tensor pred0 = net.forward(in_tensor);
    flushGpuEngine();

    auto pairs = net.getParametersAndGradients();
    auto p0_data = pairs[0].first->getData();

    Dice_BCE_Cost cost_fn(1.0f, 1.0f, 1e-7f, Execution_Target::VULKAN_GPU);
    Tensor grad;
    cost_fn.computeLossAndGradient(pred0, target_tensor, grad);
    net.backward(grad, true);
    net.clipGradients(1.0f);

    net.getTrainingContext().getOptimizer().step(net.getParametersAndGradients());
    net.invalidateLayerWeightCaches();
    flushGpuEngine();
    net.resetGradients();
    flushGpuEngine();

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

    std::vector<float> in_data = generateRandomNormalVector(b * in_c * h * w, 0.0f, 1.0f, 777);
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

    Tensor grad_main(Shape{b, out_c, h, w}, generateRandomNormalVector(b * out_c * h * w, 0.1f, 0.5f, 111), Execution_Target::CPU);
    Tensor grad_ds2(Shape{b, out_c, h / 2, w / 2}, generateRandomNormalVector(b * out_c * (h / 2) * (w / 2), 0.1f, 0.5f, 222), Execution_Target::CPU);
    Tensor grad_ds3(Shape{b, out_c, h / 4, w / 4}, generateRandomNormalVector(b * out_c * (h / 4) * (w / 4), 0.1f, 0.5f, 333), Execution_Target::CPU);

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
    flushGpuEngine();

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
    flushGpuEngine();

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
}

void runVisionSegSuite(Test_Result_Counter &counter);

namespace TestCheXpertDataset
{
bool writeSyntheticChx1(const std::filesystem::path &path)
{
    using namespace chexlocalize;
    constexpr uint32_t total = 3;
    constexpr uint32_t h = 4;
    constexpr uint32_t w = 4;
    constexpr uint32_t c = 1;
    constexpr uint32_t classes = 14;
    BinaryHeader hdr{};
    std::memcpy(hdr.magic, "CHX1", 4);
    hdr.total_samples = total;
    hdr.image_height = h;
    hdr.image_width = w;
    hdr.channels = c;
    hdr.num_classes = classes;
    hdr.record_size = 8u + classes * static_cast<uint32_t>(sizeof(float)) + h * w;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    for (uint32_t i = 0; i < total; ++i)
    {
        uint32_t orig[2] = {100u + i, 200u + i};
        out.write(reinterpret_cast<const char *>(orig), sizeof(orig));
        std::array<float, 14> labels{};
        labels.fill(0.0f);
        labels[2] = 1.0f;
        labels[5] = -1.0f;
        out.write(reinterpret_cast<const char *>(labels.data()), sizeof(labels));
        std::array<uint8_t, h * w> pixels{};
        for (size_t p = 0; p < pixels.size(); ++p)
        {
            pixels[p] = static_cast<uint8_t>((i * 40u + p * 5u) & 0xFFu);
        }
        out.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
    }
    return out.good();
}

bool runSuite()
{
    using namespace chexlocalize;
    const auto bin_path = makeUniqueTestTempPath("test_chexpert_chx1", ".bin");
    const auto bad_path = makeUniqueTestTempPath("test_chexpert_bad", ".bin");
    Scoped_Temp_File_Guard guard({bin_path.string(), bad_path.string()});
    guard.clean();

    if (!writeSyntheticChx1(bin_path)) return false;

    {
        std::ofstream bad(bad_path, std::ios::binary | std::ios::trunc);
        const char bad_hdr[64] = {'B', 'A', 'D', '0'};
        bad.write(bad_hdr, sizeof(bad_hdr));
    }
    if (!expectThrows<std::runtime_error>([&]() { CheXpert_Dataset bad_ds(bad_path.string()); }))
    {
        return false;
    }

    CheXpert_Dataset ds(bin_path.string(), 0.5f, 0.25f);
    if (!ds.isOpen() || ds.size() != 3 || ds.getHeight() != 4 || ds.getWidth() != 4 ||
        ds.getChannels() != 1 || ds.getNumClasses() != 14)
    {
        ds.close();
        return false;
    }

    auto meta0 = ds.getOriginalResolution(0);
    auto raw1 = ds.getRawImage(1);
    if (meta0.orig_height != 100 || meta0.orig_width != 200 || raw1.size() != 16 || raw1[0] != 40)
    {
        ds.close();
        return false;
    }

    auto [img_u0, lbl_u0] = ds.getSample(0, Uncertainty_Policy::U_ZEROS, true, Execution_Target::CPU);
    auto [img_u1, lbl_u1] = ds.getSample(0, Uncertainty_Policy::U_ONES, true, Execution_Target::CPU);
    auto [img_soft, lbl_soft] = ds.getSample(0, Uncertainty_Policy::U_SOFT_MULTI_LABEL, true, Execution_Target::CPU);
    if (img_u0.getShape() != Shape{1, 1, 4, 4} || lbl_u0.getShape() != Shape{1, 10})
    {
        ds.close();
        return false;
    }
    const auto v_u0 = lbl_u0.getData();
    const auto v_u1 = lbl_u1.getData();
    const auto v_soft = lbl_soft.getData();
    if (!nearlyEqual(v_u0[1], 1.0f, 1e-6f) || !nearlyEqual(v_u0[4], 0.0f, 1e-6f) ||
        !nearlyEqual(v_u1[4], 1.0f, 1e-6f) || !nearlyEqual(v_soft[4], SOFT_LABEL_Q_10[4], 1e-6f))
    {
        ds.close();
        return false;
    }

    const auto px_norm = img_u0.getData();
    const float expected_p1 = (5.0f / 255.0f - 0.5f) / 0.25f;
    if (!nearlyEqual(px_norm[1], expected_p1, 1e-4f))
    {
        ds.close();
        return false;
    }

    auto batch = ds.getBatchWithMetadata(0, 2, Uncertainty_Policy::U_STANFORD, false, Execution_Target::CPU, false);
    if (batch.images.getShape() != Shape{2, 1, 4, 4} ||
        batch.labels.getShape() != Shape{2, 14} ||
        batch.metadata.size() != 2 ||
        batch.metadata[1].orig_height != 101)
    {
        ds.close();
        return false;
    }

    const std::array<size_t, 2> indices = {2, 0};
    auto [idx_img, idx_lbl] = ds.getBatchByIndices(indices, Uncertainty_Policy::U_ZEROS, true, false, Execution_Target::CPU);
    if (idx_img.getShape() != Shape{2, 1, 4, 4} || idx_lbl.getShape() != Shape{2, 10})
    {
        ds.close();
        return false;
    }

    if (!expectThrows<std::out_of_range>([&]() { (void)ds.getSample(3); }))
    {
        ds.close();
        return false;
    }

    ds.close();
    return true;
}
}

void runVisionSegSuite(Test_Result_Counter &counter)
{
    counter.record("  Dice BCE Cost Function Suite:      ", TestDiceBceCost::runSuite());
    counter.record("  InstanceNorm2D Operator Suite:     ", TestInstanceNorm::runSuite());
    counter.record("  LeakyReLU Operator & Layer Suite:  ", TestLeakyRelu::runSuite());
    counter.record("  UpSample2D Operator & Layer Suite: ", TestUpsample::runSuite());
    counter.record("  NN-UNet Comprehensive Suite (x13): ", TestNnUnet::runSuite());
    counter.record("  CheXpert Dataset (CHX1 I/O):       ", TestCheXpertDataset::runSuite());
}
