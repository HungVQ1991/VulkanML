#include "test_common.h"
#include <atomic>
#include <chrono>
#include <format>

void Test_Result_Counter::record(const char *label, bool ok)
{
    std::cout << label << (ok ? "PASS" : "FAIL") << "\n";
    if (ok)
    {
        ++passed;
    }
    else
    {
        ++failed;
    }
}

bool nearlyEqual(float a, float b, float eps)
{
    return std::fabs(a - b) < eps;
}

bool verifyMatrix(const Tensor &mat, const std::vector<float> &expected_data, float eps)
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
    return verifyVectorsClose(actual_data, expected_data, eps);
}

double computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        return 1e9;
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
        {
            return 1e9;
        }
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

std::vector<float> generateRandomVector(size_t count, float low, float high, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = dist(rng);
    }
    return data;
}

std::vector<float> generateRandomNormalVector(size_t count, float mean, float stddev, unsigned int seed, float min_abs)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        float val = dist(rng);
        if (min_abs > 0.0f && std::fabs(val) < min_abs)
        {
            val = (val >= 0.0f) ? min_abs : -min_abs;
        }
        data[i] = val;
    }
    return data;
}

std::vector<float> generateRandomBinaryVector(size_t count, float pos_ratio, unsigned int seed)
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

bool failTestStep(std::string_view suite_tag, std::string_view message)
{
    std::cerr << "\n    [" << suite_tag << " FAIL] " << message << "\n";
    return false;
}

void flushGpuEngine(bool wait_idle)
{
    auto &engine = Execution_Engine::getInstance();
    engine.executeGraph();
    if (wait_idle)
    {
        engine.waitIdle();
    }
}

void flushIfGpu(Execution_Target target, bool wait_idle)
{
    if (target == Execution_Target::VULKAN_GPU)
    {
        flushGpuEngine(wait_idle);
    }
}

Tensor castTensorToFp16(const Tensor &src)
{
    Tensor dst(src.getExecutionTarget());
    src.to(Data_Type::FLOAT16, dst);
    return dst;
}

void convertTensorToFp16InPlace(Tensor &tensor)
{
    tensor = castTensorToFp16(tensor);
}

bool verifyVectorsClose(const std::vector<float> &a, const std::vector<float> &b, float eps)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!nearlyEqual(a[i], b[i], eps))
        {
            return false;
        }
    }
    return true;
}

bool verifyTensorsClose(const Tensor &a, const Tensor &b, float eps)
{
    if (a.getShape() != b.getShape())
    {
        return false;
    }
    return verifyVectorsClose(a.getData(), b.getData(), eps);
}

double computeNumericalGradientMaxRelErr(
    const std::vector<float> &base_data,
    float delta,
    const std::vector<float> &analytical_grad,
    const std::function<float(const std::vector<float> &)> &eval_loss)
{
    if (base_data.size() != analytical_grad.size())
    {
        return 1e9;
    }
    double max_rel_err = 0.0;
    std::vector<float> perturbed = base_data;
    for (size_t i = 0; i < base_data.size(); ++i)
    {
        const float orig = perturbed[i];
        perturbed[i] = orig + delta;
        const float loss_plus = eval_loss(perturbed);
        perturbed[i] = orig - delta;
        const float loss_minus = eval_loss(perturbed);
        perturbed[i] = orig;

        const float num_grad = (loss_plus - loss_minus) / (2.0f * delta);
        const float ana_grad = analytical_grad[i];
        if (!std::isfinite(num_grad) || !std::isfinite(ana_grad))
        {
            return 1e9;
        }
        const double abs_err = std::fabs(num_grad - ana_grad);
        const double denom = 1.0 + std::max(std::fabs(num_grad), std::fabs(ana_grad));
        max_rel_err = std::max(max_rel_err, abs_err / denom);
    }
    return max_rel_err;
}

void Scoped_Temp_File_Guard::clean() const
{
    std::error_code ec;
    for (const auto &f : files)
    {
        std::filesystem::remove(f, ec);
    }
    for (const auto &d : dirs)
    {
        if (!d.empty())
        {
            std::filesystem::remove_all(d, ec);
        }
    }
}

Scoped_Logger_Guard::~Scoped_Logger_Guard()
{
    Logger::configure(saved);
    if (reset_counters_on_exit)
    {
        Logger::resetLogCounters();
    }
}

Scoped_Engine_Flags_Guard::~Scoped_Engine_Flags_Guard()
{
    if (Runtime::isFloat16Enabled() != fp16)
    {
        Runtime::setFloat16Enabled(fp16);
    }
    Runtime::setCooperativeMatrixEnabled(coop);
    Runtime::setStaticGraphEnabled(stat);
    Runtime::setSingleSubmitEnabled(single);
    auto &engine = Execution_Engine::getInstance();
    engine.enableFusedGemmAdam(fused);
    engine.setGpuProfilingEnabled(profiling);
    engine.enableGraphCaching(graph_cache);
}

std::filesystem::path makeUniqueTestTempPath(std::string_view prefix, std::string_view ext)
{
    static std::atomic<uint64_t> counter{0};
    const uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string filename;
    filename.reserve(prefix.size() + ext.size() + 48);
    filename.append(prefix);
    filename.push_back('_');
    filename.append(std::to_string(ts));
    filename.push_back('_');
    filename.append(std::to_string(id));
    filename.append(ext);
    return std::filesystem::temp_directory_path() / filename;
}

bool hasVulkanDevice()
{
    return Runtime::isGpuAvailable();
}

bool checkVectorsClose(const std::vector<float> &a,
                       const std::vector<float> &b,
                       float tol,
                       const char *tag,
                       const char *step)
{
    if (a.size() != b.size())
    {
        return failTestStep(tag, std::string(step) + " (size mismatch)");
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
        {
            return failTestStep(tag, std::string(step) + " (non-finite value)");
        }
        max_diff = std::max(max_diff, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    if (max_diff > static_cast<double>(tol))
    {
        return failTestStep(tag, std::format("{} (max_diff={:.6f} > tol={:.6f})", step, max_diff, tol));
    }
    return true;
}
