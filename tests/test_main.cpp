#include "test_common.h"

#include <string_view>

namespace
{

void printUsage()
{
    std::cout << "Available test suites:\n"
              << "  cpu_core\n"
              << "  gpu_core\n"
              << "  engine_systems\n"
              << "  ternary_quant\n"
              << "  vision_seg\n"
              << "  heavy_unet\n"
              << "  regressions_w2_w5\n"
              << "  regressions_w6_w7\n"
              << "  perf_phases\n"
              << "  dx_infra\n"
              << "  layer_upgrades\n"
              << "  sealed_read\n"
              << "  weight_loader\n";
}

bool runNamedSuite(std::string_view suite_name, Test_Result_Counter &counter)
{
    if (suite_name == "cpu_core")
    {
        runCoreTestSuite(Execution_Target::CPU, "CPU BACKEND", counter);
        return true;
    }
    if (suite_name == "gpu_core")
    {
        runCoreTestSuite(Execution_Target::VULKAN_GPU, "VULKAN GPU BACKEND", counter);
        return true;
    }
    if (suite_name == "engine_systems")
    {
        runEngineSystemsSuite(counter);
        return true;
    }
    if (suite_name == "ternary_quant")
    {
        runTernaryQuantSuite(counter);
        return true;
    }
    if (suite_name == "vision_seg")
    {
        runVisionSegSuite(counter);
        return true;
    }
    if (suite_name == "heavy_unet")
    {
        runHeavyUnetSuite(counter);
        return true;
    }
    if (suite_name == "regressions_w2_w5")
    {
        runRegressionsW2W5Suite(counter);
        return true;
    }
    if (suite_name == "regressions_w6_w7")
    {
        runRegressionsW6W7Suite(counter);
        return true;
    }
    if (suite_name == "perf_phases")
    {
        runPerfPhasesSuite(counter);
        return true;
    }
    if (suite_name == "dx_infra")
    {
        runDxInfraSuite(counter);
        return true;
    }
    if (suite_name == "layer_upgrades")
    {
        runLayerUpgradesSuite(counter);
        return true;
    }
    if (suite_name == "sealed_read")
    {
        runSealedReadSuite(counter);
        return true;
    }
    if (suite_name == "weight_loader")
    {
        runWeightLoaderSuite(counter);
        return true;
    }
    return false;
}

}

int main(int argc, char *argv[])
{
    Logger::enableFileLogging(false);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);

    std::string selected_suite;
    bool skip_heavy = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        if (arg == "--list")
        {
            printUsage();
            return 0;
        }
        if (arg == "--skip-heavy")
        {
            skip_heavy = true;
            continue;
        }
        if (arg == "--suite" && i + 1 < argc)
        {
            selected_suite = argv[++i];
            continue;
        }
        if (arg.rfind("--suite=", 0) == 0)
        {
            selected_suite = std::string(arg.substr(8));
            continue;
        }
        std::cerr << "Unknown test runner option: " << arg << "\n";
        printUsage();
        return 2;
    }

    Test_Result_Counter counter{};

    if (!selected_suite.empty())
    {
        if (!runNamedSuite(selected_suite, counter))
        {
            std::cerr << "Unknown suite name: " << selected_suite << "\n";
            printUsage();
            return 2;
        }
    }
    else
    {
        runCoreTestSuite(Execution_Target::CPU, "CPU BACKEND", counter);
        runCoreTestSuite(Execution_Target::VULKAN_GPU, "VULKAN GPU BACKEND", counter);
        runEngineSystemsSuite(counter);
        runTernaryQuantSuite(counter);
        runVisionSegSuite(counter);
        if (!skip_heavy)
        {
            runHeavyUnetSuite(counter);
        }
        runRegressionsW2W5Suite(counter);
        runRegressionsW6W7Suite(counter);
        runPerfPhasesSuite(counter);
        runDxInfraSuite(counter);
        runLayerUpgradesSuite(counter);
        runSealedReadSuite(counter);
        runWeightLoaderSuite(counter);
    }

    std::cout << "========================================\n";
    std::cout << "  SUMMARY: PASS=" << counter.passed << " FAIL=" << counter.failed << "\n";
    std::cout << "========================================\n";
    return (counter.failed == 0) ? 0 : 1;
}
