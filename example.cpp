#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <format>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include "optimizer/adam_optimizer.h"
#include "learning_rate/cosine_annealing.h"
#include "neural_network.h"
#include "helper/layer.h"
#include "cost_function/cce_cost.h"
#include "cost_function/icost_function.h"
#include "math/matrix.h"
#include "helper/logger.h"
#include "engine/async_data_pipeline.h"


constexpr std::size_t INPUT_DIM = 784;
constexpr std::size_t OUTPUT_DIM = 10;
constexpr std::size_t BATCH_SIZE = 512;
constexpr std::size_t EPOCHS = 1;

uint32_t swapEndian(uint32_t val)
{
    return (val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24);
}

void augmentMnistImage(const float *src_ptr, float *dst_ptr, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> crop_dist(0, 4);

    int crop_x = crop_dist(rng);
    int crop_y = crop_dist(rng);

    for (int y = 0; y < 28; ++y)
    {
        int orig_y = y + crop_y - 2;
        for (int x = 0; x < 28; ++x)
        {
            int orig_x = x + crop_x - 2;
            std::size_t dst_idx = y * 28 + x;

            if (orig_y >= 0 && orig_y < 28 && orig_x >= 0 && orig_x < 28)
            {
                std::size_t src_idx = orig_y * 28 + orig_x;
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
            else
            {
                dst_ptr[dst_idx] = 0.0f;
            }
        }
    }
}

void loadMnistImages(const std::string &file_path, std::vector<float> &images_data, uint32_t &num_images)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        Logger::logMessage("loadMnistImages: Cannot open MNIST images file: " + file_path, LOG_ERROR);
        throw std::runtime_error("Cannot open MNIST images file");
    }

    uint32_t magic_number = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));

    magic_number = swapEndian(magic_number);
    if (magic_number != 2051)
    {
        Logger::logMessage("loadMnistImages: Invalid MNIST image magic number", LOG_ERROR);
        throw std::runtime_error("Invalid MNIST image magic number");
    }

    num_images = swapEndian(num_images);
    num_rows = swapEndian(num_rows);
    num_cols = swapEndian(num_cols);

    std::size_t total_pixels = static_cast<std::size_t>(num_images) * num_rows * num_cols;
    std::vector<uint8_t> raw_pixels(total_pixels);
    file.read(reinterpret_cast<char *>(raw_pixels.data()), total_pixels);

    images_data.resize(total_pixels);
    for (std::size_t i = 0; i < total_pixels; ++i)
    {
        images_data[i] = static_cast<float>(raw_pixels[i]) / 255.0f;
    }
}

void loadMnistLabels(const std::string &file_path, std::vector<float> &labels_data, uint32_t num_images)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        Logger::logMessage("loadMnistLabels: Cannot open MNIST labels file: " + file_path, LOG_ERROR);
        throw std::runtime_error("Cannot open MNIST labels file");
    }

    uint32_t magic_number = 0;
    uint32_t num_items = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));

    magic_number = swapEndian(magic_number);
    if (magic_number != 2049)
    {
        Logger::logMessage("loadMnistLabels: Invalid MNIST label magic number", LOG_ERROR);
        throw std::runtime_error("Invalid MNIST label magic number");
    }

    num_items = swapEndian(num_items);
    if (num_items < num_images)
    {
        Logger::logMessage("loadMnistLabels: Label count mismatch", LOG_ERROR);
        throw std::runtime_error("Label count mismatch");
    }

    std::vector<uint8_t> raw_labels(num_items);
    file.read(reinterpret_cast<char *>(raw_labels.data()), num_items);

    labels_data.assign(static_cast<std::size_t>(num_images) * OUTPUT_DIM, 0.0f);
    for (std::size_t i = 0; i < num_images; ++i)
    {
        uint8_t label = raw_labels[i];
        if (label < OUTPUT_DIM)
        {
            labels_data[i * OUTPUT_DIM + label] = 1.0f;
        }
    }
}

class Mnist_Data_Pipeline : public Async_Data_Pipeline
{
private:
    const std::vector<float> &images_data;
    const std::vector<float> &labels_data;
    uint32_t num_images;
    std::size_t batch_size;

    void prepareBatchHost(std::size_t batch_step, std::vector<float> &out_inputs, std::vector<float> &out_targets) override
    {
        std::size_t num_batches = num_images / batch_size;
        std::size_t b = batch_step % num_batches;

        std::size_t offset_x = b * batch_size * INPUT_DIM;
        std::size_t offset_y = b * batch_size * OUTPUT_DIM;

        out_inputs.resize(batch_size * INPUT_DIM);
        out_targets.resize(batch_size * OUTPUT_DIM);

        std::copy(images_data.begin() + offset_x,
                  images_data.begin() + offset_x + (batch_size * INPUT_DIM),
                  out_inputs.begin());

        std::copy(labels_data.begin() + offset_y,
                  labels_data.begin() + offset_y + (batch_size * OUTPUT_DIM),
                  out_targets.begin());

        float sum_x = std::accumulate(out_inputs.begin(), out_inputs.end(), 0.0f);
        float mean_x = out_inputs.empty() ? 0.0f : sum_x / static_cast<float>(out_inputs.size());
    }

public:
    Mnist_Data_Pipeline(const std::vector<float> &imgs,
                        const std::vector<float> &lbls,
                        uint32_t n_imgs,
                        std::size_t b_size,
                        VkDevice vk_device = VK_NULL_HANDLE)
        : Async_Data_Pipeline(vk_device),
          images_data(imgs),
          labels_data(lbls),
          num_images(n_imgs),
          batch_size(b_size) {}

    std::size_t getBatchSize() const override { return batch_size; }
};

double runBenchmark(Execution_Target target,
                    const std::vector<float> &images_data,
                    const std::vector<float> &labels_data,
                    uint32_t num_images,
                    Neural_Network &nn)
{
    std::size_t steps_per_epoch = num_images / BATCH_SIZE;
    nn.getLearningRate().setMaxEpoch(static_cast<int>(EPOCHS));

    Mnist_Data_Pipeline pipeline(
        images_data,
        labels_data,
        num_images,
        BATCH_SIZE,
        Execution_Engine::getInstance().getContext().getDevice());

    auto start_time = std::chrono::high_resolution_clock::now();

    nn.fit(pipeline, EPOCHS, steps_per_epoch);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    return duration.count();
}

void evaluateModel(Neural_Network &nn,
                   const std::vector<float> &test_images_data,
                   const std::vector<float> &test_labels_data,
                   uint32_t num_test_images,
                   Execution_Target target)
{
    Logger::logMessage("Evaluating model on test dataset...", LOG_INFO, true);
    nn.setTrainingMode(false);

    std::size_t correct_count = 0;
    std::array<std::array<std::size_t, OUTPUT_DIM>, OUTPUT_DIM> confusion_matrix{};

    std::size_t test_batch_size = BATCH_SIZE;
    std::size_t num_batches = (num_test_images + test_batch_size - 1) / test_batch_size;

    for (std::size_t b = 0; b < num_batches; ++b)
    {
        std::size_t current_batch_size = std::min(test_batch_size, static_cast<std::size_t>(num_test_images) - b * test_batch_size);

        std::vector<float> batch_input_host(current_batch_size * INPUT_DIM);
        std::vector<float> batch_target_host(current_batch_size * OUTPUT_DIM);

        std::copy(test_images_data.begin() + b * test_batch_size * INPUT_DIM,
                  test_images_data.begin() + (b * test_batch_size + current_batch_size) * INPUT_DIM,
                  batch_input_host.begin());

        std::copy(test_labels_data.begin() + b * test_batch_size * OUTPUT_DIM,
                  test_labels_data.begin() + (b * test_batch_size + current_batch_size) * OUTPUT_DIM,
                  batch_target_host.begin());

        Matrix input_mat(current_batch_size, INPUT_DIM, batch_input_host, target);
        Matrix target_mat(current_batch_size, OUTPUT_DIM, batch_target_host, target);

        Matrix pred_mat = nn.forward(input_mat);

        if (target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }

        std::vector<float> pred_data = pred_mat.getData();

        for (std::size_t i = 0; i < current_batch_size; ++i)
        {
            std::size_t pred_label = 0;
            float max_pred_val = pred_data[i * OUTPUT_DIM];
            for (std::size_t c = 1; c < OUTPUT_DIM; ++c)
            {
                float val = pred_data[i * OUTPUT_DIM + c];
                if (val > max_pred_val)
                {
                    max_pred_val = val;
                    pred_label = c;
                }
            }

            std::size_t true_label = 0;
            float max_true_val = batch_target_host[i * OUTPUT_DIM];
            for (std::size_t c = 1; c < OUTPUT_DIM; ++c)
            {
                float val = batch_target_host[i * OUTPUT_DIM + c];
                if (val > max_true_val)
                {
                    max_true_val = val;
                    true_label = c;
                }
            }

            if (pred_label == true_label)
            {
                correct_count++;
            }

            confusion_matrix[true_label][pred_label]++;
        }
    }

    std::size_t wrong_count = num_test_images - correct_count;
    double accuracy = (static_cast<double>(correct_count) / num_test_images) * 100.0;
    double error_rate = (static_cast<double>(wrong_count) / num_test_images) * 100.0;
    
    std::string eval_report = "\n========== Evaluation ==========\n";
    eval_report += std::format("Samples   : {}\n", num_test_images);
    eval_report += std::format("Correct   : {}\n", correct_count);
    eval_report += std::format("Wrong     : {}\n", wrong_count);
    eval_report += std::format("Accuracy  : {:.2f}%\n", accuracy);
    eval_report += std::format("Error     : {:.2f}%\n\n", error_rate);

    eval_report += "Confusion Matrix:\n[";
    for (std::size_t r = 0; r < OUTPUT_DIM; ++r)
    {
        if (r > 0)
        {
            eval_report += " ";
        }
        eval_report += "[";
        for (std::size_t c = 0; c < OUTPUT_DIM; ++c)
        {
            eval_report += std::format("{:4d}{}", confusion_matrix[r][c], (c == OUTPUT_DIM - 1 ? "" : " "));
        }
        eval_report += std::format("]{}", (r == OUTPUT_DIM - 1 ? "]" : "\n"));
    }

    Logger::logMessage(eval_report, LOG_INFO, true);

    nn.setTrainingMode(true);
}

int main()
{
    std::vector<float> train_images_data;
    std::vector<float> train_labels_data;
    uint32_t num_train_images = 0;

    loadMnistImages("data/train-images.idx3-ubyte", train_images_data, num_train_images);
    loadMnistLabels("data/train-labels.idx1-ubyte", train_labels_data, num_train_images);

    std::vector<float> test_images_data;
    std::vector<float> test_labels_data;
    uint32_t num_test_images = 0;

    loadMnistImages("data/t10k-images.idx3-ubyte", test_images_data, num_test_images);
    loadMnistLabels("data/t10k-labels.idx1-ubyte", test_labels_data, num_test_images);

    Execution_Engine::getInstance().getPipelineCacheManager().initPipelineCache("temp/pipeline_cache.bin");

    Execution_Target target = Execution_Target::VULKAN_GPU;
    Neural_Network nn(target);
    nn.setTrainingMode(true);

    nn.setLearningRate(std::make_unique<Cosine_Annealing>(0.001f, 1e-5f, static_cast<int>(EPOCHS)));
    nn.setOptimizer(std::make_unique<Adam_Optimizer>(nn.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f));
    nn.setCostFunction(std::make_unique<CCE_Cost>());

    // nn.addLayer(std::make_unique<Linear_Layer>(28 * 28, 512));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Linear_Layer>(512, 256));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Linear_Layer>(256, 128));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Linear_Layer>(128, 64));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Linear_Layer>(64, OUTPUT_DIM));
    // nn.addLayer(std::make_unique<Softmax>(true));
    
    //================================================================================
    
    nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 16, 16, 3, 1, 1));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<MaxPool2d_Layer>(28, 28, 16, 2, 2, 0));

    nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 32, 32, 3, 1, 1));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<MaxPool2d_Layer>(14, 14, 32, 2, 2, 0));

    nn.addLayer(std::make_unique<Linear_Layer>(7 * 7 * 32, 128));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Linear_Layer>(128, OUTPUT_DIM));
    nn.addLayer(std::make_unique<Softmax>(true));

    //================================================================================

    // nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(28, 28, 16));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 16, 16, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(28, 28, 16));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<MaxPool2d_Layer>(28, 28, 16, 2, 2, 0));

    // nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(14, 14, 32));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 32, 32, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(14, 14, 32));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<MaxPool2d_Layer>(14, 14, 32, 2, 2, 0));

    // nn.addLayer(std::make_unique<Linear_Layer>(7 * 7 * 32, 128));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Linear_Layer>(128, OUTPUT_DIM));
    // nn.addLayer(std::make_unique<Softmax>(true));

    //================================================================================

    // nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(28, 28, 16));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 16, 16, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(28, 28, 16));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<MaxPool2d_Layer>(28, 28, 16, 2, 2, 0));

    // nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(14, 14, 32));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 32, 32, 3, 1, 1));
    // nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(14, 14, 32));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<MaxPool2d_Layer>(14, 14, 32, 2, 2, 0));

    // nn.addLayer(std::make_unique<GlobalAvgPool2d_Layer>(7, 7, 32));

    // nn.addLayer(std::make_unique<Linear_Layer>(32, 128));
    // nn.addLayer(std::make_unique<GeLU>());

    // nn.addLayer(std::make_unique<Linear_Layer>(128, OUTPUT_DIM));
    // nn.addLayer(std::make_unique<Softmax>(true));

    nn.compileAndWarmup(BATCH_SIZE, INPUT_DIM, OUTPUT_DIM);
    Logger::logMessage("Starting training benchmark with Data Augmentation...", LOG_INFO, true);
    double duration = runBenchmark(target, train_images_data, train_labels_data, num_train_images, nn);
    Logger::logMessage("Training completed in " + std::to_string(duration / 1000.0) + " s", LOG_INFO, true);

    nn.setTrainingMode(false);
    evaluateModel(nn, test_images_data, test_labels_data, num_test_images, target);

    std::filesystem::create_directories("output/mnist");

    nn.saveInference("output/mnist/model.bin");
    Logger::logMessage("Inference model exported to model.bin", LOG_INFO, true);

    return 0;
}