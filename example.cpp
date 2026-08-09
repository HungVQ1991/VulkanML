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

    std::size_t total_pixels = num_images * num_rows * num_cols;
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

    std::vector<uint8_t> raw_labels(num_items);
    file.read(reinterpret_cast<char *>(raw_labels.data()), num_items);

    labels_data.assign(num_images * OUTPUT_DIM, 0.0f);
    for (std::size_t i = 0; i < num_images; ++i)
    {
        uint8_t label = raw_labels[i];
        labels_data[i * OUTPUT_DIM + label] = 1.0f;
    }
}

class Mnist_Data_Pipeline : public Async_Data_Pipeline
{
private:
    const std::vector<float> &images_data;
    const std::vector<float> &labels_data;
    uint32_t num_images;
    std::size_t batch_size;
    Execution_Target target;

    Batch_Data prepareBatch(std::size_t batch_step) override
    {
        std::size_t num_batches = num_images / batch_size;
        std::size_t b = batch_step % num_batches;

        std::size_t offset_x = b * batch_size * INPUT_DIM;
        std::size_t offset_y = b * batch_size * OUTPUT_DIM;

        std::vector<float> batch_x(batch_size * INPUT_DIM);
        std::vector<float> batch_y(batch_size * OUTPUT_DIM);

        thread_local std::random_device rd;
        thread_local std::mt19937 rng(rd());

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            augmentMnistImage(images_data.data() + offset_x + i * INPUT_DIM,
                              batch_x.data() + i * INPUT_DIM,
                              rng);
        }

        std::copy(labels_data.begin() + offset_y,
                  labels_data.begin() + offset_y + (batch_size * OUTPUT_DIM),
                  batch_y.begin());

        return {
            Matrix(batch_size, INPUT_DIM, std::move(batch_x), target),
            Matrix(batch_size, OUTPUT_DIM, std::move(batch_y), target)};
    }

public:
    Mnist_Data_Pipeline(const std::vector<float> &imgs,
                        const std::vector<float> &lbls,
                        uint32_t n_imgs,
                        std::size_t b_size,
                        Execution_Target exec_target)
        : images_data(imgs),
          labels_data(lbls),
          num_images(n_imgs),
          batch_size(b_size),
          target(exec_target) {}
};

double runBenchmark(Execution_Target target,
                    const std::vector<float> &images_data,
                    const std::vector<float> &labels_data,
                    uint32_t num_images,
                    Neural_Network &nn)
{
    std::size_t steps_per_epoch = num_images / BATCH_SIZE;
    nn.getLearningRate().setMaxEpoch(static_cast<int>(EPOCHS));

    Mnist_Data_Pipeline pipeline(images_data, labels_data, num_images, BATCH_SIZE, target);

    auto start_time = std::chrono::high_resolution_clock::now();

    nn.fit(pipeline, EPOCHS, steps_per_epoch);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    return duration.count();
}

int main()
{
    std::vector<float> images_data;
    std::vector<float> labels_data;
    uint32_t num_images = 0;

    loadMnistImages("data/train-images.idx3-ubyte", images_data, num_images);
    loadMnistLabels("data/train-labels.idx1-ubyte", labels_data, num_images);

    Execution_Target target = Execution_Target::VULKAN_GPU;
    Neural_Network nn(target);
    nn.setTrainingMode(true);

    nn.setLearningRate(std::make_unique<Cosine_Annealing>(0.001f, 1e-5f, static_cast<int>(EPOCHS)));
    nn.setOptimizer(std::make_unique<Adam_Optimizer>(nn.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f));
    nn.setCostFunction(std::make_unique<CCE_Cost>());

    nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1));
    nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(28, 28, 16, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());
    nn.addLayer(std::make_unique<MaxPool2d_Layer>(28, 28, 16, 2, 2, 0));

    nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1));
    nn.addLayer(std::make_unique<Batch_Norm2d_Layer>(14, 14, 32, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());
    nn.addLayer(std::make_unique<MaxPool2d_Layer>(14, 14, 32, 2, 2, 0));

    nn.addLayer(std::make_unique<Linear_Layer>(1568, 128));
    nn.addLayer(std::make_unique<Batch_Norm_Layer>(128, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Linear_Layer>(128, 10));
    nn.addLayer(std::make_unique<Softmax>(true));

    Logger::logMessage("Starting training benchmark with Data Augmentation...", LOG_INFO, true);
    double duration = runBenchmark(target, images_data, labels_data, num_images, nn);
    Logger::logMessage("Training completed in " + std::to_string(duration / 1000.0) + " s", LOG_INFO, true);

    Execution_Engine::getInstance().waitIdle();
    nn.saveInference("output/mnist/model.bin");
    Logger::logMessage("Inference model exported to model.bin", LOG_INFO, true);

    return 0;
}