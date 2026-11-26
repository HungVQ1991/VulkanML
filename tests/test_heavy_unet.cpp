#include "test_common.h"

namespace TestBatchNormUnet
{
bool runSuite()
{
    const std::string dataset_path = "data/chest-segment/chest_seg_320x320.bin";
    const std::string ckpt_path = "output/unet/unet_latest.nnck";
    if (!std::filesystem::exists(dataset_path) || !std::filesystem::exists(ckpt_path))
    {
        return true;
    }

    try
    {
        Execution_Target target = Execution_Target::VULKAN_GPU;
        auto &engine = Execution_Engine::getInstance();

        chexlocalize::Chest_Seg_Dataset dataset;
        dataset.open(dataset_path);

        Neural_Network net(target);
        net.addLayer(std::make_unique<U_Net_Layer>(320, 320, 1, 1, 16, target));
        net.enableMixedPrecision(true);

        net.loadTrainingCheckpoint(ckpt_path, 30);
        if (net.getLayers().empty() || !net.getLayers()[0]) return false;
        auto *unet = dynamic_cast<U_Net_Layer*>(net.getLayers()[0].get());
        if (unet)
        {
            unet->inspectAllBatchNorms();
        }

        std::vector<size_t> indices = { 100, 101 };
        auto [b_imgs, b_masks] = dataset.getBatch(indices, target, false, nullptr);

        net.setTrainingMode(true);
        Tensor pred_train = net.forward(b_imgs);
        engine.executeGraph();
        for (float v : pred_train.getData())
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        net.setTrainingMode(false);
        Tensor pred_eval = net.forward(b_imgs);
        engine.executeGraph();
        for (float v : pred_eval.getData())
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        std::vector<size_t> train_indices = { 0, 1 };
        auto [tr_imgs, tr_masks] = dataset.getBatch(train_indices, target, false, nullptr);
        Tensor pred_tr_eval = net.forward(tr_imgs);
        engine.executeGraph();
        for (float v : pred_tr_eval.getData())
        {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "  [FAIL] Exception in BatchNorm U-Net test: " << e.what() << "\n";
        return false;
    }
}
}

void runHeavyUnetSuite(Test_Result_Counter &counter)
{
    counter.record("  BatchNorm U-Net Verification:      ", TestBatchNormUnet::runSuite());
}
