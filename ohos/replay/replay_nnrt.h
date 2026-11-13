// The MLPs on the phone's NPU: Neural Network Runtime (NNRt) with HiAI Foundation behind it.
// OpenHarmony only. The models are the graphs of ohos/npu, all tensors fp16:
//   inference  ncr_<net>_inf_256x256_fp16.om: x (1, C, 256, 256) and the weight kernels -> y (1, 16, 256, 256)
//   training   ncr_<net>_train_<N>_fp16.om: the weight kernels, x (1, C, N, 1), target (1, 16, N, 1)
//              -> the gradient of every weight kernel and of x
// The weights are graph inputs: the inference graph gets the EMA copies, the training graph the
// training copies, both rewritten after every optimizer step (which runs on the CPU).
//
// Copy mode: the tiles are copied between the Vulkan buffers and tensors NNRt owns. Zero-copy
// (OH_NativeBuffer shared by Vulkan and NNRt) replaces the copies later.
#pragma once
#ifdef __OHOS__
#include "replay_nn.h"
#include "replay_vk.h"

#include <neural_network_runtime/neural_network_core.h>
#include <qos/qos.h>
#include <dlfcn.h>

namespace replay
{
class NnrtBackend : public NetworkBackend
{
public:
    NnrtBackend(const Mlp& diff, const Mlp& spec, const std::string& modelDir) : mModelDir(modelDir)
    {
        // HiAI Foundation registers the "HIAI_F" device with NNRt when it is loaded: load it first.
        mHiai = dlopen("libhiai_foundation.so", RTLD_NOW | RTLD_GLOBAL);
        if (!mHiai) fail(std::string("libhiai_foundation.so is not available: ") + dlerror());
        const size_t* ids = nullptr;
        uint32_t count = 0;
        if (OH_NNDevice_GetAllDevicesID(&ids, &count) != OH_NN_SUCCESS || count == 0) fail("no NNRt device");
        bool found = false;
        for (uint32_t i = 0; i < count && !found; i++)
        {
            const char* deviceName = nullptr;
            OH_NNDevice_GetName(ids[i], &deviceName);
            if (deviceName && std::strstr(deviceName, "HIAI"))
            {
                mDevice = ids[i];
                found = true;
            }
        }
        if (!found) fail("no HIAI_F device");
        // NPU calls from a thread with the default QoS are scheduled like background work.
        OH_QoS_SetThreadQoS(QOS_USER_INTERACTIVE);

        mNets[0].mlp = &diff;
        mNets[1].mlp = &spec;
        for (Net& net : mNets)
        {
            const std::string kind = net.mlp->specular ? "spec" : "diff";
            load(net.inference, *net.mlp, mModelDir + "/ncr_" + kind + "_inf_256x256_fp16.om", false, cfg::kTile);
            weightsChanged(net.mlp->specular);
        }
    }

    const char* name() const override { return "NPU (NNRt, HIAI_F), copy mode"; }

    void infer(bool specular, const uint16_t* x, uint16_t* y, uint32_t tiles) override
    {
        Model& m = mNets[specular ? 1 : 0].inference;
        const size_t xBytes = (size_t)(specular ? cfg::kSpecInputDim : cfg::kDiffInputDim) * cfg::kTile * 2;
        const size_t yBytes = (size_t)cfg::kOutputChannels * cfg::kTile * 2;
        for (uint32_t t = 0; t < tiles; t++)
        {
            auto start = std::chrono::steady_clock::now();
            std::memcpy(OH_NNTensor_GetDataBuffer(m.inputs[0]), reinterpret_cast<const char*>(x) + t * xBytes, xBytes);
            copyMs += msSince(start);
            start = std::chrono::steady_clock::now();
            run(m);
            runMs += msSince(start);
            start = std::chrono::steady_clock::now();
            std::memcpy(reinterpret_cast<char*>(y) + t * yBytes, OH_NNTensor_GetDataBuffer(m.outputs[0]), yBytes);
            copyMs += msSince(start);
            runs++;
        }
    }

    void train(bool specular, uint32_t tile, uint32_t tiles, const uint16_t* x, const uint16_t* target, float* weightGradient,
               uint16_t* inputGradient) override
    {
        Net& net = mNets[specular ? 1 : 0];
        if (net.trainingTile != tile)
        {
            if (net.trainingTile != 0) fail("the training tile size cannot change");
            load(net.training, *net.mlp, mModelDir + "/ncr_" + (specular ? "spec" : "diff") + "_train_" + std::to_string(tile) + "_fp16.om", true, tile);
            net.trainingTile = tile;
            weightsChanged(specular);
        }
        Model& m = net.training;
        const size_t layers = net.mlp->layerSizes().size();
        const size_t xBytes = (size_t)net.mlp->inputDim * tile * 2, targetBytes = (size_t)cfg::kOutputChannels * tile * 2;
        for (uint32_t t = 0; t < tiles; t++)
        {
            auto start = std::chrono::steady_clock::now();
            std::memcpy(OH_NNTensor_GetDataBuffer(m.inputs[layers]), reinterpret_cast<const char*>(x) + t * xBytes, xBytes);
            std::memcpy(OH_NNTensor_GetDataBuffer(m.inputs[layers + 1]), reinterpret_cast<const char*>(target) + t * targetBytes, targetBytes);
            trainCopyMs += msSince(start);
            start = std::chrono::steady_clock::now();
            run(m);
            trainRunMs += msSince(start);
            start = std::chrono::steady_clock::now();
            size_t offset = 0;
            for (size_t layer = 0; layer < layers; layer++)
            {
                const uint16_t* g = static_cast<const uint16_t*>(OH_NNTensor_GetDataBuffer(m.outputs[layer]));
                const size_t count = net.mlp->layerSizes()[layer];
                for (size_t i = 0; i < count; i++) weightGradient[offset + i] += halfToFloat(g[i]);
                offset += count;
            }
            std::memcpy(reinterpret_cast<char*>(inputGradient) + t * xBytes, OH_NNTensor_GetDataBuffer(m.outputs[layers]), xBytes);
            trainCopyMs += msSince(start);
            trainRuns++;
        }
    }

    void weightsChanged(bool specular) override
    {
        Net& net = mNets[specular ? 1 : 0];
        const std::vector<size_t> sizes = net.mlp->layerSizes();
        size_t offset = 0;
        for (size_t layer = 0; layer < sizes.size(); layer++)
        {
            // Weight kernels (Cout, Cin, 1, 1): the row-major matrices of the parameter block, layer after layer.
            std::memcpy(OH_NNTensor_GetDataBuffer(net.inference.inputs[1 + layer]), net.mlp->inference.data() + offset, sizes[layer] * 2);
            if (net.trainingTile != 0)
                std::memcpy(OH_NNTensor_GetDataBuffer(net.training.inputs[layer]), net.mlp->training.data() + offset, sizes[layer] * 2);
            offset += sizes[layer];
        }
    }

    // Accumulated: time inside RunSync and time copying tiles, for inference and for training.
    double runMs = 0.0, copyMs = 0.0, trainRunMs = 0.0, trainCopyMs = 0.0;
    uint32_t runs = 0, trainRuns = 0;

private:
    struct Model
    {
        std::vector<char> data;   // NNRt keeps pointing at the offline model
        OH_NNCompilation* compilation = nullptr;
        OH_NNExecutor* executor = nullptr;
        std::vector<NN_Tensor*> inputs, outputs;
    };
    struct Net
    {
        const Mlp* mlp = nullptr;
        Model inference, training;
        uint32_t trainingTile = 0;   // samples per tile of the loaded training graph; 0: not loaded
    };

    void* mHiai = nullptr;
    size_t mDevice = 0;
    std::string mModelDir;
    Net mNets[2];   // DiffNet, SpecNet

    [[noreturn]] static void fail(const std::string& what)
    {
        std::printf("FAIL NNRt: %s\n", what.c_str());
        std::exit(2);
    }

    static void run(Model& m)
    {
        const OH_NN_ReturnCode rc = OH_NNExecutor_RunSync(m.executor, m.inputs.data(), m.inputs.size(), m.outputs.data(), m.outputs.size());
        if (rc != OH_NN_SUCCESS) fail("OH_NNExecutor_RunSync failed with code " + std::to_string((int)rc));
    }

    void load(Model& m, const Mlp& mlp, const std::string& path, bool training, uint32_t tile)
    {
        const auto start = std::chrono::steady_clock::now();
        m.data = readFile(path);
        m.compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(m.data.data(), m.data.size());
        if (!m.compilation) fail("cannot construct a compilation for " + path);
        OH_NNCompilation_SetDevice(m.compilation, mDevice);
        OH_NNCompilation_SetPerformanceMode(m.compilation, OH_NN_PERFORMANCE_EXTREME);
        OH_NNCompilation_SetPriority(m.compilation, OH_NN_PRIORITY_HIGH);
        if (OH_NNCompilation_Build(m.compilation) != OH_NN_SUCCESS) fail("cannot build " + path);
        m.executor = OH_NNExecutor_Construct(m.compilation);
        if (!m.executor) fail("cannot create an executor for " + path);

        // The byte sizes the graph must have, in the order of its inputs and outputs.
        const std::vector<size_t> layers = mlp.layerSizes();
        const size_t xBytes = (size_t)mlp.inputDim * tile * 2, yBytes = (size_t)cfg::kOutputChannels * tile * 2;
        std::vector<size_t> inputBytes, outputBytes;
        if (!training) inputBytes.push_back(xBytes);
        for (size_t count : layers) inputBytes.push_back(count * 2);
        if (training)
        {
            inputBytes.push_back(xBytes);
            inputBytes.push_back(yBytes);
            for (size_t count : layers) outputBytes.push_back(count * 2);
            outputBytes.push_back(xBytes);
        }
        else
        {
            outputBytes.push_back(yBytes);
        }

        size_t inputCount = 0, outputCount = 0;
        OH_NNExecutor_GetInputCount(m.executor, &inputCount);
        OH_NNExecutor_GetOutputCount(m.executor, &outputCount);
        if (inputCount != inputBytes.size() || outputCount != outputBytes.size()) fail(path + " does not have the inputs and outputs of an NCR graph");
        for (size_t i = 0; i < inputCount + outputCount; i++)
        {
            const bool input = i < inputCount;
            const size_t index = input ? i : i - inputCount;
            NN_TensorDesc* desc = input ? OH_NNExecutor_CreateInputTensorDesc(m.executor, index) : OH_NNExecutor_CreateOutputTensorDesc(m.executor, index);
            size_t bytes = 0;
            OH_NNTensorDesc_GetByteSize(desc, &bytes);
            NN_Tensor* tensor = OH_NNTensor_Create(mDevice, desc);
            OH_NNTensorDesc_Destroy(&desc);
            const std::string which = std::string(input ? "input " : "output ") + std::to_string(index) + " of " + path;
            if (!tensor) fail("cannot create the tensor of " + which);
            if (bytes != (input ? inputBytes : outputBytes)[index]) fail(which + " has " + std::to_string(bytes) + " bytes, not what an NCR graph has there");
            std::memset(OH_NNTensor_GetDataBuffer(tensor), 0, bytes);
            (input ? m.inputs : m.outputs).push_back(tensor);
        }
        std::printf("NNRt: %s loaded in %.0f ms\n", path.substr(path.find_last_of('/') + 1).c_str(), msSince(start));
    }
};
}  // namespace replay
#endif  // __OHOS__
