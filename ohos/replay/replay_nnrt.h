// The MLPs on the phone's NPU: Neural Network Runtime (NNRt) with HiAI Foundation behind it.
// OpenHarmony only. The models are the inference graphs of ohos/npu (ncr_<net>_inf_256x256_fp16.om):
// input x (1, C, 256, 256) and the weight kernels, output y (1, 16, 256, 256), all fp16.
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
    NnrtBackend(const Checkpoint& checkpoint, const std::string& modelDir)
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

        load(mDiff, modelDir + "/ncr_diff_inf_256x256_fp16.om", checkpoint.diffWeightsHalf, {64 * 32, 64 * 64, 64 * 64, 16 * 64});
        load(mSpec, modelDir + "/ncr_spec_inf_256x256_fp16.om", checkpoint.specWeightsHalf, {64 * 64, 64 * 64, 64 * 64, 64 * 64, 16 * 64});
    }

    const char* name() const override { return "NPU (NNRt, HIAI_F), copy mode"; }

    void infer(bool specular, const uint16_t* x, uint16_t* y, uint32_t tiles) override
    {
        Model& m = specular ? mSpec : mDiff;
        const size_t xBytes = (size_t)(specular ? cfg::kSpecInputDim : cfg::kDiffInputDim) * cfg::kTile * 2;
        const size_t yBytes = (size_t)cfg::kOutputChannels * cfg::kTile * 2;
        for (uint32_t t = 0; t < tiles; t++)
        {
            auto start = std::chrono::steady_clock::now();
            std::memcpy(OH_NNTensor_GetDataBuffer(m.inputs[0]), reinterpret_cast<const char*>(x) + t * xBytes, xBytes);
            copyMs += msSince(start);
            start = std::chrono::steady_clock::now();
            const OH_NN_ReturnCode rc = OH_NNExecutor_RunSync(m.executor, m.inputs.data(), m.inputs.size(), m.outputs.data(), m.outputs.size());
            if (rc != OH_NN_SUCCESS) fail("OH_NNExecutor_RunSync failed with code " + std::to_string((int)rc));
            runMs += msSince(start);
            start = std::chrono::steady_clock::now();
            std::memcpy(reinterpret_cast<char*>(y) + t * yBytes, OH_NNTensor_GetDataBuffer(m.outputs[0]), yBytes);
            copyMs += msSince(start);
            runs++;
        }
    }

    double runMs = 0.0, copyMs = 0.0;   // accumulated: time inside RunSync, time copying tiles
    uint32_t runs = 0;

private:
    struct Model
    {
        std::vector<char> data;   // NNRt keeps pointing at the offline model
        OH_NNCompilation* compilation = nullptr;
        OH_NNExecutor* executor = nullptr;
        std::vector<NN_Tensor*> inputs, outputs;
    };

    void* mHiai = nullptr;
    size_t mDevice = 0;
    Model mDiff, mSpec;

    [[noreturn]] static void fail(const std::string& what)
    {
        std::printf("FAIL NNRt: %s\n", what.c_str());
        std::exit(2);
    }

    void load(Model& m, const std::string& path, const std::vector<uint16_t>& weights, const std::vector<size_t>& layerSizes)
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

        size_t inputCount = 0, outputCount = 0;
        OH_NNExecutor_GetInputCount(m.executor, &inputCount);
        OH_NNExecutor_GetOutputCount(m.executor, &outputCount);
        if (inputCount != layerSizes.size() + 1 || outputCount != 1) fail(path + " does not have the inputs of an NCR inference graph");
        size_t offset = 0;
        for (size_t i = 0; i < inputCount; i++)
        {
            NN_TensorDesc* desc = OH_NNExecutor_CreateInputTensorDesc(m.executor, i);
            size_t bytes = 0;
            OH_NNTensorDesc_GetByteSize(desc, &bytes);
            NN_Tensor* tensor = OH_NNTensor_Create(mDevice, desc);
            OH_NNTensorDesc_Destroy(&desc);
            if (!tensor) fail("cannot create input tensor " + std::to_string(i) + " of " + path);
            if (i > 0)
            {
                // Weight kernels (Cout, Cin, 1, 1): the checkpoint's row-major matrices, layer after layer.
                const size_t count = layerSizes[i - 1];
                if (bytes != count * 2) fail("weight tensor " + std::to_string(i) + " of " + path + " has an unexpected size");
                std::memcpy(OH_NNTensor_GetDataBuffer(tensor), weights.data() + offset, bytes);
                offset += count;
            }
            m.inputs.push_back(tensor);
        }
        NN_TensorDesc* desc = OH_NNExecutor_CreateOutputTensorDesc(m.executor, 0);
        m.outputs.push_back(OH_NNTensor_Create(mDevice, desc));
        OH_NNTensorDesc_Destroy(&desc);
        if (!m.outputs[0]) fail("cannot create the output tensor of " + path);
        std::printf("NNRt: %s loaded in %.0f ms\n", path.substr(path.find_last_of('/') + 1).c_str(), msSince(start));
    }
};
}  // namespace replay
#endif  // __OHOS__
