// Stand-alone NPU probe: loads one offline model (.om or .omc) through NNRt on the HiAI Foundation
// device, times OH_NNExecutor_RunSync and, when test vectors lie next to it, checks the outputs.
//
//   npu_probe <model file> [--runs N] [--data <prefix>] [--default-qos]
//
// --data <prefix>: inputs are read from <prefix>.in<i>.bin (fp16) and the outputs compared with
// <prefix>.out<i>.bin (float32), as written by ohos/npu/export_models.py. Without it, or for a
// missing file, the inputs are random.
// Built with the OpenHarmony NDK alone and run from a shell on the phone (see run.sh).
#include <neural_network_runtime/neural_network_core.h>
#include <qos/qos.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{
double msSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

float halfToFloat(uint16_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u))
            {
                mantissa <<= 1;
                --exponent;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
        }
    }
    else if (exponent == 31)
    {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Values the probe generates are normal and far from the fp16 limits: truncation is enough.
uint16_t floatToHalf(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000u;
    const int exponent = int((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent <= 0)
    {
        return sign;
    }
    if (exponent >= 31)
    {
        return sign | 0x7c00u;
    }
    return sign | uint16_t(exponent << 10) | uint16_t((bits >> 13) & 0x3ffu);
}

struct Tensor
{
    NN_Tensor* handle = nullptr;
    std::vector<int32_t> shape;
    OH_NN_DataType type = OH_NN_UNKNOWN;
    size_t bytes = 0;
    size_t elements = 0;

    std::string describe() const
    {
        std::string text = "{";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            text += (i ? "," : "") + std::to_string(shape[i]);
        }
        return text + "} " + (type == OH_NN_FLOAT16 ? "fp16" : type == OH_NN_FLOAT32 ? "fp32" : "type " + std::to_string((int)type));
    }
};

Tensor createTensor(size_t device, NN_TensorDesc* desc)
{
    Tensor tensor;
    int32_t* shape = nullptr;
    size_t rank = 0;
    OH_NNTensorDesc_GetShape(desc, &shape, &rank);
    tensor.shape.assign(shape, shape + rank);
    OH_NNTensorDesc_GetDataType(desc, &tensor.type);
    OH_NNTensorDesc_GetByteSize(desc, &tensor.bytes);
    tensor.elements = 1;
    for (int32_t d : tensor.shape)
    {
        tensor.elements *= (size_t)d;
    }
    tensor.handle = OH_NNTensor_Create(device, desc);
    OH_NNTensorDesc_Destroy(&desc);
    return tensor;
}

std::vector<char> readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        return {};
    }
    std::vector<char> data((size_t)file.tellg());
    file.seekg(0);
    file.read(data.data(), (std::streamsize)data.size());
    return data;
}

uint32_t gSeed = 12345;
float nextRandom()
{
    gSeed = gSeed * 1664525u + 1013904223u;
    return float(gSeed >> 8) * (1.0f / 16777216.0f);
}

/// Returns true when the tensor was filled from the test vector.
bool fillInput(const Tensor& tensor, const std::string& vectorFile)
{
    void* buffer = OH_NNTensor_GetDataBuffer(tensor.handle);
    const std::vector<char> data = vectorFile.empty() ? std::vector<char>() : readFile(vectorFile);
    if (tensor.type == OH_NN_FLOAT16 && data.size() == tensor.elements * 2)
    {
        std::memcpy(buffer, data.data(), data.size());
        return true;
    }
    // Sample tensors (leading dimension 1) in [0, 1), weight kernels in [-0.25, 0.25).
    const bool weights = !tensor.shape.empty() && tensor.shape[0] != 1;
    for (size_t i = 0; i < tensor.elements; ++i)
    {
        const float value = weights ? (nextRandom() - 0.5f) * 0.5f : nextRandom();
        if (tensor.type == OH_NN_FLOAT16)
        {
            static_cast<uint16_t*>(buffer)[i] = floatToHalf(value);
        }
        else
        {
            static_cast<float*>(buffer)[i] = value;
        }
    }
    return false;
}

float element(const Tensor& tensor, const void* buffer, size_t i)
{
    return tensor.type == OH_NN_FLOAT16 ? halfToFloat(static_cast<const uint16_t*>(buffer)[i]) : static_cast<const float*>(buffer)[i];
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: npu_probe <model file> [--runs N] [--data <prefix>] [--default-qos]\n");
        return 2;
    }
    const std::string modelFile = argv[1];
    int runs = 30;
    std::string dataPrefix;
    bool interactiveQos = true;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--runs" && i + 1 < argc)
        {
            runs = std::atoi(argv[++i]);
        }
        else if (arg == "--data" && i + 1 < argc)
        {
            dataPrefix = argv[++i];
        }
        else if (arg == "--default-qos")
        {
            interactiveQos = false;
        }
    }
    const std::string name = modelFile.substr(modelFile.find_last_of('/') + 1);

    // HiAI Foundation registers the "HIAI_F" device with NNRt when it is loaded, so load it first.
    void* hiai = dlopen("libhiai_foundation.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hiai)
    {
        std::printf("libhiai_foundation.so not available: %s\n", dlerror());
    }
    else if (auto getVersion = reinterpret_cast<const char* (*)()>(dlsym(hiai, "HMS_HiAI_GetVersion")))
    {
        const char* version = getVersion();
        std::printf("HiAI Foundation version %s\n", version ? version : "?");
    }
    if (interactiveQos)
    {
        std::printf("OH_QoS_SetThreadQoS(QOS_USER_INTERACTIVE) -> %d\n", OH_QoS_SetThreadQoS(QOS_USER_INTERACTIVE));
    }

    const size_t* ids = nullptr;
    uint32_t count = 0;
    if (OH_NNDevice_GetAllDevicesID(&ids, &count) != OH_NN_SUCCESS || count == 0)
    {
        std::printf("FAIL %s: no NNRt device\n", name.c_str());
        return 1;
    }
    size_t device = 0;
    bool found = false;
    for (uint32_t i = 0; i < count; ++i)
    {
        const char* deviceName = nullptr;
        OH_NNDevice_GetName(ids[i], &deviceName);
        std::printf("NNRt device %zu: %s\n", ids[i], deviceName ? deviceName : "?");
        if (!found && deviceName && std::strstr(deviceName, "HIAI"))
        {
            device = ids[i];
            found = true;
        }
    }
    if (!found)
    {
        std::printf("FAIL %s: no HIAI_F device\n", name.c_str());
        return 1;
    }

    if (hiai && modelFile.size() > 4 && modelFile.compare(modelFile.size() - 4, 4, ".omc") == 0)
    {
        if (auto check = reinterpret_cast<int (*)(const char*)>(dlsym(hiai, "HMS_HiAICompatibility_CheckFromFile")))
        {
            std::printf("precompiled model compatibility: %s\n", check(modelFile.c_str()) == 0 ? "compatible" : "INCOMPATIBLE");
        }
    }

    const std::vector<char> modelData = readFile(modelFile);  // NNRt keeps pointing at this buffer
    if (modelData.empty())
    {
        std::printf("FAIL %s: cannot read the model file\n", name.c_str());
        return 1;
    }
    const auto loadStart = std::chrono::steady_clock::now();
    OH_NNCompilation* compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelData.data(), modelData.size());
    if (!compilation)
    {
        std::printf("FAIL %s: OH_NNCompilation_ConstructWithOfflineModelBuffer\n", name.c_str());
        return 1;
    }
    OH_NNCompilation_SetDevice(compilation, device);
    OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_EXTREME);
    OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH);
    if (hiai)
    {
        // NPU first; an operator the NPU refuses falls back to the CPU instead of failing the build.
        if (auto setOrder = reinterpret_cast<OH_NN_ReturnCode (*)(OH_NNCompilation*, int*, size_t)>(dlsym(hiai, "HMS_HiAIOptions_SetModelDeviceOrder")))
        {
            int order[2] = {0 /* NPU */, 1 /* CPU */};
            setOrder(compilation, order, 2);
        }
        if (auto setFallback = reinterpret_cast<OH_NN_ReturnCode (*)(OH_NNCompilation*, int)>(dlsym(hiai, "HMS_HiAIOptions_SetFallbackMode")))
        {
            setFallback(compilation, 0 /* enabled */);
        }
    }
    OH_NN_ReturnCode rc = OH_NNCompilation_Build(compilation);
    if (rc != OH_NN_SUCCESS)
    {
        std::printf("FAIL %s: OH_NNCompilation_Build code %d\n", name.c_str(), (int)rc);
        return 1;
    }
    OH_NNExecutor* executor = OH_NNExecutor_Construct(compilation);
    if (!executor)
    {
        std::printf("FAIL %s: OH_NNExecutor_Construct\n", name.c_str());
        return 1;
    }
    const double loadMs = msSince(loadStart);

    size_t inputCount = 0, outputCount = 0;
    OH_NNExecutor_GetInputCount(executor, &inputCount);
    OH_NNExecutor_GetOutputCount(executor, &outputCount);
    std::vector<Tensor> inputs, outputs;
    std::vector<NN_Tensor*> inputHandles, outputHandles;
    size_t vectorInputs = 0;
    for (size_t i = 0; i < inputCount; ++i)
    {
        inputs.push_back(createTensor(device, OH_NNExecutor_CreateInputTensorDesc(executor, i)));
        if (!inputs.back().handle)
        {
            std::printf("FAIL %s: OH_NNTensor_Create for input %zu %s\n", name.c_str(), i, inputs.back().describe().c_str());
            return 1;
        }
        const bool fromVector = fillInput(inputs.back(), dataPrefix.empty() ? "" : dataPrefix + ".in" + std::to_string(i) + ".bin");
        vectorInputs += fromVector;
        std::printf("  input  %zu %s%s\n", i, inputs.back().describe().c_str(), fromVector ? " (test vector)" : "");
        inputHandles.push_back(inputs.back().handle);
    }
    for (size_t i = 0; i < outputCount; ++i)
    {
        outputs.push_back(createTensor(device, OH_NNExecutor_CreateOutputTensorDesc(executor, i)));
        if (!outputs.back().handle)
        {
            std::printf("FAIL %s: OH_NNTensor_Create for output %zu %s\n", name.c_str(), i, outputs.back().describe().c_str());
            return 1;
        }
        std::printf("  output %zu %s\n", i, outputs.back().describe().c_str());
        outputHandles.push_back(outputs.back().handle);
    }

    const auto firstStart = std::chrono::steady_clock::now();
    rc = OH_NNExecutor_RunSync(executor, inputHandles.data(), inputHandles.size(), outputHandles.data(), outputHandles.size());
    const double firstMs = msSince(firstStart);
    if (rc != OH_NN_SUCCESS)
    {
        std::printf("FAIL %s: OH_NNExecutor_RunSync code %d\n", name.c_str(), (int)rc);
        return 1;
    }

    // Numerical check against the float32 reference, when every input came from the test vectors.
    std::string check = "check none";
    if (vectorInputs == inputCount)
    {
        double worst = 0.0;       // largest element error relative to the largest reference element of that output
        double worstL2 = 0.0;     // largest L2 error relative to the reference norm
        size_t notFinite = 0;
        size_t compared = 0;
        for (size_t i = 0; i < outputCount; ++i)
        {
            const std::vector<char> data = readFile(dataPrefix + ".out" + std::to_string(i) + ".bin");
            if (data.size() != outputs[i].elements * sizeof(float))
            {
                std::printf("  output %zu: no reference of the right size (%zu bytes)\n", i, data.size());
                continue;
            }
            const float* reference = reinterpret_cast<const float*>(data.data());
            const void* buffer = OH_NNTensor_GetDataBuffer(outputs[i].handle);
            double maxError = 0.0, maxReference = 0.0, sumError = 0.0, sumReference = 0.0;
            for (size_t e = 0; e < outputs[i].elements; ++e)
            {
                const float value = element(outputs[i], buffer, e);
                if (!std::isfinite(value))
                {
                    ++notFinite;
                    continue;
                }
                const double error = std::fabs((double)value - reference[e]);
                maxError = std::max(maxError, error);
                maxReference = std::max(maxReference, (double)std::fabs(reference[e]));
                sumError += error * error;
                sumReference += (double)reference[e] * reference[e];
            }
            const double relative = maxReference > 0.0 ? maxError / maxReference : 0.0;
            const double relativeL2 = sumReference > 0.0 ? std::sqrt(sumError / sumReference) : 0.0;
            std::printf("  output %zu vs reference: L2 error %.2e, worst element %.2e of max |ref| %.3g\n", i, relativeL2, relative, maxReference);
            worst = std::max(worst, relative);
            worstL2 = std::max(worstL2, relativeL2);
            ++compared;
        }
        char text[160];
        std::snprintf(text, sizeof(text), "check %zu/%zu outputs: L2 %.2e worst %.2e nan/inf %zu", compared, outputCount, worstL2, worst, notFinite);
        check = text;
    }

    std::vector<double> times;
    for (int i = 0; i < runs; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        rc = OH_NNExecutor_RunSync(executor, inputHandles.data(), inputHandles.size(), outputHandles.data(), outputHandles.size());
        times.push_back(msSince(start));
        if (rc != OH_NN_SUCCESS)
        {
            std::printf("FAIL %s: OH_NNExecutor_RunSync code %d in run %d\n", name.c_str(), (int)rc, i);
            return 1;
        }
    }
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (double t : times)
    {
        sum += t;
    }
    std::printf("RESULT %s | load %.1f ms | first run %.2f ms | %d runs: median %.2f min %.2f avg %.2f max %.2f ms | %s\n",
                name.c_str(), loadMs, firstMs, runs, times.empty() ? 0.0 : times[times.size() / 2],
                times.empty() ? 0.0 : times.front(), times.empty() ? 0.0 : sum / times.size(),
                times.empty() ? 0.0 : times.back(), check.c_str());

    for (Tensor& tensor : inputs)
    {
        OH_NNTensor_Destroy(&tensor.handle);
    }
    for (Tensor& tensor : outputs)
    {
        OH_NNTensor_Destroy(&tensor.handle);
    }
    OH_NNExecutor_Destroy(&executor);
    OH_NNCompilation_Destroy(&compilation);
    return 0;
}
