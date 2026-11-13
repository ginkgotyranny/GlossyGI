// The networks on the CPU, in float32: the reference backend of the replayer. It follows
// ohos/npu/ncr_reference.py (checked against the CUDA implementation by verify_tcnn.py) and stands in
// for the two things the phone runs elsewhere: the hash-grid lookups (GPU compute) and the MLPs (NPU).
// Slow, but it needs nothing besides the checkpoint, and it is what the faster paths are compared with.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace replay
{
inline float halfToFloat(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ffu, bits;
    if (e == 31)
    {
        bits = sign | 0x7f800000u | (m << 13);   // infinity or NaN
    }
    else if (e != 0)
    {
        bits = sign | ((e + 112) << 23) | (m << 13);
    }
    else if (m == 0)
    {
        bits = sign;
    }
    else
    {
        // Subnormal, m * 2^-24: shift the leading one up to where a normal number keeps it.
        uint32_t shift = 0;
        while (!(m & 0x400u))
        {
            m <<= 1;
            shift++;
        }
        bits = sign | ((113 - shift) << 23) | ((m & 0x3ffu) << 13);
    }
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

/// Round to nearest, ties to even: what a float -> __half cast does in CUDA.
inline uint16_t floatToHalf(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xff) return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    const int e = (int)exponent - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0)
    {
        if (e < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - e);
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1), midpoint = 1u << (shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)e << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1))) half++;   // may carry into the exponent: still correct
    return (uint16_t)(sign | half);
}

// NCRConfig.h and NetworkParams.slang, as in ncr_reference.py.
namespace cfg
{
constexpr uint32_t kHashMapSize = 1u << 18;
constexpr uint32_t kPrimLevels = 6, kPrimFeatures = 2, kPrimBaseRes = 32;
constexpr uint32_t kClsLevels = 6, kClsFeatures = 4, kClsBaseRes = 16, kNumClusters = 4;
constexpr uint32_t kDiffInputDim = 32, kSpecInputDim = 64, kOutputDim = 4;
constexpr uint32_t kDiffPos = 12;
constexpr uint32_t kSpecClsEnc = 12, kSpecPos = 28, kSpecClsPos = 44, kSpecClsScale = 56, kSpecClsWeight = 60;
constexpr uint32_t kBatchAlign = 256;
constexpr float kSquarePlusK = 10.0f;
// NPU tiles: a batch goes to the NPU as tiles of kTile samples, channel-planar, shape (1, C, 256, 256).
constexpr uint32_t kTile = 65536;
constexpr uint32_t kOutputChannels = 16;   // the MLPs' padded output width; the first kOutputDim are used
}  // namespace cfg

inline uint32_t padUp(uint32_t x, uint32_t align) { return (x + align - 1) & ~(align - 1); }

struct GridLevel { uint32_t res, size, offset; };

inline uint32_t levelEntries(uint32_t baseRes, uint32_t level)
{
    const uint64_t res1 = (uint64_t)(baseRes << level) + 1;
    return (uint32_t)std::min<uint64_t>(res1 * res1 * res1, cfg::kHashMapSize);
}

inline GridLevel gridLevel(uint32_t baseRes, uint32_t level)
{
    GridLevel l;
    l.res = baseRes << level;
    l.size = levelEntries(baseRes, level);
    l.offset = 0;
    for (uint32_t i = 0; i < level; i++) l.offset += levelEntries(baseRes, i);
    return l;
}

inline uint32_t gridEntries(uint32_t baseRes, uint32_t levels)
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < levels; i++) total += levelEntries(baseRes, i);
    return total;
}

struct Corners { uint32_t entry[8]; float weight[8]; };

/// The eight entries around a position and their trilinear weights. Dense vertex index modulo the
/// level size on every level, and float -> unsigned saturating at 0: see ncr_reference.py.
inline Corners corners(const float* pos, const GridLevel& l)
{
    Corners c;
    float frac[3];
    uint32_t base[3];
    for (int a = 0; a < 3; a++)
    {
        const float grid = pos[a] * (float)l.res;
        base[a] = grid > 0.0f ? (uint32_t)grid : 0u;
        frac[a] = grid - (float)base[a];
    }
    const uint32_t res1 = l.res + 1;
    for (uint32_t k = 0; k < 8; k++)
    {
        const uint32_t o[3] = {k & 1, (k >> 1) & 1, (k >> 2) & 1};
        c.entry[k] = l.offset + ((res1 * (base[0] + o[0]) + (base[1] + o[1])) * res1 + (base[2] + o[2])) % l.size;
        c.weight[k] = (o[0] ? frac[0] : 1 - frac[0]) * (o[1] ? frac[1] : 1 - frac[1]) * (o[2] ? frac[2] : 1 - frac[2]);
    }
    return c;
}

/// The two cluster-grid levels that bracket a cluster radius, and their blend weights.
inline void clusterLevels(float scale, uint32_t& upper, uint32_t& lower, float& upperWeight, float& lowerWeight)
{
    auto voxel = [](int i) { return 1.0f / (float)(cfg::kClsBaseRes << i); };
    const int n = (int)cfg::kClsLevels;
    int level = n;
    for (int i = 0; i < n; ++i)
        if (scale > voxel(i)) { level = i; break; }
    if (level == 0) { upperWeight = 0.0f; upper = 0; }
    else if (level == n) { upperWeight = 1.0f; upper = (uint32_t)n - 1; }
    else { upperWeight = (voxel(level - 1) - scale) / (voxel(level - 1) - voxel(level)); upper = (uint32_t)level; }
    level = -1;
    for (int i = n - 1; i >= 0; --i)
        if (scale < voxel(i)) { level = i; break; }
    if (level == -1) { lowerWeight = 1.0f; lower = 0; }
    else if (level == n - 1) { lowerWeight = 0.0f; lower = (uint32_t)n - 1; }
    else { lowerWeight = (scale - voxel(level + 1)) / (voxel(level) - voxel(level + 1)); lower = (uint32_t)level; }
}

/// An NRCK v2 checkpoint (Model.cu): a header with the element counts, four float32 master blocks
/// (primary grid, cluster grid, DiffNet, SpecNet) and the four fp16 EMA copies inference reads.
/// The Adam moments are not part of it: training resumes with fresh moment estimates.
struct Checkpoint
{
    struct Header { char magic[4]; uint32_t version, counts[4]; };

    std::vector<float> primGridMaster, clsGridMaster, diffMaster, specMaster;
    std::vector<uint16_t> primGrid, clsGrid;   // fp16 EMA copies, as the GPU reads them
    std::vector<uint16_t> diffEma, specEma;    // MLP weights, row-major (out, in) per layer

    static Header expectedHeader()
    {
        Header h = {{'N', 'R', 'C', 'K'}, 2, {gridEntries(cfg::kPrimBaseRes, cfg::kPrimLevels) * cfg::kPrimFeatures,
                                              gridEntries(cfg::kClsBaseRes, cfg::kClsLevels) * cfg::kClsFeatures, 11264, 17408}};
        return h;
    }

    void load(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        Header header = {};
        f.read(reinterpret_cast<char*>(&header), sizeof(header));
        const Header expected = expectedHeader();
        if (!f || std::memcmp(&header, &expected, sizeof(header)) != 0)
        {
            std::printf("FAIL %s is not an NRCK v2 checkpoint of this network configuration\n", path.c_str());
            std::exit(2);
        }
        std::vector<float>* masters[4] = {&primGridMaster, &clsGridMaster, &diffMaster, &specMaster};
        std::vector<uint16_t>* emas[4] = {&primGrid, &clsGrid, &diffEma, &specEma};
        for (int i = 0; i < 4; i++)
        {
            masters[i]->resize(header.counts[i]);
            f.read(reinterpret_cast<char*>(masters[i]->data()), (std::streamsize)header.counts[i] * 4);
        }
        for (int i = 0; i < 4; i++)
        {
            emas[i]->resize(header.counts[i]);
            f.read(reinterpret_cast<char*>(emas[i]->data()), (std::streamsize)header.counts[i] * 2);
        }
        if (!f)
        {
            std::printf("FAIL %s is truncated\n", path.c_str());
            std::exit(2);
        }
    }

    /// Writes a checkpoint Falcor loads (loadModelPath). Blocks in the order of the file.
    static void save(const std::string& path, const float* const masters[4], const uint16_t* const emas[4])
    {
        const Header header = expectedHeader();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&header), sizeof(header));
        for (int i = 0; i < 4; i++) f.write(reinterpret_cast<const char*>(masters[i]), (std::streamsize)header.counts[i] * 4);
        for (int i = 0; i < 4; i++) f.write(reinterpret_cast<const char*>(emas[i]), (std::streamsize)header.counts[i] * 2);
        if (!f.good())
        {
            std::printf("FAIL cannot write %s\n", path.c_str());
            std::exit(2);
        }
    }
};

/// tiny-cuda-nn's EMA(Adam) as Model.cu configures it; see adam_ema_step() in ncr_reference.py.
namespace adam
{
constexpr float kBeta1 = 0.9f, kBeta2 = 0.999f, kEpsilon = 1e-8f;
constexpr float kL2Reg = 1e-8f;        // MLP weights only
constexpr float kLossScale = 128.0f;   // the gradients carry it; the optimizer divides it out
/// Fixed-point units per 1.0 in which the GPU sums grid gradients (ScatterGradients, GridOptimizer).
constexpr float kGridGradientScale = 16777216.0f;   // 2^24

inline float emaDebiasOld(float decay, uint32_t step) { return 1.0f - std::pow(decay, (float)(step - 1)); }
inline float emaDebiasNew(float decay, uint32_t step) { return 1.0f / (1.0f - std::pow(decay, (float)step)); }
}  // namespace adam

/// One of the two MLPs: its weights and its optimizer, which runs on the CPU (28,672 weights in all).
/// `training` is the fp16 copy of the float32 masters the training graph computes with; `inference`
/// is the fp16 EMA of those copies, which the inference graph computes with.
struct Mlp
{
    bool specular = false;
    uint32_t inputDim = 0, hiddenLayers = 0;
    std::vector<float> master, firstMoment, secondMoment;
    std::vector<uint16_t> training, inference;
    std::vector<float> trainingFloat, inferenceFloat;   // the fp16 copies as floats, for the CPU backends
    uint32_t steps = 0;

    void init(bool spec, const std::vector<float>& masters, const std::vector<uint16_t>& ema)
    {
        specular = spec;
        inputDim = spec ? cfg::kSpecInputDim : cfg::kDiffInputDim;
        hiddenLayers = spec ? 4 : 3;
        master = masters;
        inference = ema;
        firstMoment.assign(master.size(), 0.0f);
        secondMoment.assign(master.size(), 0.0f);
        training.resize(master.size());
        for (size_t i = 0; i < master.size(); i++) training[i] = floatToHalf(master[i]);
        refresh();
    }

    /// Elements of every weight matrix, in the order of the parameter block: (64, in), (64, 64)..., (16, 64).
    std::vector<size_t> layerSizes() const
    {
        std::vector<size_t> sizes = {(size_t)64 * inputDim};
        for (uint32_t i = 1; i < hiddenLayers; i++) sizes.push_back(64 * 64);
        sizes.push_back((size_t)cfg::kOutputChannels * 64);
        return sizes;
    }

    /// One step. `gradient` is dL/dw of the whole batch, with the loss scale.
    void step(const float* gradient, float learningRate, float emaDecay)
    {
        steps++;
        const float t = (float)steps;
        const float rate = learningRate * std::sqrt(1.0f - std::pow(adam::kBeta2, t)) / (1.0f - std::pow(adam::kBeta1, t));
        const float debiasOld = adam::emaDebiasOld(emaDecay, steps), debiasNew = adam::emaDebiasNew(emaDecay, steps);
        for (size_t i = 0; i < master.size(); i++)
        {
            float g = gradient[i] / adam::kLossScale;
            if (!std::isfinite(g)) g = 0.0f;
            g += adam::kL2Reg * master[i];
            const float m = firstMoment[i] = adam::kBeta1 * firstMoment[i] + (1.0f - adam::kBeta1) * g;
            const float v = secondMoment[i] = adam::kBeta2 * secondMoment[i] + (1.0f - adam::kBeta2) * g * g;
            master[i] -= rate / (std::sqrt(v) + adam::kEpsilon) * m;
            training[i] = floatToHalf(master[i]);
            inference[i] = floatToHalf((halfToFloat(inference[i]) * emaDecay * debiasOld + halfToFloat(training[i]) * (1.0f - emaDecay)) * debiasNew);
        }
        refresh();
    }

    void refresh()
    {
        trainingFloat.resize(master.size());
        inferenceFloat.resize(master.size());
        for (size_t i = 0; i < master.size(); i++)
        {
            trainingFloat[i] = halfToFloat(training[i]);
            inferenceFloat[i] = halfToFloat(inference[i]);
        }
    }
};

/// Tiles as a backend gets them: mapped for the CPU and, where the buffer is shared with the NPU
/// (Device::makeSharedBuffer), the dma-buf behind it, which the NPU reads and writes in place.
struct TileMemory
{
    uint16_t* data = nullptr;
    int fd = -1;        // -1: not shared, the backend copies
    size_t bytes = 0;   // size of the dma-buf
};

/// Runs the two MLPs on NPU tiles, channel-planar fp16. The hash-grid encodings are already in the
/// inputs (the GPU wrote them).
class NetworkBackend
{
public:
    virtual ~NetworkBackend() = default;
    virtual const char* name() const = 0;
    /// Inference: x holds `tiles` tiles of kTile samples with C channels (32 for DiffNet, 64 for
    /// SpecNet), y receives `tiles` tiles of kOutputChannels channels.
    virtual void infer(bool specular, const TileMemory& x, const TileMemory& y, uint32_t tiles) = 0;
    /// The gradients of one training step (ncr_npu_graphs.Training), for `tiles` tiles of `tile`
    /// samples: x as for inference, target with kOutputChannels channels (radiance in 0-3, the
    /// sample weight in the last). `weightGradient` receives the sum over the tiles of dL/dw and
    /// `inputGradient` the tiles of dL/dx, each tile normalised by its own sample count: the caller
    /// rescales to the batch. Both carry the loss scale.
    virtual void train(bool specular, uint32_t tile, uint32_t tiles, const TileMemory& x, const TileMemory& target, float* weightGradient,
                       const TileMemory& inputGradient) = 0;
    /// The optimizer has changed the weights of a network.
    virtual void weightsChanged(bool specular) { (void)specular; }
};

class CpuNetworks
{
public:
    CpuNetworks(const Checkpoint& checkpoint, const Mlp& diff, const Mlp& spec) : mCheckpoint(checkpoint), mDiff(diff), mSpec(spec) {}

    /// The MLP of one network on tiles, as the NPU graph computes it (float32 here).
    void inferTiles(bool specular, const uint16_t* x, uint16_t* y, uint32_t tiles) const
    {
        const uint32_t channels = specular ? cfg::kSpecInputDim : cfg::kDiffInputDim;
        const float* weights = (specular ? mSpec : mDiff).inferenceFloat.data();
        const uint32_t hidden = specular ? 4 : 3;
        for (uint32_t t = 0; t < tiles; t++)
        {
            // Local copies: the tiles live in memory shared with the GPU, slow to walk element by element.
            std::vector<uint16_t> in(x + (size_t)t * channels * cfg::kTile, x + (size_t)(t + 1) * channels * cfg::kTile);
            std::vector<uint16_t> out((size_t)cfg::kOutputChannels * cfg::kTile, 0);
            parallel(cfg::kTile, [&](uint32_t begin, uint32_t end) {
                uint16_t row[64];
                float result[cfg::kOutputDim];
                for (uint32_t n = begin; n < end; n++)
                {
                    for (uint32_t c = 0; c < channels; c++) row[c] = in[(size_t)c * cfg::kTile + n];
                    forward(weights, channels, hidden, row, result);
                    for (uint32_t c = 0; c < cfg::kOutputDim; c++) out[(size_t)c * cfg::kTile + n] = floatToHalf(result[c]);
                }
            });
            std::memcpy(y + (size_t)t * cfg::kOutputChannels * cfg::kTile, out.data(), out.size() * sizeof(uint16_t));
        }
    }

    /// Rows of DiffNetInput (32 fp16 each): writes the primary encoding into the row, as the CUDA
    /// kernel does, and the four outputs to `out`.
    void inferDiff(uint16_t* rows, uint32_t count, float* out) const
    {
        parallel(count, [&](uint32_t begin, uint32_t end) {
            for (uint32_t n = begin; n < end; n++)
            {
                uint16_t* row = rows + (size_t)n * cfg::kDiffInputDim;
                encodePrimary(row, cfg::kDiffPos);
                forward(mDiff.inferenceFloat.data(), cfg::kDiffInputDim, 3, row, out + (size_t)n * cfg::kOutputDim);
            }
        });
    }

    /// Rows of SpecNetInput (64 fp16 each): primary and cluster encodings in place, outputs to `out`.
    void inferSpec(uint16_t* rows, uint32_t count, float* out) const
    {
        parallel(count, [&](uint32_t begin, uint32_t end) {
            for (uint32_t n = begin; n < end; n++)
            {
                uint16_t* row = rows + (size_t)n * cfg::kSpecInputDim;
                encodePrimary(row, cfg::kSpecPos);
                encodeClusters(row);
                forward(mSpec.inferenceFloat.data(), cfg::kSpecInputDim, 4, row, out + (size_t)n * cfg::kOutputDim);
            }
        });
    }

    /// The gradients of one training step on tiles, as the NPU's training graph computes them
    /// (ncr_npu_graphs.Training; float32 here, with the exact ReLU derivative). Outputs as
    /// NetworkBackend::train; either may be null. Returns the sum over the samples and the four
    /// outputs of the relative squared error: the loss before its normalisation.
    double trainTiles(bool specular, uint32_t tile, uint32_t tiles, const uint16_t* x, const uint16_t* target, float* weightGradient,
                      uint16_t* inputGradient) const
    {
        const Mlp& net = specular ? mSpec : mDiff;
        const uint32_t channels = net.inputDim, hidden = net.hiddenLayers;
        const float* weights = net.trainingFloat.data();
        const size_t paramCount = net.master.size();
        const float scale = 2.0f * adam::kLossScale / ((float)tile * cfg::kOutputDim);
        const uint32_t threads = threadCount(tile);
        double loss = 0.0;
        for (uint32_t t = 0; t < tiles; t++)
        {
            // Local copies: the tiles live in memory shared with the GPU, slow to walk element by element.
            const std::vector<uint16_t> in(x + (size_t)t * channels * tile, x + (size_t)(t + 1) * channels * tile);
            const std::vector<uint16_t> want(target + (size_t)t * cfg::kOutputChannels * tile, target + (size_t)(t + 1) * cfg::kOutputChannels * tile);
            std::vector<uint16_t> gx((size_t)channels * tile, 0);
            std::vector<std::vector<double>> sums(threads, std::vector<double>(paramCount + 1, 0.0));
            parallel(tile, [&](uint32_t begin, uint32_t end, uint32_t thread) {
                double* sum = sums[thread].data();
                float a[5][64], d[64], e[64];
                for (uint32_t n = begin; n < end; n++)
                {
                    if (halfToFloat(want[(size_t)(cfg::kOutputChannels - 1) * tile + n]) == 0.0f) continue;   // a slot beyond the batch
                    // Forward, keeping the input of every layer.
                    for (uint32_t c = 0; c < channels; c++) a[0][c] = halfToFloat(in[(size_t)c * tile + n]);
                    const float* w = weights;
                    uint32_t width = channels;
                    for (uint32_t layer = 0; layer < hidden; layer++)
                    {
                        for (uint32_t j = 0; j < 64; j++)
                        {
                            float z = 0.0f;
                            for (uint32_t i = 0; i < width; i++) z += w[(size_t)j * width + i] * a[layer][i];
                            a[layer + 1][j] = z > 0.0f ? z : 0.0f;
                        }
                        w += (size_t)64 * width;
                        width = 64;
                    }
                    float y[cfg::kOutputDim], slope[cfg::kOutputDim];
                    for (uint32_t j = 0; j < cfg::kOutputDim; j++)
                    {
                        float z = 0.0f;
                        for (uint32_t i = 0; i < 64; i++) z += w[(size_t)j * 64 + i] * a[hidden][i];
                        const float q = z * cfg::kSquarePlusK, root = std::sqrt(q * q + 4.0f);
                        y[j] = 0.5f * (q + root) / cfg::kSquarePlusK;
                        slope[j] = 0.5f + 0.5f * q / root;
                    }
                    // RelativeL2Luminance, the luminance of the prediction held constant.
                    const float luminance = 0.299f * y[0] + 0.587f * y[1] + 0.114f * y[2];
                    const float denominator = luminance * luminance + 0.01f;
                    float g[cfg::kOutputDim];
                    for (uint32_t j = 0; j < cfg::kOutputDim; j++)
                    {
                        const float difference = y[j] - halfToFloat(want[(size_t)j * tile + n]);
                        sum[paramCount] += (double)difference * difference / denominator;
                        g[j] = difference * scale / denominator * slope[j];
                    }
                    // Backward: the output layer, then the hidden layers down to the input.
                    size_t offset = paramCount - (size_t)cfg::kOutputChannels * 64;
                    for (uint32_t i = 0; i < 64; i++) d[i] = 0.0f;
                    for (uint32_t j = 0; j < cfg::kOutputDim; j++)
                        for (uint32_t i = 0; i < 64; i++)
                        {
                            sum[offset + (size_t)j * 64 + i] += (double)g[j] * a[hidden][i];
                            d[i] += weights[offset + (size_t)j * 64 + i] * g[j];
                        }
                    for (uint32_t layer = hidden; layer-- > 0;)
                    {
                        const uint32_t inputs = layer == 0 ? channels : 64;
                        offset -= (size_t)64 * inputs;
                        for (uint32_t i = 0; i < inputs; i++) e[i] = 0.0f;
                        for (uint32_t j = 0; j < 64; j++)
                        {
                            if (a[layer + 1][j] <= 0.0f || d[j] == 0.0f) continue;   // ReLU
                            for (uint32_t i = 0; i < inputs; i++)
                            {
                                sum[offset + (size_t)j * inputs + i] += (double)d[j] * a[layer][i];
                                e[i] += weights[offset + (size_t)j * inputs + i] * d[j];
                            }
                        }
                        for (uint32_t i = 0; i < inputs; i++) d[i] = e[i];
                    }
                    for (uint32_t c = 0; c < channels; c++) gx[(size_t)c * tile + n] = floatToHalf(d[c]);
                }
            });
            for (uint32_t thread = 0; thread < threads; thread++) loss += sums[thread][paramCount];
            if (weightGradient)
            {
                for (size_t i = 0; i < paramCount; i++)
                {
                    double total = 0.0;
                    for (uint32_t thread = 0; thread < threads; thread++) total += sums[thread][i];
                    weightGradient[i] += halfToFloat(floatToHalf((float)total));   // the NPU returns fp16
                }
            }
            if (inputGradient) std::memcpy(inputGradient + (size_t)t * channels * tile, gx.data(), gx.size() * sizeof(uint16_t));
        }
        return loss;
    }

private:
    const Checkpoint& mCheckpoint;
    const Mlp& mDiff;
    const Mlp& mSpec;

    static uint32_t threadCount(uint32_t count) { return std::max(1u, std::min(std::thread::hardware_concurrency(), count / 1024 + 1)); }

    /// work(begin, end) or work(begin, end, thread index) over [0, count) on several threads.
    template <typename F> static void parallel(uint32_t count, F&& work)
    {
        const uint32_t threads = threadCount(count);
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < threads; t++)
        {
            const uint32_t begin = (uint32_t)((uint64_t)count * t / threads), end = (uint32_t)((uint64_t)count * (t + 1) / threads);
            pool.emplace_back([&work, begin, end, t] { invoke(work, begin, end, t, 0); });
        }
        for (std::thread& t : pool) t.join();
    }
    template <typename F> static auto invoke(F& work, uint32_t begin, uint32_t end, uint32_t thread, int) -> decltype(work(begin, end, thread))
    {
        return work(begin, end, thread);
    }
    template <typename F> static auto invoke(F& work, uint32_t begin, uint32_t end, uint32_t, long) -> decltype(work(begin, end))
    {
        return work(begin, end);
    }

    void encodePrimary(uint16_t* row, uint32_t posOffset) const
    {
        const float pos[3] = {halfToFloat(row[posOffset]), halfToFloat(row[posOffset + 1]), halfToFloat(row[posOffset + 2])};
        for (uint32_t level = 0; level < cfg::kPrimLevels; level++)
        {
            const Corners c = corners(pos, gridLevel(cfg::kPrimBaseRes, level));
            for (uint32_t f = 0; f < cfg::kPrimFeatures; f++)
            {
                float value = 0.0f;
                for (int k = 0; k < 8; k++) value += halfToFloat(mCheckpoint.primGrid[c.entry[k] * cfg::kPrimFeatures + f]) * c.weight[k];
                row[level * cfg::kPrimFeatures + f] = floatToHalf(value);
            }
        }
    }

    void encodeClusters(uint16_t* row) const
    {
        for (uint32_t k = 0; k < cfg::kNumClusters; k++)
        {
            const float pos[3] = {halfToFloat(row[cfg::kSpecClsPos + k * 3]), halfToFloat(row[cfg::kSpecClsPos + k * 3 + 1]),
                                  halfToFloat(row[cfg::kSpecClsPos + k * 3 + 2])};
            const float scale = halfToFloat(row[cfg::kSpecClsScale + k]), weight = halfToFloat(row[cfg::kSpecClsWeight + k]);
            uint32_t upper, lower;
            float upperWeight, lowerWeight;
            clusterLevels(scale, upper, lower, upperWeight, lowerWeight);
            const Corners cu = corners(pos, gridLevel(cfg::kClsBaseRes, upper));
            const Corners cl = corners(pos, gridLevel(cfg::kClsBaseRes, lower));
            for (uint32_t f = 0; f < cfg::kClsFeatures; f++)
            {
                float u = 0.0f, l = 0.0f;
                for (int j = 0; j < 8; j++)
                {
                    u += halfToFloat(mCheckpoint.clsGrid[cu.entry[j] * cfg::kClsFeatures + f]) * cu.weight[j];
                    l += halfToFloat(mCheckpoint.clsGrid[cl.entry[j] * cfg::kClsFeatures + f]) * cl.weight[j];
                }
                row[cfg::kSpecClsEnc + k * cfg::kClsFeatures + f] = floatToHalf((l * lowerWeight + u * upperWeight) * weight);
            }
        }
    }

    /// Bias-free MLP: `hidden` layers of 64 with ReLU, then the first four rows of the 16-wide output
    /// layer through SquarePlus.
    static void forward(const float* weights, uint32_t inputDim, uint32_t hidden, const uint16_t* row, float* out)
    {
        float a[64], b[64];
        for (uint32_t i = 0; i < inputDim; i++) a[i] = halfToFloat(row[i]);
        uint32_t in = inputDim;
        float* x = a;
        float* y = b;
        for (uint32_t layer = 0; layer < hidden; layer++)
        {
            for (uint32_t j = 0; j < 64; j++)
            {
                const float* w = weights + (size_t)j * in;
                float sum = 0.0f;
                for (uint32_t i = 0; i < in; i++) sum += w[i] * x[i];
                y[j] = sum > 0.0f ? sum : 0.0f;
            }
            weights += (size_t)64 * in;
            in = 64;
            std::swap(x, y);
        }
        for (uint32_t j = 0; j < cfg::kOutputDim; j++)
        {
            const float* w = weights + (size_t)j * 64;
            float sum = 0.0f;
            for (uint32_t i = 0; i < 64; i++) sum += w[i] * x[i];
            const float q = sum * cfg::kSquarePlusK;
            out[j] = 0.5f * (q + std::sqrt(q * q + 4.0f)) / cfg::kSquarePlusK;
        }
    }
};
/// The NPU's stand-in where there is none: the same tiles in, the same tiles out, computed on the CPU.
class CpuTileBackend : public NetworkBackend
{
public:
    explicit CpuTileBackend(const CpuNetworks& networks) : mNetworks(networks) {}
    const char* name() const override { return "CPU (float32), tiles"; }
    void infer(bool specular, const TileMemory& x, const TileMemory& y, uint32_t tiles) override
    {
        mNetworks.inferTiles(specular, x.data, y.data, tiles);
    }
    void train(bool specular, uint32_t tile, uint32_t tiles, const TileMemory& x, const TileMemory& target, float* weightGradient,
               const TileMemory& inputGradient) override
    {
        mNetworks.trainTiles(specular, tile, tiles, x.data, target.data, weightGradient, inputGradient.data);
    }

private:
    const CpuNetworks& mNetworks;
};
}  // namespace replay
