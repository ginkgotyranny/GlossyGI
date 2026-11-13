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
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
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

/// The inference half of an NRCK v2 checkpoint (Model.cu): the fp16 EMA copies of the two grids and
/// the two MLPs, which follow the four float32 master blocks.
struct Checkpoint
{
    std::vector<uint16_t> primGrid, clsGrid;      // fp16, as the GPU reads them
    std::vector<float> diffWeights, specWeights;  // MLP weights, row-major (out, in) per layer
    std::vector<uint16_t> diffWeightsHalf, specWeightsHalf;   // the same as stored: what the NPU's weight tensors take

    void load(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        struct Header { char magic[4]; uint32_t version, counts[4]; } header = {};
        f.read(reinterpret_cast<char*>(&header), sizeof(header));
        const uint32_t expected[4] = {gridEntries(cfg::kPrimBaseRes, cfg::kPrimLevels) * cfg::kPrimFeatures,
                                      gridEntries(cfg::kClsBaseRes, cfg::kClsLevels) * cfg::kClsFeatures, 11264, 17408};
        if (!f || std::memcmp(header.magic, "NRCK", 4) != 0 || header.version != 2 || std::memcmp(header.counts, expected, sizeof(expected)) != 0)
        {
            std::printf("FAIL %s is not an NRCK v2 checkpoint of this network configuration\n", path.c_str());
            std::exit(2);
        }
        uint64_t masters = 0;
        for (uint32_t count : header.counts) masters += (uint64_t)count * 4;
        f.seekg((std::streamoff)(sizeof(header) + masters));
        primGrid.resize(expected[0]);
        clsGrid.resize(expected[1]);
        std::vector<uint16_t> diff(expected[2]), spec(expected[3]);
        f.read(reinterpret_cast<char*>(primGrid.data()), primGrid.size() * 2);
        f.read(reinterpret_cast<char*>(clsGrid.data()), clsGrid.size() * 2);
        f.read(reinterpret_cast<char*>(diff.data()), diff.size() * 2);
        f.read(reinterpret_cast<char*>(spec.data()), spec.size() * 2);
        if (!f)
        {
            std::printf("FAIL %s is truncated\n", path.c_str());
            std::exit(2);
        }
        diffWeightsHalf = diff;
        specWeightsHalf = spec;
        diffWeights.resize(diff.size());
        specWeights.resize(spec.size());
        for (size_t i = 0; i < diff.size(); i++) diffWeights[i] = halfToFloat(diff[i]);
        for (size_t i = 0; i < spec.size(); i++) specWeights[i] = halfToFloat(spec[i]);
    }
};

/// Runs the two MLPs on NPU tiles: x holds `tiles` tiles of C channels (32 for DiffNet, 64 for
/// SpecNet), y receives `tiles` tiles of kOutputChannels channels, both channel-planar fp16.
/// The hash-grid encodings are already in x (the GPU wrote them).
class NetworkBackend
{
public:
    virtual ~NetworkBackend() = default;
    virtual const char* name() const = 0;
    virtual void infer(bool specular, const uint16_t* x, uint16_t* y, uint32_t tiles) = 0;
};

class CpuNetworks
{
public:
    explicit CpuNetworks(const Checkpoint& checkpoint) : mCheckpoint(checkpoint) {}

    /// The MLP of one network on tiles, as the NPU graph computes it (float32 here).
    void inferTiles(bool specular, const uint16_t* x, uint16_t* y, uint32_t tiles) const
    {
        const uint32_t channels = specular ? cfg::kSpecInputDim : cfg::kDiffInputDim;
        const float* weights = specular ? mCheckpoint.specWeights.data() : mCheckpoint.diffWeights.data();
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
                forward(mCheckpoint.diffWeights.data(), cfg::kDiffInputDim, 3, row, out + (size_t)n * cfg::kOutputDim);
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
                forward(mCheckpoint.specWeights.data(), cfg::kSpecInputDim, 4, row, out + (size_t)n * cfg::kOutputDim);
            }
        });
    }

private:
    const Checkpoint& mCheckpoint;

    template <typename F> static void parallel(uint32_t count, F&& work)
    {
        const uint32_t threads = std::max(1u, std::min(std::thread::hardware_concurrency(), count / 1024 + 1));
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < threads; t++)
        {
            const uint32_t begin = (uint32_t)((uint64_t)count * t / threads), end = (uint32_t)((uint64_t)count * (t + 1) / threads);
            pool.emplace_back([&work, begin, end] { work(begin, end); });
        }
        for (std::thread& t : pool) t.join();
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
    explicit CpuTileBackend(const Checkpoint& checkpoint) : mNetworks(checkpoint) {}
    const char* name() const override { return "CPU (float32), tiles"; }
    void infer(bool specular, const uint16_t* x, uint16_t* y, uint32_t tiles) override { mNetworks.inferTiles(specular, x, y, tiles); }

private:
    CpuNetworks mNetworks;
};
}  // namespace replay
