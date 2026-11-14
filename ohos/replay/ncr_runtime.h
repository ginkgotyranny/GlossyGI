// The NCR runtime: runs the frames of a Falcor bake, rendering and training, without Falcor.
//
// The bake is a recorded frame of the NeuralRadiosity pass (ohos/tools/bake_pack.py): the compute-pass
// calls in the order Falcor executed them, each with its bindings and uniform values, and the steps
// Falcor did on the host in between (prefix sums, network inference, the training step). The runtime
// executes that program again and again as consecutive frames, with the same SPIR-V and scene data;
// the camera and the scene stay as baked, the frame counter advances.
//
// Shared by the command-line replayer (ncr_replay.cpp, Windows and the phone's shell) and the
// HarmonyOS app (ohos/app). Errors are thrown as std::runtime_error.
#pragma once
#include "replay_vk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace replay
{
struct RuntimeOptions
{
    /// Where the MLPs run: "tiles" (the phone's data path with the MLPs on the CPU), "npu" (OpenHarmony:
    /// the MLPs on the NPU through NNRt) or "rows" (the float32 reference, inference only).
    std::string networks = "tiles";
    std::string shaderDir = "shaders";   // the runtime's own SPIR-V (ohos/replay/shaders)
    std::string modelDir = "models";     // the NPU models (ohos/npu/export_models.py)
    uint32_t trainTile = 4096;           // samples per tile of the training graph
    float learningRate = 0.0f;           // > 0: fixed, instead of the training schedule
    float emaDecay = 0.0f;               // > 0: fixed, instead of the training schedule
    bool noTrain = false;                // skip the recorded training step
    bool copyTiles = false;              // npu: copy tiles to NNRt instead of sharing the memory
    bool cpuPrefixSum = false;           // the prefix sums on the CPU instead of a compute pass
    bool perPass = false;                // submit every call on its own (for its time)
    uint32_t lossInterval = 0;           // every N frames, print the loss of the first training tile
    std::string dumpTrainDir;            // non-empty: write the tensors of the first training step there
};

/// Falcor's online training schedule (NeuralRadiosity::updateTrainSchedule): a heat between 0 and 1
/// that sets the learning rate and the EMA decay. It holds after the events that raised it (the
/// warmup after loading, camera motion, scene changes), then relaxes toward the level of the current
/// events. The bake holds the settings and the state after the baked frame.
struct TrainSchedule
{
    static constexpr uint32_t kEventCamera = 1u << 0, kEventEdit = 1u << 1, kEventLight = 1u << 2, kEventObject = 1u << 3;

    bool enabled = true;
    float heat = 1.0f;
    uint32_t holdLeft = 0;
    float lrHot = 5e-3f, lrCold = 2.5e-3f, emaHot = 0.95f, emaCold = 0.99f;
    uint32_t holdFrames = 30;
    float coolFrames = 300.0f;
    float cameraHeat = 0.5f, sceneHeat = 1.0f, lightMotionHeat = 0.7f, objectMotionHeat = 0.2f;
    float learningRate = 5e-3f, emaDecay = 0.95f;   // the values of the current frame

    /// One frame's update, before its training step. `events`: kEvent* bits seen since the last frame.
    void advance(uint32_t events);
};

class Runtime
{
public:
    /// `model`: "bake" for the model as it was at the start of the baked frame (before.ckpt and the
    /// training copies in before_working.bin), or the path of a checkpoint, loaded the way Falcor's
    /// loadModelPath does (training copies rebuilt from the masters).
    Runtime(Device& dev, const std::string& bakeDir, const std::string& model, const RuntimeOptions& options);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    /// One frame: the recorded program once more, as the frame after the previous one.
    void frame();

    uint32_t framesDone() const;
    uint32_t width() const;
    uint32_t height() const;
    uint32_t bakedFrame() const;   // Falcor's frame count at the baked frame
    double lastFrameMs() const;
    const std::vector<std::pair<std::string, double>>& lastTimings() const;
    /// The final image of the last frame (what the pass writes to its gColor output): RGBA32F, in
    /// the GENERAL layout, readable as a storage image.
    const Texture& outputImage() const;
    bool trains() const;
    uint32_t trainSteps() const;
    const TrainSchedule& schedule() const;
    /// Every image and ray-batch buffer the passes wrote, as the bake holds Falcor's.
    void writeResults(const std::string& dir) const;
    /// The current weights as an NRCK checkpoint Falcor loads (loadModelPath).
    void saveCheckpoint(const std::string& path) const;
    /// Sample counts against Falcor's and the NPU's counters.
    void printSummary() const;

private:
    struct State;
    std::unique_ptr<State> s;
};

/// The pass's output as it is shown: Falcor's ToneMapper with the settings of the project's graphs
/// (no auto exposure, exposure compensation 0, ACES, clamp) and sRGB encoding, as 8-bit RGBA rows in a
/// buffer, ready to be copied into a swapchain image or saved (shaders/Display).
class Display
{
public:
    Display(const Device& dev, const std::string& shaderDir, const Texture& hdr, bool bgra);
    ~Display();
    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    /// Records the pass; a barrier makes the pixels ready for a copy or the host.
    void record(VkCommandBuffer cb) const;
    const Buffer& pixels() const;
    uint32_t width() const;
    uint32_t height() const;

private:
    struct State;
    std::unique_ptr<State> s;
};

/// Writes a file whole; throws on failure.
void writeFile(const std::string& path, const void* data, size_t size);
}  // namespace replay
