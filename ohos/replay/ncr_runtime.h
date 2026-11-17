// The NCR runtime: runs the frames of a Falcor bake, rendering and training, without Falcor.
//
// The bake is a recorded frame of the NeuralRadiosity pass (ohos/tools/bake_pack.py): the compute-pass
// calls in the order Falcor executed them, each with its bindings and uniform values, and the steps
// Falcor did on the host in between (prefix sums, network inference, the training step). The runtime
// executes that program again and again as consecutive frames, with the same SPIR-V and scene data;
// the frame counter advances.
//
// Between frames the scene can change as it does on the desktop: an animated scene's bake carries a
// track (the scene's data over one loop of its animation), which is played back by time; the camera
// follows the track or is set by the caller; and the switches of the pass's panel (Settings) set the
// same uniforms the pass sets. The training schedule sees the same event classes as in Falcor.
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
    int64_t frameCount = -1;             // >= 0: Falcor's frame count for the first frame, instead of the baked frame's
};

/// NeuralRadiosity::RenderMode, with its values.
enum class RenderMode : uint32_t { Idle = 0, Render = 1, Train = 2, OnlineTrain = 3 };

/// What the desktop sets in the pass's panel (NeuralRadiosity::renderUI) and in Mogwai's scene and
/// time controls, as far as the runtime has it. The values start as they were in the baked frame, so
/// an untouched runtime replays the bake. They take effect in the next frame.
struct Settings
{
    /// Online Train trains and renders, Render only renders, Train only trains (with the baked
    /// shaders: the desktop's Train mode compiles ConeTrace for more rays, which a bake made in
    /// Online Train does not hold), Idle does neither.
    RenderMode mode = RenderMode::OnlineTrain;
    bool useNEE = true;               // "Use NEE"
    uint32_t rhsGlossyBounces = 2;    // "RHS Glossy Bounces"
    bool diffTrainOnGlossy = false;   // "Diffuse Trains On Glossy"
    bool mirrorAlbedo = true;         // "Mirror Albedo"
    uint32_t envColorEncoding = 1;    // "Env Color Encoding": 0 linear, 1 signed log, 2 log
    bool postFilter = true;           // "Post Filter"
    bool fxaa = true;                 // "FXAA"
    bool animate = true;              // the scene's animation plays (Mogwai's clock)
    bool cameraAnimated = true;       // the camera follows its path (the camera's "Animated" box); off: setCamera()
    float timeScale = 1.0f;           // animation seconds per second
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
    uint32_t warmupFrames = 600, holdFrames = 30;
    float coolFrames = 300.0f;
    float cameraHeat = 0.3f, sceneHeat = 1.0f, lightMotionHeat = 0.5f, objectMotionHeat = 0.1f;
    float learningRate = 5e-3f, emaDecay = 0.95f;   // the values of the current frame

    /// One frame's update, before its training step. `events`: kEvent* bits seen since the last frame.
    void advance(uint32_t events);
    /// NeuralRadiosity::resetTrainSchedule ("Reset schedule"): fully hot for the warmup.
    void reset();
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

    /// One frame: the recorded program once more, as the frame after the previous one. `seconds` is
    /// the time since the previous frame: an animated scene's animation advances by it (times
    /// Settings::timeScale, while Settings::animate is on).
    void frame(double seconds = 0.0);

    uint32_t framesDone() const;
    uint32_t width() const;
    uint32_t height() const;
    uint32_t bakedFrame() const;   // Falcor's frame count at the baked frame
    double lastFrameMs() const;
    const std::vector<std::pair<std::string, double>>& lastTimings() const;
    /// The final image of the last frame (what the pass writes to its gColor output): RGBA32F, in
    /// the GENERAL layout, readable as a storage image.
    const Texture& outputImage() const;
    bool trains() const;   // the bake has a training step and the networks can train
    uint32_t trainSteps() const;

    Settings& settings();
    const Settings& settings() const;
    bool hasPostFilter() const;   // the baked frame ran the post filter, so it can be switched
    bool hasLights() const;       // the scene has lights to sample, so NEE can be switched
    bool hasRenderPath() const;   // the baked frame rendered (it was not baked in Train mode)
    /// A setting or a training-schedule setting by name, as a number (booleans 0 or 1): what the
    /// command line and the app's panel go through. settingNames() lists them; set() is false for an
    /// unknown name, get() fails for one.
    static const std::vector<std::string>& settingNames();
    bool set(const std::string& name, double value);
    double get(const std::string& name) const;

    TrainSchedule& schedule();
    const TrainSchedule& schedule() const;
    /// The event classes (TrainSchedule::kEvent*) the last frame saw.
    uint32_t lastEvents() const;

    /// An animated scene: the bake has a track. The time is within one loop of the animation.
    bool animated() const;
    double animationLength() const;
    double animationTime() const;
    void setAnimationTime(double seconds);
    uint32_t trackSamples() const;
    double trackRate() const;
    uint32_t trackSample() const;   // the sample the scene is at

    /// A number the bake recorded about the baked frame (replay.txt's value records), e.g. lhs.size
    /// (the training batch), numRHS, numSpecRays.
    double bakeValue(const std::string& name, double fallback = 0.0) const;
    /// The scene's bounding box (min, max), as the passes get it; false when the bake does not say.
    bool sceneBounds(float minMax[6]) const;

    /// The camera of the last frame, and a camera for the next frames. A set camera is used while
    /// the camera does not follow the track (Settings::cameraAnimated off, or a static scene).
    CameraPose camera() const;
    void setCamera(const CameraPose& pose);

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
