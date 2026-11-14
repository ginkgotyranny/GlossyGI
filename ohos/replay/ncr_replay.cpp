// NCR replayer, the command-line front end of the runtime (ncr_runtime.h): runs frames of a Falcor
// bake without Falcor and writes what the passes produced, for the comparison with Falcor.
//
//   ncr_replay <bake dir> <model> <output dir> [options]
//
// <model> is "bake" (the model as it was at the start of the baked frame: before.ckpt and the
// training copies in before_working.bin) or the path of a checkpoint (loaded as Falcor loads one).
//
// The output directory receives every image and ray-batch buffer the passes wrote in the last frame
// (ohos/tools/replay_compare.py compares them with Falcor's), display.bin (the final image as shown:
// tone mapped, sRGB, 8-bit RGBA rows) and, after training, trained.ckpt, which Falcor loads.
//
// The networks, --networks:
//   tiles  (default) the way the phone runs them: a GPU pass looks up the hash grids and lays the
//          samples out in NPU tiles (shaders/EncodeInputs), the MLPs run on those tiles, a GPU pass
//          writes the outputs back as rows (shaders/DecodeOutputs). The MLPs run on the CPU.
//   npu    the same with the MLPs on the NPU through NNRt (OpenHarmony only; --models is the folder
//          with the .om files from ohos/npu/export_models.py).
//   rows   the float32 reference on the CPU, lookups included, straight on the sample rows.
//          Inference only: a recorded training step is skipped.
//
// Options:
//   --frames N        replay the frame N times, as consecutive frames (default 1). The camera and the
//                     scene stay as baked; the frame counter advances, so every frame draws new samples,
//                     and Falcor's training schedule continues from the baked frame.
//   --train-tile N    samples per tile of the training graph (default 4096; the NPU needs
//                     ncr_<net>_train_<N>_fp16.om)
//   --lr X, --ema X   fixed learning rate and EMA decay instead of the training schedule
//   --no-train        skip recorded training steps
//   --loss N          every N frames print the training loss of the first tile, computed on the CPU
//   --dump-train      write the tensors of the first training step for ohos/tools/train_check.py
//   --copy-tiles      with npu: keep the tiles in ordinary memory and copy them to and from the NPU,
//                     instead of sharing one buffer between Vulkan and NNRt
//   --cpu-prefix-sum  do the prefix sums on the CPU instead of in a compute pass (shaders/PrefixSum)
//   --shaders <dir>   the runtime's own SPIR-V (default: shaders next to the executable)
//   --models <dir>    the NPU models
//   --per-pass        submit every call on its own and print the time of each (slower overall)
//
// The same source builds for Windows (build_pc.bat) and for OpenHarmony (run_device.sh).
#include "ncr_runtime.h"

#include <exception>

using namespace replay;

namespace
{
int run(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: ncr_replay <bake dir> <bake|checkpoint.ckpt> <output dir> [--networks rows|tiles|npu] [--frames N] [--train-tile N]\n"
                    "                  [--lr X] [--ema X] [--no-train] [--loss N] [--dump-train] [--copy-tiles] [--cpu-prefix-sum]\n"
                    "                  [--shaders <dir>] [--models <dir>] [--per-pass]\n");
        return 2;
    }
    const std::string bakeDir = argv[1], model = argv[2], outDir = argv[3];
    const std::string executable = argv[0];
    const size_t slash = executable.find_last_of("/\\");
    RuntimeOptions options;
    options.shaderDir = (slash == std::string::npos ? std::string(".") : executable.substr(0, slash)) + "/shaders";
    options.modelDir = ".";
    uint32_t frames = 1;
    for (int i = 4; i < argc; i++)
    {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--per-pass") options.perPass = true;
        else if (arg == "--no-train") options.noTrain = true;
        else if (arg == "--dump-train") options.dumpTrainDir = outDir;
        else if (arg == "--copy-tiles") options.copyTiles = true;
        else if (arg == "--cpu-prefix-sum") options.cpuPrefixSum = true;
        else if (arg == "--networks" && hasValue) options.networks = argv[++i];
        else if (arg == "--shaders" && hasValue) options.shaderDir = argv[++i];
        else if (arg == "--models" && hasValue) options.modelDir = argv[++i];
        else if (arg == "--frames" && hasValue) frames = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--train-tile" && hasValue) options.trainTile = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--loss" && hasValue) options.lossInterval = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--lr" && hasValue) options.learningRate = std::stof(argv[++i]);
        else if (arg == "--ema" && hasValue) options.emaDecay = std::stof(argv[++i]);
        else fail("unknown option " + arg);
    }

    Device dev;
    dev.create();
    std::printf("device: %s\n", dev.name);
    Runtime runtime(dev, bakeDir, model, options);
    Display display(dev, options.shaderDir, runtime.outputImage(), false);

    std::vector<double> frameMs;
    for (uint32_t frame = 0; frame < frames; frame++)
    {
        runtime.frame();
        frameMs.push_back(runtime.lastFrameMs());
    }
    for (const auto& [name, ms] : runtime.lastTimings()) std::printf("  %8.1f ms  %s\n", ms, name.c_str());
    std::printf("frame: %.1f ms in %zu segments\n", frameMs.back(), runtime.lastTimings().size());
    if (frames > 1)
    {
        double sum = 0.0;
        for (size_t i = 1; i < frameMs.size(); i++) sum += frameMs[i];
        std::printf("frames: %u, the first %.1f ms, the others %.1f ms on average\n", frames, frameMs[0], sum / (double)(frameMs.size() - 1));
    }
    runtime.printSummary();

    // Everything the passes wrote, for the comparison with Falcor, and the image as it is shown.
    runtime.writeResults(outDir);
    VkCommandBuffer cb = dev.begin();
    display.record(cb);
    dev.submit(cb);
    writeFile(outDir + "/display.bin", display.pixels().data, (size_t)display.width() * display.height() * 4);
    if (runtime.trainSteps() > 0)
    {
        runtime.saveCheckpoint(outDir + "/trained.ckpt");
        std::printf("training: weights written to %s/trained.ckpt\n", outDir.c_str());
        // The training schedule's state after the last frame (ohos/tools/train_compare.py).
        const TrainSchedule& schedule = runtime.schedule();
        const std::string text = format("frames %u\nheat %.9g\nholdLeft %u\nlearningRate %.9g\nemaDecay %.9g\n", runtime.framesDone(), schedule.heat,
                                        schedule.holdLeft, schedule.learningRate, schedule.emaDecay);
        writeFile(outDir + "/schedule.txt", text.data(), text.size());
    }
    vkDeviceWaitIdle(dev.device);
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: MSVC rejects line buffering with a null buffer
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %s\n", e.what());
        return 2;
    }
}
