// A packed Falcor bake as live Vulkan objects: see ohos/tools/bake_pack.py for the index format.
//
// The bake is a recorded frame: an ordered list of compute-pass calls, each with the resources it
// had bound, and of the steps Falcor did on the host in between. A pass is a pipeline; every call of
// it has its own descriptor sets and its own copy of the pass's uniform block.
//
// A bake of the cluster-reuse render path (enableTemporalSpecReuse) has state that lives across
// frames: two history buffers and two lock-state buffers used as ping-pong pairs, and copies of the
// previous frame's G-buffer. The bake names them (cluster.*). The calls that bind a pair get a
// second group of descriptor sets with the pair swapped, for every other frame.
//
// The scene can change between frames in two ways. An animated scene comes with a track: what its
// data is at evenly spaced times over one loop of the animation, which applyTrackSample() writes
// into the scene's buffers, its uniform block and the top-level acceleration structure. And the
// camera can be set (setCamera), computed the way Falcor's Camera computes its shader data.
#pragma once
#include "replay_vk.h"

namespace replay
{
struct Binding
{
    uint32_t set = 0, binding = 0, count = 1;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    std::string name;   // shader path, e.g. gScene.materials.materialTextures
};

struct Element
{
    uint32_t set = 0, binding = 0, index = 0;
    std::string kind;   // buffer, texture, sampler, uniform, accel, null
    uint32_t id = 0;    // buffer / texture / sampler id; byte size of a uniform block
    std::string file;   // uniform block contents
};

struct Geometry
{
    uint32_t id, tlasInstance, geometryIndex, vbOffset, ibOffset, vertexCount, indexCount, index16, frontFaceCW, opaque;
    float transform[12];
};

struct Pass
{
    std::string name;
    uint32_t group[3] = {1, 1, 1};
    std::vector<Binding> bindings;
    std::map<std::string, std::map<std::string, std::pair<uint32_t, uint32_t>>> uniforms;   // block -> member -> (offset, size)
    std::vector<VkDescriptorSetLayout> setLayouts;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    double createMs = 0.0;
};

/// One execution of a pass in the recorded frame.
struct Call
{
    uint32_t step = 0;
    std::string pass;
    uint32_t dims[3] = {1, 1, 1};                    // as Falcor dispatched it; group counts are dims / group size, rounded up
    std::string batch;                               // render, lhs, rhs or -: the ray batch whose buffers it has bound
    std::vector<Element> elements;
    std::vector<VkDescriptorSet> sets;
    std::vector<VkDescriptorSet> altSets;            // the same with the ping-pong pairs swapped; empty if it binds none
    std::map<std::string, Buffer*> uniformBlocks;    // block path -> its buffer
};

/// One entry of the recorded frame: a call, or a host step with its arguments.
struct Step
{
    uint32_t step = 0;
    bool isCall = false;
    std::vector<std::string> host;   // e.g. {"prefixsum", "render", "diff", "230400"}
};

/// A part of the scene's data that an animated scene changes.
struct TrackRange
{
    std::string kind;   // uniform (the scene's block), buffer, geometry (3x4 transforms, 48 bytes per global instance)
    uint32_t id = 0, offset = 0, bytes = 0, recordOffset = 0;
};

/// The track of an animated scene (BakeExport.cpp): one record per sample.
struct Track
{
    static constexpr uint32_t kHeaderBytes = 28;   // of a record: events (uint32), scene bounds (6 floats)
    uint32_t samples = 0, recordBytes = 0;
    double rate = 0.0, loopSeconds = 0.0;
    std::vector<TrackRange> ranges;
    std::vector<char> data;
    bool movesGeometry = false;

    const char* record(uint32_t sample) const { return data.data() + (size_t)sample * recordBytes; }
    /// The training schedule's event classes Falcor saw in the frame of that sample.
    uint32_t events(uint32_t sample) const
    {
        uint32_t e = 0;
        std::memcpy(&e, record(sample), sizeof(e));
        return e;
    }
    void bounds(uint32_t sample, float minMax[6]) const { std::memcpy(minMax, record(sample) + 4, 6 * sizeof(float)); }
};

class Bake
{
public:
    Device& dev;
    std::string dir;
    uint32_t width = 0, height = 0, frameCount = 0;
    std::map<uint32_t, Buffer> buffers;
    std::map<uint32_t, bool> bufferLoaded;      // false: a ray-batch buffer the passes write
    std::map<uint32_t, Texture> textures;
    std::map<uint32_t, VkSampler> samplers;
    std::vector<Geometry> geometry;
    uint32_t vertexBuffer = UINT32_MAX, vertexStride = 0, indexBuffer = UINT32_MAX;
    std::vector<std::string> passOrder;
    std::map<std::string, Pass> passes;
    std::map<uint32_t, Call> calls;             // by step
    std::vector<Step> program;                  // the frame, in execution order
    std::map<std::string, uint32_t> named;      // role, e.g. render.specInput -> buffer id
    std::map<std::string, uint32_t> namedTextures;   // role, e.g. cluster.prevPos -> texture id
    std::map<uint32_t, bool> textureKept;       // a texture the runtime keeps (starts cleared, nothing to load)
    std::map<uint32_t, uint32_t> bufferSwap;    // ping-pong pairs: each buffer id -> its partner
    std::map<std::string, double> values;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    Track track;
    std::vector<float> geometryTransforms;      // 12 floats per global geometry instance id

    Bake(Device& device, const std::string& folder) : dev(device), dir(folder) {}
    Bake(const Bake&) = delete;
    Bake& operator=(const Bake&) = delete;

    ~Bake()
    {
        if (!dev.device) return;
        vkDeviceWaitIdle(dev.device);
        for (auto& [name, p] : passes)
        {
            if (p.pipeline) vkDestroyPipeline(dev.device, p.pipeline, nullptr);
            if (p.layout) vkDestroyPipelineLayout(dev.device, p.layout, nullptr);
            for (VkDescriptorSetLayout l : p.setLayouts) vkDestroyDescriptorSetLayout(dev.device, l, nullptr);
        }
        if (mPool) vkDestroyDescriptorPool(dev.device, mPool, nullptr);
        if (tlas) dev.destroyAS(dev.device, tlas, nullptr);
        for (VkAccelerationStructureKHR blas : mBlas) dev.destroyAS(dev.device, blas, nullptr);
        for (Buffer& b : mAccelerationStorage) dev.destroy(b);
        dev.destroy(mTlasInstanceBuffer);
        dev.destroy(mTlasScratch);
        for (auto& [id, b] : buffers) dev.destroy(b);
        for (auto& [id, t] : textures) dev.destroy(t);
        for (auto& [id, sampler] : samplers) vkDestroySampler(dev.device, sampler, nullptr);
        for (auto& [key, b] : mUniformBuffers) dev.destroy(b);
        dev.destroy(mDummyBuffer);
        dev.destroy(mDummyTexture);
        if (mDummySampler) vkDestroySampler(dev.device, mDummySampler, nullptr);
    }

    void load()
    {
        parseIndex();
        for (const char* pair : {"cluster.history", "cluster.lockState", "cluster.vcdRecovery"})
        {
            const std::string a = std::string(pair) + "0", b = std::string(pair) + "1";
            if (named.count(a) == 0 || named.count(b) == 0) continue;
            bufferSwap[named[a]] = named[b];
            bufferSwap[named[b]] = named[a];
        }
        createResources();
        buildAccelerationStructures();
        createPasses();
        loadTrack();
    }

    /// The scene as it is at one sample of the track: its buffers, its uniform block, the instance
    /// transforms. The GPU must be idle.
    void applyTrackSample(uint32_t sample)
    {
        const char* record = track.record(sample);
        for (const TrackRange& r : track.ranges)
        {
            char* target = r.kind == "uniform" ? sceneBlock().as<char>()
                         : r.kind == "buffer" ? buffers.at(r.id).as<char>()
                                              : reinterpret_cast<char*>(geometryTransforms.data());
            std::memcpy(target + r.offset, record + r.recordOffset, r.bytes);
        }
        if (track.movesGeometry) buildTlas();
    }

    /// The uniform block of the scene (gScene), one buffer for every call.
    Buffer& sceneBlock()
    {
        auto it = mUniformBuffers.find("gScene");
        if (it == mUniformBuffers.end()) fail("the bake has no scene uniform block (gScene)");
        return it->second;
    }

    /// A member of the scene's uniform block, e.g. camera.data.posW; null when the block has none of that name and size.
    void* sceneMember(const std::string& member, uint32_t size)
    {
        if (!mSceneMembers)
        {
            for (auto& [name, p] : passes)
            {
                auto block = p.uniforms.find("gScene");
                if (block != p.uniforms.end()) mSceneMembers = &block->second;
            }
            if (!mSceneMembers) fail("the bake does not describe the scene's uniform block");
        }
        auto m = mSceneMembers->find(member);
        if (m == mSceneMembers->end() || m->second.second < size) return nullptr;
        return sceneBlock().as<char>() + m->second.first;
    }

    CameraPose camera()
    {
        CameraPose pose;
        std::memcpy(pose.position, cameraMember("posW", 12), 12);
        std::memcpy(pose.target, cameraMember("target", 12), 12);
        std::memcpy(pose.up, cameraMember("up", 12), 12);
        return pose;
    }

    /// Falcor's Camera::calculateCameraParameters for a new pose: the lens stays as baked. What was
    /// current becomes the previous frame's (prevViewMat, prevViewProjMatNoJitter, prevPosW).
    void setCamera(const CameraPose& pose)
    {
        auto get = [&](const char* name) {
            float value = 0.0f;
            std::memcpy(&value, cameraMember(name, 4), 4);
            return (double)value;
        };
        std::memcpy(cameraMember("prevViewMat", 64), cameraMember("viewMat", 64), 64);
        std::memcpy(cameraMember("prevViewProjMatNoJitter", 64), cameraMember("viewProjMatNoJitter", 64), 64);
        std::memcpy(cameraMember("prevPosW", 12), cameraMember("posW", 12), 12);

        const double focalLength = get("focalLength"), aspect = get("aspectRatio"), nearZ = get("nearZ"), farZ = get("farZ");
        const double frameHeight = get("frameHeight"), focalDistance = get("focalDistance"), jitterX = get("jitterX"), jitterY = get("jitterY");
        double eye[3], center[3], up[3];
        for (int i = 0; i < 3; i++)
        {
            eye[i] = pose.position[i];
            center[i] = pose.target[i];
            up[i] = pose.up[i];
        }
        auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
        auto cross = [](const double* a, const double* b, double* out) {
            out[0] = a[1] * b[2] - a[2] * b[1];
            out[1] = a[2] * b[0] - a[0] * b[2];
            out[2] = a[0] * b[1] - a[1] * b[0];
        };
        auto normalize = [&](double* v) {
            const double length = std::sqrt(dot(v, v));
            for (int i = 0; i < 3; i++) v[i] /= length;
        };
        // math::matrixFromLookAt, right-handed; matrices are stored row by row.
        double f[3] = {eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]}, r[3], u[3];
        normalize(f);
        cross(up, f, r);
        normalize(r);
        cross(f, r, u);
        double view[16] = {r[0], r[1], r[2], -dot(r, eye), u[0], u[1], u[2], -dot(u, eye), f[0], f[1], f[2], -dot(f, eye), 0, 0, 0, 1};
        // math::perspective; a focal length of 0 would be an orthographic camera, which no scene here has.
        const double fovY = 2.0 * std::atan(0.5 * frameHeight / focalLength);
        const double tanHalf = std::tan(fovY * 0.5);
        double proj[16] = {};
        proj[0] = 1.0 / (aspect * tanHalf);
        proj[5] = 1.0 / tanHalf;
        proj[10] = farZ / (nearZ - farZ);
        proj[14] = -1.0;
        proj[11] = -(farZ * nearZ) / (farZ - nearZ);
        double jitter[16] = {1, 0, 0, 2.0 * jitterX, 0, 1, 0, 2.0 * jitterY, 0, 0, 1, 0, 0, 0, 0, 1};
        double viewProjNoJitter[16], projJittered[16], viewProj[16], inverse[16];
        multiply(proj, view, viewProjNoJitter);
        multiply(jitter, proj, projJittered);
        multiply(projJittered, view, viewProj);
        invert(viewProj, inverse);
        writeMatrix("viewMat", view);
        writeMatrix("projMat", projJittered);
        writeMatrix("viewProjMat", viewProj);
        writeMatrix("invViewProj", inverse);
        writeMatrix("viewProjMatNoJitter", viewProjNoJitter);
        writeMatrix("projMatNoJitter", proj);
        std::memcpy(cameraMember("posW", 12), pose.position, 12);
        std::memcpy(cameraMember("target", 12), pose.target, 12);
        std::memcpy(cameraMember("up", 12), pose.up, 12);
        // The vectors the primary rays are built from.
        double w[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]}, cu[3], cv[3];
        normalize(w);
        for (double& x : w) x *= focalDistance;
        cross(w, up, cu);
        normalize(cu);
        cross(cu, w, cv);
        normalize(cv);
        for (int i = 0; i < 3; i++)
        {
            cu[i] *= focalDistance * tanHalf * aspect;
            cv[i] *= focalDistance * tanHalf;
        }
        writeVector("cameraU", cu);
        writeVector("cameraV", cv);
        writeVector("cameraW", w);
    }

    /// What a frame leaves for the next one: the camera's matrices, which become the "previous
    /// frame" members of the camera's shader data (reprojection reads prevViewProjMatNoJitter).
    struct CameraHistory
    {
        float view[16] = {}, viewProjNoJitter[16] = {}, position[3] = {};
    };

    CameraHistory cameraHistory()
    {
        CameraHistory h;
        std::memcpy(h.view, cameraMember("viewMat", 64), 64);
        std::memcpy(h.viewProjNoJitter, cameraMember("viewProjMatNoJitter", 64), 64);
        std::memcpy(h.position, cameraMember("posW", 12), 12);
        return h;
    }

    void setCameraHistory(const CameraHistory& h)
    {
        std::memcpy(cameraMember("prevViewMat", 64), h.view, 64);
        std::memcpy(cameraMember("prevViewProjMatNoJitter", 64), h.viewProjNoJitter, 64);
        std::memcpy(cameraMember("prevPosW", 12), h.position, 12);
    }

    /// The camera differs from the previous frame's (NeuralRadiosity::execute compares the same two
    /// matrices of the camera's shader data).
    bool viewChanged() { return std::memcmp(cameraMember("viewProjMatNoJitter", 64), cameraMember("prevViewProjMatNoJitter", 64), 64) != 0; }

    bool hasBuffer(const std::string& role) const { return named.count(role) != 0; }

    Texture& texture(const std::string& role)
    {
        auto it = namedTextures.find(role);
        if (it == namedTextures.end()) fail("the bake has no texture named " + role);
        return textures.at(it->second);
    }

    Buffer& buffer(const std::string& role)
    {
        auto it = named.find(role);
        if (it == named.end()) fail("the bake has no buffer named " + role);
        return buffers.at(it->second);
    }

    /// The texture the recorded frame has bound at a shader path, e.g. gColor, the pass's output.
    uint32_t textureBoundAs(const std::string& path) const
    {
        for (const auto& [step, c] : calls)
        {
            const Pass& p = passes.at(c.pass);
            for (const Element& e : c.elements)
            {
                if (e.kind != "texture") continue;
                for (const Binding& b : p.bindings)
                    if (b.set == e.set && b.binding == e.binding && b.name == path) return e.id;
            }
        }
        fail("the bake has no texture bound as " + path);
    }

    double value(const std::string& name, double fallback = 0.0) const
    {
        auto it = values.find(name);
        return it == values.end() ? fallback : it->second;
    }

    /// Location of a uniform member of a call, or null.
    void* uniform(Call& c, const std::string& block, const std::string& member, uint32_t size)
    {
        Pass& p = passes.at(c.pass);
        auto b = p.uniforms.find(block);
        if (b == p.uniforms.end()) return nullptr;
        auto m = b->second.find(member);
        auto buffer = c.uniformBlocks.find(block);
        if (m == b->second.end() || buffer == c.uniformBlocks.end() || m->second.second < size) return nullptr;
        return buffer->second->as<char>() + m->second.first;
    }

    bool hasUniform(Call& c, const std::string& block, const std::string& member) { return uniform(c, block, member, 1) != nullptr; }

    template <typename T> T getUniform(Call& c, const std::string& block, const std::string& member)
    {
        T value = {};
        void* at = uniform(c, block, member, sizeof(T));
        if (!at) fail(c.pass + " has no uniform " + block + "." + member);
        std::memcpy(&value, at, sizeof(T));
        return value;
    }

    template <typename T> void setUniform(Call& c, const std::string& block, const std::string& member, const T& value)
    {
        void* at = uniform(c, block, member, sizeof(T));
        if (!at) fail(c.pass + " has no uniform " + block + "." + member);
        std::memcpy(at, &value, sizeof(T));
    }

    /// Records one dispatch of a call over `dims` items (Falcor's ComputePass::execute convention:
    /// the group counts are the dimensions divided by the thread group size, rounded up), followed
    /// by a barrier so that the next pass sees what this one wrote. `alternate`: with the ping-pong
    /// pairs swapped against the baked frame (a call that binds none has only one group of sets).
    void dispatch(VkCommandBuffer cb, Call& c, uint32_t dimX, uint32_t dimY = 1, uint32_t dimZ = 1, bool alternate = false)
    {
        Pass& p = passes.at(c.pass);
        const uint32_t groups[3] = {(dimX + p.group[0] - 1) / p.group[0], (dimY + p.group[1] - 1) / p.group[1], (dimZ + p.group[2] - 1) / p.group[2]};
        for (int a = 0; a < 3; a++)
        {
            // A driver runs only as many groups as it supports and drops the rest without a word.
            if (groups[a] > dev.maxGroupCount[a])
                fail(format("%s: %u thread groups in dimension %d, the device runs at most %u", p.name.c_str(), groups[a], a, dev.maxGroupCount[a]));
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        const std::vector<VkDescriptorSet>& sets = alternate && !c.altSets.empty() ? c.altSets : c.sets;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, (uint32_t)sets.size(), sets.data(), 0, nullptr);
        vkCmdDispatch(cb, groups[0], groups[1], groups[2]);
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

private:
    Buffer mDummyBuffer;
    Texture mDummyTexture;
    VkSampler mDummySampler = VK_NULL_HANDLE;
    std::map<std::string, Buffer> mUniformBuffers;   // key: see uniformKey()
    std::vector<Buffer> mAccelerationStorage;
    std::vector<VkAccelerationStructureKHR> mBlas;
    // The top-level structure is built again when instances move: its instances, the global id of
    // each one's first geometry (whose transform is the instance's), and the build's buffers.
    std::vector<VkAccelerationStructureInstanceKHR> mTlasInstances;
    std::vector<uint32_t> mTlasGeometry;
    Buffer mTlasInstanceBuffer, mTlasScratch;
    VkDescriptorPool mPool = VK_NULL_HANDLE;
    std::map<std::string, std::pair<uint32_t, uint32_t>>* mSceneMembers = nullptr;

    void* cameraMember(const char* name, uint32_t size)
    {
        void* at = sceneMember(std::string("camera.data.") + name, size);
        if (!at) fail(std::string("the scene's uniform block has no camera.data.") + name);
        return at;
    }

    void writeMatrix(const char* name, const double* m)
    {
        float values[16];
        for (int i = 0; i < 16; i++) values[i] = (float)m[i];
        std::memcpy(cameraMember(name, 64), values, 64);
    }

    void writeVector(const char* name, const double* v)
    {
        const float values[3] = {(float)v[0], (float)v[1], (float)v[2]};
        std::memcpy(cameraMember(name, 12), values, 12);
    }

    static void multiply(const double* a, const double* b, double* out)
    {
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                double sum = 0.0;
                for (int k = 0; k < 4; k++) sum += a[row * 4 + k] * b[k * 4 + col];
                out[row * 4 + col] = sum;
            }
    }

    /// Inverse of a 4x4 matrix by Gauss-Jordan elimination with partial pivoting.
    static void invert(const double* m, double* out)
    {
        double a[4][8];
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                a[row][col] = m[row * 4 + col];
                a[row][col + 4] = row == col ? 1.0 : 0.0;
            }
        for (int col = 0; col < 4; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < 4; row++)
                if (std::fabs(a[row][col]) > std::fabs(a[pivot][col])) pivot = row;
            if (a[pivot][col] == 0.0) fail("the camera's view-projection matrix is singular");
            for (int k = 0; k < 8; k++) std::swap(a[col][k], a[pivot][k]);
            const double scale = 1.0 / a[col][col];
            for (int k = 0; k < 8; k++) a[col][k] *= scale;
            for (int row = 0; row < 4; row++)
            {
                if (row == col) continue;
                const double factor = a[row][col];
                for (int k = 0; k < 8; k++) a[row][k] -= factor * a[col][k];
            }
        }
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++) out[row * 4 + col] = a[row][col + 4];
    }

    void loadTrack()
    {
        if (track.samples == 0) return;
        uint32_t at = Track::kHeaderBytes;
        for (TrackRange& r : track.ranges)
        {
            r.recordOffset = at;
            at += r.bytes;
            const size_t size = r.kind == "uniform" ? (size_t)sceneBlock().size
                              : r.kind == "buffer" ? (buffers.count(r.id) ? (size_t)buffers.at(r.id).size : 0)
                              : r.kind == "geometry" ? geometryTransforms.size() * sizeof(float) : 0;
            if ((size_t)r.offset + r.bytes > size) fail(format("the track writes beyond its target (%s %u, offset %u, %u bytes)", r.kind.c_str(), r.id, r.offset, r.bytes));
            track.movesGeometry = track.movesGeometry || r.kind == "geometry";
        }
        const std::vector<char> file = readFile(dir + "/track.bin");
        if (at != track.recordBytes || file.size() != 20 + (size_t)track.samples * track.recordBytes || std::memcmp(file.data(), "NRTK", 4) != 0)
            fail("track.bin is not what replay.txt describes");
        track.data.assign(file.begin() + 20, file.end());
        std::printf("track: %u samples at %g per second (%g s), %zu ranges, %u bytes per sample%s\n", track.samples, track.rate, track.loopSeconds,
                    track.ranges.size(), track.recordBytes, track.movesGeometry ? ", instances move" : "");
    }

    void parseIndex()
    {
        std::ifstream index(dir + "/replay.txt");
        if (!index) fail("cannot read " + dir + "/replay.txt (run ohos/tools/bake_pack.py on the bake)");
        std::string line;
        while (std::getline(index, line))
        {
            std::istringstream in(line);
            std::string kind;
            in >> kind;
            if (kind == "frame")
            {
                in >> width >> height >> frameCount;
            }
            else if (kind == "buffer")
            {
                uint32_t id, structSize, load;
                uint64_t bytes;
                in >> id >> bytes >> structSize >> load;
                Buffer b;
                b.size = bytes;
                buffers[id] = b;
                bufferLoaded[id] = load != 0;
            }
            else if (kind == "texture")
            {
                uint32_t id, format, storage;
                std::string type;
                Texture t;
                in >> id >> type >> format >> t.bytesPerPixel >> t.width >> t.height >> t.depth >> t.mips >> t.layers >> storage;
                t.format = (VkFormat)format;
                t.cube = type == "Cube";
                t.storage = storage == 1;
                textures[id] = t;
                textureKept[id] = storage == 2;
            }
            else if (kind == "sampler")
            {
                uint32_t id, minFilter, magFilter, mipFilter, addressU, addressV, addressW, anisotropy;
                float lodBias, minLod, maxLod;
                in >> id >> minFilter >> magFilter >> mipFilter >> addressU >> addressV >> addressW >> anisotropy >> lodBias >> minLod >> maxLod;
                // Falcor: filtering Point 0 / Linear 1; addressing Wrap, Mirror, Clamp, Border, MirrorOnce.
                static const VkSamplerAddressMode modes[5] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                              VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE};
                VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                ci.minFilter = minFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                ci.magFilter = magFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                ci.mipmapMode = mipFilter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                ci.addressModeU = modes[std::min(addressU, 4u)];
                ci.addressModeV = modes[std::min(addressV, 4u)];
                ci.addressModeW = modes[std::min(addressW, 4u)];
                ci.anisotropyEnable = anisotropy > 1;
                ci.maxAnisotropy = (float)anisotropy;
                ci.mipLodBias = lodBias;
                ci.minLod = std::max(minLod, 0.0f);
                ci.maxLod = maxLod;
                VK_CHECK(vkCreateSampler(dev.device, &ci, nullptr, &samplers[id]));
            }
            else if (kind == "scene")
            {
                std::string what;
                in >> what;
                if (what == "vertices") in >> vertexBuffer >> vertexStride;
                else if (what == "indices") in >> indexBuffer;
            }
            else if (kind == "geometry")
            {
                Geometry g;
                in >> g.id >> g.tlasInstance >> g.geometryIndex >> g.vbOffset >> g.ibOffset >> g.vertexCount >> g.indexCount >> g.index16 >>
                    g.frontFaceCW >> g.opaque;
                for (float& f : g.transform) in >> f;
                geometry.push_back(g);
            }
            else if (kind == "pass")
            {
                Pass p;
                in >> p.name >> p.group[0] >> p.group[1] >> p.group[2];
                passOrder.push_back(p.name);
                passes[p.name] = p;
            }
            else if (kind == "binding")
            {
                std::string passName;
                uint32_t type;
                Binding b;
                in >> passName >> b.set >> b.binding >> type >> b.count >> b.name;
                b.type = (VkDescriptorType)type;
                passes[passName].bindings.push_back(b);
            }
            else if (kind == "uniform")
            {
                std::string passName, block, member;
                uint32_t offset, size;
                in >> passName >> block >> member >> offset >> size;
                passes[passName].uniforms[block][member] = {offset, size};
            }
            else if (kind == "call")
            {
                Call c;
                in >> c.step >> c.pass >> c.dims[0] >> c.dims[1] >> c.dims[2] >> c.batch;
                calls[c.step] = c;
                Step s;
                s.step = c.step;
                s.isCall = true;
                program.push_back(s);
            }
            else if (kind == "element")
            {
                uint32_t step;
                Element e;
                in >> step >> e.set >> e.binding >> e.index >> e.kind;
                if (e.kind == "uniform") in >> e.id >> e.file;
                else if (e.kind != "accel" && e.kind != "null") in >> e.id;
                calls[step].elements.push_back(e);
            }
            else if (kind == "host")
            {
                Step s;
                in >> s.step;
                std::string word;
                while (in >> word) s.host.push_back(word);
                program.push_back(s);
            }
            else if (kind == "named")
            {
                std::string role;
                uint32_t id;
                in >> role >> id;
                named[role] = id;
            }
            else if (kind == "namedtexture")
            {
                std::string role;
                uint32_t id;
                in >> role >> id;
                namedTextures[role] = id;
            }
            else if (kind == "value")
            {
                std::string name;
                double value;
                in >> name >> value;
                values[name] = value;
            }
            else if (kind == "track")
            {
                in >> track.samples >> track.rate >> track.recordBytes >> track.loopSeconds;
            }
            else if (kind == "trackrange")
            {
                TrackRange r;
                in >> r.kind >> r.id >> r.offset >> r.bytes;
                track.ranges.push_back(r);
            }
        }
    }

    void createResources()
    {
        // Every storage buffer can also feed an acceleration structure build and be copied.
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        size_t loadedBytes = 0, scratchBytes = 0;
        for (auto& [id, b] : buffers)
        {
            const VkDeviceSize size = std::max<VkDeviceSize>(b.size, 16);
            b = dev.makeBuffer(size, usage, true);
            std::memset(b.data, 0, (size_t)size);
            if (bufferLoaded[id])
            {
                const std::vector<char> data = readFile(dir + "/data/buffer" + std::to_string(id) + ".bin");
                std::memcpy(b.data, data.data(), std::min<size_t>(data.size(), (size_t)size));
                loadedBytes += data.size();
            }
            else
            {
                scratchBytes += (size_t)size;
            }
        }
        size_t textureBytes = 0;
        for (auto& [id, t] : textures)
        {
            Texture created = dev.makeTexture(t.format, t.bytesPerPixel, t.width, t.height, t.depth, t.mips, t.layers, t.cube, t.storage);
            if (t.storage || textureKept[id])
            {
                dev.initTexture(created, nullptr);   // an output, or the runtime's own: starts cleared
            }
            else
            {
                const std::vector<char> data = readFile(dir + "/data/texture" + std::to_string(id) + ".bin");
                dev.initTexture(created, &data);
                textureBytes += data.size();
            }
            t = created;
        }
        mDummyBuffer = dev.makeBuffer(4096, usage | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
        std::memset(mDummyBuffer.data, 0, 4096);
        mDummyTexture = dev.makeTexture(VK_FORMAT_R32G32B32A32_SFLOAT, 16, 1, 1, 1, 1, 1, false, true);
        dev.initTexture(mDummyTexture, nullptr);
        VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        VK_CHECK(vkCreateSampler(dev.device, &sci, nullptr, &mDummySampler));
        std::printf("bake: %zu buffers (%.1f MB loaded, %.1f MB written by the passes), %zu textures (%.1f MB loaded), %zu samplers\n",
                    buffers.size(), loadedBytes / 1e6, scratchBytes / 1e6, textures.size(), textureBytes / 1e6, samplers.size());
    }

    VkAccelerationStructureKHR buildAS(VkAccelerationStructureTypeKHR type, const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
                                       const std::vector<uint32_t>& primitiveCounts)
    {
        VkAccelerationStructureBuildGeometryInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = type;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = (uint32_t)geometries.size();
        info.pGeometries = geometries.data();
        VkAccelerationStructureBuildSizesInfoKHR sizes = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        dev.getBuildSizes(dev.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, primitiveCounts.data(), &sizes);

        mAccelerationStorage.push_back(dev.makeBuffer(
            sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false));
        VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        ci.buffer = mAccelerationStorage.back().buffer;
        ci.size = sizes.accelerationStructureSize;
        ci.type = type;
        VkAccelerationStructureKHR as;
        VK_CHECK(dev.createAS(dev.device, &ci, nullptr, &as));

        Buffer scratch = dev.makeBuffer(sizes.buildScratchSize + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);
        info.dstAccelerationStructure = as;
        info.scratchData.deviceAddress = (dev.address(scratch) + 255) & ~VkDeviceAddress(255);
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geometries.size());
        for (size_t i = 0; i < ranges.size(); i++) ranges[i] = {primitiveCounts[i], 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
        VkCommandBuffer cb = dev.begin();
        dev.cmdBuildAS(cb, 1, &info, &pRanges);
        dev.submit(cb);
        dev.destroy(scratch);
        return as;
    }

    /// Falcor's layout: one bottom-level structure per top-level instance, its geometries in
    /// geometryIndex order, and the instance's custom index the global id of its first geometry,
    /// so that InstanceID() + GeometryIndex() of a hit is the global geometry instance id.
    void buildAccelerationStructures()
    {
        if (geometry.empty() || vertexBuffer == UINT32_MAX) fail("the bake has no geometry");
        const auto start = std::chrono::steady_clock::now();
        const VkDeviceAddress vertices = dev.address(buffers.at(vertexBuffer));
        const VkDeviceAddress indices = indexBuffer != UINT32_MAX ? dev.address(buffers.at(indexBuffer)) : 0;
        std::map<uint32_t, std::vector<Geometry>> instances;
        uint32_t maxId = 0;
        for (const Geometry& g : geometry)
        {
            instances[g.tlasInstance].push_back(g);
            maxId = std::max(maxId, g.id);
        }
        geometryTransforms.assign((size_t)(maxId + 1) * 12, 0.0f);
        for (const Geometry& g : geometry) std::memcpy(&geometryTransforms[(size_t)g.id * 12], g.transform, sizeof(float) * 12);

        uint64_t triangles = 0;
        for (auto& [index, list] : instances)
        {
            std::sort(list.begin(), list.end(), [](const Geometry& a, const Geometry& b) { return a.geometryIndex < b.geometryIndex; });
            std::vector<VkAccelerationStructureGeometryKHR> geometries;
            std::vector<uint32_t> counts;
            for (const Geometry& g : list)
            {
                VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
                geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                geo.flags = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
                auto& t = geo.geometry.triangles;
                t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                t.vertexData.deviceAddress = vertices + (VkDeviceAddress)g.vbOffset * vertexStride;
                t.vertexStride = vertexStride;
                t.maxVertex = g.vertexCount - 1;
                if (g.indexCount > 0)
                {
                    t.indexType = g.index16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                    t.indexData.deviceAddress = indices + (VkDeviceAddress)g.ibOffset * 4;
                }
                else
                {
                    t.indexType = VK_INDEX_TYPE_NONE_KHR;
                }
                geometries.push_back(geo);
                counts.push_back((g.indexCount > 0 ? g.indexCount : g.vertexCount) / 3);
                triangles += counts.back();
            }
            VkAccelerationStructureKHR blas = buildAS(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometries, counts);
            mBlas.push_back(blas);
            VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            addressInfo.accelerationStructure = blas;
            VkAccelerationStructureInstanceKHR instance = {};
            instance.instanceCustomIndex = list[0].id;
            instance.mask = 0xff;
            instance.flags = list[0].frontFaceCW ? VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR : 0;
            instance.accelerationStructureReference = dev.getASAddress(dev.device, &addressInfo);
            mTlasInstances.push_back(instance);
            mTlasGeometry.push_back(list[0].id);
        }
        mTlasInstanceBuffer = dev.makeBuffer(
            mTlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true);
        buildTlas();
        std::printf("acceleration structures: %zu geometries, %llu triangles, %zu instances, built in %.1f ms\n", geometry.size(),
                    (unsigned long long)triangles, mTlasInstances.size(), msSince(start));
    }

    /// Builds the top-level structure from the current geometry transforms: the first time, and
    /// again, into the same structure, whenever instances have moved. The GPU must be idle.
    void buildTlas()
    {
        for (size_t i = 0; i < mTlasInstances.size(); i++)
            std::memcpy(mTlasInstances[i].transform.matrix, &geometryTransforms[(size_t)mTlasGeometry[i] * 12], sizeof(float) * 12);
        std::memcpy(mTlasInstanceBuffer.data, mTlasInstances.data(), (size_t)mTlasInstanceBuffer.size);

        VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geo.geometry.instances.data.deviceAddress = dev.address(mTlasInstanceBuffer);
        VkAccelerationStructureBuildGeometryInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = 1;
        info.pGeometries = &geo;
        const uint32_t count = (uint32_t)mTlasInstances.size();
        if (tlas == VK_NULL_HANDLE)
        {
            VkAccelerationStructureBuildSizesInfoKHR sizes = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            dev.getBuildSizes(dev.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &sizes);
            mAccelerationStorage.push_back(dev.makeBuffer(
                sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false));
            VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            ci.buffer = mAccelerationStorage.back().buffer;
            ci.size = sizes.accelerationStructureSize;
            ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            VK_CHECK(dev.createAS(dev.device, &ci, nullptr, &tlas));
            mTlasScratch = dev.makeBuffer(sizes.buildScratchSize + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);
        }
        info.dstAccelerationStructure = tlas;
        info.scratchData.deviceAddress = (dev.address(mTlasScratch) + 255) & ~VkDeviceAddress(255);
        const VkAccelerationStructureBuildRangeInfoKHR range = {count, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
        VkCommandBuffer cb = dev.begin();
        dev.cmdBuildAS(cb, 1, &info, &pRange);
        dev.submit(cb);
    }

    /// The scene's blocks are the same data in every call: one buffer for all of them, so that the
    /// camera is updated in one place. A pass's own block belongs to the call: the same pass runs
    /// with different values at different points of the frame.
    static std::string uniformKey(const Call& c, const std::string& block)
    {
        return block.rfind("gScene", 0) == 0 ? block : "call" + std::to_string(c.step) + ":" + block;
    }

    /// Whether a call binds a buffer of a ping-pong pair.
    bool swaps(const Call& c) const
    {
        for (const Element& e : c.elements)
            if (e.kind == "buffer" && bufferSwap.count(e.id) != 0) return true;
        return false;
    }

    void createPasses()
    {
        // One pool for the descriptor sets of every call (two groups for a call with swapped pairs).
        std::map<VkDescriptorType, uint32_t> totals;
        uint32_t setCount = 0;
        for (auto& [step, c] : calls)
        {
            const uint32_t groups = swaps(c) ? 2 : 1;
            uint32_t maxSet = 0;
            for (const Binding& b : passes.at(c.pass).bindings)
            {
                totals[b.type] += b.count * groups;
                maxSet = std::max(maxSet, b.set);
            }
            setCount += (maxSet + 1) * groups;
        }
        std::vector<VkDescriptorPoolSize> sizes;
        for (auto& [type, count] : totals) sizes.push_back({type, count});
        VkDescriptorPoolCreateInfo pci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = setCount;
        pci.poolSizeCount = (uint32_t)sizes.size();
        pci.pPoolSizes = sizes.data();
        VK_CHECK(vkCreateDescriptorPool(dev.device, &pci, nullptr, &mPool));

        double totalMs = 0.0;
        for (const std::string& name : passOrder)
        {
            Pass& p = passes[name];
            uint32_t maxSet = 0;
            for (const Binding& b : p.bindings) maxSet = std::max(maxSet, b.set);
            p.setLayouts.resize(maxSet + 1);
            for (uint32_t set = 0; set <= maxSet; set++)
            {
                std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
                for (const Binding& b : p.bindings)
                    if (b.set == set) layoutBindings.push_back({b.binding, b.type, b.count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
                VkDescriptorSetLayoutCreateInfo sl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                sl.bindingCount = (uint32_t)layoutBindings.size();
                sl.pBindings = layoutBindings.data();
                VK_CHECK(vkCreateDescriptorSetLayout(dev.device, &sl, nullptr, &p.setLayouts[set]));
            }
            VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = (uint32_t)p.setLayouts.size();
            pl.pSetLayouts = p.setLayouts.data();
            VK_CHECK(vkCreatePipelineLayout(dev.device, &pl, nullptr, &p.layout));

            const std::vector<char> code = readFile(dir + "/spirv/" + name + ".spv");
            VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            sm.codeSize = code.size();
            sm.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            VK_CHECK(vkCreateShaderModule(dev.device, &sm, nullptr, &module));
            VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            ci.layout = p.layout;
            const auto start = std::chrono::steady_clock::now();
            VK_CHECK(vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline));
            p.createMs = msSince(start);
            totalMs += p.createMs;
            vkDestroyShaderModule(dev.device, module, nullptr);
        }
        for (auto& [step, c] : calls)
        {
            Pass& p = passes.at(c.pass);
            auto allocate = [&](std::vector<VkDescriptorSet>& sets) {
                sets.resize(p.setLayouts.size());
                for (size_t set = 0; set < p.setLayouts.size(); set++)
                {
                    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                    da.descriptorPool = mPool;
                    da.descriptorSetCount = 1;
                    da.pSetLayouts = &p.setLayouts[set];
                    VK_CHECK(vkAllocateDescriptorSets(dev.device, &da, &sets[set]));
                }
            };
            allocate(c.sets);
            writeDescriptors(p, c, c.sets, false);
            if (swaps(c))
            {
                allocate(c.altSets);
                writeDescriptors(p, c, c.altSets, true);
            }
        }
        std::printf("pipelines: %zu passes created in %.0f ms; %zu calls in the recorded frame\n", passOrder.size(), totalMs, calls.size());
    }

    /// `swapped`: a buffer of a ping-pong pair is replaced by its partner.
    void writeDescriptors(Pass& p, Call& c, const std::vector<VkDescriptorSet>& sets, bool swapped)
    {
        for (const Element& e : c.elements)
        {
            const Binding* binding = nullptr;
            for (const Binding& b : p.bindings)
                if (b.set == e.set && b.binding == e.binding) binding = &b;
            if (!binding) continue;

            VkDescriptorBufferInfo bufferInfo = {mDummyBuffer.buffer, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo imageInfo = {mDummySampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSetAccelerationStructureKHR accelInfo = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            accelInfo.accelerationStructureCount = 1;
            accelInfo.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[e.set];
            w.dstBinding = e.binding;
            w.dstArrayElement = e.index;
            w.descriptorCount = 1;
            w.descriptorType = binding->type;

            switch (binding->type)
            {
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                if (e.kind == "buffer") bufferInfo.buffer = buffers.at(swapped && bufferSwap.count(e.id) != 0 ? bufferSwap.at(e.id) : e.id).buffer;
                w.pBufferInfo = &bufferInfo;
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                if (e.kind == "uniform")
                {
                    const std::string key = uniformKey(c, binding->name);
                    auto it = mUniformBuffers.find(key);
                    if (it == mUniformBuffers.end())
                    {
                        Buffer b = dev.makeBuffer(std::max<VkDeviceSize>(e.id, 16), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
                        const std::vector<char> data = readFile(dir + "/" + e.file);
                        std::memcpy(b.data, data.data(), std::min<size_t>(data.size(), (size_t)b.size));
                        it = mUniformBuffers.emplace(key, b).first;
                    }
                    c.uniformBlocks[binding->name] = &it->second;
                    bufferInfo.buffer = it->second.buffer;
                }
                w.pBufferInfo = &bufferInfo;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            {
                const bool storage = binding->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                const Texture& t = e.kind == "texture" ? textures.at(e.id) : mDummyTexture;
                imageInfo.sampler = VK_NULL_HANDLE;
                imageInfo.imageView = storage ? t.storageView : t.view;
                w.pImageInfo = &imageInfo;
                break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                if (e.kind == "sampler") imageInfo.sampler = samplers.at(e.id);
                w.pImageInfo = &imageInfo;
                break;
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                w.pNext = &accelInfo;
                break;
            default:
                fail(format("%s: descriptor type %d of %s is not handled", p.name.c_str(), (int)binding->type, binding->name.c_str()));
            }
            vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);   // the infos above are alive until here
        }
    }
};
}  // namespace replay
