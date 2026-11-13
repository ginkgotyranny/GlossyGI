// NCR's two hash-grid encodings as GPU code: the same arithmetic as Model/HashGrid.cu (primary grid)
// and Model/HashGridInterp.cu (cluster grid), which run in CUDA on the desktop.
//
// The level tables are closed forms of the level index instead of constant arrays: with the level a
// compile-time constant after unrolling they fold to literals (a loop over constant arrays was 20x
// slower on Maleoon in the Device MRRC port).

static const uint kLog2HashMapSize = 18;                 // NCR_LOG2_HASHMAP_SIZE(_SPEC)
static const uint kHashMapSize = 1u << kLog2HashMapSize;
static const uint kPrimeY = 19349663u;                   // PRIME_X is 1
static const uint kPrimeZ = 83492791u;

static const uint kPrimLevels = 6;       // NCR_N_LEVELS
static const uint kPrimFeatures = 2;     // NCR_N_FEATURES_PER_LEVEL
static const uint kPrimBaseRes = 32;     // NCR_BASE_RESOLUTION
static const uint kClsLevels = 6;        // NCR_N_LEVELS_SPEC
static const uint kClsFeatures = 4;      // NCR_N_FEATURES_PER_LEVEL_SPEC
static const uint kClsBaseRes = 16;      // NCR_BASE_RESOLUTION_SPEC
static const uint kNumClusters = 4;      // NCR_NUM_CLUSTERS
static const float kInterpRatio = 1.0f;  // NCR_INTERP_RATIO

struct GridLevel
{
    uint res;      // cells per axis; the level has (res + 1)^3 vertices
    uint size;     // entries stored: all vertices, or the hash table size
    uint offset;   // first entry of the level in the grid buffer
    bool hashed;
};

uint levelEntries(uint baseRes, uint level)
{
    const uint res1 = (baseRes << level) + 1;
    const uint dense = res1 * res1 * res1;   // fits 32 bits up to res 1024
    return min(dense, kHashMapSize);
}

GridLevel gridLevel(uint baseRes, uint level)
{
    GridLevel l;
    l.res = baseRes << level;
    const uint res1 = l.res + 1;
    l.hashed = res1 * res1 * res1 > kHashMapSize;
    l.size = levelEntries(baseRes, level);
    l.offset = 0;
    [ForceUnroll]
    for (uint i = 0; i < 6; i++)
    {
        if (i < level) l.offset += levelEntries(baseRes, i);
    }
    return l;
}

/// Entry of grid vertex p. 32-bit wrap-around gives the low bits the CUDA code keeps.
uint gridIndex(uint3 p, GridLevel l)
{
    if (l.hashed) return (p.x + p.y * kPrimeY + p.z * kPrimeZ) & (kHashMapSize - 1);
    const uint res1 = l.res + 1;
    return ((res1 * p.x + p.y) * res1 + p.z) % l.size;
}

uint3 cornerOffset(uint c)
{
    return uint3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}

float cornerWeight(uint3 corner, float3 frac)
{
    return (corner.x != 0 ? frac.x : 1 - frac.x) * (corner.y != 0 ? frac.y : 1 - frac.y) * (corner.z != 0 ? frac.z : 1 - frac.z);
}

/// Cluster grid: the two levels bracketing the query radius and their blend weights
/// (HashGridInterp::forwardKernel). Level i has voxel scale kInterpRatio / (kClsBaseRes << i).
struct ClusterLevels
{
    uint upper;         // finer level
    uint lower;         // coarser level
    float upperWeight;
    float lowerWeight;
};

float clusterVoxelScale(uint level)
{
    return kInterpRatio / float(kClsBaseRes << level);
}

ClusterLevels clusterLevels(float scale)
{
    ClusterLevels r;
    // Finer level: first level whose voxel is smaller than the radius.
    int level = int(kClsLevels);
    for (int i = int(kClsLevels) - 1; i >= 0; --i)
    {
        if (scale > clusterVoxelScale(uint(i))) level = i;
    }
    if (level == 0)
    {
        r.upperWeight = 0.0f;
        r.upper = 0;
    }
    else if (level == int(kClsLevels))
    {
        r.upperWeight = 1.0f;
        r.upper = kClsLevels - 1;
    }
    else
    {
        const float coarser = clusterVoxelScale(uint(level - 1));
        const float finer = clusterVoxelScale(uint(level));
        r.upperWeight = (coarser - scale) / (coarser - finer);
        r.upper = uint(level);
    }
    // Coarser level: last level whose voxel is larger than the radius.
    level = -1;
    for (int j = 0; j < int(kClsLevels); ++j)
    {
        if (scale < clusterVoxelScale(uint(j))) level = j;
    }
    if (level == -1)
    {
        r.lowerWeight = 1.0f;
        r.lower = 0;
    }
    else if (level == int(kClsLevels) - 1)
    {
        r.lowerWeight = 0.0f;
        r.lower = kClsLevels - 1;
    }
    else
    {
        const float coarser = clusterVoxelScale(uint(level));
        const float finer = clusterVoxelScale(uint(level + 1));
        r.lowerWeight = (scale - finer) / (coarser - finer);
        r.lower = uint(level);
    }
    return r;
}
