#include "HashGrid.h"
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>


__constant__ uint32_t cResolutions[MAX_LEVELS];
__constant__ uint32_t cGridSizes[MAX_LEVELS];
__constant__ uint32_t cGridOffsets[MAX_LEVELS];


namespace HashGrid {

__device__ __forceinline__ uint3 _corner_offset(int c) {
    return make_uint3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}

__device__ __forceinline__ uint32_t _hash_index(uint3 index, int level, uint32_t log2HashMapSize) {
    const uint32_t maxHash = 1u << log2HashMapSize;

    uint32_t res1 = cResolutions[level] + 1;
    uint32_t denseMax = cGridSizes[level];

    uint32_t hashed = 0;

    if (denseMax > maxHash) {
        // Hash mode
        uint64_t result =
            index.x * PRIME_X +
            index.y * PRIME_Y +
            index.z * PRIME_Z;
        hashed = static_cast<uint32_t>(result & (maxHash - 1));
    } else {
        // Dense indexing mode
        // TODO: remove mod after applying bbox normalize
        hashed = ((res1 * index.x + index.y) * res1 + index.z) % denseMax;
    }

    return hashed;
}

template<int N_FEATURES_PER_LEVEL>
__global__ void forwardKernel(
    const precision_t* grids,
    precision_t* input,
    uint32_t count,
    uint32_t fullDim,
    uint32_t encOffset,
    uint32_t posOffset,
    const Config config
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level = threadIdx.y;
    if (pixIdx >= count) return;
    
    precision_t* levelOut = input + pixIdx * fullDim + encOffset + level * N_FEATURES_PER_LEVEL;
    const precision_t* posPtr = input + pixIdx * fullDim + posOffset;

    float3 pos3 = make_float3((float)posPtr[0], (float)posPtr[1], (float)posPtr[2]);

    float entry[N_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) { entry[i] = 0.0f; }

    const precision_t* grid = grids + cGridOffsets[level] * N_FEATURES_PER_LEVEL;
    
    uint32_t res = cResolutions[level];
    
    float3 posGrid = make_float3(pos3.x * res, pos3.y * res, pos3.z * res);
    uint3 base = make_uint3(posGrid.x, posGrid.y, posGrid.z);
    float3 offset = make_float3(posGrid.x - base.x, posGrid.y - base.y, posGrid.z - base.z);

    #pragma unroll
    for (int corner = 0; corner < 8; corner++) {
        // Get hash index
        uint3 corner3 = _corner_offset(corner);
        uint3 index3 = make_uint3(base.x + corner3.x, base.y + corner3.y, base.z + corner3.z);
        uint32_t index = _hash_index(index3, level, config.log2HashMapSize);

        // Calculate interpolation weight
        float weight = (corner3.x ? offset.x : (1 - offset.x)) *
                        (corner3.y ? offset.y : (1 - offset.y)) *
                        (corner3.z ? offset.z : (1 - offset.z));
        
        const precision_t* features = grid + index * N_FEATURES_PER_LEVEL;
        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
            entry[i] += (float)features[i] * weight;
        }
    }

    // Write to output
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) { levelOut[i] = (precision_t)entry[i]; }
}

template<int N_FEATURES_PER_LEVEL>
__global__ void backwardKernel(
    const precision_t* grids,
    const precision_t* input,
    const precision_t* dL_dinput,
    precision_t* dL_dgrids,
    uint32_t count,
    uint32_t fullDim,
    uint32_t encOffset,
    uint32_t posOffset,
    const Config config
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level = threadIdx.y;
    if (pixIdx >= count) return;

    const precision_t* dL_dlevelOut = dL_dinput + pixIdx * fullDim + encOffset + level * N_FEATURES_PER_LEVEL;
    const precision_t* posPtr = input + pixIdx * fullDim + posOffset;

    float3 pos3 = make_float3((float)posPtr[0], (float)posPtr[1], (float)posPtr[2]);

    float dL_dentry[N_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) { dL_dentry[i] = (float)dL_dlevelOut[i]; }

    precision_t* dL_dgrid = dL_dgrids + cGridOffsets[level] * N_FEATURES_PER_LEVEL;
    
    uint32_t res = cResolutions[level];
    
    float3 posGrid = make_float3(pos3.x * res, pos3.y * res, pos3.z * res);
    uint3 base = make_uint3(posGrid.x, posGrid.y, posGrid.z);
    float3 offset = make_float3(posGrid.x - base.x, posGrid.y - base.y, posGrid.z - base.z);

    #pragma unroll
    for (int corner = 0; corner < 8; corner++) {
        // Get hash index
        uint3 corner3 = _corner_offset(corner);
        uint3 index3 = make_uint3(base.x + corner3.x, base.y + corner3.y, base.z + corner3.z);
        uint32_t index = _hash_index(index3, level, config.log2HashMapSize);

        // Calculate interpolation weight
        float weight = (corner3.x ? offset.x : (1 - offset.x)) *
                        (corner3.y ? offset.y : (1 - offset.y)) *
                        (corner3.z ? offset.z : (1 - offset.z));

        precision_t* dL_dfeatures = dL_dgrid + index * N_FEATURES_PER_LEVEL;
        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
            atomicAdd(&dL_dfeatures[i], (precision_t)(dL_dentry[i] * weight));
        }
    }
}


// Host functions
void initializeConstants(const Config& config) {
    float resolution = config.baseResolution;
    uint32_t offset = 0;
    std::vector<uint32_t> resolutions;
    std::vector<uint32_t> gridSizes;
    std::vector<uint32_t> gridOffsets;

    for (int i = 0; i < config.nLevels; i++) {
        uint64_t elems64 = (uint64_t) resolution + 1;
        elems64 = elems64 * elems64 * elems64;
        uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << config.log2HashMapSize));

        resolutions.push_back((uint32_t) resolution);
        gridSizes.push_back(elems);
        gridOffsets.push_back(offset);

        resolution *= config.perLevelScale;
        offset += elems;
    }

    cudaMemcpyToSymbol(cResolutions, resolutions.data(), config.nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridSizes, gridSizes.data(), config.nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridOffsets, gridOffsets.data(), config.nLevels * sizeof(uint32_t));
}


void launchForward(
    cudaStream_t stream,
    const precision_t* grids,
    precision_t* input,
    uint32_t count,
    uint32_t fullDim,
    uint32_t encOffset,
    uint32_t posOffset,
    const Config& config
) {
    // Reserve threadIdx.y for level selection.
    const uint32_t blockSizeY = std::max(1u, config.nLevels);
    const uint32_t blockSizeX = std::max(1u, 256u / blockSizeY);
    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((count + blockSizeX - 1) / blockSizeX);

    switch (config.nFeaturesPerLevel) {
        case 2: forwardKernel<2><<<grid, block, 0, stream>>>(grids, input, count, fullDim, encOffset, posOffset, config); break;
        case 4: forwardKernel<4><<<grid, block, 0, stream>>>(grids, input, count, fullDim, encOffset, posOffset, config); break;
        case 8: forwardKernel<8><<<grid, block, 0, stream>>>(grids, input, count, fullDim, encOffset, posOffset, config); break;
        default: assert(false);
    }
}

void launchBackward(
    cudaStream_t stream,
    const precision_t* grids,
    const precision_t* input,
    const precision_t* dL_dinput,
    precision_t* dL_dgrids,
    uint32_t count,
    uint32_t fullDim,
    uint32_t encOffset,
    uint32_t posOffset,
    const Config& config
) {
    // Reserve threadIdx.y for level selection.
    const uint32_t blockSizeY = std::max(1u, config.nLevels);
    const uint32_t blockSizeX = std::max(1u, 256u / blockSizeY);
    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((count + blockSizeX - 1) / blockSizeX);

    switch (config.nFeaturesPerLevel) {
        case 2: backwardKernel<2><<<grid, block, 0, stream>>>(grids, input, dL_dinput, dL_dgrids, count, fullDim, encOffset, posOffset, config); break;
        case 4: backwardKernel<4><<<grid, block, 0, stream>>>(grids, input, dL_dinput, dL_dgrids, count, fullDim, encOffset, posOffset, config); break;
        case 8: backwardKernel<8><<<grid, block, 0, stream>>>(grids, input, dL_dinput, dL_dgrids, count, fullDim, encOffset, posOffset, config); break;
        default: assert(false);
    }
}

}  // namespace HashGrid
