// Exclusive prefix sum of a mask in place, and its total: what Falcor's PrefixSum utility does
// between the passes on the desktop (NeuralRadiosity::computeBatchCounts). The masks hold 0 or 1 per
// sample; the scan turns them into compacted indices and the total is the number of samples.
//
// Three dispatches over the same buffers, told apart by `stage`:
//   0  one thread per block of kBlock elements: scans its block, leaves the block's sum in gBlockSums
//   1  one thread: scans the block sums and writes the total to gCounts[countIndex]
//   2  one thread per element: adds the sum of the blocks before its own
// A thread reads and rewrites elements of a storage buffer here, which the phone's driver handles
// (the stale read-back it has concerns storage images only).
static const uint kBlock = 256;

struct Params
{
    uint count;        // elements
    uint stage;
    uint countIndex;   // 0: diffuse, 1: specular
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> gData;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gBlockSums;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> gCounts;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint blocks = (gParams.count + kBlock - 1) / kBlock;
    if (gParams.stage == 0)
    {
        if (tid.x >= blocks) return;
        const uint begin = tid.x * kBlock;
        const uint end = min(begin + kBlock, gParams.count);
        uint sum = 0;
        for (uint i = begin; i < end; i++)
        {
            const uint value = gData[i];
            gData[i] = sum;
            sum += value;
        }
        gBlockSums[tid.x] = sum;
    }
    else if (gParams.stage == 1)
    {
        if (tid.x != 0) return;
        uint sum = 0;
        for (uint b = 0; b < blocks; b++)
        {
            const uint value = gBlockSums[b];
            gBlockSums[b] = sum;
            sum += value;
        }
        gCounts[gParams.countIndex] = sum;
    }
    else
    {
        if (tid.x >= gParams.count) return;
        gData[tid.x] += gBlockSums[tid.x / kBlock];
    }
}
