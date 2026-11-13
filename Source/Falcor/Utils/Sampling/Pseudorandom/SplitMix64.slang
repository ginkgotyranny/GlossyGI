/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * SplitMix64 pseudorandom number generator.
 *
 * This is a fixed-increment version of Java 8's SplittableRandom generator.
 * The period is 2^64 and its state size is 64 bits.
 * It is a very fast generator passing BigCrush. It is recommended for use with
 * other generators like xoroshiro and xorshift to initialize their state arrays.
 *
 * Steele Jr, Guy L., Doug Lea, and Christine H. Flood., "Fast Splittable Pseudorandom Number Generators",
 * ACM SIGPLAN Notices 49.10 (2014): 453-472. http://dx.doi.org/10.1145/2714064.2660195.
 *
 * This code requires shader model 6.0 or above for 64-bit integer support, unless
 * FALCOR_SPLITMIX64_32BIT is defined to 1: the generator then carries its state as two 32-bit words
 * and computes the same sequence with 32-bit arithmetic, for devices without 64-bit integers
 * (shaderInt64). nextRandom64Parts() and nextRandom() exist in both variants.
 */

#ifndef FALCOR_SPLITMIX64_32BIT
#define FALCOR_SPLITMIX64_32BIT 0
#endif

#if FALCOR_SPLITMIX64_32BIT

struct SplitMix64
{
    uint2 state; ///< x: low 32 bits, y: high 32 bits.
};

/// 64-bit values below are uint2 (low, high).
uint2 splitMix64Add(uint2 a, uint2 b)
{
    uint lo = a.x + b.x;
    return uint2(lo, a.y + b.y + (lo < a.x ? 1u : 0u));
}

/// a ^ (a >> n) for 0 < n < 32.
uint2 splitMix64XorShift(uint2 a, uint n)
{
    return uint2(a.x ^ ((a.x >> n) | (a.y << (32u - n))), a.y ^ (a.y >> n));
}

/// Low 64 bits of a * b.
uint2 splitMix64Mul(uint2 a, uint2 b)
{
    // 32 x 32 -> 64 of the low words, from 16-bit halves.
    uint a0 = a.x & 0xffffu, a1 = a.x >> 16;
    uint b0 = b.x & 0xffffu, b1 = b.x >> 16;
    uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
    uint lo = (p00 & 0xffffu) | (mid << 16);
    uint hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
    return uint2(lo, hi + a.x * b.y + a.y * b.x);
}

/**
 * Generates the next pseudorandom number in the sequence (64 bits) as (low, high) words.
 */
uint2 nextRandom64Parts(inout SplitMix64 rng)
{
    rng.state = splitMix64Add(rng.state, uint2(0x7F4A7C15u, 0x9E3779B9u));
    uint2 z = rng.state;
    z = splitMix64Mul(splitMix64XorShift(z, 30), uint2(0x1CE4E5B9u, 0xBF58476Du));
    z = splitMix64Mul(splitMix64XorShift(z, 27), uint2(0x133111EBu, 0x94D049BBu));
    return splitMix64XorShift(z, 31);
}

/**
 * Generates the next pseudorandom number in the sequence (low 32 bits).
 */
uint nextRandom(inout SplitMix64 rng)
{
    return nextRandom64Parts(rng).x;
}

/**
 * Initialize SplitMix64 pseudorandom number generator.
 * @param[in] s0 Low bits of initial state (seed).
 * @param[in] s1 High bits of initial state (seed).
 */
SplitMix64 createSplitMix64(uint s0, uint s1)
{
    SplitMix64 rng;
    rng.state = uint2(s0, s1);
    return rng;
}

#else // FALCOR_SPLITMIX64_32BIT

struct SplitMix64
{
    uint64_t state;
};

uint64_t asuint64(uint lowbits, uint highbits)
{
    return (uint64_t(highbits) << 32) | uint64_t(lowbits);
}

/**
 * Generates the next pseudorandom number in the sequence (64 bits).
 */
uint64_t nextRandom64(inout SplitMix64 rng)
{
    uint64_t z = (rng.state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/**
 * Generates the next pseudorandom number in the sequence (64 bits) as (low, high) words.
 */
uint2 nextRandom64Parts(inout SplitMix64 rng)
{
    uint64_t z = nextRandom64(rng);
    return uint2(uint(z), uint(z >> 32));
}

/**
 * Generates the next pseudorandom number in the sequence (low 32 bits).
 */
uint nextRandom(inout SplitMix64 rng)
{
    return (uint)nextRandom64(rng);
}

/**
 * Initialize SplitMix64 pseudorandom number generator.
 * @param[in] s0 Low bits of initial state (seed).
 * @param[in] s1 High bits of initial state (seed).
 */
SplitMix64 createSplitMix64(uint s0, uint s1)
{
    SplitMix64 rng;
    rng.state = asuint64(s0, s1);
    return rng;
}

#endif // FALCOR_SPLITMIX64_32BIT
