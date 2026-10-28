"""离线精确复刻 HashGrid / HashGridInterp 的空间哈希，统计每层结构性冲突。

哈希函数（HashGrid.cu / HashGridInterp.cu 的 _hash_index）：
  h = (x*PRIME_X + y*PRIME_Y + z*PRIME_Z) & (2^18 - 1)     [uint32 自然回绕]
uint32 回绕只减 2^32 的整数倍，而 2^32 是 2^18 的倍数 → 低 18 位不变，
故 int64 精确求和后直接 & (2^18-1) 与 GPU 上的结果逐位一致。
稠密层（(res+1)^3 <= 2^18）走一一对应的索引，无冲突，仅统计哈希层。
"""
import numpy as np, time

PX, PY, PZ = 1, 19349663, 83492791
M = 1 << 18
MASK = M - 1

def level_stats(res):
    n = res + 1
    N = n ** 3
    if N <= M:
        return None  # dense 层，无冲突
    counts = np.zeros(M, dtype=np.int64)
    x = np.arange(n, dtype=np.int64)
    yterm = (np.arange(n, dtype=np.int64) * PY)[:, None]
    for zz in range(n):
        h = (x[None, :] + yterm + zz * PZ) & MASK      # n×n
        counts += np.bincount(h.ravel(), minlength=M)
    assert counts.sum() == N
    used = int((counts > 0).sum())
    shared_pts = int(counts[counts > 1].sum())          # 与别人共享槽位的点数
    pairs = int((counts * (counts - 1) // 2).sum())     # 冲突点对 C(c,2) 总和
    maxocc = int(counts.max())
    return dict(N=N, used=used, lost=N-used, shared=shared_pts, pairs=pairs, maxocc=maxocc)

def theory(N):
    """均匀随机哈希的理论期望（碰撞规模对照）"""
    used = M * (1 - (1 - 1/M)**N)
    pairs = N * (N-1) / 2 / M
    return used, pairs

def report(grid, resolutions):
    print(f'== {grid} （哈希层；稠密层无冲突）==')
    print(f'{"res":>5} {"格点数N":>12} {"占用槽位":>9} {"身份损失N-占用":>12} '
          f'{"共享槽的点":>10} {"冲突对数":>12} {"最大同槽":>8}  理论占用/理论对数')
    for res in resolutions:
        t0 = time.time()
        s = level_stats(res)
        if s is None:
            print(f'{res:>5} {"(稠密层, 一一对应)":>40}')
            continue
        tu, tp = theory(s['N'])
        print(f'{res:>5} {s["N"]:>12,} {s["used"]:>9,} {s["lost"]:>12,} '
              f'{s["shared"]:>10,} {s["pairs"]:>12,} {s["maxocc"]:>8}  '
              f'{tu:>9,.0f}/{tp:>9,.0f}   [{time.time()-t0:.1f}s]')

# HashGrid:      分辨率 32,64,128,256,512,1024（32 稠密）
# HashGridInterp: 分辨率 16,32,64,128,256,512  （16,32 稠密）
report('HashGrid（主网格）', [32, 64, 128, 256, 512, 1024])
report('HashGridInterp（簇网格）', [16, 32, 64, 128, 256, 512])
