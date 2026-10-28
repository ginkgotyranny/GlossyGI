# NCR（Neural Radiosity）算法完整管线说明文档

> 对应代码：`Falcor-stage3/Source/RenderPasses/NeuralRadiosity/`（分支 `stage4dev`）
> 适用场景：`./run.sh` → 选择 `ncr` 脚本 → 选择场景 → 在编辑器 UI 中将 NeuralRadiosity 的 **Render Mode 设为 Online Train**
> 本文档基于对以下源文件的逐行阅读整理：`NeuralRadiosity.cpp/.h`、全部 9 个 `.cs.slang` 计算着色器、`NetworkParams.slang`、`PathState.slang`、`Model/Model.cu/.h`、`Model/HashGrid.cu/.h`、`Model/HashGridInterp.cu/.h`

---

## 1. 算法概述

本工程实现的是一个**双瓣（Diffuse + Specular）神经辐射度缓存**系统，思想源自 *Neural Radiosity*（S. Hadadan, ToG 2021）：

- 场景中每个"光滑（非镜面 delta）"表面顶点的**入射/出射辐射度**由两个小型神经网络（半精度、tiny-cuda-nn 实现）参数化：
  - **DiffNet**：查询光滑漫反射顶点的辐射度；
  - **SpecNet**：查询光泽（glossy）顶点的辐射度。SpecNet 的核心创新是**锥体追踪 + KMeans 聚类**：对每个光泽顶点发射多条 GGX BSDF 采样光线，把命中距离聚类成 4 个"簇"（每个簇 = 一团可能的二次反射表面），把簇的位置/尺度/权重作为网络输入，从而让网络感知光泽反射的**空间分布不确定性**。
- 渲染时：从相机发射光线，穿过纯镜面/玻璃（delta）表面后到达第一个光滑顶点，用网络查询该顶点沿视线方向的辐射度，乘以吞吐量得到像素颜色；光泽像素再经联合双边滤波 + FXAA 去噪。
- 训练时（Online Train）：每帧从当前相机随机采样一批"左手项 LHS"光滑交互点，对每个 LHS 沿 BSDF 采样多条"右手项 RHS"路径，追踪到下一光滑顶点并用**当前网络自举（bootstrap）**估计其辐射度，多条 RHS 平均后作为 LHS 的监督目标，对网络做一步 EMA+Adam 更新——即**每帧边训练边渲染**，适用于动态场景。

## 2. 你的操作流程 ↔ 代码对应关系

| 你的操作 | 代码位置 | 发生了什么 |
|---|---|---|
| `./run.sh`，选 `ncr`，选场景（如 bathroom / anim） | `run.sh` | 启动 `build/linux-gcc/bin/Release/Mogwai --script scripts/NeuralRadiosity.py --scene media/scenes/<name>/scene-v4.pyscene`（或 `scene-animation.pyscene`），带 `LD_PRELOAD` libpng16 |
| Mogwai 加载脚本 | `scripts/NeuralRadiosity.py` | 创建 RenderGraph：`NeuralRadiosity` pass 的输出 `color` → `ToneMapper.src`，`ToneMapper.dst` 标记为图输出 |
| UI 中 NeuralRadiosity → Render Mode = **Online Train** | `NeuralRadiosity.cpp:213` `renderUI()` | ① `setConeParameters(false)`：锥采参数切换为渲染档（`NUM_SPEC_RAYS` 128→32，KMeans 迭代 10→3，重建 ConeTrace 着色器）；② `mpNRModel->setOnline(true)`：网络切换到在线（EMA）模式；③ `mVarsChanged=true` 触发资源重分配 |
| 每一帧 | `NeuralRadiosity.cpp:160` `execute()` | Online Train 模式下**先 `train()` 再 `render()`**，两管线都会执行（见下节） |

四种 Render Mode 的区别：

| 模式 | train() | render() | 说明 |
|---|---|---|---|
| Idle | ✗ | ✗ | 什么都不渲染，仅维持资源 |
| Render | ✗ | ✓ | 纯推理渲染（需已训练权重，可 "Load model state" 加载 ckpt） |
| Train | ✓ | ✗ | 离线训练：每帧从训练相机（"Load train cameras" 加载 json）随机取相机，跑完 `mTotalTrainSteps`（默认 20000）步自动停止，每 1000 步存 `<steps>.ckpt`；训练进程自适应调整批量（见 §6） |
| **Online Train** | ✓ | ✓ | **你使用的模式**：每帧用当前相机采样 + 训练一步 + 渲染，适合动画场景 |

**Train 与 Online Train 的完整参数差异**（重要前提：ConeTrace 只有**一份着色器程序**，RHS/LHS/Render 三个批共用同一份；`NUM_SPEC_RAYS`/`NUM_KMEANS_ITERS` 是编译进程序的宏，**不可能出现"训练批 128、渲染批 32"并存**——档位只由 Render Mode 决定）：

| 参数 | Train（离线） | Online Train（在线） |
|---|---|---|
| NUM_SPEC_RAYS（锥采光线数 = 线程组大小） | **128** | **32**（三个批全部） |
| KMeans 迭代次数 | **10** | **3** |
| numRHS（每 LHS 的 RHS 路径数） | 8→16→32→64（每 1/4 进度自适应 ×2） | 恒 8 |
| batchSize（每帧 LHS 数） | 32768→16384→8192→4096（自适应 ÷2） | 恒 32768 |
| LHS 采样相机 | 每帧随机切换训练相机（需 Load train cameras） | 当前视图相机 |
| 每帧执行 render() | ✗ | ✓ |
| 存 ckpt / 计步终止 | ✓（每 1000 步存，20000 步自动转 Idle） | ✗ |

设计逻辑：**Train 追求训练质量**（多光线精聚类、后期多 RHS 换更准目标、不赶帧率）；**Online Train 追求每帧预算稳定**（训练 + 渲染塞进同一帧时间），代价是簇特征统计量更糙（仅 32 条光线的 KMeans）且每步训练样本更少。

⚠ 两个切换相关的工程细节：① 模式切换触发 ConeTrace **重编译**（编译期宏变了，`mpConeTrace=nullptr`），有一次性卡顿；② 从训练中途的 Train 切到 Online Train 时 numRHS/batchSize **不会重置**（沿用当时的自适应值，见 §7）。

## 3. 每帧总流程（Online Train 模式）

```
Mogwai 渲染图调度 execute()
│
├─ updateFrameDim()                     帧尺寸
├─ prepareLighting()                    创建/更新 EmissiveUniformSampler（NEE 用）
├─ updatePrograms()                     首帧：编译 10 个 ComputePass + 创建 NRModel(CUDA)
├─ prepareResources()                   首帧/参数变化：分配 3 个 RayBatchBuffer + 中间纹理
│
├─ train()  ──────────────────────────── 训练管线（Train / OnlineTrain）
│   ├─ ① randomSmooth(LHSBatch)         随机相机光线 → LHS 光滑交互点采样
│   ├─ ② sampleRHS()                    每 LHS 采 8 条 BSDF+NEE 路径 → RHSBatch
│   ├─ ── RHS 子管线（为 RHS 点估计辐射度）──
│   │   ├─ ③ compactBatch(RHSBatch)     PrefixSum + 紧凑化 → 网络输入
│   │   ├─ ④ coneTrace(RHSBatch)        128 条锥采光线 + KMeans → specInput
│   │   ├─ ⑤ modelInferenceCUDA(RHS)    CUDA 网络推理（自举）
│   │   └─ ⑥ resolveBatch(RHSBatch)     color = thp × net + NEE 直射
│   ├─ ── LHS 子管线（生成监督目标并更新网络）──
│   │   ├─ ⑦ compactBatch(LHSBatch)
│   │   ├─ ⑧ coneTrace(LHSBatch)
│   │   ├─ ⑨ resolveRHS()              8 条 RHS 颜色平均 → LHS 训练目标
│   │   └─ ⑩ modelTrainCUDA(LHSBatch)   前向/损失/反向/EMA-Adam 更新
│
├─ render() ──────────────────────────── 渲染管线（Render / OnlineTrain）
│   ├─ ⑪ firstSmoothPass()             相机光线（穿过 delta 面）→ G-Buffer + active 掩码
│   ├─ ⑫ compactPass()                 PrefixSum + 紧凑化 → diffInput / specInput
│   ├─ ⑬ coneTrace(RenderBatch)        每 spec 像素 32 条锥采光线 + KMeans
│   ├─ ⑭ modelInferenceCUDA(Render)    CUDA 网络推理 → diffColor / specColor
│   ├─ ⑮ resolvePass()                 gColor = thp × netColor
│   └─ ⑯ postFilterPass()              联合双边滤波 + FXAA → gColor
│
└─ mFrameCount++ → 输出 color 通道 → ToneMapper → 屏幕
```

每帧 GPU 工作量（1280×720、Online Train 默认参数）：训练侧 32768 个 LHS + 32768×8 = **262144 条 RHS 路径**；渲染侧 921600 像素中 spec 像素 ×32 条锥采光线；网络**推理 2 次**（RHS 批 + 屏幕批）+ **训练 1 步**。

**执行顺序澄清**（流程图双列并排易读出歧义）：train() 内部**先整条 RHS 链（③④⑤⑥）再整条 LHS 链（⑦⑧⑨⑩）**，依赖链为 ⑤→⑥→⑨→⑩（⑩ 的训练目标要等 ⑥⑨ 备好）；而 train() 整体先于 render()。因此每帧的相对顺序是 **⑥ → ⑨ → ⑩ → ⑭**——⑩ 在同帧渲染推理之前，本帧渲染用的就是刚更新过的 EMA 权重。

## 4. 核心数据结构

### 4.1 RayBatchBuffer（`NeuralRadiosity.h:42`）

三份批次（`prepareResources()` 创建）：

| 批次 | 大小 | 用途 | needAll |
|---|---|---|---|
| `mpRenderBatch` | W×H（1280×720 = 921600） | 屏幕渲染 | false（只建 diff/spec 紧凑缓冲） |
| `mpTrainLHSBatch` | `mBatchSize` = 32768 | 训练左手项 | true |
| `mpTrainRHSBatch` | `mBatchSize × mNumRHS` = 32768×8 | 训练右手项 | true |

每份批次包含（全部为 `ResourceBindFlags::Shared` 的结构化缓冲，**可被 CUDA 直接映射**）：

- **全量 G-Buffer**（仅 needAll 批次）：`pos / dir / normal / albedo / roughness / vbuffer / color(吞吐量) / emission(NEE 颜色)`
- **漫反射紧凑列表**：`diffActive`（激活掩码）→ PrefixSum → `diffIndex`（紧凑下标）、`diffSize`；`diffInput`（DiffNet 输入向量）、`diffColor`（网络输出/训练目标）
- **光泽紧凑列表**：`specActive / specIndex / specSize`；`specInput`（SpecNet 输入向量）、`specColor`；以及锥采所需的 `specVBuffer / specPixel(伪像素坐标，作随机数种子) / specPos / specDir / specRoughness`

**字段分类**（哪些是"样本点的信息"，哪些不是——以 LHS 批为例）：

| 字段 | 类别 | 说明 |
|---|---|---|
| `pos` | 样本点属性 | 采样点位置（命中点沿法线偏移一个 epsilon，防自相交） |
| `normal / albedo / roughness` | 样本点属性 | 着色法线 / 反照率（金属用镜面反照率）/ 材质粗糙度 |
| `vbuffer` | 样本点属性（几何定位） | 命中"句柄"，见下 |
| `dir` | **不是点的属性** | 独立随机采样的查询方向 wi，属于网络查询对 (x, wi) 的方向分量 |
| `color`（吞吐量） | **不是点的属性** | **路径**的属性：到达该点沿途累积的 BSDF 权重 |

这组输出合起来是网络输入的"原料"：`(x, wi)` 中 x 由 pos（+normal/albedo/roughness 辅助特征）描述，wi 由 dir 描述，vbuffer 让下游能重建材质继续采样。

**vbuffer 的语义**：`hit.pack()` 得到的 `PackedHitInfo`（`Falcor/Scene/HitInfo.slang`，压缩时 uint2、否则 uint4），打包"命中类型 + instanceID + primitiveIndex + 重心坐标"——即**打中了哪个实例的哪个三角形的哪个位置**。它是几何句柄而非又一份属性快照：下游 pass 用 `unpackHitInfo → loadShadingData` 可以**不重新发射光线**就重建该点全部着色数据（材质类型、GGX alpha、纹理求值等）。SampleRHS 用 `lhsVBuffer` 还原 LHS 材质做 BSDF 采样与 NEE；ConeTrace 用 `specVBuffer` 还原光泽样本材质做 GGX 锥采样。与缓存的 normal/albedo/roughness 构成"**廉价快照 vs 权威重建**"的冗余：快照供直接读取（如 0.5 阈值分流只需 roughness 一次比较），句柄供完整还原。

**allColor 的语义变迁**（同一缓冲在 RHS 批上会被覆写，必须区分阶段理解）：

| 阶段 | 写入者 | 语义 | 值 |
|---|---|---|---|
| ① | RandomSmooth（LHS 批） | LHS 路径吞吐量 thp | 初始 (1,1,1)，穿带色镜面/染色玻璃时按 BSDF 权重相乘（LHS 批此值后续无人消费，属遗留写入） |
| ② | SampleRHS（RHS 批） | RHS 路径段吞吐量 | 终点为光滑面时 = thp（等待网络补上终点辐射度）；终点为自发光面/环境时 = thp×emission / thp×env（已是终值，样本标记 inactive） |
| ⑥ | **ResolveBatch（RHS 批，覆写）** | **该条 RHS 的完整单样本辐射度估计** | `allColor = thp × netColor + allEmission(NEE)`；inactive 样本（直接命中发光面/环境）保持 ② 的原值 |
| ⑨ | ResolveRHS（消费） | 训练目标 | 每 LHS 的 8 条 allColor 平均 → LHS 的 diffColor/specColor |

`allEmission` 是它的对照组：RandomSmooth 写 0，SampleRHS 写 NEE 贡献（MIS 加权），ResolveBatch 时并入 allColor 后完成使命。

另外注意：**批式稀疏 G-Buffer**（`all*` 前缀，按样本编号索引，点随机分布在场景表面）与 FirstSmoothPass 的**屏幕 G-Buffer**（`g*` 前缀，按像素索引，点严格对应屏幕像素）同名字段、不同域；两者字段构成几乎相同，因为都服务于"把一个光滑交互点完整打包传给网络"这一目的。

### 4.2 CUDA 互操作与同步（`NeuralRadiosity.cpp:735`）

- Slang 写完缓冲后，`pRenderContext->waitForFalcor(cudaStream)` 让 CUDA 流等待图形队列完成；
- `NRModel` 在自己的流上执行完毕后，`pRenderContext->waitForCuda(cudaStream)` 让图形队列等待 CUDA；
- 网络输入/输出直接通过 `getCudaMemory()->getMappedData()` 拿到 CUDA 指针（`RayBatchBuffer::getDiffPtrs()/getSpecPtrs()`），零拷贝。

## 5. 逐 Pass 详解

以下每个 Pass 同时给出：**输入 → 处理 → 输出**。所有屏幕 Pass 线程组 `[16,16,1]`，批处理 Pass `[256,1,1]`。

### ⑪ FirstSmoothPass —— 相机光线追踪 + G-Buffer 生成
`FirstSmoothPass.cs.slang`，dispatch `W×H`

- **输入**：相机（`gScene.camera`）、场景加速结构、材质系统。
- **处理**：每像素生成 pinhole 主光线；循环（最多 16 次）追踪：
  - 未命中 → 环境贴图辐射度 × 吞吐量，结束；
  - 命中自发光面 → 自发光 × 吞吐量，结束；
  - 命中背面（非双面材质）→ 结束；
  - 命中 **delta 纯镜面/玻璃** → 按 BSDF 采样继续传播（镜面穿透）；
  - 命中**首个光滑表面** → 记录该交互点，结束。
- **输出**：屏幕空间 G-Buffer（`gPos / gDir / gNormal / gAlbedo / gRoughness / gVBuffer / gActive`，同时是 pass 的输出通道）；`gColor` = 直射辐射度（环境/自发光命中）或**吞吐量**（光滑命中）；`diffActive/specActive` 掩码（**粗糙度 > 0.5 → 漫反射瓣；≤ 0.5 → 光泽瓣**）。
- 细节：漫反射反照率 < 0.01 时用镜面反照率代替（处理金属）。

### ⑫ CompactPass —— 流式紧凑化
`CompactPass.cs.slang`（前置两遍 `PrefixSum`，`Utils/Algorithm/PrefixSum`）

- **输入**：`diffActive/specActive` 掩码、屏幕 G-Buffer、场景 AABB。
- **处理**：PrefixSum 把掩码变成紧凑下标（同时读出 `diffSize/specSize`）；每个激活像素把 G-Buffer 写入紧凑数组。`diffInput` 中**位置归一化到场景 AABB**；`specInput` 的位置保留世界坐标（归一化由 ConeTrace 完成）。
- **输出**：`diffInput`（漫反射网络输入的 GBuffer 段）、`specInput` + `specPos/specDir/specRoughness/specVBuffer/specPixel`。

### ⑬④⑧ ConeTrace —— 锥体采样 + KMeans 聚类（本算法核心）
`ConeTrace.cs.slang`，线程组 `[NUM_SPEC_RAYS,1,1]`（Train 模式 128 / Render·OnlineTrain 模式 32——**Online Train 下三个批全部用 32**，见 §2 参数对比表），**一个线程组处理一个光泽样本**，dispatch `specSize`

- **输入**：`specPos/specDir/specRoughness/specVBuffer/specPixel`、场景 AABB、场景（光线追踪）、预滤波环境立方图（当前被禁用，回退为直接 `envMap.eval`）。
- **处理**（分四步）：
  1. `loadGroupData`：线程 0 加载交互点着色数据（**禁用漫反射瓣**，只留镜面瓣），计算反射方向 `dir = reflect(dir, N)`；用 **32 项 LUT** 按粗糙度查询单位距离处镜面瓣半径 `tanLobe / cosLobe`；共享给全组；
  2. `traceRandomDepth`：每线程独立采样一条 GGX BSDF 方向（最多重试 10 次采样有效方向），追踪得到**命中距离 rayT**；
  3. `kMeansInt`：全组协同（wave 归约 + groupshared）把 32/128 个 rayT 做 **定点整数 KMeans 聚类**（尺度 1e5；min-max 均匀初始化 4 个簇心；Train 模式 10 / Render·OnlineTrain 模式 3 次迭代），得到每簇 `mean（中心距离）/ std（方差）/ count`；
  4. `writeOutput`：每簇换算成网络特征——
     - `clsPos = pos + mean × reflectDir`（簇的预测命中位置，归一化到 AABB）
     - `clsScale = (std + tanLobe × mean) / 2`（**统计方差与锥角张开量的融合**，即该簇的等效光斑半径）
     - `clsWeight = count / numSpecRays`（簇的权重）
     - 线程 0 另写入归一化主位置与 `envColor`（沿反射方向的环境光颜色）。
- **输出**：`specInput` 的簇特征段（+主位置、envColor）。
- 注：时序复用（TemporalMerge / TemporalToHistory / ScreenSpaceDenoise 三种模式及 UI 开关）的代码**当前全部被注释禁用**，每帧独立聚类。
- 深入解析（GGX/NDF 采样原理、四个子步骤的逐步机制、kMeansInt 统计量计算、miss 处理、specInput 三位写入者与装配时间线）见 **§9.5–9.6**。

### ⑭⑤ 网络推理（CUDA）—— modelInferenceCUDA
`Model.cu:325 inference()`，推理流，批大小 padUp(size, 256)

- **输入**：`diffInput → diffColor`、`specInput → specColor`（CUDA 映射指针）。
- **处理**：见 §6——HashGrid 编码（漫反射与光泽共用主网格）→（光泽再加 HashGridInterp 簇编码）→ MLP 前向；使用 **EMA 半精度推理权重**。
- **输出**：16 维输出向量的前 3 分量即 RGB 辐射度（`NetworkOutput::toFloat4()`）。

### ⑮ ResolvePass —— 屏幕合成
`ResolvePass.cs.slang`

- **输入**：`gColor`（吞吐量/直射）、`diffColor/specColor`（网络输出）、`diffIndex/specIndex`。
- **处理**：spec 像素 `gColor = thp × specColor`（优先），否则 diff 像素 `gColor = thp × diffColor`；直射像素（环境/自发光）保持不变；同时写 `tempColor` 供后滤波。
- **输出**：`gColor`、`tempColor`。

### ⑯ PostFilterPass —— 联合双边滤波 + FXAA
`PostFilterPass.cs.slang`（两个入口 `main` / `mainFXAA`）

- **处理**：
  1. **spec 像素**做 7×7（半径 3）联合双边滤波，权重 = 高斯(空间 σ=4) × 高斯(法线差 σ=0.05) × 高斯(反照率差 σ=0.02) × 高斯(颜色差 σ=2)；非 spec 像素直通 → 写入中间纹理；
  2. FXAA（可 UI 关闭）沿亮度梯度方向做抗锯齿 → 写回 `gColor`。
- **输出**：最终 `gColor`（pass 的 `color` 输出通道）→ ToneMapper → 屏幕。

### ① RandomSmooth —— LHS 训练点采样
`RandomSmooth.cs.slang`，dispatch `batchSize`(32768)

- **输入**：相机（OnlineTrain = **当前视图相机**；Train 模式且加载了训练相机 json 时每帧随机切换）、场景。
- **处理**：每线程从相机按**随机 uv** 生成 pinhole 光线（不是屏幕像素对齐的）；追踪：
  - miss / 自发光面 / 背面 → **重采样**（最多 16 轮，换随机种子重投光线）；
  - 命中 delta 表面 → 继续镜面传播；
  - 俄罗斯轮盘：`u ≥ 0.4`（60%）继续按 BSDF 采样传播；`u < 0.4`（40%）**停止并选中该光滑顶点为训练样本**；
  - 选中后**随机生成 cosine 半球入射方向 wi**（kRandomWi，作为网络查询方向）。
- **输出**：LHSBatch 的 G-Buffer + 吞吐量 + `diff/spec Active/Index` 掩码（同 0.5 阈值分流）。

### ② SampleRHS —— RHS 路径采样 + NEE（MIS）
`SampleRHS.cs.slang`，线程组 `[NUM_RHS,1,1]`（**组 = LHS 样本，线程 = RHS 编号**），dispatch `batchSize`

- **输入**：LHS 的 `pos/dir/vbuffer/active`（反解材质）、`emissiveSampler`（均匀自发光采样器）、解析光源、场景。
- **处理**（每条 RHS，初始 8 条/LHS）：
  1. **NEE**：按光源类型（自发光/解析）选择并采样光源，阴影光线遮挡测试；MIS 权重——解析光 = 1.0（BSDF 射线打不到），自发光光 = 幂启发式 `pdf/(pdf+bsdfPdf)`；BSDF 采样命中自发光三角时也计算反向 MIS `bs.pdf/(bs.pdf+pEdB)`；
  2. **BSDF 采样**：在 LHS 交互点按 BSDF 采一条散射光线，追踪到**下一个光滑交互点 / 自发光面 / 环境**（中间 delta 面继续传播）；首个光滑点记录为 RHS 交互（G-Buffer），`rhsColor = 吞吐量 × misBSDF`，`rhsEmission = Li × BSDF值 × misNEE`。
- **输出**：RHSBatch（32768×8）完整 G-Buffer + `rhsColor / rhsEmission` + 掩码。

**每 LHS 的光线账单**（一个 LHS = 8 个 RHS 线程，各带独立随机数）：

| 光线类型 | 每条 RHS 路径 | 每 LHS（×8） | 目的 |
|---|---|---|---|
| NEE 阴影光线 | 0 或 1 条（UI 开启 NEE 且有灯时） | 0 或 8 | **不找点**——测试 LHS 点到灯的连线通视性，只测遮挡、不求交点属性，很便宜 |
| BSDF 散射光线 | 恰好 1 条 | 8 | **找 RHS 点的光线**：按材质重要性采样方向，追到下一个非 delta 交互 |
| delta 续接光线 | 0–15 条 | 0–120 | BSDF 光线中途撞纯镜面/玻璃时继续穿行（与首条共享 16 次循环预算） |

典型无大量玻璃场景：每 LHS ≈ 16 条光线；每帧 32768 × 8 = 262144 条 RHS 路径。

**"first smooth" 追踪循环的终止条件**（代码注释原文 *Trace the first smooth ray*；**无轮盘赌、无作废重投**——这是与 RandomSmooth 最本质的区别：RHS 不需要“选点”，方向由 BSDF 重要性采样决定，所有结局照单全收）：

| 终止情形 | active？ | allColor 写什么 | 后续待遇 |
|---|---|---|---|
| 命中光滑面（★成功） | ✓ active | thp（暂存，等网络） | ③④⑤ 紧凑化→锥采→**网络自举**；⑥ 覆写 |
| miss（飞出场景） | ✗ | thp × env(dir) | ⑥ 不动，直接计入平均 |
| 命中自发光面 | ✗ | thp × emission × misBSDF | 同上 |
| 背面（非双面材质） | ✗ | 0 | 贡献为零 |
| delta 面上 BSDF 采样失败 | ✗ | 0 | 同上 |

⚠ 边界情况：若一条路径**整整 16 次都撞 delta 面**（镜子走廊光陷阱），循环耗尽后既没 break 也没 terminate——path 仍 active 而 `pos/dir/normal` 还是初始 0、`roughness=1`，会被当成“原点处的漫反射样本”进入网络（RandomSmooth 的预算耗尽同样存在此角落）。需要 16 连续 delta 命中，概率极低，但排查训练异常时可留意。

**NEE 的三个精确事实**（概念详解见 §9.4）：
- 每线程**至多一次** NEE（一条阴影光线）；每 LHS 实际得到 8 个独立 NEE 样本，⑨ 平均时直接光方差同样降 1/8；
- **只在 LHS 点做**：RHS 端点的直接光包含在网络预测里（神经辐射度的递归结构）；delta 中间顶点数学上不可做光源采样（delta 瓣是针状分布，光源方向恰好等于反射方向的概率为零）；
- **光源选择是两级随机**：① 等概率抽类别（本管线环境光恒不启用 → 自发光/解析 50/50，或单类 100%）→ ② 类别内 `EmissiveUniformSampler` **等概率**抽自发光三角形（1/三角形数，**不按亮度加权**）再在三角形上均匀采点 → 两级概率都需补偿（`ls.pdf *= selectionPdf; ls.Li /= selectionPdf`）。亮度悬殊场景（小射灯 vs 大面光）可换 `EmissivePowerSampler`（构造处 `NeuralRadiosity.cpp:1077`）以降方差。

### ③⑦ CompactBatch —— 批次紧凑化
`CompactBatch.cs.slang`，与 CompactPass 逻辑相同，作用于 RHS（262144 槽）/LHS（32768 槽）批次。

**两阶段执行**（C++ 侧 `compactBatch()`）：

1. **PrefixSum ×2**（diff/spec 各一次）：对 Index 缓冲做 **in-place exclusive 扫描**（`y[i]=x[0]+…+x[i-1]`），把掩码变成紧凑下标，同时把总数回读到 `diffSize/specSize`（CPU 数值，**每次带一次 GPU 同步**——每帧共 6 个这样的同步点）：

```
槽:           0    1    2    3    4    5    6
写入时两份相同（来自①/②）:
 diffActive:  0    1    0    1    1    0    1   ← 掩码
 diffIndex:   0    1    0    1    1    0    1   ← 掩码的副本
PrefixSum 后:
 diffActive:  0    1    0    1    1    0    1   ← 永不改动：回答"槽 i 是否有效"
 diffIndex:   0    0    1    1    2    3    3   ← 变成映射表：回答"槽 i 在紧凑数组第几位"
 diffSize = 4
```

**双缓冲约定**：写入时 `Active = Index`（同一掩码写两遍）；扫描后分工不可互换——Active 是后续所有判断的布尔开关（③ 搬不搬、⑥/⑨ 合不合并），Index 是稀疏↔紧凑的寻址桥梁（inactive 槽的 Index 值无意义，但永不被读）。此约定贯穿 spec 侧、屏幕路径 ⑫、以及所有 scatter 回 pass。

2. **CompactBatch 着色器**（总线程 = 批大小）：每个线程代表一个稀疏槽，active 则把 G-Buffer **gather** 到紧凑槽：
   - diff：写入 `diffInput` 的 GBuffer 段，**pos 归一化到场景 AABB**；
   - spec：写 `specInput` GBuffer 段 + `specPos/specDir/specRoughness/specVBuffer`，**保持世界坐标不归一化**——因为还要经过 ④ ConeTrace，锥采必须在世界系做光线追踪与簇位置计算（`clsPos = pos + mean×dir`），归一化推迟到 ConeTrace 的 writeOutput；`specPixel` 写入**伪像素坐标** `uint2(idx&0xFFF, idx>>12)`（batch idx 编码的随机种子，供 ConeTrace 的 GGX 采样）。

**为什么必须紧凑化**：① 网络只为有效样本花算力；② DiffNet（32 维输入）与 SpecNet（64 维）必须各排各的稠密数组才能批量推理；③ 集中做归一化预处理。输出布局（GBuffer 段就绪、编码段/簇段留空待填）见 §6.1——两网的输入维度即"每个样本的特征向量长度"，`padUp(…,16)` 对齐是 tcnn FullyFusedMLP 的半精度要求。

### ⑥ ResolveBatch —— RHS 辐射度合成
`ResolveBatch.cs.slang`

- **处理**：`allColor = 吞吐量 × netColor + emission`——把网络对 RHS 下一顶点的出射辐射度估计与 NEE 直射光合并，得到每条 RHS 路径对 LHS 辐射度的单样本估计。
- **输出**：RHSBatch 的 `allColor`。

### ⑨ ResolveRHS —— 训练目标合成
`ResolveRHS.cs.slang`

- **处理**：对每个 LHS 的 8 条 RHS 的 `allColor` **求平均**（NaN/Inf → 0），按 diff/spec 分流写入 LHS 的 `diffColor / specColor`——即**单步路径追踪估计的监督目标**。
- **输出**：LHSBatch 的 `diffColor/specColor`（训练 target）。

### ⑩ 网络训练（CUDA）—— modelTrainCUDA
`Model.cu:380 train()`，训练流，详见 §6。

## 6. 神经网络结构（CUDA / tiny-cuda-nn）

### 6.1 两套输入向量布局（`NetworkParams.slang`，`Real = fp16`）

**DiffNetInput（32 维，padUp(25,16)）**

| 偏移 | 内容 | 写入者 |
|---|---|---|
| 0–11 | 主哈希网格编码（6 级 × 2 特征） | CUDA HashGrid 前向 |
| 12–14 | pos（归一化到场景 AABB） | Slang CompactPass |
| 15–17 | dir（查询方向 wi） | Slang |
| 18–20 | normal | Slang |
| 21–23 | albedo | Slang |
| 24 | roughness | Slang |
| 25–31 | 对齐填充 0 | Slang |

**SpecNetInput（64 维，padUp(64,16)）**

| 偏移 | 内容 | 写入者 |
|---|---|---|
| 0–11 | 主哈希网格编码（主位置） | CUDA HashGrid 前向 |
| 12–27 | 簇哈希网格编码（4 簇 × 4 特征） | CUDA HashGridInterp 前向 |
| 28–30 | pos（主位置，归一化） | Slang ConeTrace |
| 31–33 | dir（反射方向） | Slang |
| 34–36 | normal | Slang |
| 37–39 | albedo | Slang |
| 40 | roughness | Slang |
| 41–43 | envColor（反射方向环境光） | Slang ConeTrace |
| 44–55 | 4 × clsPos（簇位置，归一化） | Slang ConeTrace |
| 56–59 | 4 × clsScale（簇半径） | Slang ConeTrace |
| 60–63 | 4 × clsWeight（簇权重） | Slang ConeTrace |

网络输出 `NetworkOutput` 为 16 维（`OUTPUT_DIM=16`），实际只使用前 3 维 RGB。

### 6.2 网络与编码超参（`Model.cu:84` NRModel 构造）

| 组件 | 配置 |
|---|---|
| **HashGrid 主网格** | 多分辨率哈希：6 级 × 2 特征/级，基础分辨率 32，每级 ×2，每级哈希表 2^18 条目（约 135 万元素 × 2 特征 ≈ 5 MB fp16） |
| **HashGridInterp 簇网格** | 6 级 × 4 特征/簇 × 4 簇，基础分辨率 16，`interpRatio=1.0`；以**簇位置**查询，**clsScale 决定相邻两级之间的插值权重**（尺度感知：小簇查细层、大簇查粗层），输出再乘簇权重 clsWeight |
| **DiffNet** | FullyFusedMLP：32 → 64 → 64 → 64 → 16，ReLU 隐层，SquarePlus 输出激活，半精度 |
| **SpecNet** | FullyFusedMLP：64 → 64 → 64 → 64 → 64 → 16，ReLU + SquarePlus，半精度 |
| **损失** | `RelativeL2Luminance`（相对 L2 亮度加权） |
| **优化器** | 每组参数各一个 **EMA(decay=0.95) 嵌套 Adam(lr=5e-3)**，共 4 组：主网格、簇网格、DiffNet、SpecNet |
| **精度** | 双份权重：**float 主权重**（优化器更新）+ **fp16 工作副本**（前向/反向）+ **fp16 EMA 推理权重**（渲染用，即 `custom_weights()`）；批大小对齐到 256 |
| **流** | 独立的 `mTrainStream` / `mInferenceStream`，与 Falcor 图形队列用外部信号量互相同步 |

### 6.3 训练一步的流程（`Model.cu:380`）

```
清零网格梯度 →
[diff]  HashGrid 前向(工作权重) → DiffNet forward → 损失(vs target) → DiffNet backward → HashGrid backward
[spec]  HashGrid 前向 + HashGridInterp 前向 → SpecNet forward → 损失 → backward ×2 →
DiffNetOpt.step → SpecNetOpt.step → ClsGridOpt.step → PrimGridOpt.step
（EMA 优化器内部：Adam 更新 float 主权重 → 滑动平均进 fp16 EMA 推理权重）
```

### 6.4 遗留代码（当前未启用）

- `Model/NeuralModel.cu`、`Model/NeuralConeModel.cu`：早期把编码+MLP 封装成 tcnn Module 的自定义模型（内含**梯度钳制** `clampGradients`，阈值 1.0），仅被已注释的 `createOfflineModelsIfNeeded()` 引用——当前路径不经过它们，因此**梯度钳制实际未生效**；`Model.cu` 里的 `clampGradients` kernel 同样只是定义未调用。
- 离线/在线双模型切换的 `setOnline()` 主体逻辑也被注释，现在只有 `mOnline` 标志位本身（恒为 true，推理直接走 EMA 权重）。

## 7. 关键参数速查

| 参数 | 默认值 | 来源 | 说明 |
|---|---|---|---|
| `mSpecularRoughnessThreshold` | 0.5 | 编译宏 `SPECULAR_ROUGHNESS_THRESHOLD` | diff/spec 瓣分流阈值 |
| `NUM_SPEC_RAYS` | Train 模式 128 / Render·OnlineTrain 模式 32（三个批统一，程序级宏） | `setConeParameters()` 切档 | 每光泽样本锥采光线数（= ConeTrace 线程组大小） |
| `NUM_KMEANS_ITERS` | Train 模式 10 / Render·OnlineTrain 模式 3 | 同上 | KMeans 迭代次数 |
| `NUM_CLUSTERS` | 4 | 编译宏 | KMeans 簇数 |
| `mBatchSizeInit / mNumRHSInit` | 32768 / 8 | C++ | 训练批大小与 RHS 数 |
| `RS_RR_PROB` | 0.4 | C++ 传入 | LHS 停止概率（60% 继续/40% 选中） |
| `mSaveCKPTInterval / mTotalTrainSteps` | 1000 / 20000 | C++ | 仅 Train 模式生效 |
| 自适应批量（仅 Train） | 每 1/4 进度 | `updateAdaptiveRHSState()` | numRHS ×2 且 batchSize ÷2（8→16→32→64；32768→…→4096），RHS 总射线数恒为 2^18。该调度调用位于 `if (mRenderMode == Train)` 块内，Online Train 不执行、恒为 8×32768；⚠ 但从训练中途的 Train 切换到 Online Train 时不会重置（`setConeParameters(false)` 只恢复锥采参数），会沿用当时的 numRHS/batchSize |
| PostFilter | 半径 3，σ_spatial=4, σ_normal=0.05, σ_albedo=0.02, σ_color=2 | C++ 硬编码 | 仅 spec 像素滤波 |
| NEE | UI 开关，需场景有自发光/解析光才实际启用 | `mUseNEE` | 环境光不参与 NEE（由 BSDF 射线 miss 时求值） |

## 8. 当前禁用/注意的功能（阅读代码时发现的）

1. **时序复用（Temporal Spec Reuse / Cone Trace Reuse Mode）**：UI 开关仍在，但 `mpRenderSpecTemporalHistory` 缓冲的创建与 ConeTrace 中的历史聚类路径全部被注释——**当前每帧独立聚类**，开关无实际效果。
2. **预滤波环境立方图**：`preparePrefiltEnvCubemap()` 调用被注释，ConeTrace 中 `gPrefiltEnv.valid=0` → envColor 回退为**未按粗糙度预滤波**的 `envMap.eval(reflectDir)`。
3. **梯度钳制**：如 §6.4 所述未生效。
4. `useNEE` 初始置 false，只有存在自发光集合或解析光时才置 true；环境光只能经 BSDF 采样命中。
5. Online Train 模式下**不执行**自适应批量与 ckpt 保存（那段逻辑仅 `RenderMode::Train` 走），batchSize 恒为 32768 / 8。
6. pass 声明了 8 个输出通道（color/vbuffer/active/pos/dir/normal/albedo/roughness），RenderGraph 里只有 `color` 被 ToneMapper 消费；输入通道全部注释（**该 pass 自己做光线追踪，不依赖上游 G-Buffer pass**）。

---

## 9. 附录：核心概念详解（吞吐量 / 采样权重 / BSDF）

### 9.1 吞吐量 thp（Path Throughput）

**定义**（`PathState.slang:98` 的 `thp` 字段）：路径从起点（相机 / LHS 点）走到当前点，**沿途每一次散射的 BSDF 采样权重的连乘积**。它回答的问题是：*"如果路径末端发出 1 单位的辐射度，逆着路径传回起点时还剩多少？"*

**数学出处**——路径追踪估计器：

```
L = Σ_j [ ∏_{i<j} f_i·|cosθ_i| / p_i ] × Le_j
             └──────── 这一项就是 thp ────────┘
```

**计算**（核心只有一行，[RayTraceHelper.slang:161](../Falcor-stage3/Source/RenderPasses/NeuralRadiosity/RayTraceHelper.slang#L161)）：

```cpp
path.thp = float3(1.f);      // ① 初始化（相机权重 W = 1）
path.thp *= bs.weight;       // ② 每弹射一次乘一次（在 generateScatterRay 内）
allColor[idx] = color;       // ③ 选中样本时 color = path.thp
```

数值示例（RandomSmooth 中的一条路径）：

```
thp = (1, 1, 1)                          ← 相机出发
  ↓ 穿过染色玻璃，权重 (0.9, 0.8, 0.7)
(0.9, 0.8, 0.7)
  ↓ 镜面反射（delta），权重 (1, 1, 1)
(0.9, 0.8, 0.7)
  ↓ 命中光滑面，被轮盘跳过：按 BSDF 采样继续，权重 0.5
(0.45, 0.40, 0.35)                       ← 选中样本，allColor = thp
```

三个细节：
- 被轮盘“跳过”的光滑面**同样乘权重**（走的是同一个 `generateScatterRay`）；
- 轮盘的 60%/40% 概率本身**不进 thp**（没有乘 1/0.4 之类的补偿）——它是“选训练点”的策略，不做无偏求和，无需补偿；
- 纯镜面下 thp 恒为 (1,1,1)；染色玻璃使 thp 变色——它是 float3，携带“路径被染成什么颜色”。

**RHS 批 allColor 的完整计算**（真正被消费的那个）：

```
在 LHS 点 x（查询方向 V = −wi）：
  thp = f(V,ωo)·|cosθ|/pdf （BSDF 采样）→ 沿 delta 链继续连乘
  终点为自发光面 / 环境：  allColor = thp × Le 或 thp × env(dir)   （终值，inactive）
  终点为光滑面：          allColor = thp                            （暂存）
  → ResolveBatch 覆写：   allColor = thp × netColor + allEmission(NEE)
                         └── 即单样本估计 L(x,V) ≈ (f·cosθ/p) × L_i ──┘
```

渲染侧完全同构：FirstSmoothPass 的 `gColor = thp`（只穿 delta 链）→ ResolvePass `gColor = thp × netColor`。“穿过镜子和茶色玻璃看到的地板，其网络预测辐射度”要乘上沿途吞吐量才是屏幕颜色。

### 9.2 采样权重 bs.weight = f(V,ω)·|cosθ| / pdf(ω)

着色点渲染方程 `L_o(x,V) = ∫ f(V,ω)·L_i(x,ω)·|cosθ| dω` 的蒙特卡洛估计（N=1）：

```
L_o ≈ [ f(V,ω)·|cosθ| / pdf(ω) ] × L_i(x,ω)
       └────── bs.weight ──────┘   └ 未知，递归/网络提供 ┘
```

| 因子 | 含义 |
|---|---|
| `f(V, ω)` | BSDF 值：这对方向之间材质的传递比例 |
| `\|cosθ\|` | 投影项：掠射光束照在单位表面积上的几何打折（能量守恒修正） |
| `pdf(ω)` | 采样补偿：每个样本“代表 1/p 大小的域”——采得勤的方向权重压低、采得稀的抬高，保证估计**无偏** |

两个算例：
- **纯镜面（delta）**：Falcor 特判 pdf=1，weight = 镜面反射率 → 无色镜 (1,1,1)，茶色镜直接给 thp 染色；
- **Lambert + cosine 加权采样**：`f = albedo/π`，`p = cosθ/π` → `weight = (albedo/π·cosθ)/(cosθ/π) = albedo`，**恰好约掉**——采样分布与被积函数形状匹配时权重变常数、方差最小。GGX 光泽面用 NDF 采样时抵消不干净 → weight 方差大 → **光泽路径噪声的根源**。

### 9.3 BSDF 与 BRDF

- **BSDF**（Bidirectional **Scattering** Distribution Function）：`f(ωi, ωo)`，描述表面点把光从任意入射方向散射到任意出射方向的系数（单位 sr⁻¹）。
- 家族关系：**BSDF = BRDF（反射，同侧半球）+ BTDF（透射，穿到另一侧）**。BRDF 是不透明表面下的特例；玻璃/水必须用透射部分，所以引擎接口统一叫 BSDF。

| 术语 | 管辖范围 | 典型材质 |
|---|---|---|
| BRDF | 仅反射（入射出射同在法线一侧） | 墙面、木头、金属 |
| BTDF | 仅透射（穿过表面到另一侧） | 玻璃、水 |
| BSDF | 反射 + 透射 | Falcor 的 `IMaterialInstance` 统一接口 |

Falcor 中的对应物：`mi.sample / mi.eval / mi.getProperties` 即 BSDF 接口；`BSDFSample.lobeType` 标记本次散射属于哪个“瓣”（Diffuse / SpecularReflection / Transmission / Delta…，BSDF 是多瓣叠加）。与本管线相关的位置：

- `generateScatterRay` 中 `bs.isLobe(LobeType::Transmission)` 分支：透射时光线跨到表面另一侧，并维护 `PathState.interiorList`（跟踪当前处于哪种介质体内，如玻璃内部）；
- “背面终止”检查（`isDoubleSided / frontFacing`）：有透射才需要处理背面语义；
- FirstSmoothPass / RandomSmooth 沿 delta 链传播时可以**穿过玻璃**继续走，靠的就是 BSDF 的透射瓣（`deltaTransmissionPath` 标志）；
- ConeTrace 中“禁用漫反射瓣”（`activeLobes &= ~LobeType::Diffuse`）：锥采只按光泽瓣采样。

### 9.4 NEE（Next Event Estimation，下一事件估计）

**解决什么问题**：路径追踪默认靠 BSDF 采样得到直接光——在着色点按材质分布随机采方向，“碰运气”打中光源。小而亮的灯（如大房间里的吸顶灯）只占极小立体角（可能 0.01%）→ 上万条光线才有 1 条命中，命中那条被 pdf 补偿抬得很高 → 图像表现为“萤火虫”（fireflies）噪声，收敛极慢。

**做法**：不碰运气，**直接在光源表面上采样一个点 y，连一条线**：

```
向 y 发阴影光线（traceVisibilityRay，只测通视、不求交点属性，便宜）：
  被挡 → 贡献 0
  通视 → 贡献 = f(V, y→x)·|cosθ|·Le(y)·G(x,y) / pdf_light
```

“Next Event”名字来源：路径的“下一事件”本是“下一次命中”，其中最重要的就是命中光源——NEE 把这个事件显式地直接采样，而不是等它随机发生。

**为什么必须配 MIS（防双重计数）**：直接光有两条路都能算出——NEE 直连灯 + BSDF 采样碰巧打中灯（`path.terminate` + `color = emission×thp` 分支）。两条都算会重复计入。MIS 用幂启发式给两边各分一个 ≤1 的权重（和为 1）：

| 权重 | 适用 | 代码（SampleRHS.cs.slang:276-277） |
|---|---|---|
| `misNEE = 1.0` | 解析光：无几何体，BSDF 光线永远打不中，不存在重复计数 | 三元判断前半支 |
| `misNEE = p_light/(p_light+p_bsdf)` | 自发光几何：两种技术都覆盖，按各自 pdf 分权 | `pBdE = lMi.evalPdf(lSd, ls.dir)` |
| `misBSDF = p_bsdf/(p_bsdf+p_light)` | BSDF 光线实际命中自发光三角时的另一侧权重 | `pEdB = emissiveSampler.evalPdf(...)` |

效果：灯大且近（光源采样高效）→ NEE 权重自动大；灯小而亮且落在光泽瓣里（BSDF 采样高效）→ misBSDF 自动大。MIS 自动在两种策略间平滑切换，永远不劣于单独使用任一种。

**本管线中的角色**：渲染管线（FirstSmoothPass）**没有 NEE**——渲染时直接光只能靠相机光线直接命中自发光面。NEE 只出现在训练侧：

```
训练目标 = thp × netColor（间接：网络自举） + allEmission（直接：NEE 估计，方差低）
```

即网络学的 `L(x,V)` 包含直接光；训练样本里直接光部分用 NEE 显式估计，比等 BSDF 碰运气方差小得多——小光源场景训练收敛快很多。渲染时网络一次性预测“直接+间接”总和。

**边界条件**：
- UI "Use NEE"（`mUseNEE`）初始为 false，**仅当场景存在自发光集合（`mpEmissiveSampler`）或解析光时才置 true**（`NeuralRadiosity.cpp:620-631`）；
- **环境光不走 NEE**：`useEnvLights` 声明但从未置 true——环境光只能由 BSDF 光线 miss 时求值（`envMap.eval × thp`）。天空很大时没问题；小而亮的环境光源（如太阳）会偏噪，是已知限制；
- 光源选择的两级随机与 Uniform 采样器的细节见 §5 ② 的“NEE 的三个精确事实”。

### 9.5 PBR、GGX 与 NDF 采样（镜面方向从哪来）

**PBR（Physically Based Rendering，基于物理的渲染）**：用物理原理（能量守恒、微表面理论、菲涅尔、线性空间光度量）建模材质与光照的方法论；采用它的引擎即“PBR 引擎”（UE/Unity/Falcor）。与之对照的经验模型（Phong）用“反光度”这类无物理含义的调参旋钮。Falcor 材质参数 `diffuseReflectionAlbedo / specularReflectionAlbedo / roughness`（metallic 工作流）即 PBR 三件套——管线里“漫反射反照率 < 0.01 时用镜面反照率”的代码正是金属工作流的体现（金属无漫反射分量）。

**微表面理论与 GGX**：粗糙镜面 = 无数微型完美镜子；**GGX**（Trowbridge-Reitz）是“小镜子法线朝向 h”的概率分布 D(h)，重尾特性（亮芯+柔和边缘）与实测吻合，是工业标准：

```
D(h) = α² / (π·((N·h)²(α²−1)+1)²)        α = roughness²
roughness 0 → 针状瓣（完美镜面）；1 → 宽瓣（模糊反射）
```

**NDF 采样**（采镜面方向的标准技巧，Walter 2007）：不直接采方向，**先按 D(h) 采一个微表面法线 h、再反射**：

```
tan²θh = α²·ξ₁/(1−ξ₁)（逆 CDF）, φh = 2πξ₂ → h
wo = 2(V·h)h − V（反射）
pdf(h) = D(h)·cosθh；pdf(wo) = pdf(h)/(4(V·h))（雅可比）
```

ConeTrace 里被 `if(false)` 关闭的 `stratifiedSample` 就是这三行；`mi.sample()` 内部是同一套数学。**抵消分析**：f = D·G·F/(4(N·V)(N·wo))，weight = f·(N·wo)/pdf(wo) 中 **D 与 pdf 相消**（选 D 做采样分布的原因），但 G（遮挡）、F（菲涅尔）、(V·h) 雅可比仍在 → 这就是“GGX 抵消不干净 → 光泽噪声”的数学根源。标准 NDF 采样的缺陷（掠射时采出背面法线，需重试）由 **VNDF**（Heitz 2018，可见 NDF）修复——`traceRandomDepth` 的“最多重试 10 次”正是该缺陷的工程补丁。

**为什么 32 个方向聚在反射锚周围**：三个事实叠加——① NDF 分布中心在 N；② h = N 时 wo 恰等于锚方向 M = reflect(入射, N)；③ 反射把偏差加倍（镜子偏 δ → 反射线偏 2δ）。⟹ wo 分布 ≈ **以 M 为中心、张角约 2 倍 h 分布张角的 GGX 锥**。锚方向的两个用途（clsPos 折算、envColor 采样方向）都建立在此。

### 9.6 ConeTrace 内部机制详解

**两个输入的用途**：`gScene`——①材质重建（loadShadingData 用 vbuffer 句柄 + 几何/材质数据）②锥采光线追踪（BVAS 加速结构）；场景 AABB——writeOutput 里把主位置和簇位置归一化到 [0,1]³（哈希编码的定义域；与 diff 侧同一 AABB 同一公式，保证两网共用的主 HashGrid 查询域一致）。

**四个子步骤**：

1. **loadGroupData**（线程 0 + groupshared + 屏障）：全组 32/128 条光线共享同一交互点，材质只加载一次（省 31/32 的“long latency”开销）。三件预处理：禁用漫反射瓣（spec 样本只要镜面瓣，防止 mi.sample 采出漫反射方向浪费光线）；`dir = reflect(入射, N)`（反射锚，覆盖 groupshared 的 dir）；按粗糙度查 **32 项 LUT** 插值得 `tanLobe`（单位距离处瓣半径）/`cosLobe`（瓣内判定）——GGX 能量锥角无闭式，离线积分算好查表。
2. **traceRandomDepth**（每线程）：GGX NDF 采一条方向（`mi.sample` 失败最多重试 10 次）→ 追一条光线取命中距离 rayT。**miss 不重试**：重试只覆盖“采样失败”；miss/invalid 在 KMeans 三处被 valid 掩码剔除（min-max 归约、count、sum），逃逸能量由 envColor 另行补偿。设计理由：这里的线程职责是“提供一个统计观测”，“没有观测”是合法结果；若 miss 重采会引入条件偏差（深度分布被限制在必中几何的条件下）。
3. **kMeansInt**（全组协同，定点整数 scale=1e5）：整数原子加**精确、帧间可复现**（浮点累加顺序不定会抖动）；min-max 等距初始化簇心（wave 归约求 min/max，128 线程=4 wave 需跨 wave 二级归约）；迭代 = 分配（|t−center| 最近）+ 更新（整数和/count）共 3/10 次；**方差在循环后单独一步**算（`diff²/scale` 防溢出写法，÷scale 还原成场景单位²）。三个统计量的计算路径：
   - **count**：`InterlockedAdd(gsCounts[bestK], valid?1:0)`，每轮清零重计，**最后一轮的计数留存为输出**；
   - **mean**：迭代中的“簇心”即最终输出（最后一轮更新值，整数除法），不再单独计算；
   - **std** = √(Σ(t−mean)²/count)，用最终分配+最终簇心补算一次（总体标准差）；
   - **空簇**：min-max 等距初始化的已知弱点——簇心变 0、count=0 → clsPos=pos、clsScale=0、**clsWeight=0** → 被 HashGridInterp 的簇权重归零，无害；
   - 数值例子：有效 rayT 分布 2m×12 / 5m×10 / 9m×6（4 条 miss）→ min-max 初始 [1.9, 4.33, 6.77, 9.2]m → 一轮后簇心 (2.03, 5.05, 空, 8.90)m → 最终 std₀=√(ΣΔt²/12)≈0.082m → clsWeight = 12/32（**分母固定为光线总数**，miss 降低总权重）。
4. **writeOutput**：簇主线程 0..3 写 4×(clsPos/clsScale/clsWeight)；线程 0 写**主位置**（specPos 归一化到 AABB——**覆写**③写入的未归一化版本；它就是主 HashGrid 的查询坐标）与 **envColor**。
   - clsScale 单位细节：clsScale 保持世界单位，而 HashGridInterp 的层级阶梯是单位立方体体素尺度——比较隐含假设场景 AABB 尺寸 O(1)；大/小场景会系统性偏移层级选择，可用 `interpRatio`（Model.h 的 mClsGridConfig）补偿。

**envColor 三分支**（[PrefiltEnvCubemap.slang:64-79](../Falcor-stage3/Source/RenderPasses/NeuralRadiosity/PrefiltEnvCubemap.slang#L64-L79)）：

| 分支 | 条件 | 取值 | 状态 |
|---|---|---|---|
| 预滤波立方图 | valid（立方图已构建） | `eval(锚方向, roughness)`：r=roughness/2 → α → **90% 能量锥** tanθ=α×3 → lod=log₂(tanθ×faceSize) → 采模糊 mip（split-sum 式 IBL 预滤波，粗糙度=模糊度） | **禁用**（preparePrefiltEnvCubemap 调用被注释 → valid 恒 0） |
| 原图回退 | 场景有环境光 | `envMap.eval(锚方向)`：未模糊单点采样 | **当前实际路径** |
| 置零 | 无环境光 | (0,0,0) | 封闭场景 |

每个 spec 样本只采一次（解析提示特征，非 MC 估计）；粗糙表面的“锥平均”只能靠（禁用的）预滤波近似。

**specInput 的三位写入者与装配时间线**：

| 偏移 | 内容 | 写入者 | 时机 |
|---|---|---|---|
| [0..11] | 主哈希编码 | ⑤⑭ **CUDA HashGrid** | 网络前向紧前（**in-place** 原地写入） |
| [12..27] | 簇哈希编码 | ⑤⑭ **CUDA HashGridInterp** | 同上 |
| [28..30] | 主位置（归一化） | ④⑧⑬ **ConeTrace 线程0** | 覆写③写的世界坐标版 |
| [31..40] | dir/normal/albedo/roughness | ③⑦⑫ **CompactBatch/CompactPass** | 紧凑化时 |
| [41..43] | envColor | ④⑧⑬ ConeTrace 线程0 | 锥采后 |
| [44..63] | 4×(clsPos/clsScale/clsWeight) | ④⑧⑬ ConeTrace 簇主线程 | 锥采后 |

28 个编码维全程不经过任何 Slang pass；CUDA 的写入是 in-place 变换（从偏移 28 读 pos、往偏移 0..11 写编码，同一块共享缓冲）。**padUp(specSize,256) 对齐尾部是陈旧条目**（上帧残留，网络多算 ≤255 个废样本，输出落在 specColor 尾部无人读取——无害但浪费）。

### 9.7 权重的三份副本与混合精度训练

**⑤/⑭/⑩ 是跨 API 调用**：`modelInferenceCUDA/modelTrainCUDA` 是 C++ 函数而非 Slang pass——`waitForFalcor → NRModel::inference/train → waitForCuda`，两种 API 的命令队列通过外部信号量互相排队。"调用网络" = **编码（HashGrid + HashGridInterp）+ MLP 前向**的完整链条，不只是 MLP。

**一套参数、六份物理拷贝**（每参数 18 字节）：

| 拷贝 | 类型 | 职责 | 谁读写 |
|---|---|---|---|
| 主权重 | **f32** | 账本——**唯一被梯度下降更新的那份** | 只有 Adam step |
| 工作权重 | f16 | 每步被主权重**覆写**的低精度快照（不是独立参数） | 前向/反向 |
| EMA 权重 | f16 | 滑动平均副本，⑤⑭ 推理专用 | EMA step 写、推理读 |
| 梯度 | f16 | backward 写、step 读（见下） | kernel 间交接 |
| Adam 矩 m, v | **f32** | 每参数一份的滑动统计（见 §9.8） | Adam step |
| （无独立存储） | — | EMA 副本由优化器拥有（`custom_weights()`） | — |

单步数据流：`前向/反向(f16) → Adam 在 f32 上更新主权重 → 转半精度覆写工作副本 → EMA 滑动`。

**为什么账本必须 f32**：fp16 在 1.0 附近相邻可表示数间隔（ULP）≈ 0.001——小于 ~0.0005 的单步增量会被舍入**完全吞掉**；f32 的 ULP ≈ 1e-7，增量一分不丢。训练是两万步的积分，小步长被系统性舍掉会让网络"学不动"。f32 记账 + f16 计算 = **"一次性的量可以粗，累积的量必须精"**（梯度噪声本来就大，f16 无所谓；权重和 m/v 是跨步累积量，必须 f32）。**loss scale** 对付另一半问题：f16 梯度下溢（<6e-5 变 0），把损失乘大常数抬梯度进可表示范围，step 内部再除回。

**为什么梯度要常驻缓冲**：① backward 和 step 是两个独立 kernel，显存是它们唯一的交接媒介；② **哈希网格的梯度是散射累加**——每个样本只碰自己体素的几十个格子，26 万样本用原子加各自累加进同一格子的梯度缓冲，过程横跨整批，必须有落地处（MLP 梯度是稠密的，无此问题）；③ 每步先 `cudaMemsetAsync` 清零防梯度串步；④ tcnn 构造时一次性静态分配，热路径零分配。

**EMA 的双重语义**：① 时间平滑——每帧的随机梯度更新让工作权重抖动，直接渲染会闪烁；EMA(0.95) ≈ 最近 20 帧的滑动平均，出图稳定；② **target network**——⑤ 自举用 EMA（滞后），⑩ 更新工作权重（超前），目标与被训练量不是同一快照，避免"追自己正在移动的靶子"的 bootstrapped 回归正反馈发散（与 DQN target network 同源，EMA 是连续软同步）。

**显存账单**：全网络 ≈ 708 万参数（网格 539万+436万、MLP 1.15万+1.77万）× 18 B ≈ **127 MB**——这就是提交历史 "Enable half precision, save GPU Memory" 的背景。

### 9.8 Adam 优化器与梯度

**优化器** = 训练闭环（前向→损失→反向→**更新权重**）中"拿着梯度决定权重怎么动"的策略组件。谱系：

| 一代 | 策略 | 修的毛病 |
|---|---|---|
| SGD | θ ← θ − lr·g 固定步长 | — |
| +Momentum | 累积速度（惯性） | 噪声导致之字抖动 |
| RMSProp | 除以梯度均方（自适应步长） | 一套 lr 不适合所有参数 |
| **Adam**（2015） | 动量 + 自适应 | 两者都修，深度学习默认 |

**Adam 的两个矩（每参数一份）**：

```
m ← β₁·m + (1−β₁)·g     一阶矩：梯度滑动平均（惯性方向）β₁=0.9 → 记忆 ≈ 10 步
v ← β₂·v + (1−β₂)·g²    二阶矩：梯度均方（抖动尺度）  β₂=0.999 → 记忆 ≈ 1000 步
θ ← θ − lr · m̂/(√v̂ + ε)  更新 = 步长 × 信噪比
```

不对称是刻意的：**方向要灵（10 步），尺规要稳（1000 步）**——g² 本身是波动量，短窗平均无意义。m/√v ≈ 信噪比：梯度一致 → 迈大步；纯噪声 → 几乎不动；不同量级的参数自动归一。

**逐成分标注**：lr=5e-3 是步长上限（|m̂/√v̂|≲1）；**ε=1e-8 不是步长**，是分母护栏——防除零与防爆炸，健康时形同虚设（比真实梯度尺度小几个数量级），病态时救命。**哈希网格每帧都靠它保命**：539 万格子大多数帧内没被任何样本碰到（g=0, v=0），有 ε 更新量 = 0/1e-8 = 0 安静不动；没 ε 则 0/0 = NaN 毒化全网。数值安全三件套：ε（分母护栏）、loss scale（f16 梯度下溢）、gradient clamp（遗迹未启用）。

**超参的来源**：β₁/β₂/ε 是 Adam 论文推荐的经验默认值（全行业沿用，本工程取 tcnn 默认未调）；**真正手调过的是 lr=5e-3**（tcnn 常规默认 1e-3，在线训练总步数少故加大）**和 EMA decay=0.95**（≈20 帧平滑窗口）。本工程属"非平稳目标"教科书场景（动画场景每帧分布漂移），可考虑降 β₂，但作者保持默认、由外层 EMA 兜底非平稳稳定性——分工：Adam 管单步稳定，EMA 管跨帧呈现。

**梯度的粒度**：**每个参数一个梯度**（与参数同形状，全网络合成一个梯度向量）。易混的两类梯度：

| 类型 | 形状 | 角色 |
|---|---|---|
| 对激活的梯度 dL/d(激活) | 同该层输出（如 16×批） | 链式法则接力棒，中间量 |
| 对参数的梯度 dL/dθ | 同参数形状 | 最终产物，交给优化器 |

backward 按层计算只是"生产工艺"，产物是每参数一份；批量训练 = 逐样本梯度累加（MLP 在矩阵乘里隐式完成，哈希网格靠原子加显式完成——539 万格子的稀疏更新）。m/v 同样每参数一份、每帧更新一次（逐元素并行 kernel，比前向/反向便宜得多）。

### 9.9 网格编码详解：HashGrid 与 HashGridInterp

**为什么需要编码**：小 MLP 直接吃低维输入学不了高频细节（谱偏差）。网格编码 = **可学习查找表 + 三线性插值 + 多分辨率**：把 [0,1]³ 切成多层体素网格，格点存可训练特征，查询时三线性插值；粗层给全局趋势、细层给局部细节。思想源自 Instant-NGP（NVIDIA 2022）。

#### 9.9.1 HashGrid（主位置网格）

**结构**（6 级，分辨率 32→1024 每级 ×2，每条目 2 个 fp16 特征）：

| 层级 | 分辨率 | 角点数 (res+1)³ | 表条目数 | 模式 |
|---|---|---|---|---|
| L0 | 32 | 35,937 | 35,937 | **稠密**（装得下，一格一位，零碰撞） |
| L1–L5 | 64…1024 | 27万~10.7亿 | 各 262,144 (=2¹⁸) | **哈希**（装不下，有损压缩） |

**前向 kernel**（threadIdx.y = **层级**，一个样本的 6 层由 6 线程并行）：

```
① 读 pos（input 向量的 posOffset 段）
② pos×res → base（体素角点整数坐标）+ offset（体素内小数位置）
③ 8 角循环：
     角坐标 = base + (cx,cy,cz)，cx∈{0,1} 是角的轴偏移位
     哈希 h = (x·1 + y·19349663 + z·83492791) & (2¹⁸−1)
     三线性权重 = ∏(cx ? offset : 1−offset)（三轴系数连乘）
     entry[i] += features[h][i] × weight      ← i = 特征分量编号（每条目 2 个）
④ 写出：entry → input 向量 encOffset + level×2 段
     地址 = input + pixIdx×fullDim + encOffset + level×2
     （L0 写 [0][1]，L1 写 [2][3]…L5 写 [10][11]；6 线程各写各的 2 格窗口）
```

**三线性权重的由来**（1D→2D→3D）：一维两个端点 `value=(1−t)v₀+tv₁`；三维 8 角每角系数 = 三轴因子连乘。权重和恒为 1（加权平均）；查询点跨体素边界时输出**连续**（共享面上的角相同、权重此消彼长）——这是网格编码对梯度下降友好的关键。

**PRIMES**：`PRIME_X=1, PRIME_Y=19349663, PRIME_Z=83492791`——Instant-NGP 官方质数表前三位。作用是把格点坐标"搅拌"成伪均匀槽位：相邻格点**永不碰撞**（x+1 邻居 → 槽+1；y+1 → 槽+213151；z+1 → 槽+130999），碰撞只发生在远处无关位置之间。全 1 的朴素哈希会让整个对角平面共享条目（结构化踩踏）。质数与 2 的幂表尺寸互质、不产生共振模式；uint32 自然回绕参与混合；`& (2¹⁸−1)` 位掩码代替取模（这就是表大小取 2 的幂的原因）。

**反向**：同样的 8 角循环，`dL/dentry × 权重` 用 `atomicAdd` 累加进格点梯度（详见 9.8 的梯度累加机制；反向用 input 里留存的 pos **重算**同一组 base/offset/权重，保证与前向一字不差——权重是前向插值系数，链式法则里线性组合的导数就是它自己）。

#### 9.9.2 哈希碰撞为什么可接受

L1 需要 65³=274,625 个角点席位、表只有 262,144 → 鸽笼原理至少 12,481 次碰撞；L2 超编 8 倍。碰撞 = 不同格点共享特征。五重保护使其在实践中几乎无害：

1. **被共享的是编码特征，不是颜色**——最终输出 = MLP(编码, pos, normal, **albedo**, roughness, envColor, 簇特征)，表面颜色是 G-Buffer 直通输入，不经过哈希；会被混淆的只有"该尺度下辐射场结构模式"；
2. **碰撞只发生在远处**——相邻格点永不碰撞（PRIMES 性质），不存在"小物体和旁边的墙被平均"的局部模糊；被平均的是场景两端不相干的角落；
3. **场景稀疏性**——超编算的是含空气的理论格点；样本全落在表面附近（2D 流形），实际使用的条目 << 26 万，相关条目间碰撞概率低；
4. **多层冗余**——同一世界位置在各层整数坐标不同 → 哈希值不同；这层撞了别层不撞，6 层全撞概率天文数字级小；
5. **raw pos 兜底**——输入向量里有未编码的原始归一化坐标，最坏情况 MLP 仍可区分。

何时咬人：场景表面总量超表容量、或需超高频细节（如超出容量的锐利阴影边）→ 碰撞伪影/细节糊化。解法：调大 `log2HashMapSize`（每 +1 显存翻倍；2¹⁸ 对房间级场景宽裕）。

#### 9.9.3 HashGridInterp（簇网格，本工程自创）

与 HashGrid 的两点本质差异：**threadIdx.y = 簇编号**（4 簇并行，每簇用自己的 clsPos 查询）；**层级不全查**——由 clsScale 从尺度阶梯中选**相邻两层**做层间插值。

**尺度阶梯**：`gridScales[i] = interpRatio / resolution_i` = [1/16, 1/32, 1/64, 1/128, 1/256, 1/512]（interpRatio=1 时）——每级体素边长，每级减半。

**选层算法**：上搜（自底向上）找第一个"体素比簇小"的层级、下搜（自顶向下）找第一个"体素比簇大"的层级——正常情况恰为相邻两层（coarserScale/finerScale = 这两层的体素边长），层间权重线性插值：

```
upperWeight = (coarserScale − scaleVal)/(coarserScale − finerScale)   // 越接近细层越大
数值例：scaleVal=0.02 落在 1/32(0.031) 和 1/64(0.016) 之间 → upper=0.72 / lower=0.28
边界：簇比最细体素还小 → 全用最细层；比最粗还大 → 全用最粗层
```

这与三线性的单轴权重（t 和 1−t）**同构**——HashGridInterp 的完整查询 = **"3+1 维插值"**：空间 3 维三线性（8 角）× 尺度 1 维线性（2 层），空间连续、尺度连续。

**输出** = (细层特征 × upperWeight + 粗层特征 × lowerWeight) × **clsWeight**（空簇自动归零），写入 [12..27]（4 簇 × 4 特征，每簇窗口 = encOffset + clusterIdx×4）。

| | HashGrid | HashGridInterp |
|---|---|---|
| 查询坐标 | 主位置（每样本 1 个） | 每簇 clsPos（每样本 4 个） |
| 层级策略 | 6 层全查、拼接 | 只查相邻 2 层、按尺度插值 |
| 层级由谁决定 | — | clsScale |
| 分辨率 / 特征 | 32 起 / 每条目 2 | 16 起 / 每条目 4 |
| 输出 / threadIdx.y | 12 维 [0..11] / 层级 | 16 维 [12..27] / 簇 |
| 参数量 / 消费者 | ≈270 万 / diff+spec 共用 | ≈436 万 / 仅 spec |

#### 9.9.4 批处理视角：所有样本如何进网络

输入**物理上**是一整块大数组（所有样本的 32/64 维向量背靠背——正是 Slang/CUDA 共享零拷贝的那块缓冲）；网络**逻辑上**把它当 **32×N / 64×N 的列主序矩阵**（每列 = 一个样本），一次 kernel、一串 GEMM 批量前向——数学上等价于 N 次独立前向（第 j 列输出只依赖第 j 列输入），执行上是 GPU 的看家批处理。输出同构（16×N，第 j 列 = 样本 j 的预测）。批大小 padUp(…,256) 对齐的尾部为陈旧列（多算无害）。

### 9.10 interpRatio 与场景尺度校准（实测数据）

**interpRatio 的作用**：尺度阶梯的整体倍率。选层满足 `interpRatio/resolution ≈ clsScale`，反解 `resolution ≈ interpRatio/clsScale`——它的物理含义是"**1 个归一化单位对应多大的世界尺寸**"，即场景 AABB 跨度。调大 → 同一簇选更细层（更高频）；调小 → 更粗。

**单位错位回顾**：clsScale 保持世界单位（米），而阶梯是无量纲归一化单位；"阶梯数值当米用"隐含假设 **AABB 跨度 ≈ 1 且三轴接近**（归一化逐轴独立，非立方 AABB 还有各向异性拉伸）。阶梯数值 [6.25, 3.1, 1.6, 0.78, 0.39, 0.20] cm 即 1/16~1/512 假设 1 单位=1 米的换算。

**实测 AABB**（解析各场景 pyscene/pbrt 的网格+实例变换计算，脚本 `media/scenes/calc_aabb.py`；Falcor 按帧取 getSceneBounds，动画场景量级不变）：

| 场景 | AABB 跨度 Sx×Sy×Sz（米） | 均值跨度 | 建议 interpRatio |
|---|---|---|---|
| cornell-box | 2.00×2.00×2.00 | 2.00 | **2** |
| bathroom | 5.40×2.80×2.45 | 3.55 | **3.5** |
| bathroom2（动画基准位姿） | 3.91×3.94×7.65 | 5.17 | **5** |
| living-room | 5.64×2.88×4.23 | 4.25 | **4~4.5** |
| kitchen | 6.92×3.39×8.65 | 6.32 | **6~6.5** |
| veach-ajar | 17.08×3.82×8.20 | 9.70 | **10**（走廊型，x 轴独大） |
| teapot | 32.0×6.2×32.0（地面主导） | 23.4 | （无 pyscene，run.sh 不可用） |

**结论：没有一个场景满足"≈1"**——当前 interpRatio=1.0 下所有场景的 clsScale 被高估 3.5~10 倍，簇系统性挤在粗层；veach-ajar 最严重。

**修改方案（三档）**：① 快速实验——直接改 `Model.h:98` 的默认值（改动在 NRModel 构造时生效，**必须重新训练**，层级分配变了特征语义不连续）；② 正规——`setScene` 时把 `getSceneBounds()` 跨度传入 NRModel，按 (Sx+Sy+Sz)/3 自动设置；③ 最干净——在 ConeTrace 的 writeOutput 给 clsScale 除以 AABB 均值跨度（与 clsPos 归一化对称），interpRatio 永远保持 1.0。注意这些是"按设计语义"的理论正确值；当前偏粗因多层插值连续性表现为可容忍的平滑偏差，实际收益需 A/B 对比验证。
