---
title: "AI Infra 面试题"
description: 用面试口吻回答 AI Infra 基础题，覆盖 GPU 硬件、推理、模型优化和性能瓶颈，附口算估算，中英对照。
pubDatetime: 2026-11-11T08:00:00Z
tags:
  - interview
  - gpu
  - inference
draft: false
---

## 硬件与性能基础

#### CPU 和 GPU 区别

-   CPU是针对低延迟串行处理优化的通用处理器，而GPU是针对高吞吐量计算优化的大规模并行处理器。

    >   A CPU is a general-purpose processor optimized for low-latency serial processing, while a GPU is a massively parallel processor optimized for high-throughput computation.

<br>

![](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/1fe40e5c3d4c4bc7)

#### GPU 执行层级（Thread / Warp / Block / Grid / SM）

-   Thread：最小的执行单位，跑你写的那段标量代码，有自己的寄存器。

    >   A thread is the smallest unit of execution; it runs your scalar kernel code and has its own registers.

-   warp 是 GPU 调度的基本单位，由 32 个线程组成，同时执行同一条指令。

    >   A warp is a group of 32 threads that execute the same instruction together; it's the basic scheduling unit.

-   Block：由你定义，比如 256 个线程，也就是 8 个 warp。一个 block 只会放在一个 SM 上运行，块内线程可以共享 shared memory，也可以互相同步。

    >   A block runs on a single SM, and the threads in a block can share shared memory and synchronize with each other. 

-   Grid：一次 kernel 启动的全部 block，分散到所有 SM 上执行。

    >   A grid is all the blocks from one kernel launch, distributed across all SMs.

-   SM：GPU 的核心计算模块，里面有 CUDA Core、Tensor Core、寄存器堆、shared memory/L1 和 warp 调度器。一块 GPU 有几十到上百个 SM（A100 有 108 个）。

    >   An SM is the GPU's core compute unit, containing CUDA cores, Tensor Cores, a register file, shared memory/L1, and warp schedulers; a GPU has dozens to over a hundred SMs — 108 on an A100.



<br>

#### CUDA Core 和 Tensor Core 的区别？

-   CUDA Core 做标量运算，Tensor Core 专门做矩阵乘加，吞吐高一个数量级。

    >   CUDA cores do scalar operations, while Tensor Cores perform small matrix multiply-accumulates in one instruction, giving much higher throughput.

<br>

#### GPU 如何掩盖内存延迟？

-   一个 warp 等待数据时，调度器立即切换到另一个 warp 去执行，让计算单元保持忙碌。

    >   When one warp stalls on memory, the scheduler switches to another ready warp, keeping the compute units busy.

<br>

#### 什么是 occupancy？是不是越高越好？

-   occupancy 是每个 SM 上活跃 warp 的比例。它不是越高越好，只要够掩盖延迟就行。

    >   Occupancy is the ratio of active warps per SM; higher isn't always better — it just needs to be enough to hide latency.

<br>

#### 什么是 warp divergence？

-   同一个 warp 里的线程走了不同的分支，各分支只能依次执行，造成浪费。

    >   Warp divergence happens when threads in a warp take different branches, so the paths execute serially.

<br>

#### GPU 的存储层级？

-   寄存器 → 共享内存/L1 → L2 → 显存（HBM），越往下越大、越慢。

    >   Registers, shared memory/L1, L2, then HBM — each level is larger but slower.

<br>

#### 什么是共享内存（shared memory）？

-   片上的高速 SRAM，同一个 block 的线程共享，用来复用数据，减少对显存的访问。**SRAM（Static Random-Access Memory，静态随机存储器）** 是一种做在芯片上的高速存储器，速度快，但容量小、造价高。

    >   Shared memory is fast on-chip SRAM shared within a thread block, used to reuse data and reduce HBM traffic.
    >
    >   SRAM is the fast but small on-chip memory (registers, shared memory, L1/L2 cache), while HBM is the large but slower off-chip memory.

<br>

##### L1/L2 cache 是什么？

-   L1 在每个 SM 内部，和 shared memory 共用同一块 SRAM，容量约 200KB 量级，延迟只有几十个周期；L2 是所有 SM 共享的，A100 约 40MB，H100 约 50MB，延迟约 200 个周期，所有对显存的读写都会经过 L2。HBM 约 500 个周期以上，所以 L2 大约快 2 到 3 倍. L2 在芯片上，带宽是 HBM 的好几倍

    -   **一个时钟周期的时长 = 1 ÷ 时钟频率**。GPU 的频率大约是 1.5 到 2 GHz，所以一个周期约为 1 ÷ 1.5 GHz ≈ 0.5 到 0.7 纳秒。换算成时间来看：
        -   L1/shared memory 约 30 个周期，大约 **20 ns**
        -   L2 约 200 个周期，大约 **100 到 150 ns**
        -   显存（HBM）约 500 个周期，大约 **300 到 400 ns**

    >   L1 lives inside each SM and shares SRAM with shared memory — a couple hundred KB per SM, tens of cycles; L2 is shared by all SMs — about 40MB on A100 and 50MB on H100, around 200 cycles — and every HBM access goes through it.

-   它们的作用是让反复访问的数据不必每次都去读慢的显存。

    >   They exist so repeated accesses don’t have to go all the way to slow HBM.

<br>

##### L1 和 shared memory 有什么区别？

-   L1 是硬件自动管理的缓存；shared memory 用的是同一块 SRAM，但由程序员显式控制。两者的容量划分可以配置。

    >   L1 is a hardware-managed cache; shared memory is the same SRAM but explicitly managed by the programmer, and the split is configurable.

<br>

##### L2 对推理有什么意义？

-   模型权重和 KV Cache 远大于 L2，所以 decode 时大部分数据还是要从显存读，依然是内存受限。L2 主要帮助小的、反复访问的数据，比如激活值。从 Ampere 架构（A100）开始，CUDA 提供了 **L2 persistence** 功能，可以让指定的数据优先留在 L2 里。

    >   Weights and the KV cache are far bigger than L2, so decode still streams from HBM and stays memory-bound; L2 mainly helps small, frequently reused data like activations.

<br>

#### HBM 和 GDDR 的区别？

-   HBM 把显存芯片堆叠在 GPU 旁边，总线很宽，带宽极高（H100 约 3.35 TB/s），但很贵，用在数据中心卡；GDDR 是独立的显存颗粒，带宽只有几百 GB/s（L4 约 300 GB/s），但便宜、功耗低，用在边缘卡和消费卡。

    >   HBM stacks memory right next to the GPU with a very wide bus, giving very high bandwidth (~3.35 TB/s on H100) at high cost for datacenter GPUs; GDDR uses ==separate chips== with a few hundred GB/s (~300 GB/s on L4), but is cheaper and lower-power for edge and consumer GPUs.

-   所以边缘卡更容易内存受限，量化和 batching 在边缘设备上收益更大。

    >   That's why edge GPUs are more often memory-bound, and quantization and batching matter even more there.

<br>

#### 显存容量和带宽分别决定什么？

-   容量决定装不装得下，带宽决定跑多快。

    >   Capacity determines what fits; bandwidth determines how fast it runs.

<br>

#### 什么是合并访存（coalesced access）？

-   一个 warp 的线程访问连续的内存地址，硬件可以把这些访问合并成少量事务，从而充分利用带宽。

    >   Coalesced access means threads in a warp read contiguous addresses, so the hardware combines them into few memory transactions.

<br>

#### 为什么说"计算便宜，搬数据贵"？

-   现代 GPU 的算力增长远快于带宽，大多数时间都花在等数据上。

    >   Compute has grown much faster than memory bandwidth, so most workloads spend their time waiting on data movement.

<br>

#### 什么是寄存器溢出（register spilling）？

-   寄存器不够用时，数据被放到很慢的 local memory（实际在显存里），性能会下降。

    >   Register spilling is when a kernel needs more registers than available, so data spills to slow local memory backed by HBM.

<br>

#### FLOPs 和 FLOPS 的区别？

-   FLOPs 是浮点运算的次数，FLOPS 是每秒做多少次浮点运算。

    >   FLOPs is a count of operations; FLOPS is operations per second.

<br>

#### 什么是算术强度（arithmetic intensity）？

-   每搬运 1 字节数据要做多少次计算，等于 FLOPs ÷ 字节数。

    >   Arithmetic intensity is FLOPs per byte moved.

<br>

#### 什么是 Roofline 模型？

-   用来判断任务是受限于算力还是带宽：性能上限 = min(峰值算力, 带宽 × 算术强度)。

    >   The roofline model bounds performance by min(peak compute, bandwidth × arithmetic intensity).

<br>

#### 什么是拐点（ridge point）？

-   拐点 = 峰值算力 ÷ 带宽，是区分内存受限和计算受限的分界线。

    >   The ridge point is peak FLOPS divided by bandwidth — the intensity where a workload switches from memory-bound to compute-bound.

<br>

#### memory-bound 和 compute-bound 的区别？

-   内存受限是计算单元在等数据；计算受限是数据够用，但算力满了。

    >   Memory-bound means the compute units are waiting for data, and compute-bound means the compute units are fully busy.

<br>

#### 延迟（latency）和吞吐（throughput）的区别？

-   延迟是单个请求要多久完成，吞吐是单位时间能处理多少请求。两者通常需要取舍。

    >   Latency is how long one request takes; throughput is how many requests you finish per second — they usually trade off.

<br>

#### BF16 和 FP16 的区别？

-   两者都是 2 字节。BF16 的指数位和 FP32 一样，范围大、不容易溢出；FP16 精度更高，但范围小。

    >   Both are 2 bytes; BF16 keeps FP32's exponent range so it rarely overflows, while FP16 has more precision but a smaller range.

<br>

#### 为什么低精度更快？

-   要搬的字节数更少，而且 Tensor Core 的低精度吞吐更高。

    >   Lower precision moves fewer bytes and Tensor Cores deliver more FLOPS at lower precision.

<br>

#### 什么是量化？

-   用更低的位数（INT8、INT4、FP8）来表示权重或激活，以节省显存、加快推理，代价是可能掉精度。

    >   Quantization represents weights or activations in fewer bits to save memory and speed up inference, at some accuracy risk.

<br>

#### PCIe 和 NVLink 的区别？

-   PCIe 连接 CPU 和 GPU，约 32 到 64 GB/s；NVLink 连接 GPU 和 GPU，可达几百 GB/s。

    >   PCIe connects CPU to GPU at tens of GB/s; NVLink connects GPUs at hundreds of GB/s.

<br>

#### 什么是 kernel launch 开销？

-   CPU 每启动一个 GPU kernel 都有几微秒的固定开销，kernel 很小或很多时会成为瓶颈。可以用 CUDA Graphs 或 kernel fusion 来减少。

    >   Each kernel launch costs a few microseconds of CPU overhead, which dominates when kernels are small; CUDA Graphs or fusion reduce it.
