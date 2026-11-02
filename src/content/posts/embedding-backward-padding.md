---
title: "Embedding 反向：token 越多，显存和耗时按平方增长"
description: RL-Kernel 的确定性 embedding 反向把重复 token 的梯度补零到最常见 token 的次数，显存按平方增长、32k token 就 OOM；改成 Triton 分段求和后逐 bit 不变，单卡快 3.4～114 倍、显存降到 2 GiB 左右。
pubDatetime: 2026-11-02T20:06:00Z
tags:
  - rl-kernel
  - issue
draft: false
---

## 描述

Embedding 反向在累加重复 token 的梯度时，会先把数据塞进一个"按最常见 token 的次数补齐"的大张量。token 一多，这个张量就会爆：16k token 时要 25.4 GiB；单卡和 2 卡都在 32k 就 OOM。

## 根因

代码在 [`embedding.py:44-66`](https://github.com/RL-Align/RL-Kernel/blob/d00d444b/rl_engine/backends/cuda/embedding/embedding.py#L44-L66) 的 `_deterministic_embedding_grad_weight`。核心问题在这几行：

```python
max_count = int(counts.max().item())             # M：最常见 token 的出现次数
...
padded = torch.zeros(
    (num_unique, max_count, hidden_size),        # [U, M, H]：每个 token 都补到 M 行
    device=sorted_rows.device,
    dtype=sorted_rows.dtype,
)
padded[segment, slot] = sorted_rows
acc = padded[:, 0]
for slot_idx in range(1, max_count):             # 再按 M 步逐列相加
    acc = acc + padded[:, slot_idx]
```

为了让加法顺序固定，它把每个 token 的梯度行排成一列，再把所有列**补零到同样长度**，这个长度就是最常见 token 的出现次数 M。

> 例子：1k 个 token 里，逗号出现 37 次；另外 453 种 token 大多只出现一两次，却都要补到 37 行。

补零张量的大小 = 不同 token 数 U × M × 4096 × 4 字节。U 和 M 都随 token 数增长，所以总量按平方增长。

## 修复

### v1（舍弃）

**方案**：不补零，纯 PyTorch。把 token 按出现次数从多到少排好，第 j 步还没加完的 token 正好是最前面的 k 个，只对它们加上第 j 行。加法顺序和原实现一样，所以结果逐 bit 不变。下面是核心逻辑。

```python
acc = sorted_rows[starts_d]                      # 每个 token 的第 0 行
for j in range(1, M):                            # 和原实现一样走 M 步
    k = active[j]                                # 出现次数 > j 的 token 个数
    acc[:k] = acc[:k] + sorted_rows[starts_d[:k] + j]
```

补零张量没有了，每一步只处理还没加完的 k 个 token。但循环仍然在 Python 里走 M 步，每一步要启动 4 个小 kernel（算下标、取行、相加、写回），原实现每一步只有 1 个。

### v2（采用）

**方案**：把整个循环放进一个 Triton kernel。每个 token 交给一组线程（每组负责 512 列），在 kernel 里按原来的顺序把它的梯度行逐个加起来，直接写成 bf16。下面是核心逻辑。

```python
@triton.jit
def _segment_sum(order, rows, starts, counts, max_count, dst, out, hidden, BLOCK: tl.constexpr):
    seg = tl.program_id(0)                                     # 第几个 token
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)      # 负责哪 512 列
    start, count = tl.load(starts + seg), tl.load(counts + seg)
    acc = tl.load(rows + tl.load(order + start) * hidden + cols)          # 第 0 行
    for j in range(1, count):                                             # 按原顺序逐行相加
        acc = acc + tl.load(rows + tl.load(order + start + j) * hidden + cols)
    if count < tl.load(max_count):
        acc = acc + 0.0                  # 复现原实现补零的副作用：-0.0 变成 +0.0
    tl.store(out + tl.load(dst + seg) * hidden + cols, acc)               # 直接写 bf16
```

- **结果一个 bit 都不变**：每个 token 都从第 0 行开始，按原来的顺序做 fp32 加法。
- **kernel 数固定**：循环在 kernel 里面，launch 次数和 M 无关。
- **不要 fp32 全词表中间结果**：直接写成 bf16。
- **bf16 / fp16 梯度也走这个 kernel**（单卡 `SM90EmbeddingOp.forward` 传进来的是 bf16）：原实现在这条路径上是用 bf16 逐次相加的，kernel 每加一行就按梯度的类型舍入一次（就近舍入到偶数，和 PyTorch 在 GPU 上的 bf16 加法一样），所以也逐 bit 一致。只有设备不是 CUDA（如 Ascend）或没装 Triton 时才用纯 PyTorch 实现。
- 仓库里已有的 Triton 反向不能直接拿来用：它在 −0.0 上和原实现差一个符号位。

## 测试流程

回归测试提交在仓库里，CI 会自动跑；以后有人改这段代码，测试会拦住。拿修复前的补零实现做标准答案，按位比较（`torch.equal` 分不出 −0.0 和 +0.0）。在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,2 python -m pytest -q \
    tests/ops/embedding/test_embedding_backward_padding.py \
    tests/distributed/cp/test_cp_embedding_backward.py
```

| 文件 | 卡数 | 检查什么 |
|---|---:|---|
| `test_embedding_backward_padding.py` | 1 | 边界情况（空输入、非法 id、−0.0 / NaN / inf、非规格化数、hidden 不是 512 的倍数）和真实 hidden（H=4096）下逐 bit 一致，fp32 / bf16 / fp16 梯度都测；显存只随 token 数线性增长；GPU 操作数不随最常见 token 的出现次数增长；每个调用入口的 `weight.grad` 逐 bit 一致；CP=2 / 4 单进程模拟 |
| `test_cp_embedding_backward.py` | 2 | 真实 2 卡 CP=2 和 TP=2：每张卡和原实现逐 bit 一致，两卡合并后的梯度和单卡整条序列一致；不足 2 张卡时自动跳过 |

结果（H100，GPU 0 + GPU 2）：

```
...............................                                          [100%]
31 passed in 10.84s
```

## 性能结果

只测规约函数本身。NVIDIA H100 NVL，H=4096，全词表 V=151936，梯度 fp32，耗时取 7 次中位数。N 从 256 开始每次翻倍，一直到 64k（1k = 1024）。"快几倍"= 原实现耗时 ÷ 新实现耗时，"省几倍"= 原实现显存 ÷ 新实现显存；原实现 OOM 的行记为"—"。显存指调用期间新增的峰值。

**数据**：用 Qwen3-8B tokenizer 对本仓库 `docs/` 下的 Markdown 按路径顺序分词，再首尾相接，长度为 N 时取前 N 个 token（64k 一共用到 37 篇已提交的文档）。**不是 RL 训练数据**；梯度值是固定种子的随机数，因为结果只和 token 的分布有关。2 卡时把 token 切成长度 200～3000 的样本，再按仓库的 zigzag 方式分到两张卡。

### v1

**单卡**（GPU 1）

| N | 原实现耗时 (ms) | v1 耗时 (ms) | v1 快几倍 | 原实现显存 (GiB) | v1 显存 (GiB) | 显存省几倍 | 原实现 GPU 操作数 | v1 GPU 操作数 | 逐 bit 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.34 | 2.89 | 0.8× | 3.53 | 3.49 | 1.0× | 77 | 140 | 是 |
| 512 | 2.35 | 2.89 | 0.8× | 3.59 | 3.50 | 1.0× | 77 | 140 | 是 |
| 1k | 2.52 | 3.38 | 0.7× | 3.77 | 3.52 | 1.1× | 90 | 192 | 是 |
| 2k | 3.84 | 5.78 | 0.7× | 4.75 | 3.55 | 1.3× | 161 | 476 | 是 |
| 4k | 7.59 | 7.46 | 1.0× | 6.86 | 3.63 | 1.9× | 361 | 660 | 是 |
| 8k | 22.02 | 11.59 | 1.9× | 12.77 | 3.76 | 3.4× | 2224 | 1139 | 是 |
| 16k | 55.72 | 17.82 | 3.1× | 25.40 | 4.03 | 6.3× | 7176 | 1839 | 是 |
| 32k | 165.82 | 30.95 | 5.4× | 57.85 | 4.55 | 12.7× | 25768 | 3292 | 是 |
| 64k | **OOM** | 62.17 | — | **OOM** | 5.59 | — | **OOM** | 6816 | — |

![v1 与原实现对比：耗时](https://img.voidindex.net/posts/embedding-backward-padding/8533806bb6c040d4.png)![v1 与原实现对比：显存](https://img.voidindex.net/posts/embedding-backward-padding/8613758613a6ab85.png)

**分析**

- **显存解决了**：N 每翻一倍，原实现的显存涨到约 2 倍（16k 时 25.4 GiB，32k 时 57.9 GiB，64k OOM）；v1 从 3.5 GiB 只涨到 5.6 GiB，64k 也能跑。
- **小 N 反而变慢**：256～2k 时 v1 只有原实现的 0.7～0.8 倍，4k 时持平，8k 起才更快。原因是 GPU 操作数：例如 1k 时 192 个对 90 个。v1 每一步启动 4 个 kernel，原实现每一步 1 个；N 小的时候每个 kernel 的计算量很少，耗时主要花在启动 kernel 上。
- **大 N 时的耗时仍然随 M 线性增长**：64k 时 62.2 ms，因为 Python 循环还是走 M 步（64k 时 M=1684）。
- **结论**：一个修复不该让任何长度变慢，所以舍弃 v1，改成 v2：把循环放进一个 kernel，让 kernel 数不随 M 增长。

### v2

**单卡**（GPU 1）

| N | 原实现耗时 (ms) | v2 耗时 (ms) | v2 快几倍 | 原实现显存 (GiB) | v2 显存 (GiB) | 显存省几倍 | 逐 bit 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.34 | 0.70 | 3.4× | 3.53 | 1.16 | 3.0× | 是 |
| 512 | 2.35 | 0.68 | 3.4× | 3.59 | 1.17 | 3.1× | 是 |
| 1k | 2.52 | 0.67 | 3.7× | 3.77 | 1.18 | 3.2× | 是 |
| 2k | 3.84 | 0.71 | 5.4× | 4.75 | 1.19 | 4.0× | 是 |
| 4k | 7.59 | 0.76 | 10.0× | 6.86 | 1.22 | 5.6× | 是 |
| 8k | 22.02 | 0.92 | 23.9× | 12.77 | 1.29 | 9.9× | 是 |
| 16k | 55.72 | 1.09 | 51.0× | 25.40 | 1.41 | 18.0× | 是 |
| 32k | 165.82 | 1.46 | 113.6× | 57.85 | 1.66 | 34.8× | 是 |
| 64k | **OOM** | 2.10 | — | **OOM** | 2.16 | — | — |

![v2 与原实现、v1 对比：耗时](https://img.voidindex.net/posts/embedding-backward-padding/3a061e7f6deb9235.png)![v2 与原实现、v1 对比：显存](https://img.voidindex.net/posts/embedding-backward-padding/90749ebd0cf4a331.png)

**2 卡 CP=2**（GPU 2 + GPU 3，NVLink；每卡一段序列，耗时包含卡间 all-gather，取 rank 0）

| N | 原实现耗时 (ms) | v2 耗时 (ms) | v2 快几倍 | 原实现显存 (GiB) | v2 显存 (GiB) | 显存省几倍 | 逐 bit 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 3.1 | 1.5 | 2.0× | 3.72 | 2.32 | 1.6× | 是 |
| 512 | 3.1 | 1.5 | 2.0× | 3.60 | 2.32 | 1.5× | 是 |
| 1k | 3.3 | 1.5 | 2.2× | 3.78 | 2.33 | 1.6× | 是 |
| 2k | 4.7 | 1.6 | 3.0× | 4.77 | 2.34 | 2.0× | 是 |
| 4k | 8.6 | 1.8 | 4.7× | 6.89 | 2.35 | 2.9× | 是 |
| 8k | 23.3 | 2.2 | 10.8× | 12.83 | 2.38 | 5.4× | 是 |
| 16k | 58.4 | 2.7 | 21.7× | 25.52 | 2.45 | 10.4× | 是 |
| 32k | **OOM** | 3.7 | — | **OOM** | 2.57 | — | 是（合并后和单卡原实现一致） |
| 64k | **OOM** | 5.4 | — | **OOM** | 3.66 | — | 是（合并后和单卡 v2 一致） |

**2 卡 TP=2**（GPU 2 + GPU 3，NVLink；每卡一半词表，取 rank 0）

| N | 原实现耗时 (ms) | v2 耗时 (ms) | v2 快几倍 | 原实现显存 (GiB) | v2 显存 (GiB) | 显存省几倍 | 逐 bit 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.5 | 0.6 | 2.6× | 1.79 | 0.59 | 3.0× | 是 |
| 512 | 1.5 | 0.6 | 2.4× | 1.86 | 0.60 | 3.1× | 是 |
| 1k | 1.7 | 0.6 | 2.9× | 2.03 | 0.62 | 3.3× | 是 |
| 2k | 3.0 | 0.6 | 4.9× | 2.98 | 0.66 | 4.5× | 是 |
| 4k | 6.8 | 0.7 | 9.4× | 5.03 | 0.73 | 6.9× | 是 |
| 8k | 21.1 | 1.0 | 22.2× | 10.71 | 0.89 | 12.0× | 是 |
| 16k | 55.0 | 1.3 | 43.4× | 22.76 | 1.20 | 19.0× | 是 |
| 32k | **OOM** | 2.0 | — | **OOM** | 1.82 | — | 是（合并后和单卡原实现一致） |
| 64k | **OOM** | 3.1 | — | **OOM** | 3.05 | — | 是（合并后和单卡 v2 一致） |

2 卡的"逐 bit 一致"= 每张卡和原实现一致，且两卡合并后的梯度和单卡原实现一致；原实现 OOM 的长度只比合并后的结果。

![v2 在 2 卡上的耗时：CP=2 和 TP=2](https://img.voidindex.net/posts/embedding-backward-padding/35707a1332f8ddba.png)![v2 在 2 卡上的显存：CP=2 和 TP=2](https://img.voidindex.net/posts/embedding-backward-padding/0a6c1a89e4eee60f.png)

**分析**

- **单卡每个长度都更快**：快 3.4～114 倍；token 数涨了 256 倍，v2 耗时只从 0.70 ms 涨到 2.10 ms。一次调用只有 28～47 个 GPU 操作，和 M 无关。
- **显存基本不随 N 涨**：单卡 1.2～2.2 GiB，原实现 3.5～57.9 GiB 后 OOM；N 越大省得越多，32k 时省 34.8 倍。
- **2 卡时原实现从 32k 起就 OOM，v2 到 64k 仍只要 3～4 GiB**：32k 时原实现要申请 53.3 GiB 的补零张量（TP 为 50.6 GiB），卡上只剩 31.8 GiB（TP 为 37.6 GiB），其余被通信缓冲区等占用。
- **CP 的 v2 显存约为单卡的两倍**：CP 路径里 rank 0 把梯度乘以 cp 时新建了一张全词表梯度，和规约无关。
- **CP 的耗时包含卡间 all-gather**：拆开测，64k 时 all-gather 约 1.9 ms，规约约 2.0 ms。
- **bf16 梯度路径也更快**：单卡 `SM90EmbeddingOp.forward` 传进来的是 bf16 梯度，也走同一个 kernel，256 时 1.03 → 0.73 ms，16k 时快 35 倍（36.42 → 1.03 ms），见下面「受影响的调用方」。

## 结果验证

逐 bit 比较，不用 `torch.equal`，因为它分不出 −0.0 和 +0.0：

- 13 个边界情况（−0.0、NaN、inf、空输入、非法 id、hidden 不是 512 的倍数等）加 16k 真实数据：全部一致，每个都比较 fp32 / bf16 / fp16 输出和 bf16 梯度输入。
- 规约函数，单卡 256～32k：v2 和原实现全部一致（64k 时原实现 OOM，无法对照）。
- 真实 2 卡 CP=2 和 TP=2，256～64k：每张卡上 v2 都和原实现一致（原实现 OOM 的长度除外）；多卡合并后的最终梯度和单卡结果也一致。
- 回归测试（见「测试流程」）：31 个用例全部通过（单卡 29 个，真实 2 卡 2 个）。
- 只改了反向，前向不变，所以不影响训推一致。

**受影响的调用方**：从每个入口用真实的前向和反向跑一遍，原实现和 v2 各一次，比较 `weight.grad`（单卡 GPU 1）：

| 入口 | 梯度 | 原实现最长能跑 | 256 原实现 (ms) | 256 v2 (ms) | 16k 原实现 (ms) | 16k v2 (ms) | 16k 原实现 (GiB) | 16k v2 (GiB) | 逐 bit 一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SM90EmbeddingOp.forward_fp32` | fp32 | 16k | 2.25 | 0.74 | 55.60 | 1.10 | 25.40 | 1.41 | 7/7 个长度 |
| `SM90EmbeddingOp.forward`（bf16 输出） | bf16 | 32k | 1.03 | 0.73 | 36.42 | 1.03 | 12.27 | 1.41 | 8/8 个长度 |
| `canonical_embedding`（cuda-sm90） | fp32 | 16k | 2.53 | 1.00 | 56.52 | 2.05 | 25.77 | 2.04 | 7/7 个长度 |
| `canonical_embedding`（pytorch） | fp32 | 16k | 2.53 | 0.99 | 56.54 | 2.06 | 25.77 | 2.04 | 7/7 个长度 |
| `CPEmbedding`（cp=1） | fp32 | 16k | 2.30 | 0.76 | 56.00 | 1.52 | 25.65 | 1.91 | 7/7 个长度 |
| Ascend 后端 | — | — | — | — | — | — | — | — | 未测（这台机器没有 NPU） |

- 从真实入口调用时，卡上还有权重、前向输出、上游梯度等张量，所以原实现在 fp32 路径上 **32k 就 OOM**，比只测规约函数时更早。
- bf16 路径的补零张量是 bf16，只有 fp32 的一半，所以原实现能撑到 32k。
