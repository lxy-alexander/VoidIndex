---
title: LeetCode 知识点题单
description: 按知识点整理的 LeetCode 题单，分高频和中频两档、共 269 道题，每题给出中英文链接和一句话思路；114 道加粗的经典题附完整题解和带编号注释的 Python 代码，写法尽量参考灵神（灵茶山艾府）的题解。
pubDatetime: 2026-11-20T10:00:00Z
tags:
  - leetcode
  - algorithm
draft: false

---

这份题单按知识点分类，每个子类列出 3–15 道代表题。**加粗**的是最经典、应该优先刷的题，一共 114 道，每道都附了完整题解；其余的题给出一句话思路，方便刷完加粗题后自己练习。🔒 表示会员题。

每道题都有 CN（leetcode.cn）和 EN（leetcode.com）两个链接。代码统一用 Python，写法尽量参考灵神（灵茶山艾府）的题解，注释按 1、2、3… 标出步骤。代码里直接使用的 `List`、`Counter`、`deque`、`heappush`、`bisect_left`、`inf` 等名字，LeetCode 的 Python 环境已经默认导入。

## 目录

| 档位 | 题单 | 核心思路 |
|---|---|---|
| 🔴 高频 | [1. 滑动窗口与双指针](#1-滑动窗口与双指针) | 维护窗口 / 指针的不变量，每个元素进出各一次 |
| 🔴 高频 | [2. 二分](#2-二分) | 找到单调性：check(mid) 为真时答案在左边还是右边 |
| 🔴 高频 | [3. 单调栈](#3-单调栈) | 栈内保持单调，弹出时就找到了被弹元素的「下一个更大 / 更小」 |
| 🔴 高频 | [4. 网格图](#4-网格图) | 网格就是图：上下左右四个邻居，DFS 求连通块，BFS 求最短步数 |
| 🔴 高频 | [5. 图论](#5-图论) | 建图（邻接表）+ 遍历；有依赖用拓扑排序，有边权用 Dijkstra |
| 🔴 高频 | [6. 动态规划](#6-动态规划) | 定义状态 → 写转移 → 定边界；先写记忆化搜索，再翻译成递推 |
| 🔴 高频 | [7. 常用数据结构](#7-常用数据结构) | 用合适的结构把暴力的某一层循环降成 O(1) 或 O(log n) |
| 🔴 高频 | [8. 链表、树与回溯](#8-链表树与回溯) | 链表画图 + 哨兵；树用递归定义子问题；回溯 = 选择 → 递归 → 撤销 |
| 🔴 高频 | [9. 贪心](#9-贪心) | 每一步做当前最优的选择，再用交换论证说明不会更差 |
| 🟡 中频 | [10. 中频 · 滑动窗口](#10-中频--滑动窗口) | 三指针、分组循环 |
| 🟡 中频 | [11. 中频 · 二分](#11-中频--二分) | 最小化最大值、最大化最小值、第 K 小 |
| 🟡 中频 | [12. 中频 · 单调栈](#12-中频--单调栈) | 贡献法、最小字典序 |
| 🟡 中频 | [13. 中频 · 位运算](#13-中频--位运算) | 基础与性质 |
| 🟡 中频 | [14. 中频 · 图论](#14-中频--图论) | 最小生成树、Floyd、Bellman-Ford |
| 🟡 中频 | [15. 中频 · 动态规划](#15-中频--动态规划) | 划分型、区间、树形、状压 |
| 🟡 中频 | [16. 中频 · 常用数据结构](#16-中频--常用数据结构) | 差分、单调队列、树状数组 / 线段树 |
| 🟡 中频 | [17. 中频 · 贪心](#17-中频--贪心) | 字典序、反悔贪心 |
| 🟡 中频 | [18. 中频 · 字符串](#18-中频--字符串) | KMP |
| 🟡 中频 | [19. 中频 · 数学](#19-中频--数学) | 质数、最大公约数、快速幂 |

## 怎么刷

1. 先刷高频的加粗题（94 道），基本覆盖 HOT 100 的核心考点。每道题先自己想 10–15 分钟，再看思路和代码。
2. 每个子类刷完加粗题后，用同类的非加粗题检验是否真的掌握了套路：能独立写出来才算会。
3. 高频全部过完再刷中频。中频的题更依赖特定技巧（贡献法、反悔贪心、KMP、状压 DP 等），适合冲刺阶段查漏补缺。
4. 同一道题可能出现在多个子类里（比如 42 接雨水既是双指针也是单调栈），题解只写一次，其他位置的表格会链接过去。

<br>
<br>

## 1. 滑动窗口与双指针

维护窗口 / 指针的不变量，每个元素进出各一次。

<br>

### 定长窗口

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 643 | 子数组最大平均数 I | [CN](https://leetcode.cn/problems/maximum-average-subarray-i/) · [EN](https://leetcode.com/problems/maximum-average-subarray-i/) | 维护长度为 k 的窗口和，右进左出 |
| 1456 | 定长子串中元音的最大数目 | [CN](https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/) · [EN](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/) | 窗口内元音计数，进一个出一个 |
| **438** | **找到字符串中所有字母异位词** | [CN](https://leetcode.cn/problems/find-all-anagrams-in-a-string/) · [EN](https://leetcode.com/problems/find-all-anagrams-in-a-string/) | [题解 ↓](#438-找到字符串中所有字母异位词--find-all-anagrams-in-a-string) |
| **567** | **字符串的排列** | [CN](https://leetcode.cn/problems/permutation-in-string/) · [EN](https://leetcode.com/problems/permutation-in-string/) | [题解 ↓](#567-字符串的排列--permutation-in-string) |

<br>

#### 438. 找到字符串中所有字母异位词 / Find All Anagrams in a String

[CN](https://leetcode.cn/problems/find-all-anagrams-in-a-string/) · [EN](https://leetcode.com/problems/find-all-anagrams-in-a-string/) · Medium

**题意**：找出 s 中所有是 p 的字母异位词的子串，返回它们的起始下标。

**思路**：定长滑动窗口，窗口长度固定为 len(p)。右端点进入窗口后，如果窗口长度达到 len(p)，比较窗口的字母计数和 p 的字母计数，再让左端点离开。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 1. 统计 p 的字母个数
        cnt_p = Counter(p)
        cnt_s = Counter()
        ans = []
        for right, c in enumerate(s):
            # 2. 右端点字母进入窗口
            cnt_s[c] += 1
            left = right - len(p) + 1
            if left < 0:  # 窗口长度还不足 len(p)
                continue
            # 3. 窗口长度等于 len(p)，比较两个计数（Python 3.10+ 的 Counter 比较会忽略 0）
            if cnt_s == cnt_p:
                ans.append(left)
            # 4. 左端点字母离开窗口
            cnt_s[s[left]] -= 1
        return ans
```

**复杂度**：时间 O(|Σ|·n)，|Σ| = 26；空间 O(|Σ|)。

<br>

#### 567. 字符串的排列 / Permutation in String

[CN](https://leetcode.cn/problems/permutation-in-string/) · [EN](https://leetcode.com/problems/permutation-in-string/) · Medium

**题意**：判断 s2 是否包含 s1 的某个排列作为子串。

**思路**：和 438 完全一样：在 s2 上滑动长度为 len(s1) 的窗口，只要有一个窗口的字母计数等于 s1 的，就返回 True。

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 1. 统计 s1 的字母个数
        cnt1 = Counter(s1)
        cnt2 = Counter()
        for right, c in enumerate(s2):
            # 2. 右端点进入窗口
            cnt2[c] += 1
            left = right - len(s1) + 1
            if left < 0:
                continue
            # 3. 窗口长度等于 len(s1)，计数相同说明是一个排列
            if cnt1 == cnt2:
                return True
            # 4. 左端点离开窗口
            cnt2[s2[left]] -= 1
        return False
```

**复杂度**：时间 O(|Σ|·n)；空间 O(|Σ|)。

<br>

### 不定长窗口

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **3** | **无重复字符的最长子串** | [CN](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) · [EN](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | [题解 ↓](#3-无重复字符的最长子串--longest-substring-without-repeating-characters) |
| **209** | **长度最小的子数组** | [CN](https://leetcode.cn/problems/minimum-size-subarray-sum/) · [EN](https://leetcode.com/problems/minimum-size-subarray-sum/) | [题解 ↓](#209-长度最小的子数组--minimum-size-subarray-sum) |
| **76** | **最小覆盖子串** | [CN](https://leetcode.cn/problems/minimum-window-substring/) · [EN](https://leetcode.com/problems/minimum-window-substring/) | [题解 ↓](#76-最小覆盖子串--minimum-window-substring) |
| 904 | 水果成篮 | [CN](https://leetcode.cn/problems/fruit-into-baskets/) · [EN](https://leetcode.com/problems/fruit-into-baskets/) | 窗口内水果种类超过 2 就收缩左端 |
| 1004 | 最大连续1的个数 III | [CN](https://leetcode.cn/problems/max-consecutive-ones-iii/) · [EN](https://leetcode.com/problems/max-consecutive-ones-iii/) | 窗口内 0 的个数超过 k 就收缩左端 |
| 713 | 乘积小于 K 的子数组 | [CN](https://leetcode.cn/problems/subarray-product-less-than-k/) · [EN](https://leetcode.com/problems/subarray-product-less-than-k/) | 乘积 ≥ k 时收缩；每个右端点贡献 right-left+1 个子数组 |
| 424 | 替换后的最长重复字符 | [CN](https://leetcode.cn/problems/longest-repeating-character-replacement/) · [EN](https://leetcode.com/problems/longest-repeating-character-replacement/) | 窗口长度 - 窗口内最多字符的次数 ≤ k |

<br>

#### 3. 无重复字符的最长子串 / Longest Substring Without Repeating Characters

[CN](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) · [EN](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · Medium

**题意**：求字符串中不含重复字符的最长子串的长度。

**思路**：不定长滑动窗口：枚举右端点，把字符加入窗口；如果它出现了两次，就不断移出左端点直到它只出现一次。此时窗口无重复，用窗口长度更新答案。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        cnt = defaultdict(int)  # 窗口内每个字符的出现次数
        ans = left = 0
        for right, c in enumerate(s):
            # 1. 右端点字符进入窗口
            cnt[c] += 1
            # 2. 窗口内有重复，收缩左端点直到 c 只出现一次
            while cnt[c] > 1:
                cnt[s[left]] -= 1
                left += 1
            # 3. 此时窗口内没有重复字符，更新答案
            ans = max(ans, right - left + 1)
        return ans
```

**复杂度**：时间 O(n)；空间 O(|Σ|)。

<br>

#### 209. 长度最小的子数组 / Minimum Size Subarray Sum

[CN](https://leetcode.cn/problems/minimum-size-subarray-sum/) · [EN](https://leetcode.com/problems/minimum-size-subarray-sum/) · Medium

**题意**：正整数数组中，求和 ≥ target 的最短子数组长度，不存在返回 0。

**思路**：数组全是正数，窗口越长和越大，满足单调性。枚举右端点，只要窗口和 ≥ target，就用当前长度更新答案并收缩左端点。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        ans = n + 1
        s = left = 0
        for right, x in enumerate(nums):
            # 1. 右端点进入窗口
            s += x
            # 2. 窗口和满足要求时，记录答案并尽量缩短窗口
            while s >= target:
                ans = min(ans, right - left + 1)
                s -= nums[left]
                left += 1
        # 3. 没有满足要求的子数组时返回 0
        return ans if ans <= n else 0
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 76. 最小覆盖子串 / Minimum Window Substring

[CN](https://leetcode.cn/problems/minimum-window-substring/) · [EN](https://leetcode.com/problems/minimum-window-substring/) · Hard

**题意**：求 s 中涵盖 t 所有字符（包括重复次数）的最短子串。

**思路**：cnt[c] 表示窗口还缺多少个字母 c，less 表示还有几种字母没凑齐。右端点进入时 cnt 减一，某种字母减到 0 时 less 减一。less == 0 说明窗口已经覆盖 t，这时记录答案并收缩左端点。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # 1. cnt[c] = 窗口还缺几个 c（可以为负，表示多了）；less = 还缺几种字母
        cnt = Counter(t)
        less = len(cnt)
        ans_left, ans_right = -1, len(s)
        left = 0
        for right, c in enumerate(s):
            # 2. 右端点进入窗口
            cnt[c] -= 1
            if cnt[c] == 0:  # 字母 c 刚好凑齐
                less -= 1
            # 3. 窗口已经覆盖 t：更新答案，然后收缩左端点
            while less == 0:
                if right - left < ans_right - ans_left:
                    ans_left, ans_right = left, right
                x = s[left]
                if cnt[x] == 0:  # x 移出后就不够了
                    less += 1
                cnt[x] += 1
                left += 1
        # 4. 没找到时 ans_left 仍为 -1
        return "" if ans_left < 0 else s[ans_left: ans_right + 1]
```

**复杂度**：时间 O(m + n)；空间 O(|Σ|)。

<br>

### 相向双指针

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 167 | 两数之和 II - 输入有序数组 | [CN](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/) · [EN](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | 和太小左指针右移，和太大右指针左移 |
| **15** | **三数之和** | [CN](https://leetcode.cn/problems/3sum/) · [EN](https://leetcode.com/problems/3sum/) | [题解 ↓](#15-三数之和--3sum) |
| 16 | 最接近的三数之和 | [CN](https://leetcode.cn/problems/3sum-closest/) · [EN](https://leetcode.com/problems/3sum-closest/) | 排序 + 枚举第一个数 + 双指针，记录差值最小的和 |
| 18 | 四数之和 | [CN](https://leetcode.cn/problems/4sum/) · [EN](https://leetcode.com/problems/4sum/) | 两层枚举 + 双指针，注意去重与剪枝 |
| **11** | **盛最多水的容器** | [CN](https://leetcode.cn/problems/container-with-most-water/) · [EN](https://leetcode.com/problems/container-with-most-water/) | [题解 ↓](#11-盛最多水的容器--container-with-most-water) |
| **42** | **接雨水** | [CN](https://leetcode.cn/problems/trapping-rain-water/) · [EN](https://leetcode.com/problems/trapping-rain-water/) | [题解 ↓](#42-接雨水--trapping-rain-water) |
| 125 | 验证回文串 | [CN](https://leetcode.cn/problems/valid-palindrome/) · [EN](https://leetcode.com/problems/valid-palindrome/) | 跳过非字母数字字符，两端向中间比较 |

<br>

#### 15. 三数之和 / 3Sum

[CN](https://leetcode.cn/problems/3sum/) · [EN](https://leetcode.com/problems/3sum/) · Medium

**题意**：找出所有和为 0 且不重复的三元组。

**思路**：排序后枚举第一个数 x = nums[i]，剩下的问题是在 i 右边找两数之和为 -x，用相向双指针。去重：x 和前一个数相同就跳过；找到一组后，j 和 k 也要跳过相同的数。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 1. 排序，才能用相向双指针并方便去重
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n - 2):
            x = nums[i]
            # 2. 第一个数去重
            if i > 0 and x == nums[i - 1]:
                continue
            # 3. 剪枝：最小的三个数之和都 > 0，后面不可能有解
            if x + nums[i + 1] + nums[i + 2] > 0:
                break

            # 4. 相向双指针找两数之和 = -x
            j, k = i + 1, n - 1
            while j < k:
                s = x + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:
                    ans.append([x, nums[j], nums[k]])
                    # 5. 跳过重复的 nums[j] 和 nums[k]
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while k > j and nums[k] == nums[k + 1]:
                        k -= 1
        return ans
```

**复杂度**：时间 O(n²)；空间 O(1)（不计排序栈空间和答案）。

<br>

#### 11. 盛最多水的容器 / Container With Most Water

[CN](https://leetcode.cn/problems/container-with-most-water/) · [EN](https://leetcode.com/problems/container-with-most-water/) · Medium

**题意**：选两条竖线与 x 轴构成容器，求最多能装多少水。

**思路**：相向双指针。面积由较短的那根线决定，如果移动较长的那根，宽度变小、高度不会超过短线，面积只会更小；所以每次移动较短的那根。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        left, right = 0, len(height) - 1
        while left < right:
            # 1. 计算当前两根线围成的面积
            area = (right - left) * min(height[left], height[right])
            ans = max(ans, area)
            # 2. 移动较短的那根线（移动长的不可能更优）
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 42. 接雨水 / Trapping Rain Water

[CN](https://leetcode.cn/problems/trapping-rain-water/) · [EN](https://leetcode.com/problems/trapping-rain-water/) · Hard

**题意**：给定柱子高度，求下雨后能接多少雨水。

**思路**：竖着看：位置 i 的水位 = min(左边最高, 右边最高)。相向双指针：维护 pre_max 和 suf_max，哪边的最大值更小，哪边的水位就已经确定，计算后向中间移动。单调栈解法是横着一层层算：栈底到栈顶递减，遇到更高的柱子时，弹出的栈顶是坑底，左右两边较矮者决定这一层水的高度。

**解法一：相向双指针**

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = pre_max = suf_max = 0
        left, right = 0, len(height) - 1
        while left < right:
            # 1. 更新左边和右边的最大高度
            pre_max = max(pre_max, height[left])
            suf_max = max(suf_max, height[right])
            # 2. 较小的一边水位已经确定，计算这一格的水量并移动
            if pre_max < suf_max:
                ans += pre_max - height[left]
                left += 1
            else:
                ans += suf_max - height[right]
                right -= 1
        return ans
```

**解法二：单调栈**

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        st = []  # 存下标，对应高度单调递减
        for i, h in enumerate(height):
            # 1. 当前柱子比栈顶高，栈顶就是一个坑底
            while st and h >= height[st[-1]]:
                bottom_h = height[st.pop()]
                if not st:  # 左边没有柱子，接不住水
                    break
                left = st[-1]
                # 2. 这一层水的高度和宽度
                dh = min(height[left], h) - bottom_h
                ans += dh * (i - left - 1)
            # 3. 当前下标入栈
            st.append(i)
        return ans
```

**复杂度**：两种解法时间都是 O(n)；双指针空间 O(1)，单调栈空间 O(n)。

<br>

### 同向双指针

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **283** | **移动零** | [CN](https://leetcode.cn/problems/move-zeroes/) · [EN](https://leetcode.com/problems/move-zeroes/) | [题解 ↓](#283-移动零--move-zeroes) |
| 26 | 删除有序数组中的重复项 | [CN](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/) · [EN](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) | 慢指针指向下一个写入位置，遇到新值才写入 |
| 27 | 移除元素 | [CN](https://leetcode.cn/problems/remove-element/) · [EN](https://leetcode.com/problems/remove-element/) | 不等于 val 的元素依次写到慢指针处 |
| 88 | 合并两个有序数组 | [CN](https://leetcode.cn/problems/merge-sorted-array/) · [EN](https://leetcode.com/problems/merge-sorted-array/) | 从后往前合并，不会覆盖未处理的元素 |

<br>

#### 283. 移动零 / Move Zeroes

[CN](https://leetcode.cn/problems/move-zeroes/) · [EN](https://leetcode.com/problems/move-zeroes/) · Easy

**题意**：把所有 0 移到数组末尾，保持非零元素的相对顺序，原地操作。

**思路**：同向双指针：i0 指向下一个非零数应该放的位置。遍历到非零数就和 nums[i0] 交换，i0 后移。这样非零数保持原顺序，0 被换到后面。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        i0 = 0  # 1. 下一个非零数要放的位置
        for i in range(len(nums)):
            # 2. 遇到非零数，换到 i0 处
            if nums[i]:
                nums[i], nums[i0] = nums[i0], nums[i]
                i0 += 1
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>
<br>

## 2. 二分

找到单调性：check(mid) 为真时答案在左边还是右边。

<br>

### 基础查找

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **704** | **二分查找** | [CN](https://leetcode.cn/problems/binary-search/) · [EN](https://leetcode.com/problems/binary-search/) | [题解 ↓](#704-二分查找--binary-search) |
| 35 | 搜索插入位置 | [CN](https://leetcode.cn/problems/search-insert-position/) · [EN](https://leetcode.com/problems/search-insert-position/) | lower_bound：第一个 ≥ target 的位置 |
| **34** | **在排序数组中查找元素的第一个和最后一个位置** | [CN](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/) · [EN](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | [题解 ↓](#34-在排序数组中查找元素的第一个和最后一个位置--find-first-and-last-position-of-element-in-sorted-array) |
| 74 | 搜索二维矩阵 | [CN](https://leetcode.cn/problems/search-a-2d-matrix/) · [EN](https://leetcode.com/problems/search-a-2d-matrix/) | 把矩阵看成长度 m·n 的有序数组二分 |
| 240 | 搜索二维矩阵 II | [CN](https://leetcode.cn/problems/search-a-2d-matrix-ii/) · [EN](https://leetcode.com/problems/search-a-2d-matrix-ii/) | 从右上角出发：大了左移，小了下移 |
| 162 | 寻找峰值 | [CN](https://leetcode.cn/problems/find-peak-element/) · [EN](https://leetcode.com/problems/find-peak-element/) | 比较 nums[mid] 和 nums[mid+1]，往上坡方向收缩 |
| 69 | x 的平方根 | [CN](https://leetcode.cn/problems/sqrtx/) · [EN](https://leetcode.com/problems/sqrtx/) | 二分最大的 m 使 m² ≤ x |

<br>

#### 704. 二分查找 / Binary Search

[CN](https://leetcode.cn/problems/binary-search/) · [EN](https://leetcode.com/problems/binary-search/) · Easy

**题意**：在升序数组中查找 target，返回下标，不存在返回 -1。

**思路**：闭区间写法：[left, right] 是待查找的区间。循环不变量：left 左边都 < target，right 右边都 ≥ target。循环结束后 left 就是第一个 ≥ target 的位置。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 1. 闭区间 [left, right]
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            # 2. nums[mid] < target 说明答案在右边
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        # 3. left 是第一个 >= target 的位置，检查是否等于 target
        if left < len(nums) and nums[left] == target:
            return left
        return -1
```

**复杂度**：时间 O(log n)；空间 O(1)。

<br>

#### 34. 在排序数组中查找元素的第一个和最后一个位置 / Find First and Last Position of Element in Sorted Array

[CN](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/) · [EN](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) · Medium

**题意**：在非递减数组中找 target 的开始和结束位置，要求 O(log n)。

**思路**：写一个 lower_bound（第一个 ≥ target 的位置）。开始位置 = lower_bound(target)；结束位置 = lower_bound(target + 1) - 1。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 1. lower_bound：返回第一个 >= target 的下标，不存在返回 len(nums)
        def lower_bound(target: int) -> int:
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left

        # 2. 开始位置
        start = lower_bound(target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        # 3. 结束位置 = 第一个 >= target+1 的位置 - 1
        end = lower_bound(target + 1) - 1
        return [start, end]
```

**复杂度**：时间 O(log n)；空间 O(1)。

<br>

### 旋转数组

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **33** | **搜索旋转排序数组** | [CN](https://leetcode.cn/problems/search-in-rotated-sorted-array/) · [EN](https://leetcode.com/problems/search-in-rotated-sorted-array/) | [题解 ↓](#33-搜索旋转排序数组--search-in-rotated-sorted-array) |
| **153** | **寻找旋转排序数组中的最小值** | [CN](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/) · [EN](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) | [题解 ↓](#153-寻找旋转排序数组中的最小值--find-minimum-in-rotated-sorted-array) |
| 81 | 搜索旋转排序数组 II | [CN](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/) · [EN](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/) | 有重复元素：nums[mid] == nums[right] 时无法判断，right -= 1 |
| 154 | 寻找旋转排序数组中的最小值 II | [CN](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/) · [EN](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/) | 和 nums[right] 比较，相等时 right -= 1 |

<br>

#### 33. 搜索旋转排序数组 / Search in Rotated Sorted Array

[CN](https://leetcode.cn/problems/search-in-rotated-sorted-array/) · [EN](https://leetcode.com/problems/search-in-rotated-sorted-array/) · Medium

**题意**：在旋转过的升序数组（元素互不相同）中查找 target 的下标。

**思路**：两次二分：先用 153 的方法找到最小值下标 i，把数组分成两个有序段 [0, i-1] 和 [i, n-1]。target > nums[-1] 时只可能在第一段，否则在第二段，然后在那一段里做普通二分。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)

        # 1. 153 题：返回最小值的下标
        def find_min() -> int:
            left, right = 0, n - 2
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < nums[-1]:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        # 2. 在有序区间 [left, right] 中二分查找 target
        def lower_bound(left: int, right: int) -> int:
            lo, hi = left, right
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo if lo <= right and nums[lo] == target else -1

        i = find_min()
        # 3. target 比最后一个数大，只可能在第一段；否则在第二段
        if target > nums[-1]:
            return lower_bound(0, i - 1)
        return lower_bound(i, n - 1)
```

**复杂度**：时间 O(log n)；空间 O(1)。

<br>

#### 153. 寻找旋转排序数组中的最小值 / Find Minimum in Rotated Sorted Array

[CN](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/) · [EN](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) · Medium

**题意**：求旋转过的升序数组（元素互不相同）的最小值。

**思路**：和最后一个数 nums[-1] 比较：nums[mid] < nums[-1] 说明 mid 在第二段（或数组没旋转），最小值在 mid 或它左边；否则 mid 在第一段，最小值在 mid 右边。nums[-1] 一定不在第一段，所以二分范围是 [0, n-2]。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # 1. 闭区间 [0, n-2]；nums[n-1] 本身不用参与二分
        left, right = 0, len(nums) - 2
        while left <= right:
            mid = (left + right) // 2
            # 2. mid 在第二段，最小值在 mid 或左边
            if nums[mid] < nums[-1]:
                right = mid - 1
            # 3. mid 在第一段，最小值在右边
            else:
                left = mid + 1
        # 4. left 就是最小值的下标
        return nums[left]
```

**复杂度**：时间 O(log n)；空间 O(1)。

<br>

### 二分答案

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **875** | **爱吃香蕉的珂珂** | [CN](https://leetcode.cn/problems/koko-eating-bananas/) · [EN](https://leetcode.com/problems/koko-eating-bananas/) | [题解 ↓](#875-爱吃香蕉的珂珂--koko-eating-bananas) |
| 1011 | 在 D 天内送达包裹的能力 | [CN](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/) · [EN](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) | 二分载重，贪心装船算需要几天 |
| 1482 | 制作 m 束花所需的最少天数 | [CN](https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/) · [EN](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/) | 二分天数，数连续开花段能做几束 |
| 2187 | 完成旅途的最少时间 | [CN](https://leetcode.cn/problems/minimum-time-to-complete-trips/) · [EN](https://leetcode.com/problems/minimum-time-to-complete-trips/) | 二分时间 t，检查 Σ t // time[i] ≥ totalTrips |

<br>

#### 875. 爱吃香蕉的珂珂 / Koko Eating Bananas

[CN](https://leetcode.cn/problems/koko-eating-bananas/) · [EN](https://leetcode.com/problems/koko-eating-bananas/) · Medium

**题意**：珂珂每小时最多吃 k 根香蕉（一堆吃完这一小时就不再吃），求 h 小时内吃完的最小 k。

**思路**：二分答案：速度 k 越大，吃完需要的小时数越少，满足单调性。check(k) = Σ ceil(p / k) ≤ h，二分找最小的满足 check 的 k。

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 1. 速度为 k 时能否在 h 小时内吃完；ceil(p/k) = (p-1)//k + 1
        def check(k: int) -> bool:
            return sum((p - 1) // k + 1 for p in piles) <= h

        # 2. 速度范围 [1, max(piles)]，找第一个满足 check 的 k
        left, right = 1, max(piles)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
```

**复杂度**：时间 O(n log U)，U = max(piles)；空间 O(1)。

<br>

### 困难经典

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **4** | **寻找两个正序数组的中位数** | [CN](https://leetcode.cn/problems/median-of-two-sorted-arrays/) · [EN](https://leetcode.com/problems/median-of-two-sorted-arrays/) | [题解 ↓](#4-寻找两个正序数组的中位数--median-of-two-sorted-arrays) |

<br>

#### 4. 寻找两个正序数组的中位数 / Median of Two Sorted Arrays

[CN](https://leetcode.cn/problems/median-of-two-sorted-arrays/) · [EN](https://leetcode.com/problems/median-of-two-sorted-arrays/) · Hard

**题意**：求两个正序数组合并后的中位数，要求 O(log(m+n))。

**思路**：在较短的数组 a 上二分分割点 i（a 的前 i 个放左半边），b 的分割点 j = (m+n+1)//2 - i 随之确定。合法分割要求 a[i-1] ≤ b[j] 且 b[j-1] ≤ a[i]。a[i-1] > b[j] 说明 i 太大，往左找；否则往右找。

```python
class Solution:
    def findMedianSortedArrays(self, a: List[int], b: List[int]) -> float:
        # 1. 保证 a 更短，在 a 上二分
        if len(a) > len(b):
            a, b = b, a
        m, n = len(a), len(b)
        half = (m + n + 1) // 2  # 左半边的元素个数
        left, right = 0, m
        while left <= right:
            # 2. a 取前 i 个，b 取前 j 个，组成左半边
            i = (left + right) // 2
            j = half - i
            a_left = a[i - 1] if i > 0 else -inf
            a_right = a[i] if i < m else inf
            b_left = b[j - 1] if j > 0 else -inf
            b_right = b[j] if j < n else inf
            # 3. 合法分割：左半边的最大值 <= 右半边的最小值
            if a_left <= b_right and b_left <= a_right:
                max_left = max(a_left, b_left)
                if (m + n) % 2:
                    return max_left
                return (max_left + min(a_right, b_right)) / 2
            # 4. a 的左半边取多了，i 往左；否则往右
            if a_left > b_right:
                right = i - 1
            else:
                left = i + 1
```

**复杂度**：时间 O(log min(m, n))；空间 O(1)。

<br>
<br>

## 3. 单调栈

栈内保持单调，弹出时就找到了被弹元素的「下一个更大 / 更小」。

<br>

### 基础

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **739** | **每日温度** | [CN](https://leetcode.cn/problems/daily-temperatures/) · [EN](https://leetcode.com/problems/daily-temperatures/) | [题解 ↓](#739-每日温度--daily-temperatures) |
| 496 | 下一个更大元素 I | [CN](https://leetcode.cn/problems/next-greater-element-i/) · [EN](https://leetcode.com/problems/next-greater-element-i/) | 对 nums2 用单调栈求下一个更大元素，存进哈希表再查询 |
| **503** | **下一个更大元素 II** | [CN](https://leetcode.cn/problems/next-greater-element-ii/) · [EN](https://leetcode.com/problems/next-greater-element-ii/) | [题解 ↓](#503-下一个更大元素-ii--next-greater-element-ii) |
| 901 | 股票价格跨度 | [CN](https://leetcode.cn/problems/online-stock-span/) · [EN](https://leetcode.com/problems/online-stock-span/) | 单调递减栈存 (价格, 下标)，弹出所有 ≤ 当前价格的 |

<br>

#### 739. 每日温度 / Daily Temperatures

[CN](https://leetcode.cn/problems/daily-temperatures/) · [EN](https://leetcode.com/problems/daily-temperatures/) · Medium

**题意**：对每一天，求要再等几天才会有更高的温度，没有则为 0。

**思路**：从左往右遍历，栈里存还没找到更高温度的下标（对应温度单调递减）。当前温度比栈顶高时，栈顶的答案就是当前下标，弹出并记录。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        st = []  # 1. 存下标，对应温度单调递减（都在等待更高的温度）
        for i, t in enumerate(temperatures):
            # 2. 当前温度比栈顶高，栈顶找到了答案
            while st and t > temperatures[st[-1]]:
                j = st.pop()
                ans[j] = i - j
            # 3. 当前下标入栈，等待它的更高温度
            st.append(i)
        return ans
```

**复杂度**：时间 O(n)，每个下标最多入栈出栈各一次；空间 O(n)。

<br>

#### 503. 下一个更大元素 II / Next Greater Element II

[CN](https://leetcode.cn/problems/next-greater-element-ii/) · [EN](https://leetcode.com/problems/next-greater-element-ii/) · Medium

**题意**：循环数组中，求每个元素的下一个更大元素，没有则为 -1。

**思路**：循环数组：把数组遍历两遍（下标 i % n）。第二遍只负责给第一遍留在栈里的元素找答案，不再入栈。

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1] * n
        st = []  # 存下标，对应值单调递减
        # 1. 遍历两遍，模拟循环数组
        for i in range(2 * n):
            x = nums[i % n]
            # 2. x 比栈顶大，栈顶的下一个更大元素就是 x
            while st and x > nums[st[-1]]:
                ans[st.pop()] = x
            # 3. 只在第一遍入栈
            if i < n:
                st.append(i)
        return ans
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

### 矩形面积

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **84** | **柱状图中最大的矩形** | [CN](https://leetcode.cn/problems/largest-rectangle-in-histogram/) · [EN](https://leetcode.com/problems/largest-rectangle-in-histogram/) | [题解 ↓](#84-柱状图中最大的矩形--largest-rectangle-in-histogram) |
| **85** | **最大矩形** | [CN](https://leetcode.cn/problems/maximal-rectangle/) · [EN](https://leetcode.com/problems/maximal-rectangle/) | [题解 ↓](#85-最大矩形--maximal-rectangle) |
| 42 | 接雨水（单调栈解法） | [CN](https://leetcode.cn/problems/trapping-rain-water/) · [EN](https://leetcode.com/problems/trapping-rain-water/) | 横着算：弹出栈顶作为坑底，左右两边较矮者决定水位，[题解](#42-接雨水--trapping-rain-water) |

<br>

#### 84. 柱状图中最大的矩形 / Largest Rectangle in Histogram

[CN](https://leetcode.cn/problems/largest-rectangle-in-histogram/) · [EN](https://leetcode.com/problems/largest-rectangle-in-histogram/) · Hard

**题意**：求柱状图中能勾勒出的最大矩形面积。

**思路**：枚举每根柱子作为矩形的高 h，矩形能向左右延伸到第一根比它矮的柱子之前。用单调栈求出每根柱子左边、右边第一个更矮的位置 left[i] 和 right[i]，面积 = h × (right[i] - left[i] - 1)。

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        # 1. left[i]：左边第一个比 heights[i] 矮的下标，没有则为 -1
        left = [-1] * n
        st = []
        for i, h in enumerate(heights):
            while st and heights[st[-1]] >= h:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        # 2. right[i]：右边第一个比 heights[i] 矮的下标，没有则为 n
        right = [n] * n
        st.clear()
        for i in range(n - 1, -1, -1):
            h = heights[i]
            while st and heights[st[-1]] >= h:
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        # 3. 以每根柱子为高，计算最大面积
        return max(h * (r - l - 1) for h, l, r in zip(heights, left, right))
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 85. 最大矩形 / Maximal Rectangle

[CN](https://leetcode.cn/problems/maximal-rectangle/) · [EN](https://leetcode.com/problems/maximal-rectangle/) · Hard

**题意**：在 01 矩阵中找只包含 1 的最大矩形面积。

**思路**：把每一行当作底，heights[j] 表示从这一行往上连续 '1' 的个数，就变成了 84 题。逐行更新 heights，对每一行求一次柱状图最大矩形。

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # 1. 84 题：柱状图中最大的矩形（这里用一次遍历的写法）
        def largest(heights: List[int]) -> int:
            heights = heights + [0]  # 末尾放 0，保证最后栈被清空
            st = [-1]  # 栈底哨兵
            ans = 0
            for i, h in enumerate(heights):
                while len(st) > 1 and heights[st[-1]] >= h:
                    height = heights[st.pop()]
                    ans = max(ans, height * (i - st[-1] - 1))
                st.append(i)
            return ans

        n = len(matrix[0])
        heights = [0] * n
        ans = 0
        for row in matrix:
            # 2. 更新以当前行为底的柱子高度
            for j, c in enumerate(row):
                heights[j] = heights[j] + 1 if c == '1' else 0
            # 3. 对这一行求最大矩形
            ans = max(ans, largest(heights))
        return ans
```

**复杂度**：时间 O(mn)；空间 O(n)。

<br>
<br>

## 4. 网格图

网格就是图：上下左右四个邻居，DFS 求连通块，BFS 求最短步数。

<br>

### DFS

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **200** | **岛屿数量** | [CN](https://leetcode.cn/problems/number-of-islands/) · [EN](https://leetcode.com/problems/number-of-islands/) | [题解 ↓](#200-岛屿数量--number-of-islands) |
| **695** | **岛屿的最大面积** | [CN](https://leetcode.cn/problems/max-area-of-island/) · [EN](https://leetcode.com/problems/max-area-of-island/) | [题解 ↓](#695-岛屿的最大面积--max-area-of-island) |
| 463 | 岛屿的周长 | [CN](https://leetcode.cn/problems/island-perimeter/) · [EN](https://leetcode.com/problems/island-perimeter/) | 每块陆地贡献 4 条边，每对相邻陆地减 2，[题解 ↓](#463-岛屿的周长--island-perimeter) |
| 130 | 被围绕的区域 | [CN](https://leetcode.cn/problems/surrounded-regions/) · [EN](https://leetcode.com/problems/surrounded-regions/) | 从边界上的 O 出发 DFS 标记，剩下没标记的 O 改成 X，[题解 ↓](#130-被围绕的区域--surrounded-regions) |
| 417 | 太平洋大西洋水流问题 | [CN](https://leetcode.cn/problems/pacific-atlantic-water-flow/) · [EN](https://leetcode.com/problems/pacific-atlantic-water-flow/) | 从两个大洋的边界反向 DFS（往高处走），取交集 |
| **79** | **单词搜索** | [CN](https://leetcode.cn/problems/word-search/) · [EN](https://leetcode.com/problems/word-search/) | [题解 ↓](#79-单词搜索--word-search) |

<br>

#### 79. 单词搜索 / Word Search

[CN](https://leetcode.cn/problems/word-search/) · [EN](https://leetcode.com/problems/word-search/) · Medium

**题意**：判断单词能否由网格中上下左右相邻、且不重复使用的格子拼出。

**思路**：从每个格子出发回溯：当前格子要等于 word[k]，然后标记为已访问，向四个方向找 word[k+1]，回溯时恢复。两个优化：网格里某个字母不够用就直接返回 False；如果 word 末尾字母在网格里更少，就把 word 反转，从稀有的一端开始搜，能大幅减少分支。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # 1. 优化一：网格中字母个数不够，直接返回 False
        cnt = Counter(c for row in board for c in row)
        if not cnt >= Counter(word):
            return False
        # 2. 优化二：从出现次数更少的一端开始搜
        if cnt[word[-1]] < cnt[word[0]]:
            word = word[::-1]

        m, n = len(board), len(board[0])

        # 3. 从 (i, j) 出发能否匹配 word[k:]
        def dfs(i: int, j: int, k: int) -> bool:
            if board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            board[i][j] = ''  # 标记为已访问
            for x, y in (i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j):
                if 0 <= x < m and 0 <= y < n and dfs(x, y, k + 1):
                    return True
            board[i][j] = word[k]  # 4. 回溯：恢复现场
            return False

        return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

**复杂度**：时间 O(mn·3^k)，k = len(word)；空间 O(k)。

<br>

### BFS

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **994** | **腐烂的橘子** | [CN](https://leetcode.cn/problems/rotting-oranges/) · [EN](https://leetcode.com/problems/rotting-oranges/) | [题解 ↓](#994-腐烂的橘子--rotting-oranges) |
| 542 | 01 矩阵 | [CN](https://leetcode.cn/problems/01-matrix/) · [EN](https://leetcode.com/problems/01-matrix/) | 多源 BFS：所有 0 同时入队，[题解 ↓](#542-01-矩阵--01-matrix) |
| 1091 | 二进制矩阵中的最短路径 | [CN](https://leetcode.cn/problems/shortest-path-in-binary-matrix/) · [EN](https://leetcode.com/problems/shortest-path-in-binary-matrix/) | 8 个方向的 BFS，[题解 ↓](#1091-二进制矩阵中的最短路径--shortest-path-in-binary-matrix) |
| 934 | 最短的桥 | [CN](https://leetcode.cn/problems/shortest-bridge/) · [EN](https://leetcode.com/problems/shortest-bridge/) | DFS 把第一座岛全部入队，再多源 BFS 扩展到第二座岛 |

<br>
<br>

## 5. 图论

看题目里有没有**「东西之间的关系」**。只要能说清楚「谁和谁有关系」，就可以把东西当成节点、关系当成边，用图来做。做法分三步：先找出节点和边，再按题目在问什么归到下面 8 类中的一类，最后套这一类的模板。

<br>

### 怎么看出是图

| 类型 | 信号 | 节点 | 边 | 例题 |
|---|---|---|---|---|
| 明显 | 出现「节点、边、连接、相邻、邻居」 | 题目给的节点 | 题目给的边 | 797 所有可能的路径 |
| 明显 | 输入是边列表 `edges = [[0,1],[1,2]]` | 编号 | 每一对 `[u, v]` | 1971 寻找图中是否存在路径 |
| 明显 | 输入是邻接表 `graph[i]` 或邻接矩阵 `isConnected[i][j]` | 下标 i | `graph[i]` 里的元素 / 值为 1 的位置 | 547 省份数量、841 钥匙和房间 |
| 隐藏 | **二维网格** `grid[i][j]`，能上下左右走 | 每个格子 | 相邻格子 | 200 岛屿数量、994 腐烂的橘子 |
| 隐藏 | **依赖关系**：「学 A 之前必须先学 B」 | 每个任务 | 依赖关系（有向） | 207 课程表 |
| 隐藏 | **一步变换**：「每次只能改一个字母或一位数字」 | 每个状态 | 一步能变过去 | 127 单词接龙、752 打开转盘锁 |
| 隐藏 | **传递关系**：「a 和 b 是朋友，b 和 c 是朋友」 | 每个人 | 直接关系 | 547 省份数量 |
| 隐藏 | **比例或换算**：a/b = 2，b/c = 3 | 每个变量 | 带权重的边 | 399 除法求值 |
| 隐藏 | **分组或合并**：「把有关系的分到同一组」 | 每个元素 | 有关系就连边 | 721 账户合并 |

<br>

### 8 类题型总览

按**「题目在问什么」**分成 8 类。每一类都有固定的模板，认出类别就知道该怎么写。

| # | 类别 | 题目在问什么（关键词） | 方法 | 模板 | 面试频率 |
|---|---|---|---|---|---|
| 1 | [连通块](#1-连通块) | 有几组？能不能到？（几个、多少块、合并、能否访问所有） | DFS / BFS / 并查集 | A、F | ★★★ |
| 2 | [网格遍历](#2-网格遍历) | 二维网格里的岛屿、区域 | DFS / BFS（四个方向） | A | ★★★ |
| 3 | [无权最短路](#3-无权最短路bfs) | 最少几步？（最短、最少、最小步数） | BFS | C | ★★★ |
| 4 | [拓扑排序](#4-拓扑排序) | 按什么顺序？有没有循环依赖？（先后、依赖、能否完成） | Kahn（BFS + 入度）/ DFS 后序 | D、E | ★★★ |
| 5 | [路径枚举](#5-路径枚举dfs--回溯) | 列出所有路径或方案 | DFS + 回溯 | B | ★★ |
| 6 | [带权最短路](#6-带权最短路dijkstra) | 边有权重，求最短距离 / 最小代价 | Dijkstra | G | ★★ |
| 7 | [二分图](#7-二分图) | 能不能分成两组？ | 染色（DFS / BFS） | H | ★ |
| 8 | [最小生成树](#8-最小生成树) | 用最小代价把所有点连起来 | Kruskal（排序 + 并查集）/ Prim | F | ★ |

拿到题按这个顺序判断类别（越靠上的特征越「特殊」，先排除）：

```
先找出节点和边，建邻接表
  ├─ 有先后依赖？              → 4 拓扑排序
  ├─ 边有不同权重？            → 6 Dijkstra
  ├─ 问最少几步？              → 3 BFS
  ├─ 要列出所有路径？          → 5 回溯
  ├─ 问能不能分成两组？        → 7 二分图
  ├─ 要用最小代价连通所有点？  → 8 最小生成树
  ├─ 输入是二维网格？          → 2 网格遍历
  └─ 其他（有几组、能不能到）  → 1 连通块
```

<br>

### 通用模板

题目给的输入经常不是现成的图，比如边列表 `[[0,1],[1,2]]`，或者要你自己找关系（127 单词接龙）。第一步先转成**邻接表**：每个节点对应一个「能直接走到的节点」列表。转好以后，取邻居统一写成 `for nxt in g[node]`，后面的模板都能直接套。

#### 建图

```python
from collections import defaultdict, deque

def build_graph(edges, directed=True):
    g = defaultdict(list)
    # 1. 每条边 u -> v 加进邻接表
    for u, v in edges:
        g[u].append(v)
        # 2. 无向图要加两个方向
        if not directed:
            g[v].append(u)
    return g
# 例：edges = [[0,1],[1,2]] → g = {0: [1], 1: [2]}
```

#### 模板 A：DFS + 全局 visited（能不能到、有几块）

```python
visited = set()

def dfs(node):
    # 1. 标记已访问，每个节点只访问一次
    visited.add(node)
    # 2. 递归访问所有没去过的邻居
    for nxt in graph[node]:
        if nxt not in visited:
            dfs(nxt)

# 3. 数连通块：每遇到一个没访问过的节点，就是一个新块
count = 0
for node in all_nodes:
    if node not in visited:
        count += 1
        dfs(node)
```

#### 模板 B：DFS + 回溯（所有路径）

```python
res, path, on_path = [], [src], {src}

def dfs(node):
    # 1. 到达终点，记录一条路径
    if node == dst:
        res.append(path[:])
        return
    for nxt in graph[node]:
        # 2. 不走回当前路径上的点（有向无环图可以省掉 on_path）
        if nxt in on_path:
            continue
        path.append(nxt); on_path.add(nxt)
        dfs(nxt)
        # 3. 回溯：撤销选择，换下一个邻居
        path.pop(); on_path.remove(nxt)

dfs(src)
```

和模板 A 的区别：A 的 `visited` 是全局的，一个点访问过就再也不访问；B 的 `on_path` 只记录当前这条路径，回溯时撤销，所以同一个点可以出现在不同的路径里。

#### 模板 C：BFS 最短路（最少步数）

```python
def bfs(src, dst):
    q = deque([(src, 0)])
    visited = {src}
    while q:
        # 1. 按距离从近到远取出节点
        node, step = q.popleft()
        if node == dst:
            return step
        for nxt in graph[node]:
            # 2. 入队时就标记，避免同一个点重复入队
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, step + 1))
    # 3. 走不到终点
    return -1
```

多源 BFS（994 腐烂的橘子、542 01 矩阵）：一开始就把所有起点一起入队，其余不变。

#### 模板 D：拓扑排序 Kahn（BFS + 入度）

```python
def topo_sort(n, edges):  # edges: u -> v，表示先 u 后 v
    # 1. 建图并统计入度
    g = defaultdict(list)
    indeg = [0] * n
    for u, v in edges:
        g[u].append(v)
        indeg[v] += 1
    # 2. 入度为 0 的点没有前置依赖，可以先做
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        # 3. 做完 u，后继的入度减一，减到 0 就能做了
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # 4. 长度不够 n，说明剩下的点在环里
    return order if len(order) == n else []
```

#### 模板 E：拓扑排序 DFS（后序反转）

```python
def topo_dfs(n):
    visited, order = set(), []

    def dfs(x):
        visited.add(x)
        for y in graph[x]:
            if y not in visited:
                dfs(y)
        # 1. 后序：x 的所有后继都处理完，才把 x 加进去
        order.append(x)

    # 2. 从每个没访问过的点出发
    for i in range(n):
        if i not in visited:
            dfs(i)
    # 3. 后序反转就是拓扑序
    return order[::-1]
```

这个模板默认图里没有环。要判断有没有环，用三色标记（见 [207 课程表](#207-课程表--course-schedule)）：DFS 时遇到「正在访问中」的点就是有环。

#### 模板 F：并查集

什么时候必须用并查集？当题目是「**边一条条加进来，每加一条就要判断**」的时候，并查集明显更好。比如 **684 冗余连接**：按顺序加边，找出第一条加进去就会成环的边。用 DFS 的话，每加一条边都要重新遍历一次图；用并查集，每条边只要做一次 `find` 就能判断。

```python
parent = list(range(n))

def find(x):
    # 1. 找根，顺便把路径上的点直接挂到根上（路径压缩）
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rx, ry = find(x), find(y)
    # 2. 根相同，说明本来就连通
    if rx == ry:
        return False
    # 3. 合并两个集合
    parent[rx] = ry
    return True
```

#### 模板 G：Dijkstra（带权最短路）

模板 C 的 BFS 能求最短路，前提是**每条边的长度都一样**（都算 1 步）。如果每条边的长度不一样，比如 A→B 要 5 分钟、A→C 要 1 分钟，BFS 先碰到的终点就不一定是最近的。Dijkstra 把 BFS 的普通队列换成**优先队列（最小堆）**，每次先取出当前距离最小的节点。要求边权非负。

```python
import heapq

def dijkstra(g, src):
    # g[u] = [(v, w), ...]，表示 u 到 v 的边权是 w
    dist = {src: 0}
    heap = [(0, src)]  # (到 src 的距离, 节点)
    while heap:
        # 1. 取出当前距离最小的节点
        d, u = heapq.heappop(heap)
        # 2. 过时的记录（u 已经被更短的距离更新过），跳过
        if d > dist.get(u, float('inf')):
            continue
        # 3. 用 u 去更新邻居
        for v, w in g[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

#### 模板 H：二分图染色

```python
def is_bipartite(n):
    color = [0] * n  # 0 未染色，1 和 -1 是两种颜色

    def dfs(x, c):
        color[x] = c
        for y in graph[x]:
            # 1. 邻居和自己同色，不是二分图
            if color[y] == c:
                return False
            # 2. 邻居没染色，染成相反的颜色继续
            if color[y] == 0 and not dfs(y, -c):
                return False
        return True

    # 3. 图可能不连通，每个连通块都要染
    return all(color[i] or dfs(i, 1) for i in range(n))
```

<br>

### 1. 连通块

**特征**：数有几组、判断两点是否连通、分组。模板 A 或 F。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| **547** | **省份数量** | [CN](https://leetcode.cn/problems/number-of-provinces/) · [EN](https://leetcode.com/problems/number-of-provinces/) | 中等 | 入门必做，邻接矩阵，[题解 ↓](#547-省份数量--number-of-provinces) |
| 1971 | 寻找图中是否存在路径 | [CN](https://leetcode.cn/problems/find-if-path-exists-in-graph/) · [EN](https://leetcode.com/problems/find-if-path-exists-in-graph/) | 简单 | 判断能不能到，[题解 ↓](#1971-寻找图中是否存在路径--find-if-path-exists-in-graph) |
| 841 | 钥匙和房间 | [CN](https://leetcode.cn/problems/keys-and-rooms/) · [EN](https://leetcode.com/problems/keys-and-rooms/) | 中等 | 能不能访问所有节点，[题解 ↓](#841-钥匙和房间--keys-and-rooms) |
| 133 | 克隆图 | [CN](https://leetcode.cn/problems/clone-graph/) · [EN](https://leetcode.com/problems/clone-graph/) | 中等 | 遍历 + 哈希表，[题解 ↓](#133-克隆图--clone-graph) |
| 399 | 除法求值 | [CN](https://leetcode.cn/problems/evaluate-division/) · [EN](https://leetcode.com/problems/evaluate-division/) | 中等 | 带权图 DFS，[题解 ↓](#399-除法求值--evaluate-division) |
| 684 | 冗余连接 | [CN](https://leetcode.cn/problems/redundant-connection/) · [EN](https://leetcode.com/problems/redundant-connection/) | 中等 | 并查集找环，[题解 ↓](#684-冗余连接--redundant-connection) |
| 721 | 账户合并 | [CN](https://leetcode.cn/problems/accounts-merge/) · [EN](https://leetcode.com/problems/accounts-merge/) | 中等 | 并查集分组，[题解 ↓](#721-账户合并--accounts-merge) |
| — | 按距离给点分组 | 非力扣原题 | 中等 | **Ambient.ai 可能考**，547 换了个外壳，[题解 ↓](#按距离给点分组非力扣原题) |

<br>

#### 547. 省份数量 / Number of Provinces

[CN](https://leetcode.cn/problems/number-of-provinces/) · [EN](https://leetcode.com/problems/number-of-provinces/) · Medium

**题意**：给定城市之间的邻接矩阵，求省份（连通块）数量。

**思路**：省份就是连通块。DFS 解法：遍历每个城市，没访问过就答案加一，并 DFS 标记它能到达的所有城市。并查集解法：相连的城市合并，最后数有几个根。

**解法一：DFS**

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        vis = [False] * n

        # 1. 标记 i 所在连通块的所有城市
        def dfs(i: int) -> None:
            vis[i] = True
            for j, conn in enumerate(isConnected[i]):
                if conn and not vis[j]:
                    dfs(j)

        ans = 0
        for i in range(n):
            # 2. 没访问过的城市属于一个新省份
            if not vis[i]:
                dfs(i)
                ans += 1
        return ans
```

**解法二：并查集**

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        fa = list(range(n))

        # 1. 查找根节点，路径压缩
        def find(x: int) -> int:
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        # 2. 合并相连的城市，每次成功合并连通块数减一
        ans = n
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j]:
                    fi, fj = find(i), find(j)
                    if fi != fj:
                        fa[fi] = fj
                        ans -= 1
        return ans
```

**复杂度**：时间 O(n²)；空间 O(n)。

<br>

#### 1971. 寻找图中是否存在路径 / Find if Path Exists in Graph

[CN](https://leetcode.cn/problems/find-if-path-exists-in-graph/) · [EN](https://leetcode.com/problems/find-if-path-exists-in-graph/) · Easy

**题意**：给一个有 n 个节点的无向图（边列表 `edges`），判断从 source 能否走到 destination。

**思路**：模板 A。先把边列表转成邻接表（无向图两个方向都要加），再从 source 出发遍历，看能否访问到 destination。n 最大 2×10⁵，递归可能爆栈，这里用栈写成迭代版。

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        # 1. 建图（无向图）
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        # 2. 从 source 出发 DFS（用栈代替递归）
        vis = [False] * n
        vis[source] = True
        st = [source]
        while st:
            x = st.pop()
            # 3. 走到终点
            if x == destination:
                return True
            for y in g[x]:
                if not vis[y]:
                    vis[y] = True
                    st.append(y)
        return False
```

**复杂度**：时间 O(n + m)；空间 O(n + m)。

<br>

#### 841. 钥匙和房间 / Keys and Rooms

[CN](https://leetcode.cn/problems/keys-and-rooms/) · [EN](https://leetcode.com/problems/keys-and-rooms/) · Medium

**题意**：n 个房间，只有 0 号房间没锁。`rooms[i]` 是房间 i 里的钥匙（能打开的房间号），判断能否进入所有房间。

**思路**：模板 A。房间是节点，钥匙是有向边，`rooms` 本身就是邻接表。从 0 号房间 DFS，最后看访问过的房间数是否等于 n。

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        vis = [False] * len(rooms)

        # 1. 进入房间 x，用里面的钥匙打开没去过的房间
        def dfs(x: int) -> None:
            vis[x] = True
            for y in rooms[x]:
                if not vis[y]:
                    dfs(y)

        dfs(0)
        # 2. 所有房间都去过了吗
        return all(vis)
```

**复杂度**：时间 O(n + m)，m 是钥匙总数；空间 O(n)。

<br>

#### 133. 克隆图 / Clone Graph

[CN](https://leetcode.cn/problems/clone-graph/) · [EN](https://leetcode.com/problems/clone-graph/) · Medium

**题意**：深拷贝一个无向连通图。

**思路**：DFS + 哈希表：哈希表记录「原节点 → 克隆节点」。访问一个节点时先创建克隆并登记，再递归克隆所有邻居。已经登记过的节点直接返回，避免死循环。

```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        clones = {}  # 1. 原节点 -> 克隆节点

        def dfs(node: 'Node') -> 'Node':
            # 2. 已经克隆过，直接返回
            if node in clones:
                return clones[node]
            # 3. 先创建并登记，再克隆邻居（邻居可能指回自己）
            copy = Node(node.val)
            clones[node] = copy
            copy.neighbors = [dfs(nb) for nb in node.neighbors]
            return copy

        return dfs(node) if node else None
```

**复杂度**：时间 O(n + m)；空间 O(n)。

<br>

#### 399. 除法求值 / Evaluate Division

[CN](https://leetcode.cn/problems/evaluate-division/) · [EN](https://leetcode.com/problems/evaluate-division/) · Medium

**题意**：已知若干等式 `a / b = k`，回答一组询问 `c / d = ?`，无法确定的返回 -1.0。

**思路**：变量是节点，`a / b = k` 是两条带权边：a→b 权重 k，b→a 权重 1/k。`c / d` 就是从 c 走到 d 的路径上所有边权的乘积。每个询问从 c 出发 DFS，沿途累乘；变量没出现过或者走不到 d，答案是 -1.0。

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 1. 建带权图：a / b = k → a->b 权 k，b->a 权 1/k
        g = defaultdict(list)
        for (a, b), k in zip(equations, values):
            g[a].append((b, k))
            g[b].append((a, 1 / k))

        # 2. 从 x 出发找 target，返回路径上的乘积；找不到返回 -1
        def dfs(x: str, target: str, vis: Set[str]) -> float:
            if x == target:
                return 1.0
            vis.add(x)
            for y, k in g[x]:
                if y not in vis:
                    res = dfs(y, target, vis)
                    if res >= 0:
                        return res * k
            return -1.0

        # 3. 每个询问单独搜一次；没出现过的变量直接 -1
        return [dfs(c, d, set()) if c in g and d in g else -1.0 for c, d in queries]
```

**复杂度**：时间 O(q·(n + m))，q 是询问数；空间 O(n + m)。询问很多时可以改用带权并查集。

<br>

#### 684. 冗余连接 / Redundant Connection

[CN](https://leetcode.cn/problems/redundant-connection/) · [EN](https://leetcode.com/problems/redundant-connection/) · Medium

**题意**：在树上多加了一条边变成有环图，找出可以删除的那条边（有多个答案时返回最后出现的）。

**思路**：按顺序加边，用并查集维护连通性。如果一条边的两个端点已经连通，加上它就会形成环，它就是答案（题目要求返回最后出现的那条，而形成环的第一条边就是环上最后加入的）。

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        fa = list(range(n + 1))

        # 1. 查找根，路径压缩
        def find(x: int) -> int:
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        for x, y in edges:
            fx, fy = find(x), find(y)
            # 2. 已经连通，这条边多余
            if fx == fy:
                return [x, y]
            # 3. 合并
            fa[fx] = fy
```

**复杂度**：时间 O(n α(n))；空间 O(n)。

<br>

#### 721. 账户合并 / Accounts Merge

[CN](https://leetcode.cn/problems/accounts-merge/) · [EN](https://leetcode.com/problems/accounts-merge/) · Medium

**题意**：`accounts[i] = [名字, 邮箱1, 邮箱2, ...]`。两个账户只要有一个相同的邮箱，就属于同一个人。合并同一个人的所有账户，返回 `[名字, 排好序的邮箱...]`。

**思路**：模板 F。账户是节点，「共用一个邮箱」是边。遍历邮箱时记录它第一次出现在哪个账户，再遇到时就把两个账户合并。最后按根把邮箱分组、排序，名字取根账户的名字。

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        n = len(accounts)
        fa = list(range(n))

        def find(x: int) -> int:
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        # 1. 同一个邮箱出现在两个账户里，就合并这两个账户
        owner = {}  # 邮箱 -> 第一次出现的账户下标
        for i, acc in enumerate(accounts):
            for email in acc[1:]:
                if email in owner:
                    fa[find(i)] = find(owner[email])
                else:
                    owner[email] = i

        # 2. 按根账户把邮箱分组
        groups = defaultdict(list)
        for email, i in owner.items():
            groups[find(i)].append(email)

        # 3. 名字 + 排序后的邮箱
        return [[accounts[root][0]] + sorted(emails) for root, emails in groups.items()]
```

**复杂度**：时间 O(L log L)，L 是邮箱总数（排序占主要部分）；空间 O(L)。

<br>

#### 按距离给点分组（非力扣原题）

**题意**：给平面上 n 个点和距离阈值 d。两个点的距离 ≤ d 就属于同一组，并且分组可以传递（A 和 B 同组、B 和 C 同组，那么 A、B、C 都在一组）。返回分组结果（或组数）。

**思路**：就是 547 省份数量换了个外壳：547 的「直接相连」由邻接矩阵给出，这里要自己算，两点距离 ≤ d 就连一条边。枚举所有点对，用并查集合并，最后按根分组。比较时用距离的平方，避免开根号带来的浮点误差。

```python
def group_points(points: List[Tuple[int, int]], d: int) -> List[List[int]]:
    n = len(points)
    # 1. 建邻接矩阵，只算上三角，两个方向都填
    graph = [[False] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= d * d:
                graph[i][j] = graph[j][i] = True

    vis = [False] * n

    # 2. 把 i 所在组的点都放进 group
    def dfs(i: int, group: List[int]) -> None:
        vis[i] = True
        group.append(i)
        for j in range(n):
            if graph[i][j] and not vis[j]:
                dfs(j, group)

    # 3. 每个没访问过的点是新一组的起点
    ans = []
    for i in range(n):
        if not vis[i]:
            group = []
            dfs(i, group)
            ans.append(group)
    return ans


def group_points(points: List[Tuple[int, int]], d: int) -> List[List[int]]:
    n = len(points)
    vis = [False] * n
    groups = []
    for s in range(n):
        # 1. 已经分过组的点跳过
        if vis[s]:
            continue
        # 2. 从 s 出发 BFS，能走到的点都和 s 同组
        vis[s] = True
        q = deque([s])
        group = []
        while q:
            i = q.popleft()
            group.append(i)
            x1, y1 = points[i]
            # 3. 邻居不用提前建图：没访问过、且距离 <= d 的点就是邻居
            for j in range(n):
                if not vis[j]:
                    x2, y2 = points[j]
                    if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= d * d:
                        vis[j] = True  # 入队时就标记
                        q.append(j)
        # 4. 这一轮 BFS 访问到的点就是一组
        groups.append(group)
    return groups
```

**复杂度**：时间 O(n² α(n))；空间 O(n)。点很多时可以按 d 划分网格，只比较相邻格子里的点。

<br>

### 2. 网格遍历

**特征**：输入是 `grid[i][j]`，节点是格子，邻居是上下左右。模板 A，邻居写成四个方向。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| 733 | 图像渲染 | [CN](https://leetcode.cn/problems/flood-fill/) · [EN](https://leetcode.com/problems/flood-fill/) | 简单 | 最简单的入门题，[题解 ↓](#733-图像渲染--flood-fill) |
| **200** | **岛屿数量** | [CN](https://leetcode.cn/problems/number-of-islands/) · [EN](https://leetcode.com/problems/number-of-islands/) | 中等 | 必做，[题解 ↓](#200-岛屿数量--number-of-islands) |
| 695 | 岛屿的最大面积 | [CN](https://leetcode.cn/problems/max-area-of-island/) · [EN](https://leetcode.com/problems/max-area-of-island/) | 中等 | 200 + 统计面积，[题解 ↓](#695-岛屿的最大面积--max-area-of-island) |
| 463 | 岛屿的周长 | [CN](https://leetcode.cn/problems/island-perimeter/) · [EN](https://leetcode.com/problems/island-perimeter/) | 简单 | 数边，[题解 ↓](#463-岛屿的周长--island-perimeter) |
| 1020 | 飞地的数量 | [CN](https://leetcode.cn/problems/number-of-enclaves/) · [EN](https://leetcode.com/problems/number-of-enclaves/) | 中等 | 从边界出发 DFS，[题解 ↓](#1020-飞地的数量--number-of-enclaves) |
| 130 | 被围绕的区域 | [CN](https://leetcode.cn/problems/surrounded-regions/) · [EN](https://leetcode.com/problems/surrounded-regions/) | 中等 | 从边界出发 DFS，[题解 ↓](#130-被围绕的区域--surrounded-regions) |

<br>

#### 733. 图像渲染 / Flood Fill

[CN](https://leetcode.cn/problems/flood-fill/) · [EN](https://leetcode.com/problems/flood-fill/) · Easy

**题意**：从像素 (sr, sc) 开始，把和它颜色相同、上下左右相连的所有像素都改成 color。

**思路**：模板 A 的网格版。记下起点原来的颜色 old，从起点 DFS，遇到颜色为 old 的格子就改色并继续。如果 old 和 color 一样，改完的格子颜色还是 old，会无限递归，所以直接返回。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        old = image[sr][sc]
        # 1. 颜色没变，不用改（否则会无限递归）
        if old == color:
            return image
        m, n = len(image), len(image[0])

        # 2. 改色并向四个方向扩散
        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or image[i][j] != old:
                return
            image[i][j] = color
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        dfs(sr, sc)
        return image
```

**复杂度**：时间 O(mn)；空间 O(mn)（递归栈）。

<br>

#### 200. 岛屿数量 / Number of Islands

[CN](https://leetcode.cn/problems/number-of-islands/) · [EN](https://leetcode.com/problems/number-of-islands/) · Medium

**题意**：求网格中岛屿（上下左右相连的 '1'）的数量。

**思路**：遍历网格，遇到没访问过的陆地就答案加一，并从它出发 DFS，把整座岛标记为已访问（改成 '2'）。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])

        # 1. 把 (i, j) 所在的岛全部标记为 '2'
        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            grid[i][j] = '2'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        ans = 0
        for i in range(m):
            for j in range(n):
                # 2. 找到一座新岛
                if grid[i][j] == '1':
                    dfs(i, j)
                    ans += 1
        return ans
```

**复杂度**：时间 O(mn)；空间 O(mn)（递归栈）。

<br>

#### 695. 岛屿的最大面积 / Max Area of Island

[CN](https://leetcode.cn/problems/max-area-of-island/) · [EN](https://leetcode.com/problems/max-area-of-island/) · Medium

**题意**：求网格中面积最大的岛屿的面积。

**思路**：和 200 一样 DFS 标记整座岛，DFS 返回这座岛的格子数，取最大值。

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        # 1. 返回 (i, j) 所在岛的面积，并把访问过的格子改成 0
        def dfs(i: int, j: int) -> int:
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
                return 0
            grid[i][j] = 0
            return 1 + dfs(i - 1, j) + dfs(i + 1, j) + dfs(i, j - 1) + dfs(i, j + 1)

        # 2. 对每个格子求一次，取最大
        return max(dfs(i, j) for i in range(m) for j in range(n))
```

**复杂度**：时间 O(mn)；空间 O(mn)。

<br>

#### 463. 岛屿的周长 / Island Perimeter

[CN](https://leetcode.cn/problems/island-perimeter/) · [EN](https://leetcode.com/problems/island-perimeter/) · Easy

**题意**：网格中只有一座岛（1 是陆地），岛内没有湖，求岛的周长。

**思路**：不用遍历，直接数边。每块陆地有 4 条边；两块陆地相邻时，它们之间的那条边两边各算了一次，但不属于周长，所以每对相邻陆地减 2。只看每块陆地的上边和左边，每对相邻陆地正好被数一次。

```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        ans = 0
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if x == 0:
                    continue
                # 1. 每块陆地贡献 4 条边
                ans += 4
                # 2. 和上方、左方的陆地共用一条边，两块各减 1
                if i > 0 and grid[i - 1][j]:
                    ans -= 2
                if j > 0 and grid[i][j - 1]:
                    ans -= 2
        return ans
```

**复杂度**：时间 O(mn)；空间 O(1)。

<br>

#### 1020. 飞地的数量 / Number of Enclaves

[CN](https://leetcode.cn/problems/number-of-enclaves/) · [EN](https://leetcode.com/problems/number-of-enclaves/) · Medium

**题意**：1 是陆地，0 是海洋。每次可以走到上下左右相邻的陆地，或者从网格边界走出去。求无论怎么走都走不出网格的陆地格子数。

**思路**：反过来想：能走出去的陆地，一定和边界上的某块陆地连通。从边界上的所有陆地出发 DFS，把能走到的陆地都改成 0，剩下的 1 就是飞地。

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        # 1. 把和 (i, j) 连通的陆地都改成 0
        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 0:
                return
            grid[i][j] = 0
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        # 2. 从四条边界出发
        for i in range(m):
            dfs(i, 0)
            dfs(i, n - 1)
        for j in range(n):
            dfs(0, j)
            dfs(m - 1, j)
        # 3. 剩下的陆地都走不出去
        return sum(grid[i][j] for i in range(m) for j in range(n))
```

**复杂度**：时间 O(mn)；空间 O(mn)。

<br>

#### 130. 被围绕的区域 / Surrounded Regions

[CN](https://leetcode.cn/problems/surrounded-regions/) · [EN](https://leetcode.com/problems/surrounded-regions/) · Medium

**题意**：矩阵由 'X' 和 'O' 组成，把所有被 'X' 完全围住的 'O' 区域改成 'X'。和边界连通的 'O' 不算被围住。原地修改。

**思路**：和 1020 一样从边界出发。先把和边界连通的 'O' 都标记成 '#'（不能改），再遍历整个矩阵：剩下的 'O' 都被围住了，改成 'X'；'#' 恢复成 'O'。

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        m, n = len(board), len(board[0])

        # 1. 把和 (i, j) 连通的 'O' 标记成 '#'
        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O':
                return
            board[i][j] = '#'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        # 2. 从四条边界上的 'O' 出发
        for i in range(m):
            dfs(i, 0)
            dfs(i, n - 1)
        for j in range(n):
            dfs(0, j)
            dfs(m - 1, j)

        # 3. 没被标记的 'O' 被围住了；'#' 恢复成 'O'
        for row in board:
            for j, c in enumerate(row):
                if c == 'O':
                    row[j] = 'X'
                elif c == '#':
                    row[j] = 'O'
```

**复杂度**：时间 O(mn)；空间 O(mn)。

<br>

### 3. 无权最短路（BFS）

**特征**：问最少几步、最短距离，每一步代价相同。模板 C。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| 1091 | 二进制矩阵中的最短路径 | [CN](https://leetcode.cn/problems/shortest-path-in-binary-matrix/) · [EN](https://leetcode.com/problems/shortest-path-in-binary-matrix/) | 中等 | BFS 模板题，[题解 ↓](#1091-二进制矩阵中的最短路径--shortest-path-in-binary-matrix) |
| **994** | **腐烂的橘子** | [CN](https://leetcode.cn/problems/rotting-oranges/) · [EN](https://leetcode.com/problems/rotting-oranges/) | 中等 | 多源 BFS，必做，[题解 ↓](#994-腐烂的橘子--rotting-oranges) |
| 542 | 01 矩阵 | [CN](https://leetcode.cn/problems/01-matrix/) · [EN](https://leetcode.com/problems/01-matrix/) | 中等 | 多源 BFS，[题解 ↓](#542-01-矩阵--01-matrix) |
| 127 | 单词接龙 | [CN](https://leetcode.cn/problems/word-ladder/) · [EN](https://leetcode.com/problems/word-ladder/) | 困难 | 隐式图，邻居要自己生成，[题解 ↓](#127-单词接龙--word-ladder) |
| 752 | 打开转盘锁 | [CN](https://leetcode.cn/problems/open-the-lock/) · [EN](https://leetcode.com/problems/open-the-lock/) | 中等 | 隐式图，[题解 ↓](#752-打开转盘锁--open-the-lock) |

<br>

#### 1091. 二进制矩阵中的最短路径 / Shortest Path in Binary Matrix

[CN](https://leetcode.cn/problems/shortest-path-in-binary-matrix/) · [EN](https://leetcode.com/problems/shortest-path-in-binary-matrix/) · Medium

**题意**：n×n 的 01 矩阵，只能走值为 0 的格子，每步可以走到 8 个方向的相邻格子。求从左上角到右下角最少经过几个格子，走不到返回 -1。

**思路**：模板 C。每一步代价相同，用 BFS 一层层往外扩，第一次到达右下角时的层数就是答案。访问过的格子直接改成 1，省掉 visited。起点或终点是 1 时直接返回 -1。

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # 1. 起点或终点被堵住
        if grid[0][0] or grid[n - 1][n - 1]:
            return -1
        grid[0][0] = 1  # 标记已访问
        q = [(0, 0)]
        step = 1  # 路径包含的格子数
        while q:
            tmp = q
            q = []
            for i, j in tmp:
                # 2. 到达终点
                if i == n - 1 and j == n - 1:
                    return step
                # 3. 枚举 8 个方向
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                            grid[x][y] = 1
                            q.append((x, y))
            step += 1
        return -1
```

**复杂度**：时间 O(n²)；空间 O(n²)。

<br>

#### 994. 腐烂的橘子 / Rotting Oranges

[CN](https://leetcode.cn/problems/rotting-oranges/) · [EN](https://leetcode.com/problems/rotting-oranges/) · Medium

**题意**：每分钟腐烂橘子会让相邻的新鲜橘子腐烂，求全部腐烂的最少分钟数，不可能返回 -1。

**思路**：多源 BFS：所有腐烂的橘子同时作为起点入队，每一轮把队列里的橘子周围的新鲜橘子变腐烂，轮数就是分钟数。最后还有新鲜橘子就返回 -1。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        # 1. 统计新鲜橘子个数，所有腐烂橘子作为 BFS 的起点
        fresh = 0
        q = []
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if x == 1:
                    fresh += 1
                elif x == 2:
                    q.append((i, j))

        ans = 0
        # 2. 一层一层扩散，每层代表一分钟
        while q and fresh:
            ans += 1
            tmp = q
            q = []
            for x, y in tmp:
                for i, j in (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1):
                    if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                        fresh -= 1
                        grid[i][j] = 2  # 3. 新鲜橘子变腐烂
                        q.append((i, j))
        # 4. 仍有新鲜橘子说明有橘子永远不会腐烂
        return -1 if fresh else ans
```

**复杂度**：时间 O(mn)；空间 O(mn)。

<br>

#### 542. 01 矩阵 / 01 Matrix

[CN](https://leetcode.cn/problems/01-matrix/) · [EN](https://leetcode.com/problems/01-matrix/) · Medium

**题意**：01 矩阵中，求每个格子到最近的 0 的距离（上下左右走一步算 1）。

**思路**：多源 BFS。如果从每个 1 出发找最近的 0，要做很多次 BFS；反过来，把所有 0 同时作为起点入队，一次 BFS 往外扩，每个格子第一次被访问时的层数就是它到最近 0 的距离。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        # 1. 所有 0 是起点，距离为 0；其余格子先设为 -1（未访问）
        dis = [[-1] * n for _ in range(m)]
        q = deque()
        for i, row in enumerate(mat):
            for j, x in enumerate(row):
                if x == 0:
                    dis[i][j] = 0
                    q.append((i, j))
        # 2. BFS：邻居的距离 = 当前距离 + 1
        while q:
            i, j = q.popleft()
            for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < m and 0 <= y < n and dis[x][y] < 0:
                    dis[x][y] = dis[i][j] + 1
                    q.append((x, y))
        return dis
```

**复杂度**：时间 O(mn)；空间 O(mn)。

<br>

#### 127. 单词接龙 / Word Ladder

[CN](https://leetcode.cn/problems/word-ladder/) · [EN](https://leetcode.com/problems/word-ladder/) · Hard

**题意**：每次只能改一个字母且中间单词必须在字典中，求 beginWord 变到 endWord 的最短序列长度。

**思路**：求最短转换次数，用 BFS。每个单词枚举每一位换成 26 个字母，如果在单词表里就是一个邻居；访问过的单词从集合里删掉，防止重复访问。

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        words = set(wordList)
        if endWord not in words:
            return 0
        # 1. BFS，step 是当前层的单词数（包含起点）
        q = [beginWord]
        words.discard(beginWord)
        step = 1
        while q:
            step += 1
            tmp = q
            q = []
            for word in tmp:
                # 2. 枚举每一位换成其他字母
                for i in range(len(word)):
                    for c in ascii_lowercase:
                        nxt = word[:i] + c + word[i + 1:]
                        if nxt not in words:
                            continue
                        if nxt == endWord:
                            return step
                        # 3. 删除表示已访问
                        words.remove(nxt)
                        q.append(nxt)
        return 0
```

**复杂度**：时间 O(n·L·26·L)，n 是单词数、L 是单词长度（拼接字符串需要 O(L)）；空间 O(n·L)。

<br>

#### 752. 打开转盘锁 / Open the Lock

[CN](https://leetcode.cn/problems/open-the-lock/) · [EN](https://leetcode.com/problems/open-the-lock/) · Medium

**题意**：4 位转盘锁从 "0000" 开始，每次把某一位拨上或拨下一格（9 和 0 相邻）。不能经过 deadends 里的状态，求变成 target 的最少次数，不可能返回 -1。

**思路**：隐式图 + 模板 C。每个状态是节点，拨一次能到的 8 个状态是邻居。BFS 从 "0000" 开始，第一次碰到 target 时的层数就是答案。deadends 直接放进 visited，就不会被走到。

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        vis = set(deadends)
        # 1. 起点就是死亡数字
        if '0000' in vis:
            return -1
        vis.add('0000')
        q = ['0000']
        step = 0
        while q:
            tmp = q
            q = []
            for s in tmp:
                if s == target:
                    return step
                # 2. 每一位拨上或拨下，得到 8 个邻居
                for i in range(4):
                    d = int(s[i])
                    for nd in (d + 1) % 10, (d - 1) % 10:
                        nxt = s[:i] + str(nd) + s[i + 1:]
                        if nxt not in vis:
                            vis.add(nxt)
                            q.append(nxt)
            step += 1
        return -1
```

**复杂度**：时间 O(10⁴ · 8 · 4)，状态最多 10⁴ 个；空间 O(10⁴)。

<br>

### 4. 拓扑排序

**特征**：有先后依赖，问能不能完成、按什么顺序、有没有环。模板 D 或 E。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| **207** | **课程表** | [CN](https://leetcode.cn/problems/course-schedule/) · [EN](https://leetcode.com/problems/course-schedule/) | 中等 | 必做，判断有没有环，[题解 ↓](#207-课程表--course-schedule) |
| 210 | 课程表 II | [CN](https://leetcode.cn/problems/course-schedule-ii/) · [EN](https://leetcode.com/problems/course-schedule-ii/) | 中等 | 输出顺序，[题解 ↓](#210-课程表-ii--course-schedule-ii) |
| 802 | 找到最终的安全状态 | [CN](https://leetcode.cn/problems/find-eventual-safe-states/) · [EN](https://leetcode.com/problems/find-eventual-safe-states/) | 中等 | 反向拓扑 / 三色标记，[题解 ↓](#802-找到最终的安全状态--find-eventual-safe-states) |
| 631 | 设计 Excel 求和公式 🔒 | [CN](https://leetcode.cn/problems/design-excel-sum-formula/) · [EN](https://leetcode.com/problems/design-excel-sum-formula/) | 困难 | **对应 Ambient.ai 电子表格题**，[题解 ↓](#631-设计-excel-求和公式--design-excel-sum-formula) |
| 310 | 最小高度树 | [CN](https://leetcode.cn/problems/minimum-height-trees/) · [EN](https://leetcode.com/problems/minimum-height-trees/) | 中等 | 从叶子一层层剥，[题解 ↓](#310-最小高度树--minimum-height-trees) |
| 269 | 火星词典 🔒 | [CN](https://leetcode.cn/problems/alien-dictionary/) · [EN](https://leetcode.com/problems/alien-dictionary/) | 困难 | 先建图再拓扑排序，[题解 ↓](#269-火星词典--alien-dictionary) |

<br>

#### 207. 课程表 / Course Schedule

[CN](https://leetcode.cn/problems/course-schedule/) · [EN](https://leetcode.com/problems/course-schedule/) · Medium

**题意**：给定先修关系，判断能否修完所有课程。

**思路**：能上完所有课 ⇔ 有向图没有环。三色标记 DFS：0 表示没访问，1 表示在当前递归栈里，2 表示已经确认从它出发没有环。DFS 时遇到颜色为 1 的节点说明找到了环。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 1. 建图：b -> a 表示先上 b 再上 a
        g = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            g[b].append(a)

        colors = [0] * numCourses  # 0 未访问，1 访问中，2 已完成

        # 2. 返回从 x 出发是否能找到环
        def dfs(x: int) -> bool:
            colors[x] = 1
            for y in g[x]:
                if colors[y] == 1 or colors[y] == 0 and dfs(y):
                    return True
            colors[x] = 2
            return False

        # 3. 任意一个起点找到环就不能完成
        for i, c in enumerate(colors):
            if c == 0 and dfs(i):
                return False
        return True
```

**复杂度**：时间 O(n + m)；空间 O(n + m)。

<br>

#### 210. 课程表 II / Course Schedule II

[CN](https://leetcode.cn/problems/course-schedule-ii/) · [EN](https://leetcode.com/problems/course-schedule-ii/) · Medium

**题意**：返回修完所有课程的一种顺序，不可能则返回空数组。

**思路**：Kahn 拓扑排序：统计每门课的入度，入度为 0 的课入队。每次出队一门课加入答案，并把它的后继课入度减一，减到 0 就入队。最后答案长度不足 n 说明有环。

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 1. 建图并统计入度
        g = [[] for _ in range(numCourses)]
        deg = [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            deg[a] += 1

        # 2. 入度为 0 的课可以直接上
        q = deque(i for i, d in enumerate(deg) if d == 0)
        order = []
        while q:
            x = q.popleft()
            order.append(x)
            # 3. 上完 x 后，后继课的入度减一
            for y in g[x]:
                deg[y] -= 1
                if deg[y] == 0:
                    q.append(y)
        # 4. 有环时无法上完所有课
        return order if len(order) == numCourses else []
```

**复杂度**：时间 O(n + m)；空间 O(n + m)。

<br>

#### 802. 找到最终的安全状态 / Find Eventual Safe States

[CN](https://leetcode.cn/problems/find-eventual-safe-states/) · [EN](https://leetcode.com/problems/find-eventual-safe-states/) · Medium

**题意**：有向图中，如果从节点 i 出发的所有路径最终都走到终端节点（没有出边的节点），i 就是安全节点。按升序返回所有安全节点。

**思路**：一个点不安全，等价于从它出发能走到环。用 207 的三色标记 DFS：0 未访问，1 访问中（在当前递归栈里），2 已确认安全。DFS 中遇到颜色为 1 的点说明有环，这条路径上的点都保持颜色 1（不安全）；所有邻居都安全，自己才标成 2。

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        n = len(graph)
        colors = [0] * n  # 0 未访问，1 访问中 / 不安全，2 安全

        # 1. 返回从 x 出发是否能走到环
        def has_cycle(x: int) -> bool:
            colors[x] = 1
            for y in graph[x]:
                # 2. 走到访问中的点（有环），或者邻居能走到环
                if colors[y] == 1 or colors[y] == 0 and has_cycle(y):
                    return True  # x 保持颜色 1，不安全
            # 3. 所有邻居都安全，x 也安全
            colors[x] = 2
            return False

        return [i for i in range(n) if colors[i] == 2 or colors[i] == 0 and not has_cycle(i)]
```

**复杂度**：时间 O(n + m)；空间 O(n)。

<br>

#### 631. 设计 Excel 求和公式 / Design Excel Sum Formula

[CN](https://leetcode.cn/problems/design-excel-sum-formula/) · [EN](https://leetcode.com/problems/design-excel-sum-formula/) · Hard

**题意**：实现一个 height 行、列为 'A' 到 width 的表格，所有格子初始为 0。`set(r, c, v)` 把格子设成 v；`get(r, c)` 返回格子的值；`sum(r, c, numbers)` 把格子设成公式，值等于 numbers 中所有格子（"A1"）和区域（"A1:B2"）之和，并返回当前值。公式会随着被引用格子的变化而变化，保证没有循环引用。

**思路**：单元格是节点，「格子 X 的公式引用了格子 Y」是一条依赖边，整张表是一个有向无环图。公式格子存「被引用的格子 → 引用次数」（同一个格子可能被多个区域重复引用）。`set` 会删掉这个格子的公式，变回普通数字。`get` 时沿着依赖递归求值，并用 memo 记住本次已经算过的格子，相当于按拓扑序（先算被依赖的格子）计算，共享的依赖只算一次。

```python
class Excel:
    def __init__(self, height: int, width: str):
        self.vals = {}      # 1. 普通格子：(行, 列) -> 数值
        self.formulas = {}  # 公式格子：(行, 列) -> Counter{被引用的格子: 次数}

    def set(self, row: int, column: str, val: int) -> None:
        # 2. 设成数字后，原来的公式失效
        self.formulas.pop((row, column), None)
        self.vals[(row, column)] = val

    def get(self, row: int, column: str) -> int:
        memo = {}

        # 3. 递归求值：先算出被依赖的格子（拓扑序）
        def value(cell: Tuple[int, str]) -> int:
            if cell in memo:
                return memo[cell]
            if cell in self.formulas:
                res = sum(value(c) * k for c, k in self.formulas[cell].items())
            else:
                res = self.vals.get(cell, 0)
            memo[cell] = res
            return res

        return value((row, column))

    def sum(self, row: int, column: str, numbers: List[str]) -> int:
        # 4. 解析 "A1" 和 "A1:B2"，统计每个格子被引用了几次
        cnt = Counter()
        for s in numbers:
            if ':' in s:
                a, b = s.split(':')
                for r in range(int(a[1:]), int(b[1:]) + 1):
                    for c in range(ord(a[0]), ord(b[0]) + 1):
                        cnt[(r, chr(c))] += 1
            else:
                cnt[(int(s[1:]), s[0])] += 1
        # 5. 保存公式并返回当前值
        self.formulas[(row, column)] = cnt
        return self.get(row, column)
```

**复杂度**：`set` O(1)；`sum` O(引用的格子数 + 求值)；`get` 最坏 O(H·W·引用数)，每个格子在一次求值中只算一次。

<br>

#### 310. 最小高度树 / Minimum Height Trees

[CN](https://leetcode.cn/problems/minimum-height-trees/) · [EN](https://leetcode.com/problems/minimum-height-trees/) · Medium

**题意**：给一棵 n 个节点的树（无向边列表），选一个节点作为根使树的高度最小，返回所有这样的根。

**思路**：类似拓扑排序，从外往里剥洋葱。度为 1 的点是叶子，叶子不可能是最优的根（换成它的邻居，高度不会更大）。每轮把当前所有叶子删掉，它们的邻居度数减一，变成新的叶子。剩下不超过 2 个点时，它们就是树的中心，也就是答案。

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        # 1. 建图并统计度数
        g = [[] for _ in range(n)]
        deg = [0] * n
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
            deg[x] += 1
            deg[y] += 1
        # 2. 所有叶子（度为 1）入队
        leaves = [i for i in range(n) if deg[i] == 1]
        remain = n
        # 3. 一层层删除叶子，直到剩下不超过 2 个点
        while remain > 2:
            remain -= len(leaves)
            nxt = []
            for x in leaves:
                for y in g[x]:
                    deg[y] -= 1
                    if deg[y] == 1:
                        nxt.append(y)
            leaves = nxt
        return leaves
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 269. 火星词典 / Alien Dictionary

[CN](https://leetcode.cn/problems/alien-dictionary/) · [EN](https://leetcode.com/problems/alien-dictionary/) · Hard

**题意**：给一组已经按外星语字母顺序排好序的单词，推出字母之间的顺序，返回任意一种合法顺序；不合法返回空串。

**思路**：建图 + 模板 D。相邻的两个单词，第一个不同的字母决定了顺序：w1[i] 在 w2[i] 前面，连一条边。如果 w2 是 w1 的真前缀（比如 "abc" 排在 "ab" 前面），输入本身就不合法。所有出现过的字母都是节点。拓扑排序后如果有字母没排进去，说明有环，也不合法。

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # 1. 所有出现过的字母都是节点
        g = {c: set() for w in words for c in w}
        deg = {c: 0 for c in g}
        # 2. 相邻单词第一个不同的字母连边
        for w1, w2 in pairwise(words):
            for a, b in zip(w1, w2):
                if a != b:
                    if b not in g[a]:
                        g[a].add(b)
                        deg[b] += 1
                    break
            else:
                # 3. 没有不同的字母，且 w1 更长：前缀排在后面，非法
                if len(w1) > len(w2):
                    return ''
        # 4. Kahn 拓扑排序
        q = deque(c for c, d in deg.items() if d == 0)
        order = []
        while q:
            c = q.popleft()
            order.append(c)
            for nxt in g[c]:
                deg[nxt] -= 1
                if deg[nxt] == 0:
                    q.append(nxt)
        # 5. 有字母没排进去，说明有环
        return ''.join(order) if len(order) == len(g) else ''
```

**复杂度**：时间 O(L)，L 是所有单词的总长度；空间 O(|Σ|²)。

<br>

### 5. 路径枚举（DFS + 回溯）

**特征**：列出所有路径、所有方案。模板 B。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| **797** | **所有可能的路径** | [CN](https://leetcode.cn/problems/all-paths-from-source-to-target/) · [EN](https://leetcode.com/problems/all-paths-from-source-to-target/) | 中等 | 模板题，[题解 ↓](#797-所有可能的路径--all-paths-from-source-to-target) |
| 332 | 重新安排行程 | [CN](https://leetcode.cn/problems/reconstruct-itinerary/) · [EN](https://leetcode.com/problems/reconstruct-itinerary/) | 困难 | 进阶，[题解 ↓](#332-重新安排行程--reconstruct-itinerary) |

<br>

#### 797. 所有可能的路径 / All Paths From Source to Target

[CN](https://leetcode.cn/problems/all-paths-from-source-to-target/) · [EN](https://leetcode.com/problems/all-paths-from-source-to-target/) · Medium

**题意**：给一个有 n 个节点的有向无环图（邻接表 `graph`），找出所有从节点 0 到节点 n-1 的路径。

**思路**：模板 B。从 0 出发 DFS，`path` 记录当前路径，走到 n-1 就把路径复制一份加入答案，递归返回时把最后一个点弹出（回溯）。图是有向无环图，不会走回当前路径上的点，所以不需要 `on_path`。

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        n = len(graph)
        ans = []
        path = [0]  # 1. 当前路径，从节点 0 出发

        def dfs(x: int) -> None:
            # 2. 到达终点，记录路径
            if x == n - 1:
                ans.append(path.copy())
                return
            # 3. 枚举下一步走到哪个邻居
            for y in graph[x]:
                path.append(y)
                dfs(y)
                path.pop()  # 4. 回溯

        dfs(0)
        return ans
```

**复杂度**：时间 O(n·2ⁿ)，路径数最多 2ⁿ⁻² 条，每条复制 O(n)；空间 O(n)（不计答案）。

<br>

#### 332. 重新安排行程 / Reconstruct Itinerary

[CN](https://leetcode.cn/problems/reconstruct-itinerary/) · [EN](https://leetcode.com/problems/reconstruct-itinerary/) · Hard

**题意**：给一组机票 `[from, to]`，从 "JFK" 出发，每张机票恰好用一次，返回字典序最小的行程。保证至少存在一种合法行程。

**思路**：「每条边恰好走一次」是欧拉路径，用 Hierholzer 算法：从起点出发 DFS，每走一条边就把它删掉；一个点的边都走完了，才把它加入答案（后序）。最后把答案反转。为什么后序是对的：最先卡住的点一定是终点，走进「死胡同」的那部分会被排在最后。邻居按字典序从小到大走，就得到字典序最小的行程；把邻居倒序存进列表，`pop()` 就能 O(1) 取出最小的。

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # 1. 建图，邻居倒序存放，pop() 取出字典序最小的
        g = defaultdict(list)
        for a, b in sorted(tickets, reverse=True):
            g[a].append(b)
        ans = []

        def dfs(x: str) -> None:
            # 2. 每走一条边就删掉它（pop）
            while g[x]:
                dfs(g[x].pop())
            # 3. x 的边都走完了，后序加入答案
            ans.append(x)

        dfs('JFK')
        # 4. 后序反转就是行程
        return ans[::-1]
```

**复杂度**：时间 O(m log m)，m 是机票数（排序）；空间 O(m)。

<br>

### 6. 带权最短路（Dijkstra）

**特征**：边的权重不同，求最短距离或最小代价。模板 G。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| **743** | **网络延迟时间** | [CN](https://leetcode.cn/problems/network-delay-time/) · [EN](https://leetcode.com/problems/network-delay-time/) | 中等 | Dijkstra 模板题，[题解 ↓](#743-网络延迟时间--network-delay-time) |
| 1631 | 最小体力消耗路径 | [CN](https://leetcode.cn/problems/path-with-minimum-effort/) · [EN](https://leetcode.com/problems/path-with-minimum-effort/) | 中等 | 网格上的 Dijkstra，[题解 ↓](#1631-最小体力消耗路径--path-with-minimum-effort) |
| 787 | K 站中转内最便宜的航班 | [CN](https://leetcode.cn/problems/cheapest-flights-within-k-stops/) · [EN](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | 中等 | 带步数限制，[题解 ↓](#787-k-站中转内最便宜的航班--cheapest-flights-within-k-stops) |
| 1514 | 概率最大的路径 | [CN](https://leetcode.cn/problems/path-with-maximum-probability/) · [EN](https://leetcode.com/problems/path-with-maximum-probability/) | 中等 | Dijkstra 变形，[题解 ↓](#1514-概率最大的路径--path-with-maximum-probability) |

<br>

#### 743. 网络延迟时间 / Network Delay Time

[CN](https://leetcode.cn/problems/network-delay-time/) · [EN](https://leetcode.com/problems/network-delay-time/) · Medium

**题意**：从节点 k 发出信号，求所有节点都收到信号的最短时间，不可能返回 -1。

**思路**：单源最短路，边权非负，用堆优化 Dijkstra。答案是所有节点最短路的最大值；有节点到不了就返回 -1。

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # 1. 建图（节点编号改成从 0 开始）
        g = [[] for _ in range(n)]
        for x, y, d in times:
            g[x - 1].append((y - 1, d))

        # 2. Dijkstra：堆中存 (距离, 节点)
        dis = [inf] * n
        dis[k - 1] = 0
        h = [(0, k - 1)]
        while h:
            dx, x = heappop(h)
            if dx > dis[x]:  # 3. 过期的堆元素，跳过
                continue
            for y, d in g[x]:
                new_dis = dx + d
                if new_dis < dis[y]:
                    dis[y] = new_dis
                    heappush(h, (new_dis, y))

        # 4. 所有节点都收到信号的时间
        mx = max(dis)
        return mx if mx < inf else -1
```

**复杂度**：时间 O(m log m)；空间 O(n + m)。

<br>

#### 1631. 最小体力消耗路径 / Path With Minimum Effort

[CN](https://leetcode.cn/problems/path-with-minimum-effort/) · [EN](https://leetcode.com/problems/path-with-minimum-effort/) · Medium

**题意**：网格中每个格子有高度，从左上角走到右下角，一条路径的体力消耗是相邻格子高度差绝对值的**最大值**。求最小的体力消耗。

**思路**：模板 G 的网格版，只是路径代价从「边权之和」换成「边权的最大值」。Dijkstra 仍然成立，因为沿着路径走，最大值只会变大不会变小（和非负边权求和一样单调）。格子是节点，相邻格子之间的边权是高度差。

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m, n = len(heights), len(heights[0])
        # 1. dis[i][j] = 到 (i, j) 的路径的最小体力消耗
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        h = [(0, 0, 0)]  # (体力消耗, 行, 列)
        while h:
            d, i, j = heappop(h)
            # 2. 第一次弹出终点时就是答案
            if i == m - 1 and j == n - 1:
                return d
            if d > dis[i][j]:  # 过时的记录
                continue
            for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < m and 0 <= y < n:
                    # 3. 路径代价取最大值而不是求和
                    nd = max(d, abs(heights[x][y] - heights[i][j]))
                    if nd < dis[x][y]:
                        dis[x][y] = nd
                        heappush(h, (nd, x, y))
```

**复杂度**：时间 O(mn log(mn))；空间 O(mn)。

<br>

#### 787. K 站中转内最便宜的航班 / Cheapest Flights Within K Stops

[CN](https://leetcode.cn/problems/cheapest-flights-within-k-stops/) · [EN](https://leetcode.com/problems/cheapest-flights-within-k-stops/) · Medium

**题意**：求从 src 到 dst 最多经过 k 次中转的最便宜价格，不存在返回 -1。

**思路**：限制边数的最短路，用 Bellman-Ford：最多 k 次中转即最多 k+1 条边，做 k+1 轮松弛。每一轮必须基于上一轮的 dis 副本更新，否则一轮里会连续走多条边。

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # 1. dis[i] = 最多用 t 条边时，src 到 i 的最小花费
        dis = [inf] * n
        dis[src] = 0
        # 2. 最多 k+1 条边，做 k+1 轮松弛
        for _ in range(k + 1):
            pre = dis.copy()  # 3. 基于上一轮的结果更新
            for x, y, price in flights:
                if pre[x] + price < dis[y]:
                    dis[y] = pre[x] + price
        return dis[dst] if dis[dst] < inf else -1
```

**复杂度**：时间 O(k·m)；空间 O(n)。

<br>

#### 1514. 概率最大的路径 / Path with Maximum Probability

[CN](https://leetcode.cn/problems/path-with-maximum-probability/) · [EN](https://leetcode.com/problems/path-with-maximum-probability/) · Medium

**题意**：无向图中每条边有通过的成功概率，求从 start_node 到 end_node 成功概率最大的路径的概率，没有路径返回 0。

**思路**：模板 G 的变形：路径的值是边权**相乘**，要求**最大**。概率都在 [0, 1] 之间，沿路径越乘越小，和「非负边权越加越大」是对称的，所以 Dijkstra 仍然成立：把最小堆换成最大堆（Python 里存相反数），每次取出当前概率最大的点。

```python
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start_node: int, end_node: int) -> float:
        # 1. 建图（无向图）
        g = [[] for _ in range(n)]
        for (x, y), p in zip(edges, succProb):
            g[x].append((y, p))
            g[y].append((x, p))
        # 2. prob[i] = 到 i 的最大成功概率
        prob = [0.0] * n
        prob[start_node] = 1.0
        h = [(-1.0, start_node)]  # 最大堆：存概率的相反数
        while h:
            p, x = heappop(h)
            p = -p
            # 3. 第一次弹出终点时就是答案
            if x == end_node:
                return p
            if p < prob[x]:  # 过时的记录
                continue
            for y, py in g[x]:
                # 4. 概率相乘
                np = p * py
                if np > prob[y]:
                    prob[y] = np
                    heappush(h, (-np, y))
        return 0.0
```

**复杂度**：时间 O(m log m)；空间 O(n + m)。

<br>

### 7. 二分图

**特征**：能不能把节点分成两组，让每条边都连接不同的组。模板 H。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| **785** | **判断二分图** | [CN](https://leetcode.cn/problems/is-graph-bipartite/) · [EN](https://leetcode.com/problems/is-graph-bipartite/) | 中等 | 染色模板题，[题解 ↓](#785-判断二分图--is-graph-bipartite) |
| 886 | 可能的二分法 | [CN](https://leetcode.cn/problems/possible-bipartition/) · [EN](https://leetcode.com/problems/possible-bipartition/) | 中等 | 先建图再染色，[题解 ↓](#886-可能的二分法--possible-bipartition) |

<br>

#### 785. 判断二分图 / Is Graph Bipartite?

[CN](https://leetcode.cn/problems/is-graph-bipartite/) · [EN](https://leetcode.com/problems/is-graph-bipartite/) · Medium

**题意**：给一个无向图的邻接表 `graph`（可能不连通），判断能否把节点分成两组，使每条边的两个端点属于不同的组。

**思路**：模板 H。从一个没染色的点出发，染成颜色 1，DFS 时把邻居染成相反的颜色。如果某个邻居已经和自己同色，就不是二分图。图可能不连通，每个连通块都要单独染一次。

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        colors = [0] * n  # 1. 0 表示未染色，1 和 -1 表示两种颜色

        # 2. 把 x 染成 c，返回它所在的连通块能否完成染色
        def dfs(x: int, c: int) -> bool:
            colors[x] = c
            for y in graph[x]:
                # 3. 邻居同色，冲突
                if colors[y] == c:
                    return False
                # 4. 邻居未染色，染成相反的颜色
                if colors[y] == 0 and not dfs(y, -c):
                    return False
            return True

        # 5. 每个还没染色的点都是一个新的连通块
        return all(colors[i] or dfs(i, 1) for i in range(n))
```

**复杂度**：时间 O(n + m)；空间 O(n)。

<br>

#### 886. 可能的二分法 / Possible Bipartition

[CN](https://leetcode.cn/problems/possible-bipartition/) · [EN](https://leetcode.com/problems/possible-bipartition/) · Medium

**题意**：把编号 1 到 n 的人分成两组，`dislikes[i] = [a, b]` 表示 a 和 b 不能在同一组。判断能否分组。

**思路**：人是节点，「互相不喜欢」是无向边，问题就是判断这个图是不是二分图。先用 dislikes 建邻接表，再按 785 染色。编号从 1 开始，转成从 0 开始。

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        # 1. 建图（无向图，编号改成从 0 开始）
        g = [[] for _ in range(n)]
        for x, y in dislikes:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)

        colors = [0] * n  # 0 未染色，1 和 -1 是两组

        # 2. 把 x 染成 c，邻居染成 -c
        def dfs(x: int, c: int) -> bool:
            colors[x] = c
            for y in g[x]:
                if colors[y] == c or colors[y] == 0 and not dfs(y, -c):
                    return False
            return True

        # 3. 每个连通块都要染色
        return all(colors[i] or dfs(i, 1) for i in range(n))
```

**复杂度**：时间 O(n + m)；空间 O(n + m)。

<br>

### 8. 最小生成树

**特征**：把所有点连起来，总代价最小。Kruskal：边按权重排序，从小到大用并查集（模板 F）加边，跳过会成环的边；Prim：从一个点出发，每次把离当前树最近的点加进来，适合稠密图。

| 题号 | 题目 | 链接 | 难度 | 备注 |
|---|---|---|---|---|
| 1584 | 连接所有点的最小费用 | [CN](https://leetcode.cn/problems/min-cost-to-connect-all-points/) · [EN](https://leetcode.com/problems/min-cost-to-connect-all-points/) | 中等 | Kruskal / Prim 模板题，[题解 ↓](#1584-连接所有点的最小费用--min-cost-to-connect-all-points) |
| 1135 | 最低成本连通所有城市 🔒 | [CN](https://leetcode.cn/problems/connecting-cities-with-minimum-cost/) · [EN](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) | 中等 | Kruskal，[题解 ↓](#1135-最低成本连通所有城市--connecting-cities-with-minimum-cost) |

<br>

#### 1584. 连接所有点的最小费用 / Min Cost to Connect All Points

[CN](https://leetcode.cn/problems/min-cost-to-connect-all-points/) · [EN](https://leetcode.com/problems/min-cost-to-connect-all-points/) · Medium

**题意**：平面上的点之间的费用是曼哈顿距离，求连接所有点的最小总费用。

**思路**：稠密图（完全图）的最小生成树，用 O(n²) 的 Prim：dis[i] 表示点 i 到当前生成树的最短距离。每次选一个不在树中、dis 最小的点加入，再用它更新其他点的 dis。

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        # 1. dis[i] = 点 i 到生成树的最短距离；先把 0 号点加入树
        dis = [inf] * n
        dis[0] = 0
        in_tree = [False] * n
        ans = 0
        for _ in range(n):
            # 2. 选不在树中、距离最近的点
            u = -1
            for i in range(n):
                if not in_tree[i] and (u < 0 or dis[i] < dis[u]):
                    u = i
            in_tree[u] = True
            ans += dis[u]
            # 3. 用 u 更新其他点到树的距离
            ux, uy = points[u]
            for i in range(n):
                if not in_tree[i]:
                    x, y = points[i]
                    dis[i] = min(dis[i], abs(x - ux) + abs(y - uy))
        return ans
```

**复杂度**：时间 O(n²)；空间 O(n)。

<br>

#### 1135. 最低成本连通所有城市 / Connecting Cities With Minimum Cost

[CN](https://leetcode.cn/problems/connecting-cities-with-minimum-cost/) · [EN](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) · Medium

**题意**：n 个城市（编号 1 到 n），`connections[i] = [x, y, cost]` 表示修一条连接 x 和 y 的路要花 cost。求连通所有城市的最小成本，无法连通返回 -1。

**思路**：Kruskal。边按成本从小到大排序，依次尝试加入：两端已经连通就跳过（加了会成环），否则用并查集（模板 F）合并并计入成本。成功加入 n-1 条边说明所有城市已连通。

```python
class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        fa = list(range(n + 1))

        def find(x: int) -> int:
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        ans = 0
        edges = 0
        # 1. 按成本从小到大考虑每条边
        for x, y, cost in sorted(connections, key=lambda e: e[2]):
            fx, fy = find(x), find(y)
            # 2. 已经连通，加这条边会成环
            if fx == fy:
                continue
            # 3. 合并，计入成本
            fa[fx] = fy
            ans += cost
            edges += 1
            if edges == n - 1:
                return ans
        # 4. 边不够 n-1 条，无法连通
        return -1 if n > 1 else 0
```

**复杂度**：时间 O(m log m)；空间 O(n)。

<br>
<br>

## 6. 动态规划

定义状态 → 写转移 → 定边界；先写记忆化搜索，再翻译成递推。

<br>

### 入门

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **70** | **爬楼梯** | [CN](https://leetcode.cn/problems/climbing-stairs/) · [EN](https://leetcode.com/problems/climbing-stairs/) | [题解 ↓](#70-爬楼梯--climbing-stairs) |
| 746 | 使用最小花费爬楼梯 | [CN](https://leetcode.cn/problems/min-cost-climbing-stairs/) · [EN](https://leetcode.com/problems/min-cost-climbing-stairs/) | f[i] = min(f[i-1] + cost[i-1], f[i-2] + cost[i-2]) |
| **198** | **打家劫舍** | [CN](https://leetcode.cn/problems/house-robber/) · [EN](https://leetcode.com/problems/house-robber/) | [题解 ↓](#198-打家劫舍--house-robber) |
| 213 | 打家劫舍 II | [CN](https://leetcode.cn/problems/house-robber-ii/) · [EN](https://leetcode.com/problems/house-robber-ii/) | 环形：分「不偷第一间」和「不偷最后一间」各做一次 198 |
| **53** | **最大子数组和** | [CN](https://leetcode.cn/problems/maximum-subarray/) · [EN](https://leetcode.com/problems/maximum-subarray/) | [题解 ↓](#53-最大子数组和--maximum-subarray) |
| **62** | **不同路径** | [CN](https://leetcode.cn/problems/unique-paths/) · [EN](https://leetcode.com/problems/unique-paths/) | [题解 ↓](#62-不同路径--unique-paths) |
| 64 | 最小路径和 | [CN](https://leetcode.cn/problems/minimum-path-sum/) · [EN](https://leetcode.com/problems/minimum-path-sum/) | f[i][j] = min(上, 左) + grid[i][j] |
| 120 | 三角形最小路径和 | [CN](https://leetcode.cn/problems/triangle/) · [EN](https://leetcode.com/problems/triangle/) | 自底向上：f[j] = min(f[j], f[j+1]) + row[j] |
| 221 | 最大正方形 | [CN](https://leetcode.cn/problems/maximal-square/) · [EN](https://leetcode.com/problems/maximal-square/) | 以 (i,j) 为右下角的最大边长 = min(左, 上, 左上) + 1 |
| 152 | 乘积最大子数组 | [CN](https://leetcode.cn/problems/maximum-product-subarray/) · [EN](https://leetcode.com/problems/maximum-product-subarray/) | 同时维护以 i 结尾的最大积和最小积（负数会翻转） |

<br>

#### 70. 爬楼梯 / Climbing Stairs

[CN](https://leetcode.cn/problems/climbing-stairs/) · [EN](https://leetcode.com/problems/climbing-stairs/) · Easy

**题意**：每次爬 1 或 2 级台阶，求爬到第 n 级的方法数。

**思路**：最后一步要么跨 1 级、要么跨 2 级，所以 f[i] = f[i-1] + f[i-2]，f[0] = f[1] = 1。只依赖前两项，用两个变量滚动。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        # 1. f0 = f[i-2]，f1 = f[i-1]，初始为 f[0] = f[1] = 1
        f0 = f1 = 1
        for _ in range(2, n + 1):
            # 2. f[i] = f[i-1] + f[i-2]
            f0, f1 = f1, f0 + f1
        return f1
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 198. 打家劫舍 / House Robber

[CN](https://leetcode.cn/problems/house-robber/) · [EN](https://leetcode.com/problems/house-robber/) · Medium

**题意**：不能偷相邻的房屋，求能偷到的最高金额。

**思路**：考虑最后一间房偷不偷：不偷，答案是前 i-1 间的最大值；偷，答案是前 i-2 间的最大值加上它。f[i] = max(f[i-1], f[i-2] + nums[i])。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 1. f0 = f[i-2]，f1 = f[i-1]
        f0 = f1 = 0
        for x in nums:
            # 2. 不偷 x：f1；偷 x：f0 + x
            f0, f1 = f1, max(f1, f0 + x)
        return f1
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 53. 最大子数组和 / Maximum Subarray

[CN](https://leetcode.cn/problems/maximum-subarray/) · [EN](https://leetcode.com/problems/maximum-subarray/) · Medium

**题意**：求和最大的非空连续子数组的和。

**思路**：f[i] 表示以 nums[i] 结尾的最大子数组和。要么只选 nums[i]，要么接在前面的子数组后面：f[i] = max(f[i-1], 0) + nums[i]。答案是所有 f[i] 的最大值。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = -inf
        f = 0  # 1. 以当前元素结尾的最大子数组和
        for x in nums:
            # 2. 前面的和是负数就丢掉，从 x 重新开始
            f = max(f, 0) + x
            # 3. 更新答案
            ans = max(ans, f)
        return ans
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 62. 不同路径 / Unique Paths

[CN](https://leetcode.cn/problems/unique-paths/) · [EN](https://leetcode.com/problems/unique-paths/) · Medium

**题意**：从 m×n 网格左上角走到右下角，每次只能向右或向下，求路径数。

**思路**：只能向右或向下，所以 f[i][j] = f[i-1][j] + f[i][j-1]。每一行只依赖上一行和本行左边，压缩成一维：f[j] += f[j-1]。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 1. 第一行每个格子都只有 1 条路径
        f = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                # 2. f[j]（上方） + f[j-1]（左方）
                f[j] += f[j - 1]
        return f[-1]
```

**复杂度**：时间 O(mn)；空间 O(n)。也可以直接算组合数 C(m+n-2, m-1)。

<br>

### 背包

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **416** | **分割等和子集** | [CN](https://leetcode.cn/problems/partition-equal-subset-sum/) · [EN](https://leetcode.com/problems/partition-equal-subset-sum/) | [题解 ↓](#416-分割等和子集--partition-equal-subset-sum) |
| **494** | **目标和** | [CN](https://leetcode.cn/problems/target-sum/) · [EN](https://leetcode.com/problems/target-sum/) | [题解 ↓](#494-目标和--target-sum) |
| **322** | **零钱兑换** | [CN](https://leetcode.cn/problems/coin-change/) · [EN](https://leetcode.com/problems/coin-change/) | [题解 ↓](#322-零钱兑换--coin-change) |
| 518 | 零钱兑换 II | [CN](https://leetcode.cn/problems/coin-change-ii/) · [EN](https://leetcode.com/problems/coin-change-ii/) | 完全背包求方案数：外层枚举硬币，内层正序枚举金额 |
| 279 | 完全平方数 | [CN](https://leetcode.cn/problems/perfect-squares/) · [EN](https://leetcode.com/problems/perfect-squares/) | 完全背包：物品是 1, 4, 9, …，求最少个数 |
| 474 | 一和零 | [CN](https://leetcode.cn/problems/ones-and-zeroes/) · [EN](https://leetcode.com/problems/ones-and-zeroes/) | 二维费用的 0-1 背包，0 和 1 的容量都倒序枚举 |
| 1049 | 最后一块石头的重量 II | [CN](https://leetcode.cn/problems/last-stone-weight-ii/) · [EN](https://leetcode.com/problems/last-stone-weight-ii/) | 分成两堆使差最小 → 0-1 背包求不超过 sum/2 的最大和 |

<br>

#### 416. 分割等和子集 / Partition Equal Subset Sum

[CN](https://leetcode.cn/problems/partition-equal-subset-sum/) · [EN](https://leetcode.com/problems/partition-equal-subset-sum/) · Medium

**题意**：判断能否把数组分成两个和相等的子集。

**思路**：总和必须是偶数，目标是选一些数使和恰好为 sum/2，这是 0-1 背包。f[j] 表示能否凑出和 j，倒序枚举 j 保证每个数只用一次。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        # 1. 总和是奇数，不可能平分
        if s % 2:
            return False
        target = s // 2
        # 2. f[j] = 能否从前面的数中选出和为 j 的子集
        f = [True] + [False] * target
        for x in nums:
            # 3. 0-1 背包倒序枚举，保证 x 只用一次
            for j in range(target, x - 1, -1):
                f[j] = f[j] or f[j - x]
            if f[target]:
                return True
        return f[target]
```

**复杂度**：时间 O(n·sum)；空间 O(sum)。

<br>

#### 494. 目标和 / Target Sum

[CN](https://leetcode.cn/problems/target-sum/) · [EN](https://leetcode.com/problems/target-sum/) · Medium

**题意**：给每个数添加 + 或 -，求表达式结果等于 target 的方案数。

**思路**：设加正号的数之和为 p，那么 p - (s - p) = target，即 p = (s + target) / 2。问题变成：选一些数使和为 p 的方案数，0-1 背包计数。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # 1. 正数和 p = (s + target) / 2，必须是非负整数
        s = sum(nums) + target
        if s < 0 or s % 2:
            return 0
        p = s // 2
        # 2. f[j] = 选出和为 j 的方案数
        f = [1] + [0] * p
        for x in nums:
            # 3. 0-1 背包倒序
            for j in range(p, x - 1, -1):
                f[j] += f[j - x]
        return f[p]
```

**复杂度**：时间 O(n·p)；空间 O(p)。

<br>

#### 322. 零钱兑换 / Coin Change

[CN](https://leetcode.cn/problems/coin-change/) · [EN](https://leetcode.com/problems/coin-change/) · Medium

**题意**：每种硬币无限个，求凑成 amount 的最少硬币数，凑不出返回 -1。

**思路**：完全背包求最少物品数。f[j] 表示凑出金额 j 的最少硬币数，f[j] = min(f[j], f[j - c] + 1)。完全背包正序枚举金额，同一枚硬币可以重复使用。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 1. f[j] = 凑出 j 的最少硬币数，凑不出为 inf
        f = [0] + [inf] * amount
        for c in coins:
            # 2. 完全背包：正序枚举，允许重复使用 c
            for j in range(c, amount + 1):
                f[j] = min(f[j], f[j - c] + 1)
        # 3. 凑不出返回 -1
        return f[amount] if f[amount] < inf else -1
```

**复杂度**：时间 O(n·amount)；空间 O(amount)。

<br>

### 子序列与字符串

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **300** | **最长递增子序列** | [CN](https://leetcode.cn/problems/longest-increasing-subsequence/) · [EN](https://leetcode.com/problems/longest-increasing-subsequence/) | [题解 ↓](#300-最长递增子序列--longest-increasing-subsequence) |
| 673 | 最长递增子序列的个数 | [CN](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/) · [EN](https://leetcode.com/problems/number-of-longest-increasing-subsequence/) | O(n²) DP，同时记录以 i 结尾的 LIS 长度和个数 |
| 354 | 俄罗斯套娃信封问题 | [CN](https://leetcode.cn/problems/russian-doll-envelopes/) · [EN](https://leetcode.com/problems/russian-doll-envelopes/) | 宽升序、宽相同时高降序，再对高求 LIS（贪心 + 二分） |
| **1143** | **最长公共子序列** | [CN](https://leetcode.cn/problems/longest-common-subsequence/) · [EN](https://leetcode.com/problems/longest-common-subsequence/) | [题解 ↓](#1143-最长公共子序列--longest-common-subsequence) |
| **72** | **编辑距离** | [CN](https://leetcode.cn/problems/edit-distance/) · [EN](https://leetcode.com/problems/edit-distance/) | [题解 ↓](#72-编辑距离--edit-distance) |
| 583 | 两个字符串的删除操作 | [CN](https://leetcode.cn/problems/delete-operation-for-two-strings/) · [EN](https://leetcode.com/problems/delete-operation-for-two-strings/) | 答案 = m + n - 2·LCS |
| 115 | 不同的子序列 | [CN](https://leetcode.cn/problems/distinct-subsequences/) · [EN](https://leetcode.com/problems/distinct-subsequences/) | f[i][j] = f[i-1][j] + (s[i]==t[j] ? f[i-1][j-1] : 0) |
| **5** | **最长回文子串** | [CN](https://leetcode.cn/problems/longest-palindromic-substring/) · [EN](https://leetcode.com/problems/longest-palindromic-substring/) | [题解 ↓](#5-最长回文子串--longest-palindromic-substring) |
| **139** | **单词拆分** | [CN](https://leetcode.cn/problems/word-break/) · [EN](https://leetcode.com/problems/word-break/) | [题解 ↓](#139-单词拆分--word-break) |

<br>

#### 300. 最长递增子序列 / Longest Increasing Subsequence

[CN](https://leetcode.cn/problems/longest-increasing-subsequence/) · [EN](https://leetcode.com/problems/longest-increasing-subsequence/) · Medium

**题意**：求最长严格递增子序列的长度。

**思路**：O(n²) 的 DP 是 f[i] = max(f[j]) + 1（j < i 且 nums[j] < nums[i]）。更优的贪心 + 二分：g[k] 表示长度为 k+1 的递增子序列的最小结尾。g 是严格递增的，对每个 x 用二分找第一个 ≥ x 的位置替换，找不到就追加。g 的长度就是答案。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        g = []  # 1. g[k] = 长度为 k+1 的递增子序列的最小结尾
        for x in nums:
            # 2. 找第一个 >= x 的位置
            j = bisect_left(g, x)
            if j == len(g):
                g.append(x)  # 3. x 比所有结尾都大，LIS 变长
            else:
                g[j] = x  # 4. 让长度为 j+1 的子序列结尾更小
        return len(g)
```

**复杂度**：时间 O(n log n)；空间 O(n)。

<br>

#### 1143. 最长公共子序列 / Longest Common Subsequence

[CN](https://leetcode.cn/problems/longest-common-subsequence/) · [EN](https://leetcode.com/problems/longest-common-subsequence/) · Medium

**题意**：求两个字符串的最长公共子序列长度。

**思路**：f[i][j] 表示 text1 前 i 个字符和 text2 前 j 个字符的 LCS 长度。最后一个字符相同：f[i-1][j-1] + 1；不同：max(f[i-1][j], f[i][j-1])。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        # 1. f[i][j]：text1[:i] 和 text2[:j] 的 LCS，多一行一列作为边界
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i, x in enumerate(text1):
            for j, y in enumerate(text2):
                # 2. 最后一个字符相同，一起选上
                if x == y:
                    f[i + 1][j + 1] = f[i][j] + 1
                # 3. 不同，去掉其中一个
                else:
                    f[i + 1][j + 1] = max(f[i][j + 1], f[i + 1][j])
        return f[m][n]
```

**复杂度**：时间 O(mn)；空间 O(mn)，可以优化到 O(n)。

<br>

#### 72. 编辑距离 / Edit Distance

[CN](https://leetcode.cn/problems/edit-distance/) · [EN](https://leetcode.com/problems/edit-distance/) · Medium

**题意**：用插入、删除、替换三种操作，求把 word1 变成 word2 的最少操作数。

**思路**：f[i][j] 表示 word1 前 i 个字符变成 word2 前 j 个字符的最少操作数。最后一个字符相同时不用操作：f[i-1][j-1]；不同时取插入 f[i][j-1]、删除 f[i-1][j]、替换 f[i-1][j-1] 的最小值加一。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        # 1. 边界：空串变成长度为 j 的串需要 j 次插入；反之需要 i 次删除
        f[0] = list(range(n + 1))
        for i, x in enumerate(word1):
            f[i + 1][0] = i + 1
            for j, y in enumerate(word2):
                # 2. 字符相同，不需要操作
                if x == y:
                    f[i + 1][j + 1] = f[i][j]
                # 3. 删除 / 插入 / 替换取最小，再加一次操作
                else:
                    f[i + 1][j + 1] = min(f[i][j + 1], f[i + 1][j], f[i][j]) + 1
        return f[m][n]
```

**复杂度**：时间 O(mn)；空间 O(mn)。

<br>

#### 5. 最长回文子串 / Longest Palindromic Substring

[CN](https://leetcode.cn/problems/longest-palindromic-substring/) · [EN](https://leetcode.com/problems/longest-palindromic-substring/) · Medium

**题意**：求字符串中最长的回文子串。

**思路**：中心扩展：回文串的中心可能是一个字符（奇数长度）或两个字符之间（偶数长度），共 2n-1 个中心。从每个中心向两边扩展，记录最长的。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        ans_left = ans_right = 0
        # 1. 枚举 2n-1 个回文中心：i 为偶数时中心是字符，为奇数时是两个字符之间
        for i in range(2 * n - 1):
            left, right = i // 2, (i + 1) // 2
            # 2. 向两边扩展
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            # 3. 循环结束时回文串是 s[left+1: right]
            if right - left - 1 > ans_right - ans_left:
                ans_left, ans_right = left + 1, right
        return s[ans_left: ans_right]
```

**复杂度**：时间 O(n²)；空间 O(1)。Manacher 算法可以做到 O(n)。

<br>

#### 139. 单词拆分 / Word Break

[CN](https://leetcode.cn/problems/word-break/) · [EN](https://leetcode.com/problems/word-break/) · Medium

**题意**：判断字符串能否拆分成字典中的单词（单词可重复使用）。

**思路**：划分型 DP：f[i] 表示 s 的前 i 个字符能否被拆分。枚举最后一个单词的起点 j，如果 f[j] 为真且 s[j:i] 在字典里，f[i] 就为真。最后一个单词的长度不会超过字典中最长单词，以此缩小枚举范围。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        max_len = max(map(len, words))
        n = len(s)
        # 1. f[i] = s[:i] 能否被拆分；空串可以
        f = [True] + [False] * n
        for i in range(1, n + 1):
            # 2. 枚举最后一个单词 s[j:i]，长度不超过 max_len
            for j in range(i - 1, max(i - max_len - 1, -1), -1):
                if f[j] and s[j:i] in words:
                    f[i] = True
                    break
        return f[n]
```

**复杂度**：时间 O(n·L²)，L 是最长单词长度（切片和哈希需要 O(L)）；空间 O(n + 字典总长)。

<br>

### 状态机（股票系列）

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **121** | **买卖股票的最佳时机** | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | [题解 ↓](#121-买卖股票的最佳时机--best-time-to-buy-and-sell-stock) |
| **122** | **买卖股票的最佳时机 II** | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) | [题解 ↓](#122-买卖股票的最佳时机-ii--best-time-to-buy-and-sell-stock-ii) |
| 123 | 买卖股票的最佳时机 III | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) | 最多 2 笔：buy1 / sell1 / buy2 / sell2 四个状态 |
| 188 | 买卖股票的最佳时机 IV | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) | 最多 k 笔：f[j][0/1] 表示已完成 j 笔、是否持有 |
| **309** | **买卖股票的最佳时机含冷冻期** | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | [题解 ↓](#309-买卖股票的最佳时机含冷冻期--best-time-to-buy-and-sell-stock-with-cooldown) |
| 714 | 买卖股票的最佳时机含手续费 | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) | 122 的状态机，卖出时减去手续费 |

<br>

#### 121. 买卖股票的最佳时机 / Best Time to Buy and Sell Stock

[CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) · Easy

**题意**：只能买卖一次股票，求最大利润。

**思路**：枚举卖出的那一天，买入价一定是之前的最低价。遍历时维护前缀最小值即可。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        min_price = prices[0]  # 1. 到目前为止的最低价格
        for p in prices:
            # 2. 今天卖出的最大利润
            ans = max(ans, p - min_price)
            # 3. 更新最低价格
            min_price = min(min_price, p)
        return ans
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 122. 买卖股票的最佳时机 II / Best Time to Buy and Sell Stock II

[CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) · Medium

**题意**：可以多次买卖股票（同一时间最多持有一股），求最大利润。

**思路**：状态机 DP：每天结束时有两种状态，f0 不持有股票，f1 持有股票。f0 = max(f0, f1 + p)（昨天就不持有，或今天卖出）；f1 = max(f1, f0 - p)（昨天就持有，或今天买入）。贪心的看法：把所有上涨的差价都赚到。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 1. f0：不持有股票的最大利润；f1：持有股票的最大利润
        f0, f1 = 0, -inf
        for p in prices:
            # 2. 状态转移（右边用的都是昨天的值）
            f0, f1 = max(f0, f1 + p), max(f1, f0 - p)
        # 3. 最后不持有股票才是最大利润
        return f0
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 309. 买卖股票的最佳时机含冷冻期 / Best Time to Buy and Sell Stock with Cooldown

[CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) · Medium

**题意**：可以多次买卖，但卖出后第二天不能买入（冷冻期一天），求最大利润。

**思路**：在 122 的基础上，卖出后要冷冻一天，所以买入要从「前天不持有」转移：f1[i] = max(f1[i-1], f0[i-2] - p)。用三个变量滚动：pre0 是前天的 f0。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 1. pre0 = 前天不持有，f0 = 昨天不持有，f1 = 昨天持有
        pre0, f0, f1 = 0, 0, -inf
        for p in prices:
            # 2. 卖出：f1 + p；买入只能从前天不持有转移（冷冻期）
            pre0, f0, f1 = f0, max(f0, f1 + p), max(f1, pre0 - p)
        return f0
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>
<br>

## 7. 常用数据结构

用合适的结构把暴力的某一层循环降成 O(1) 或 O(log n)。

<br>

### 前缀和

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 303 | 区域和检索 - 数组不可变 | [CN](https://leetcode.cn/problems/range-sum-query-immutable/) · [EN](https://leetcode.com/problems/range-sum-query-immutable/) | 前缀和：sum(l..r) = s[r+1] - s[l] |
| **560** | **和为 K 的子数组** | [CN](https://leetcode.cn/problems/subarray-sum-equals-k/) · [EN](https://leetcode.com/problems/subarray-sum-equals-k/) | [题解 ↓](#560-和为-k-的子数组--subarray-sum-equals-k) |
| **238** | **除自身以外数组的乘积** | [CN](https://leetcode.cn/problems/product-of-array-except-self/) · [EN](https://leetcode.com/problems/product-of-array-except-self/) | [题解 ↓](#238-除自身以外数组的乘积--product-of-array-except-self) |
| 523 | 连续的子数组和 | [CN](https://leetcode.cn/problems/continuous-subarray-sum/) · [EN](https://leetcode.com/problems/continuous-subarray-sum/) | 前缀和模 k 相同、且下标相差 ≥ 2 |
| 974 | 和可被 K 整除的子数组 | [CN](https://leetcode.cn/problems/subarray-sums-divisible-by-k/) · [EN](https://leetcode.com/problems/subarray-sums-divisible-by-k/) | 统计前缀和模 k 相同的对数（Python 的 % 已非负） |
| 525 | 连续数组 | [CN](https://leetcode.cn/problems/contiguous-array/) · [EN](https://leetcode.com/problems/contiguous-array/) | 0 看成 -1，求和为 0 的最长子数组：记录每个前缀和第一次出现的位置 |

<br>

#### 560. 和为 K 的子数组 / Subarray Sum Equals K

[CN](https://leetcode.cn/problems/subarray-sum-equals-k/) · [EN](https://leetcode.com/problems/subarray-sum-equals-k/) · Medium

**题意**：求和为 k 的连续子数组个数（数组可能有负数）。

**思路**：子数组和 = s[j] - s[i]（前缀和之差）。枚举右端点 j，需要统计左边有多少个 s[i] = s[j] - k，用哈希表记录每个前缀和出现的次数。数组有负数，不能用滑动窗口。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = s = 0
        cnt = defaultdict(int)
        cnt[0] = 1  # 1. 空前缀的和为 0
        for x in nums:
            # 2. s 是当前前缀和
            s += x
            # 3. 左边有多少个前缀和等于 s - k
            ans += cnt[s - k]
            # 4. 记录当前前缀和
            cnt[s] += 1
        return ans
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 238. 除自身以外数组的乘积 / Product of Array Except Self

[CN](https://leetcode.cn/problems/product-of-array-except-self/) · [EN](https://leetcode.com/problems/product-of-array-except-self/) · Medium

**题意**：求数组中除自身以外所有元素的乘积，不能用除法，要求 O(n)。

**思路**：answer[i] = 左边所有数的乘积 × 右边所有数的乘积。先从左往右把前缀积写进答案数组，再从右往左用一个变量维护后缀积乘上去，不用除法，额外空间 O(1)。

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [1] * n
        # 1. ans[i] = nums[0] * ... * nums[i-1]（前缀积）
        for i in range(1, n):
            ans[i] = ans[i - 1] * nums[i - 1]
        # 2. 从右往左乘上后缀积
        suf = 1
        for i in range(n - 1, -1, -1):
            ans[i] *= suf
            suf *= nums[i]
        return ans
```

**复杂度**：时间 O(n)；空间 O(1)（不计答案）。

<br>

### 栈

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **20** | **有效的括号** | [CN](https://leetcode.cn/problems/valid-parentheses/) · [EN](https://leetcode.com/problems/valid-parentheses/) | [题解 ↓](#20-有效的括号--valid-parentheses) |
| **155** | **最小栈** | [CN](https://leetcode.cn/problems/min-stack/) · [EN](https://leetcode.com/problems/min-stack/) | [题解 ↓](#155-最小栈--min-stack) |
| 150 | 逆波兰表达式求值 | [CN](https://leetcode.cn/problems/evaluate-reverse-polish-notation/) · [EN](https://leetcode.com/problems/evaluate-reverse-polish-notation/) | 数字入栈，遇到运算符弹出两个计算；除法向零取整 |
| **394** | **字符串解码** | [CN](https://leetcode.cn/problems/decode-string/) · [EN](https://leetcode.com/problems/decode-string/) | [题解 ↓](#394-字符串解码--decode-string) |
| 71 | 简化路径 | [CN](https://leetcode.cn/problems/simplify-path/) · [EN](https://leetcode.com/problems/simplify-path/) | 按 / 分割：.. 出栈，. 和空串跳过，其余入栈 |
| **227** | **基本计算器 II** | [CN](https://leetcode.cn/problems/basic-calculator-ii/) · [EN](https://leetcode.com/problems/basic-calculator-ii/) | [题解 ↓](#227-基本计算器-ii--basic-calculator-ii) |

<br>

#### 20. 有效的括号 / Valid Parentheses

[CN](https://leetcode.cn/problems/valid-parentheses/) · [EN](https://leetcode.com/problems/valid-parentheses/) · Easy

**题意**：判断只含 ()[]{} 的括号字符串是否有效。

**思路**：左括号入栈；遇到右括号时，栈顶必须是对应的左括号，否则不合法。最后栈必须为空。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        # 1. 长度为奇数一定不合法
        if len(s) % 2:
            return False
        pairs = {')': '(', ']': '[', '}': '{'}
        st = []
        for c in s:
            # 2. 左括号入栈
            if c not in pairs:
                st.append(c)
            # 3. 右括号必须和栈顶匹配
            elif not st or st.pop() != pairs[c]:
                return False
        # 4. 所有左括号都要被匹配
        return not st
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 155. 最小栈 / Min Stack

[CN](https://leetcode.cn/problems/min-stack/) · [EN](https://leetcode.com/problems/min-stack/) · Medium

**题意**：实现一个栈，支持 push、pop、top，并能在 O(1) 时间取到最小值。

**思路**：栈里每个元素存 (值, 入栈时的最小值)。最小值只和当前元素及其下面的元素有关，出栈后下面元素记录的最小值仍然正确。

```python
class MinStack:
    def __init__(self):
        # 1. 栈中存 (val, 从栈底到这个元素的最小值)；哨兵避免判空
        self.st = [(0, inf)]

    def push(self, val: int) -> None:
        # 2. 新的最小值 = min(val, 之前的最小值)
        self.st.append((val, min(val, self.st[-1][1])))

    def pop(self) -> None:
        self.st.pop()

    def top(self) -> int:
        return self.st[-1][0]

    def getMin(self) -> int:
        # 3. 栈顶记录的就是当前最小值
        return self.st[-1][1]
```

**复杂度**：所有操作 O(1)；空间 O(n)。

<br>

#### 394. 字符串解码 / Decode String

[CN](https://leetcode.cn/problems/decode-string/) · [EN](https://leetcode.com/problems/decode-string/) · Medium

**题意**：解码形如 3[a2[c]] 的字符串，k[s] 表示 s 重复 k 次。

**思路**：用栈处理嵌套。遇到 '[' 时把「当前已解码的字符串、重复次数」入栈，然后开始解码括号内部；遇到 ']' 时出栈，把括号内的结果重复 k 次接到之前的字符串后面。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        st = []  # 存 (进入括号前的字符串, 括号的重复次数)
        res, k = '', 0
        for c in s:
            if c.isdigit():
                # 1. 次数可能是多位数
                k = k * 10 + int(c)
            elif c == '[':
                # 2. 进入一层括号：保存现场，重新开始
                st.append((res, k))
                res, k = '', 0
            elif c == ']':
                # 3. 离开括号：括号内的结果重复 k 次，接到之前的字符串后面
                pre, cnt = st.pop()
                res = pre + res * cnt
            else:
                # 4. 普通字母
                res += c
        return res
```

**复杂度**：时间 O(答案长度)；空间 O(答案长度)。

<br>

#### 227. 基本计算器 II / Basic Calculator II

[CN](https://leetcode.cn/problems/basic-calculator-ii/) · [EN](https://leetcode.com/problems/basic-calculator-ii/) · Medium

**题意**：计算只含非负整数、+ - * / 和空格的表达式的值，除法向零取整。

**思路**：乘除优先级高，先算。遍历时记录上一个运算符 op：遇到一个完整的数后，op 是 + 就把数入栈，- 就把相反数入栈，* 和 / 就和栈顶计算后替换栈顶。最后栈中所有数求和。

```python
class Solution:
    def calculate(self, s: str) -> int:
        st = []
        num = 0
        op = '+'  # 1. 当前数前面的运算符
        for i, c in enumerate(s):
            if c.isdigit():
                num = num * 10 + int(c)
            # 2. 遇到运算符或到达末尾，处理 op 和 num
            if c in '+-*/' or i == len(s) - 1:
                if op == '+':
                    st.append(num)
                elif op == '-':
                    st.append(-num)
                elif op == '*':
                    st.append(st.pop() * num)
                else:
                    st.append(int(st.pop() / num))  # 3. 向零取整
                op = c
                num = 0
        # 4. 剩下的只有加法
        return sum(st)
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

### 队列

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 232 | 用栈实现队列 | [CN](https://leetcode.cn/problems/implement-queue-using-stacks/) · [EN](https://leetcode.com/problems/implement-queue-using-stacks/) | 输入栈 + 输出栈，输出栈空时把输入栈倒过去 |
| 225 | 用队列实现栈 | [CN](https://leetcode.cn/problems/implement-stack-using-queues/) · [EN](https://leetcode.com/problems/implement-stack-using-queues/) | push 后把前面的元素依次出队再入队 |
| 933 | 最近的请求次数 | [CN](https://leetcode.cn/problems/number-of-recent-calls/) · [EN](https://leetcode.com/problems/number-of-recent-calls/) | 队列存时间戳，弹出早于 t - 3000 的 |

<br>

### 堆

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **215** | **数组中的第K个最大元素** | [CN](https://leetcode.cn/problems/kth-largest-element-in-an-array/) · [EN](https://leetcode.com/problems/kth-largest-element-in-an-array/) | [题解 ↓](#215-数组中的第k个最大元素--kth-largest-element-in-an-array) |
| **347** | **前 K 个高频元素** | [CN](https://leetcode.cn/problems/top-k-frequent-elements/) · [EN](https://leetcode.com/problems/top-k-frequent-elements/) | [题解 ↓](#347-前-k-个高频元素--top-k-frequent-elements) |
| **23** | **合并 K 个升序链表** | [CN](https://leetcode.cn/problems/merge-k-sorted-lists/) · [EN](https://leetcode.com/problems/merge-k-sorted-lists/) | [题解 ↓](#23-合并-k-个升序链表--merge-k-sorted-lists) |
| **295** | **数据流的中位数** | [CN](https://leetcode.cn/problems/find-median-from-data-stream/) · [EN](https://leetcode.com/problems/find-median-from-data-stream/) | [题解 ↓](#295-数据流的中位数--find-median-from-data-stream) |
| 373 | 查找和最小的 K 对数字 | [CN](https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/) · [EN](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/) | 小顶堆初始放入 (i, 0)，弹出 (i, j) 后放入 (i, j+1) |
| 621 | 任务调度器 | [CN](https://leetcode.cn/problems/task-scheduler/) · [EN](https://leetcode.com/problems/task-scheduler/) | 答案 = max(n_tasks, (maxCnt-1)·(n+1) + 次数等于 maxCnt 的任务数) |

<br>

#### 215. 数组中的第K个最大元素 / Kth Largest Element in an Array

[CN](https://leetcode.cn/problems/kth-largest-element-in-an-array/) · [EN](https://leetcode.com/problems/kth-largest-element-in-an-array/) · Medium

**题意**：求数组中第 k 个最大的元素，要求 O(n)。

**思路**：快速选择：随机选一个 pivot，把数分成大于、等于、小于 pivot 三组。根据 k 落在哪一组，只递归那一组。三路划分可以防止大量重复元素时退化。也可以用大小为 k 的小顶堆，时间 O(n log k)。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 1. 随机选 pivot，三路划分
        pivot = random.choice(nums)
        big = [x for x in nums if x > pivot]
        equal = [x for x in nums if x == pivot]
        small = [x for x in nums if x < pivot]
        # 2. 第 k 大在比 pivot 大的那组里
        if k <= len(big):
            return self.findKthLargest(big, k)
        # 3. 第 k 大在比 pivot 小的那组里
        if k > len(big) + len(equal):
            return self.findKthLargest(small, k - len(big) - len(equal))
        # 4. 否则就是 pivot
        return pivot
```

**复杂度**：期望时间 O(n)；空间 O(n)。

<br>

#### 347. 前 K 个高频元素 / Top K Frequent Elements

[CN](https://leetcode.cn/problems/top-k-frequent-elements/) · [EN](https://leetcode.com/problems/top-k-frequent-elements/) · Medium

**题意**：返回出现频率前 k 高的元素。

**思路**：先统计频率。出现次数最多是 n，按频率做桶排序：buckets[c] 放出现 c 次的元素，从高频桶往低频桶取，取够 k 个为止。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 1. 统计每个元素的出现次数
        cnt = Counter(nums)
        # 2. 桶排序：buckets[c] = 出现 c 次的元素
        max_cnt = max(cnt.values())
        buckets = [[] for _ in range(max_cnt + 1)]
        for x, c in cnt.items():
            buckets[c].append(x)
        # 3. 从高频到低频取 k 个（题目保证答案唯一）
        ans = []
        for bucket in reversed(buckets):
            ans += bucket
            if len(ans) == k:
                return ans
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 23. 合并 K 个升序链表 / Merge k Sorted Lists

[CN](https://leetcode.cn/problems/merge-k-sorted-lists/) · [EN](https://leetcode.com/problems/merge-k-sorted-lists/) · Hard

**题意**：合并 k 个升序链表。

**思路**：最小堆：把每条链表的头节点放进堆，每次弹出最小的节点接到答案后面，再把它的下一个节点放进堆。堆中存 (val, 编号, 节点)，编号避免值相同时比较节点。

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 1. 把每条链表的头节点放进最小堆
        h = [(head.val, i, head) for i, head in enumerate(lists) if head]
        heapify(h)
        dummy = cur = ListNode()  # 哨兵节点
        while h:
            # 2. 弹出当前最小的节点，接到答案后面
            _, i, node = heappop(h)
            cur.next = node
            cur = cur.next
            # 3. 把它的下一个节点放进堆
            if node.next:
                heappush(h, (node.next.val, i, node.next))
        return dummy.next
```

**复杂度**：时间 O(N log k)，N 是节点总数；空间 O(k)。

<br>

#### 295. 数据流的中位数 / Find Median from Data Stream

[CN](https://leetcode.cn/problems/find-median-from-data-stream/) · [EN](https://leetcode.com/problems/find-median-from-data-stream/) · Hard

**题意**：数据流中不断添加整数，随时返回当前所有数的中位数。

**思路**：对顶堆：left 是大顶堆（存较小的一半，Python 用取负模拟），right 是小顶堆（存较大的一半）。保持 len(left) 等于 len(right) 或多一个。添加时先放进一边再把那边的堆顶挪到另一边，保证 left 的所有数 ≤ right 的所有数。

```python
class MedianFinder:
    def __init__(self):
        self.left = []   # 1. 大顶堆（存相反数），存较小的一半
        self.right = []  # 小顶堆，存较大的一半

    def addNum(self, num: int) -> None:
        # 2. 两边一样多：num 先进 right，再把 right 的最小值移到 left
        if len(self.left) == len(self.right):
            heappush(self.left, -heappushpop(self.right, num))
        # 3. left 多一个：num 先进 left，再把 left 的最大值移到 right
        else:
            heappush(self.right, -heappushpop(self.left, -num))

    def findMedian(self) -> float:
        # 4. 奇数个时中位数是 left 的堆顶；偶数个时是两个堆顶的平均值
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (self.right[0] - self.left[0]) / 2
```

**复杂度**：addNum O(log n)，findMedian O(1)；空间 O(n)。

<br>

### 哈希表

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **1** | **两数之和** | [CN](https://leetcode.cn/problems/two-sum/) · [EN](https://leetcode.com/problems/two-sum/) | [题解 ↓](#1-两数之和--two-sum) |
| **49** | **字母异位词分组** | [CN](https://leetcode.cn/problems/group-anagrams/) · [EN](https://leetcode.com/problems/group-anagrams/) | [题解 ↓](#49-字母异位词分组--group-anagrams) |
| **128** | **最长连续序列** | [CN](https://leetcode.cn/problems/longest-consecutive-sequence/) · [EN](https://leetcode.com/problems/longest-consecutive-sequence/) | [题解 ↓](#128-最长连续序列--longest-consecutive-sequence) |
| 41 | 缺失的第一个正数 | [CN](https://leetcode.cn/problems/first-missing-positive/) · [EN](https://leetcode.com/problems/first-missing-positive/) | 原地哈希：把 x 交换到下标 x-1，再找第一个不对的位置 |
| **146** | **LRU 缓存** | [CN](https://leetcode.cn/problems/lru-cache/) · [EN](https://leetcode.com/problems/lru-cache/) | [题解 ↓](#146-lru-缓存--lru-cache) |

<br>

#### 1. 两数之和 / Two Sum

[CN](https://leetcode.cn/problems/two-sum/) · [EN](https://leetcode.com/problems/two-sum/) · Easy

**题意**：找出和为 target 的两个数的下标。

**思路**：枚举右边的数 x，问左边有没有 target - x。用哈希表记录「值 → 下标」，边遍历边查边存。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        idx = {}  # 1. 值 -> 下标
        for j, x in enumerate(nums):
            # 2. 左边有没有 target - x
            if target - x in idx:
                return [idx[target - x], j]
            # 3. 记录当前数
            idx[x] = j
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 49. 字母异位词分组 / Group Anagrams

[CN](https://leetcode.cn/problems/group-anagrams/) · [EN](https://leetcode.com/problems/group-anagrams/) · Medium

**题意**：把字母异位词分到同一组。

**思路**：字母异位词排序后相同。用排序后的字符串作为哈希表的 key 分组。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for s in strs:
            # 1. 排序后的字符串作为 key
            groups[''.join(sorted(s))].append(s)
        # 2. 每组就是一类异位词
        return list(groups.values())
```

**复杂度**：时间 O(n·L log L)；空间 O(n·L)。

<br>

#### 128. 最长连续序列 / Longest Consecutive Sequence

[CN](https://leetcode.cn/problems/longest-consecutive-sequence/) · [EN](https://leetcode.com/problems/longest-consecutive-sequence/) · Medium

**题意**：求未排序数组中数字连续的最长序列的长度，要求 O(n)。

**思路**：把数放进集合。只从序列的起点 x（x-1 不在集合中）开始往后数 x+1, x+2, …。每个数最多被数一次，总时间 O(n)。如果当前最长已经 ≥ 集合大小的一半，后面不可能更长，可以提前退出。

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        st = set(nums)
        ans = 0
        for x in st:
            # 1. x 不是序列起点，跳过
            if x - 1 in st:
                continue
            # 2. 从起点 x 往后数
            y = x + 1
            while y in st:
                y += 1
            # 3. 序列 [x, y-1] 的长度
            ans = max(ans, y - x)
            if ans * 2 >= len(st):  # 4. 不可能有更长的了
                break
        return ans
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 146. LRU 缓存 / LRU Cache

[CN](https://leetcode.cn/problems/lru-cache/) · [EN](https://leetcode.com/problems/lru-cache/) · Medium

**题意**：实现 LRU 缓存，get 和 put 都要 O(1)，超出容量时淘汰最久未使用的。

**思路**：把缓存想象成一摞书：哈希表存 key → 节点，双向链表按使用时间排序，最近用过的放最上面（链表头）。get 和 put 时把节点抽出来放到最上面；超出容量时删除最下面的那本（链表尾）。用一个哨兵节点 dummy 把链表连成环，dummy.next 是头，dummy.prev 是尾。

```python
class Node:
    __slots__ = 'prev', 'next', 'key', 'value'

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # 1. 哨兵节点，链表成环：dummy.next 是最近使用的，dummy.prev 是最久未使用的
        self.dummy = Node()
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.key_to_node = {}

    # 2. 删除节点
    def remove(self, x: Node) -> None:
        x.prev.next = x.next
        x.next.prev = x.prev

    # 3. 在链表头插入节点（放到最上面）
    def push_front(self, x: Node) -> None:
        x.prev = self.dummy
        x.next = self.dummy.next
        x.prev.next = x
        x.next.prev = x

    # 4. 取出节点并放到最上面；不存在返回 None
    def get_node(self, key: int) -> Optional[Node]:
        if key not in self.key_to_node:
            return None
        node = self.key_to_node[key]
        self.remove(node)
        self.push_front(node)
        return node

    def get(self, key: int) -> int:
        node = self.get_node(key)
        return node.value if node else -1

    def put(self, key: int, value: int) -> None:
        # 5. 已存在：更新值（get_node 已经放到最上面）
        node = self.get_node(key)
        if node:
            node.value = value
            return
        # 6. 不存在：新建并放到最上面
        node = Node(key, value)
        self.key_to_node[key] = node
        self.push_front(node)
        # 7. 超出容量，删除最下面的
        if len(self.key_to_node) > self.capacity:
            back = self.dummy.prev
            del self.key_to_node[back.key]
            self.remove(back)
```

**复杂度**：get 和 put 都是 O(1)；空间 O(capacity)。

<br>

### 字典树

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **208** | **实现 Trie (前缀树)** | [CN](https://leetcode.cn/problems/implement-trie-prefix-tree/) · [EN](https://leetcode.com/problems/implement-trie-prefix-tree/) | [题解 ↓](#208-实现-trie-前缀树--implement-trie-prefix-tree) |
| 211 | 添加与搜索单词 - 数据结构设计 | [CN](https://leetcode.cn/problems/design-add-and-search-words-data-structure/) · [EN](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | Trie；遇到 . 时 DFS 所有孩子 |
| 212 | 单词搜索 II | [CN](https://leetcode.cn/problems/word-search-ii/) · [EN](https://leetcode.com/problems/word-search-ii/) | 单词建 Trie，网格回溯时沿 Trie 往下走，找到单词后删除以剪枝 |

<br>

#### 208. 实现 Trie (前缀树) / Implement Trie (Prefix Tree)

[CN](https://leetcode.cn/problems/implement-trie-prefix-tree/) · [EN](https://leetcode.com/problems/implement-trie-prefix-tree/) · Medium

**题意**：实现前缀树，支持 insert、search、startsWith。

**思路**：每个节点有 26 个孩子（这里用字典）和一个 end 标记。insert 沿路创建节点；search 和 startsWith 沿路查找，区别是 search 要求最后的节点 end 为真。

```python
class Node:
    __slots__ = 'son', 'end'

    def __init__(self):
        self.son = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            # 1. 没有这个孩子就创建
            if c not in cur.son:
                cur.son[c] = Node()
            cur = cur.son[c]
        # 2. 标记单词结尾
        cur.end = True

    # 3. 返回 0：不存在；1：是前缀；2：是完整单词
    def find(self, word: str) -> int:
        cur = self.root
        for c in word:
            if c not in cur.son:
                return 0
            cur = cur.son[c]
        return 2 if cur.end else 1

    def search(self, word: str) -> bool:
        return self.find(word) == 2

    def startsWith(self, prefix: str) -> bool:
        return self.find(prefix) != 0
```

**复杂度**：每次操作 O(L)；空间 O(所有单词总长 × |Σ|)。

<br>

### 并查集

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 547 | 省份数量 | [CN](https://leetcode.cn/problems/number-of-provinces/) · [EN](https://leetcode.com/problems/number-of-provinces/) | 并查集合并相连的城市，数根的个数，[题解](#547-省份数量--number-of-provinces) |
| **684** | **冗余连接** | [CN](https://leetcode.cn/problems/redundant-connection/) · [EN](https://leetcode.com/problems/redundant-connection/) | [题解](#684-冗余连接--redundant-connection) |
| 721 | 账户合并 | [CN](https://leetcode.cn/problems/accounts-merge/) · [EN](https://leetcode.com/problems/accounts-merge/) | 同一账户的邮箱合并到一个集合，再按根分组，[题解](#721-账户合并--accounts-merge) |
| 990 | 等式方程的可满足性 | [CN](https://leetcode.cn/problems/satisfiability-of-equality-equations/) · [EN](https://leetcode.com/problems/satisfiability-of-equality-equations/) | 先合并所有 ==，再检查每个 != 的两端不在同一集合 |

<br>
<br>

## 8. 链表、树与回溯

链表画图 + 哨兵；树用递归定义子问题；回溯 = 选择 → 递归 → 撤销。

<br>

### 链表

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **206** | **反转链表** | [CN](https://leetcode.cn/problems/reverse-linked-list/) · [EN](https://leetcode.com/problems/reverse-linked-list/) | [题解 ↓](#206-反转链表--reverse-linked-list) |
| **92** | **反转链表 II** | [CN](https://leetcode.cn/problems/reverse-linked-list-ii/) · [EN](https://leetcode.com/problems/reverse-linked-list-ii/) | [题解 ↓](#92-反转链表-ii--reverse-linked-list-ii) |
| **25** | **K 个一组翻转链表** | [CN](https://leetcode.cn/problems/reverse-nodes-in-k-group/) · [EN](https://leetcode.com/problems/reverse-nodes-in-k-group/) | [题解 ↓](#25-k-个一组翻转链表--reverse-nodes-in-k-group) |
| **21** | **合并两个有序链表** | [CN](https://leetcode.cn/problems/merge-two-sorted-lists/) · [EN](https://leetcode.com/problems/merge-two-sorted-lists/) | [题解 ↓](#21-合并两个有序链表--merge-two-sorted-lists) |
| 24 | 两两交换链表中的节点 | [CN](https://leetcode.cn/problems/swap-nodes-in-pairs/) · [EN](https://leetcode.com/problems/swap-nodes-in-pairs/) | 哨兵节点 + 每次交换相邻两个节点 |
| **19** | **删除链表的倒数第 N 个结点** | [CN](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) · [EN](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) | [题解 ↓](#19-删除链表的倒数第-n-个结点--remove-nth-node-from-end-of-list) |
| **141** | **环形链表** | [CN](https://leetcode.cn/problems/linked-list-cycle/) · [EN](https://leetcode.com/problems/linked-list-cycle/) | [题解 ↓](#141-环形链表--linked-list-cycle) |
| **142** | **环形链表 II** | [CN](https://leetcode.cn/problems/linked-list-cycle-ii/) · [EN](https://leetcode.com/problems/linked-list-cycle-ii/) | [题解 ↓](#142-环形链表-ii--linked-list-cycle-ii) |
| 876 | 链表的中间结点 | [CN](https://leetcode.cn/problems/middle-of-the-linked-list/) · [EN](https://leetcode.com/problems/middle-of-the-linked-list/) | 快慢指针，快指针走到底时慢指针在中间 |
| 234 | 回文链表 | [CN](https://leetcode.cn/problems/palindrome-linked-list/) · [EN](https://leetcode.com/problems/palindrome-linked-list/) | 找中点 + 反转后半段 + 逐个比较 |
| **160** | **相交链表** | [CN](https://leetcode.cn/problems/intersection-of-two-linked-lists/) · [EN](https://leetcode.com/problems/intersection-of-two-linked-lists/) | [题解 ↓](#160-相交链表--intersection-of-two-linked-lists) |
| **2** | **两数相加** | [CN](https://leetcode.cn/problems/add-two-numbers/) · [EN](https://leetcode.com/problems/add-two-numbers/) | [题解 ↓](#2-两数相加--add-two-numbers) |
| 143 | 重排链表 | [CN](https://leetcode.cn/problems/reorder-list/) · [EN](https://leetcode.com/problems/reorder-list/) | 找中点 + 反转后半段 + 交错合并 |
| **148** | **排序链表** | [CN](https://leetcode.cn/problems/sort-list/) · [EN](https://leetcode.com/problems/sort-list/) | [题解 ↓](#148-排序链表--sort-list) |
| 138 | 随机链表的复制 | [CN](https://leetcode.cn/problems/copy-list-with-random-pointer/) · [EN](https://leetcode.com/problems/copy-list-with-random-pointer/) | 每个节点后插入复制节点，设置 random，再拆成两条链表 |

<br>

#### 206. 反转链表 / Reverse Linked List

[CN](https://leetcode.cn/problems/reverse-linked-list/) · [EN](https://leetcode.com/problems/reverse-linked-list/) · Easy

**题意**：反转单链表。

**思路**：头插法：维护已反转部分的头 pre。遍历原链表，把当前节点 cur 的 next 指向 pre，再整体前进。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre, cur = None, head
        while cur:
            # 1. 先记下下一个节点
            nxt = cur.next
            # 2. 当前节点指向已反转部分
            cur.next = pre
            # 3. 整体前进
            pre = cur
            cur = nxt
        # 4. 循环结束时 pre 是新的头节点
        return pre
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 92. 反转链表 II / Reverse Linked List II

[CN](https://leetcode.cn/problems/reverse-linked-list-ii/) · [EN](https://leetcode.com/problems/reverse-linked-list-ii/) · Medium

**题意**：反转链表中从第 left 个到第 right 个节点的部分。

**思路**：先走到第 left-1 个节点 p0（用哨兵处理 left = 1）。从 p0.next 开始用 206 的方法反转 right-left+1 个节点。反转后 pre 是这段的新头，cur 是这段后面的节点，p0.next 是这段的新尾，把三者重新接好。

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # 1. 哨兵节点，p0 走到要反转部分的前一个节点
        p0 = dummy = ListNode(next=head)
        for _ in range(left - 1):
            p0 = p0.next

        # 2. 反转 right-left+1 个节点
        pre, cur = None, p0.next
        for _ in range(right - left + 1):
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt

        # 3. 接回去：原来的第一个节点（现在的尾）指向 cur，p0 指向新头 pre
        p0.next.next = cur
        p0.next = pre
        return dummy.next
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 25. K 个一组翻转链表 / Reverse Nodes in k-Group

[CN](https://leetcode.cn/problems/reverse-nodes-in-k-group/) · [EN](https://leetcode.com/problems/reverse-nodes-in-k-group/) · Hard

**题意**：每 k 个节点一组翻转链表，最后不足 k 个的保持原样。

**思路**：先数出链表长度 n。每次剩余节点 ≥ k 时，用 92 的方法反转 p0 后面的 k 个节点并接好；接好之后，原来的第一个节点变成了这一组的尾，也就是下一组的 p0。

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 1. 统计节点个数
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next

        p0 = dummy = ListNode(next=head)
        pre, cur = None, head
        # 2. 剩余节点够 k 个才反转
        while n >= k:
            n -= k
            # 3. 反转 k 个节点
            for _ in range(k):
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            # 4. 接回去，并让 p0 移到这一组的尾部
            tail = p0.next
            tail.next = cur
            p0.next = pre
            p0 = tail
        return dummy.next
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 21. 合并两个有序链表 / Merge Two Sorted Lists

[CN](https://leetcode.cn/problems/merge-two-sorted-lists/) · [EN](https://leetcode.com/problems/merge-two-sorted-lists/) · Easy

**题意**：合并两个升序链表。

**思路**：哨兵节点 + 双指针：每次把两个链表中较小的头节点接到答案后面。一个链表用完后，把另一个剩下的部分整体接上。

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = dummy = ListNode()  # 1. 哨兵节点
        while list1 and list2:
            # 2. 较小的节点接到后面
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        # 3. 接上剩余部分
        cur.next = list1 or list2
        return dummy.next
```

**复杂度**：时间 O(m + n)；空间 O(1)。

<br>

#### 19. 删除链表的倒数第 N 个结点 / Remove Nth Node From End of List

[CN](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) · [EN](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) · Medium

**题意**：删除链表的倒数第 n 个节点。

**思路**：前后指针：右指针先走 n 步，然后左右指针一起走，右指针到最后一个节点时，左指针正好在倒数第 n+1 个节点，删除它的下一个。用哨兵处理删除头节点的情况。

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 1. 哨兵节点，方便删除头节点
        left = right = dummy = ListNode(next=head)
        # 2. 右指针先走 n 步
        for _ in range(n):
            right = right.next
        # 3. 一起走，直到右指针到达最后一个节点
        while right.next:
            left = left.next
            right = right.next
        # 4. left 的下一个就是倒数第 n 个节点
        left.next = left.next.next
        return dummy.next
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 141. 环形链表 / Linked List Cycle

[CN](https://leetcode.cn/problems/linked-list-cycle/) · [EN](https://leetcode.com/problems/linked-list-cycle/) · Easy

**题意**：判断链表中是否有环。

**思路**：快慢指针：慢指针每次走一步，快指针每次走两步。有环时快指针一定会在环里追上慢指针；无环时快指针会走到末尾。

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        # 1. 快指针能继续走两步
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # 2. 相遇说明有环
            if fast is slow:
                return True
        return False
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 142. 环形链表 II / Linked List Cycle II

[CN](https://leetcode.cn/problems/linked-list-cycle-ii/) · [EN](https://leetcode.com/problems/linked-list-cycle-ii/) · Medium

**题意**：返回链表开始入环的第一个节点，无环返回 null。

**思路**：快慢指针相遇后，设头到入口距离为 a，入口到相遇点为 b，环长为 c。可以推出 a = (k-1)c + (c-b)，即从头节点和相遇点同时出发、每次各走一步，会在入口相遇。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # 1. 快慢指针相遇
            if fast is slow:
                # 2. head 和 slow 同时出发，相遇点就是入口
                while slow is not head:
                    slow = slow.next
                    head = head.next
                return slow
        # 3. 没有环
        return None
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 160. 相交链表 / Intersection of Two Linked Lists

[CN](https://leetcode.cn/problems/intersection-of-two-linked-lists/) · [EN](https://leetcode.com/problems/intersection-of-two-linked-lists/) · Easy

**题意**：返回两个单链表相交的起始节点，不相交返回 null。

**思路**：p 走完 A 再走 B，q 走完 B 再走 A。两者走的总长度都是 a + b + c（c 是公共部分），所以一定同时到达交点；不相交时同时到达 None。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        p, q = headA, headB
        # 1. 两个指针走的总路程相同，必定同时到达交点或 None
        while p is not q:
            # 2. 走到头就换到另一条链表
            p = p.next if p else headB
            q = q.next if q else headA
        return p
```

**复杂度**：时间 O(m + n)；空间 O(1)。

<br>

#### 2. 两数相加 / Add Two Numbers

[CN](https://leetcode.cn/problems/add-two-numbers/) · [EN](https://leetcode.com/problems/add-two-numbers/) · Medium

**题意**：两个逆序存储的非负整数链表相加，以同样形式返回和。

**思路**：链表从低位到高位存储，正好按竖式加法的顺序遍历。每一位相加再加上进位，新节点存个位，进位留给下一位。两个链表都走完、进位也为 0 时结束。

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        cur = dummy = ListNode()  # 1. 哨兵节点
        carry = 0  # 进位
        # 2. 还有数字或者还有进位，就继续
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            # 3. 当前位是 carry % 10，进位是 carry // 10
            cur.next = ListNode(carry % 10)
            cur = cur.next
            carry //= 10
        return dummy.next
```

**复杂度**：时间 O(max(m, n))；空间 O(1)（不计答案）。

<br>

#### 148. 排序链表 / Sort List

[CN](https://leetcode.cn/problems/sort-list/) · [EN](https://leetcode.com/problems/sort-list/) · Medium

**题意**：对链表升序排序，要求 O(n log n)。

**思路**：归并排序：快慢指针找到中点，把链表从中间断开，分别递归排序两半，再用 21 题合并。

```python
class Solution:
    # 1. 找中点并断开，返回后半段的头
    def middle(self, head: ListNode) -> ListNode:
        slow = fast = head
        pre = None
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        pre.next = None  # 断开前后两半
        return slow

    # 2. 合并两个有序链表（21 题）
    def merge(self, a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
        cur = dummy = ListNode()
        while a and b:
            if a.val <= b.val:
                cur.next = a
                a = a.next
            else:
                cur.next = b
                b = b.next
            cur = cur.next
        cur.next = a or b
        return dummy.next

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 3. 空链表或只有一个节点，已经有序
        if head is None or head.next is None:
            return head
        # 4. 分成两半，分别排序再合并
        head2 = self.middle(head)
        return self.merge(self.sortList(head), self.sortList(head2))
```

**复杂度**：时间 O(n log n)；空间 O(log n)（递归栈）。自底向上的归并可以做到 O(1) 空间。

<br>

### 二叉树遍历

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 94 | 二叉树的中序遍历 | [CN](https://leetcode.cn/problems/binary-tree-inorder-traversal/) · [EN](https://leetcode.com/problems/binary-tree-inorder-traversal/) | 递归：左 → 根 → 右；迭代用栈一路向左 |
| 144 | 二叉树的前序遍历 | [CN](https://leetcode.cn/problems/binary-tree-preorder-traversal/) · [EN](https://leetcode.com/problems/binary-tree-preorder-traversal/) | 递归：根 → 左 → 右；迭代时先压右孩子再压左孩子 |
| 145 | 二叉树的后序遍历 | [CN](https://leetcode.cn/problems/binary-tree-postorder-traversal/) · [EN](https://leetcode.com/problems/binary-tree-postorder-traversal/) | 递归：左 → 右 → 根；迭代可按 根 → 右 → 左 遍历再反转 |
| **102** | **二叉树的层序遍历** | [CN](https://leetcode.cn/problems/binary-tree-level-order-traversal/) · [EN](https://leetcode.com/problems/binary-tree-level-order-traversal/) | [题解 ↓](#102-二叉树的层序遍历--binary-tree-level-order-traversal) |
| 103 | 二叉树的锯齿形层序遍历 | [CN](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/) · [EN](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) | 层序遍历，奇数层的结果反转 |
| **199** | **二叉树的右视图** | [CN](https://leetcode.cn/problems/binary-tree-right-side-view/) · [EN](https://leetcode.com/problems/binary-tree-right-side-view/) | [题解 ↓](#199-二叉树的右视图--binary-tree-right-side-view) |
| **104** | **二叉树的最大深度** | [CN](https://leetcode.cn/problems/maximum-depth-of-binary-tree/) · [EN](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | [题解 ↓](#104-二叉树的最大深度--maximum-depth-of-binary-tree) |
| 111 | 二叉树的最小深度 | [CN](https://leetcode.cn/problems/minimum-depth-of-binary-tree/) · [EN](https://leetcode.com/problems/minimum-depth-of-binary-tree/) | 只有一个孩子时不能取 min，要走有孩子的那边 |
| **226** | **翻转二叉树** | [CN](https://leetcode.cn/problems/invert-binary-tree/) · [EN](https://leetcode.com/problems/invert-binary-tree/) | [题解 ↓](#226-翻转二叉树--invert-binary-tree) |
| **101** | **对称二叉树** | [CN](https://leetcode.cn/problems/symmetric-tree/) · [EN](https://leetcode.com/problems/symmetric-tree/) | [题解 ↓](#101-对称二叉树--symmetric-tree) |
| 100 | 相同的树 | [CN](https://leetcode.cn/problems/same-tree/) · [EN](https://leetcode.com/problems/same-tree/) | 同时递归两棵树，根相同且左右子树都相同 |
| 572 | 另一棵树的子树 | [CN](https://leetcode.cn/problems/subtree-of-another-tree/) · [EN](https://leetcode.com/problems/subtree-of-another-tree/) | 对 root 的每个节点调用「相同的树」 |

<br>

#### 102. 二叉树的层序遍历 / Binary Tree Level Order Traversal

[CN](https://leetcode.cn/problems/binary-tree-level-order-traversal/) · [EN](https://leetcode.com/problems/binary-tree-level-order-traversal/) · Medium

**题意**：返回二叉树的层序遍历结果（每层一个列表）。

**思路**：BFS：队列里放当前层的所有节点，每次把一整层取出，记录它们的值，同时把下一层的孩子入队。

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        ans = []
        q = deque([root])
        while q:
            vals = []
            # 1. 当前队列里正好是一整层
            for _ in range(len(q)):
                node = q.popleft()
                vals.append(node.val)
                # 2. 下一层的节点入队
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(vals)
        return ans
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 199. 二叉树的右视图 / Binary Tree Right Side View

[CN](https://leetcode.cn/problems/binary-tree-right-side-view/) · [EN](https://leetcode.com/problems/binary-tree-right-side-view/) · Medium

**题意**：返回从右侧看二叉树时，从上到下能看到的节点值。

**思路**：DFS 先递归右子树再递归左子树。每一层第一个访问到的节点就是从右边看到的节点：递归深度等于答案长度时，说明这一层还没有记录过。

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []

        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if node is None:
                return
            # 1. 这一层第一次访问到节点
            if depth == len(ans):
                ans.append(node.val)
            # 2. 先右后左，保证每层先看到最右边的
            dfs(node.right, depth + 1)
            dfs(node.left, depth + 1)

        dfs(root, 0)
        return ans
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

#### 104. 二叉树的最大深度 / Maximum Depth of Binary Tree

[CN](https://leetcode.cn/problems/maximum-depth-of-binary-tree/) · [EN](https://leetcode.com/problems/maximum-depth-of-binary-tree/) · Easy

**题意**：求二叉树的最大深度。

**思路**：子问题：树的最大深度 = max(左子树深度, 右子树深度) + 1，空节点深度为 0。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 1. 空节点深度为 0
        if root is None:
            return 0
        # 2. 左右子树的深度取最大，再加上自己
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

#### 226. 翻转二叉树 / Invert Binary Tree

[CN](https://leetcode.cn/problems/invert-binary-tree/) · [EN](https://leetcode.com/problems/invert-binary-tree/) · Easy

**题意**：翻转二叉树（交换每个节点的左右子树）。

**思路**：递归翻转左右子树，再交换它们。

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        # 1. 翻转左右子树
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        # 2. 交换
        root.left = right
        root.right = left
        return root
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

#### 101. 对称二叉树 / Symmetric Tree

[CN](https://leetcode.cn/problems/symmetric-tree/) · [EN](https://leetcode.com/problems/symmetric-tree/) · Easy

**题意**：判断二叉树是否轴对称。

**思路**：一棵树对称 ⇔ 左子树和右子树互为镜像。两棵树 p、q 互为镜像：根的值相同，且 p.left 和 q.right 镜像、p.right 和 q.left 镜像。

```python
class Solution:
    # 1. 判断两棵树是否互为镜像
    def isMirror(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None or q is None:
            return p is q  # 都为空才对称
        # 2. 根相同，外侧和外侧比，内侧和内侧比
        return p.val == q.val and self.isMirror(p.left, q.right) and self.isMirror(p.right, q.left)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self.isMirror(root.left, root.right)
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

### 构造与二叉搜索树

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **105** | **从前序与中序遍历序列构造二叉树** | [CN](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) · [EN](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | [题解 ↓](#105-从前序与中序遍历序列构造二叉树--construct-binary-tree-from-preorder-and-inorder-traversal) |
| 106 | 从中序与后序遍历序列构造二叉树 | [CN](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) · [EN](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) | 后序最后一个是根，在中序里定位后切分左右子树 |
| 114 | 二叉树展开为链表 | [CN](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/) · [EN](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/) | 按 右 → 左 → 根 的顺序遍历，用头插法串起来 |
| **297** | **二叉树的序列化与反序列化** | [CN](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/) · [EN](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | [题解 ↓](#297-二叉树的序列化与反序列化--serialize-and-deserialize-binary-tree) |
| **98** | **验证二叉搜索树** | [CN](https://leetcode.cn/problems/validate-binary-search-tree/) · [EN](https://leetcode.com/problems/validate-binary-search-tree/) | [题解 ↓](#98-验证二叉搜索树--validate-binary-search-tree) |
| 230 | 二叉搜索树中第 K 小的元素 | [CN](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/) · [EN](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) | 中序遍历的第 k 个节点 |
| 108 | 将有序数组转换为二叉搜索树 | [CN](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/) · [EN](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/) | 取中点作为根，左右两半递归 |

<br>

#### 105. 从前序与中序遍历序列构造二叉树 / Construct Binary Tree from Preorder and Inorder Traversal

[CN](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) · [EN](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) · Medium

**题意**：根据前序和中序遍历结果构造二叉树（值互不相同）。

**思路**：前序的第一个数是根，在中序里找到根的位置 k，中序左边 k 个是左子树，右边是右子树；前序里根后面紧跟的 k 个也是左子树。用哈希表 O(1) 找根在中序的位置，递归时传下标区间。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 1. 值 -> 在中序遍历中的下标
        index = {x: i for i, x in enumerate(inorder)}

        # 2. 前序区间 [pre_l, pre_r)，中序区间 [in_l, in_r)
        def dfs(pre_l: int, pre_r: int, in_l: int, in_r: int) -> Optional[TreeNode]:
            if pre_l == pre_r:
                return None
            # 3. 根在中序中的位置，左子树大小
            root_val = preorder[pre_l]
            left_size = index[root_val] - in_l
            # 4. 递归构造左右子树
            left = dfs(pre_l + 1, pre_l + 1 + left_size, in_l, in_l + left_size)
            right = dfs(pre_l + 1 + left_size, pre_r, in_l + 1 + left_size, in_r)
            return TreeNode(root_val, left, right)

        n = len(preorder)
        return dfs(0, n, 0, n)
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

#### 297. 二叉树的序列化与反序列化 / Serialize and Deserialize Binary Tree

[CN](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/) · [EN](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) · Hard

**题意**：设计二叉树的序列化与反序列化算法。

**思路**：前序遍历序列化，空节点记为 #，用逗号分隔。反序列化时按同样的前序顺序消费 token：遇到 # 返回空，否则建节点再依次构造左右子树。

```python
class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        res = []

        # 1. 前序遍历，空节点写 #
        def dfs(node: Optional[TreeNode]) -> None:
            if node is None:
                res.append('#')
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ','.join(res)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        tokens = iter(data.split(','))

        # 2. 按前序顺序依次读取 token
        def build() -> Optional[TreeNode]:
            val = next(tokens)
            if val == '#':
                return None
            # 3. 先建根，再建左右子树
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node

        return build()
```

**复杂度**：两个操作都是时间 O(n)、空间 O(n)。

<br>

#### 98. 验证二叉搜索树 / Validate Binary Search Tree

[CN](https://leetcode.cn/problems/validate-binary-search-tree/) · [EN](https://leetcode.com/problems/validate-binary-search-tree/) · Medium

**题意**：判断二叉树是否为有效的二叉搜索树。

**思路**：前序遍历时传递取值范围 (left, right)：根的值必须在开区间内；递归左子树时把上界改成根的值，递归右子树时把下界改成根的值。

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode], left=-inf, right=inf) -> bool:
        # 1. 空树是 BST
        if root is None:
            return True
        x = root.val
        # 2. 当前值必须在 (left, right) 内，左子树上界变为 x，右子树下界变为 x
        return left < x < right and \
            self.isValidBST(root.left, left, x) and \
            self.isValidBST(root.right, x, right)
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

### 路径

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 112 | 路径总和 | [CN](https://leetcode.cn/problems/path-sum/) · [EN](https://leetcode.com/problems/path-sum/) | 往下递归时减去节点值，到叶子判断是否为 0 |
| 113 | 路径总和 II | [CN](https://leetcode.cn/problems/path-sum-ii/) · [EN](https://leetcode.com/problems/path-sum-ii/) | 112 + 回溯记录路径 |
| **437** | **路径总和 III** | [CN](https://leetcode.cn/problems/path-sum-iii/) · [EN](https://leetcode.com/problems/path-sum-iii/) | [题解 ↓](#437-路径总和-iii--path-sum-iii) |

<br>

#### 437. 路径总和 III / Path Sum III

[CN](https://leetcode.cn/problems/path-sum-iii/) · [EN](https://leetcode.com/problems/path-sum-iii/) · Medium

**题意**：求路径和等于 targetSum 的向下路径数（不必从根开始、到叶子结束）。

**思路**：和 560 一样用前缀和 + 哈希表，只是数组换成了从根到当前节点的路径。DFS 进入节点时把当前前缀和计数加一，离开时减一（回溯），保证哈希表里只有当前路径上的前缀和。

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        ans = 0
        cnt = defaultdict(int)
        cnt[0] = 1  # 1. 空路径的前缀和为 0

        def dfs(node: Optional[TreeNode], s: int) -> None:
            nonlocal ans
            if node is None:
                return
            # 2. s 是根到当前节点的路径和
            s += node.val
            ans += cnt[s - targetSum]
            # 3. 当前前缀和加入哈希表，递归左右子树
            cnt[s] += 1
            dfs(node.left, s)
            dfs(node.right, s)
            # 4. 回溯：离开当前节点时撤销
            cnt[s] -= 1

        dfs(root, 0)
        return ans
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

### 最近公共祖先

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **236** | **二叉树的最近公共祖先** | [CN](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/) · [EN](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | [题解 ↓](#236-二叉树的最近公共祖先--lowest-common-ancestor-of-a-binary-tree) |
| 235 | 二叉搜索树的最近公共祖先 | [CN](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/) · [EN](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | p、q 都比根小走左边，都大走右边，否则根就是答案 |

<br>

#### 236. 二叉树的最近公共祖先 / Lowest Common Ancestor of a Binary Tree

[CN](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/) · [EN](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) · Medium

**题意**：求二叉树中两个节点的最近公共祖先。

**思路**：分类讨论：当前节点是空、p 或 q，直接返回它。否则递归左右子树：两边都找到了，当前节点就是 LCA；只有一边找到，就返回那一边的结果。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 1. 空节点、p 或 q，直接返回
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 2. 左右都找到，当前节点就是 LCA
        if left and right:
            return root
        # 3. 只有一边找到，返回那一边
        return left or right
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

### 直径

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **543** | **二叉树的直径** | [CN](https://leetcode.cn/problems/diameter-of-binary-tree/) · [EN](https://leetcode.com/problems/diameter-of-binary-tree/) | [题解 ↓](#543-二叉树的直径--diameter-of-binary-tree) |
| **124** | **二叉树中的最大路径和** | [CN](https://leetcode.cn/problems/binary-tree-maximum-path-sum/) · [EN](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | [题解 ↓](#124-二叉树中的最大路径和--binary-tree-maximum-path-sum) |
| 687 | 最长同值路径 | [CN](https://leetcode.cn/problems/longest-univalue-path/) · [EN](https://leetcode.com/problems/longest-univalue-path/) | 543 的变形：只有和孩子值相同才能接上 |

<br>

#### 543. 二叉树的直径 / Diameter of Binary Tree

[CN](https://leetcode.cn/problems/diameter-of-binary-tree/) · [EN](https://leetcode.com/problems/diameter-of-binary-tree/) · Easy

**题意**：求二叉树的直径（任意两节点间最长路径的边数）。

**思路**：直径一定在某个节点「拐弯」：经过该节点的最长路径 = 左子树的最长链 + 右子树的最长链。DFS 返回以当前节点为端点向下的最长链长度，同时在每个节点用左右链之和更新答案。

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0

        # 1. 返回从 node 往下的最长链的边数；空节点返回 -1
        def dfs(node: Optional[TreeNode]) -> int:
            nonlocal ans
            if node is None:
                return -1
            left_len = dfs(node.left) + 1   # 2. 左边的链，加上 node 到左孩子的边
            right_len = dfs(node.right) + 1
            # 3. 在 node 处拐弯的路径
            ans = max(ans, left_len + right_len)
            return max(left_len, right_len)

        dfs(root)
        return ans
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

#### 124. 二叉树中的最大路径和 / Binary Tree Maximum Path Sum

[CN](https://leetcode.cn/problems/binary-tree-maximum-path-sum/) · [EN](https://leetcode.com/problems/binary-tree-maximum-path-sum/) · Hard

**题意**：求二叉树中路径和最大的非空路径的和（路径可以不经过根）。

**思路**：和 543 一样在每个节点拐弯，只是链的长度换成节点值之和。DFS 返回从当前节点往下的最大链和；链和为负数时不如不要，所以和 0 取最大。

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = -inf

        # 1. 返回从 node 往下的最大链和
        def dfs(node: Optional[TreeNode]) -> int:
            nonlocal ans
            if node is None:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            # 2. 在 node 处拐弯的路径和
            ans = max(ans, left + right + node.val)
            # 3. 往上只能选一条链；负数就不要了
            return max(max(left, right) + node.val, 0)

        dfs(root)
        return ans
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

### 回溯

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **46** | **全排列** | [CN](https://leetcode.cn/problems/permutations/) · [EN](https://leetcode.com/problems/permutations/) | [题解 ↓](#46-全排列--permutations) |
| 47 | 全排列 II | [CN](https://leetcode.cn/problems/permutations-ii/) · [EN](https://leetcode.com/problems/permutations-ii/) | 排序；同一层跳过「与前一个相同且前一个没用过」的数 |
| **78** | **子集** | [CN](https://leetcode.cn/problems/subsets/) · [EN](https://leetcode.com/problems/subsets/) | [题解 ↓](#78-子集--subsets) |
| 90 | 子集 II | [CN](https://leetcode.cn/problems/subsets-ii/) · [EN](https://leetcode.com/problems/subsets-ii/) | 排序；同一层跳过重复的数 |
| 77 | 组合 | [CN](https://leetcode.cn/problems/combinations/) · [EN](https://leetcode.com/problems/combinations/) | 枚举下一个选谁；剩余数字不够 k 个时剪枝 |
| **39** | **组合总和** | [CN](https://leetcode.cn/problems/combination-sum/) · [EN](https://leetcode.com/problems/combination-sum/) | [题解 ↓](#39-组合总和--combination-sum) |
| 40 | 组合总和 II | [CN](https://leetcode.cn/problems/combination-sum-ii/) · [EN](https://leetcode.com/problems/combination-sum-ii/) | 排序；每个数只用一次，同一层去重 |
| 216 | 组合总和 III | [CN](https://leetcode.cn/problems/combination-sum-iii/) · [EN](https://leetcode.com/problems/combination-sum-iii/) | 从 1~9 里选 k 个数使和为 n，和超了就剪枝 |
| **17** | **电话号码的字母组合** | [CN](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/) · [EN](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) | [题解 ↓](#17-电话号码的字母组合--letter-combinations-of-a-phone-number) |
| **22** | **括号生成** | [CN](https://leetcode.cn/problems/generate-parentheses/) · [EN](https://leetcode.com/problems/generate-parentheses/) | [题解 ↓](#22-括号生成--generate-parentheses) |
| 131 | 分割回文串 | [CN](https://leetcode.cn/problems/palindrome-partitioning/) · [EN](https://leetcode.com/problems/palindrome-partitioning/) | 枚举下一段的结束位置，是回文才递归 |
| **51** | **N 皇后** | [CN](https://leetcode.cn/problems/n-queens/) · [EN](https://leetcode.com/problems/n-queens/) | [题解 ↓](#51-n-皇后--n-queens) |
| 93 | 复原 IP 地址 | [CN](https://leetcode.cn/problems/restore-ip-addresses/) · [EN](https://leetcode.com/problems/restore-ip-addresses/) | 枚举每段长度 1~3，检查前导零和 ≤ 255 |

<br>

#### 46. 全排列 / Permutations

[CN](https://leetcode.cn/problems/permutations/) · [EN](https://leetcode.com/problems/permutations/) · Medium

**题意**：返回不含重复数字的数组的所有全排列。

**思路**：排列型回溯：path[i] 填哪个数，枚举所有还没用过的数。用 on_path 数组标记已经用过的数，递归返回后恢复。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        path = [0] * n
        on_path = [False] * n

        # 1. 填 path[i]
        def dfs(i: int) -> None:
            if i == n:
                ans.append(path.copy())
                return
            # 2. 枚举还没用过的数
            for j, on in enumerate(on_path):
                if not on:
                    path[i] = nums[j]
                    on_path[j] = True
                    dfs(i + 1)
                    on_path[j] = False  # 3. 恢复现场

        dfs(0)
        return ans
```

**复杂度**：时间 O(n·n!)；空间 O(n)。

<br>

#### 78. 子集 / Subsets

[CN](https://leetcode.cn/problems/subsets/) · [EN](https://leetcode.com/problems/subsets/) · Medium

**题意**：返回元素互不相同的数组的所有子集。

**思路**：子集型回溯，「选或不选」：对每个数决定选或不选，递归到末尾时记录当前子集。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        path = []

        # 1. 考虑 nums[i] 选或不选
        def dfs(i: int) -> None:
            if i == n:
                ans.append(path.copy())
                return
            # 2. 不选 nums[i]
            dfs(i + 1)
            # 3. 选 nums[i]，递归后恢复现场
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

        dfs(0)
        return ans
```

**复杂度**：时间 O(n·2ⁿ)；空间 O(n)。

<br>

#### 39. 组合总和 / Combination Sum

[CN](https://leetcode.cn/problems/combination-sum/) · [EN](https://leetcode.com/problems/combination-sum/) · Medium

**题意**：从无重复的正整数数组中选数（可重复选）使和为 target，返回所有组合。

**思路**：每个数可以重复选。考虑第 i 个数：不选，跳到 i+1；选，留在 i（还能继续选它）。先排序，剩余的和 left 比 candidates[i] 小时后面都不可能，直接剪枝。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        ans = []
        path = []

        # 1. 考虑 candidates[i]，还需要凑出 left
        def dfs(i: int, left: int) -> None:
            if left == 0:
                ans.append(path.copy())
                return
            # 2. 剪枝：数用完了，或者最小的数都放不下
            if i == len(candidates) or left < candidates[i]:
                return
            # 3. 不选 candidates[i]
            dfs(i + 1, left)
            # 4. 选 candidates[i]，下次还可以选它
            path.append(candidates[i])
            dfs(i, left - candidates[i])
            path.pop()

        dfs(0, target)
        return ans
```

**复杂度**：时间与答案规模相关，最坏指数级；空间 O(target / min(candidates))。

<br>

#### 17. 电话号码的字母组合 / Letter Combinations of a Phone Number

[CN](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/) · [EN](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) · Medium

**题意**：返回数字 2-9 组成的字符串在电话按键上能表示的所有字母组合。

**思路**：回溯：path[i] 填 digits[i] 对应的某个字母，枚举所有可能，填满就记录。

```python
MAPPING = "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        n = len(digits)
        if n == 0:
            return []
        ans = []
        path = [''] * n

        # 1. 填 path[i]
        def dfs(i: int) -> None:
            if i == n:
                ans.append(''.join(path))
                return
            # 2. 枚举 digits[i] 对应的字母
            for c in MAPPING[int(digits[i])]:
                path[i] = c  # 直接覆盖，不需要恢复
                dfs(i + 1)

        dfs(0)
        return ans
```

**复杂度**：时间 O(n·4ⁿ)；空间 O(n)。

<br>

#### 22. 括号生成 / Generate Parentheses

[CN](https://leetcode.cn/problems/generate-parentheses/) · [EN](https://leetcode.com/problems/generate-parentheses/) · Medium

**题意**：生成所有 n 对括号的有效组合。

**思路**：一共要填 2n 个位置。左括号数量 < n 时可以填左括号；右括号数量 < 左括号数量时可以填右括号。这样生成的一定合法。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        path = []

        # 1. open：已填左括号数，close：已填右括号数
        def dfs(open: int, close: int) -> None:
            if close == n:
                ans.append(''.join(path))
                return
            # 2. 还可以填左括号
            if open < n:
                path.append('(')
                dfs(open + 1, close)
                path.pop()
            # 3. 右括号不能多于左括号
            if close < open:
                path.append(')')
                dfs(open, close + 1)
                path.pop()

        dfs(0, 0)
        return ans
```

**复杂度**：时间 O(n·C(2n, n) / (n+1))（卡特兰数）；空间 O(n)。

<br>

#### 51. N 皇后 / N-Queens

[CN](https://leetcode.cn/problems/n-queens/) · [EN](https://leetcode.com/problems/n-queens/) · Hard

**题意**：在 n×n 棋盘上放 n 个互不攻击的皇后，返回所有方案。

**思路**：每行恰好放一个皇后，queens[r] 表示第 r 行皇后所在的列，本质是列的全排列。用三个数组标记列、主对角线（r+c 相同）、副对角线（r-c 相同）是否被占用。

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        queens = [0] * n  # 1. queens[r] = 第 r 行皇后的列
        col = [False] * n
        diag1 = [False] * (2 * n - 1)  # r + c
        diag2 = [False] * (2 * n - 1)  # r - c + n - 1

        # 2. 在第 r 行放皇后
        def dfs(r: int) -> None:
            if r == n:
                ans.append(['.' * c + 'Q' + '.' * (n - 1 - c) for c in queens])
                return
            for c in range(n):
                rc = r - c + n - 1
                # 3. 列和两条对角线都没被占用
                if not col[c] and not diag1[r + c] and not diag2[rc]:
                    queens[r] = c
                    col[c] = diag1[r + c] = diag2[rc] = True
                    dfs(r + 1)
                    col[c] = diag1[r + c] = diag2[rc] = False  # 4. 恢复现场

        dfs(0)
        return ans
```

**复杂度**：时间 O(n²·n!)；空间 O(n)。

<br>
<br>

## 9. 贪心

每一步做当前最优的选择，再用交换论证说明不会更差。

<br>

### 基本贪心

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 455 | 分发饼干 | [CN](https://leetcode.cn/problems/assign-cookies/) · [EN](https://leetcode.com/problems/assign-cookies/) | 两个数组排序，用能满足的最小饼干喂胃口最小的孩子 |
| **122** | **买卖股票的最佳时机 II** | [CN](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/) · [EN](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) | [题解](#122-买卖股票的最佳时机-ii--best-time-to-buy-and-sell-stock-ii) |
| **55** | **跳跃游戏** | [CN](https://leetcode.cn/problems/jump-game/) · [EN](https://leetcode.com/problems/jump-game/) | [题解 ↓](#55-跳跃游戏--jump-game) |
| **45** | **跳跃游戏 II** | [CN](https://leetcode.cn/problems/jump-game-ii/) · [EN](https://leetcode.com/problems/jump-game-ii/) | [题解 ↓](#45-跳跃游戏-ii--jump-game-ii) |
| **134** | **加油站** | [CN](https://leetcode.cn/problems/gas-station/) · [EN](https://leetcode.com/problems/gas-station/) | [题解 ↓](#134-加油站--gas-station) |
| 135 | 分发糖果 | [CN](https://leetcode.cn/problems/candy/) · [EN](https://leetcode.com/problems/candy/) | 从左往右、从右往左各扫一遍，每个孩子取两边要求的较大值 |
| **763** | **划分字母区间** | [CN](https://leetcode.cn/problems/partition-labels/) · [EN](https://leetcode.com/problems/partition-labels/) | [题解 ↓](#763-划分字母区间--partition-labels) |

<br>

#### 55. 跳跃游戏 / Jump Game

[CN](https://leetcode.cn/problems/jump-game/) · [EN](https://leetcode.com/problems/jump-game/) · Medium

**题意**：nums[i] 是在位置 i 最多能跳的步数，判断能否到达最后一个位置。

**思路**：维护当前能到达的最远位置 mx。遍历每个位置 i：如果 i > mx，说明 i 到不了，返回 False；否则用 i + nums[i] 更新 mx。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        mx = 0  # 1. 当前能到达的最远位置
        for i, jump in enumerate(nums):
            # 2. 位置 i 到不了
            if i > mx:
                return False
            # 3. 从 i 出发能跳到更远
            mx = max(mx, i + jump)
            if mx >= len(nums) - 1:
                return True
        return True
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 45. 跳跃游戏 II / Jump Game II

[CN](https://leetcode.cn/problems/jump-game-ii/) · [EN](https://leetcode.com/problems/jump-game-ii/) · Medium

**题意**：同 55，保证能到达终点，求最少跳跃次数。

**思路**：把跳跃看成「建桥」：cur_right 是已建的桥的终点，next_right 是下一座桥最远能到哪。走到 cur_right 时必须再建一座桥（跳一次），新桥终点就是 next_right。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        ans = 0
        cur_right = 0   # 1. 已建造的桥的右端点
        next_right = 0  # 下一座桥最远能到的位置
        # 2. 走到 n-2 即可，到达 n-1 不需要再建桥
        for i in range(len(nums) - 1):
            next_right = max(next_right, i + nums[i])
            # 3. 到达桥的终点，必须建新桥
            if i == cur_right:
                cur_right = next_right
                ans += 1
        return ans
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 134. 加油站 / Gas Station

[CN](https://leetcode.cn/problems/gas-station/) · [EN](https://leetcode.com/problems/gas-station/) · Medium

**题意**：环形路线上有 n 个加油站，求能绕一圈的出发站编号，不存在返回 -1。

**思路**：总油量 < 总消耗时无解。否则从 0 出发累计油量 s，如果在 i 处 s < 0，说明从起点到 i 之间的任何一站出发都到不了 i+1，起点改为 i+1，s 清零。

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # 1. 总油量不够，一定无解
        if sum(gas) < sum(cost):
            return -1
        start = s = 0
        for i, (g, c) in enumerate(zip(gas, cost)):
            s += g - c
            # 2. 从 start 到不了 i+1，[start, i] 都不能作为起点
            if s < 0:
                start = i + 1
                s = 0
        # 3. 总油量够时，剩下的 start 一定可行
        return start
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>

#### 763. 划分字母区间 / Partition Labels

[CN](https://leetcode.cn/problems/partition-labels/) · [EN](https://leetcode.com/problems/partition-labels/) · Medium

**题意**：把字符串划分成尽可能多的片段，使每个字母只出现在一个片段中，返回各片段长度。

**思路**：先记录每个字母最后一次出现的位置。遍历时维护当前片段必须延伸到的最远位置 end（片段内所有字母的最后位置的最大值）；走到 end 时当前片段可以结束。本质是合并区间。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 1. 每个字母最后出现的下标
        last = {c: i for i, c in enumerate(s)}
        ans = []
        start = end = 0
        for i, c in enumerate(s):
            # 2. 当前片段至少要延伸到 c 最后出现的位置
            end = max(end, last[c])
            # 3. 到达片段终点，切一刀
            if i == end:
                ans.append(end - start + 1)
                start = i + 1
        return ans
```

**复杂度**：时间 O(n)；空间 O(|Σ|)。

<br>

### 区间

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **56** | **合并区间** | [CN](https://leetcode.cn/problems/merge-intervals/) · [EN](https://leetcode.com/problems/merge-intervals/) | [题解 ↓](#56-合并区间--merge-intervals) |
| 57 | 插入区间 | [CN](https://leetcode.cn/problems/insert-interval/) · [EN](https://leetcode.com/problems/insert-interval/) | 分三段：左边不重叠的、和新区间合并的、右边不重叠的 |
| **435** | **无重叠区间** | [CN](https://leetcode.cn/problems/non-overlapping-intervals/) · [EN](https://leetcode.com/problems/non-overlapping-intervals/) | [题解 ↓](#435-无重叠区间--non-overlapping-intervals) |
| 452 | 用最少数量的箭引爆气球 | [CN](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/) · [EN](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/) | 按右端点排序，在右端点射箭，射不到的再加一支 |
| 253 | 会议室 II 🔒 | [CN](https://leetcode.cn/problems/meeting-rooms-ii/) · [EN](https://leetcode.com/problems/meeting-rooms-ii/) | 按开始时间排序 + 小顶堆存结束时间；或差分扫描线 |
| 986 | 区间列表的交集 | [CN](https://leetcode.cn/problems/interval-list-intersections/) · [EN](https://leetcode.com/problems/interval-list-intersections/) | 双指针，交集为 [max(左), min(右)]，右端点小的前进 |

<br>

#### 56. 合并区间 / Merge Intervals

[CN](https://leetcode.cn/problems/merge-intervals/) · [EN](https://leetcode.com/problems/merge-intervals/) · Medium

**题意**：合并所有重叠的区间。

**思路**：按左端点排序。遍历区间，如果当前区间的左端点 ≤ 答案中最后一个区间的右端点，就重叠，合并（更新右端点为两者最大值）；否则作为新区间加入答案。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 1. 按左端点排序
        intervals.sort(key=lambda p: p[0])
        ans = []
        for p in intervals:
            # 2. 和最后一个区间重叠，合并
            if ans and p[0] <= ans[-1][1]:
                ans[-1][1] = max(ans[-1][1], p[1])
            # 3. 不重叠，作为新区间
            else:
                ans.append(p)
        return ans
```

**复杂度**：时间 O(n log n)；空间 O(1)（不计排序和答案）。

<br>

#### 435. 无重叠区间 / Non-overlapping Intervals

[CN](https://leetcode.cn/problems/non-overlapping-intervals/) · [EN](https://leetcode.com/problems/non-overlapping-intervals/) · Medium

**题意**：求最少移除多少个区间，使剩余区间互不重叠。

**思路**：等价于选出最多的互不重叠区间。按右端点排序，贪心地选结束最早的区间，这样留给后面的空间最多。答案 = 总数 - 能选的最多个数。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # 1. 按右端点排序
        intervals.sort(key=lambda p: p[1])
        keep = 0
        pre_end = -inf  # 上一个选中区间的右端点
        for start, end in intervals:
            # 2. 不和上一个选中的区间重叠，就选它
            if start >= pre_end:
                keep += 1
                pre_end = end
        # 3. 其余的都要删除
        return len(intervals) - keep
```

**复杂度**：时间 O(n log n)；空间 O(1)（不计排序）。

<br>
<br>

## 10. 中频 · 滑动窗口

<br>

### 三指针

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 2444 | 统计定界子数组的数目 | [CN](https://leetcode.cn/problems/count-subarrays-with-fixed-bounds/) · [EN](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/) | 记录最近的 minK、maxK 和越界元素位置，每个右端点贡献 max(0, min(i1, i2) - i0) |
| 795 | 区间子数组个数 | [CN](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/) · [EN](https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/) | 记录最近一个 > right 的位置和最近一个 ∈ [left, right] 的位置 |

<br>

### 分组循环

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 228 | 汇总区间 | [CN](https://leetcode.cn/problems/summary-ranges/) · [EN](https://leetcode.com/problems/summary-ranges/) | 外层确定段的起点，内层走到连续段末尾 |
| 1446 | 连续字符 | [CN](https://leetcode.cn/problems/consecutive-characters/) · [EN](https://leetcode.com/problems/consecutive-characters/) | 分组循环，求最长的一段 |
| 1869 | 哪种连续子字符串更长 | [CN](https://leetcode.cn/problems/longer-contiguous-segments-of-ones-than-zeros/) · [EN](https://leetcode.com/problems/longer-contiguous-segments-of-ones-than-zeros/) | 分组循环，分别统计 1 段和 0 段的最大长度 |

<br>
<br>

## 11. 中频 · 二分

<br>

### 最小化最大值

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **410** | **分割数组的最大值** | [CN](https://leetcode.cn/problems/split-array-largest-sum/) · [EN](https://leetcode.com/problems/split-array-largest-sum/) | [题解 ↓](#410-分割数组的最大值--split-array-largest-sum) |
| 1760 | 袋子里最少数目的球 | [CN](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/) · [EN](https://leetcode.com/problems/minimum-limit-of-balls-in-a-bag/) | 二分上限 m，大小为 x 的袋子需要 (x-1) // m 次操作 |
| 2064 | 分配给商店的最多商品的最小值 | [CN](https://leetcode.cn/problems/minimized-maximum-of-products-distributed-to-any-store/) · [EN](https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/) | 二分上限 m，检查 Σ ceil(q / m) ≤ n |

<br>

#### 410. 分割数组的最大值 / Split Array Largest Sum

[CN](https://leetcode.cn/problems/split-array-largest-sum/) · [EN](https://leetcode.com/problems/split-array-largest-sum/) · Hard

**题意**：把数组分成 k 个非空连续子数组，使各段和的最大值最小。

**思路**：二分答案（最小化最大值）：上限 mx 越大，需要的段数越少。check(mx)：贪心地往当前段里加数，超过 mx 就新开一段，看段数是否 ≤ k。下界是 max(nums)，上界是 sum(nums)。

```python
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        # 1. 每段和不超过 mx 时，最少分几段，是否 <= k
        def check(mx: int) -> bool:
            cnt, s = 1, 0
            for x in nums:
                if s + x <= mx:
                    s += x
                else:  # 2. 放不下，新开一段
                    cnt += 1
                    s = x
            return cnt <= k

        # 3. 二分最小的满足 check 的 mx
        left, right = max(nums), sum(nums)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
```

**复杂度**：时间 O(n log(sum))；空间 O(1)。

<br>

### 最大化最小值

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **1552** | **两球之间的磁力** | [CN](https://leetcode.cn/problems/magnetic-force-between-two-balls/) · [EN](https://leetcode.com/problems/magnetic-force-between-two-balls/) | [题解 ↓](#1552-两球之间的磁力--magnetic-force-between-two-balls) |
| 2517 | 礼盒的最大甜蜜度 | [CN](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/) · [EN](https://leetcode.com/problems/maximum-tastiness-of-candy-basket/) | 排序后二分最小差 d，贪心检查能否选出 k 个 |

<br>

#### 1552. 两球之间的磁力 / Magnetic Force Between Two Balls

[CN](https://leetcode.cn/problems/magnetic-force-between-two-balls/) · [EN](https://leetcode.com/problems/magnetic-force-between-two-balls/) · Medium

**题意**：在 n 个位置中放 m 个球，使任意两球之间的最小距离最大。

**思路**：二分答案（最大化最小值）：最小间距 d 越大，能放的球越少。排序后 check(d)：从第一个位置开始，贪心地每次放在距离上一个球 ≥ d 的最近位置，看能否放下 m 个。

```python
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        position.sort()

        # 1. 最小间距至少为 d 时，能否放下 m 个球
        def check(d: int) -> bool:
            cnt = 1
            pre = position[0]  # 第一个球放在最左边
            for x in position:
                # 2. 距离够了就放一个
                if x - pre >= d:
                    cnt += 1
                    pre = x
            return cnt >= m

        # 3. 二分最大的满足 check 的 d
        left, right = 1, (position[-1] - position[0]) // (m - 1)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        return right
```

**复杂度**：时间 O(n log n + n log U)；空间 O(1)（不计排序）。

<br>

### 第 K 小

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **378** | **有序矩阵中第 K 小的元素** | [CN](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/) · [EN](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) | [题解 ↓](#378-有序矩阵中第-k-小的元素--kth-smallest-element-in-a-sorted-matrix) |
| 668 | 乘法表中第k小的数 | [CN](https://leetcode.cn/problems/kth-smallest-number-in-multiplication-table/) · [EN](https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/) | 二分 x，第 i 行 ≤ x 的个数是 min(x // i, n) |
| 719 | 找出第 K 小的数对距离 | [CN](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/) · [EN](https://leetcode.com/problems/find-k-th-smallest-pair-distance/) | 排序 + 二分距离 d + 双指针统计距离 ≤ d 的数对 |

<br>

#### 378. 有序矩阵中第 K 小的元素 / Kth Smallest Element in a Sorted Matrix

[CN](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/) · [EN](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) · Medium

**题意**：在每行每列都升序的 n×n 矩阵中，求第 k 小的元素。

**思路**：二分答案：值 x 越大，矩阵中 ≤ x 的元素越多。统计 ≤ x 的个数时，从左下角出发：当前元素 ≤ x 就把这一列上方都算上并右移，否则上移，O(n) 完成。找第一个使个数 ≥ k 的 x。

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)

        # 1. 矩阵中 <= mx 的元素个数是否 >= k
        def check(mx: int) -> bool:
            cnt = 0
            i, j = n - 1, 0  # 从左下角出发
            while i >= 0 and j < n:
                if matrix[i][j] <= mx:
                    cnt += i + 1  # 2. 这一列第 0~i 行都 <= mx
                    j += 1
                else:
                    i -= 1
            return cnt >= k

        # 3. 二分第一个满足 check 的值，它一定在矩阵中
        left, right = matrix[0][0], matrix[-1][-1]
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
```

**复杂度**：时间 O(n log U)，U 是值域；空间 O(1)。

<br>
<br>

## 12. 中频 · 单调栈

<br>

### 贡献法

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **907** | **子数组的最小值之和** | [CN](https://leetcode.cn/problems/sum-of-subarray-minimums/) · [EN](https://leetcode.com/problems/sum-of-subarray-minimums/) | [题解 ↓](#907-子数组的最小值之和--sum-of-subarray-minimums) |
| 2104 | 子数组范围和 | [CN](https://leetcode.cn/problems/sum-of-subarray-ranges/) · [EN](https://leetcode.com/problems/sum-of-subarray-ranges/) | Σ 每个数作为最大值的贡献 - Σ 作为最小值的贡献 |
| 1856 | 子数组最小乘积的最大值 | [CN](https://leetcode.cn/problems/maximum-subarray-min-product/) · [EN](https://leetcode.com/problems/maximum-subarray-min-product/) | 单调栈求每个数作为最小值的最大范围 + 前缀和求区间和 |

<br>

#### 907. 子数组的最小值之和 / Sum of Subarray Minimums

[CN](https://leetcode.cn/problems/sum-of-subarray-minimums/) · [EN](https://leetcode.com/problems/sum-of-subarray-minimums/) · Medium

**题意**：求所有连续子数组最小值之和，结果模 10⁹+7。

**思路**：贡献法：每个 arr[i] 作为最小值的子数组个数 = (i - left[i]) × (right[i] - i)，其中 left[i] 是左边第一个 < arr[i] 的位置，right[i] 是右边第一个 ≤ arr[i] 的位置（一边严格一边不严格，避免重复数被算两次）。一次单调栈同时求出两边。

```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        MOD = 1_000_000_007
        n = len(arr)
        left = [-1] * n  # 1. 左边第一个 < arr[i] 的位置
        right = [n] * n  # 右边第一个 <= arr[i] 的位置
        st = []
        for i, x in enumerate(arr):
            # 2. 弹出的元素 j 找到了它右边第一个 <= 它的数
            while st and arr[st[-1]] >= x:
                right[st.pop()] = i
            # 3. 栈顶是左边第一个 < x 的数
            if st:
                left[i] = st[-1]
            st.append(i)
        # 4. 每个数的贡献 = 值 × 以它为最小值的子数组个数
        ans = sum(x * (i - l) * (r - i) for i, (x, l, r) in enumerate(zip(arr, left, right)))
        return ans % MOD
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>

### 最小字典序

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **316** | **去除重复字母** | [CN](https://leetcode.cn/problems/remove-duplicate-letters/) · [EN](https://leetcode.com/problems/remove-duplicate-letters/) | [题解 ↓](#316-去除重复字母--remove-duplicate-letters) |
| **402** | **移掉 K 位数字** | [CN](https://leetcode.cn/problems/remove-k-digits/) · [EN](https://leetcode.com/problems/remove-k-digits/) | [题解 ↓](#402-移掉-k-位数字--remove-k-digits) |
| 321 | 拼接最大数 | [CN](https://leetcode.cn/problems/create-maximum-number/) · [EN](https://leetcode.com/problems/create-maximum-number/) | 枚举两个数组各取几个，单调栈取最大子序列，再按字典序归并 |

<br>

#### 316. 去除重复字母 / Remove Duplicate Letters

[CN](https://leetcode.cn/problems/remove-duplicate-letters/) · [EN](https://leetcode.com/problems/remove-duplicate-letters/) · Medium

**题意**：去除重复字母，使每个字母只出现一次，且结果的字典序最小。

**思路**：要字典序最小，前面的字母越小越好。用单调栈：当前字母 c 没在栈中时，如果栈顶比 c 大且栈顶字母在后面还会出现，就弹出栈顶（以后再加回来），最后把 c 入栈。已在栈中的字母直接跳过。

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        left = Counter(s)  # 1. 每个字母剩余的出现次数
        st = []
        in_st = set()
        for c in s:
            left[c] -= 1
            # 2. 已经在栈中，跳过
            if c in in_st:
                continue
            # 3. 栈顶更大且后面还有，弹出可以让字典序更小
            while st and c < st[-1] and left[st[-1]]:
                in_st.remove(st.pop())
            # 4. c 入栈
            st.append(c)
            in_st.add(c)
        return ''.join(st)
```

**复杂度**：时间 O(n)；空间 O(|Σ|)。

<br>

#### 402. 移掉 K 位数字 / Remove K Digits

[CN](https://leetcode.cn/problems/remove-k-digits/) · [EN](https://leetcode.com/problems/remove-k-digits/) · Medium

**题意**：从数字字符串中移除 k 位，使剩下的数字最小。

**思路**：要结果最小，高位越小越好。单调栈：当前数字比栈顶小，并且还能删（k > 0），就弹出栈顶。遍历完如果还没删够，从末尾删（此时栈单调不减）。最后去掉前导零。

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        st = []
        for d in num:
            # 1. 栈顶比当前数字大，删掉栈顶
            while k and st and st[-1] > d:
                st.pop()
                k -= 1
            st.append(d)
        # 2. 还没删够，从末尾删
        if k:
            st = st[:-k]
        # 3. 去掉前导零，空串返回 "0"
        return ''.join(st).lstrip('0') or '0'
```

**复杂度**：时间 O(n)；空间 O(n)。

<br>
<br>

## 13. 中频 · 位运算

<br>

### 基础与性质

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **136** | **只出现一次的数字** | [CN](https://leetcode.cn/problems/single-number/) · [EN](https://leetcode.com/problems/single-number/) | [题解 ↓](#136-只出现一次的数字--single-number) |
| 137 | 只出现一次的数字 II | [CN](https://leetcode.cn/problems/single-number-ii/) · [EN](https://leetcode.com/problems/single-number-ii/) | 按位统计 1 的个数，模 3 后剩下的就是答案的这一位 |
| 260 | 只出现一次的数字 III | [CN](https://leetcode.cn/problems/single-number-iii/) · [EN](https://leetcode.com/problems/single-number-iii/) | 全部异或得到 a^b，按 lowbit 把数分成两组分别异或 |
| 191 | 位1的个数 | [CN](https://leetcode.cn/problems/number-of-1-bits/) · [EN](https://leetcode.com/problems/number-of-1-bits/) | n &= n - 1 每次去掉最低位的 1 |
| 338 | 比特位计数 | [CN](https://leetcode.cn/problems/counting-bits/) · [EN](https://leetcode.com/problems/counting-bits/) | f[i] = f[i >> 1] + (i & 1) |
| 231 | 2 的幂 | [CN](https://leetcode.cn/problems/power-of-two/) · [EN](https://leetcode.com/problems/power-of-two/) | n > 0 且 n & (n - 1) == 0 |
| 190 | 颠倒二进制位 | [CN](https://leetcode.cn/problems/reverse-bits/) · [EN](https://leetcode.com/problems/reverse-bits/) | 逐位取出，放到对称的位置 |
| 268 | 丢失的数字 | [CN](https://leetcode.cn/problems/missing-number/) · [EN](https://leetcode.com/problems/missing-number/) | 0..n 和数组所有元素一起异或 |
| 371 | 两整数之和 | [CN](https://leetcode.cn/problems/sum-of-two-integers/) · [EN](https://leetcode.com/problems/sum-of-two-integers/) | 异或是无进位加法，(a & b) << 1 是进位；Python 需要模拟 32 位 |

<br>

#### 136. 只出现一次的数字 / Single Number

[CN](https://leetcode.cn/problems/single-number/) · [EN](https://leetcode.com/problems/single-number/) · Easy

**题意**：数组中只有一个元素出现一次，其余都出现两次，找出它。

**思路**：异或的性质：x ^ x = 0，x ^ 0 = x，并且满足交换律、结合律。把所有数异或起来，出现两次的都抵消了，剩下的就是只出现一次的数。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 1. 成对的数异或为 0，剩下只出现一次的数
        return reduce(xor, nums)
```

**复杂度**：时间 O(n)；空间 O(1)。

<br>
<br>

## 14. 中频 · 图论

<br>

### 最小生成树

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **1584** | **连接所有点的最小费用** | [CN](https://leetcode.cn/problems/min-cost-to-connect-all-points/) · [EN](https://leetcode.com/problems/min-cost-to-connect-all-points/) | [题解](#1584-连接所有点的最小费用--min-cost-to-connect-all-points) |
| 1135 | 最低成本连通所有城市 🔒 | [CN](https://leetcode.cn/problems/connecting-cities-with-minimum-cost/) · [EN](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) | Kruskal：按边权排序 + 并查集，最后检查是否连通，[题解](#1135-最低成本连通所有城市--connecting-cities-with-minimum-cost) |

<br>

### Floyd

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 1334 | 阈值距离内邻居最少的城市 | [CN](https://leetcode.cn/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/) · [EN](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/) | Floyd 求任意两点最短路，统计每个城市 ≤ threshold 的邻居数 |

<br>

### Bellman-Ford

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **787** | **K 站中转内最便宜的航班** | [CN](https://leetcode.cn/problems/cheapest-flights-within-k-stops/) · [EN](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | [题解](#787-k-站中转内最便宜的航班--cheapest-flights-within-k-stops) |

<br>
<br>

## 15. 中频 · 动态规划

<br>

### 划分型

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 91 | 解码方法 | [CN](https://leetcode.cn/problems/decode-ways/) · [EN](https://leetcode.com/problems/decode-ways/) | f[i] 看最后一位、最后两位能否单独解码 |
| 132 | 分割回文串 II | [CN](https://leetcode.cn/problems/palindrome-partitioning-ii/) · [EN](https://leetcode.com/problems/palindrome-partitioning-ii/) | 预处理回文表，f[i] = min(f[j] + 1)，其中 s[j..i] 是回文 |
| 410 | 分割数组的最大值 | [CN](https://leetcode.cn/problems/split-array-largest-sum/) · [EN](https://leetcode.com/problems/split-array-largest-sum/) | 划分型 DP：f[i][j] 表示前 j 个数分成 i 段；二分答案更快，[题解](#410-分割数组的最大值--split-array-largest-sum) |

<br>

### 区间

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **516** | **最长回文子序列** | [CN](https://leetcode.cn/problems/longest-palindromic-subsequence/) · [EN](https://leetcode.com/problems/longest-palindromic-subsequence/) | [题解 ↓](#516-最长回文子序列--longest-palindromic-subsequence) |
| 647 | 回文子串 | [CN](https://leetcode.cn/problems/palindromic-substrings/) · [EN](https://leetcode.com/problems/palindromic-substrings/) | 中心扩展，统计所有回文子串 |
| **312** | **戳气球** | [CN](https://leetcode.cn/problems/burst-balloons/) · [EN](https://leetcode.com/problems/burst-balloons/) | [题解 ↓](#312-戳气球--burst-balloons) |
| 375 | 猜数字大小 II | [CN](https://leetcode.cn/problems/guess-number-higher-or-lower-ii/) · [EN](https://leetcode.com/problems/guess-number-higher-or-lower-ii/) | f[i][j] = min over k (k + max(f[i][k-1], f[k+1][j])) |
| 1039 | 多边形三角剖分的最低得分 | [CN](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/) · [EN](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/) | f[i][j] = min over k (f[i][k] + f[k][j] + v[i]·v[k]·v[j]) |

<br>

#### 516. 最长回文子序列 / Longest Palindromic Subsequence

[CN](https://leetcode.cn/problems/longest-palindromic-subsequence/) · [EN](https://leetcode.com/problems/longest-palindromic-subsequence/) · Medium

**题意**：求字符串的最长回文子序列的长度。

**思路**：区间 DP：f[i][j] 表示 s[i..j] 的最长回文子序列。两端字符相同，都选上：f[i+1][j-1] + 2；不同，去掉一端：max(f[i+1][j], f[i][j-1])。i 倒序、j 正序枚举。

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        f = [[0] * n for _ in range(n)]
        # 1. f[i] 依赖 f[i+1]，所以 i 倒序
        for i in range(n - 1, -1, -1):
            f[i][i] = 1  # 单个字符是回文
            for j in range(i + 1, n):
                # 2. 两端相同，一起选
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1] + 2
                # 3. 两端不同，去掉其中一端
                else:
                    f[i][j] = max(f[i + 1][j], f[i][j - 1])
        return f[0][n - 1]
```

**复杂度**：时间 O(n²)；空间 O(n²)。

<br>

#### 312. 戳气球 / Burst Balloons

[CN](https://leetcode.cn/problems/burst-balloons/) · [EN](https://leetcode.com/problems/burst-balloons/) · Hard

**题意**：戳破气球 i 得到 nums[i-1]·nums[i]·nums[i+1] 枚硬币，求戳破所有气球能获得的最大硬币数。

**思路**：逆向思考：枚举区间 (i, j) 内最后一个被戳破的气球 k，它被戳时左右邻居正好是 i 和 j。在两端补上值为 1 的气球，f[i][j] = max(f[i][k] + f[k][j] + v[i]·v[k]·v[j])，i < k < j 都是开区间。

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # 1. 两端补 1
        v = [1] + nums + [1]
        n = len(v)
        # 2. f[i][j] = 戳破开区间 (i, j) 内所有气球的最大硬币数
        f = [[0] * n for _ in range(n)]
        for i in range(n - 3, -1, -1):
            for j in range(i + 2, n):
                # 3. 枚举最后戳破的气球 k
                f[i][j] = max(f[i][k] + f[k][j] + v[i] * v[k] * v[j] for k in range(i + 1, j))
        return f[0][n - 1]
```

**复杂度**：时间 O(n³)；空间 O(n²)。

<br>

### 树形

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **337** | **打家劫舍 III** | [CN](https://leetcode.cn/problems/house-robber-iii/) · [EN](https://leetcode.com/problems/house-robber-iii/) | [题解 ↓](#337-打家劫舍-iii--house-robber-iii) |
| 124 | 二叉树中的最大路径和 | [CN](https://leetcode.cn/problems/binary-tree-maximum-path-sum/) · [EN](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | 树形 DP：返回向下的最大链，在每个节点处拼接左右链，[题解](#124-二叉树中的最大路径和--binary-tree-maximum-path-sum) |
| 968 | 监控二叉树 | [CN](https://leetcode.cn/problems/binary-tree-cameras/) · [EN](https://leetcode.com/problems/binary-tree-cameras/) | 每个节点三种状态：装摄像头 / 被孩子覆盖 / 被父节点覆盖 |

<br>

#### 337. 打家劫舍 III / House Robber III

[CN](https://leetcode.cn/problems/house-robber-iii/) · [EN](https://leetcode.com/problems/house-robber-iii/) · Medium

**题意**：房屋排成二叉树，相连的两个房屋不能同时偷，求最高金额。

**思路**：树形 DP：DFS 返回两个值，(选当前节点的最大值, 不选当前节点的最大值)。选当前节点，左右孩子都不能选；不选当前节点，左右孩子各自选或不选取最大。

```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        # 1. 返回 (偷 node, 不偷 node) 时子树的最大金额
        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            if node is None:
                return 0, 0
            l_rob, l_not = dfs(node.left)
            r_rob, r_not = dfs(node.right)
            # 2. 偷 node：孩子都不能偷
            rob = node.val + l_not + r_not
            # 3. 不偷 node：孩子偷不偷都行
            not_rob = max(l_rob, l_not) + max(r_rob, r_not)
            return rob, not_rob

        return max(dfs(root))
```

**复杂度**：时间 O(n)；空间 O(h)。

<br>

### 状压

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 698 | 划分为k个相等的子集 | [CN](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/) · [EN](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/) | 状压 DP：f[mask] 表示用 mask 中的数时当前桶已装的和（模 target） |
| 526 | 优美的排列 | [CN](https://leetcode.cn/problems/beautiful-arrangement/) · [EN](https://leetcode.com/problems/beautiful-arrangement/) | f[mask]：前 popcount(mask) 个位置用 mask 中的数的方案数 |
| 847 | 访问所有节点的最短路径 | [CN](https://leetcode.cn/problems/shortest-path-visiting-all-nodes/) · [EN](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) | BFS，状态是 (当前节点, 已访问集合) |

<br>
<br>

## 16. 中频 · 常用数据结构

<br>

### 差分

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **1094** | **拼车** | [CN](https://leetcode.cn/problems/car-pooling/) · [EN](https://leetcode.com/problems/car-pooling/) | [题解 ↓](#1094-拼车--car-pooling) |
| 1109 | 航班预订统计 | [CN](https://leetcode.cn/problems/corporate-flight-bookings/) · [EN](https://leetcode.com/problems/corporate-flight-bookings/) | 差分数组：d[l] += x，d[r+1] -= x，最后求前缀和 |
| 370 | 区间加法 🔒 | [CN](https://leetcode.cn/problems/range-addition/) · [EN](https://leetcode.com/problems/range-addition/) | 差分模板题 |

<br>

#### 1094. 拼车 / Car Pooling

[CN](https://leetcode.cn/problems/car-pooling/) · [EN](https://leetcode.com/problems/car-pooling/) · Medium

**题意**：trips[i] = [人数, 上车点, 下车点]，车只能向东开，判断容量为 capacity 的车能否接送所有乘客。

**思路**：差分数组：乘客在 from 上车、to 下车，就是给区间 [from, to-1] 加 num。用差分 d[from] += num、d[to] -= num，再从左到右求前缀和，任何时刻超过 capacity 就返回 False。

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        # 1. 差分数组
        d = [0] * 1001
        for num, start, end in trips:
            d[start] += num  # 2. start 处上车
            d[end] -= num    # end 处下车
        # 3. 前缀和就是每个位置车上的人数
        s = 0
        for v in d:
            s += v
            if s > capacity:
                return False
        return True
```

**复杂度**：时间 O(n + U)，U = 1001；空间 O(U)。

<br>

### 单调队列

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **239** | **滑动窗口最大值** | [CN](https://leetcode.cn/problems/sliding-window-maximum/) · [EN](https://leetcode.com/problems/sliding-window-maximum/) | [题解 ↓](#239-滑动窗口最大值--sliding-window-maximum) |
| 1438 | 绝对差不超过限制的最长连续子数组 | [CN](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/) · [EN](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/) | 两个单调队列分别维护窗口最大值和最小值 |
| 862 | 和至少为 K 的最短子数组 | [CN](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/) · [EN](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/) | 前缀和 + 单调递增队列：队首满足条件就弹出并更新答案 |
| 1696 | 跳跃游戏 VI | [CN](https://leetcode.cn/problems/jump-game-vi/) · [EN](https://leetcode.com/problems/jump-game-vi/) | f[i] = nums[i] + max(f[i-k..i-1])，用单调队列维护窗口最大值 |

<br>

#### 239. 滑动窗口最大值 / Sliding Window Maximum

[CN](https://leetcode.cn/problems/sliding-window-maximum/) · [EN](https://leetcode.com/problems/sliding-window-maximum/) · Hard

**题意**：求大小为 k 的滑动窗口从左移到右时，每个窗口的最大值。

**思路**：单调队列：队列里存下标，对应的值单调递减。新元素入队前，把队尾比它小的都弹出（它们不可能再成为最大值）；队首如果已经离开窗口就弹出。队首就是当前窗口的最大值。

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        q = deque()  # 存下标，对应值单调递减
        for i, x in enumerate(nums):
            # 1. 入：弹出队尾所有 <= x 的元素
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            # 2. 出：队首离开窗口
            if i - q[0] >= k:
                q.popleft()
            # 3. 记录答案：窗口形成后，队首就是最大值
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans
```

**复杂度**：时间 O(n)；空间 O(k)。

<br>

### 树状数组 / 线段树

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **307** | **区域和检索 - 数组可修改** | [CN](https://leetcode.cn/problems/range-sum-query-mutable/) · [EN](https://leetcode.com/problems/range-sum-query-mutable/) | [题解 ↓](#307-区域和检索---数组可修改--range-sum-query---mutable) |
| 315 | 计算右侧小于当前元素的个数 | [CN](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) · [EN](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | 离散化 + 从右往左用树状数组计数；也可以归并排序 |
| 493 | 翻转对 | [CN](https://leetcode.cn/problems/reverse-pairs/) · [EN](https://leetcode.com/problems/reverse-pairs/) | 归并排序，合并前用双指针统计 nums[i] > 2·nums[j] 的对数 |

<br>

#### 307. 区域和检索 - 数组可修改 / Range Sum Query - Mutable

[CN](https://leetcode.cn/problems/range-sum-query-mutable/) · [EN](https://leetcode.com/problems/range-sum-query-mutable/) · Medium

**题意**：支持单点修改和区间求和。

**思路**：树状数组：tree[i] 存区间 (i - lowbit(i), i] 的和。单点更新时沿 i += lowbit(i) 往上更新；前缀和查询时沿 i -= lowbit(i) 往下累加。update 要加的是新旧值之差。

```python
class NumArray:
    def __init__(self, nums: List[int]):
        n = len(nums)
        self.nums = [0] * n
        self.tree = [0] * (n + 1)  # 1. 下标从 1 开始
        for i, x in enumerate(nums):
            self.update(i, x)

    def update(self, index: int, val: int) -> None:
        # 2. 计算增量，沿 lowbit 往上更新
        delta = val - self.nums[index]
        self.nums[index] = val
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    # 3. 前缀和 nums[0] + ... + nums[i-1]
    def prefix_sum(self, i: int) -> int:
        s = 0
        while i:
            s += self.tree[i]
            i &= i - 1  # 等价于 i -= lowbit(i)
        return s

    def sumRange(self, left: int, right: int) -> int:
        # 4. 两个前缀和相减
        return self.prefix_sum(right + 1) - self.prefix_sum(left)
```

**复杂度**：初始化 O(n log n)，update 和 sumRange 都是 O(log n)；空间 O(n)。

<br>
<br>

## 17. 中频 · 贪心

<br>

### 字典序

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **179** | **最大数** | [CN](https://leetcode.cn/problems/largest-number/) · [EN](https://leetcode.com/problems/largest-number/) | [题解 ↓](#179-最大数--largest-number) |
| 670 | 最大交换 | [CN](https://leetcode.cn/problems/maximum-swap/) · [EN](https://leetcode.com/problems/maximum-swap/) | 从右往左记录最大数字的位置，找到最左边能变大的位置交换 |
| 738 | 单调递增的数字 | [CN](https://leetcode.cn/problems/monotone-increasing-digits/) · [EN](https://leetcode.com/problems/monotone-increasing-digits/) | 从右往左，遇到前一位更大就把前一位减一，后面全填 9 |

<br>

#### 179. 最大数 / Largest Number

[CN](https://leetcode.cn/problems/largest-number/) · [EN](https://leetcode.com/problems/largest-number/) · Medium

**题意**：重新排列非负整数，使它们拼接成的数最大。

**思路**：自定义排序：对于两个数 a、b，如果 a+b（字符串拼接）> b+a，a 就应该排在前面。按这个规则排序后拼接；全是 0 时要返回 "0"。

```python
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        strs = list(map(str, nums))
        # 1. a 排在 b 前面 ⇔ a+b > b+a
        def cmp(a: str, b: str) -> int:
            if a + b > b + a:
                return -1
            if a + b < b + a:
                return 1
            return 0

        strs.sort(key=cmp_to_key(cmp))
        # 2. 处理全是 0 的情况
        return '0' if strs[0] == '0' else ''.join(strs)
```

**复杂度**：时间 O(n log n · L)；空间 O(n·L)。

<br>

### 反悔贪心

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **630** | **课程表 III** | [CN](https://leetcode.cn/problems/course-schedule-iii/) · [EN](https://leetcode.com/problems/course-schedule-iii/) | [题解 ↓](#630-课程表-iii--course-schedule-iii) |
| 871 | 最低加油次数 | [CN](https://leetcode.cn/problems/minimum-number-of-refueling-stops/) · [EN](https://leetcode.com/problems/minimum-number-of-refueling-stops/) | 路过的加油站放进大顶堆，油不够时取油量最大的加 |
| 1642 | 可以到达的最远建筑 | [CN](https://leetcode.cn/problems/furthest-building-you-can-reach/) · [EN](https://leetcode.com/problems/furthest-building-you-can-reach/) | 梯子留给最大的 k 次落差：小顶堆存落差，超过 k 个就用砖块填最小的 |

<br>

#### 630. 课程表 III / Course Schedule III

[CN](https://leetcode.cn/problems/course-schedule-iii/) · [EN](https://leetcode.com/problems/course-schedule-iii/) · Hard

**题意**：每门课有时长和最晚结束时间，同一时间只能上一门，求最多能上几门课。

**思路**：按截止时间排序，依次尝试上课。能在截止前上完就上，课时放进大顶堆；上不完时，如果堆里最长的课比当前课更长，就反悔：退掉那门长课改上当前课，总时间变短、课数不变。

```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        # 1. 按截止时间排序
        courses.sort(key=lambda c: c[1])
        h = []  # 大顶堆（存相反数），存已选课程的时长
        day = 0
        for duration, last_day in courses:
            # 2. 能上完就直接上
            if day + duration <= last_day:
                day += duration
                heappush(h, -duration)
            # 3. 上不完：用当前课替换已选的最长课（反悔）
            elif h and duration < -h[0]:
                day += duration + heappop(h)  # heappop 返回的是负数
                heappush(h, -duration)
        return len(h)
```

**复杂度**：时间 O(n log n)；空间 O(n)。

<br>
<br>

## 18. 中频 · 字符串

<br>

### KMP

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **28** | **找出字符串中第一个匹配项的下标** | [CN](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) · [EN](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/) | [题解 ↓](#28-找出字符串中第一个匹配项的下标--find-the-index-of-the-first-occurrence-in-a-string) |
| 459 | 重复的子字符串 | [CN](https://leetcode.cn/problems/repeated-substring-pattern/) · [EN](https://leetcode.com/problems/repeated-substring-pattern/) | 前缀函数：π[n-1] > 0 且 n % (n - π[n-1]) == 0 |
| 214 | 最短回文串 | [CN](https://leetcode.cn/problems/shortest-palindrome/) · [EN](https://leetcode.com/problems/shortest-palindrome/) | 对 s + '#' + reverse(s) 求前缀函数，得到最长回文前缀 |
| 1392 | 最长快乐前缀 | [CN](https://leetcode.cn/problems/longest-happy-prefix/) · [EN](https://leetcode.com/problems/longest-happy-prefix/) | 前缀函数的最后一个值就是答案长度 |

<br>

#### 28. 找出字符串中第一个匹配项的下标 / Find the Index of the First Occurrence in a String

[CN](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) · [EN](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/) · Easy

**题意**：返回 needle 在 haystack 中第一次出现的下标，不存在返回 -1。

**思路**：KMP：先求模式串的前缀函数 pi[i]（needle[:i+1] 的最长相等真前后缀长度）。匹配时失配就让 j = pi[j-1] 跳回去，主串指针不回退。

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        m = len(needle)
        # 1. 前缀函数
        pi = [0] * m
        j = 0
        for i in range(1, m):
            b = needle[i]
            while j and b != needle[j]:
                j = pi[j - 1]  # 2. 失配，跳到更短的相等前后缀
            if b == needle[j]:
                j += 1
            pi[i] = j

        # 3. 在主串上匹配
        j = 0
        for i, b in enumerate(haystack):
            while j and b != needle[j]:
                j = pi[j - 1]
            if b == needle[j]:
                j += 1
            # 4. 匹配成功，返回起点
            if j == m:
                return i - m + 1
        return -1
```

**复杂度**：时间 O(n + m)；空间 O(m)。

<br>
<br>

## 19. 中频 · 数学

<br>

### 质数

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **204** | **计数质数** | [CN](https://leetcode.cn/problems/count-primes/) · [EN](https://leetcode.com/problems/count-primes/) | [题解 ↓](#204-计数质数--count-primes) |
| 263 | 丑数 | [CN](https://leetcode.cn/problems/ugly-number/) · [EN](https://leetcode.com/problems/ugly-number/) | 反复除以 2、3、5，看最后是否为 1 |
| 264 | 丑数 II | [CN](https://leetcode.cn/problems/ugly-number-ii/) · [EN](https://leetcode.com/problems/ugly-number-ii/) | 三指针分别乘 2、3、5，每次取最小的 |

<br>

#### 204. 计数质数 / Count Primes

[CN](https://leetcode.cn/problems/count-primes/) · [EN](https://leetcode.com/problems/count-primes/) · Medium

**题意**：求小于 n 的质数个数。

**思路**：埃氏筛：从 2 开始，如果 i 是质数，就把 i², i²+i, … 都标记为合数（小于 i² 的倍数已经被更小的质因子标记过了）。

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 3:
            return 0
        # 1. is_prime[i] 表示 i 是否为质数
        is_prime = [True] * n
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n - 1) + 1):
            if is_prime[i]:
                # 2. 从 i*i 开始标记 i 的倍数
                for j in range(i * i, n, i):
                    is_prime[j] = False
        # 3. 统计质数个数
        return sum(is_prime)
```

**复杂度**：时间 O(n log log n)；空间 O(n)。

<br>

### 最大公约数

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| 1979 | 找出数组的最大公约数 | [CN](https://leetcode.cn/problems/find-greatest-common-divisor-of-array/) · [EN](https://leetcode.com/problems/find-greatest-common-divisor-of-array/) | gcd(min(nums), max(nums)) |
| 914 | 卡牌分组 | [CN](https://leetcode.cn/problems/x-of-a-kind-in-a-deck-of-cards/) · [EN](https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/) | 所有数字出现次数的 gcd ≥ 2 |

<br>

### 快速幂

| 题号 | 题目 | 链接 | 思路 |
|---|---|---|---|
| **50** | **Pow(x, n)** | [CN](https://leetcode.cn/problems/powx-n/) · [EN](https://leetcode.com/problems/powx-n/) | [题解 ↓](#50-powx-n--powx-n) |
| 372 | 超级次方 | [CN](https://leetcode.cn/problems/super-pow/) · [EN](https://leetcode.com/problems/super-pow/) | a^(10x+d) = (a^x)^10 · a^d，按 b 的每一位迭代，模 1337 |

<br>

#### 50. Pow(x, n) / Pow(x, n)

[CN](https://leetcode.cn/problems/powx-n/) · [EN](https://leetcode.com/problems/powx-n/) · Medium

**题意**：实现 pow(x, n)。

**思路**：快速幂：把 n 写成二进制，x^n 就是 x 的若干个 2^k 次方的乘积。每次看 n 的最低位，是 1 就把当前的 x 乘进答案，然后 x 平方、n 右移。n 为负数时先取 x 的倒数。

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 1. 负指数：x^n = (1/x)^(-n)
        if n < 0:
            x, n = 1 / x, -n
        ans = 1.0
        while n:
            # 2. 最低位是 1，乘上当前的 x^(2^k)
            if n & 1:
                ans *= x
            # 3. x 平方，n 右移一位
            x *= x
            n >>= 1
        return ans
```

**复杂度**：时间 O(log n)；空间 O(1)。
