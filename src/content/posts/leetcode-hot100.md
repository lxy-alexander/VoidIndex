---
title: LeetCode Hot 100 题解
description: 按 LeetCode「热题 100」的 17 个分类整理全部 100 题，每题包含题目与示例、解法思路、复杂度和 Python 代码。
pubDatetime: 2026-11-13T23:00:00Z
tags:
  - leetcode
  - algorithm
draft: false
---

按 LeetCode「热题 100」学习计划的 17 个分类整理，共 100 题。全部分类已合并到本文件。每题包含：完整题目与示例（点标题右侧 🔗 跳转原题）、解法思路、复杂度、Python 代码。

代码可直接粘贴到 LeetCode 提交。本地运行时需自行补充 `typing`、`collections`、`heapq`、`functools` 等 import，以及 `ListNode` / `TreeNode` 定义。

| # | 分类 | 题数 | 核心思路 |
|---|------|------|----------|
| 01 | [哈希 Hash](#哈希-hash) | 3 | 用空间换查找时间 |
| 02 | [双指针 Two Pointers](#双指针-two-pointers) | 4 | 排序后相向收缩，或快慢指针 |
| 03 | [滑动窗口 Sliding Window](#滑动窗口-sliding-window) | 2 | 维护窗口不变量 |
| 04 | [子串 Substring](#子串-substring) | 3 | 前缀和、单调队列、可变窗口 |
| 05 | [普通数组 Array](#普通数组-array) | 5 | Kadane、排序、原地哈希 |
| 06 | [矩阵 Matrix](#矩阵-matrix) | 4 | 原地标记、边界模拟 |
| 07 | [链表 Linked List](#链表-linked-list) | 14 | 哨兵节点、快慢指针、反转 |
| 08 | [二叉树 Binary Tree](#二叉树-binary-tree) | 15 | 递归定义子问题、后序汇总 |
| 09 | [图论 Graph](#图论-graph) | 4 | DFS/BFS、拓扑排序、Trie |
| 10 | [回溯 Backtracking](#回溯-backtracking) | 8 | 选择、递归、撤销 |
| 11 | [二分查找 Binary Search](#二分查找-binary-search) | 6 | 找到可二分的单调性质 |
| 12 | [栈 Stack](#栈-stack) | 5 | 匹配、单调栈 |
| 13 | [堆 Heap](#堆-heap) | 3 | Top K、对顶堆 |
| 14 | [贪心 Greedy](#贪心-greedy) | 4 | 维护当前最优边界 |
| 15 | [动态规划 DP](#动态规划-dp) | 10 | 定义状态、转移、边界 |
| 16 | [多维动态规划 Multi-dim DP](#多维动态规划-multi-dimensional-dp) | 5 | 网格 / 双序列 / 区间 |
| 17 | [技巧 Tricks](#技巧-tricks) | 5 | 位运算、投票、环检测 |

<br>
<br>

# 哈希 Hash

<br>

## 1. 两数之和 / Two Sum [🔗](https://leetcode.com/problems/two-sum/)

难度：Easy

给定一个整数数组 `nums` 和一个整数目标值 `target`，在数组中找出和为 `target` 的那两个整数，并返回它们的数组下标。

可以假设每种输入只对应一个答案，且同一个元素不能使用两次。答案可以按任意顺序返回。

**示例 1：**

```text
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：nums[0] + nums[1] == 9，返回 [0, 1]。
```

**示例 2：**

```text
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**示例 3：**

```text
输入：nums = [3,3], target = 6
输出：[0,1]
```

**进阶：** 你能想出时间复杂度小于 O(n²) 的算法吗？

**解法**：遍历时用哈希表记录「值 → 下标」，对每个数查 `target - x` 是否已出现。先查后存，避免同一元素用两次。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, x in enumerate(nums):
            if target - x in seen:
                return [seen[target - x], i]
            seen[x] = i
        return []
```

<br>

## 49. 字母异位词分组 / Group Anagrams [🔗](https://leetcode.com/problems/group-anagrams/)

难度：Medium

给你一个字符串数组 `strs`，请把字母异位词组合在一起，可以按任意顺序返回结果列表。

字母异位词是指由相同字母重新排列得到的单词，每个字母的出现次数也相同。

**示例 1：**

```text
输入：strs = ["eat","tea","tan","ate","nat","bat"]
输出：[["bat"],["nat","tan"],["ate","eat","tea"]]
```

**示例 2：**

```text
输入：strs = [""]
输出：[[""]]
```

**示例 3：**

```text
输入：strs = ["a"]
输出：[["a"]]
```

**解法**：异位词排序后相同，用排序后的字符串作哈希键。也可用 26 维计数元组作键，把单词排序的 O(k log k) 降到 O(k)。

**复杂度**：时间 O(nk log k)，空间 O(nk)，k 为最长单词长度。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for s in strs:
            groups["".join(sorted(s))].append(s)
        return list(groups.values())
```

<br>

## 128. 最长连续序列 / Longest Consecutive Sequence [🔗](https://leetcode.com/problems/longest-consecutive-sequence/)

难度：Medium

给定一个未排序的整数数组 `nums`，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请设计并实现时间复杂度为 O(n) 的算法。

**示例 1：**

```text
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]，长度为 4。
```

**示例 2：**

```text
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

**示例 3：**

```text
输入：nums = [1,0,1,2]
输出：3
```

**解法**：放入集合后，只从序列起点（`x - 1` 不在集合中）开始向后数。每个数最多被访问常数次。注意要遍历集合而不是原数组，否则大量重复值会退化。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        best = 0
        for x in s:
            if x - 1 not in s:
                y = x + 1
                while y in s:
                    y += 1
                best = max(best, y - x)
        return best
```

<br>

## 560. Subarray Sum Equals K [🔗](https://leetcode.com/problems/subarray-sum-equals-k/)

Given an array of integers `nums` and an integer `k`, return *the total number of subarrays whose sum equals to* `k`.

A subarray is a contiguous **non-empty** sequence of elements within an array.

**Example 1:**

```
Input: nums = [1,1,1], k = 2
Output: 2
```

**Example 2:**

```
Input: nums = [1,2,3], k = 3
Output: 2
```

````python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        cnt = {0: 1}
        prefix = 0
        for x in nums:
            prefix += x
            diff = prefix - k
            if diff in cnt:
                ans += cnt[diff]
            cnt[prefix] = cnt.get(prefix, 0) + 1
        return ans
        
        
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        cnt = defaultdict(int)
        cnt[0] = 1
        prefix = 0
        for x in nums:
            prefix += x
            diff = prefix - k
            if diff in cnt:
                ans += cnt[diff]
            cnt[prefix] += 1
        return ans
        
````

<br>
<br>

# 双指针 Two Pointers

<br>

## 283. 移动零 / Move Zeroes [🔗](https://leetcode.com/problems/move-zeroes/)

难度：Easy

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组末尾，同时保持非零元素的相对顺序。

必须原地操作，不能复制数组。

**示例 1：**

```text
输入：nums = [0,1,0,3,12]
输出：[1,3,12,0,0]
```

**示例 2：**

```text
输入：nums = [0]
输出：[0]
```

**进阶：** 你能尽量减少完成的操作次数吗？

**解法**：慢指针 `slow` 指向下一个非零元素应放的位置，快指针遇到非零就交换。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
```

<br>

## 11. 盛最多水的容器 / Container With Most Water [🔗](https://leetcode.com/problems/container-with-most-water/)

难度：Medium

给定一个长度为 `n` 的整数数组 `height`。有 `n` 条竖线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])`。

找出其中两条线，使它们与 x 轴构成的容器能装最多的水，返回容器能储存的最大水量。容器不能倾斜。

**示例 1：**

```text
输入：height = [1,8,6,2,5,4,8,3,7]
输出：49
解释：选下标 1 和 8 的两条线，宽 7，高 min(8, 7) = 7，面积 49。
```

**示例 2：**

```text
输入：height = [1,1]
输出：1
```

**解法**：左右指针相向移动，每次移动较矮的一侧。因为面积受矮边限制，移动高边时宽度变小而高度不会变大，不可能更优。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def maxArea(self, height: list[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            ans = max(ans, area)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans
```

<br>

## 15. 三数之和 / 3Sum [🔗](https://leetcode.com/problems/3sum/)

难度：Medium

给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k`、`j != k`，且 `nums[i] + nums[j] + nums[k] == 0`。请返回所有和为 0 且不重复的三元组。

答案中不能包含重复的三元组。

**示例 1：**

```text
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0
不同的三元组是 [-1,0,1] 和 [-1,-1,2]，输出顺序不重要。
```

**示例 2：**

```text
输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0。
```

**示例 3：**

```text
输入：nums = [0,0,0]
输出：[[0,0,0]]
```

**解法**：排序后固定第一个数，其余部分用相向双指针。三处去重：跳过相同的第一个数；找到答案后左右指针各跳过相同值。`nums[i] > 0` 时可提前结束。

**复杂度**：时间 O(n²)，空间 O(log n)（排序栈，不计输出）。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
        return res
```

<br>

## 42. 接雨水 / Trapping Rain Water [🔗](https://leetcode.com/problems/trapping-rain-water/)

难度：Hard

给定 `n` 个非负整数，表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子下雨之后能接多少雨水。

**示例 1：**

```text
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：该高度图最多可以接 6 个单位的雨水。
```

**示例 2：**

```text
输入：height = [4,2,0,3,2,5]
输出：9
```

**解法**：位置 i 的水量是 `min(左侧最高, 右侧最高) - height[i]`。双指针维护 `lmax`、`rmax`：哪边的最大值小，那边的水量就已确定，结算后向内移动。也可用单调栈按层计算。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        lmax = rmax = res = 0
        while l < r:
            lmax = max(lmax, height[l])
            rmax = max(rmax, height[r])
            if lmax < rmax:
                res += lmax - height[l]
                l += 1
            else:
                res += rmax - height[r]
                r -= 1
        return res
```

<br>
<br>

# 滑动窗口 Sliding Window

<br>

## 3. 无重复字符的最长子串 / Longest Substring Without Repeating Characters [🔗](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

难度：Medium

给定一个字符串 `s`，找出其中不含有重复字符的最长子串的长度。

**示例 1：**

```text
输入：s = "abcabcbb"
输出：3
解释：无重复字符的最长子串是 "abc"，长度为 3。
```

**示例 2：**

```text
输入：s = "bbbbb"
输出：1
解释：最长子串是 "b"，长度为 1。
```

**示例 3：**

```text
输入：s = "pwwkew"
输出：3
解释：最长子串是 "wke"，长度为 3。注意答案必须是子串，"pwke" 是子序列，不是子串。
```

**解法**：记录每个字符最后出现的位置。遇到窗口内的重复字符时，左边界直接跳到该字符上次位置之后。

**复杂度**：时间 O(n)，空间 O(字符集大小)。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}
        left = best = 0
        for right, c in enumerate(s):
            if c in last and last[c] >= left:
                left = last[c] + 1
            last[c] = right
            best = max(best, right - left + 1)
        return best
```

<br>

## 438. 找到字符串中所有字母异位词 / Find All Anagrams in a String [🔗](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

难度：Medium

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的异位词子串，返回这些子串的起始下标，不考虑输出顺序。

异位词指由相同字母重排形成的字符串（包括相同的字符串）。

**示例 1：**

```text
输入：s = "cbaebabacd", p = "abc"
输出：[0,6]
解释：
起始下标 0 的子串是 "cba"，是 "abc" 的异位词。
起始下标 6 的子串是 "bac"，是 "abc" 的异位词。
```

**示例 2：**

```text
输入：s = "abab", p = "ab"
输出：[0,1,2]
解释：
下标 0 的子串 "ab"、下标 1 的子串 "ba"、下标 2 的子串 "ab" 都是 "ab" 的异位词。
```

**解法**：长度固定为 `len(p)` 的窗口，维护 26 个字母的计数，比较窗口计数与 `p` 的计数。

**复杂度**：时间 O(26n)，空间 O(26)。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        n, m = len(s), len(p)
        if n < m:
            return []
        need, win = [0] * 26, [0] * 26
        for c in p:
            need[ord(c) - 97] += 1
        res = []
        for i in range(n):
            win[ord(s[i]) - 97] += 1
            if i >= m:
                win[ord(s[i - m]) - 97] -= 1
            if win == need:
                res.append(i - m + 1)
        return res
```

<br>
<br>

# 子串 Substring

<br>

## 560. 和为 K 的子数组 / Subarray Sum Equals K [🔗](https://leetcode.com/problems/subarray-sum-equals-k/)

难度：Medium

给你一个整数数组 `nums` 和一个整数 `k`，统计并返回该数组中和为 `k` 的子数组个数。

子数组是数组中元素的连续非空序列。

**示例 1：**

```text
输入：nums = [1,1,1], k = 2
输出：2
```

**示例 2：**

```text
输入：nums = [1,2,3], k = 3
输出：2
```

**解法**：前缀和 + 哈希。子数组 `(j, i]` 的和为 `pre[i] - pre[j]`，所以统计之前有多少个前缀和等于 `pre - k`。数组含负数，不能用滑动窗口。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = defaultdict(int)
        cnt[0] = 1
        pre = res = 0
        for x in nums:
            pre += x
            res += cnt[pre - k]
            cnt[pre] += 1
        return res
```

<br>

## 239. 滑动窗口最大值 / Sliding Window Maximum [🔗](https://leetcode.com/problems/sliding-window-maximum/)

难度：Hard

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组最左侧移动到最右侧，每次只向右移动一位，你只能看到窗口内的 `k` 个数字。

返回每个窗口中的最大值。

**示例 1：**

```text
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```text
输入：nums = [1], k = 1
输出：[1]
```

**解法**：单调递减队列存下标。新元素入队前弹出所有不大于它的队尾元素（它们不可能再成为最大值），队首过期则弹出，队首即窗口最大值。

**复杂度**：时间 O(n)，空间 O(k)。

```python
class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        q = deque()
        ans = []
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)

            if i < k - 1:
                continue

            ans.append(nums[q[0]])

            if q[0] == i - k + 1:
                q.popleft()
        return ans
```

<br>

## 76. 最小覆盖子串 / Minimum Window Substring [🔗](https://leetcode.com/problems/minimum-window-substring/)

难度：Hard

给你一个字符串 `s` 和一个字符串 `t`，返回 `s` 中涵盖 `t` 所有字符的最小子串。如果不存在这样的子串，返回空字符串 `""`。

对于 `t` 中重复的字符，子串中该字符的数量必须不少于 `t` 中的数量。如果存在答案，可以保证它是唯一的。

**示例 1：**

```text
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含了 t 中的 'A'、'B' 和 'C'。
```

**示例 2：**

```text
输入：s = "a", t = "a"
输出："a"
```

**示例 3：**

```text
输入：s = "a", t = "aa"
输出：""
解释：t 中的两个 'a' 都要出现在子串中，s 只有一个 'a'，所以返回空字符串。
```

**进阶：** 你能设计一个 O(m + n) 时间的算法吗？

**解法**：可变窗口。`need` 记录还缺的字符数，`missing` 记录总缺口。右指针扩张到 `missing == 0` 后，收缩左指针去掉多余字符，再更新答案。

**复杂度**：时间 O(|s| + |t|)，空间 O(字符集大小)。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = Counter(t)
        missing = len(t)
        left = start = 0
        best = float("inf")
        for right, c in enumerate(s):
            if need[c] > 0:
                missing -= 1
            need[c] -= 1
            if missing == 0:
                while need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                if right - left + 1 < best:
                    best = right - left + 1
                    start = left
                need[s[left]] += 1
                missing += 1
                left += 1
        return "" if best == float("inf") else s[start:start + best]
```

<br>
<br>

# 普通数组 Array

<br>

## 53. 最大子数组和 / Maximum Subarray [🔗](https://leetcode.com/problems/maximum-subarray/)

难度：Medium

给你一个整数数组 `nums`，找出一个和最大的连续子数组（至少包含一个元素），返回其最大和。

**示例 1：**

```text
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**示例 2：**

```text
输入：nums = [1]
输出：1
```

**示例 3：**

```text
输入：nums = [5,4,-1,7,8]
输出：23
```

**进阶：** 如果你已经实现了 O(n) 的解法，尝试用更精妙的分治法求解。

**解法**：Kadane 算法。`cur` 是以当前元素结尾的最大和；前面的和为负时直接丢弃。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur = best = nums[0]
        for x in nums[1:]:
            cur = max(cur + x, x)
            best = max(best, cur)
        return best
```

<br>

## 56. 合并区间 / Merge Intervals [🔗](https://leetcode.com/problems/merge-intervals/)

难度：Medium

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [start_i, end_i]`。请合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

**示例 1：**

```text
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠，合并为 [1,6]。
```

**示例 2：**

```text
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 端点相接，也视为重叠。
```

**解法**：按左端点排序。当前区间与结果末尾重叠就扩展右端点，否则追加。

**复杂度**：时间 O(n log n)，空间 O(n)。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        res = []
        for s, e in intervals:
            if res and s <= res[-1][1]:
                res[-1][1] = max(res[-1][1], e)
            else:
                res.append([s, e])
        return res
```

<br>

## 189. 轮转数组 / Rotate Array [🔗](https://leetcode.com/problems/rotate-array/)

难度：Medium

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

**示例 1：**

```text
输入：nums = [1,2,3,4,5,6,7], k = 3
输出：[5,6,7,1,2,3,4]
解释：
向右轮转 1 步：[7,1,2,3,4,5,6]
向右轮转 2 步：[6,7,1,2,3,4,5]
向右轮转 3 步：[5,6,7,1,2,3,4]
```

**示例 2：**

```text
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
```

**进阶：** 尽可能想出多种解法，至少有三种。你能用空间复杂度为 O(1) 的原地算法解决吗？

**解法**：三次反转：整体反转，再分别反转前 k 个和剩余部分。`k` 需先对 n 取模。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        def rev(l: int, r: int) -> None:
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1

        n = len(nums)
        k %= n
        rev(0, n - 1)
        rev(0, k - 1)
        rev(k, n - 1)
```

<br>

## 238. 除自身以外数组的乘积 / Product of Array Except Self [🔗](https://leetcode.com/problems/product-of-array-except-self/)

难度：Medium

给你一个整数数组 `nums`，返回数组 `answer`，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

题目保证任意前缀和后缀元素的乘积都在 32 位整数范围内。请不要使用除法，且在 O(n) 时间内完成。

**示例 1：**

```text
输入：nums = [1,2,3,4]
输出：[24,12,8,6]
```

**示例 2：**

```text
输入：nums = [-1,1,0,-3,3]
输出：[0,0,9,0,0]
```

**进阶：** 你能在 O(1) 的额外空间内完成吗？（输出数组不算额外空间。）

**解法**：结果 = 左侧乘积 × 右侧乘积。先用输出数组存左侧乘积，再从右向左用一个变量累乘。

**复杂度**：时间 O(n)，空间 O(1)（不计输出）。

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        for i in range(1, n):
            res[i] = res[i - 1] * nums[i - 1]
        right = 1
        for i in range(n - 1, -1, -1):
            res[i] *= right
            right *= nums[i]
        return res
```

<br>

## 41. 缺失的第一个正数 / First Missing Positive [🔗](https://leetcode.com/problems/first-missing-positive/)

难度：Hard

给你一个未排序的整数数组 `nums`，找出其中没有出现的最小的正整数。

请实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

**示例 1：**

```text
输入：nums = [1,2,0]
输出：3
解释：1 和 2 都在数组中，3 不在。
```

**示例 2：**

```text
输入：nums = [3,4,-1,1]
输出：2
解释：1 在数组中，2 不在。
```

**示例 3：**

```text
输入：nums = [7,8,9,11,12]
输出：1
解释：最小的正整数 1 不在数组中。
```

**解法**：答案一定在 `[1, n+1]` 内。原地哈希：把值 `x` 交换到下标 `x-1`，然后找第一个 `nums[i] != i+1` 的位置。`while` 中的相等判断可防止重复值造成死循环。

**复杂度**：时间 O(n)（每次交换归位一个数），空间 O(1)。

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                j = nums[i] - 1
                nums[i], nums[j] = nums[j], nums[i]
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1
```

<br>
<br>

# 矩阵 Matrix

<br>

## 73. 矩阵置零 / Set Matrix Zeroes [🔗](https://leetcode.com/problems/set-matrix-zeroes/)

难度：Medium

给定一个 `m x n` 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。

**示例 1：**

```text
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]
```

**示例 2：**

```text
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

**进阶：** 直观的 O(mn) 额外空间方案不是好办法；O(m + n) 额外空间有所改进，但仍不是最好的。你能想出一个只用常量空间的方案吗？

**解法**：用第一行、第一列作标记位，另用两个布尔值记录它们本身是否含 0。最后再处理第一行和第一列，避免标记被污染。

**复杂度**：时间 O(mn)，空间 O(1)。

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        row0 = any(matrix[0][j] == 0 for j in range(n))
        col0 = any(matrix[i][0] == 0 for i in range(m))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row0:
            for j in range(n):
                matrix[0][j] = 0
        if col0:
            for i in range(m):
                matrix[i][0] = 0
```

<br>

## 54. 螺旋矩阵 / Spiral Matrix [🔗](https://leetcode.com/problems/spiral-matrix/)

难度：Medium

给你一个 `m` 行 `n` 列的矩阵 `matrix`，按照顺时针螺旋顺序，返回矩阵中的所有元素。

**示例 1：**

```text
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 2：**

```text
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

**解法**：维护上下左右四条边界，每走完一条边收缩一次。走下边和左边前要再判断边界，防止单行或单列被重复访问。

**复杂度**：时间 O(mn)，空间 O(1)（不计输出）。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        res = []
        while top <= bottom and left <= right:
            for j in range(left, right + 1):
                res.append(matrix[top][j])
            top += 1
            for i in range(top, bottom + 1):
                res.append(matrix[i][right])
            right -= 1
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    res.append(matrix[bottom][j])
                bottom -= 1
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    res.append(matrix[i][left])
                left += 1
        return res
```

<br>

## 48. 旋转图像 / Rotate Image [🔗](https://leetcode.com/problems/rotate-image/)

难度：Medium

给定一个 `n × n` 的二维矩阵 `matrix` 表示一个图像，请将图像顺时针旋转 90 度。

必须原地旋转，即直接修改输入的二维矩阵，不要使用另一个矩阵。

**示例 1：**

```text
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**示例 2：**

```text
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

**解法**：顺时针 90 度 = 先沿主对角线转置，再反转每一行。

**复杂度**：时间 O(n²)，空间 O(1)。

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()
```

<br>

## 240. 搜索二维矩阵 II / Search a 2D Matrix II [🔗](https://leetcode.com/problems/search-a-2d-matrix-ii/)

难度：Medium

编写一个高效的算法，在 `m x n` 矩阵 `matrix` 中搜索目标值 `target`。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

**示例 1：**

```text
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```

**示例 2：**

```text
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

**解法**：从右上角出发。当前值大于 target 则左移（排除一列），小于则下移（排除一行）。

**复杂度**：时间 O(m + n)，空间 O(1)。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        i, j = 0, len(matrix[0]) - 1
        while i < len(matrix) and j >= 0:
            if matrix[i][j] == target:
                return True
            if matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False
```

<br>
<br>

# 链表 Linked List

<br>

## 160. 相交链表 / Intersection of Two Linked Lists [🔗](https://leetcode.com/problems/intersection-of-two-linked-lists/)

难度：Easy

给你两个单链表的头节点 `headA` 和 `headB`，找出并返回两个单链表相交的起始节点。如果两个链表不相交，返回 `null`。

题目保证整个链式结构中不存在环。函数返回后，链表必须保持原有结构。

评测时会给出 `intersectVal`（相交节点的值，不相交时为 0）、`listA`、`listB`，以及 `skipA`、`skipB`（分别从 A、B 头部跳过多少个节点到达相交节点）。

**示例 1：**

```text
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8。A 中相交节点前有 2 个节点，B 中相交节点前有 3 个节点。
注意：值为 1 的节点不是相交节点，A 中第二个节点和 B 中第三个节点是内存中两个不同的节点。
```

**示例 2：**

```text
输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
```

**示例 3：**

```text
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：No intersection
解释：两个链表不相交，返回 null。
```

**进阶：** 你能否设计一个时间复杂度 O(m + n)、仅用 O(1) 内存的解决方案？

**解法**：两指针各走完自己的链表后切到对方链表，走过的总长度相同（a + b），因此会在交点相遇；不相交时同时到达 None。

**复杂度**：时间 O(m + n)，空间 O(1)。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        a, b = headA, headB
        while a is not b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a
```

<br>

## 206. 反转链表 / Reverse Linked List [🔗](https://leetcode.com/problems/reverse-linked-list/)

难度：Easy

给你单链表的头节点 `head`，请你反转链表，并返回反转后的链表。

**示例 1：**

```text
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

**示例 2：**

```text
输入：head = [1,2]
输出：[2,1]
```

**示例 3：**

```text
输入：head = []
输出：[]
```

**进阶：** 链表可以选用迭代或递归方式完成反转，你能用两种方法解决吗？

**解法**：迭代，用 `prev` 和 `cur` 逐个改变 next 指向。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev
```

<br>

## 234. 回文链表 / Palindrome Linked List [🔗](https://leetcode.com/problems/palindrome-linked-list/)

难度：Easy

给你一个单链表的头节点 `head`，判断该链表是否为回文链表。如果是，返回 `true`；否则，返回 `false`。

**示例 1：**

```text
输入：head = [1,2,2,1]
输出：true
```

**示例 2：**

```text
输入：head = [1,2]
输出：false
```

**进阶：** 你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

**解法**：快慢指针找到中点，反转后半段，再与前半段逐一比较。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        while slow:
            nxt = slow.next
            slow.next = prev
            prev = slow
            slow = nxt
        a, b = head, prev
        while b:
            if a.val != b.val:
                return False
            a = a.next
            b = b.next
        return True
```

<br>

## 141. 环形链表 / Linked List Cycle [🔗](https://leetcode.com/problems/linked-list-cycle/)

难度：Easy

给你一个链表的头节点 `head`，判断链表中是否有环。

如果链表中某个节点可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。评测系统用 `pos` 表示链表尾连接到链表中的位置（下标从 0 开始），`pos` 不作为参数传入。

如果链表中存在环，返回 `true`；否则，返回 `false`。

**示例 1：**

```text
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

```text
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

```text
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

**进阶：** 你能用 O(1)（即常量）内存解决此问题吗？

**解法**：Floyd 判圈：快指针每次走两步，慢指针走一步，有环必相遇。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False
```

<br>

## 142. 环形链表 II / Linked List Cycle II [🔗](https://leetcode.com/problems/linked-list-cycle-ii/)

难度：Medium

给定一个链表的头节点 `head`，返回链表开始入环的第一个节点。如果链表无环，返回 `null`。

评测系统用 `pos` 表示链表尾连接到链表中的位置（下标从 0 开始），`pos` 不作为参数传入。不允许修改链表。

**示例 1：**

```text
输入：head = [3,2,0,-4], pos = 1
输出：返回下标为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

```text
输入：head = [1,2], pos = 0
输出：返回下标为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

```text
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

**进阶：** 你是否可以使用 O(1) 空间解决此题？

**解法**：快慢指针相遇后，将一个指针放回头节点，两者同速前进，再次相遇处即入环点。由 `a = c·圈长 - b` 可得：头到入环点的距离等于相遇点继续走到入环点的距离（模圈长）。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                p = head
                while p is not slow:
                    p = p.next
                    slow = slow.next
                return p
        return None
```

<br>

## 21. 合并两个有序链表 / Merge Two Sorted Lists [🔗](https://leetcode.com/problems/merge-two-sorted-lists/)

难度：Easy

将两个升序链表合并为一个新的升序链表并返回。新链表由拼接给定的两个链表的所有节点组成。

**示例 1：**

```text
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**示例 2：**

```text
输入：l1 = [], l2 = []
输出：[]
```

**示例 3：**

```text
输入：l1 = [], l2 = [0]
输出：[0]
```

**解法**：哨兵节点 + 每次接上较小的节点，最后接上剩余部分。

**复杂度**：时间 O(m + n)，空间 O(1)。

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = tail = ListNode()
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        tail.next = list1 or list2
        return dummy.next
```

<br>

## 2. 两数相加 / Add Two Numbers [🔗](https://leetcode.com/problems/add-two-numbers/)

难度：Medium

给你两个非空链表，表示两个非负整数。它们每位数字都按照逆序方式存储，并且每个节点只能存储一位数字。

请将两个数相加，并以相同形式返回一个表示和的链表。除了数字 0 之外，这两个数都不会以 0 开头。

**示例 1：**

```text
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807。
```

**示例 2：**

```text
输入：l1 = [0], l2 = [0]
输出：[0]
```

**示例 3：**

```text
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

**解法**：按位相加并维护进位，循环条件包含进位，以处理最高位的额外进位。

**复杂度**：时间 O(max(m, n))，空间 O(max(m, n))。

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = tail = ListNode()
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            carry, digit = divmod(carry, 10)
            tail.next = ListNode(digit)
            tail = tail.next
        return dummy.next
```

<br>

## 19. 删除链表的倒数第 N 个结点 / Remove Nth Node From End of List [🔗](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

难度：Medium

给你一个链表，删除链表的倒数第 `n` 个结点，并返回链表的头结点。

**示例 1：**

```text
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**示例 2：**

```text
输入：head = [1], n = 1
输出：[]
```

**示例 3：**

```text
输入：head = [1,2], n = 1
输出：[1]
```

**进阶：** 你能尝试使用一趟扫描实现吗？

**解法**：哨兵 + 前后指针。快指针先走 n 步，再同步走，慢指针停在待删节点的前驱。哨兵处理删除头节点的情况。

**复杂度**：时间 O(L)，空间 O(1)。

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast = slow = dummy
        for _ in range(n):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

<br>

## 24. 两两交换链表中的节点 / Swap Nodes in Pairs [🔗](https://leetcode.com/problems/swap-nodes-in-pairs/)

难度：Medium

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。

必须在不修改节点内部值的情况下完成，只能进行节点交换。

**示例 1：**

```text
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```

**示例 2：**

```text
输入：head = []
输出：[]
```

**示例 3：**

```text
输入：head = [1]
输出：[1]
```

**解法**：哨兵，`prev` 指向每对的前驱，每次修改三根指针。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        prev = dummy
        while prev.next and prev.next.next:
            a, b = prev.next, prev.next.next
            a.next = b.next
            b.next = a
            prev.next = b
            prev = a
        return dummy.next
```

<br>

## 25. K 个一组翻转链表 / Reverse Nodes in k-Group [🔗](https://leetcode.com/problems/reverse-nodes-in-k-group/)

难度：Hard

给你链表的头节点 `head`，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，最后剩余的节点保持原有顺序。

不能只是单纯地改变节点内部的值，而是需要实际进行节点交换。

**示例 1：**

```text
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
```

**示例 2：**

```text
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```

**进阶：** 你可以设计一个只用 O(1) 额外内存空间的算法解决此问题吗？

**解法**：先探测是否还有 k 个节点；有则反转这一段，把前驱接到新头，把旧头（现在的尾）作为下一段的前驱。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        group_prev = dummy
        while True:
            kth = group_prev
            for _ in range(k):
                kth = kth.next
                if not kth:
                    return dummy.next
            group_next = kth.next
            prev, cur = group_next, group_prev.next
            while cur is not group_next:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
            new_tail = group_prev.next
            group_prev.next = kth
            group_prev = new_tail
```

<br>

## 138. 随机链表的复制 / Copy List with Random Pointer [🔗](https://leetcode.com/problems/copy-list-with-random-pointer/)

难度：Medium

给你一个长度为 `n` 的链表，每个节点包含一个额外的随机指针 `random`，它可以指向链表中的任何节点或空节点。

构造这个链表的深拷贝：新链表由 `n` 个全新节点组成，每个新节点的 `val` 与对应原节点相同，`next` 和 `random` 也都指向新链表中的节点，使原链表和复制链表能表示相同的链表状态。复制链表中的指针不能指向原链表中的节点。

输入输出中，每个节点用 `[val, random_index]` 表示，`random_index` 是随机指针指向的节点下标（0 到 n-1），不指向任何节点时为 `null`。函数只接收原链表的头节点 `head`。

**示例 1：**

```text
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**示例 2：**

```text
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
```

**示例 3：**

```text
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```

**解法**：第一遍建立「原节点 → 新节点」的映射，第二遍连接 next 和 random。`None` 映射到 `None`。追问 O(1) 空间时，可把新节点穿插到原节点后面，再拆开。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def copyRandomList(self, head: "Optional[Node]") -> "Optional[Node]":
        mp = {None: None}
        cur = head
        while cur:
            mp[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur:
            mp[cur].next = mp[cur.next]
            mp[cur].random = mp[cur.random]
            cur = cur.next
        return mp[head]
```

<br>

## 148. 排序链表 / Sort List [🔗](https://leetcode.com/problems/sort-list/)

难度：Medium

给你链表的头结点 `head`，请将其按升序排列并返回排序后的链表。

**示例 1：**

```text
输入：head = [4,2,1,3]
输出：[1,2,3,4]
```

**示例 2：**

```text
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

**示例 3：**

```text
输入：head = []
输出：[]
```

**进阶：** 你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

**解法**：归并排序。快慢指针找中点并断开，递归排序两半后合并。`fast` 从 `head.next` 起步，保证两个节点时能正确切分。

**复杂度**：时间 O(n log n)，空间 O(log n)（递归栈）。

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        a, b = self.sortList(head), self.sortList(mid)
        dummy = tail = ListNode()
        while a and b:
            if a.val <= b.val:
                tail.next = a
                a = a.next
            else:
                tail.next = b
                b = b.next
            tail = tail.next
        tail.next = a or b
        return dummy.next
```

<br>

## 23. 合并 K 个升序链表 / Merge k Sorted Lists [🔗](https://leetcode.com/problems/merge-k-sorted-lists/)

难度：Hard

给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例 1：**

```text
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：把三个有序链表合并到一个有序链表中得到 1->1->2->3->4->4->5->6。
```

**示例 2：**

```text
输入：lists = []
输出：[]
```

**示例 3：**

```text
输入：lists = [[]]
输出：[]
```

**解法**：小顶堆保存每条链表的当前头节点。堆元素加入下标 `i` 作为第二关键字，避免值相同时比较 ListNode 报错。也可两两分治合并，复杂度相同。

**复杂度**：时间 O(N log k)，空间 O(k)。

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
        heapq.heapify(heap)
        dummy = tail = ListNode()
        while heap:
            _, i, node = heapq.heappop(heap)
            tail.next = node
            tail = node
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        return dummy.next
```

<br>

## 146. LRU 缓存 / LRU Cache [🔗](https://leetcode.com/problems/lru-cache/)

难度：Medium

请你设计并实现一个满足 LRU（最近最少使用）缓存约束的数据结构，实现 `LRUCache` 类：

- `LRUCache(int capacity)`：以正整数作为容量 `capacity` 初始化 LRU 缓存。
- `int get(int key)`：如果关键字 `key` 存在于缓存中，返回它的值，否则返回 `-1`。
- `void put(int key, int value)`：如果关键字已存在，变更它的值；如果不存在，插入该组 `key-value`。如果插入导致关键字数量超过 `capacity`，则应该逐出最久未使用的关键字。

函数 `get` 和 `put` 必须以 O(1) 的平均时间复杂度运行。

**示例 1：**

```text
输入：
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出：
[null, null, null, 1, null, -1, null, -1, 3, 4]
解释：
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作使关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作使关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

**解法**：哈希表 + 双向链表。哈希表定位节点，链表维护使用顺序（头部最新，尾部最旧），两个哨兵节点简化边界。面试中 `OrderedDict` 写法通常不够，要会手写。

**复杂度**：每次操作时间 O(1)，空间 O(capacity)。

```python
class ListNode:
    def __init__(self, key = 0, value = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        nxt = self.head.next
        self.head.next = node
        node.next = nxt
        nxt.prev = node
        node.prev = self.head

    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_front(node)
            return 
        new_node = ListNode(key, value)
        self._add_front(new_node)
        self.cache[key] = new_node
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]





# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

<br>
<br>

# 二叉树 Binary Tree

<br>

## 94. 二叉树的中序遍历 / Binary Tree Inorder Traversal [🔗](https://leetcode.com/problems/binary-tree-inorder-traversal/)

难度：Easy

给定一个二叉树的根节点 `root`，返回它的中序遍历结果。

**示例 1：**

```text
输入：root = [1,null,2,3]
输出：[1,3,2]
```

**示例 2：**

```text
输入：root = [1,2,3,4,5,null,8,null,null,6,7,9]
输出：[4,2,6,5,7,1,3,9,8]
```

**示例 3：**

```text
输入：root = []
输出：[]
```

**示例 4：**

```text
输入：root = [1]
输出：[1]
```

**进阶：** 递归算法很简单，你可以通过迭代算法完成吗？

**解法**：迭代写法：一路向左压栈，弹出访问后转向右子树。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res, stack = [], []
        cur = root
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
```

<br>

## 104. 二叉树的最大深度 / Maximum Depth of Binary Tree [🔗](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

难度：Easy

给定一个二叉树 `root`，返回其最大深度。

二叉树的最大深度是指从根节点到最远叶子节点的最长路径上的节点数。

**示例 1：**

```text
输入：root = [3,9,20,null,null,15,7]
输出：3
```

**示例 2：**

```text
输入：root = [1,null,2]
输出：2
```

**解法**：深度 = 1 + max(左子树深度, 右子树深度)。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

<br>

## 226. 翻转二叉树 / Invert Binary Tree [🔗](https://leetcode.com/problems/invert-binary-tree/)

难度：Easy

给你一棵二叉树的根节点 `root`，翻转这棵二叉树（交换每个节点的左右子树），并返回其根节点。

**示例 1：**

```text
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

**示例 2：**

```text
输入：root = [2,1,3]
输出：[2,3,1]
```

**示例 3：**

```text
输入：root = []
输出：[]
```

**解法**：递归交换每个节点的左右子树。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

<br>

## 101. 对称二叉树 / Symmetric Tree [🔗](https://leetcode.com/problems/symmetric-tree/)

难度：Easy

给你一个二叉树的根节点 `root`，检查它是否轴对称（即左右镜像）。

**示例 1：**

```text
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2：**

```text
输入：root = [1,2,2,null,3,null,3]
输出：false
```

**进阶：** 你可以运用递归和迭代两种方法解决这个问题吗？

**解法**：同时递归两棵子树：值相同，且 a 的左对 b 的右、a 的右对 b 的左。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def mirror(a, b) -> bool:
            if not a or not b:
                return a is b
            return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)

        return mirror(root.left, root.right)
```

<br>

## 543. 二叉树的直径 / Diameter of Binary Tree [🔗](https://leetcode.com/problems/diameter-of-binary-tree/)

难度：Easy

给你一棵二叉树的根节点，返回该树的直径。

二叉树的直径是指树中任意两个节点之间最长路径的长度。这条路径可能经过也可能不经过根节点 `root`。两节点之间路径的长度由它们之间的边数表示。

**示例 1：**

```text
输入：root = [1,2,3,4,5]
输出：3
解释：3 是路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
```

**示例 2：**

```text
输入：root = [1,2]
输出：1
```

**解法**：后序遍历求深度。经过某节点的最长路径是左深度 + 右深度，遍历时更新全局最大值。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        best = 0

        def depth(node) -> int:
            nonlocal best
            if not node:
                return 0
            l, r = depth(node.left), depth(node.right)
            best = max(best, l + r)
            return 1 + max(l, r)

        depth(root)
        return best
```

<br>

## 102. 二叉树的层序遍历 / Binary Tree Level Order Traversal [🔗](https://leetcode.com/problems/binary-tree-level-order-traversal/)

难度：Medium

给你二叉树的根节点 `root`，返回其节点值的层序遍历（即逐层地、从左到右访问所有节点）。

**示例 1：**

```text
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```

**示例 2：**

```text
输入：root = [1]
输出：[[1]]
```

**示例 3：**

```text
输入：root = []
输出：[]
```

**解法**：BFS，每轮处理当前层的全部节点。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        level = [root]
        while level:
            res.append([node.val for node in level])
            level = [c for node in level for c in (node.left, node.right) if c]
        return res
```

<br>

## 108. 将有序数组转换为二叉搜索树 / Convert Sorted Array to Binary Search Tree [🔗](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

难度：Easy

给你一个整数数组 `nums`，其中元素已经按升序排列，请你将其转换为一棵平衡二叉搜索树。

平衡二叉树是指每个节点的左右两个子树的高度差的绝对值不超过 1。

**示例 1：**

```text
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也是正确答案。
```

**示例 2：**

```text
输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

**解法**：取中点作根，递归构建左右子树。

**复杂度**：时间 O(n)，空间 O(log n)。

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(l: int, r: int):
            if l > r:
                return None
            mid = (l + r) // 2
            return TreeNode(nums[mid], build(l, mid - 1), build(mid + 1, r))

        return build(0, len(nums) - 1)
```

<br>

## 98. 验证二叉搜索树 / Validate Binary Search Tree [🔗](https://leetcode.com/problems/validate-binary-search-tree/)

难度：Medium

给你一个二叉树的根节点 `root`，判断其是否是一个有效的二叉搜索树。

有效二叉搜索树定义如下：

- 节点的左子树只包含严格小于当前节点的数。
- 节点的右子树只包含严格大于当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例 1：**

```text
输入：root = [2,1,3]
输出：true
```

**示例 2：**

```text
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5，但是右子节点的值是 4。
```

**解法**：递归时传入取值开区间 `(low, high)`。只比较父子节点是不够的，右子树的所有节点都要大于祖先。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def check(node, low, high) -> bool:
            if not node:
                return True
            if not low < node.val < high:
                return False
            return check(node.left, low, node.val) and check(node.right, node.val, high)

        return check(root, float("-inf"), float("inf"))
```

<br>

## 230. 二叉搜索树中第 K 小的元素 / Kth Smallest Element in a BST [🔗](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

难度：Medium

给定一个二叉搜索树的根节点 `root` 和一个整数 `k`，查找其中第 `k` 小的元素（`k` 从 1 开始计数）。

**示例 1：**

```text
输入：root = [3,1,4,null,2], k = 1
输出：1
```

**示例 2：**

```text
输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
```

**进阶：** 如果二叉搜索树经常被修改（插入 / 删除操作），并且你需要频繁地查找第 k 小的值，你将如何优化算法？

**解法**：BST 的中序遍历是升序，迭代中序到第 k 个即返回。

**复杂度**：时间 O(h + k)，空间 O(h)。

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        cur = root
        while True:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            k -= 1
            if k == 0:
                return cur.val
            cur = cur.right
```

<br>

## 199. 二叉树的右视图 / Binary Tree Right Side View [🔗](https://leetcode.com/problems/binary-tree-right-side-view/)

难度：Medium

给定一个二叉树的根节点 `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**示例 1：**

```text
输入：root = [1,2,3,null,5,null,4]
输出：[1,3,4]
```

**示例 2：**

```text
输入：root = [1,2,3,4,null,null,null,5]
输出：[1,3,4,5]
```

**示例 3：**

```text
输入：root = [1,null,3]
输出：[1,3]
```

**示例 4：**

```text
输入：root = []
输出：[]
```

**解法**：层序遍历，取每层最后一个节点。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        level = [root] if root else []
        while level:
            res.append(level[-1].val)
            level = [c for node in level for c in (node.left, node.right) if c]
        return res
```

<br>

## 114. 二叉树展开为链表 / Flatten Binary Tree to Linked List [🔗](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

难度：Medium

给你二叉树的根结点 `root`，请你将它原地展开为一个单链表：

- 展开后的单链表同样使用 `TreeNode`，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null`。
- 展开后的单链表与二叉树先序遍历的顺序相同。

**示例 1：**

```text
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```

**示例 2：**

```text
输入：root = []
输出：[]
```

**示例 3：**

```text
输入：root = [0]
输出：[0]
```

**进阶：** 你可以使用原地算法（O(1) 额外空间）展开这棵树吗？

**解法**：对每个有左子树的节点，找到左子树最右节点，把原右子树接上去，再把左子树移到右边。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        cur = root
        while cur:
            if cur.left:
                p = cur.left
                while p.right:
                    p = p.right
                p.right = cur.right
                cur.right = cur.left
                cur.left = None
            cur = cur.right
```

<br>

## 105. 从前序与中序遍历序列构造二叉树 / Construct Binary Tree from Preorder and Inorder Traversal [🔗](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

难度：Medium

给定两个整数数组 `preorder` 和 `inorder`，其中 `preorder` 是二叉树的先序遍历，`inorder` 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

题目保证树中的值互不相同。

**示例 1：**

```text
输入：preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出：[3,9,20,null,null,15,7]
```

**示例 2：**

```text
输入：preorder = [-1], inorder = [-1]
输出：[-1]
```

**解法**：前序首元素是根；在中序中定位根，左边就是左子树，其长度决定了前序的切分。用哈希表 O(1) 定位。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        pos = {v: i for i, v in enumerate(inorder)}

        def build(pre_l: int, in_l: int, size: int):
            if size == 0:
                return None
            root_val = preorder[pre_l]
            left_size = pos[root_val] - in_l
            left = build(pre_l + 1, in_l, left_size)
            right = build(pre_l + 1 + left_size, pos[root_val] + 1, size - 1 - left_size)
            return TreeNode(root_val, left, right)

        return build(0, 0, len(preorder))
```

<br>

## 437. 路径总和 III / Path Sum III [🔗](https://leetcode.com/problems/path-sum-iii/)

难度：Medium

给定一个二叉树的根节点 `root` 和一个整数 `targetSum`，求该二叉树里节点值之和等于 `targetSum` 的路径的数目。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

**示例 1：**

```text
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条：5 -> 3、5 -> 2 -> 1、-3 -> 11。
```

**示例 2：**

```text
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3
```

**解法**：树上前缀和，等同于第 560 题。DFS 时记录根到当前路径上各前缀和的出现次数，离开节点时撤销。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        cnt = defaultdict(int)
        cnt[0] = 1

        def dfs(node, pre: int) -> int:
            if not node:
                return 0
            pre += node.val
            res = cnt[pre - targetSum]
            cnt[pre] += 1
            res += dfs(node.left, pre) + dfs(node.right, pre)
            cnt[pre] -= 1
            return res

        return dfs(root, 0)
```

<br>

## 236. 二叉树的最近公共祖先 / Lowest Common Ancestor of a Binary Tree [🔗](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

难度：Medium

给定一个二叉树，找到该树中两个指定节点 `p`、`q` 的最近公共祖先。

最近公共祖先的定义：对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。

题目保证所有节点的值互不相同，且 `p`、`q` 均存在于给定的二叉树中。

**示例 1：**

```text
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3。
```

**示例 2：**

```text
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5，因为根据定义，最近公共祖先节点可以为节点本身。
```

**示例 3：**

```text
输入：root = [1,2], p = 1, q = 2
输出：1
```

**解法**：后序递归。遇到 p/q/空就返回；左右都找到说明当前节点是 LCA，否则返回非空的一侧。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def lowestCommonAncestor(self, root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
        if not root or root is p or root is q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left or right
```

<br>

## 124. 二叉树中的最大路径和 / Binary Tree Maximum Path Sum [🔗](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

难度：Hard

二叉树中的路径被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中至多出现一次。该路径至少包含一个节点，且不一定经过根节点。

路径和是路径中各节点值的总和。给你一个二叉树的根节点 `root`，返回其最大路径和。

**示例 1：**

```text
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3，路径和为 2 + 1 + 3 = 6。
```

**示例 2：**

```text
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7，路径和为 15 + 20 + 7 = 42。
```

**解法**：后序。`gain(node)` 返回向上贡献的单边最大值，负贡献按 0 算。以当前节点为拐点的路径和是 `val + 左 + 右`，用它更新答案。

**复杂度**：时间 O(n)，空间 O(h)。

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        best = float("-inf")

        def gain(node) -> int:
            nonlocal best
            if not node:
                return 0
            l = max(gain(node.left), 0)
            r = max(gain(node.right), 0)
            best = max(best, node.val + l + r)
            return node.val + max(l, r)

        gain(root)
        return best
```

<br>
<br>

# 图论 Graph

<br>

## 200. 岛屿数量 / Number of Islands [🔗](https://leetcode.com/problems/number-of-islands/)

难度：Medium

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。可以假设网格的四条边均被水包围。

**示例 1：**

```text
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```text
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

**解法**：遍历网格，遇到陆地就计数，并 DFS 把整座岛淹掉，防止重复计数。网格很大时递归可能超限，可改 BFS。

**复杂度**：时间 O(mn)，空间 O(mn)。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(i: int, j: int) -> None:
            if not (0 <= i < m and 0 <= j < n) or grid[i][j] != "1":
                return
            grid[i][j] = "0"
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    res += 1
                    dfs(i, j)
        return res
```

<br>

## 994. 腐烂的橘子 / Rotting Oranges [🔗](https://leetcode.com/problems/rotting-oranges/)

难度：Medium

在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，腐烂的橘子周围 4 个方向上相邻的新鲜橘子都会腐烂。返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`。

**示例 1：**

```text
输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
输出：4
```

**示例 2：**

```text
输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行，第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个方向上。
```

**示例 3：**

```text
输入：grid = [[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0。
```

**解法**：多源 BFS。所有腐烂橘子同时入队，按层扩散，层数即分钟数；最后检查是否还有新鲜橘子。

**复杂度**：时间 O(mn)，空间 O(mn)。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        fresh = 0
        q = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    q.append((i, j))
        minutes = 0
        while q and fresh:
            minutes += 1
            for _ in range(len(q)):
                i, j = q.popleft()
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                        grid[x][y] = 2
                        fresh -= 1
                        q.append((x, y))
        return -1 if fresh else minutes
```

<br>

## 207. 课程表 / Course Schedule [🔗](https://leetcode.com/problems/course-schedule/)

难度：Medium

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1`。

在选修某些课程之前需要一些先修课程。先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [a_i, b_i]` 表示如果要学习课程 `a_i`，则必须先学习课程 `b_i`。

请你判断是否可能完成所有课程的学习。如果可以，返回 `true`；否则，返回 `false`。

**示例 1：**

```text
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。这是可能的。
```

**示例 2：**

```text
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：学习课程 1 之前要先完成课程 0；学习课程 0 之前，又要先完成课程 1。这是不可能的。
```

**解法**：等价于判断有向图是否有环。Kahn 拓扑排序：入度为 0 的节点入队，弹出时减少后继入度，最后看处理过的节点数是否等于总数。

**复杂度**：时间 O(V + E)，空间 O(V + E)。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for a, b in prerequisites:
            graph[b].append(a)
            indeg[a] += 1
        q = deque(i for i in range(numCourses) if indeg[i] == 0)
        done = 0
        while q:
            u = q.popleft()
            done += 1
            for v in graph[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return done == numCourses
```

<br>

## 208. 实现 Trie（前缀树）/ Implement Trie (Prefix Tree) [🔗](https://leetcode.com/problems/implement-trie-prefix-tree/)

难度：Medium

Trie（发音类似 "try"，又称前缀树）是一种树形数据结构，用于高效地存储和检索字符串数据集中的键，常用于自动补全和拼写检查。

请你实现 `Trie` 类：

- `Trie()`：初始化前缀树对象。
- `void insert(String word)`：向前缀树中插入字符串 `word`。
- `boolean search(String word)`：如果字符串 `word` 在前缀树中（即之前已经插入），返回 `true`；否则，返回 `false`。
- `boolean startsWith(String prefix)`：如果之前已经插入的字符串中，有以 `prefix` 为前缀的，返回 `true`；否则，返回 `false`。

`word` 和 `prefix` 仅由小写英文字母组成。

**示例 1：**

```text
输入：
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出：
[null, null, true, false, true, null, true]
解释：
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

**解法**：每个节点存子节点字典和单词结束标记。`search` 要求结束标记为真，`startsWith` 只要路径存在。

**复杂度**：每次操作时间 O(L)，空间 O(总字符数)。

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def _find(self, s: str):
        node = self
        for c in s:
            node = node.children.get(c)
            if node is None:
                return None
        return node

    def insert(self, word: str) -> None:
        node = self
        for c in word:
            node = node.children.setdefault(c, Trie())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find(prefix) is not None
```

<br>
<br>

# 回溯 Backtracking

<br>

## 46. 全排列 / Permutations [🔗](https://leetcode.com/problems/permutations/)

难度：Medium

给定一个不含重复数字的数组 `nums`，返回其所有可能的全排列。你可以按任意顺序返回答案。

**示例 1：**

```text
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**示例 2：**

```text
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

**示例 3：**

```text
输入：nums = [1]
输出：[[1]]
```

**解法**：逐位交换：把 `[i, n)` 中的每个数换到位置 i，递归处理 i+1，再换回来。

**复杂度**：时间 O(n·n!)，空间 O(n)。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)

        def dfs(i: int) -> None:
            if i == n:
                res.append(nums[:])
                return
            for j in range(i, n):
                nums[i], nums[j] = nums[j], nums[i]
                dfs(i + 1)
                nums[i], nums[j] = nums[j], nums[i]

        dfs(0)
        return res
```

<br>

## 78. 子集 / Subsets [🔗](https://leetcode.com/problems/subsets/)

难度：Medium

给你一个整数数组 `nums`，数组中的元素互不相同。返回该数组所有可能的子集（幂集）。

解集不能包含重复的子集，可以按任意顺序返回解集。

**示例 1：**

```text
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```text
输入：nums = [0]
输出：[[],[0]]
```

**解法**：每层从 `start` 向后选，递归树的每个节点都是一个子集。

**复杂度**：时间 O(n·2ⁿ)，空间 O(n)。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res, path = [], []

        def dfs(start: int) -> None:
            res.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res
```

<br>

## 17. 电话号码的字母组合 / Letter Combinations of a Phone Number [🔗](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

难度：Medium

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。

数字到字母的映射与电话按键相同：2 对应 abc，3 对应 def，4 对应 ghi，5 对应 jkl，6 对应 mno，7 对应 pqrs，8 对应 tuv，9 对应 wxyz。注意 1 不对应任何字母。

**示例 1：**

```text
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**示例 2：**

```text
输入：digits = "2"
输出：["a","b","c"]
```

**解法**：每位数字枚举对应字母。输入为空时返回空列表。

**复杂度**：时间 O(n·4ⁿ)，空间 O(n)。

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        board = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
                 "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res, path = [], []

        def dfs(i: int) -> None:
            if i == len(digits):
                res.append("".join(path))
                return
            for c in board[digits[i]]:
                path.append(c)
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res
```

<br>

## 39. 组合总和 / Combination Sum [🔗](https://leetcode.com/problems/combination-sum/)

难度：Medium

给你一个无重复元素的整数数组 `candidates` 和一个目标整数 `target`，找出 `candidates` 中可以使数字和为 `target` 的所有不同组合，以列表形式返回。你可以按任意顺序返回这些组合。

`candidates` 中的同一个数字可以无限制重复被选取。如果至少一个数字的被选数量不同，则两种组合是不同的。题目保证和为 `target` 的不同组合数少于 150 个。

**示例 1：**

```text
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7。注意 2 可以使用多次。
7 也是一个候选，7 = 7。
仅有这两种组合。
```

**示例 2：**

```text
输入：candidates = [2,3,5], target = 8
输出：[[2,2,2,2],[2,3,3],[3,5]]
```

**示例 3：**

```text
输入：candidates = [2], target = 1
输出：[]
```

**解法**：排序后回溯，下一层从 `i` 开始以允许重复选，并通过只向后选避免重复组合。候选数超过剩余值时直接剪枝。

**复杂度**：时间取决于搜索树大小，空间 O(target / min)。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res, path = [], []

        def dfs(start: int, remain: int) -> None:
            if remain == 0:
                res.append(path[:])
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remain:
                    break
                path.append(candidates[i])
                dfs(i, remain - candidates[i])
                path.pop()

        dfs(0, target)
        return res
```

<br>

## 22. 括号生成 / Generate Parentheses [🔗](https://leetcode.com/problems/generate-parentheses/)

难度：Medium

数字 `n` 代表生成括号的对数，请你设计一个函数，生成所有可能的并且有效的括号组合。

**示例 1：**

```text
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

**示例 2：**

```text
输入：n = 1
输出：["()"]
```

**解法**：左括号数量小于 n 时可加左括号；右括号数量小于左括号时可加右括号。

**复杂度**：时间 O(4ⁿ/√n)，空间 O(n)。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res, path = [], []

        def dfs(left: int, right: int) -> None:
            if len(path) == 2 * n:
                res.append("".join(path))
                return
            if left < n:
                path.append("(")
                dfs(left + 1, right)
                path.pop()
            if right < left:
                path.append(")")
                dfs(left, right + 1)
                path.pop()

        dfs(0, 0)
        return res
```

<br>

## 79. 单词搜索 / Word Search [🔗](https://leetcode.com/problems/word-search/)

难度：Medium

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word`。如果 `word` 存在于网格中，返回 `true`；否则，返回 `false`。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中「相邻」单元格是水平或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例 1：**

```text
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

```text
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
```

**示例 3：**

```text
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

**进阶：** 你可以使用搜索剪枝的技术来优化解决方案，使其在 `board` 更大的情况下可以更快解决问题吗？

**解法**：从每个格子开始 DFS，临时把走过的格子改成 `#` 作访问标记，回溯时恢复。

**复杂度**：时间 O(mn·3ᴸ)，空间 O(L)。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])

        def dfs(i: int, j: int, k: int) -> bool:
            if not (0 <= i < m and 0 <= j < n) or board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            board[i][j] = "#"
            found = (dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1)
                     or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1))
            board[i][j] = word[k]
            return found

        return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

<br>

## 131. 分割回文串 / Palindrome Partitioning [🔗](https://leetcode.com/problems/palindrome-partitioning/)

难度：Medium

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文串。返回 `s` 所有可能的分割方案。

**示例 1：**

```text
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```text
输入：s = "a"
输出：[["a"]]
```

**解法**：枚举当前段的终点，是回文才继续递归。

**复杂度**：时间 O(n·2ⁿ)，空间 O(n)。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res, path = [], []
        n = len(s)

        def dfs(start: int) -> None:
            if start == n:
                res.append(path[:])
                return
            for end in range(start + 1, n + 1):
                sub = s[start:end]
                if sub == sub[::-1]:
                    path.append(sub)
                    dfs(end)
                    path.pop()

        dfs(0)
        return res
```

<br>

## 51. N 皇后 / N-Queens [🔗](https://leetcode.com/problems/n-queens/)

难度：Hard

按照国际象棋的规则，皇后可以攻击与之处在同一行、同一列或同一斜线上的棋子。

n 皇后问题研究的是如何将 `n` 个皇后放置在 `n × n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n`，返回所有不同的 n 皇后问题的解决方案。每一种解法包含一个不同的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表皇后和空位。

**示例 1：**

```text
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：4 皇后问题存在两个不同的解法。
```

**示例 2：**

```text
输入：n = 1
输出：[["Q"]]
```

**解法**：逐行放置，用三个集合记录被占的列和两条对角线：`r - c` 与 `r + c` 分别在对角线上恒定。

**复杂度**：时间 O(n!)，空间 O(n)。

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res, queens = [], []
        cols, diag, anti = set(), set(), set()

        def dfs(r: int) -> None:
            if r == n:
                res.append(["." * c + "Q" + "." * (n - c - 1) for c in queens])
                return
            for c in range(n):
                if c in cols or r - c in diag or r + c in anti:
                    continue
                cols.add(c)
                diag.add(r - c)
                anti.add(r + c)
                queens.append(c)
                dfs(r + 1)
                queens.pop()
                cols.remove(c)
                diag.remove(r - c)
                anti.remove(r + c)

        dfs(0)
        return res
```

<br>
<br>

# 二分查找 Binary Search

<br>

## 35. 搜索插入位置 / Search Insert Position [🔗](https://leetcode.com/problems/search-insert-position/)

难度：Easy

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

数组中的元素互不相同。请必须使用时间复杂度为 O(log n) 的算法。

**示例 1：**

```text
输入：nums = [1,3,5,6], target = 5
输出：2
```

**示例 2：**

```text
输入：nums = [1,3,5,6], target = 2
输出：1
```

**示例 3：**

```text
输入：nums = [1,3,5,6], target = 7
输出：4
```

**解法**：找第一个 `>= target` 的位置（lower bound）。循环结束时 `l` 就是答案。

**复杂度**：时间 O(log n)，空间 O(1)。

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)
        while l < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l
```

<br>

## 74. 搜索二维矩阵 / Search a 2D Matrix [🔗](https://leetcode.com/problems/search-a-2d-matrix/)

难度：Medium

给你一个满足下述两条属性的 `m x n` 整数矩阵：

- 每行中的整数从左到右按非严格递增顺序排列。
- 每行的第一个整数大于前一行的最后一个整数。

给你一个整数 `target`，如果 `target` 在矩阵中，返回 `true`；否则，返回 `false`。

**示例 1：**

```text
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
```

**示例 2：**

```text
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false
```

**解法**：把矩阵看成长度 `m*n` 的有序数组，用 `divmod(mid, n)` 映射回行列。

**复杂度**：时间 O(log mn)，空间 O(1)。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        l, r = 0, m * n - 1
        while l <= r:
            mid = (l + r) // 2
            i, j = divmod(mid, n)
            if matrix[i][j] == target:
                return True
            if matrix[i][j] < target:
                l = mid + 1
            else:
                r = mid - 1
        return False
```

<br>

## 34. 在排序数组中查找元素的第一个和最后一个位置 / Find First and Last Position of Element in Sorted Array [🔗](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

难度：Medium

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。你必须设计并实现时间复杂度为 O(log n) 的算法。

**示例 1：**

```text
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```text
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```text
输入：nums = [], target = 0
输出：[-1,-1]
```

**解法**：两次 lower bound：第一个 `>= target` 的位置是起点，第一个 `>= target + 1` 的位置减 1 是终点。

**复杂度**：时间 O(log n)，空间 O(1)。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def lower(x: int) -> int:
            l, r = 0, len(nums)
            while l < r:
                mid = (l + r) // 2
                if nums[mid] < x:
                    l = mid + 1
                else:
                    r = mid
            return l

        start = lower(target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        return [start, lower(target + 1) - 1]
```

<br>

## 33. 搜索旋转排序数组 / Search in Rotated Sorted Array [🔗](https://leetcode.com/problems/search-in-rotated-sorted-array/)

难度：Medium

整数数组 `nums` 按升序排列，数组中的值互不相同。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了旋转，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标从 0 开始计数）。例如，`[0,1,2,4,5,6,7]` 在下标 3 处经旋转后可能变为 `[4,5,6,7,0,1,2]`。

给你旋转后的数组 `nums` 和一个整数 `target`，如果 `nums` 中存在这个目标值，返回它的下标，否则返回 `-1`。你必须设计一个时间复杂度为 O(log n) 的算法。

**示例 1：**

```text
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

**示例 2：**

```text
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**示例 3：**

```text
输入：nums = [1], target = 0
输出：-1
```

**解法**：`mid` 两侧至少有一侧有序。判断哪侧有序，再看 target 是否落在该侧范围内。

**复杂度**：时间 O(log n)，空间 O(1)。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            if nums[l] <= nums[mid]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
```

<br>

## 153. 寻找旋转排序数组中的最小值 / Find Minimum in Rotated Sorted Array [🔗](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

难度：Medium

已知一个长度为 `n` 的数组，预先按照升序排列，经由 1 到 `n` 次旋转后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 4 次，则可以得到 `[4,5,6,7,0,1,2]`；
- 若旋转 7 次，则可以得到 `[0,1,2,4,5,6,7]`。

数组 `[a[0], a[1], ..., a[n-1]]` 旋转一次的结果为 `[a[n-1], a[0], a[1], ..., a[n-2]]`。

给你一个元素值互不相同的数组 `nums`，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的最小元素。你必须设计一个时间复杂度为 O(log n) 的算法。

**示例 1：**

```text
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5]，旋转 3 次得到输入数组。
```

**示例 2：**

```text
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7]，旋转 4 次得到输入数组。
```

**示例 3：**

```text
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17]，旋转 4 次得到输入数组。
```

**解法**：比较 `nums[mid]` 与 `nums[r]`：大于说明最小值在右半段（不含 mid），否则在左半段（含 mid）。

**复杂度**：时间 O(log n)，空间 O(1)。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        return nums[l]
```

<br>

## 4. 寻找两个正序数组的中位数 / Median of Two Sorted Arrays [🔗](https://leetcode.com/problems/median-of-two-sorted-arrays/)

难度：Hard

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的中位数。

算法的时间复杂度应该为 O(log (m+n))。

**示例 1：**

```text
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3]，中位数 2。
```

**示例 2：**

```text
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4]，中位数 (2 + 3) / 2 = 2.5。
```

**解法**：在较短数组上二分分割点 i，另一数组分割点 j 由总数一半确定。合法分割满足 `A[i-1] <= B[j]` 且 `B[j-1] <= A[i]`，此时中位数由分割线两侧的四个值决定。

**复杂度**：时间 O(log min(m, n))，空间 O(1)。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        a, b = nums1, nums2
        if len(a) > len(b):
            a, b = b, a
        m, n = len(a), len(b)
        half = (m + n + 1) // 2
        inf = float("inf")
        l, r = 0, m
        while l <= r:
            i = (l + r) // 2
            j = half - i
            a_left = a[i - 1] if i > 0 else -inf
            a_right = a[i] if i < m else inf
            b_left = b[j - 1] if j > 0 else -inf
            b_right = b[j] if j < n else inf
            if a_left > b_right:
                r = i - 1
            elif b_left > a_right:
                l = i + 1
            else:
                if (m + n) % 2:
                    return float(max(a_left, b_left))
                return (max(a_left, b_left) + min(a_right, b_right)) / 2
        return 0.0
```

<br>
<br>

# 栈 Stack

<br>

## 20. 有效的括号 / Valid Parentheses [🔗](https://leetcode.com/problems/valid-parentheses/)

难度：Easy

给定一个只包括 `'('`、`')'`、`'{'`、`'}'`、`'['`、`']'` 的字符串 `s`，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例 1：**

```text
输入：s = "()"
输出：true
```

**示例 2：**

```text
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```text
输入：s = "(]"
输出：false
```

**示例 4：**

```text
输入：s = "([])"
输出：true
```

**解法**：左括号入栈，右括号必须与栈顶匹配，最后栈应为空。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        pairs = {")": "(", "]": "[", "}": "{"}
        stack = []
        for c in s:
            if c in pairs:
                if not stack or stack.pop() != pairs[c]:
                    return False
            else:
                stack.append(c)
        return not stack
```

<br>

## 155. 最小栈 / Min Stack [🔗](https://leetcode.com/problems/min-stack/)

难度：Medium

设计一个支持 `push`、`pop`、`top` 操作，并能在常数时间内检索到最小元素的栈。

实现 `MinStack` 类：

- `MinStack()`：初始化堆栈对象。
- `void push(int val)`：将元素 `val` 推入堆栈。
- `void pop()`：删除堆栈顶部的元素。
- `int top()`：获取堆栈顶部的元素。
- `int getMin()`：获取堆栈中的最小元素。

每个函数的时间复杂度都必须是 O(1)。`pop`、`top` 和 `getMin` 总是在非空栈上调用。

**示例 1：**

```text
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
输出：
[null,null,null,null,-3,null,0,-2]
解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3
minStack.pop();
minStack.top();      --> 返回 0
minStack.getMin();   --> 返回 -2
```

**解法**：每个位置同时保存「值」和「到该位置为止的最小值」。

**复杂度**：每次操作时间 O(1)，空间 O(n)。

```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        cur_min = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, cur_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

<br>

## 394. 字符串解码 / Decode String [🔗](https://leetcode.com/problems/decode-string/)

难度：Medium

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为：`k[encoded_string]`，表示方括号内部的 `encoded_string` 正好重复 `k` 次。注意 `k` 保证为正整数。

你可以认为输入字符串总是有效的：没有额外的空格，方括号总是符合格式要求。此外，原始数据不包含数字，所有的数字只表示重复的次数 `k`，例如不会出现像 `3a` 或 `2[4]` 这样的输入。

**示例 1：**

```text
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

**示例 2：**

```text
输入：s = "3[a2[c]]"
输出："accaccacc"
```

**示例 3：**

```text
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

**示例 4：**

```text
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```

**解法**：遇到 `[` 把当前字符串和倍数压栈并重置，遇到 `]` 弹出并拼接。数字可能有多位。

**复杂度**：时间 O(输出长度)，空间 O(输出长度)。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        cur = ""
        num = 0
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == "[":
                stack.append((cur, num))
                cur, num = "", 0
            elif c == "]":
                prev, times = stack.pop()
                cur = prev + cur * times
            else:
                cur += c
        return cur
```

<br>

## 739. 每日温度 / Daily Temperatures [🔗](https://leetcode.com/problems/daily-temperatures/)

难度：Medium

给定一个整数数组 `temperatures`，表示每天的温度，返回一个数组 `answer`，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

**示例 1：**

```text
输入：temperatures = [73,74,75,71,69,72,76,73]
输出：[1,1,4,2,1,1,0,0]
```

**示例 2：**

```text
输入：temperatures = [30,40,50,60]
输出：[1,1,1,0]
```

**示例 3：**

```text
输入：temperatures = [30,60,90]
输出：[1,1,0]
```

**解法**：单调递减栈存下标。当前温度更高时，弹出栈顶并结算其等待天数。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < t:
                j = stack.pop()
                res[j] = i - j
            stack.append(i)
        return res
```

<br>

## 84. 柱状图中最大的矩形 / Largest Rectangle in Histogram [🔗](https://leetcode.com/problems/largest-rectangle-in-histogram/)

难度：Hard

给定 `n` 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

**示例 1：**

```text
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形由高度为 5 和 6 的两根柱子组成，面积为 10 个单位。
```

**示例 2：**

```text
输入：heights = [2,4]
输出：4
```

**解法**：单调递增栈。柱子被弹出时，当前下标是其右边界，新栈顶是其左边界。末尾加高度 0 的哨兵，保证所有柱子都被结算。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        best = 0
        for i, h in enumerate(heights + [0]):
            while stack[-1] != -1 and heights[stack[-1]] >= h:
                height = heights[stack.pop()]
                best = max(best, height * (i - stack[-1] - 1))
            stack.append(i)
        return best
```

<br>
<br>

# 堆 Heap

<br>

## 215. 数组中的第 K 个最大元素 / Kth Largest Element in an Array [🔗](https://leetcode.com/problems/kth-largest-element-in-an-array/)

难度：Medium

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。你必须设计并实现时间复杂度为 O(n) 的算法。

**示例 1：**

```text
输入：nums = [3,2,1,5,6,4], k = 2
输出：5
```

**示例 2：**

```text
输入：nums = [3,2,3,1,2,4,5,5,6], k = 4
输出：4
```

**解法**：快速选择，随机选 pivot 做三路划分，只进入答案所在的一侧。三路划分能避免大量重复值导致的退化。也可维护大小为 k 的小顶堆，时间 O(n log k)。

**复杂度**：平均时间 O(n)，空间 O(n)（此写法用了额外数组）。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        pivot = random.choice(nums)
        big = [x for x in nums if x > pivot]
        equal = [x for x in nums if x == pivot]
        small = [x for x in nums if x < pivot]
        if k <= len(big):
            return self.findKthLargest(big, k)
        if k <= len(big) + len(equal):
            return pivot
        return self.findKthLargest(small, k - len(big) - len(equal))
```

<br>

## 347. 前 K 个高频元素 / Top K Frequent Elements [🔗](https://leetcode.com/problems/top-k-frequent-elements/)

难度：Medium

给你一个整数数组 `nums` 和一个整数 `k`，请你返回其中出现频率前 `k` 高的元素。你可以按任意顺序返回答案。

题目保证答案唯一，也就是说前 `k` 个高频元素的集合是唯一的。

**示例 1：**

```text
输入：nums = [1,1,1,2,2,3], k = 2
输出：[1,2]
```

**示例 2：**

```text
输入：nums = [1], k = 1
输出：[1]
```

**进阶：** 你所设计算法的时间复杂度必须优于 O(n log n)，其中 n 是数组大小。

**解法**：桶排序：下标是频率，桶里放该频率的数字，从高频桶往低频收集。也可用大小为 k 的小顶堆，时间 O(n log k)。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        buckets = [[] for _ in range(len(nums) + 1)]
        for x, c in count.items():
            buckets[c].append(x)
        res = []
        for c in range(len(nums), 0, -1):
            for x in buckets[c]:
                res.append(x)
                if len(res) == k:
                    return res
        return res
```

<br>

## 295. 数据流的中位数 / Find Median from Data Stream [🔗](https://leetcode.com/problems/find-median-from-data-stream/)

难度：Hard

中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。例如 `[2,3,4]` 的中位数是 3，`[2,3]` 的中位数是 (2 + 3) / 2 = 2.5。

实现 `MedianFinder` 类：

- `MedianFinder()`：初始化 `MedianFinder` 对象。
- `void addNum(int num)`：将数据流中的整数 `num` 添加到数据结构中。
- `double findMedian()`：返回到目前为止所有元素的中位数。与实际答案相差 10⁻⁵ 以内的答案将被接受。

**示例 1：**

```text
输入：
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出：
[null, null, null, 1.5, null, 2.0]
解释：
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr = [1, 2, 3]
medianFinder.findMedian(); // 返回 2.0
```

**进阶：** 如果数据流中所有整数都在 0 到 100 范围内，你将如何优化你的算法？如果数据流中 99% 的整数都在 0 到 100 范围内，你将如何优化你的算法？

**解法**：对顶堆。大顶堆 `small` 存较小一半（Python 中用取负模拟），小顶堆 `large` 存较大一半，保持 `len(small) - len(large)` 为 0 或 1。新数先经过 `large` 过滤，再平衡。

**复杂度**：`addNum` O(log n)，`findMedian` O(1)，空间 O(n)。

```python
class MedianFinder:
    def __init__(self):
        self.small = []
        self.large = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2
```

<br>
<br>

# 贪心 Greedy

<br>

## 121. 买卖股票的最佳时机 / Best Time to Buy and Sell Stock [🔗](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

难度：Easy

给定一个数组 `prices`，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出。设计一个算法来计算你所能获取的最大利润。如果你不能获取任何利润，返回 `0`。

**示例 1：**

```text
输入：prices = [7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）买入，在第 5 天（股票价格 = 6）卖出，最大利润 = 6 - 1 = 5。
注意利润不能是 7 - 1 = 6，因为卖出价格需要大于买入价格，同时你不能在买入前卖出股票。
```

**示例 2：**

```text
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下，没有交易完成，所以最大利润为 0。
```

**解法**：遍历时维护历史最低价，用当天价格减去它更新答案。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        low = prices[0]
        best = 0
        for p in prices:
            low = min(low, p)
            best = max(best, p - low)
        return best
```

<br>

## 55. 跳跃游戏 / Jump Game [🔗](https://leetcode.com/problems/jump-game/)

难度：Medium

给你一个非负整数数组 `nums`，你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。如果可以，返回 `true`；否则，返回 `false`。

**示例 1：**

```text
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1，然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```text
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该位置的最大跳跃长度是 0，所以永远不可能到达最后一个下标。
```

**解法**：维护最远可达位置 `far`，一旦 `i > far` 就到不了。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        far = 0
        for i, x in enumerate(nums):
            if i > far:
                return False
            far = max(far, i + x)
        return True
```

<br>

## 45. 跳跃游戏 II / Jump Game II [🔗](https://leetcode.com/problems/jump-game-ii/)

难度：Medium

给定一个长度为 `n` 的整数数组 `nums`，初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向后跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处，其中 `0 <= j <= nums[i]` 且 `i + j < n`。

返回到达 `nums[n - 1]` 的最小跳跃次数。题目保证可以到达 `nums[n - 1]`。

**示例 1：**

```text
输入：nums = [2,3,1,1,4]
输出：2
解释：跳到最后一个位置的最小跳跃数是 2。从下标 0 跳到下标 1，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**示例 2：**

```text
输入：nums = [2,3,0,1,4]
输出：2
```

**解法**：隐式 BFS。`end` 是当前步数能到的边界，`far` 是下一步最远能到的位置，走到 `end` 时必须再跳一次。循环到 `n-2` 为止，避免在终点多算一跳。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = end = far = 0
        for i in range(len(nums) - 1):
            far = max(far, i + nums[i])
            if i == end:
                jumps += 1
                end = far
        return jumps
```

<br>

## 763. 划分字母区间 / Partition Labels [🔗](https://leetcode.com/problems/partition-labels/)

难度：Medium

给你一个字符串 `s`。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s`。返回一个表示每个字符串片段长度的列表。

**示例 1：**

```text
输入：s = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca"、"defegde"、"hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde"、"hijhklij" 这样的划分是错误的，因为划分的片段数较少。
```

**示例 2：**

```text
输入：s = "eccbbbbdec"
输出：[10]
```

**解法**：记录每个字母最后出现的位置。遍历时扩展当前片段右边界，走到边界就切一段。

**复杂度**：时间 O(n)，空间 O(26)。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {c: i for i, c in enumerate(s)}
        res = []
        start = end = 0
        for i, c in enumerate(s):
            end = max(end, last[c])
            if i == end:
                res.append(end - start + 1)
                start = i + 1
        return res
```

<br>
<br>

# 动态规划 DP

<br>

## 70. 爬楼梯 / Climbing Stairs [🔗](https://leetcode.com/problems/climbing-stairs/)

难度：Easy

假设你正在爬楼梯，需要 `n` 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**示例 1：**

```text
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

**示例 2：**

```text
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

**解法**：`f(n) = f(n-1) + f(n-2)`，滚动变量即可。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
```

<br>

## 118. 杨辉三角 / Pascal's Triangle [🔗](https://leetcode.com/problems/pascals-triangle/)

难度：Easy

给定一个非负整数 `numRows`，生成杨辉三角的前 `numRows` 行。

在杨辉三角中，每个数是它左上方和右上方的数的和。

**示例 1：**

```text
输入：numRows = 5
输出：[[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

**示例 2：**

```text
输入：numRows = 1
输出：[[1]]
```

**解法**：每行首尾为 1，中间元素等于上一行相邻两数之和。

**复杂度**：时间 O(n²)，空间 O(n²)。

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for _ in range(numRows - 1):
            prev = res[-1]
            res.append([1] + [prev[i] + prev[i + 1] for i in range(len(prev) - 1)] + [1])
        return res
```

<br>

## 198. 打家劫舍 / House Robber [🔗](https://leetcode.com/problems/house-robber/)

难度：Medium

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统：如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，一夜之内能够偷窃到的最高金额。

**示例 1：**

```text
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。偷窃到的最高金额 = 1 + 3 = 4。
```

**示例 2：**

```text
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋（金额 = 2）、3 号房屋（金额 = 9），接着偷窃 5 号房屋（金额 = 1）。偷窃到的最高金额 = 2 + 9 + 1 = 12。
```

**解法**：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`，只需保留前两个状态。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        prev, cur = 0, 0
        for x in nums:
            prev, cur = cur, max(cur, prev + x)
        return cur
```

<br>

## 279. 完全平方数 / Perfect Squares [🔗](https://leetcode.com/problems/perfect-squares/)

难度：Medium

给你一个整数 `n`，返回和为 `n` 的完全平方数的最少数量。

完全平方数是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

**示例 1：**

```text
输入：n = 12
输出：3
解释：12 = 4 + 4 + 4
```

**示例 2：**

```text
输入：n = 13
输出：2
解释：13 = 4 + 9
```

**解法**：完全背包。`dp[i] = min(dp[i - j²]) + 1`。

**复杂度**：时间 O(n√n)，空间 O(n)。

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0] + [n + 1] * n
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        return dp[n]
```

<br>

## 322. 零钱兑换 / Coin Change [🔗](https://leetcode.com/problems/coin-change/)

难度：Medium

给你一个整数数组 `coins`，表示不同面额的硬币；以及一个整数 `amount`，表示总金额。

计算并返回可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。你可以认为每种硬币的数量是无限的。

**示例 1：**

```text
输入：coins = [1, 2, 5], amount = 11
输出：3
解释：11 = 5 + 5 + 1
```

**示例 2：**

```text
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```text
输入：coins = [1], amount = 0
输出：0
```

**解法**：完全背包。`dp[i] = min(dp[i - coin] + 1)`，初值 `amount + 1` 表示不可达。

**复杂度**：时间 O(amount·k)，空间 O(amount)。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0] + [amount + 1] * amount
        for i in range(1, amount + 1):
            for c in coins:
                if c <= i:
                    dp[i] = min(dp[i], dp[i - c] + 1)
        return -1 if dp[amount] > amount else dp[amount]
```

<br>

## 139. 单词拆分 / Word Break [🔗](https://leetcode.com/problems/word-break/)

难度：Medium

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s`，返回 `true`。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

**示例 1：**

```text
输入：s = "leetcode", wordDict = ["leet", "code"]
输出：true
解释：返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

**示例 2：**

```text
输入：s = "applepenapple", wordDict = ["apple", "pen"]
输出：true
解释：返回 true 因为 "applepenapple" 可以由 "apple"、"pen"、"apple" 拼接成。注意，你可以重复使用字典中的单词。
```

**示例 3：**

```text
输入：s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出：false
```

**解法**：`dp[i]` 表示前 i 个字符能否拆分。枚举最后一个词的起点 j，长度不超过字典最长词以剪枝。

**复杂度**：时间 O(nL²)，空间 O(n + 字典大小)，L 为最长单词长度。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        max_len = max(map(len, words))
        n = len(s)
        dp = [True] + [False] * n
        for i in range(1, n + 1):
            for j in range(i - 1, max(i - max_len, 0) - 1, -1):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break
        return dp[n]
```

<br>

## 300. 最长递增子序列 / Longest Increasing Subsequence [🔗](https://leetcode.com/problems/longest-increasing-subsequence/)

难度：Medium

给你一个整数数组 `nums`，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**

```text
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4。
```

**示例 2：**

```text
输入：nums = [0,1,0,3,2,3]
输出：4
```

**示例 3：**

```text
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

**进阶：** 你能将算法的时间复杂度降低到 O(n log n) 吗？

**解法**：贪心 + 二分。`tails[i]` 是长度为 i+1 的递增子序列的最小末尾。用 `bisect_left` 替换第一个 `>= x` 的值，或追加到末尾。`tails` 本身不一定是真实子序列。

**复杂度**：时间 O(n log n)，空间 O(n)。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = []
        for x in nums:
            i = bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)
```

<br>

## 152. 乘积最大子数组 / Maximum Product Subarray [🔗](https://leetcode.com/problems/maximum-product-subarray/)

难度：Medium

给你一个整数数组 `nums`，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 32 位整数。

**示例 1：**

```text
输入：nums = [2,3,-2,4]
输出：6
解释：子数组 [2,3] 有最大乘积 6。
```

**示例 2：**

```text
输入：nums = [-2,0,-1]
输出：0
解释：结果不能为 2，因为 [-2,-1] 不是子数组。
```

**解法**：同时维护以当前元素结尾的最大和最小乘积，因为负数乘最小值可能变成最大值。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        hi = lo = best = nums[0]
        for x in nums[1:]:
            candidates = (x, hi * x, lo * x)
            hi, lo = max(candidates), min(candidates)
            best = max(best, hi)
        return best
```

<br>

## 416. 分割等和子集 / Partition Equal Subset Sum [🔗](https://leetcode.com/problems/partition-equal-subset-sum/)

难度：Medium

给你一个只包含正整数的非空数组 `nums`。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**示例 1：**

```text
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11]。
```

**示例 2：**

```text
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

**解法**：总和为奇数直接否。否则是 0-1 背包：能否选出和为 `sum/2` 的子集。容量倒序遍历，保证每个数只用一次。

**复杂度**：时间 O(n·sum)，空间 O(sum)。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        target = total // 2
        dp = [True] + [False] * target
        for x in nums:
            for j in range(target, x - 1, -1):
                dp[j] = dp[j] or dp[j - x]
        return dp[target]
```

<br>

## 32. 最长有效括号 / Longest Valid Parentheses [🔗](https://leetcode.com/problems/longest-valid-parentheses/)

难度：Hard

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例 1：**

```text
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"。
```

**示例 2：**

```text
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"。
```

**示例 3：**

```text
输入：s = ""
输出：0
```

**解法**：栈存下标，栈底保存「最后一个未匹配的右括号」作为基准。匹配后长度为 `i - 栈顶`。也可用 DP：`dp[i]` 表示以 i 结尾的最长有效长度。

**复杂度**：时间 O(n)，空间 O(n)。

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        best = 0
        for i, c in enumerate(s):
            if c == "(":
                stack.append(i)
            else:
                stack.pop()
                if stack:
                    best = max(best, i - stack[-1])
                else:
                    stack.append(i)
        return best
```

<br>
<br>

# 多维动态规划 Multi-dimensional DP

<br>

## 62. 不同路径 / Unique Paths [🔗](https://leetcode.com/problems/unique-paths/)

难度：Medium

一个机器人位于一个 `m x n` 网格的左上角。机器人每次只能向下或者向右移动一步，它试图达到网格的右下角。

问总共有多少条不同的路径？

**示例 1：**

```text
输入：m = 3, n = 7
输出：28
```

**示例 2：**

```text
输入：m = 3, n = 2
输出：3
解释：从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```

**示例 3：**

```text
输入：m = 7, n = 3
输出：28
```

**示例 4：**

```text
输入：m = 3, n = 3
输出：6
```

**解法**：`dp[j] += dp[j-1]`，一维滚动。也可直接算组合数 `C(m+n-2, m-1)`。

**复杂度**：时间 O(mn)，空间 O(n)。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

<br>

## 64. 最小路径和 / Minimum Path Sum [🔗](https://leetcode.com/problems/minimum-path-sum/)

难度：Medium

给定一个包含非负整数的 `m x n` 网格 `grid`，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

**示例 1：**

```text
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```text
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

**解法**：`dp[i][j] = grid[i][j] + min(上, 左)`，一维滚动。

**复杂度**：时间 O(mn)，空间 O(n)。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n = len(grid[0])
        dp = [float("inf")] * n
        dp[0] = 0
        for row in grid:
            dp[0] += row[0]
            for j in range(1, n):
                dp[j] = min(dp[j], dp[j - 1]) + row[j]
        return dp[-1]
```

<br>

## 5. 最长回文子串 / Longest Palindromic Substring [🔗](https://leetcode.com/problems/longest-palindromic-substring/)

难度：Medium

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

**示例 1：**

```text
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**示例 2：**

```text
输入：s = "cbbd"
输出："bb"
```

**解法**：中心扩展，枚举奇数和偶数两种中心，比区间 DP 更省空间。区间 DP 的转移为 `dp[i][j] = s[i]==s[j] and dp[i+1][j-1]`。

**复杂度**：时间 O(n²)，空间 O(1)。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        start, best = 0, 0

        def expand(l: int, r: int) -> None:
            nonlocal start, best
            while l >= 0 and r < n and s[l] == s[r]:
                l -= 1
                r += 1
            if r - l - 1 > best:
                best = r - l - 1
                start = l + 1

        for i in range(n):
            expand(i, i)
            expand(i, i + 1)
        return s[start:start + best]
```

<br>

## 1143. 最长公共子序列 / Longest Common Subsequence [🔗](https://leetcode.com/problems/longest-common-subsequence/)

难度：Medium

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长公共子序列的长度。如果不存在公共子序列，返回 `0`。

一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是。两个字符串的公共子序列是这两个字符串所共同拥有的子序列。

**示例 1：**

```text
输入：text1 = "abcde", text2 = "ace"
输出：3
解释：最长公共子序列是 "ace"，它的长度为 3。
```

**示例 2：**

```text
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。
```

**示例 3：**

```text
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。
```

**解法**：`dp[i][j]` 是前 i、j 个字符的 LCS。字符相等取 `dp[i-1][j-1] + 1`，否则取 `max(dp[i-1][j], dp[i][j-1])`。

**复杂度**：时间 O(mn)，空间 O(mn)。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
```

<br>

## 72. 编辑距离 / Edit Distance [🔗](https://leetcode.com/problems/edit-distance/)

难度：Medium

给你两个单词 `word1` 和 `word2`，请返回将 `word1` 转换成 `word2` 所使用的最少操作数。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

**示例 1：**

```text
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```text
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

**解法**：`dp[i][j]` 是前 i、j 个字符的编辑距离。字符相等时继承 `dp[i-1][j-1]`，否则取删除、插入、替换三者的最小值加 1。

**复杂度**：时间 O(mn)，空间 O(mn)。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]
```

<br>
<br>

# 技巧 Tricks

<br>

## 136. 只出现一次的数字 / Single Number [🔗](https://leetcode.com/problems/single-number/)

难度：Easy

给你一个非空整数数组 `nums`，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

**示例 1：**

```text
输入：nums = [2,2,1]
输出：1
```

**示例 2：**

```text
输入：nums = [4,1,2,1,2]
输出：4
```

**示例 3：**

```text
输入：nums = [1]
输出：1
```

**解法**：异或：`x ^ x = 0`，`x ^ 0 = x`，全部异或后只剩单独的数。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for x in nums:
            res ^= x
        return res
```

<br>

## 169. 多数元素 / Majority Element [🔗](https://leetcode.com/problems/majority-element/)

难度：Easy

给定一个大小为 `n` 的数组 `nums`，返回其中的多数元素。多数元素是指在数组中出现次数大于 `⌊n/2⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**示例 1：**

```text
输入：nums = [3,2,3]
输出：3
```

**示例 2：**

```text
输入：nums = [2,2,1,1,1,2,2]
输出：2
```

**进阶：** 尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。

**解法**：Boyer-Moore 投票：计数为 0 时换候选，相同加一，不同减一。多数元素的票数抵消不完。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate, count = None, 0
        for x in nums:
            if count == 0:
                candidate = x
            count += 1 if x == candidate else -1
        return candidate
```

<br>

## 75. 颜色分类 / Sort Colors [🔗](https://leetcode.com/problems/sort-colors/)

难度：Medium

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums`，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、`1` 和 `2` 分别表示红色、白色和蓝色。必须在不使用库内置 sort 函数的情况下解决这个问题。

**示例 1：**

```text
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

**示例 2：**

```text
输入：nums = [2,0,1]
输出：[0,1,2]
```

**进阶：** 你能想出一个仅使用常数空间的一趟扫描算法吗？

**解法**：荷兰国旗三路划分。`[0, l)` 全 0，`[l, i)` 全 1，`(r, n)` 全 2。与右侧交换后 `i` 不前进，因为换来的值还没检查。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        l, i, r = 0, 0, len(nums) - 1
        while i <= r:
            if nums[i] == 0:
                nums[l], nums[i] = nums[i], nums[l]
                l += 1
                i += 1
            elif nums[i] == 2:
                nums[r], nums[i] = nums[i], nums[r]
                r -= 1
            else:
                i += 1
```

<br>

## 31. 下一个排列 / Next Permutation [🔗](https://leetcode.com/problems/next-permutation/)

难度：Medium

整数数组的一个排列就是将其所有成员以序列或线性顺序排列。例如，`arr = [1,2,3]`，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]`。

整数数组的下一个排列是指其整数的下一个字典序更大的排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即其元素按升序排列）。例如：

- `arr = [1,2,3]` 的下一个排列是 `[1,3,2]`。
- `arr = [2,3,1]` 的下一个排列是 `[3,1,2]`。
- `arr = [3,2,1]` 的下一个排列是 `[1,2,3]`，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums`，找出 `nums` 的下一个排列。必须原地修改，只允许使用额外常数空间。

**示例 1：**

```text
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```text
输入：nums = [3,2,1]
输出：[1,2,3]
```

**示例 3：**

```text
输入：nums = [1,1,5]
输出：[1,5,1]
```

**解法**：从右找第一个 `nums[i] < nums[i+1]`；再从右找第一个大于 `nums[i]` 的数并交换；最后反转 `i` 右侧的降序段。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        l, r = i + 1, len(nums) - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
```

<br>

## 287. 寻找重复数 / Find the Duplicate Number [🔗](https://leetcode.com/problems/find-the-duplicate-number/)

难度：Medium

给定一个包含 `n + 1` 个整数的数组 `nums`，其数字都在 `[1, n]` 范围内（包括 1 和 n），可知至少存在一个重复的整数。

假设 `nums` 只有一个重复的整数，返回这个重复的数。你设计的解决方案必须不修改数组 `nums` 且只用常量级 O(1) 的额外空间。

**示例 1：**

```text
输入：nums = [1,3,4,2,2]
输出：2
```

**示例 2：**

```text
输入：nums = [3,1,3,4,2]
输出：3
```

**示例 3：**

```text
输入：nums = [3,3,3,3,3]
输出：3
```

**进阶：** 如何证明 `nums` 中至少存在一个重复的数字？你可以设计一个线性级时间复杂度 O(n) 的解决方案吗？

**解法**：把 `i → nums[i]` 看成链表。重复值使两个下标指向同一节点，形成环，入环点就是答案。做法与第 142 题相同。

**复杂度**：时间 O(n)，空间 O(1)。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        p = 0
        while p != slow:
            p = nums[p]
            slow = nums[slow]
        return p
```
