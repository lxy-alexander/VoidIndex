---
title: LeetCode 数据结构设计题合集
description: LeetCode 设计题按 9 类整理（缓存、基础结构、随机化、迭代器、数据流、区间查询、字典树、业务模拟、多线程），每题给出中英文链接、思路和带编号注释的 Python 代码，写法尽量参考灵神（灵茶山艾府）的题解。
pubDatetime: 2026-11-14T08:17:00Z
tags:
  - leetcode
  - design
draft: true

---

设计题（Design）让你实现一个类，在里面写好几个方法，比如 LRU 缓存、推特、地铁系统。下面按类型整理，🔒 表示会员题。每道题都给出 CN（leetcode.cn）和 EN（leetcode.com）两个链接；代码统一用 Python，写法尽量参考灵神（灵茶山艾府）的题解，代码注释按 1、2、3… 标出步骤。

## 目录

| 分类 | 识别信号（题目里出现这些词） | 核心结构 |
|---|---|---|
| [1. 缓存类](#1-缓存类) | 最近使用、淘汰、O(1) 调整顺序 | 哈希表 + 双向链表 |
| [2. 基础数据结构实现](#2-基础数据结构实现) | 栈、队列、循环、哈希、最小值、最大频率 | 辅助栈 / 数组 + 指针 / 分桶 |
| [3. 随机化 / O(1) 操作](#3-随机化--o1-操作) | 等概率、随机返回、权重、洗牌 | 数组 + 哈希表存下标 / 蓄水池 / 前缀和 + 二分 |
| [4. 迭代器](#4-迭代器) | next / hasNext / peek | 栈 / 指针 / 缓存下一个元素 |
| [5. 数据流与统计](#5-数据流与统计) | 数据流、中位数、第 K 大、最近 t 毫秒、跨度 | 堆 / 队列 / 单调栈 / 有序集合 |
| [6. 区间与范围查询](#6-区间与范围查询) | 区间求和、单点修改、预订不重叠、时间戳、快照 | 前缀和 / 树状数组 / 线段树 / 二分 |
| [7. 字符串 / 字典树](#7-字符串--字典树) | 前缀、后缀、通配符 `.`、自动补全、字符流、路径 | Trie + 节点附加信息 |
| [8. 业务系统模拟](#8-业务系统模拟) | 用户、账户、车站、余额、评分、租借 | 多个哈希表 + 堆 / 有序集合 |
| [9. 多线程 / 并发](#9-多线程--并发) | 按序、交替、凑齐再放行、死锁 | Lock / Event / Semaphore / Condition |

另有 [附录](#附录)：开放式系统设计、北美高频题、安防场景变形题、按优先级刷题。

<br>

## 设计题：五步法

设计题考的不是新算法，而是“给每个方法挑对数据结构”。每道题都按下面五步走，先在纸上写完前三步再敲代码。

1. 列接口和频率：把每个方法写成一行，标出它的调用频率和数据规模（例如 `get` 10^5 次、`put` 10^5 次）。调用最多的方法决定结构。
2. 定状态：问自己“回答每个查询，最少要记住什么？”写成字段，例如 `food → (rating, cuisine)`、`cuisine → 堆`。
3. 给每个方法配结构：按查询方式选——按 key 找用哈希表；取最大/最小用堆；要有序、要删任意元素用 SortedList；要按最近顺序调整用双向链表；按前缀找用 Trie；区间求和用前缀和或树状数组。
4. 逐个方法写，先写改状态的方法：先写 `add / update / remove`，保证状态始终一致；再写查询。多个结构互相引用时，每次修改都要同步更新所有结构。
5. 算复杂度并自查边界：每个方法写出时间复杂度；检查空结构、重复 key、同值 tie-break、更新成原值这几种情况。

### 设计题常用套路

- **要 O(1) 的增删查** → 哈希表 + 双向链表（146、460、432）
- **要 O(1) 随机取** → 数组 + 哈希表存下标（380、381）
- **要随时取最大 / 最小 / 中位数** → 堆，或两个堆（295、355）
- **要按排序维护、还要能删除** → 有序集合（`SortedList`），或堆 + 懒删除（2034、2349、2353）
- **按时间或版本查询** → 每个键存一个列表，查询时二分（981、1146）
- **区间增删合并** → 有序集合 / 线段树（715、729-732）
- **立即维护 vs 延迟维护**：结构支持 O(log n) 删除任意元素（`SortedList`）就立即删；不支持（`heapq`）就用懒删除，查询时再清理。
- **全局时间**：凡是“最近”“最新”“历史版本”，都给每个事件一个自增时间戳，后面排序、二分、合并都靠它。

**面试高频建议优先刷**：146、155、380、295、208、355、981、1396、460、173、225/232、703、2296。

LeetCode 的 Python 环境预装了 `sortedcontainers`，下文的 `SortedList` 都可以直接 `from sortedcontainers import SortedList`。

<br>
<br>

## 1. 缓存类

哈希表负责 O(1) 找到节点，双向链表负责 O(1) 把节点挖出来、插到头尾。用一个哨兵节点 `dummy` 同时当头和尾（循环链表），能省掉所有空指针判断。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 146 | LRU 缓存 | [CN](https://leetcode.cn/problems/lru-cache/) · [EN](https://leetcode.com/problems/lru-cache/) | 哈希表 + 双向链表 |
| 460 | LFU 缓存 | [CN](https://leetcode.cn/problems/lfu-cache/) · [EN](https://leetcode.com/problems/lfu-cache/) | 哈希表 + 频次 → 双向链表 |
| 432 | 全 O(1) 的数据结构 | [CN](https://leetcode.cn/problems/all-oone-data-structure/) · [EN](https://leetcode.com/problems/all-oone-data-structure/) | 哈希表 + 双向链表（按计数分桶） |

<br>

### 146. LRU 缓存 / LRU Cache

[CN](https://leetcode.cn/problems/lru-cache/) · [EN](https://leetcode.com/problems/lru-cache/) · Medium

**题意**：实现容量为 `capacity` 的缓存，`get(key)` 返回值（不存在返回 -1），`put(key, value)` 插入或更新；超出容量时淘汰最久未使用的 key。两个操作都要 O(1)。

**思路**：把缓存想成一摞书。`get` 和 `put` 都会把这本书抽出来放到最上面；放不下时扔掉最下面那本。“抽出来、放最上面、扔最下面”用双向链表 O(1) 完成，“找到这本书”用哈希表 `key → 节点` O(1) 完成。

```python
# dict: key -> Node，O(1) 查找节点
# Doubly Linked List: 维护 LRU -> MRU 顺序，O(1) 移动/删除节点
# Node 必须存 key：淘汰节点时，通过 node.key 在 O(1) 内删除 dict 对应记录
class Node:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        
    def _add_end(self, node):
        tmp = self.tail.prev
        tmp.next = node
        node.prev = tmp
        node.next = self.tail
        self.tail.prev = node
        
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 在类里调用自己的方法要写成 self
        self._remove(node)
        self._add_end(node)
        return node.val      

    def put(self, key: int, val: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = val
            # 在类里调用自己的方法要写成 self.
            self._remove(node)
            self._add_end(node)
        else:
            node = Node(key, val)
            self._add_end(node)
            self.cache[key] = node
            # 必须大于capacity, 才能删除
            if len(self.cache) > self.capacity:
                lru_node = self.head.next
                self._remove(lru_node)
                del self.cache[lru_node.key]
                

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()   # 头部 = 最近使用，尾部 = 最久未使用

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key, last=False)   # 移到头部
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value                    # 新 key 会加到尾部，旧 key 位置不变
        self.cache.move_to_end(key, last=False)    # 所以都要移到头部
        if len(self.cache) > self.capacity:
            self.cache.popitem()                   # 默认 last=True，删尾部（最久未使用）
```

复杂度：`get` / `put` 都是 O(1)；空间 O(capacity)。

Python 也可以直接用 `OrderedDict`：`move_to_end(key, last=False)` 移到头部，`popitem()` 删尾部。面试时最好两种都会写。

靠"环形"。只有一个 `dummy` 时，链表首尾相连，`dummy` 同时当头哨兵和尾哨兵

<br>

### 460. LFU 缓存 / LFU Cache

[CN](https://leetcode.cn/problems/lfu-cache/) · [EN](https://leetcode.com/problems/lfu-cache/) · Hard

**题意**：容量满时淘汰使用次数最少的 key；次数相同时淘汰最久未使用的。`get` / `put` 都要 O(1)。

**思路**：在 LRU 的基础上，按使用次数 `freq` 分成多摞书，每摞内部按最近使用排序。访问一本书时，把它从 `freq` 那摞抽出来，放到 `freq + 1` 那摞的最上面。维护 `min_freq`：淘汰时直接去 `min_freq` 那摞的最下面拿。

```python
from collections import defaultdict
from typing import Optional


class Node:

    def __init__(self, key=0, val=0):
        self.key = key
        self.value = val
        self.freq = 1  # 新书只读了一次
        self.prev = None
        self.next = None


class DLinkedList:
    # 一摞书：head.next 是最上面的书，tail.prev 是最下面的书
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def is_empty(self) -> bool:
        return self.head.next is self.tail


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.d = {}  # key -> Node
        # 1. freq -> 这摞书（每个 freq 一个带 head/tail 的双向链表）
        self.freq_list = defaultdict(DLinkedList)
        self.min_freq = 0

    def get_node(self, key: int) -> Optional[Node]:
        if key not in self.d:  # 没有这本书
            return None
        node = self.d[key]  # 有这本书
        # 2. 从 freq 那摞抽出来
        self.remove(node)
        lst = self.freq_list[node.freq]
        if lst.is_empty():  # 抽出来后，这摞书是空的
            del self.freq_list[node.freq]  # 移除空链表
            if self.min_freq == node.freq:  # 这摞书是最左边的
                self.min_freq += 1
        # 3. 放到 freq + 1 那摞的最上面
        node.freq += 1
        self.push_front(self.freq_list[node.freq], node)
        return node

    def get(self, key: int) -> int:
        node = self.get_node(key)
        return node.value if node else -1

    def put(self, key: int, value: int) -> None:
        node = self.get_node(key)
        if node:  # 有这本书
            node.value = value  # 更新 value
            return
        # 4. 书太多了：去掉 min_freq 那摞最下面的书
        if len(self.d) == self.capacity:
            lst = self.freq_list[self.min_freq]
            back_node = lst.tail.prev  # 最左边那摞书的最下面的书
            del self.d[back_node.key]
            self.remove(back_node)
            if lst.is_empty():  # 这摞书是空的
                del self.freq_list[self.min_freq]
        # 5. 新书放在 freq = 1 那摞的最上面
        self.d[key] = node = Node(key, value)
        self.push_front(self.freq_list[1], node)
        self.min_freq = 1

    def remove(self, x: Node) -> None:
        x.prev.next = x.next
        x.next.prev = x.prev

    # 放到这摞书的最上面：插在 head 后面
    def push_front(self, lst: DLinkedList, x: Node) -> None:
        x.prev = lst.head
        x.next = lst.head.next
        lst.head.next.prev = x
        lst.head.next = x
```

复杂度：`get` / `put` 都是 O(1)；空间 O(capacity)。

<br>

### 432. 全 O(1) 的数据结构 / All O'one Data Structure

[CN](https://leetcode.cn/problems/all-oone-data-structure/) · [EN](https://leetcode.com/problems/all-oone-data-structure/) · Hard

**题意**：`inc(key)` / `dec(key)` 让 key 的计数 ±1（减到 0 删除），`getMaxKey()` / `getMinKey()` 返回任一计数最大 / 最小的 key。全部 O(1)。

**思路**：链表的每个节点是一个“计数桶”，存计数 `cnt` 和所有计数等于 `cnt` 的 key；桶按 `cnt` 从小到大排列。计数 ±1 时，key 只会搬到相邻的桶（没有就新建），所以是 O(1)。最小在 `dummy.next`，最大在 `dummy.prev`。

```python
class Node:
    __slots__ = 'prev', 'next', 'keys', 'cnt'

    def __init__(self, cnt=0):
        self.cnt = cnt
        self.keys = set()


class AllOne:
    def __init__(self):
        # 1. 循环双向链表，桶按 cnt 递增；dummy.next 最小，dummy.prev 最大
        self.dummy = Node()
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.key_to_node = {}  # key -> 所在的桶

    # 在 node 后面插入一个计数为 cnt 的新桶
    def insert_after(self, node: Node, cnt: int) -> Node:
        new = Node(cnt)
        new.prev = node
        new.next = node.next
        node.next.prev = new
        node.next = new
        return new

    def remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def inc(self, key: str) -> None:
        if key not in self.key_to_node:
            # 2. 新 key：放进 cnt = 1 的桶（就是 dummy.next，不存在就新建）
            nxt = self.dummy.next
            if nxt is self.dummy or nxt.cnt > 1:
                nxt = self.insert_after(self.dummy, 1)
        else:
            # 3. 老 key：从当前桶搬到 cnt + 1 的桶
            cur = self.key_to_node[key]
            nxt = cur.next
            if nxt is self.dummy or nxt.cnt > cur.cnt + 1:
                nxt = self.insert_after(cur, cur.cnt + 1)
            cur.keys.remove(key)
            if not cur.keys:  # 4. 桶空了就删掉
                self.remove(cur)
        nxt.keys.add(key)
        self.key_to_node[key] = nxt

    def dec(self, key: str) -> None:
        cur = self.key_to_node[key]
        if cur.cnt == 1:
            # 5. 计数减到 0：直接删除 key
            del self.key_to_node[key]
        else:
            # 6. 搬到 cnt - 1 的桶
            pre = cur.prev
            if pre is self.dummy or pre.cnt < cur.cnt - 1:
                pre = self.insert_after(cur.prev, cur.cnt - 1)
            pre.keys.add(key)
            self.key_to_node[key] = pre
        cur.keys.remove(key)
        if not cur.keys:
            self.remove(cur)

    def getMaxKey(self) -> str:
        # 7. 最后一个桶的任意 key
        last = self.dummy.prev
        return next(iter(last.keys)) if last is not self.dummy else ""

    def getMinKey(self) -> str:
        first = self.dummy.next
        return next(iter(first.keys)) if first is not self.dummy else ""
```

复杂度：所有操作 O(1)；空间 O(n)。

<br>
<br>

## 2. 基础数据结构实现

这一类是把栈、队列、哈希表、链表自己实现一遍，或者在它们上面加一个 O(1) 的额外查询（最小值、最大频率、增量）。关键是想清楚“额外信息存在哪里”：存在每个元素旁边（155），按频率分组（895），还是懒标记到用时再算（1381）。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 155 | 最小栈 | [CN](https://leetcode.cn/problems/min-stack/) · [EN](https://leetcode.com/problems/min-stack/) | 栈里存 (值, 前缀最小值) |
| 225 | 用队列实现栈 | [CN](https://leetcode.cn/problems/implement-stack-using-queues/) · [EN](https://leetcode.com/problems/implement-stack-using-queues/) | 一个队列，入队后旋转 |
| 232 | 用栈实现队列 | [CN](https://leetcode.cn/problems/implement-queue-using-stacks/) · [EN](https://leetcode.com/problems/implement-queue-using-stacks/) | 输入栈 + 输出栈 |
| 622 | 设计循环队列 | [CN](https://leetcode.cn/problems/design-circular-queue/) · [EN](https://leetcode.com/problems/design-circular-queue/) | 数组 + 头指针 + 长度 |
| 641 | 设计循环双端队列 | [CN](https://leetcode.cn/problems/design-circular-deque/) · [EN](https://leetcode.com/problems/design-circular-deque/) | 数组 + 头指针 + 长度 |
| 705 | 设计哈希集合 | [CN](https://leetcode.cn/problems/design-hashset/) · [EN](https://leetcode.com/problems/design-hashset/) | 分桶 + 链地址法 |
| 706 | 设计哈希映射 | [CN](https://leetcode.cn/problems/design-hashmap/) · [EN](https://leetcode.com/problems/design-hashmap/) | 分桶 + 链地址法 |
| 707 | 设计链表 | [CN](https://leetcode.cn/problems/design-linked-list/) · [EN](https://leetcode.com/problems/design-linked-list/) | 哨兵头结点 + size |
| 895 | 最大频率栈 | [CN](https://leetcode.cn/problems/maximum-frequency-stack/) · [EN](https://leetcode.com/problems/maximum-frequency-stack/) | 频次 → 栈 |
| 716 🔒 | 最大栈 | [CN](https://leetcode.cn/problems/max-stack/) · [EN](https://leetcode.com/problems/max-stack/) | 栈 + 堆 + 懒删除 |
| 1381 | 设计一个支持增量操作的栈 | [CN](https://leetcode.cn/problems/design-a-stack-with-increment-operation/) · [EN](https://leetcode.com/problems/design-a-stack-with-increment-operation/) | 懒标记数组 |
| 1172 | 餐盘栈 | [CN](https://leetcode.cn/problems/dinner-plate-stacks/) · [EN](https://leetcode.com/problems/dinner-plate-stacks/) | 栈数组 + 最小堆 |
| 1670 | 设计前中后队列 | [CN](https://leetcode.cn/problems/design-front-middle-back-queue/) · [EN](https://leetcode.com/problems/design-front-middle-back-queue/) | 两个双端队列 |
| 1756 🔒 | 设计最近使用（MRU）队列 | [CN](https://leetcode.cn/problems/design-most-recently-used-queue/) · [EN](https://leetcode.com/problems/design-most-recently-used-queue/) | 数组（进阶：树状数组） |
| 1570 🔒 | 两个稀疏向量的点积 | [CN](https://leetcode.cn/problems/dot-product-of-two-sparse-vectors/) · [EN](https://leetcode.com/problems/dot-product-of-two-sparse-vectors/) | 只存非零项 |

<br>

### 155. 最小栈 / Min Stack

[CN](https://leetcode.cn/problems/min-stack/) · [EN](https://leetcode.com/problems/min-stack/) · Medium

**题意**：实现栈的 `push / pop / top`，外加 O(1) 的 `getMin()`。

**思路**：栈里每个元素存一对 `(val, 从栈底到它的最小值)`。出栈后，新栈顶自带的最小值就是剩下元素的最小值，不用重新算。栈底放一个哨兵 `(0, inf)`，省掉判空。

```python
class MinStack:
    def __init__(self):
        # 1. 哨兵：前缀最小值从 inf 开始，push 时不用判空
        self.st = [(0, inf)]  # (val, 前缀最小值)

    def push(self, val: int) -> None:
        # 2. 新的前缀最小值 = min(上一个前缀最小值, val)
        self.st.append((val, min(self.st[-1][1], val)))

    def pop(self) -> None:
        self.st.pop()

    def top(self) -> int:
        return self.st[-1][0]

    def getMin(self) -> int:
        # 3. 栈顶记录的就是整个栈的最小值
        return self.st[-1][1]
```

复杂度：全部 O(1)；空间 O(n)。

<br>

### 225. 用队列实现栈 / Implement Stack using Queues

[CN](https://leetcode.cn/problems/implement-stack-using-queues/) · [EN](https://leetcode.com/problems/implement-stack-using-queues/) · Easy

**题意**：只用队列的 `append / popleft / 看队首 / 长度` 实现栈。

**思路**：只用一个队列。每次 push 后，把它前面的 n-1 个元素依次出队再入队，新元素就转到了队首，队首就是栈顶。

```python
class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        # 1. 新元素入队
        self.q.append(x)
        # 2. 把前面的 n-1 个元素转到它后面，新元素到了队首
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        # 3. 队首就是栈顶
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not self.q
```

复杂度：`push` O(n)，其余 O(1)。

<br>

### 232. 用栈实现队列 / Implement Queue using Stacks

[CN](https://leetcode.cn/problems/implement-queue-using-stacks/) · [EN](https://leetcode.com/problems/implement-queue-using-stacks/) · Easy

**题意**：只用栈实现队列的 `push / pop / peek / empty`。

**思路**：一摞书倒过来就是先进先出。`push` 放进输入栈；要取队首时，如果输出栈空了，把输入栈整个倒进输出栈，输出栈顶就是最早进来的元素。每个元素最多被倒一次，均摊 O(1)。

```python
class MyQueue:
    def __init__(self):
        self.in_st = []   # 1. 输入栈：只负责 push
        self.out_st = []  # 2. 输出栈：栈顶是队首

    def push(self, x: int) -> None:
        self.in_st.append(x)

    def pop(self) -> int:
        self.peek()  # 3. 保证输出栈不空
        return self.out_st.pop()

    def peek(self) -> int:
        # 4. 输出栈空了才倒，倒过来后顺序正好反转
        if not self.out_st:
            while self.in_st:
                self.out_st.append(self.in_st.pop())
        return self.out_st[-1]

    def empty(self) -> bool:
        return not self.in_st and not self.out_st
```

复杂度：均摊 O(1)。

<br>

### 622. 设计循环队列 / Design Circular Queue

[CN](https://leetcode.cn/problems/design-circular-queue/) · [EN](https://leetcode.com/problems/design-circular-queue/) · Medium

**题意**：用固定容量 k 实现循环队列：`enQueue / deQueue / Front / Rear / isEmpty / isFull`。

**思路**：数组 + 队首下标 `front` + 元素个数 `size`。队尾下标是 `(front + size - 1) % k`。用 `size` 而不是两个指针，判空判满最简单。

```python
class MyCircularQueue:
    def __init__(self, k: int):
        self.q = [0] * k
        self.front = 0  # 1. 队首下标
        self.size = 0   # 2. 当前元素个数

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        # 3. 新元素放在队尾的下一个位置
        self.q[(self.front + self.size) % len(self.q)] = value
        self.size += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        # 4. 队首后移一格
        self.front = (self.front + 1) % len(self.q)
        self.size -= 1
        return True

    def Front(self) -> int:
        return -1 if self.isEmpty() else self.q[self.front]

    def Rear(self) -> int:
        return -1 if self.isEmpty() else self.q[(self.front + self.size - 1) % len(self.q)]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == len(self.q)
```

复杂度：全部 O(1)；空间 O(k)。

<br>

### 641. 设计循环双端队列 / Design Circular Deque

[CN](https://leetcode.cn/problems/design-circular-deque/) · [EN](https://leetcode.com/problems/design-circular-deque/) · Medium

**题意**：在 622 的基础上，队首也能插入，队尾也能删除。

**思路**：同 622。队首插入时 `front` 先往左退一格（取模），队尾删除只需 `size -= 1`。

```python
class MyCircularDeque:
    def __init__(self, k: int):
        self.q = [0] * k
        self.front = 0
        self.size = 0

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        # 1. front 左移一格（Python 的 % 对负数也返回非负）
        self.front = (self.front - 1) % len(self.q)
        self.q[self.front] = value
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        # 2. 放在队尾的下一个位置
        self.q[(self.front + self.size) % len(self.q)] = value
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % len(self.q)
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        # 3. 队尾删除：长度减一即可
        self.size -= 1
        return True

    def getFront(self) -> int:
        return -1 if self.isEmpty() else self.q[self.front]

    def getRear(self) -> int:
        return -1 if self.isEmpty() else self.q[(self.front + self.size - 1) % len(self.q)]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == len(self.q)
```

复杂度：全部 O(1)。

<br>

### 705. 设计哈希集合 / Design HashSet

[CN](https://leetcode.cn/problems/design-hashset/) · [EN](https://leetcode.com/problems/design-hashset/) · Easy

**题意**：不用内置哈希表，实现 `add / remove / contains`，key 范围 `[0, 10^6]`。

**思路**：链地址法。取一个质数个桶，`key % BASE` 决定落在哪个桶，桶里用列表存冲突的 key。（key 范围小时也可以直接开 `10^6 + 1` 的布尔数组。）

```python
class MyHashSet:
    BASE = 769  # 1. 桶数取质数，让 key 分布更均匀

    def __init__(self):
        self.buckets = [[] for _ in range(self.BASE)]

    def add(self, key: int) -> None:
        # 2. 找到桶，不存在才加入
        b = self.buckets[key % self.BASE]
        if key not in b:
            b.append(key)

    def remove(self, key: int) -> None:
        b = self.buckets[key % self.BASE]
        if key in b:
            b.remove(key)

    def contains(self, key: int) -> bool:
        return key in self.buckets[key % self.BASE]
```

复杂度：均摊 O(n / BASE)；空间 O(BASE + n)。

<br>

### 706. 设计哈希映射 / Design HashMap

[CN](https://leetcode.cn/problems/design-hashmap/) · [EN](https://leetcode.com/problems/design-hashmap/) · Easy

**题意**：不用内置哈希表，实现 `put / get / remove`。

**思路**：同 705，桶里存 `[key, value]`。

```python
class MyHashMap:
    BASE = 769

    def __init__(self):
        self.buckets = [[] for _ in range(self.BASE)]

    def put(self, key: int, value: int) -> None:
        b = self.buckets[key % self.BASE]
        # 1. 已存在就更新
        for pair in b:
            if pair[0] == key:
                pair[1] = value
                return
        # 2. 不存在就追加
        b.append([key, value])

    def get(self, key: int) -> int:
        for k, v in self.buckets[key % self.BASE]:
            if k == key:
                return v
        return -1

    def remove(self, key: int) -> None:
        b = self.buckets[key % self.BASE]
        for i, (k, _) in enumerate(b):
            if k == key:
                # 3. 和最后一个交换后删除，O(1)
                b[i] = b[-1]
                b.pop()
                return
```

复杂度：均摊 O(n / BASE)。

<br>

### 707. 设计链表 / Design Linked List

[CN](https://leetcode.cn/problems/design-linked-list/) · [EN](https://leetcode.com/problems/design-linked-list/) · Medium

**题意**：实现单链表的 `get(index) / addAtHead / addAtTail / addAtIndex / deleteAtIndex`。

**思路**：哨兵头结点 + `size`。所有插入、删除都先走到第 `index - 1` 个节点（哨兵算第 -1 个），头插、尾插都是 `addAtIndex` 的特例。

```python
class ListNode:
    __slots__ = 'val', 'next'

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MyLinkedList:
    def __init__(self):
        self.dummy = ListNode()  # 1. 哨兵，dummy.next 是第 0 个节点
        self.size = 0

    # 2. 返回下标为 index 的节点；index = -1 返回哨兵
    def _node(self, index: int) -> ListNode:
        cur = self.dummy
        for _ in range(index + 1):
            cur = cur.next
        return cur

    def get(self, index: int) -> int:
        if not 0 <= index < self.size:
            return -1
        return self._node(index).val

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        index = max(index, 0)
        # 3. 走到前驱，接上新节点
        pre = self._node(index - 1)
        pre.next = ListNode(val, pre.next)
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if not 0 <= index < self.size:
            return
        # 4. 走到前驱，跳过被删节点
        pre = self._node(index - 1)
        pre.next = pre.next.next
        self.size -= 1
```

复杂度：按下标操作 O(index)；头插 O(1)。

<br>

### 895. 最大频率栈 / Maximum Frequency Stack

[CN](https://leetcode.cn/problems/maximum-frequency-stack/) · [EN](https://leetcode.com/problems/maximum-frequency-stack/) · Hard

**题意**：`push(val)`；`pop()` 弹出出现频率最高的元素，频率相同弹最靠近栈顶的。

**思路**：把栈“按频率拆开”：第 1 次出现的元素压进第 0 个栈，第 2 次出现的压进第 1 个栈……同一个值会出现在多个栈里。`pop` 时只看最后一个栈的栈顶，它就是频率最高且最近压入的。最后一个栈空了就扔掉。

```python
class FreqStack:
    def __init__(self):
        self.cnt = defaultdict(int)  # 1. 每个值当前出现了几次
        self.stacks = []             # 2. stacks[i]：第 i+1 次出现的元素

    def push(self, val: int) -> None:
        # 3. 第 cnt 次出现，压入下标为 cnt 的栈（不存在就新建）
        if self.cnt[val] == len(self.stacks):
            self.stacks.append([val])
        else:
            self.stacks[self.cnt[val]].append(val)
        self.cnt[val] += 1

    def pop(self) -> int:
        # 4. 最后一个栈的栈顶：频率最高、离栈顶最近
        val = self.stacks[-1].pop()
        if not self.stacks[-1]:
            self.stacks.pop()
        self.cnt[val] -= 1
        return val
```

复杂度：全部 O(1)；空间 O(n)。

<br>

### 716. 最大栈 / Max Stack 🔒

[CN](https://leetcode.cn/problems/max-stack/) · [EN](https://leetcode.com/problems/max-stack/) · Hard

**题意**：栈操作外加 `peekMax()` 返回最大值、`popMax()` 删除最大值（有多个删最靠近栈顶的）。`top` 要 O(1)，其余 O(log n)。

**思路**：给每个元素一个自增 id。栈存 `(val, id)`，最大堆存 `(-val, -id)`（值大优先，值相同时 id 大的即靠近栈顶的优先）。一个元素从一边删掉后，另一边懒删除：把 id 记进 `removed`，下次看到时再丢弃。

```python
class MaxStack:
    def __init__(self):
        self.st = []        # 1. (val, id)
        self.h = []         # 2. (-val, -id)：最大值、最靠近栈顶优先
        self.removed = set()  # 3. 已经从另一边删掉的 id
        self.id = 0

    def push(self, x: int) -> None:
        self.st.append((x, self.id))
        heappush(self.h, (-x, -self.id))
        self.id += 1

    # 4. 清理栈顶 / 堆顶已被删掉的元素
    def _clean_st(self) -> None:
        while self.st[-1][1] in self.removed:
            self.removed.remove(self.st.pop()[1])

    def _clean_h(self) -> None:
        while -self.h[0][1] in self.removed:
            self.removed.remove(-heappop(self.h)[1])

    def pop(self) -> int:
        self._clean_st()
        x, i = self.st.pop()
        self.removed.add(i)  # 5. 堆里那份留着，懒删除
        return x

    def top(self) -> int:
        self._clean_st()
        return self.st[-1][0]

    def peekMax(self) -> int:
        self._clean_h()
        return -self.h[0][0]

    def popMax(self) -> int:
        self._clean_h()
        x, i = heappop(self.h)
        self.removed.add(-i)  # 6. 栈里那份留着，懒删除
        return -x
```

复杂度：`push / pop / popMax / peekMax` 均摊 O(log n)；`top` 均摊 O(1)。

<br>

### 1381. 设计一个支持增量操作的栈 / Design a Stack With Increment Operation

[CN](https://leetcode.cn/problems/design-a-stack-with-increment-operation/) · [EN](https://leetcode.com/problems/design-a-stack-with-increment-operation/) · Medium

**题意**：容量为 `maxSize` 的栈，额外支持 `increment(k, val)`：栈底的 k 个元素都加 `val`。

**思路**：懒标记。`add[i]` 表示“栈底前 i+1 个元素都要加 add[i]”，`increment` 只改 `add[min(k, size) - 1]`。`pop` 第 i 个元素时，把 `add[i]` 加给它，再把 `add[i]` 传给 `add[i-1]`（下面的元素也要加）。

```python
class CustomStack:
    def __init__(self, maxSize: int):
        self.st = []
        self.add = [0] * maxSize  # 1. 懒标记
        self.max_size = maxSize

    def push(self, x: int) -> None:
        if len(self.st) < self.max_size:
            self.st.append(x)

    def pop(self) -> int:
        if not self.st:
            return -1
        i = len(self.st) - 1
        # 2. 弹出时才把懒标记加上
        res = self.st.pop() + self.add[i]
        # 3. 标记往下传给 i-1
        if i > 0:
            self.add[i - 1] += self.add[i]
        self.add[i] = 0
        return res

    def increment(self, k: int, val: int) -> None:
        # 4. 只在第 min(k, size) 个位置打标记
        i = min(k, len(self.st)) - 1
        if i >= 0:
            self.add[i] += val
```

复杂度：全部 O(1)。

<br>

### 1172. 餐盘栈 / Dinner Plate Stacks

[CN](https://leetcode.cn/problems/dinner-plate-stacks/) · [EN](https://leetcode.com/problems/dinner-plate-stacks/) · Hard

**题意**：无限个容量为 `capacity` 的栈排成一行。`push` 放到最左边未满的栈；`pop` 从最右边非空栈弹出；`popAtStack(i)` 从第 i 个栈弹出。

**思路**（灵神写法）：栈列表 `stacks` + 最小堆 `idx` 存“未满栈的下标”。`push` 时堆顶就是最左未满的栈，但堆里可能有下标已经越界（右边的空栈被删了）的脏数据，先弹掉；`pop` 时从右边删掉所有空栈，保证 `stacks[-1]` 非空。

```python
class DinnerPlates:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []  # 1. 所有栈
        self.idx = []     # 2. 最小堆：未满栈的下标

    def push(self, val: int) -> None:
        # 3. 堆里的下标可能越界了（对应的栈已经被删除），懒删除
        if self.idx and self.idx[0] >= len(self.stacks):
            self.idx = []  # 越界说明堆里全是越界下标，直接清空
        if self.idx:
            # 4. 放进最左边的未满栈；放满了就出堆
            self.stacks[self.idx[0]].append(val)
            if len(self.stacks[self.idx[0]]) == self.capacity:
                heappop(self.idx)
        else:
            # 5. 没有未满栈：在最右边新开一个
            self.stacks.append([val])
            if self.capacity > 1:
                heappush(self.idx, len(self.stacks) - 1)

    def pop(self) -> int:
        return self.popAtStack(len(self.stacks) - 1)

    def popAtStack(self, index: int) -> int:
        if index < 0 or index >= len(self.stacks) or not self.stacks[index]:
            return -1
        # 6. 从满变成未满，下标入堆
        if len(self.stacks[index]) == self.capacity:
            heappush(self.idx, index)
        val = self.stacks[index].pop()
        # 7. 去掉末尾的空栈，保证最右边的栈非空
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()
        return val
```

第 3 步能直接清空，是因为第 7 步只删末尾的空栈：堆顶越界说明堆里最小的下标都越界了，其他下标只会更大。

复杂度：`push / popAtStack` O(log n)，`pop` 均摊 O(log n)。

<br>

### 1670. 设计前中后队列 / Design Front Middle Back Queue

[CN](https://leetcode.cn/problems/design-front-middle-back-queue/) · [EN](https://leetcode.com/problems/design-front-middle-back-queue/) · Medium

**题意**：支持在前、中、后三个位置 push 和 pop。中间位置有两个时取靠前的。

**思路**：两个双端队列 `left`、`right` 各存一半，始终保持 `len(left) == len(right)` 或 `len(left) + 1 == len(right)`。这样中间位置就在两个队列的交界处。每次操作后调用 `balance()` 恢复这个约束。

```python
class FrontMiddleBackQueue:
    def __init__(self):
        self.left = deque()
        self.right = deque()

    # 1. 保持 len(right) - len(left) ∈ {0, 1}
    def balance(self) -> None:
        if len(self.left) > len(self.right):
            self.right.appendleft(self.left.pop())
        elif len(self.right) > len(self.left) + 1:
            self.left.append(self.right.popleft())

    def pushFront(self, val: int) -> None:
        self.left.appendleft(val)
        self.balance()

    def pushMiddle(self, val: int) -> None:
        # 2. 两边一样长放 right 开头；right 多一个放 left 末尾
        if len(self.left) < len(self.right):
            self.left.append(val)
        else:
            self.right.appendleft(val)

    def pushBack(self, val: int) -> None:
        self.right.append(val)
        self.balance()

    def popFront(self) -> int:
        if not self.right:  # 3. right 为空说明整个队列为空
            return -1
        val = self.left.popleft() if self.left else self.right.popleft()
        self.balance()
        return val

    def popMiddle(self) -> int:
        if not self.right:
            return -1
        # 4. 总数为偶数时中间偏左的在 left 末尾，奇数时在 right 开头
        if len(self.left) == len(self.right):
            return self.left.pop()
        return self.right.popleft()

    def popBack(self) -> int:
        if not self.right:
            return -1
        val = self.right.pop()
        self.balance()
        return val
```

复杂度：全部 O(1)。

<br>

### 1756. 设计最近使用（MRU）队列 / Design Most Recently Used Queue 🔒

[CN](https://leetcode.cn/problems/design-most-recently-used-queue/) · [EN](https://leetcode.com/problems/design-most-recently-used-queue/) · Medium

**题意**：队列初始为 `[1, 2, ..., n]`。`fetch(k)` 把第 k 个元素（从 1 开始）移到队尾并返回它。

**思路**：n ≤ 2000，直接用列表 `pop(k-1)` + `append` 就够了。进阶做法：在长度为 `n + 调用次数` 的数组上用树状数组标记“这个位置有没有元素”，`fetch` 时用树状数组二分找第 k 个有元素的位置，做到 O(log n)。

```python
class MRUQueue:
    def __init__(self, n: int):
        # 1. 初始队列 1..n
        self.q = list(range(1, n + 1))

    def fetch(self, k: int) -> int:
        # 2. 取出第 k 个，放到队尾
        val = self.q.pop(k - 1)
        self.q.append(val)
        return val
```

复杂度：`fetch` O(n)。

<br>

### 1570. 两个稀疏向量的点积 / Dot Product of Two Sparse Vectors 🔒

[CN](https://leetcode.cn/problems/dot-product-of-two-sparse-vectors/) · [EN](https://leetcode.com/problems/dot-product-of-two-sparse-vectors/) · Medium

**题意**：用一个类存稀疏向量，实现 `dotProduct(vec)` 求两个向量的点积。

**思路**：只存非零项 `下标 → 值`。点积时遍历非零项较少的那个，到另一个里查。（Meta 的追问：如果只有一个向量稀疏怎么办？——遍历稀疏的那个即可；两个都用有序列表存时可以双指针。）

```python
class SparseVector:
    def __init__(self, nums: List[int]):
        # 1. 只存非零项
        self.d = {i: x for i, x in enumerate(nums) if x}

    def dotProduct(self, vec: 'SparseVector') -> int:
        a, b = self.d, vec.d
        # 2. 遍历非零项少的那个
        if len(a) > len(b):
            a, b = b, a
        # 3. 两边下标都非零才有贡献
        return sum(x * b[i] for i, x in a.items() if i in b)
```

复杂度：构造 O(n)；点积 O(min(非零项数))。

<br>
<br>

## 3. 随机化 / O(1) 操作

“等概率随机返回一个元素”要求元素放在连续的数组里，才能 `choice`。数组删除中间元素是 O(n)，技巧是“和末尾交换再删末尾”，这需要哈希表记录每个值的下标。其余几题是三个经典随机算法：Fisher-Yates 洗牌、蓄水池抽样、前缀和 + 二分。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 380 | O(1) 时间插入、删除和获取随机元素 | [CN](https://leetcode.cn/problems/insert-delete-getrandom-o1/) · [EN](https://leetcode.com/problems/insert-delete-getrandom-o1/) | 数组 + 哈希表（和末尾交换后删除） |
| 381 | 同上，允许重复 | [CN](https://leetcode.cn/problems/insert-delete-getrandom-o1-duplicates-allowed/) · [EN](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) | 数组 + 哈希表 → 下标集合 |
| 384 | 打乱数组 | [CN](https://leetcode.cn/problems/shuffle-an-array/) · [EN](https://leetcode.com/problems/shuffle-an-array/) | Fisher-Yates 洗牌 |
| 382 | 链表随机节点 | [CN](https://leetcode.cn/problems/linked-list-random-node/) · [EN](https://leetcode.com/problems/linked-list-random-node/) | 蓄水池抽样 |
| 398 | 随机数索引 | [CN](https://leetcode.cn/problems/random-pick-index/) · [EN](https://leetcode.com/problems/random-pick-index/) | 蓄水池抽样 / 哈希表存下标 |
| 528 | 按权重随机选择 | [CN](https://leetcode.cn/problems/random-pick-with-weight/) · [EN](https://leetcode.com/problems/random-pick-with-weight/) | 前缀和 + 二分 |
| 710 | 黑名单中的随机数 | [CN](https://leetcode.cn/problems/random-pick-with-blacklist/) · [EN](https://leetcode.com/problems/random-pick-with-blacklist/) | 哈希映射重排 |
| 519 | 随机翻转矩阵 | [CN](https://leetcode.cn/problems/random-flip-matrix/) · [EN](https://leetcode.com/problems/random-flip-matrix/) | 哈希表模拟洗牌 |

<br>

### 380. O(1) 时间插入、删除和获取随机元素 / Insert Delete GetRandom O(1)

[CN](https://leetcode.cn/problems/insert-delete-getrandom-o1/) · [EN](https://leetcode.com/problems/insert-delete-getrandom-o1/) · Medium

**题意**：实现集合的 `insert / remove / getRandom`，三个操作都要平均 O(1)，`getRandom` 等概率返回一个元素。

**思路**：数组 `nums` 存元素（支持随机下标），哈希表 `idx` 存 `值 → 下标`（支持 O(1) 查找）。删除时把末尾元素搬到被删元素的位置，再删末尾，数组保持连续。

```python
class RandomizedSet:
    def __init__(self):
        self.nums = []  # 1. 连续存放所有元素
        self.idx = {}   # 2. 值 -> 在 nums 中的下标

    def insert(self, val: int) -> bool:
        if val in self.idx:
            return False
        self.idx[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.idx:
            return False
        # 3. 用末尾元素覆盖 val 的位置，并更新末尾元素的下标
        i = self.idx[val]
        last = self.nums[-1]
        self.nums[i] = last
        self.idx[last] = i
        # 4. 删除末尾和 val 的记录（先更新再删，val 恰好是末尾时也正确）
        self.nums.pop()
        del self.idx[val]
        return True

    def getRandom(self) -> int:
        return choice(self.nums)
```

复杂度：全部平均 O(1)。

<br>

### 381. O(1) 时间插入、删除和获取随机元素 - 允许重复 / Insert Delete GetRandom O(1) - Duplicates allowed

[CN](https://leetcode.cn/problems/insert-delete-getrandom-o1-duplicates-allowed/) · [EN](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) · Hard

**题意**：同 380，但允许重复元素，`getRandom` 返回每个元素的概率和它出现的次数成正比。

**思路**：同一个值可能出现在多个下标，把 `值 → 下标` 改成 `值 → 下标集合`。删除时从集合里任取一个下标 i，用末尾元素覆盖它，再修正末尾元素的下标集合。

```python
class RandomizedCollection:
    def __init__(self):
        self.nums = []
        self.idx = defaultdict(set)  # 1. 值 -> 下标集合

    def insert(self, val: int) -> bool:
        self.idx[val].add(len(self.nums))
        self.nums.append(val)
        return len(self.idx[val]) == 1  # 2. 之前不存在才返回 True

    def remove(self, val: int) -> bool:
        if not self.idx[val]:
            return False
        # 3. 任取 val 的一个下标 i，末尾元素 last 的下标是 n-1
        i = self.idx[val].pop()
        last = self.nums[-1]
        n = len(self.nums)
        self.nums[i] = last
        # 4. last 从 n-1 搬到 i（顺序很重要：val 就是 last 时也正确）
        self.idx[last].add(i)
        self.idx[last].discard(n - 1)
        self.nums.pop()
        return True

    def getRandom(self) -> int:
        return choice(self.nums)
```

复杂度：全部平均 O(1)。

<br>

### 384. 打乱数组 / Shuffle an Array

[CN](https://leetcode.cn/problems/shuffle-an-array/) · [EN](https://leetcode.com/problems/shuffle-an-array/) · Medium

**题意**：`reset()` 返回原数组，`shuffle()` 返回一个等概率的随机排列。

**思路**：Fisher-Yates 洗牌。从左到右，第 i 个位置从 `[i, n-1]` 里等概率挑一个数换过来。第 i 个位置填哪个数有 n-i 种等概率选择，总共 n! 种结果，每种概率相同。

```python
class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums

    def reset(self) -> List[int]:
        return self.nums

    def shuffle(self) -> List[int]:
        a = self.nums.copy()  # 1. 不能改原数组，reset 还要用
        for i in range(len(a)):
            # 2. 从 [i, n-1] 中等概率选一个换到位置 i
            j = randint(i, len(a) - 1)
            a[i], a[j] = a[j], a[i]
        return a
```

复杂度：`shuffle` O(n)。

<br>

### 382. 链表随机节点 / Linked List Random Node

[CN](https://leetcode.cn/problems/linked-list-random-node/) · [EN](https://leetcode.com/problems/linked-list-random-node/) · Medium

**题意**：等概率返回链表中一个节点的值。进阶：链表很长、长度未知，不用额外空间。

**思路**：蓄水池抽样。遍历到第 i 个节点（从 1 开始）时，以 1/i 的概率把答案换成它。第 i 个节点最终被选中的概率 = 1/i × i/(i+1) × … × (n-1)/n = 1/n。

```python
class Solution:
    def __init__(self, head: Optional[ListNode]):
        self.head = head

    def getRandom(self) -> int:
        ans = 0
        node, i = self.head, 1
        while node:
            # 1. 以 1/i 的概率选当前节点
            if randrange(i) == 0:
                ans = node.val
            node = node.next
            i += 1
        return ans
```

复杂度：`getRandom` O(n)，额外空间 O(1)。

<br>

### 398. 随机数索引 / Random Pick Index

[CN](https://leetcode.cn/problems/random-pick-index/) · [EN](https://leetcode.com/problems/random-pick-index/) · Medium

**题意**：数组中可能有重复元素，`pick(target)` 等概率返回一个满足 `nums[i] == target` 的下标。

**思路**：同 382 的蓄水池抽样，只在 `nums[i] == target` 时计数。（空间允许的话，预处理 `值 → 下标列表` 再 `choice` 更快。）

```python
class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        ans = cnt = 0
        for i, x in enumerate(self.nums):
            if x == target:
                cnt += 1  # 1. 这是第 cnt 个等于 target 的下标
                # 2. 以 1/cnt 的概率选它
                if randrange(cnt) == 0:
                    ans = i
        return ans
```

复杂度：`pick` O(n)，额外空间 O(1)。

<br>

### 528. 按权重随机选择 / Random Pick with Weight

[CN](https://leetcode.cn/problems/random-pick-with-weight/) · [EN](https://leetcode.com/problems/random-pick-with-weight/) · Medium

**题意**：`pickIndex()` 随机返回下标 i，概率为 `w[i] / sum(w)`。

**思路**：把下标 i 想成数轴上长度为 `w[i]` 的一段。算前缀和 `s`，在 `[1, total]` 里随机一个整数 x，二分找第一个 `s[i] >= x` 的 i。

```python
class Solution:
    def __init__(self, w: List[int]):
        # 1. 前缀和：下标 i 对应区间 (s[i-1], s[i]]
        self.s = list(accumulate(w))

    def pickIndex(self) -> int:
        # 2. 在 [1, total] 中随机
        x = randint(1, self.s[-1])
        # 3. 第一个 >= x 的前缀和
        return bisect_left(self.s, x)
```

复杂度：构造 O(n)；`pickIndex` O(log n)。

<br>

### 710. 黑名单中的随机数 / Random Pick with Blacklist

[CN](https://leetcode.cn/problems/random-pick-with-blacklist/) · [EN](https://leetcode.com/problems/random-pick-with-blacklist/) · Hard

**题意**：在 `[0, n)` 中等概率随机返回一个不在黑名单里的整数，尽量少调用随机函数。

**思路**：白名单个数 `m = n - len(blacklist)`。只在 `[0, m)` 里随机；`[0, m)` 里的黑名单数字，各自映射到 `[m, n)` 里的一个白名单数字。这样 `[0, m)` 每个数都对应唯一的白名单数字，等概率。

```python
class Solution:
    def __init__(self, n: int, blacklist: List[int]):
        self.m = m = n - len(blacklist)
        black = set(blacklist)
        self.mp = {}
        w = m  # 1. 指向 [m, n) 中下一个候选
        for b in blacklist:
            if b < m:
                # 2. 跳过 [m, n) 中的黑名单数字
                while w in black:
                    w += 1
                # 3. 把 [0, m) 里的黑名单 b 映射到白名单 w
                self.mp[b] = w
                w += 1

    def pick(self) -> int:
        x = randrange(self.m)
        return self.mp.get(x, x)
```

复杂度：构造 O(len(blacklist))；`pick` O(1)。

<br>

### 519. 随机翻转矩阵 / Random Flip Matrix

[CN](https://leetcode.cn/problems/random-flip-matrix/) · [EN](https://leetcode.com/problems/random-flip-matrix/) · Medium

**题意**：`m × n` 的 0 矩阵，`flip()` 等概率选一个值为 0 的格子翻成 1 并返回坐标；`reset()` 全部归零。m、n 可达 10^4，调用最多 1000 次。

**思路**：把格子拉平成 `[0, m*n)`，用 Fisher-Yates 的思路：在 `[0, total)` 里随机选 x，返回“x 位置当前放的数”，然后把末尾位置 `total-1` 的数换到 x，`total -= 1`。数组太大，只用哈希表记录被换过的位置。

```python
class Solution:
    def __init__(self, m: int, n: int):
        self.m, self.n = m, n
        self.total = m * n
        self.mp = {}  # 1. 位置 -> 当前放的数（没被换过的位置放的就是自己）

    def flip(self) -> List[int]:
        # 2. 在剩余的 total 个位置中随机
        x = randrange(self.total)
        self.total -= 1
        idx = self.mp.get(x, x)
        # 3. 把末尾位置的数换到 x，相当于删掉 x
        self.mp[x] = self.mp.get(self.total, self.total)
        return [idx // self.n, idx % self.n]

    def reset(self) -> None:
        self.total = self.m * self.n
        self.mp.clear()
```

复杂度：`flip` O(1)；空间 O(调用次数)。

<br>
<br>

## 4. 迭代器

迭代器题的要点是“懒”：不要在构造函数里把所有结果展开（那样空间是 O(n)，也失去了迭代器的意义），而是只保存“继续往下走所需的最少状态”，每次 `next` 走一步。常见状态是一个栈（树、嵌套结构）或几个指针（数组）。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 173 | 二叉搜索树迭代器 | [CN](https://leetcode.cn/problems/binary-search-tree-iterator/) · [EN](https://leetcode.com/problems/binary-search-tree-iterator/) | 栈模拟中序遍历 |
| 284 | 窥视迭代器 | [CN](https://leetcode.cn/problems/peeking-iterator/) · [EN](https://leetcode.com/problems/peeking-iterator/) | 缓存下一个元素 |
| 341 | 扁平化嵌套列表迭代器 | [CN](https://leetcode.cn/problems/flatten-nested-list-iterator/) · [EN](https://leetcode.com/problems/flatten-nested-list-iterator/) | 栈 |
| 251 🔒 | 展开二维向量 | [CN](https://leetcode.cn/problems/flatten-2d-vector/) · [EN](https://leetcode.com/problems/flatten-2d-vector/) | 双指针 |
| 281 🔒 | 锯齿迭代器 | [CN](https://leetcode.cn/problems/zigzag-iterator/) · [EN](https://leetcode.com/problems/zigzag-iterator/) | 队列 |
| 900 | RLE 迭代器 | [CN](https://leetcode.cn/problems/rle-iterator/) · [EN](https://leetcode.com/problems/rle-iterator/) | 指针 |
| 1286 | 字母组合迭代器 | [CN](https://leetcode.cn/problems/iterator-for-combination/) · [EN](https://leetcode.com/problems/iterator-for-combination/) | 下标数组求下一个组合 |

<br>

### 173. 二叉搜索树迭代器 / Binary Search Tree Iterator

[CN](https://leetcode.cn/problems/binary-search-tree-iterator/) · [EN](https://leetcode.com/problems/binary-search-tree-iterator/) · Medium

**题意**：按中序遍历（从小到大）依次返回 BST 的节点值。`next / hasNext` 均摊 O(1)，空间 O(h)。

**思路**：把递归中序遍历拆开：栈里存“已经往左走、但还没访问”的节点。`next` 弹出栈顶访问，然后把它右子树的左链压栈。每个节点进出栈各一次，所以均摊 O(1)。

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.st = []
        self.push_left(root)  # 1. 先把最左链压栈，栈顶就是最小值

    # 2. 从 node 开始一路向左，全部压栈
    def push_left(self, node: Optional[TreeNode]) -> None:
        while node:
            self.st.append(node)
            node = node.left

    def next(self) -> int:
        # 3. 栈顶是当前最小的未访问节点
        node = self.st.pop()
        # 4. 访问完它，下一个要访问的是右子树的最左节点
        self.push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return bool(self.st)
```

复杂度：`next` 均摊 O(1)；空间 O(h)。

<br>

### 284. 窥视迭代器 / Peeking Iterator

[CN](https://leetcode.cn/problems/peeking-iterator/) · [EN](https://leetcode.com/problems/peeking-iterator/) · Medium

**题意**：给一个只有 `next / hasNext` 的迭代器，包装出额外的 `peek()`：看下一个元素但不移动。

**思路**：提前取出一个元素缓存起来。`peek` 返回缓存；`next` 返回缓存并再取一个；`hasNext` 看缓存是否为空。

```python
class PeekingIterator:
    def __init__(self, iterator):
        self.it = iterator
        # 1. 预取下一个元素，None 表示没有了
        self.nxt = iterator.next() if iterator.hasNext() else None

    def peek(self):
        return self.nxt

    def next(self):
        res = self.nxt
        # 2. 返回缓存，并预取再下一个
        self.nxt = self.it.next() if self.it.hasNext() else None
        return res

    def hasNext(self):
        return self.nxt is not None
```

复杂度：全部 O(1)。

<br>

### 341. 扁平化嵌套列表迭代器 / Flatten Nested List Iterator

[CN](https://leetcode.cn/problems/flatten-nested-list-iterator/) · [EN](https://leetcode.com/problems/flatten-nested-list-iterator/) · Medium

**题意**：嵌套列表里的元素是整数或列表，实现迭代器按顺序返回所有整数。

**思路**：栈里倒序存 `NestedInteger`（栈顶是下一个要处理的）。`hasNext` 负责“整理”：栈顶是列表就弹出、把它的元素倒序压栈，直到栈顶是整数或栈空。注意空列表 `[[]]` 的情况必须在 `hasNext` 里处理掉。

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        # 1. 倒序压栈，栈顶是第一个元素
        self.st = nestedList[::-1]

    def next(self) -> int:
        # 2. hasNext 保证了栈顶是整数
        return self.st.pop().getInteger()

    def hasNext(self) -> bool:
        # 3. 栈顶是列表就展开，直到栈顶是整数
        while self.st and not self.st[-1].isInteger():
            self.st.extend(self.st.pop().getList()[::-1])
        return bool(self.st)
```

复杂度：均摊 O(1)；空间 O(n)。

<br>

### 251. 展开二维向量 / Flatten 2D Vector 🔒

[CN](https://leetcode.cn/problems/flatten-2d-vector/) · [EN](https://leetcode.com/problems/flatten-2d-vector/) · Medium

**题意**：按顺序遍历二维数组（每行长度不同，可能有空行），实现 `next / hasNext`。

**思路**：两个指针 `(i, j)` 指向下一个元素。`hasNext` 负责跳过已经走完的行和空行。

```python
class Vector2D:
    def __init__(self, vec: List[List[int]]):
        self.vec = vec
        self.i = 0  # 1. 行指针
        self.j = 0  # 2. 列指针

    def next(self) -> int:
        self.hasNext()  # 3. 先跳到合法位置
        val = self.vec[self.i][self.j]
        self.j += 1
        return val

    def hasNext(self) -> bool:
        # 4. 当前行走完了（或是空行）就换下一行
        while self.i < len(self.vec) and self.j == len(self.vec[self.i]):
            self.i += 1
            self.j = 0
        return self.i < len(self.vec)
```

复杂度：均摊 O(1)。

<br>

### 281. 锯齿迭代器 / Zigzag Iterator 🔒

[CN](https://leetcode.cn/problems/zigzag-iterator/) · [EN](https://leetcode.com/problems/zigzag-iterator/) · Medium

**题意**：两个数组交替输出元素，一个用完后输出另一个剩下的。进阶：推广到 k 个数组。

**思路**：队列里放 `(数组, 下一个下标)`。每次从队首取一个元素，如果该数组还有剩余就放回队尾。天然支持 k 个数组。

```python
class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        # 1. 只放非空数组
        self.q = deque((v, 0) for v in (v1, v2) if v)

    def next(self) -> int:
        # 2. 取队首数组的下一个元素
        v, i = self.q.popleft()
        # 3. 还有剩余就放回队尾，实现轮流
        if i + 1 < len(v):
            self.q.append((v, i + 1))
        return v[i]

    def hasNext(self) -> bool:
        return bool(self.q)
```

复杂度：全部 O(1)。

<br>

### 900. RLE 迭代器 / RLE Iterator

[CN](https://leetcode.cn/problems/rle-iterator/) · [EN](https://leetcode.com/problems/rle-iterator/) · Medium

**题意**：游程编码数组 `encoding = [次数0, 值0, 次数1, 值1, ...]`。`next(n)` 消耗 n 个元素，返回最后一个被消耗的值；不够返回 -1。

**思路**：指针 `i` 指向当前段，就地减少 `encoding[i]` 表示这段已经用掉多少。

```python
class RLEIterator:
    def __init__(self, encoding: List[int]):
        self.a = encoding
        self.i = 0  # 1. 当前段的下标（偶数）

    def next(self, n: int) -> int:
        a = self.a
        while self.i < len(a):
            # 2. 当前段不够，全部用掉，跳到下一段
            if a[self.i] < n:
                n -= a[self.i]
                self.i += 2
            else:
                # 3. 当前段够用，扣掉 n
                a[self.i] -= n
                return a[self.i + 1]
        return -1
```

复杂度：所有 `next` 加起来 O(len(encoding) + 调用次数)。

<br>

### 1286. 字母组合迭代器 / Iterator for Combination

[CN](https://leetcode.cn/problems/iterator-for-combination/) · [EN](https://leetcode.com/problems/iterator-for-combination/) · Medium

**题意**：给有序且字符互不相同的字符串 `characters` 和长度 `k`，按字典序依次返回所有长为 k 的组合。

**思路**：用下标数组 `idx` 表示当前组合，初始为 `[0, 1, ..., k-1]`。求下一个组合：从右往左找第一个还能右移的位置 i（`idx[i] < n - k + i`），把它加一，后面的下标依次紧跟。找不到就说明已经是最后一个组合。（字符串很短，回溯预生成全部组合也能过。）

```python
class CombinationIterator:
    def __init__(self, characters: str, combinationLength: int):
        self.s = characters
        self.k = combinationLength
        self.idx = list(range(combinationLength))  # 1. 第一个组合
        self.has_next = True

    def next(self) -> str:
        s, k, idx = self.s, self.k, self.idx
        res = ''.join(s[i] for i in idx)
        # 2. 从右往左找第一个还能右移的位置
        n = len(s)
        i = k - 1
        while i >= 0 and idx[i] == n - k + i:
            i -= 1
        if i < 0:
            self.has_next = False  # 3. 已经是最后一个组合
        else:
            # 4. 该位置加一，后面的下标紧跟其后
            idx[i] += 1
            for j in range(i + 1, k):
                idx[j] = idx[j - 1] + 1
        return res

    def hasNext(self) -> bool:
        return self.has_next
```

复杂度：`next` O(k)。

<br>
<br>

## 5. 数据流与统计

数据一个个到来，要随时回答统计查询。先判断查询依赖“全部历史”还是“最近一段”：全部历史用堆 / 有序集合 / 前缀；最近一段用队列（按时间或个数淘汰旧数据）。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 295 | 数据流的中位数 | [CN](https://leetcode.cn/problems/find-median-from-data-stream/) · [EN](https://leetcode.com/problems/find-median-from-data-stream/) | 大顶堆 + 小顶堆 |
| 703 | 数据流中的第 K 大元素 | [CN](https://leetcode.cn/problems/kth-largest-element-in-a-stream/) · [EN](https://leetcode.com/problems/kth-largest-element-in-a-stream/) | 大小为 K 的小顶堆 |
| 346 🔒 | 数据流中的移动平均值 | [CN](https://leetcode.cn/problems/moving-average-from-data-stream/) · [EN](https://leetcode.com/problems/moving-average-from-data-stream/) | 队列 + 窗口和 |
| 352 | 将数据流变为多个不相交区间 | [CN](https://leetcode.cn/problems/data-stream-as-disjoint-intervals/) · [EN](https://leetcode.com/problems/data-stream-as-disjoint-intervals/) | 有序字典 |
| 901 | 股票价格跨度 | [CN](https://leetcode.cn/problems/online-stock-span/) · [EN](https://leetcode.com/problems/online-stock-span/) | 单调栈 |
| 2034 | 股票价格波动 | [CN](https://leetcode.cn/problems/stock-price-fluctuation/) · [EN](https://leetcode.com/problems/stock-price-fluctuation/) | 哈希表 + 有序集合 / 懒删除堆 |
| 1352 | 最后 K 个数的乘积 | [CN](https://leetcode.cn/problems/product-of-the-last-k-numbers/) · [EN](https://leetcode.com/problems/product-of-the-last-k-numbers/) | 前缀积 |
| 933 | 最近的请求次数 | [CN](https://leetcode.cn/problems/number-of-recent-calls/) · [EN](https://leetcode.com/problems/number-of-recent-calls/) | 队列 |
| 359 🔒 | 日志速率限制器 | [CN](https://leetcode.cn/problems/logger-rate-limiter/) · [EN](https://leetcode.com/problems/logger-rate-limiter/) | 哈希表 |
| 362 🔒 | 敲击计数器 | [CN](https://leetcode.cn/problems/design-hit-counter/) · [EN](https://leetcode.com/problems/design-hit-counter/) | 队列 / 环形数组 |
| 1825 | 求出 MK 平均值 | [CN](https://leetcode.cn/problems/finding-mk-average/) · [EN](https://leetcode.com/problems/finding-mk-average/) | 三个有序集合 |
| 2526 | 找到数据流中的连续整数 | [CN](https://leetcode.cn/problems/find-consecutive-integers-from-a-data-stream/) · [EN](https://leetcode.com/problems/find-consecutive-integers-from-a-data-stream/) | 计数器 |
| 1429 🔒 | 第一个唯一数字 | [CN](https://leetcode.cn/problems/first-unique-number/) · [EN](https://leetcode.com/problems/first-unique-number/) | 计数 + 队列懒删除 |

<br>

### 295. 数据流的中位数 / Find Median from Data Stream

[CN](https://leetcode.cn/problems/find-median-from-data-stream/) · [EN](https://leetcode.com/problems/find-median-from-data-stream/) · Hard

**题意**：`addNum(num)` 加入一个数，`findMedian()` 返回当前所有数的中位数。

**思路**：把数分成较小的一半 `left`（最大堆）和较大的一半 `right`（最小堆），保持 `len(left) == len(right)` 或 `len(left) == len(right) + 1`。中位数只和两个堆顶有关。加入时先让 num 进另一个堆“过一遍”，再把那个堆的堆顶移过来，这样不用分情况比较大小。

```python
class MedianFinder:
    def __init__(self):
        self.left = []   # 1. 最大堆（存相反数）：较小的一半
        self.right = []  # 2. 最小堆：较大的一半

    def addNum(self, num: int) -> None:
        if len(self.left) == len(self.right):
            # 3. 要给 left 加一个：num 先进 right，再把 right 的最小值移到 left
            x = heappushpop(self.right, num)
            heappush(self.left, -x)
        else:
            # 4. 要给 right 加一个：num 先进 left，再把 left 的最大值移到 right
            x = -heappushpop(self.left, -num)
            heappush(self.right, x)

    def findMedian(self) -> float:
        # 5. 奇数个：left 堆顶；偶数个：两个堆顶的平均
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (self.right[0] - self.left[0]) / 2
```

复杂度：`addNum` O(log n)；`findMedian` O(1)。

<br>

### 703. 数据流中的第 K 大元素 / Kth Largest Element in a Stream

[CN](https://leetcode.cn/problems/kth-largest-element-in-a-stream/) · [EN](https://leetcode.com/problems/kth-largest-element-in-a-stream/) · Easy

**题意**：初始给一个数组和 k，每次 `add(val)` 后返回当前第 k 大的元素。

**思路**：只保留最大的 k 个数，用最小堆存，堆顶就是第 k 大。新数进来后堆超过 k 个就弹出最小的。

```python
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.h = []  # 1. 最小堆，只存最大的 k 个数
        for x in nums:
            self.add(x)

    def add(self, val: int) -> int:
        heappush(self.h, val)
        # 2. 超过 k 个就去掉最小的
        if len(self.h) > self.k:
            heappop(self.h)
        # 3. 堆顶 = 最大的 k 个数里最小的 = 第 k 大
        return self.h[0]
```

复杂度：`add` O(log k)。

<br>

### 346. 数据流中的移动平均值 / Moving Average from Data Stream 🔒

[CN](https://leetcode.cn/problems/moving-average-from-data-stream/) · [EN](https://leetcode.com/problems/moving-average-from-data-stream/) · Easy

**题意**：给窗口大小 `size`，每次 `next(val)` 返回最近 `size` 个数的平均值。

**思路**：队列存窗口内的数，同时维护窗口和。超出窗口就把最早的数出队并从和里减掉。

```python
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.q = deque()
        self.s = 0  # 1. 窗口内元素之和

    def next(self, val: int) -> float:
        self.q.append(val)
        self.s += val
        # 2. 超出窗口，移出最早的数
        if len(self.q) > self.size:
            self.s -= self.q.popleft()
        return self.s / len(self.q)
```

复杂度：O(1)。

<br>

### 352. 将数据流变为多个不相交区间 / Data Stream as Disjoint Intervals

[CN](https://leetcode.cn/problems/data-stream-as-disjoint-intervals/) · [EN](https://leetcode.com/problems/data-stream-as-disjoint-intervals/) · Hard

**题意**：不断加入非负整数，`getIntervals()` 返回把已加入的数合并成的不相交区间（按左端点排序）。

**思路**：有序字典 `左端点 → 右端点`。加入 v 时找到左边相邻区间 `[l0, r0]` 和右边相邻区间 `[v+1, r1]`：v 已被覆盖就跳过；否则看能否和左边、右边、或两边合并。

```python
from sortedcontainers import SortedDict

class SummaryRanges:
    def __init__(self):
        self.sd = SortedDict()  # 1. 左端点 -> 右端点

    def addNum(self, v: int) -> None:
        sd = self.sd
        i = sd.bisect_right(v)  # 2. 左端点 > v 的第一个区间
        # 3. 左边相邻区间 [l0, r0]：已经覆盖 v 就直接返回
        left = i > 0
        if left:
            l0, r0 = sd.peekitem(i - 1)
            if r0 >= v:
                return
            left = r0 + 1 == v  # 能和左边接上
        right = v + 1 in sd     # 4. 能和右边 [v+1, r1] 接上
        # 5. 按四种情况合并
        if left and right:
            sd[l0] = sd.pop(v + 1)
        elif left:
            sd[l0] = v
        elif right:
            sd[v] = sd.pop(v + 1)
        else:
            sd[v] = v

    def getIntervals(self) -> List[List[int]]:
        return [[l, r] for l, r in self.sd.items()]
```

复杂度：`addNum` O(log n)；`getIntervals` O(n)。

<br>

### 901. 股票价格跨度 / Online Stock Span

[CN](https://leetcode.cn/problems/online-stock-span/) · [EN](https://leetcode.com/problems/online-stock-span/) · Medium

**题意**：每天给出股价，返回“从今天往前数，连续多少天的股价 ≤ 今天”（含今天）。

**思路**：找左边第一个比今天大的那天，跨度就是天数差。单调栈存 `(日期, 价格)`，价格从栈底到栈顶严格递减。今天的价格把栈顶所有 ≤ 它的弹掉（它们以后不可能再是“左边第一个更大的”），剩下的栈顶就是答案。栈底放一个价格为无穷大的哨兵。

```python
class StockSpanner:
    def __init__(self):
        self.st = [(-1, inf)]  # 1. 哨兵：(日期, 价格)，保证栈不空
        self.cur_day = -1      # 2. 第一个 next 调用算第 0 天

    def next(self, price: int) -> int:
        # 3. 弹出所有 <= 今天价格的天
        while price >= self.st[-1][1]:
            self.st.pop()
        self.cur_day += 1
        # 4. 栈顶是左边第一个更大的那天
        ans = self.cur_day - self.st[-1][0]
        self.st.append((self.cur_day, price))
        return ans
```

复杂度：均摊 O(1)。

<br>

### 2034. 股票价格波动 / Stock Price Fluctuation

[CN](https://leetcode.cn/problems/stock-price-fluctuation/) · [EN](https://leetcode.com/problems/stock-price-fluctuation/) · Medium

**题意**：`update(timestamp, price)` 记录或更正某时刻的价格；`current()` 返回最新时刻的价格；`maximum() / minimum()` 返回所有时刻（以更正后的为准）中的最高 / 最低价。

**思路**：哈希表 `时间 → 价格` 处理更正；有序集合存所有价格，更正时先删旧价格再加新价格，最大最小就是两端。也可以用两个堆 + 懒删除：堆里存 `(价格, 时间)`，堆顶的价格和哈希表里对不上就是过期数据，弹掉。

```python
from sortedcontainers import SortedList

class StockPrice:
    def __init__(self):
        self.price = {}           # 1. 时间 -> 价格
        self.prices = SortedList()  # 2. 所有当前有效的价格
        self.cur = 0              # 3. 最新时间戳

    def update(self, timestamp: int, price: int) -> None:
        # 4. 是更正：先删掉旧价格
        if timestamp in self.price:
            self.prices.remove(self.price[timestamp])
        self.prices.add(price)
        self.price[timestamp] = price
        self.cur = max(self.cur, timestamp)

    def current(self) -> int:
        return self.price[self.cur]

    def maximum(self) -> int:
        return self.prices[-1]

    def minimum(self) -> int:
        return self.prices[0]
```

懒删除堆的写法：

```python
class StockPrice:
    def __init__(self):
        self.price = {}
        self.cur = 0
        self.max_h = []  # 1. (-价格, 时间)
        self.min_h = []  # 2. (价格, 时间)

    def update(self, timestamp: int, price: int) -> None:
        # 3. 旧数据不删，留在堆里等查询时再清理
        self.price[timestamp] = price
        self.cur = max(self.cur, timestamp)
        heappush(self.max_h, (-price, timestamp))
        heappush(self.min_h, (price, timestamp))

    def current(self) -> int:
        return self.price[self.cur]

    def maximum(self) -> int:
        # 4. 堆顶价格和哈希表不一致，说明这个时刻被更正过，弹掉
        while -self.max_h[0][0] != self.price[self.max_h[0][1]]:
            heappop(self.max_h)
        return -self.max_h[0][0]

    def minimum(self) -> int:
        while self.min_h[0][0] != self.price[self.min_h[0][1]]:
            heappop(self.min_h)
        return self.min_h[0][0]
```

复杂度：都是 O(log n)（堆写法均摊）。

<br>

### 1352. 最后 K 个数的乘积 / Product of the Last K Numbers

[CN](https://leetcode.cn/problems/product-of-the-last-k-numbers/) · [EN](https://leetcode.com/problems/product-of-the-last-k-numbers/) · Medium

**题意**：`add(num)` 追加一个数，`getProduct(k)` 返回最后 k 个数的乘积。

**思路**：前缀积，最后 k 个数的乘积 = `pre[-1] // pre[-k-1]`。0 会让前缀积失效：遇到 0 就清空前缀积，只保留 0 之后的部分。如果 k 超过了 0 之后的长度，说明最后 k 个数里有 0，答案是 0。

```python
class ProductOfNumbers:
    def __init__(self):
        self.pre = [1]  # 1. 前缀积，pre[0] = 1 是哨兵

    def add(self, num: int) -> None:
        if num == 0:
            # 2. 遇到 0：之前的都作废
            self.pre = [1]
        else:
            self.pre.append(self.pre[-1] * num)

    def getProduct(self, k: int) -> int:
        # 3. 最后 k 个数跨过了 0
        if k >= len(self.pre):
            return 0
        return self.pre[-1] // self.pre[-k - 1]
```

复杂度：都是 O(1)。

<br>

### 933. 最近的请求次数 / Number of Recent Calls

[CN](https://leetcode.cn/problems/number-of-recent-calls/) · [EN](https://leetcode.com/problems/number-of-recent-calls/) · Easy

**题意**：`ping(t)` 记录时刻 t 的请求（t 严格递增），返回 `[t-3000, t]` 内的请求数。

**思路**：队列存请求时刻，每次把早于 `t-3000` 的出队，队列长度就是答案。

```python
class RecentCounter:
    def __init__(self):
        self.q = deque()

    def ping(self, t: int) -> int:
        self.q.append(t)
        # 1. 移出窗口外的旧请求
        while self.q[0] < t - 3000:
            self.q.popleft()
        return len(self.q)
```

复杂度：均摊 O(1)。

<br>

### 359. 日志速率限制器 / Logger Rate Limiter 🔒

[CN](https://leetcode.cn/problems/logger-rate-limiter/) · [EN](https://leetcode.com/problems/logger-rate-limiter/) · Easy

**题意**：同一条消息 10 秒内只打印一次。`shouldPrintMessage(timestamp, message)` 返回是否应该打印（时间戳非递减）。

**思路**：哈希表记录每条消息“下一次允许打印的时间”。

```python
class Logger:
    def __init__(self):
        self.next_time = {}  # 1. message -> 下一次允许打印的时间

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        # 2. 还没到允许时间
        if timestamp < self.next_time.get(message, 0):
            return False
        self.next_time[message] = timestamp + 10
        return True
```

复杂度：O(1)。消息种类很多时，可以再用队列按时间清理过期的 key，控制内存。

<br>

### 362. 敲击计数器 / Design Hit Counter 🔒

[CN](https://leetcode.cn/problems/design-hit-counter/) · [EN](https://leetcode.com/problems/design-hit-counter/) · Medium

**题意**：`hit(timestamp)` 记录一次敲击，`getHits(timestamp)` 返回过去 300 秒内（`(timestamp-300, timestamp]`）的敲击数。时间戳非递减。

**思路**：环形数组：300 个桶，桶 `t % 300` 存“时间 t 的敲击数”和“这个桶记录的是哪一秒”。时间对不上说明是旧数据，直接覆盖。这样同一秒大量敲击也只占一个桶，空间 O(300)。

```python
class HitCounter:
    def __init__(self):
        self.times = [0] * 300  # 1. 每个桶记录的是哪一秒
        self.hits = [0] * 300   # 2. 这一秒的敲击数

    def hit(self, timestamp: int) -> None:
        i = timestamp % 300
        # 3. 桶里是旧的时间，重置
        if self.times[i] != timestamp:
            self.times[i] = timestamp
            self.hits[i] = 0
        self.hits[i] += 1

    def getHits(self, timestamp: int) -> int:
        # 4. 只统计 300 秒内的桶
        return sum(h for t, h in zip(self.times, self.hits) if timestamp - t < 300)
```

复杂度：`hit` O(1)；`getHits` O(300)。用 `deque` 存时间戳、`getHits` 时出队过期的也可以，但同一秒大量敲击时空间更大。

<br>

### 1825. 求出 MK 平均值 / Finding MK Average

[CN](https://leetcode.cn/problems/finding-mk-average/) · [EN](https://leetcode.com/problems/finding-mk-average/) · Hard

**题意**：数据流中取最后 m 个数，去掉最小的 k 个和最大的 k 个，返回剩下数的平均值（向下取整）；不足 m 个返回 -1。保证 `3k < m`。

**思路**：队列保存最后 m 个数；三个有序集合 `lo`（最小的 k 个）、`mid`、`hi`（最大的 k 个），并维护 `mid` 的和 `s`。每次加入新数前先删掉窗口里最老的数，然后把新数放进对应的集合，最后在三个集合之间搬运，恢复 `len(lo) == len(hi) == k`。

```python
from sortedcontainers import SortedList

class MKAverage:
    def __init__(self, m: int, k: int):
        self.m, self.k = m, k
        self.q = deque()
        self.lo, self.mid, self.hi = SortedList(), SortedList(), SortedList()
        self.s = 0  # 1. mid 的元素和

    def addElement(self, num: int) -> None:
        m, k = self.m, self.k
        lo, mid, hi = self.lo, self.mid, self.hi
        self.q.append(num)
        if len(self.q) < m:
            return
        if len(self.q) == m and not mid:
            # 2. 第一次凑满 m 个：排序后切成三段
            a = sorted(self.q)
            lo.update(a[:k])
            mid.update(a[k:m - k])
            hi.update(a[m - k:])
            self.s = sum(mid)
            return
        # 3. 删掉最老的数（等于 lo 最大值时从 lo 删，等于 hi 最小值时从 hi 删）
        x = self.q.popleft()
        if x <= lo[-1]:
            lo.remove(x)
        elif x >= hi[0]:
            hi.remove(x)
        else:
            mid.remove(x)
            self.s -= x
        # 4. 放入新数
        if lo and num < lo[-1]:
            lo.add(num)
        elif hi and num > hi[0]:
            hi.add(num)
        else:
            mid.add(num)
            self.s += num
        # 5. 恢复 lo、hi 各 k 个：先处理多出来的，再处理不够的
        if len(lo) > k:
            x = lo.pop()
            mid.add(x)
            self.s += x
        if len(hi) > k:
            x = hi.pop(0)
            mid.add(x)
            self.s += x
        if len(lo) < k:
            x = mid.pop(0)
            lo.add(x)
            self.s -= x
        if len(hi) < k:
            x = mid.pop()
            hi.add(x)
            self.s -= x

    def calculateMKAverage(self) -> int:
        if len(self.q) < self.m:
            return -1
        return self.s // (self.m - 2 * self.k)
```

复杂度：`addElement` O(log m)；`calculateMKAverage` O(1)。

<br>

### 2526. 找到数据流中的连续整数 / Find Consecutive Integers from a Data Stream

[CN](https://leetcode.cn/problems/find-consecutive-integers-from-a-data-stream/) · [EN](https://leetcode.com/problems/find-consecutive-integers-from-a-data-stream/) · Medium

**题意**：给定 `value` 和 `k`，每次 `consec(num)` 加入一个数，返回最后 k 个数是否都等于 `value`。

**思路**：只需要记录“末尾连续等于 value 的个数”。

```python
class DataStream:
    def __init__(self, value: int, k: int):
        self.value = value
        self.k = k
        self.cnt = 0  # 1. 末尾连续等于 value 的个数

    def consec(self, num: int) -> bool:
        # 2. 相等就加一，否则清零
        self.cnt = self.cnt + 1 if num == self.value else 0
        return self.cnt >= self.k
```

复杂度：O(1)。

<br>

### 1429. 第一个唯一数字 / First Unique Number 🔒

[CN](https://leetcode.cn/problems/first-unique-number/) · [EN](https://leetcode.com/problems/first-unique-number/) · Medium

**题意**：初始给一个数组，`add(value)` 追加一个数，`showFirstUnique()` 返回队列中第一个只出现一次的数，没有返回 -1。

**思路**：计数表 + 队列。队列按加入顺序存数，`showFirstUnique` 时把队首出现次数 > 1 的数弹掉（懒删除）：一个数一旦重复，就再也不会变回唯一。

```python
class FirstUnique:
    def __init__(self, nums: List[int]):
        self.cnt = Counter()  # 1. 每个数出现的次数
        self.q = deque()      # 2. 按加入顺序存放的候选
        for x in nums:
            self.add(x)

    def showFirstUnique(self) -> int:
        # 3. 懒删除：队首已经重复了就弹掉
        while self.q and self.cnt[self.q[0]] > 1:
            self.q.popleft()
        return self.q[0] if self.q else -1

    def add(self, value: int) -> None:
        self.cnt[value] += 1
        # 4. 只有第一次出现才进队列
        if self.cnt[value] == 1:
            self.q.append(value)
```

复杂度：均摊 O(1)。

<br>
<br>

## 6. 区间与范围查询

先看“数组改不改”：不修改用前缀和（303、304）；单点修改 + 区间查询用树状数组（307、308）；区间修改 / 动态值域用线段树（732、715）；区间不重叠、合并用有序集合（729、2276）。按“时间 / 版本”查询则是每个 key 存一个按时间递增的列表，查询时二分（981、1146）。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 303 | 区域和检索 - 数组不可变 | [CN](https://leetcode.cn/problems/range-sum-query-immutable/) · [EN](https://leetcode.com/problems/range-sum-query-immutable/) | 前缀和 |
| 304 | 二维区域和检索 - 矩阵不可变 | [CN](https://leetcode.cn/problems/range-sum-query-2d-immutable/) · [EN](https://leetcode.com/problems/range-sum-query-2d-immutable/) | 二维前缀和 |
| 307 | 区域和检索 - 数组可修改 | [CN](https://leetcode.cn/problems/range-sum-query-mutable/) · [EN](https://leetcode.com/problems/range-sum-query-mutable/) | 树状数组 |
| 308 🔒 | 二维区域和检索 - 矩阵可修改 | [CN](https://leetcode.cn/problems/range-sum-query-2d-mutable/) · [EN](https://leetcode.com/problems/range-sum-query-2d-mutable/) | 二维树状数组 |
| 729 | 我的日程安排表 I | [CN](https://leetcode.cn/problems/my-calendar-i/) · [EN](https://leetcode.com/problems/my-calendar-i/) | 有序集合 + 二分 |
| 731 | 我的日程安排表 II | [CN](https://leetcode.cn/problems/my-calendar-ii/) · [EN](https://leetcode.com/problems/my-calendar-ii/) | 差分（有序字典） |
| 732 | 我的日程安排表 III | [CN](https://leetcode.cn/problems/my-calendar-iii/) · [EN](https://leetcode.com/problems/my-calendar-iii/) | 差分 / 动态开点线段树 |
| 715 | Range 模块 | [CN](https://leetcode.cn/problems/range-module/) · [EN](https://leetcode.com/problems/range-module/) | 有序区间合并 |
| 2276 | 统计区间中的整数数目 | [CN](https://leetcode.cn/problems/count-integers-in-intervals/) · [EN](https://leetcode.com/problems/count-integers-in-intervals/) | 有序集合 / 动态开点线段树 |
| 981 | 基于时间的键值存储 | [CN](https://leetcode.cn/problems/time-based-key-value-store/) · [EN](https://leetcode.com/problems/time-based-key-value-store/) | 哈希表 + 二分 |
| 1146 | 快照数组 | [CN](https://leetcode.cn/problems/snapshot-array/) · [EN](https://leetcode.com/problems/snapshot-array/) | 每个位置的版本列表 + 二分 |
| 1157 | 子数组中占绝大多数的元素 | [CN](https://leetcode.cn/problems/online-majority-element-in-subarray/) · [EN](https://leetcode.com/problems/online-majority-element-in-subarray/) | 位置列表 + 随机化 |
| 2080 | 区间内查询数字的频率 | [CN](https://leetcode.cn/problems/range-frequency-queries/) · [EN](https://leetcode.com/problems/range-frequency-queries/) | 位置列表 + 二分 |
| 635 🔒 | 设计日志存储系统 | [CN](https://leetcode.cn/problems/design-log-storage-system/) · [EN](https://leetcode.com/problems/design-log-storage-system/) | 截断时间戳 + 字符串比较 |
| 1348 | 推文计数 | [CN](https://leetcode.cn/problems/tweet-counts-per-frequency/) · [EN](https://leetcode.com/problems/tweet-counts-per-frequency/) | 哈希表 + 有序列表 + 二分 |

<br>

### 303. 区域和检索 - 数组不可变 / Range Sum Query - Immutable

[CN](https://leetcode.cn/problems/range-sum-query-immutable/) · [EN](https://leetcode.com/problems/range-sum-query-immutable/) · Easy

**题意**：数组不变，多次查询 `sumRange(left, right)`（闭区间）。

**思路**：前缀和 `s[i] = nums[0] + ... + nums[i-1]`，`s[0] = 0`。区间和 = `s[right+1] - s[left]`。多开一位让 `left = 0` 时不用特判。

```python
class NumArray:
    def __init__(self, nums: List[int]):
        # 1. s[i+1] = nums[0] + ... + nums[i]
        self.s = list(accumulate(nums, initial=0))

    def sumRange(self, left: int, right: int) -> int:
        # 2. [left, right] 的和 = s[right+1] - s[left]
        return self.s[right + 1] - self.s[left]
```

复杂度：构造 O(n)；查询 O(1)。

<br>

### 304. 二维区域和检索 - 矩阵不可变 / Range Sum Query 2D - Immutable

[CN](https://leetcode.cn/problems/range-sum-query-2d-immutable/) · [EN](https://leetcode.com/problems/range-sum-query-2d-immutable/) · Medium

**题意**：矩阵不变，多次查询子矩形 `(row1, col1)` 到 `(row2, col2)` 的元素和。

**思路**：二维前缀和 `s[i+1][j+1]` = 左上角 `(0,0)` 到 `(i,j)` 的和。递推和查询都用容斥。

```python
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        s = [[0] * (n + 1) for _ in range(m + 1)]
        for i, row in enumerate(matrix):
            for j, x in enumerate(row):
                # 1. 上 + 左 - 左上（被加了两次） + 自己
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + x
        self.s = s

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        s = self.s
        # 2. 大矩形 - 上面 - 左边 + 左上（被减了两次）
        return s[row2 + 1][col2 + 1] - s[row2 + 1][col1] - s[row1][col2 + 1] + s[row1][col1]
```

复杂度：构造 O(mn)；查询 O(1)。

<br>

### 307. 区域和检索 - 数组可修改 / Range Sum Query - Mutable

[CN](https://leetcode.cn/problems/range-sum-query-mutable/) · [EN](https://leetcode.com/problems/range-sum-query-mutable/) · Medium

**题意**：支持单点修改 `update(index, val)` 和区间求和 `sumRange(left, right)`。

**思路**：树状数组（Fenwick Tree）。`tree[i]` 管理一段长度为 `lowbit(i) = i & -i` 的区间。单点加 和 前缀和 都只走 O(log n) 个节点。`update` 是“改成 val”，所以加的是差值 `val - nums[index]`。构造时用 O(n) 建树：每个节点把自己的值加到父节点 `i + lowbit(i)` 上。

```python
class NumArray:
    def __init__(self, nums: List[int]):
        n = len(nums)
        tree = [0] * (n + 1)  # 1. 下标从 1 开始
        for i, x in enumerate(nums, 1):
            tree[i] += x
            # 2. O(n) 建树：把自己加到父节点
            nxt = i + (i & -i)
            if nxt <= n:
                tree[nxt] += tree[i]
        self.nums = nums
        self.tree = tree

    def update(self, index: int, val: int) -> None:
        delta = val - self.nums[index]
        self.nums[index] = val
        # 3. 单点加：沿 i += lowbit(i) 往上更新所有覆盖它的节点
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    # 4. 前缀和 nums[0] + ... + nums[i-1]：沿 i -= lowbit(i) 往下累加
    def prefix_sum(self, i: int) -> int:
        s = 0
        while i:
            s += self.tree[i]
            i &= i - 1  # 等价于 i -= i & -i
        return s

    def sumRange(self, left: int, right: int) -> int:
        return self.prefix_sum(right + 1) - self.prefix_sum(left)
```

复杂度：构造 O(n)；`update / sumRange` O(log n)。

<br>

### 308. 二维区域和检索 - 矩阵可修改 / Range Sum Query 2D - Mutable 🔒

[CN](https://leetcode.cn/problems/range-sum-query-2d-mutable/) · [EN](https://leetcode.com/problems/range-sum-query-2d-mutable/) · Hard

**题意**：304 加上单点修改。

**思路**：二维树状数组：两层 lowbit 循环，查询用二维前缀和的容斥。

```python
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        self.m, self.n = m, n
        self.a = [[0] * n for _ in range(m)]
        self.tree = [[0] * (n + 1) for _ in range(m + 1)]
        # 1. 逐个 update 建树
        for i in range(m):
            for j in range(n):
                self.update(i, j, matrix[i][j])

    def update(self, row: int, col: int, val: int) -> None:
        delta = val - self.a[row][col]
        self.a[row][col] = val
        # 2. 两层 lowbit：行方向、列方向都往上走
        i = row + 1
        while i <= self.m:
            j = col + 1
            while j <= self.n:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i

    # 3. 左上角 (0,0) 到 (r-1,c-1) 的和
    def query(self, r: int, c: int) -> int:
        s = 0
        i = r
        while i:
            j = c
            while j:
                s += self.tree[i][j]
                j &= j - 1
            i &= i - 1
        return s

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        # 4. 容斥
        return (self.query(row2 + 1, col2 + 1) - self.query(row1, col2 + 1)
                - self.query(row2 + 1, col1) + self.query(row1, col1))
```

复杂度：`update / sumRegion` O(log m · log n)。

<br>

### 729. 我的日程安排表 I / My Calendar I

[CN](https://leetcode.cn/problems/my-calendar-i/) · [EN](https://leetcode.com/problems/my-calendar-i/) · Medium

**题意**：`book(start, end)` 预订半开区间 `[start, end)`，和已有预订重叠就返回 False，否则加入并返回 True。

**思路**：已有预订互不重叠，按开始时间排序后，结束时间也是有序的。有序集合里存 `(end, start)`，二分找第一个 `end > start` 的预订：只有它可能和新区间重叠，检查它的 `start < end` 即可。

```python
from sortedcontainers import SortedList

class MyCalendar:
    def __init__(self):
        self.sl = SortedList()  # 1. 存 (end, start)，按 end 排序

    def book(self, start: int, end: int) -> bool:
        # 2. 第一个 end' > start 的预订
        i = self.sl.bisect_right((start, inf))
        # 3. 它的 start' < end 就重叠了
        if i < len(self.sl) and self.sl[i][1] < end:
            return False
        self.sl.add((end, start))
        return True
```

复杂度：`book` O(log n)。

<br>

### 731. 我的日程安排表 II / My Calendar II

[CN](https://leetcode.cn/problems/my-calendar-ii/) · [EN](https://leetcode.com/problems/my-calendar-ii/) · Medium

**题意**：允许两重预订，但不能出现三重预订。

**思路**：差分。有序字典 `diff[start] += 1, diff[end] -= 1`，按时间顺序累加就得到每个时刻的预订数。先加进去，如果最大值 ≥ 3 就撤回。调用次数 ≤ 1000，O(n) 扫一遍可以接受。（也可以维护“单重区间”和“双重区间”两个列表，O(n) 判断。）

```python
from sortedcontainers import SortedDict

class MyCalendarTwo:
    def __init__(self):
        self.diff = SortedDict()  # 1. 时间 -> 差分值

    def book(self, start: int, end: int) -> bool:
        d = self.diff
        # 2. 先尝试加入
        d[start] = d.get(start, 0) + 1
        d[end] = d.get(end, 0) - 1
        # 3. 按时间累加，任何时刻 >= 3 就失败
        cur = 0
        for v in d.values():
            cur += v
            if cur >= 3:
                # 4. 撤回刚才的修改
                d[start] -= 1
                d[end] += 1
                return False
        return True
```

复杂度：`book` O(n)。

<br>

### 732. 我的日程安排表 III / My Calendar III

[CN](https://leetcode.cn/problems/my-calendar-iii/) · [EN](https://leetcode.com/problems/my-calendar-iii/) · Hard

**题意**：每次 `book(start, end)` 后，返回当前所有预订的最大重叠数 k。

**思路**：同 731 的差分，每次扫一遍求最大前缀和，O(n)。进阶是动态开点线段树：区间加 1、查全局最大值，每次 O(log C)，下面给出线段树写法（`C = 10^9`，用字典开点 + 懒标记）。

```python
class MyCalendarThree:
    def __init__(self):
        self.mx = defaultdict(int)    # 1. 节点区间内的最大重叠数
        self.lazy = defaultdict(int)  # 2. 懒标记：整个区间都加了多少

    # 3. 给 [l, r] 区间加 1；当前节点 o 管理 [s, e]
    def update(self, o: int, s: int, e: int, l: int, r: int) -> None:
        if l <= s and e <= r:
            # 4. 完全覆盖：打懒标记，不再往下
            self.mx[o] += 1
            self.lazy[o] += 1
            return
        m = (s + e) // 2
        if l <= m:
            self.update(o * 2, s, m, l, r)
        if m < r:
            self.update(o * 2 + 1, m + 1, e, l, r)
        # 5. 自己的最大值 = 子节点最大值 + 自己的懒标记（标记永久化，不下传）
        self.mx[o] = max(self.mx[o * 2], self.mx[o * 2 + 1]) + self.lazy[o]

    def book(self, startTime: int, endTime: int) -> int:
        self.update(1, 0, 10 ** 9, startTime, endTime - 1)
        return self.mx[1]
```

用 `o * 2` 编号在字典里不会冲突，深度只有 30 层左右，所以节点编号不会溢出（Python 整数也不会溢出）。

复杂度：`book` O(log C)；空间 O(n log C)。

<br>

### 715. Range 模块 / Range Module

[CN](https://leetcode.cn/problems/range-module/) · [EN](https://leetcode.com/problems/range-module/) · Hard

**题意**：跟踪一组半开区间。`addRange(l, r)` 加入、`removeRange(l, r)` 删除、`queryRange(l, r)` 判断 `[l, r)` 是否被完全覆盖。

**思路**：维护一组互不相交、按左端点排序的区间。用一个有序列表存所有端点 `[l0, r0, l1, r1, ...]`，那么“某个数 x 被覆盖”等价于 `bisect_right(x)` 是奇数。加入、删除区间就是把 `[l, r)` 范围内的端点整段替换掉。

```python
class RangeModule:
    def __init__(self):
        self.a = []  # 1. 端点列表：[l0, r0, l1, r1, ...]，偶数位是左端点

    def addRange(self, left: int, right: int) -> None:
        i = bisect_left(self.a, left)
        j = bisect_right(self.a, right)
        # 2. 被替换的这段端点里，left 在区间外（i 是偶数）才需要加入 left 作为新左端点
        #    right 在区间外（j 是偶数）才需要加入 right 作为新右端点
        mid = []
        if i % 2 == 0:
            mid.append(left)
        if j % 2 == 0:
            mid.append(right)
        self.a[i:j] = mid

    def removeRange(self, left: int, right: int) -> None:
        i = bisect_left(self.a, left)
        j = bisect_right(self.a, right)
        # 3. left 在某个区间内部（i 是奇数），把它变成那个区间的新右端点
        #    right 在某个区间内部（j 是奇数），把它变成新左端点
        mid = []
        if i % 2 == 1:
            mid.append(left)
        if j % 2 == 1:
            mid.append(right)
        self.a[i:j] = mid

    def queryRange(self, left: int, right: int) -> bool:
        # 4. left 和 right 落在同一个区间 [a[i], a[i+1]] 里
        i = bisect_right(self.a, left)
        j = bisect_left(self.a, right)
        return i == j and i % 2 == 1
```

复杂度：切片替换 O(n)，但常数极小；严格 O(log n) 可以用动态开点线段树。

<br>

### 2276. 统计区间中的整数数目 / Count Integers in Intervals

[CN](https://leetcode.cn/problems/count-integers-in-intervals/) · [EN](https://leetcode.com/problems/count-integers-in-intervals/) · Hard

**题意**：`add(left, right)` 加入闭区间，`count()` 返回至少被一个区间覆盖的整数个数。

**思路**（灵神写法之一）：有序字典 `右端点 → 左端点`，维护互不相交的区间和覆盖总数 `cnt`。加入 `[left, right]` 时，找出所有和它相交的区间（右端点 ≥ left、左端点 ≤ right），把它们删掉、合并成一个大区间。每个区间最多被删一次，均摊 O(log n)。

```python
from sortedcontainers import SortedDict

class CountIntervals:
    def __init__(self):
        self.d = SortedDict()  # 1. 右端点 -> 左端点
        self.cnt = 0           # 2. 所有区间覆盖的整数个数

    def add(self, left: int, right: int) -> None:
        d = self.d
        # 3. 从第一个右端点 >= left 的区间开始，左端点 <= right 就相交
        i = d.bisect_left(left)
        while i < len(d) and d.peekitem(i)[1] <= right:
            r, l = d.popitem(i)
            # 4. 合并：扩大 [left, right]，减掉被合并区间的贡献
            left = min(left, l)
            right = max(right, r)
            self.cnt -= r - l + 1
        self.cnt += right - left + 1
        d[right] = left

    def count(self) -> int:
        return self.cnt
```

复杂度：`add` 均摊 O(log n)；`count` O(1)。

<br>

### 981. 基于时间的键值存储 / Time Based Key-Value Store

[CN](https://leetcode.cn/problems/time-based-key-value-store/) · [EN](https://leetcode.com/problems/time-based-key-value-store/) · Medium

**题意**：`set(key, value, timestamp)` 存一个带时间戳的值（同一个 key 的时间戳严格递增）；`get(key, timestamp)` 返回时间戳 ≤ timestamp 的最新值，没有返回 `""`。

**思路**：每个 key 存一个按时间递增的列表 `[(timestamp, value)]`。查询时二分找最后一个时间戳 ≤ timestamp 的位置。

```python
class TimeMap:
    def __init__(self):
        self.d = defaultdict(list)  # 1. key -> [(timestamp, value)]，timestamp 递增

    def set(self, key: str, value: str, timestamp: int) -> None:
        # 2. 时间戳严格递增，直接追加就是有序的
        self.d[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        a = self.d[key]
        # 3. 第一个时间戳 > timestamp 的位置，前一个就是答案
        #    用 chr(127) 当 value，保证 (timestamp, 任何 value) 都排在它前面
        i = bisect_right(a, (timestamp, chr(127)))
        return a[i - 1][1] if i else ""
```

复杂度：`set` O(1)；`get` O(log n)。

<br>

### 1146. 快照数组 / Snapshot Array

[CN](https://leetcode.cn/problems/snapshot-array/) · [EN](https://leetcode.com/problems/snapshot-array/) · Medium

**题意**：长度为 n 的数组，`set(index, val)` 修改；`snap()` 拍快照并返回快照编号；`get(index, snap_id)` 返回某次快照时该位置的值。

**思路**：不要每次快照都复制数组。每个位置只记录它被修改的历史 `[(snap_id, val)]`。查询 `(index, snap_id)` 时，二分找最后一个 `snap_id' <= snap_id` 的记录。

```python
class SnapshotArray:
    def __init__(self, length: int):
        self.cur_snap_id = 0
        self.history = defaultdict(list)  # 1. 每个 index 的历史修改记录 [(snap_id, val)]

    def set(self, index: int, val: int) -> None:
        # 2. 同一个快照内多次修改，也直接追加（查询时取最后一个）
        self.history[index].append((self.cur_snap_id, val))

    def snap(self) -> int:
        self.cur_snap_id += 1
        return self.cur_snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        h = self.history[index]
        # 3. 找第一个 >= snap_id + 1 的记录，前一个就是 snap_id 时的值
        j = bisect_left(h, (snap_id + 1,)) - 1
        return h[j][1] if j >= 0 else 0
```

复杂度：`set / snap` O(1)；`get` O(log q)。

<br>

### 1157. 子数组中占绝大多数的元素 / Online Majority Element In Subarray

[CN](https://leetcode.cn/problems/online-majority-element-in-subarray/) · [EN](https://leetcode.com/problems/online-majority-element-in-subarray/) · Hard

**题意**：`query(left, right, threshold)` 返回 `arr[left..right]` 中出现次数 ≥ threshold 的元素，保证 `2 * threshold > 区间长度`；没有返回 -1。

**思路**：随机化。满足条件的元素占区间一半以上，随机抽一个位置，抽中它的概率 > 1/2。抽 20 次都没抽中的概率 < 2^-20。对抽中的值，用“值 → 下标列表”二分统计它在区间内的出现次数。

```python
class MajorityChecker:
    K = 20  # 1. 随机次数，失败概率 < 2^-20

    def __init__(self, arr: List[int]):
        self.arr = arr
        self.pos = defaultdict(list)  # 2. 值 -> 出现的下标（递增）
        for i, x in enumerate(arr):
            self.pos[x].append(i)

    def query(self, left: int, right: int, threshold: int) -> int:
        for _ in range(self.K):
            # 3. 随机抽一个位置的值
            x = self.arr[randint(left, right)]
            p = self.pos[x]
            # 4. 二分统计 x 在 [left, right] 中出现的次数
            cnt = bisect_right(p, right) - bisect_left(p, left)
            if cnt >= threshold:
                return x
            # 5. 剪枝：出现次数超过一半却不够 threshold，就不可能有答案了
            if cnt * 2 >= right - left + 1:
                break
        return -1
```

复杂度：`query` O(K log n)。确定性做法是线段树维护摩尔投票（候选 + 计数），再二分验证。

<br>

### 2080. 区间内查询数字的频率 / Range Frequency Queries

[CN](https://leetcode.cn/problems/range-frequency-queries/) · [EN](https://leetcode.com/problems/range-frequency-queries/) · Medium

**题意**：`query(left, right, value)` 返回 `value` 在 `arr[left..right]` 中出现的次数。

**思路**：预处理“值 → 出现的下标列表”，查询时二分出下标落在 `[left, right]` 的个数。

```python
class RangeFreqQuery:
    def __init__(self, arr: List[int]):
        self.pos = defaultdict(list)  # 1. 值 -> 递增的下标列表
        for i, x in enumerate(arr):
            self.pos[x].append(i)

    def query(self, left: int, right: int, value: int) -> int:
        a = self.pos[value]
        # 2. 下标在 [left, right] 内的个数
        return bisect_right(a, right) - bisect_left(a, left)
```

复杂度：构造 O(n)；查询 O(log n)。

<br>

### 635. 设计日志存储系统 / Design Log Storage System 🔒

[CN](https://leetcode.cn/problems/design-log-storage-system/) · [EN](https://leetcode.com/problems/design-log-storage-system/) · Medium

**题意**：时间戳格式 `Year:Month:Day:Hour:Minute:Second`（各段补零定长）。`put(id, timestamp)` 存日志；`retrieve(start, end, granularity)` 按粒度（如 `"Day"`）返回时间在 `[start, end]` 内的日志 id。

**思路**：定长字符串可以直接按字典序比较。按粒度截断前缀：`start` 截到粒度为止、`end` 也截到粒度为止，比较日志时间戳的同长度前缀即可。日志多时可以存有序列表再二分。

```python
class LogSystem:
    # 1. 每个粒度对应的前缀长度："2017:01:01:23:59:59"
    LEN = {"Year": 4, "Month": 7, "Day": 10, "Hour": 13, "Minute": 16, "Second": 19}

    def __init__(self):
        self.logs = []  # 2. (timestamp, id)

    def put(self, id: int, timestamp: str) -> None:
        self.logs.append((timestamp, id))

    def retrieve(self, start: str, end: str, granularity: str) -> List[int]:
        k = self.LEN[granularity]
        s, e = start[:k], end[:k]
        # 3. 截断到同一粒度后，按字符串比较
        return [i for t, i in self.logs if s <= t[:k] <= e]
```

复杂度：`put` O(1)；`retrieve` O(n)。

<br>

### 1348. 推文计数 / Tweet Counts Per Frequency

[CN](https://leetcode.cn/problems/tweet-counts-per-frequency/) · [EN](https://leetcode.com/problems/tweet-counts-per-frequency/) · Medium

**题意**：`recordTweet(name, time)` 记录推文；`getTweetCountsPerFrequency(freq, name, start, end)` 把 `[start, end]` 按每分钟 / 每小时 / 每天切块，返回每块内的推文数。

**思路**：每个 name 存一个有序列表（时间可能乱序到来，用 `SortedList` 或 `insort`）。查询时对每个块 `[l, min(l + size - 1, end)]` 二分计数。

```python
from sortedcontainers import SortedList

class TweetCounts:
    SIZE = {"minute": 60, "hour": 3600, "day": 86400}  # 1. 块大小

    def __init__(self):
        self.d = defaultdict(SortedList)  # 2. name -> 有序时间列表

    def recordTweet(self, tweetName: str, time: int) -> None:
        self.d[tweetName].add(time)

    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        size = self.SIZE[freq]
        a = self.d[tweetName]
        ans = []
        # 3. 每个块 [l, r] 二分计数
        for l in range(startTime, endTime + 1, size):
            r = min(l + size - 1, endTime)
            ans.append(a.bisect_right(r) - a.bisect_left(l))
        return ans
```

复杂度：`recordTweet` O(log n)；查询 O(块数 · log n)。

<br>
<br>

## 7. 字符串 / 字典树

Trie 的每个节点代表一个前缀，`son` 指向下一个字符，`end` 标记“有单词在这里结束”。题目的变化都在“节点上还存什么”：前缀和（677）、热度 Top 3（642）、文件内容（588）、单词下标（745）。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 208 | 实现 Trie（前缀树） | [CN](https://leetcode.cn/problems/implement-trie-prefix-tree/) · [EN](https://leetcode.com/problems/implement-trie-prefix-tree/) | Trie |
| 211 | 添加与搜索单词 - 数据结构设计 | [CN](https://leetcode.cn/problems/design-add-and-search-words-data-structure/) · [EN](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | Trie + DFS（通配符） |
| 676 | 实现一个魔法字典 | [CN](https://leetcode.cn/problems/implement-magic-dictionary/) · [EN](https://leetcode.com/problems/implement-magic-dictionary/) | Trie + DFS（恰好改一次） |
| 677 | 键值映射 | [CN](https://leetcode.cn/problems/map-sum-pairs/) · [EN](https://leetcode.com/problems/map-sum-pairs/) | Trie 节点里存前缀和 |
| 745 | 前缀和后缀搜索 | [CN](https://leetcode.cn/problems/prefix-and-suffix-search/) · [EN](https://leetcode.com/problems/prefix-and-suffix-search/) | 组合键 `后缀#单词` 的 Trie |
| 1032 | 字符流 | [CN](https://leetcode.cn/problems/stream-of-characters/) · [EN](https://leetcode.com/problems/stream-of-characters/) | 反向 Trie |
| 642 🔒 | 设计搜索自动补全系统 | [CN](https://leetcode.cn/problems/design-search-autocomplete-system/) · [EN](https://leetcode.com/problems/design-search-autocomplete-system/) | Trie + 排序 |
| 588 🔒 | 设计内存文件系统 | [CN](https://leetcode.cn/problems/design-in-memory-file-system/) · [EN](https://leetcode.com/problems/design-in-memory-file-system/) | 按路径分段的 Trie |
| 1166 🔒 | 设计文件系统 | [CN](https://leetcode.cn/problems/design-file-system/) · [EN](https://leetcode.com/problems/design-file-system/) | 哈希表（路径 → 值） |

<br>

### 208. 实现 Trie（前缀树） / Implement Trie (Prefix Tree)

[CN](https://leetcode.cn/problems/implement-trie-prefix-tree/) · [EN](https://leetcode.com/problems/implement-trie-prefix-tree/) · Medium

**题意**：实现 `insert(word)`、`search(word)`（完整单词是否存在）、`startsWith(prefix)`（是否有单词以它开头）。

**思路**：`search` 和 `startsWith` 都是沿着字符往下走，区别只在于最后是否要求 `end` 为真。灵神的写法把它们合并成一个 `find`：返回 0 表示走不下去，1 表示是前缀，2 表示是完整单词。

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
            # 1. 没有这条边就新建
            if c not in cur.son:
                cur.son[c] = Node()
            cur = cur.son[c]
        # 2. 标记单词结尾
        cur.end = True

    # 3. 0：不存在；1：是某个单词的前缀；2：是完整单词
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

复杂度：每次操作 O(L)，L 是字符串长度。

<br>

### 211. 添加与搜索单词 - 数据结构设计 / Design Add and Search Words Data Structure

[CN](https://leetcode.cn/problems/design-add-and-search-words-data-structure/) · [EN](https://leetcode.com/problems/design-add-and-search-words-data-structure/) · Medium

**题意**：`addWord(word)` 加入单词；`search(word)` 判断是否存在匹配的单词，其中 `.` 可以匹配任意一个字母。

**思路**：Trie + DFS。普通字符沿着唯一的边走；遇到 `.` 就尝试所有子节点，有一个成功就返回 True。

```python
class WordDictionary:
    def __init__(self):
        self.root = {}  # 1. 用字典当节点，'#' 键表示单词结尾

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            cur = cur.setdefault(c, {})
        cur['#'] = True

    def search(self, word: str) -> bool:
        # 2. dfs(i, node)：从 node 出发能否匹配 word[i:]
        def dfs(i: int, node: dict) -> bool:
            if i == len(word):
                return '#' in node
            c = word[i]
            if c != '.':
                return c in node and dfs(i + 1, node[c])
            # 3. 通配符：枚举所有子节点（跳过结尾标记）
            return any(dfs(i + 1, child) for k, child in node.items() if k != '#')
        return dfs(0, self.root)
```

复杂度：`addWord` O(L)；`search` 最坏 O(26^(.的个数) · L)，题目保证 `.` 至多 2 个。

<br>

### 676. 实现一个魔法字典 / Implement Magic Dictionary

[CN](https://leetcode.cn/problems/implement-magic-dictionary/) · [EN](https://leetcode.com/problems/implement-magic-dictionary/) · Medium

**题意**：`buildDict(words)` 建字典；`search(word)` 判断能否**恰好**修改 word 的一个字母，使它变成字典里的单词。

**思路**：Trie + DFS，多带一个参数 `changed`：沿原字符走不消耗修改次数；走其它字符要消耗唯一的一次修改。走到结尾时必须恰好改过一次。

```python
class MagicDictionary:
    def __init__(self):
        self.root = {}

    def buildDict(self, dictionary: List[str]) -> None:
        for w in dictionary:
            cur = self.root
            for c in w:
                cur = cur.setdefault(c, {})
            cur['#'] = True

    def search(self, searchWord: str) -> bool:
        # 1. dfs(i, node, changed)：已经改过 changed 次
        def dfs(i: int, node: dict, changed: bool) -> bool:
            if i == len(searchWord):
                # 2. 必须恰好改一次，并且是完整单词
                return changed and '#' in node
            c = searchWord[i]
            # 3. 不修改这个字母
            if c in node and dfs(i + 1, node[c], changed):
                return True
            # 4. 修改这个字母（只能改一次）
            if not changed:
                for k, child in node.items():
                    if k != c and k != '#' and dfs(i + 1, child, True):
                        return True
            return False
        return dfs(0, self.root, False)
```

复杂度：`search` O(26 · L^2) 量级。也可以按长度分组，直接逐个比较“恰好一个字母不同”。

<br>

### 677. 键值映射 / Map Sum Pairs

[CN](https://leetcode.cn/problems/map-sum-pairs/) · [EN](https://leetcode.com/problems/map-sum-pairs/) · Medium

**题意**：`insert(key, val)` 插入或覆盖；`sum(prefix)` 返回所有以 prefix 开头的 key 的值之和。

**思路**：每个 Trie 节点存“经过这个节点的所有 key 的值之和”。插入时，沿路每个节点加上 `delta = val - 旧值`（覆盖时只加差值）。查询时走到前缀节点直接返回。

```python
class MapSum:
    def __init__(self):
        self.root = {}
        self.vals = {}  # 1. key -> 当前值，用来算差值

    def insert(self, key: str, val: int) -> None:
        # 2. 覆盖时只加差值
        delta = val - self.vals.get(key, 0)
        self.vals[key] = val
        cur = self.root
        for c in key:
            cur = cur.setdefault(c, {'#': 0})
            # 3. 沿路每个前缀节点都加上 delta
            cur['#'] += delta

    def sum(self, prefix: str) -> int:
        cur = self.root
        for c in prefix:
            if c not in cur:
                return 0
            cur = cur[c]
        return cur['#']
```

复杂度：都是 O(L)。

<br>

### 745. 前缀和后缀搜索 / Prefix and Suffix Search

[CN](https://leetcode.cn/problems/prefix-and-suffix-search/) · [EN](https://leetcode.com/problems/prefix-and-suffix-search/) · Hard

**题意**：给定单词列表，`f(pref, suff)` 返回同时有该前缀和后缀的单词的最大下标，没有返回 -1。单词长度 ≤ 7。

**思路**：把 `后缀 + '#' + 单词` 的所有组合插入一个 Trie（单词很短，每个单词只有 8 种后缀），每个节点记录经过它的最大下标。查询时搜索 `suff + '#' + pref`。

```python
class WordFilter:
    def __init__(self, words: List[str]):
        self.root = {}
        for idx, w in enumerate(words):
            # 1. 枚举所有后缀（包括空后缀），插入 "后缀#单词"
            for i in range(len(w) + 1):
                cur = self.root
                for c in w[i:] + '#' + w:
                    cur = cur.setdefault(c, {})
                    # 2. 下标递增，后插入的覆盖，节点上就是最大下标
                    cur['idx'] = idx

    def f(self, pref: str, suff: str) -> int:
        cur = self.root
        # 3. 搜索 "后缀#前缀"
        for c in suff + '#' + pref:
            if c not in cur:
                return -1
            cur = cur[c]
        return cur['idx']
```

（`'idx'` 是多字符的键，不会和单字符的边冲突。）

复杂度：构造 O(n · L^2)；查询 O(L)。

<br>

### 1032. 字符流 / Stream of Characters

[CN](https://leetcode.cn/problems/stream-of-characters/) · [EN](https://leetcode.com/problems/stream-of-characters/) · Hard

**题意**：给定单词表，每次 `query(letter)` 追加一个字符，判断当前字符流是否有某个后缀在单词表中。

**思路**：把每个单词反着插入 Trie。每来一个字符，就从字符流末尾往前在 Trie 里走，碰到单词结尾就返回 True。只需要保存最近 `最长单词长度` 个字符。

```python
class StreamChecker:
    def __init__(self, words: List[str]):
        self.root = {}
        for w in words:
            cur = self.root
            # 1. 反向插入
            for c in reversed(w):
                cur = cur.setdefault(c, {})
            cur['#'] = True
        # 2. 只保留最近 max_len 个字符
        self.s = deque(maxlen=max(map(len, words)))

    def query(self, letter: str) -> bool:
        self.s.append(letter)
        cur = self.root
        # 3. 从最新的字符往前匹配
        for c in reversed(self.s):
            if c not in cur:
                return False
            cur = cur[c]
            if '#' in cur:
                return True
        return False
```

复杂度：`query` O(最长单词长度)。

<br>

### 642. 设计搜索自动补全系统 / Design Search Autocomplete System 🔒

[CN](https://leetcode.cn/problems/design-search-autocomplete-system/) · [EN](https://leetcode.com/problems/design-search-autocomplete-system/) · Hard

**题意**：给历史句子和热度。用户逐字符输入，每输入一个字符（非 `#`）返回以当前输入为前缀、热度最高的 3 个句子（热度相同按字典序）；输入 `#` 表示句子结束，把它的热度加 1 并重置输入。

**思路**：Trie 节点存“经过它的所有句子 → 热度”（数据量小时直接存计数表即可）。输入时维护当前节点指针，走不下去就置为 None，后续都返回空；每次在当前节点的计数表里排序取前 3。

```python
class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.root = {}
        for s, t in zip(sentences, times):
            self._add(s, t)
        self.prefix = []
        self.cur = self.root  # 1. 当前输入对应的节点

    def _add(self, s: str, t: int) -> None:
        cur = self.root
        for c in s:
            cur = cur.setdefault(c, {})
            # 2. 节点上的 '$' 存经过它的句子 -> 热度
            cnt = cur.setdefault('$', {})
            cnt[s] = cnt.get(s, 0) + t

    def input(self, c: str) -> List[str]:
        if c == '#':
            # 3. 句子结束：热度 +1，重置状态
            self._add(''.join(self.prefix), 1)
            self.prefix = []
            self.cur = self.root
            return []
        self.prefix.append(c)
        # 4. 沿着输入往下走，走不下去就一直是 None
        if self.cur is not None:
            self.cur = self.cur.get(c)
        if self.cur is None:
            return []
        # 5. 热度降序、字典序升序，取前 3
        cnt = self.cur['$']
        return sorted(cnt, key=lambda s: (-cnt[s], s))[:3]
```

复杂度：`input` O(k log k)，k 是当前前缀下的句子数；数据大时可以在节点上只维护 Top 3。

<br>

### 588. 设计内存文件系统 / Design In-Memory File System 🔒

[CN](https://leetcode.cn/problems/design-in-memory-file-system/) · [EN](https://leetcode.com/problems/design-in-memory-file-system/) · Hard

**题意**：实现 `ls(path)`（是文件返回 `[文件名]`，是目录返回按字典序排列的子项）、`mkdir(path)`（递归创建）、`addContentToFile(path, content)`（追加内容，不存在就创建）、`readContentFromFile(path)`。

**思路**：Trie 的每条边是一个路径段（目录名或文件名）。节点存子节点字典和文件内容；`content is None` 表示是目录。

```python
class Node:
    def __init__(self):
        self.son = {}         # 1. 名字 -> 子节点
        self.content = None   # 2. None 表示目录，否则是文件内容


class FileSystem:
    def __init__(self):
        self.root = Node()

    # 3. 沿路径走到节点，不存在就创建
    def _walk(self, path: str) -> Node:
        cur = self.root
        for name in path.split('/')[1:]:
            if not name:  # 根目录 "/" 切出来是空串
                continue
            if name not in cur.son:
                cur.son[name] = Node()
            cur = cur.son[name]
        return cur

    def ls(self, path: str) -> List[str]:
        node = self._walk(path)
        # 4. 文件：只返回自己的名字
        if node.content is not None:
            return [path.split('/')[-1]]
        return sorted(node.son)

    def mkdir(self, path: str) -> None:
        self._walk(path)

    def addContentToFile(self, filePath: str, content: str) -> None:
        node = self._walk(filePath)
        # 5. 新文件从空串开始
        node.content = (node.content or '') + content

    def readContentFromFile(self, filePath: str) -> str:
        return self._walk(filePath).content
```

复杂度：每次操作 O(路径长度)，`ls` 额外 O(k log k) 排序。

<br>

### 1166. 设计文件系统 / Design File System 🔒

[CN](https://leetcode.cn/problems/design-file-system/) · [EN](https://leetcode.com/problems/design-file-system/) · Medium

**题意**：`createPath(path, value)`：路径已存在或父路径不存在时返回 False，否则创建并返回 True；`get(path)` 返回值，不存在返回 -1。

**思路**：哈希表 `路径 → 值` 就够了。父路径就是 `path` 去掉最后一段。用 Trie 也可以，但哈希表最简单。

```python
class FileSystem:
    def __init__(self):
        self.paths = {"": -1}  # 1. 根的父路径记为空串

    def createPath(self, path: str, value: int) -> bool:
        if path in self.paths:
            return False
        # 2. 父路径：去掉最后一个 '/' 及其后面的部分
        parent = path[:path.rfind('/')]
        if parent not in self.paths:
            return False
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        return self.paths.get(path, -1)
```

复杂度：O(L)（哈希字符串）。

<br>
<br>

## 8. 业务系统模拟

最像“应用题”的一类：题面很长，但拆开后每个方法都是前面几类的组合。先按五步法列出“每个查询需要什么”，再把哈希表、堆、有序集合拼起来。最容易错的是**多个结构之间的同步**：一个实体的状态变了，所有引用它的结构都要更新（或懒删除）。

| 题号 | 题目 | 链接 | 核心结构 |
|---|---|---|---|
| 355 | 设计推特 | [CN](https://leetcode.cn/problems/design-twitter/) · [EN](https://leetcode.com/problems/design-twitter/) | 哈希表 + 堆（多路归并取 Top 10） |
| 535 | TinyURL 的加密与解密 | [CN](https://leetcode.cn/problems/encode-and-decode-tinyurl/) · [EN](https://leetcode.com/problems/encode-and-decode-tinyurl/) | 哈希表 + 自增 id |
| 1396 | 设计地铁系统 | [CN](https://leetcode.cn/problems/design-underground-system/) · [EN](https://leetcode.com/problems/design-underground-system/) | 两个哈希表 |
| 1603 | 设计停车系统 | [CN](https://leetcode.cn/problems/design-parking-system/) · [EN](https://leetcode.com/problems/design-parking-system/) | 计数 |
| 1472 | 设计浏览器历史记录 | [CN](https://leetcode.cn/problems/design-browser-history/) · [EN](https://leetcode.com/problems/design-browser-history/) | 数组 + 指针 |
| 1600 | 王位继承顺序 | [CN](https://leetcode.cn/problems/throne-inheritance/) · [EN](https://leetcode.com/problems/throne-inheritance/) | 多叉树 + 前序遍历 |
| 1797 | 设计一个验证系统 | [CN](https://leetcode.cn/problems/design-authentication-manager/) · [EN](https://leetcode.com/problems/design-authentication-manager/) | 哈希表 + 过期时间 |
| 1845 | 座位预约管理系统 | [CN](https://leetcode.cn/problems/seat-reservation-manager/) · [EN](https://leetcode.com/problems/seat-reservation-manager/) | 最小堆 |
| 1912 | 设计电影租借系统 | [CN](https://leetcode.cn/problems/design-movie-rental-system/) · [EN](https://leetcode.com/problems/design-movie-rental-system/) | 多个有序集合 |
| 1993 | 树上的操作 | [CN](https://leetcode.cn/problems/operations-on-tree/) · [EN](https://leetcode.com/problems/operations-on-tree/) | 树 + DFS |
| 2043 | 简易银行系统 | [CN](https://leetcode.cn/problems/simple-bank-system/) · [EN](https://leetcode.com/problems/simple-bank-system/) | 数组模拟 |
| 2241 | 设计一个 ATM 机器 | [CN](https://leetcode.cn/problems/design-an-atm-machine/) · [EN](https://leetcode.com/problems/design-an-atm-machine/) | 贪心 |
| 2296 | 设计一个文本编辑器 | [CN](https://leetcode.cn/problems/design-a-text-editor/) · [EN](https://leetcode.com/problems/design-a-text-editor/) | 对顶栈 |
| 2336 | 无限集中的最小数字 | [CN](https://leetcode.cn/problems/smallest-number-in-infinite-set/) · [EN](https://leetcode.com/problems/smallest-number-in-infinite-set/) | 有序集合 / 堆 |
| 2349 | 设计数字容器系统 | [CN](https://leetcode.cn/problems/design-a-number-container-system/) · [EN](https://leetcode.com/problems/design-a-number-container-system/) | 哈希表 + 懒删除堆 |
| 2353 | 设计食物评分系统 | [CN](https://leetcode.cn/problems/design-a-food-rating-system/) · [EN](https://leetcode.com/problems/design-a-food-rating-system/) | 哈希表 + 有序集合 / 堆 |
| 2502 | 设计内存分配器 | [CN](https://leetcode.cn/problems/design-memory-allocator/) · [EN](https://leetcode.com/problems/design-memory-allocator/) | 数组模拟 |
| 2642 | 设计可以求最短路径的图类 | [CN](https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/) · [EN](https://leetcode.com/problems/design-graph-with-shortest-path-calculator/) | Dijkstra / Floyd |
| 1656 | 设计有序流 | [CN](https://leetcode.cn/problems/design-an-ordered-stream/) · [EN](https://leetcode.com/problems/design-an-ordered-stream/) | 数组 + 指针 |
| 348 🔒 | 设计井字棋 | [CN](https://leetcode.cn/problems/design-tic-tac-toe/) · [EN](https://leetcode.com/problems/design-tic-tac-toe/) | 行列计数 |
| 353 🔒 | 设计贪吃蛇游戏 | [CN](https://leetcode.cn/problems/design-snake-game/) · [EN](https://leetcode.com/problems/design-snake-game/) | 双端队列 + 哈希集合 |
| 379 🔒 | 电话目录管理系统 | [CN](https://leetcode.cn/problems/design-phone-directory/) · [EN](https://leetcode.com/problems/design-phone-directory/) | 队列 + 集合 |
| 1244 🔒 | 力扣排行榜 | [CN](https://leetcode.cn/problems/design-a-leaderboard/) · [EN](https://leetcode.com/problems/design-a-leaderboard/) | 哈希表 + 有序集合 |

<br>

### 355. 设计推特 / Design Twitter

[CN](https://leetcode.cn/problems/design-twitter/) · [EN](https://leetcode.com/problems/design-twitter/) · Medium

**题意**：`postTweet(userId, tweetId)` 发推；`getNewsFeed(userId)` 返回自己和关注的人最近的 10 条推文 id（从新到旧）；`follow / unfollow`。

**思路**：全局自增时间戳 `time` 给每条推文排序。每个用户的推文按时间顺序存在列表里（最新的在末尾）。取 feed 就是“多路归并取前 10”：每个人的列表末尾放进最大堆，弹出一条就把这个人的下一条（更旧的）放进去。

```python
class Twitter:

    def __init__(self):
        # 1. 全局时间戳：每发一条推文就 +1，数字越大表示越新
        self.time = 0
        # 2. tweets[用户] = 这个用户发过的推文列表，每项是 (推文 id, 时间戳)
        #    按发布顺序追加，所以列表末尾就是这个用户最新的推文
        self.tweets = defaultdict(list)
        # 3. followees[用户] = 这个用户关注的人的集合（用 set 方便增删、自动去重）
        self.followees = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        # 4. 发推：记下推文 id 和当前时间，然后时间 +1
        self.tweets[userId].append((tweetId, self.time))
        self.time += 1

    def getNewsFeed(self, userId: int) -> list[int]:
        # 5. 要看的人 = 我关注的人 + 我自己
        users = self.followees[userId] | {userId}
        # 6. 每个人的推文列表都已经按时间排好序，相当于 K 个有序列表做多路归并
        #    先把每个人「最新的一条」放进堆里
        heap = []
        for uid in users:
            if self.tweets[uid]:
                i = len(self.tweets[uid]) - 1  # 最后一条就是最新的
                tid, time = self.tweets[uid][i]
                # 7. heapq 是最小堆，时间取负数，堆顶就是最新的推文
                #    同时记下 uid 和下标 i，弹出后才知道下一条去哪拿
                heap.append((-time, tid, uid, i))
        heapq.heapify(heap)  # O(K) 建堆
        ans = []
        # 8. 每次弹出所有候选中最新的一条，最多取 10 条
        while heap and len(ans) < 10:
            time, tid, uid, i = heapq.heappop(heap)
            ans.append(tid)
            # 9. 这条被取走后，把同一个人的上一条（更旧的一条）补进堆里
            if i - 1 >= 0:
                tid, time = self.tweets[uid][i - 1]
                heapq.heappush(heap, (-time, tid, uid, i - 1))
        return ans

    def follow(self, followerId: int, followeeId: int) -> None:
        # 10. 关注：加入集合，重复关注也不会出问题
        self.followees[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        # 11. 取关：用 discard 而不是 remove，没关注过也不会报错
        self.followees[followerId].discard(followeeId)
```

复杂度：`getNewsFeed` O(f + 10 log f)，f 是关注人数；其余 O(1)。

<br>

### 535. TinyURL 的加密与解密 / Encode and Decode TinyURL

[CN](https://leetcode.cn/problems/encode-and-decode-tinyurl/) · [EN](https://leetcode.com/problems/encode-and-decode-tinyurl/) · Medium

**题意**：设计 `encode(longUrl)` 生成短链接、`decode(shortUrl)` 还原。

**思路**：自增 id + 哈希表。同一个长链接重复 encode 时复用已有短链接。面试时可以继续聊：id 转 62 进制让短链接更短；分布式下用号段发号器；随机字符串要处理冲突。

```python
class Codec:
    PREFIX = "http://tinyurl.com/"

    def __init__(self):
        self.long_to_id = {}  # 1. 长链接 -> id（去重）
        self.urls = []        # 2. id -> 长链接

    def encode(self, longUrl: str) -> str:
        if longUrl not in self.long_to_id:
            self.long_to_id[longUrl] = len(self.urls)
            self.urls.append(longUrl)
        return self.PREFIX + str(self.long_to_id[longUrl])

    def decode(self, shortUrl: str) -> str:
        # 3. 去掉前缀就是 id
        return self.urls[int(shortUrl[len(self.PREFIX):])]
```

复杂度：O(L)。

<br>

### 1396. 设计地铁系统 / Design Underground System

[CN](https://leetcode.cn/problems/design-underground-system/) · [EN](https://leetcode.com/problems/design-underground-system/) · Medium

**题意**：`checkIn(id, station, t)` 进站，`checkOut(id, station, t)` 出站，`getAverageTime(start, end)` 返回从 start 到 end 的所有行程的平均用时。

**思路**：两个哈希表。`check_in`：乘客 id → (进站站名, 时间)；`stats`：(起点, 终点) → [总用时, 次数]。出站时把这段行程累加到 `stats`。

```python
class UndergroundSystem:
    def __init__(self):
        self.check_in = {}                         # 1. id -> (进站站名, 进站时间)
        self.stats = defaultdict(lambda: [0, 0])   # 2. (起点, 终点) -> [总用时, 次数]

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.check_in[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        # 3. 取出进站记录，累加这段行程
        start, t0 = self.check_in.pop(id)
        s = self.stats[(start, stationName)]
        s[0] += t - t0
        s[1] += 1

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        total, cnt = self.stats[(startStation, endStation)]
        return total / cnt
```

复杂度：全部 O(1)。

<br>

### 1603. 设计停车系统 / Design Parking System

[CN](https://leetcode.cn/problems/design-parking-system/) · [EN](https://leetcode.com/problems/design-parking-system/) · Easy

**题意**：大、中、小三种车位各有若干个，`addCar(carType)`（1/2/3）有对应空位就停入并返回 True。

**思路**：数组记录每种车位剩余数量。

```python
class ParkingSystem:
    def __init__(self, big: int, medium: int, small: int):
        # 1. 下标 1、2、3 对应三种车位，下标 0 占位
        self.left = [0, big, medium, small]

    def addCar(self, carType: int) -> bool:
        # 2. 没有空位
        if self.left[carType] == 0:
            return False
        self.left[carType] -= 1
        return True
```

复杂度：O(1)。

<br>

### 1472. 设计浏览器历史记录 / Design Browser History

[CN](https://leetcode.cn/problems/design-browser-history/) · [EN](https://leetcode.com/problems/design-browser-history/) · Medium

**题意**：`visit(url)` 访问新页面（清空前进记录）；`back(steps)` / `forward(steps)` 最多后退 / 前进 steps 步，返回当前页面。

**思路**：数组 + 当前位置 `cur` + 有效末尾 `end`。`visit` 覆盖 `cur + 1` 并把 `end` 设为它，这样“清空前进记录”是 O(1)，不需要真的删除。

```python
class BrowserHistory:
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.cur = 0  # 1. 当前页面下标
        self.end = 0  # 2. 有效历史的最后一个下标（之后的是作废的前进记录）

    def visit(self, url: str) -> None:
        self.cur += 1
        # 3. 覆盖或追加，然后丢弃前进记录
        if self.cur == len(self.history):
            self.history.append(url)
        else:
            self.history[self.cur] = url
        self.end = self.cur

    def back(self, steps: int) -> str:
        self.cur = max(self.cur - steps, 0)
        return self.history[self.cur]

    def forward(self, steps: int) -> str:
        self.cur = min(self.cur + steps, self.end)
        return self.history[self.cur]
```

复杂度：全部 O(1)。

<br>

### 1600. 王位继承顺序 / Throne Inheritance

[CN](https://leetcode.cn/problems/throne-inheritance/) · [EN](https://leetcode.com/problems/throne-inheritance/) · Medium

**题意**：国王是根，`birth(parent, child)` 加孩子，`death(name)` 标记死亡，`getInheritanceOrder()` 返回继承顺序（排除死者）。

**思路**：继承顺序就是这棵多叉树的前序遍历（孩子按出生顺序）。死亡只打标记，不改树结构。

```python
class ThroneInheritance:
    def __init__(self, kingName: str):
        self.king = kingName
        self.children = defaultdict(list)  # 1. 每个人的孩子，按出生顺序
        self.dead = set()                  # 2. 死亡标记

    def birth(self, parentName: str, childName: str) -> None:
        self.children[parentName].append(childName)

    def death(self, name: str) -> None:
        self.dead.add(name)

    def getInheritanceOrder(self) -> List[str]:
        ans = []
        # 3. 前序遍历，跳过死者（但继续遍历其子孙）
        def dfs(x: str) -> None:
            if x not in self.dead:
                ans.append(x)
            for y in self.children[x]:
                dfs(y)
        dfs(self.king)
        return ans
```

复杂度：`birth / death` O(1)；`getInheritanceOrder` O(n)。人数多时把递归改成显式栈，避免爆栈。

<br>

### 1797. 设计一个验证系统 / Design Authentication Manager

[CN](https://leetcode.cn/problems/design-authentication-manager/) · [EN](https://leetcode.com/problems/design-authentication-manager/) · Medium

**题意**：每个 token 生成后 `timeToLive` 秒过期。`generate(id, t)` 生成；`renew(id, t)` 续期未过期的 token；`countUnexpiredTokens(t)` 统计未过期的个数。过期和其它操作同一时刻发生时，过期优先。

**思路**（灵神写法）：有序字典 `token → 过期时间`。因为 `currentTime` 递增，`renew` 时把 token 移到末尾，字典就按过期时间有序；统计时从头部删掉所有已过期的。

```python
class AuthenticationManager:
    def __init__(self, timeToLive: int):
        self.ttl = timeToLive
        self.expire = OrderedDict()  # 1. tokenId -> 过期时间，按过期时间递增

    # 2. 删掉所有过期 token（头部的最早过期）
    def _clean(self, t: int) -> None:
        while self.expire and next(iter(self.expire.values())) <= t:
            self.expire.popitem(last=False)

    def generate(self, tokenId: str, currentTime: int) -> None:
        self.expire[tokenId] = currentTime + self.ttl

    def renew(self, tokenId: str, currentTime: int) -> None:
        self._clean(currentTime)
        if tokenId in self.expire:
            # 3. 续期后过期时间最晚，移到末尾保持有序
            self.expire.move_to_end(tokenId)
            self.expire[tokenId] = currentTime + self.ttl

    def countUnexpiredTokens(self, currentTime: int) -> int:
        self._clean(currentTime)
        return len(self.expire)
```

复杂度：均摊 O(1)。

<br>

### 1845. 座位预约管理系统 / Seat Reservation Manager

[CN](https://leetcode.cn/problems/seat-reservation-manager/) · [EN](https://leetcode.com/problems/seat-reservation-manager/) · Medium

**题意**：座位 1..n。`reserve()` 预约编号最小的空座位并返回；`unreserve(seatNumber)` 取消预约。

**思路**（灵神写法）：不要一开始就把 n 个座位放进堆。用 `cnt` 表示“1..cnt 都被预约过一次”；堆里只放被取消的座位。`reserve` 时堆非空就取堆顶（它一定比 cnt+1 小），否则返回 cnt+1。

```python
class SeatManager:
    def __init__(self, n: int):
        self.cnt = 0         # 1. 1..cnt 被顺序预约过
        self.available = []  # 2. 最小堆：被取消预约的座位

    def reserve(self) -> int:
        # 3. 优先用取消的座位（都 <= cnt，比 cnt+1 小）
        if self.available:
            return heappop(self.available)
        self.cnt += 1
        return self.cnt

    def unreserve(self, seatNumber: int) -> None:
        heappush(self.available, seatNumber)
```

复杂度：O(log n)；构造 O(1)。

<br>

### 1912. 设计电影租借系统 / Design Movie Rental System

[CN](https://leetcode.cn/problems/design-movie-rental-system/) · [EN](https://leetcode.com/problems/design-movie-rental-system/) · Hard

**题意**：`entries[i] = [shop, movie, price]`。`search(movie)`：未借出的该电影中最便宜的 5 家店（价格、店号升序）；`rent / drop` 借出、归还；`report()`：已借出的最便宜的 5 部 `[shop, movie]`（价格、店号、电影号升序）。

**思路**：三个结构。`price[(shop, movie)]` 查价格；`unrented[movie]` 是 `SortedList((price, shop))`；`rented` 是 `SortedList((price, shop, movie))`。借出 / 归还就是在两边之间移动一条记录。

```python
from sortedcontainers import SortedList

class MovieRentingSystem:
    def __init__(self, n: int, entries: List[List[int]]):
        self.price = {}                         # 1. (shop, movie) -> price
        self.unrented = defaultdict(SortedList)  # 2. movie -> {(price, shop)}
        self.rented = SortedList()               # 3. {(price, shop, movie)}
        for shop, movie, p in entries:
            self.price[(shop, movie)] = p
            self.unrented[movie].add((p, shop))

    def search(self, movie: int) -> List[int]:
        return [shop for _, shop in self.unrented[movie][:5]]

    def rent(self, shop: int, movie: int) -> None:
        p = self.price[(shop, movie)]
        # 4. 从“未借出”移到“已借出”
        self.unrented[movie].remove((p, shop))
        self.rented.add((p, shop, movie))

    def drop(self, shop: int, movie: int) -> None:
        p = self.price[(shop, movie)]
        # 5. 从“已借出”移回“未借出”
        self.rented.remove((p, shop, movie))
        self.unrented[movie].add((p, shop))

    def report(self) -> List[List[int]]:
        return [[shop, movie] for _, shop, movie in self.rented[:5]]
```

复杂度：`rent / drop` O(log n)；`search / report` O(log n + 5)。

<br>

### 1993. 树上的操作 / Operations on Tree

[CN](https://leetcode.cn/problems/operations-on-tree/) · [EN](https://leetcode.com/problems/operations-on-tree/) · Medium

**题意**：给 `parent` 数组表示的树。`lock(num, user)`：未上锁则上锁；`unlock(num, user)`：被同一用户锁住则解锁；`upgrade(num, user)`：节点未上锁、至少有一个子孙上锁、所有祖先都未上锁时，给它上锁并解锁所有子孙。

**思路**：模拟。`lock_user[i]` 记录谁锁了节点 i（-1 表示没锁）。`upgrade` 时往上走检查祖先，DFS 子树检查并解锁子孙。n ≤ 2000，每次 O(n) 可以接受。

```python
class LockingTree:
    def __init__(self, parent: List[int]):
        n = len(parent)
        self.parent = parent
        self.lock_user = [-1] * n               # 1. -1 表示未上锁
        self.children = [[] for _ in range(n)]  # 2. 建子节点表
        for i in range(1, n):
            self.children[parent[i]].append(i)

    def lock(self, num: int, user: int) -> bool:
        if self.lock_user[num] != -1:
            return False
        self.lock_user[num] = user
        return True

    def unlock(self, num: int, user: int) -> bool:
        if self.lock_user[num] != user:
            return False
        self.lock_user[num] = -1
        return True

    def upgrade(self, num: int, user: int) -> bool:
        # 3. 自己和所有祖先都不能上锁
        x = num
        while x != -1:
            if self.lock_user[x] != -1:
                return False
            x = self.parent[x]

        # 4. DFS 解锁所有子孙，同时记录是否至少有一个上锁的子孙
        def dfs(x: int) -> bool:
            found = self.lock_user[x] != -1
            self.lock_user[x] = -1
            for y in self.children[x]:
                found |= dfs(y)  # 用 |= 而不是 or，保证每个子树都会被解锁
            return found

        if not any([dfs(y) for y in self.children[num]]):
            return False  # 5. 没有上锁的子孙（此时子树本来就全未上锁，dfs 没改任何东西）
        self.lock_user[num] = user
        return True
```

复杂度：`lock / unlock` O(1)；`upgrade` O(n)。

<br>

### 2043. 简易银行系统 / Simple Bank System

[CN](https://leetcode.cn/problems/simple-bank-system/) · [EN](https://leetcode.com/problems/simple-bank-system/) · Medium

**题意**：账户编号 1..n，`transfer / deposit / withdraw`，账户不存在或余额不足返回 False。

**思路**：数组模拟。先统一写一个“账户是否合法”的检查，再写每个操作。

```python
class Bank:
    def __init__(self, balance: List[int]):
        self.balance = balance

    # 1. 账户编号从 1 开始
    def _valid(self, account: int) -> bool:
        return 1 <= account <= len(self.balance)

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        # 2. 两个账户都合法，并且转出账户余额足够
        if not self._valid(account2) or not self.withdraw(account1, money):
            return False
        self.balance[account2 - 1] += money
        return True

    def deposit(self, account: int, money: int) -> bool:
        if not self._valid(account):
            return False
        self.balance[account - 1] += money
        return True

    def withdraw(self, account: int, money: int) -> bool:
        if not self._valid(account) or self.balance[account - 1] < money:
            return False
        self.balance[account - 1] -= money
        return True
```

`transfer` 里先检查 `account2`，再调用 `withdraw` 扣钱，顺序不能反：否则 `account2` 不合法时钱已经被扣了。

复杂度：O(1)。

<br>

### 2241. 设计一个 ATM 机器 / Design an ATM Machine

[CN](https://leetcode.cn/problems/design-an-atm-machine/) · [EN](https://leetcode.com/problems/design-an-atm-machine/) · Medium

**题意**：五种面额 20、50、100、200、500。`deposit(cnt)` 存入各面额张数；`withdraw(amount)` 从大面额到小面额贪心取钞（每种取尽量多），取不出恰好的金额就返回 `[-1]` 且不取任何钱。

**思路**：按题意贪心模拟。先算出每种面额取几张，最后检查剩余金额是否为 0，为 0 才真正扣减。

```python
class ATM:
    DENOM = (20, 50, 100, 200, 500)

    def __init__(self):
        self.cnt = [0] * 5  # 1. 每种面额的张数

    def deposit(self, banknotesCount: List[int]) -> None:
        for i, c in enumerate(banknotesCount):
            self.cnt[i] += c

    def withdraw(self, amount: int) -> List[int]:
        ans = [0] * 5
        # 2. 从大面额到小面额，每种尽量多取
        for i in range(4, -1, -1):
            ans[i] = min(self.cnt[i], amount // self.DENOM[i])
            amount -= ans[i] * self.DENOM[i]
        # 3. 取不出恰好的金额：不扣任何钱
        if amount:
            return [-1]
        for i, c in enumerate(ans):
            self.cnt[i] -= c
        return ans
```

复杂度：O(1)（5 种面额）。

<br>

### 2296. 设计一个文本编辑器 / Design a Text Editor

[CN](https://leetcode.cn/problems/design-a-text-editor/) · [EN](https://leetcode.com/problems/design-a-text-editor/) · Hard

**题意**：带光标的文本编辑器：`addText(text)` 在光标处插入；`deleteText(k)` 删除光标左边 k 个字符；`cursorLeft(k) / cursorRight(k)` 移动光标，返回光标左边最近的 10 个字符。

**思路**（灵神写法）：对顶栈。光标左边的字符存在栈 `left`（栈顶挨着光标），右边的存在栈 `right`（栈顶挨着光标）。插入是压 `left`；删除是弹 `left`；光标左移就是从 `left` 弹出压入 `right`。每次操作只和 k 有关，与文本总长度无关。

```python
class TextEditor:
    def __init__(self):
        self.left = []   # 1. 光标左侧字符，栈顶挨着光标
        self.right = []  # 2. 光标右侧字符（倒序），栈顶挨着光标

    def addText(self, text: str) -> None:
        self.left.extend(text)  # 3. 插入：压入左栈

    def deleteText(self, k: int) -> int:
        # 4. 删除：弹出左栈最多 k 个
        pre = len(self.left)
        del self.left[-k:]
        return pre - len(self.left)

    def text(self) -> str:
        # 5. 光标左边最近的 10 个字符
        return ''.join(self.left[-10:])

    def cursorLeft(self, k: int) -> str:
        # 6. 光标左移：左栈的字符逐个挪到右栈
        while k and self.left:
            self.right.append(self.left.pop())
            k -= 1
        return self.text()

    def cursorRight(self, k: int) -> str:
        while k and self.right:
            self.left.append(self.right.pop())
            k -= 1
        return self.text()
```

复杂度：每次 O(k)，与文本长度无关。

<br>

### 2336. 无限集中的最小数字 / Smallest Number in Infinite Set

[CN](https://leetcode.cn/problems/smallest-number-in-infinite-set/) · [EN](https://leetcode.com/problems/smallest-number-in-infinite-set/) · Medium

**题意**：集合初始包含所有正整数。`popSmallest()` 删除并返回最小的数；`addBack(num)` 把 num 加回来（已在集合中则无操作）。

**思路**：和 1845 同一个套路。`thres` 表示 `>= thres` 的数都还在集合里；小于 `thres` 的被加回来的数，存在有序集合（或堆 + 集合去重）里。

```python
from sortedcontainers import SortedSet

class SmallestInfiniteSet:
    def __init__(self):
        self.thres = 1         # 1. [thres, ∞) 都在集合里
        self.s = SortedSet()   # 2. 小于 thres、被加回来的数

    def popSmallest(self) -> int:
        # 3. 加回来的数都 < thres，优先返回
        if self.s:
            return self.s.pop(0)
        self.thres += 1
        return self.thres - 1

    def addBack(self, num: int) -> None:
        # 4. 只有被删过的数（< thres）才需要加回
        if num < self.thres:
            self.s.add(num)
```

复杂度：O(log n)。

<br>

### 2349. 设计数字容器系统 / Design a Number Container System

[CN](https://leetcode.cn/problems/design-a-number-container-system/) · [EN](https://leetcode.com/problems/design-a-number-container-system/) · Medium

**题意**：`change(index, number)` 把下标 index 的数改成 number（可以覆盖）；`find(number)` 返回装着 number 的最小下标，没有返回 -1。

**思路**（灵神写法）：懒删除堆。`index_to_number` 记录每个下标当前的数；`number_to_indices[number]` 是最小堆，`change` 时只往新数的堆里加下标，不从旧数的堆里删。`find` 时如果堆顶下标现在装的已经不是 number，说明是过期数据，弹掉。

```python
class NumberContainers:
    def __init__(self):
        self.index_to_number = {}                 # 1. 下标 -> 当前的数
        self.number_to_indices = defaultdict(list)  # 2. 数 -> 下标最小堆（可能有过期数据）

    def change(self, index: int, number: int) -> None:
        # 3. 只更新映射，旧数堆里的 index 留着，懒删除
        self.index_to_number[index] = number
        heappush(self.number_to_indices[number], index)

    def find(self, number: int) -> int:
        h = self.number_to_indices[number]
        # 4. 堆顶下标装的不是 number 了，就是过期数据
        while h and self.index_to_number[h[0]] != number:
            heappop(h)
        return h[0] if h else -1
```

复杂度：均摊 O(log n)。

<br>

### 2353. 设计食物评分系统 / Design a Food Rating System

[CN](https://leetcode.cn/problems/design-a-food-rating-system/) · [EN](https://leetcode.com/problems/design-a-food-rating-system/) · Medium

**题意**：`changeRating(food, newRating)` 修改评分；`highestRated(cuisine)` 返回该烹饪方式下评分最高的食物，评分相同返回字典序最小的。

**思路**：`food → [rating, cuisine]` 处理修改；`cuisine → SortedList((-rating, food))`，第一个元素就是答案。修改时先删旧的 `(-old, food)` 再加新的。（也可以用懒删除堆，和 2349 一样。）

```python
from sortedcontainers import SortedList

class FoodRatings:
    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.food_info = {}                 # 1. food -> [rating, cuisine]
        self.cs = defaultdict(SortedList)   # 2. cuisine -> {(-rating, food)}
        for f, c, r in zip(foods, cuisines, ratings):
            self.food_info[f] = [r, c]
            self.cs[c].add((-r, f))

    def changeRating(self, food: str, newRating: int) -> None:
        info = self.food_info[food]
        sl = self.cs[info[1]]
        # 3. 删旧加新，保持两个结构同步
        sl.remove((-info[0], food))
        sl.add((-newRating, food))
        info[0] = newRating

    def highestRated(self, cuisine: str) -> str:
        # 4. 评分最高、字典序最小的在第一个
        return self.cs[cuisine][0][1]
```

复杂度：`changeRating` O(log n)；`highestRated` O(1)（`SortedList` 下标访问是 O(log n)，常数很小）。

<br>

### 2502. 设计内存分配器 / Design Memory Allocator

[CN](https://leetcode.cn/problems/design-memory-allocator/) · [EN](https://leetcode.com/problems/design-memory-allocator/) · Medium

**题意**：大小为 n 的内存。`allocate(size, mID)` 找最左边连续 size 个空闲单元分配给 mID，返回起始下标（没有返回 -1）；`freeMemory(mID)` 释放 mID 的所有单元，返回释放的个数。

**思路**：n ≤ 1000，数组模拟：`memory[i]` 存占用者 id，0 表示空闲。分配时从左往右数连续空闲长度，够 size 就填上。

```python
class Allocator:
    def __init__(self, n: int):
        self.memory = [0] * n  # 1. 0 表示空闲，否则是 mID

    def allocate(self, size: int, mID: int) -> int:
        free = 0  # 2. 当前连续空闲长度
        for i, id in enumerate(self.memory):
            if id:
                free = 0
                continue
            free += 1
            if free == size:
                # 3. [i-size+1, i] 分配给 mID
                self.memory[i - size + 1: i + 1] = [mID] * size
                return i - size + 1
        return -1

    def freeMemory(self, mID: int) -> int:
        cnt = 0
        for i, id in enumerate(self.memory):
            if id == mID:
                self.memory[i] = 0
                cnt += 1
        return cnt
```

复杂度：O(n)。n 很大时可以用有序集合维护空闲区间，或线段树维护“最长连续空闲段”。

<br>

### 2642. 设计可以求最短路径的图类 / Design Graph With Shortest Path Calculator

[CN](https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/) · [EN](https://leetcode.com/problems/design-graph-with-shortest-path-calculator/) · Hard

**题意**：有向带权图，`addEdge(edge)` 加边，`shortestPath(node1, node2)` 返回最短路，不可达返回 -1。

**思路**：每次查询跑一遍堆优化 Dijkstra，到达终点就提前返回。（如果查询远多于加边，可以用 Floyd 预处理，加边 `x→y` 时用 O(n^2) 更新所有 `i→x→y→j`。）

```python
class Graph:
    def __init__(self, n: int, edges: List[List[int]]):
        self.g = [[] for _ in range(n)]  # 1. 邻接表
        for e in edges:
            self.addEdge(e)

    def addEdge(self, edge: List[int]) -> None:
        x, y, w = edge
        self.g[x].append((y, w))

    def shortestPath(self, node1: int, node2: int) -> int:
        dis = [inf] * len(self.g)
        dis[node1] = 0
        h = [(0, node1)]
        while h:
            d, x = heappop(h)
            # 2. 第一次弹出终点，就是最短路
            if x == node2:
                return d
            # 3. 过期数据：x 之前已经以更短距离出过堆
            if d > dis[x]:
                continue
            for y, w in self.g[x]:
                # 4. 松弛
                if d + w < dis[y]:
                    dis[y] = d + w
                    heappush(h, (dis[y], y))
        return -1
```

复杂度：`shortestPath` O(m log m)；`addEdge` O(1)。

<br>

### 1656. 设计有序流 / Design an Ordered Stream

[CN](https://leetcode.cn/problems/design-an-ordered-stream/) · [EN](https://leetcode.com/problems/design-an-ordered-stream/) · Easy

**题意**：n 个 `(idKey, value)` 乱序到来（idKey 1..n），每次 `insert` 后返回从当前指针开始、连续已到达的最长一段 value，并移动指针。

**思路**：数组 + 指针 `ptr`。插入后从 `ptr` 往后扫连续非空的部分。

```python
class OrderedStream:
    def __init__(self, n: int):
        self.stream = [None] * (n + 1)  # 1. 下标 1..n
        self.ptr = 1                    # 2. 下一个要输出的位置

    def insert(self, idKey: int, value: str) -> List[str]:
        self.stream[idKey] = value
        start = self.ptr
        # 3. 从 ptr 往后输出连续已到达的部分
        while self.ptr < len(self.stream) and self.stream[self.ptr]:
            self.ptr += 1
        return self.stream[start:self.ptr]
```

复杂度：所有 `insert` 加起来 O(n)。

<br>

### 348. 设计井字棋 / Design Tic-Tac-Toe 🔒

[CN](https://leetcode.cn/problems/design-tic-tac-toe/) · [EN](https://leetcode.com/problems/design-tic-tac-toe/) · Medium

**题意**：n × n 棋盘，`move(row, col, player)` 落子，如果该玩家连成一行、一列或一条对角线就返回玩家编号，否则返回 0。要求比 O(n^2) 更快。

**思路**：不存棋盘，只存计数。玩家 1 记 +1、玩家 2 记 -1，每行、每列、两条对角线各一个计数器。某个计数器的绝对值达到 n，说明这条线全是同一个玩家。

```python
class TicTacToe:
    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n  # 1. 每行的计数：玩家 1 +1，玩家 2 -1
        self.cols = [0] * n
        self.diag = 0        # 2. 主对角线 row == col
        self.anti = 0        # 3. 副对角线 row + col == n - 1

    def move(self, row: int, col: int, player: int) -> int:
        v = 1 if player == 1 else -1
        n = self.n
        self.rows[row] += v
        self.cols[col] += v
        if row == col:
            self.diag += v
        if row + col == n - 1:
            self.anti += v
        # 4. 任意一条线的绝对值达到 n 就获胜
        if n in (abs(self.rows[row]), abs(self.cols[col]), abs(self.diag), abs(self.anti)):
            return player
        return 0
```

复杂度：`move` O(1)；空间 O(n)。

<br>

### 353. 设计贪吃蛇游戏 / Design Snake Game 🔒

[CN](https://leetcode.cn/problems/design-snake-game/) · [EN](https://leetcode.com/problems/design-snake-game/) · Medium

**题意**：`width × height` 的屏幕，蛇从 `(0, 0)` 出发，食物按顺序出现。`move(direction)` 返回当前得分；撞墙或撞到自己返回 -1。

**思路**：双端队列存蛇身（队首是头），哈希集合存蛇身坐标判断撞自己。每一步：算新头；没吃到食物就先移除尾巴（尾巴让出的格子新头可以走进去）；再判断撞墙 / 撞自己；最后加入新头。

```python
class SnakeGame:
    DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.w, self.h = width, height
        self.food = food
        self.fi = 0                   # 1. 下一个食物的下标
        self.body = deque([(0, 0)])   # 2. 队首是蛇头
        self.cells = {(0, 0)}         # 3. 蛇身占用的格子

    def move(self, direction: str) -> int:
        dr, dc = self.DIRS[direction]
        r, c = self.body[0][0] + dr, self.body[0][1] + dc
        # 4. 撞墙
        if not (0 <= r < self.h and 0 <= c < self.w):
            return -1
        # 5. 吃到食物：蛇身变长（不移除尾巴）
        if self.fi < len(self.food) and [r, c] == self.food[self.fi]:
            self.fi += 1
        else:
            # 6. 没吃到：先移除尾巴，新头可以走进尾巴让出的格子
            self.cells.remove(self.body.pop())
        # 7. 撞到自己
        if (r, c) in self.cells:
            return -1
        self.body.appendleft((r, c))
        self.cells.add((r, c))
        return self.fi  # 得分 = 吃到的食物数
```

复杂度：`move` O(1)。

<br>

### 379. 电话目录管理系统 / Design Phone Directory 🔒

[CN](https://leetcode.cn/problems/design-phone-directory/) · [EN](https://leetcode.com/problems/design-phone-directory/) · Medium

**题意**：号码 0..maxNumbers-1。`get()` 分配一个可用号码（没有返回 -1）；`check(number)` 判断是否可用；`release(number)` 回收。

**思路**：队列存可用号码，集合存已分配号码。`release` 一个没分配过的号码要忽略，否则会重复入队。

```python
class PhoneDirectory:
    def __init__(self, maxNumbers: int):
        self.available = deque(range(maxNumbers))  # 1. 可分配的号码
        self.used = set()                          # 2. 已分配的号码

    def get(self) -> int:
        if not self.available:
            return -1
        x = self.available.popleft()
        self.used.add(x)
        return x

    def check(self, number: int) -> bool:
        return number not in self.used

    def release(self, number: int) -> None:
        # 3. 只回收已分配的，避免重复入队
        if number in self.used:
            self.used.remove(number)
            self.available.append(number)
```

复杂度：全部 O(1)。

<br>

### 1244. 力扣排行榜 / Design A Leaderboard 🔒

[CN](https://leetcode.cn/problems/design-a-leaderboard/) · [EN](https://leetcode.com/problems/design-a-leaderboard/) · Medium

**题意**：`addScore(playerId, score)` 给玩家加分（新玩家从 0 开始）；`top(K)` 返回前 K 名分数之和；`reset(playerId)` 把玩家分数清零（移出榜单）。

**思路**：哈希表 `玩家 → 分数` + 有序集合存所有分数。加分时删旧分数、加新分数；`top(K)` 取最大的 K 个求和。

```python
from sortedcontainers import SortedList

class Leaderboard:
    def __init__(self):
        self.score = {}            # 1. playerId -> 分数
        self.scores = SortedList()  # 2. 所有玩家的分数

    def addScore(self, playerId: int, score: int) -> None:
        # 3. 老玩家：先删旧分数
        if playerId in self.score:
            self.scores.remove(self.score[playerId])
        self.score[playerId] = self.score.get(playerId, 0) + score
        self.scores.add(self.score[playerId])

    def top(self, K: int) -> int:
        # 4. 最大的 K 个
        return sum(self.scores[-K:])

    def reset(self, playerId: int) -> None:
        self.scores.remove(self.score.pop(playerId))
```

复杂度：`addScore / reset` O(log n)；`top` O(K + log n)。

<br>
<br>

## 9. 多线程 / 并发

LeetCode 有一个专门的「多线程」（Concurrency）题库，Python 都用 `threading` 写。几乎所有题都是同一个模型：**每个线程在执行前等一个“轮到我了”的信号，执行完把信号交给下一个线程**。信号可以是：

- `Event`：一次性开关，适合“某事发生后放行”（1114）。
- `Semaphore(n)`：计数信号量，`acquire` 减一（为 0 时阻塞），`release` 加一。初始值 0 表示“还不能走”，1 表示“可以走一次”。轮流执行最常用（1115、1116、1195）。
- `Condition`：配合 `while not 条件: wait()`，适合条件比较复杂的情况（1188、1195）。
- `Barrier(n)`：凑齐 n 个线程再一起放行（1117）。

| 题号 | 题目 | 链接 | 核心工具 |
|---|---|---|---|
| 1114 | 按序打印 | [CN](https://leetcode.cn/problems/print-in-order/) · [EN](https://leetcode.com/problems/print-in-order/) | Event |
| 1115 | 交替打印 FooBar | [CN](https://leetcode.cn/problems/print-foobar-alternately/) · [EN](https://leetcode.com/problems/print-foobar-alternately/) | Semaphore |
| 1116 | 打印零与奇偶数 | [CN](https://leetcode.cn/problems/print-zero-even-odd/) · [EN](https://leetcode.com/problems/print-zero-even-odd/) | Semaphore |
| 1117 | H2O 生成 | [CN](https://leetcode.cn/problems/building-h2o/) · [EN](https://leetcode.com/problems/building-h2o/) | Semaphore + Barrier |
| 1195 | 交替打印字符串（多线程 Fizz Buzz） | [CN](https://leetcode.cn/problems/fizz-buzz-multithreaded/) · [EN](https://leetcode.com/problems/fizz-buzz-multithreaded/) | Semaphore |
| 1226 | 哲学家进餐 | [CN](https://leetcode.cn/problems/the-dining-philosophers/) · [EN](https://leetcode.com/problems/the-dining-philosophers/) | Lock（避免死锁） |
| 1188 🔒 | 设计有限阻塞队列 | [CN](https://leetcode.cn/problems/design-bounded-blocking-queue/) · [EN](https://leetcode.com/problems/design-bounded-blocking-queue/) | Condition（生产者-消费者） |
| 1242 🔒 | 多线程网络爬虫 | [CN](https://leetcode.cn/problems/web-crawler-multithreaded/) · [EN](https://leetcode.com/problems/web-crawler-multithreaded/) | 线程池 + Lock |
| 1279 🔒 | 红绿灯路口 | [CN](https://leetcode.cn/problems/traffic-light-controlled-intersection/) · [EN](https://leetcode.com/problems/traffic-light-controlled-intersection/) | Lock |

<br>

### 1114. 按序打印 / Print in Order

[CN](https://leetcode.cn/problems/print-in-order/) · [EN](https://leetcode.com/problems/print-in-order/) · Easy

**题意**：三个线程分别调用 `first()`、`second()`、`third()`，调用顺序不确定，要保证输出顺序是 first、second、third。

**思路**：两个 `Event`：`first` 执行完设置 `e1`，`second` 等 `e1`、执行完设置 `e2`，`third` 等 `e2`。

```python
from threading import Event

class Foo:
    def __init__(self):
        self.e1 = Event()  # 1. first 已完成
        self.e2 = Event()  # 2. second 已完成

    def first(self, printFirst: 'Callable[[], None]') -> None:
        printFirst()
        self.e1.set()

    def second(self, printSecond: 'Callable[[], None]') -> None:
        self.e1.wait()  # 3. 等 first 完成
        printSecond()
        self.e2.set()

    def third(self, printThird: 'Callable[[], None]') -> None:
        self.e2.wait()  # 4. 等 second 完成
        printThird()
```

<br>

### 1115. 交替打印 FooBar / Print FooBar Alternately

[CN](https://leetcode.cn/problems/print-foobar-alternately/) · [EN](https://leetcode.com/problems/print-foobar-alternately/) · Medium

**题意**：两个线程分别循环 n 次打印 `foo` 和 `bar`，要求输出 `foobar` 重复 n 次。

**思路**：两个信号量交替“传球”。`foo_sem` 初始 1（foo 先走），`bar_sem` 初始 0。

```python
from threading import Semaphore

class FooBar:
    def __init__(self, n):
        self.n = n
        self.foo_sem = Semaphore(1)  # 1. foo 先走
        self.bar_sem = Semaphore(0)  # 2. bar 等 foo

    def foo(self, printFoo: 'Callable[[], None]') -> None:
        for _ in range(self.n):
            self.foo_sem.acquire()  # 3. 等轮到 foo
            printFoo()
            self.bar_sem.release()  # 4. 把球传给 bar

    def bar(self, printBar: 'Callable[[], None]') -> None:
        for _ in range(self.n):
            self.bar_sem.acquire()
            printBar()
            self.foo_sem.release()  # 5. 把球传回 foo
```

<br>

### 1116. 打印零与奇偶数 / Print Zero Even Odd

[CN](https://leetcode.cn/problems/print-zero-even-odd/) · [EN](https://leetcode.com/problems/print-zero-even-odd/) · Medium

**题意**：三个线程分别调用 `zero / even / odd`，输出 `010203...0n`。

**思路**：`zero` 打印 n 次 0，每次打完根据下一个数的奇偶把球传给 `odd` 或 `even`；它们打完再把球传回 `zero`。

```python
from threading import Semaphore

class ZeroEvenOdd:
    def __init__(self, n):
        self.n = n
        self.z = Semaphore(1)  # 1. zero 先走
        self.e = Semaphore(0)
        self.o = Semaphore(0)

    def zero(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1):
            self.z.acquire()
            printNumber(0)
            # 2. 下一个数是 i：奇数交给 odd，偶数交给 even
            (self.o if i % 2 else self.e).release()

    def even(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(2, self.n + 1, 2):
            self.e.acquire()
            printNumber(i)
            self.z.release()  # 3. 球传回 zero

    def odd(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1, 2):
            self.o.acquire()
            printNumber(i)
            self.z.release()
```

<br>

### 1117. H2O 生成 / Building H2O

[CN](https://leetcode.cn/problems/building-h2o/) · [EN](https://leetcode.com/problems/building-h2o/) · Medium

**题意**：氢线程调用 `hydrogen`、氧线程调用 `oxygen`，每 3 个线程（2 个 H、1 个 O）组成一个水分子后一起放行，不同分子的线程不能交错。

**思路**：`Semaphore(2)` 限制同时进入的 H 不超过 2 个，`Semaphore(1)` 限制 O 不超过 1 个；`Barrier(3)` 等凑齐 3 个线程再一起放行，放行后各自释放名额。

```python
from threading import Barrier, Semaphore

class H2O:
    def __init__(self):
        self.h = Semaphore(2)    # 1. 一个分子最多 2 个 H
        self.o = Semaphore(1)    # 2. 一个分子最多 1 个 O
        self.b = Barrier(3)      # 3. 凑齐 3 个再放行

    def hydrogen(self, releaseHydrogen: 'Callable[[], None]') -> None:
        self.h.acquire()
        self.b.wait()            # 4. 等另外两个原子
        releaseHydrogen()
        self.h.release()         # 5. 释放名额给下一个分子

    def oxygen(self, releaseOxygen: 'Callable[[], None]') -> None:
        self.o.acquire()
        self.b.wait()
        releaseOxygen()
        self.o.release()
```

<br>

### 1195. 交替打印字符串 / Fizz Buzz Multithreaded

[CN](https://leetcode.cn/problems/fizz-buzz-multithreaded/) · [EN](https://leetcode.com/problems/fizz-buzz-multithreaded/) · Medium

**题意**：四个线程分别负责输出 `fizz`（3 的倍数）、`buzz`（5 的倍数）、`fizzbuzz`（15 的倍数）、数字，按 1..n 的顺序输出。

**思路**：`number` 线程当调度者，遍历 1..n：普通数字自己打印；其它情况把球传给对应线程，并等它打完传回来。其它三个线程只需要按自己负责的数的个数循环。

```python
from threading import Semaphore

class FizzBuzz:
    def __init__(self, n: int):
        self.n = n
        self.f = Semaphore(0)
        self.b = Semaphore(0)
        self.fb = Semaphore(0)
        self.done = Semaphore(0)  # 1. 其它线程打完后通知调度者

    def fizz(self, printFizz: 'Callable[[], None]') -> None:
        # 2. 3 的倍数但不是 15 的倍数，共多少个就循环多少次
        for _ in range(self.n // 3 - self.n // 15):
            self.f.acquire()
            printFizz()
            self.done.release()

    def buzz(self, printBuzz: 'Callable[[], None]') -> None:
        for _ in range(self.n // 5 - self.n // 15):
            self.b.acquire()
            printBuzz()
            self.done.release()

    def fizzbuzz(self, printFizzBuzz: 'Callable[[], None]') -> None:
        for _ in range(self.n // 15):
            self.fb.acquire()
            printFizzBuzz()
            self.done.release()

    def number(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1):
            # 3. 调度：按 i 选出负责的线程
            if i % 15 == 0:
                sem = self.fb
            elif i % 3 == 0:
                sem = self.f
            elif i % 5 == 0:
                sem = self.b
            else:
                printNumber(i)
                continue
            sem.release()
            self.done.acquire()  # 4. 等它打完再继续，保证顺序
```

<br>

### 1226. 哲学家进餐 / The Dining Philosophers

[CN](https://leetcode.cn/problems/the-dining-philosophers/) · [EN](https://leetcode.com/problems/the-dining-philosophers/) · Medium

**题意**：5 个哲学家围坐，每人左右各一把叉子，要同时拿到两把才能吃。实现 `wantsToEat`，不能死锁。

**思路**：死锁的原因是“每人都拿了左叉、等右叉”，形成环。打破环的经典方法：给叉子编号，每人**先拿编号小的叉子**。这样最后一个人的拿取顺序和其他人相反，不会成环。

```python
from threading import Lock

class DiningPhilosophers:
    def __init__(self):
        self.forks = [Lock() for _ in range(5)]  # 1. 5 把叉子

    def wantsToEat(self, philosopher: int,
                   pickLeftFork: 'Callable[[], None]', pickRightFork: 'Callable[[], None]',
                   eat: 'Callable[[], None]',
                   putLeftFork: 'Callable[[], None]', putRightFork: 'Callable[[], None]') -> None:
        left, right = philosopher, (philosopher + 1) % 5
        # 2. 先锁编号小的，再锁编号大的，打破循环等待
        first, second = min(left, right), max(left, right)
        with self.forks[first], self.forks[second]:
            pickLeftFork()
            pickRightFork()
            eat()
            putLeftFork()
            putRightFork()
```

其它解法：最多允许 4 个人同时拿叉子（`Semaphore(4)`）；或者用一把全局锁让拿两把叉子成为原子操作。

<br>

### 1188. 设计有限阻塞队列 / Design Bounded Blocking Queue 🔒

[CN](https://leetcode.cn/problems/design-bounded-blocking-queue/) · [EN](https://leetcode.com/problems/design-bounded-blocking-queue/) · Medium

**题意**：线程安全的有界队列：`enqueue` 时队列满就阻塞，`dequeue` 时队列空就阻塞，`size()` 返回元素个数。

**思路**：生产者-消费者模型。一把锁 + 两个条件变量 `not_full`、`not_empty`。等待条件必须写成 `while` 而不是 `if`，防止虚假唤醒。

```python
from threading import Condition, Lock

class BoundedBlockingQueue(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.q = deque()
        lock = Lock()
        self.not_full = Condition(lock)   # 1. 两个条件共用一把锁
        self.not_empty = Condition(lock)

    def enqueue(self, element: int) -> None:
        with self.not_full:
            # 2. 满了就等，while 防虚假唤醒
            while len(self.q) == self.capacity:
                self.not_full.wait()
            self.q.append(element)
            self.not_empty.notify()  # 3. 唤醒一个消费者

    def dequeue(self) -> int:
        with self.not_empty:
            while not self.q:
                self.not_empty.wait()
            x = self.q.popleft()
            self.not_full.notify()   # 4. 唤醒一个生产者
            return x

    def size(self) -> int:
        return len(self.q)
```

<br>

### 1242. 多线程网络爬虫 / Web Crawler Multithreaded 🔒

[CN](https://leetcode.cn/problems/web-crawler-multithreaded/) · [EN](https://leetcode.com/problems/web-crawler-multithreaded/) · Medium

**题意**：从 `startUrl` 开始，用多线程爬取所有和它同一主机名的链接（`htmlParser.getUrls(url)` 返回页面中的链接，调用较慢）。

**思路**：BFS，每一层的 `getUrls` 交给线程池并发执行；`visited` 只在主线程里读写，就不需要加锁。

```python
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        # 1. 主机名：去掉 "http://" 后取第一个 '/' 之前的部分
        host = startUrl.split('/')[2]
        visited = {startUrl}
        frontier = [startUrl]
        with ThreadPoolExecutor(max_workers=16) as pool:
            while frontier:
                # 2. 这一层的所有页面并发抓取
                results = pool.map(htmlParser.getUrls, frontier)
                frontier = []
                # 3. 只在主线程里更新 visited，无需加锁
                for urls in results:
                    for u in urls:
                        if u not in visited and u.split('/')[2] == host:
                            visited.add(u)
                            frontier.append(u)
        return list(visited)
```

按层同步的缺点是一层里最慢的页面会拖住整层；更彻底的做法是每个任务完成后立刻提交新任务，并用锁保护 `visited`。

<br>

### 1279. 红绿灯路口 / Traffic Light Controlled Intersection 🔒

[CN](https://leetcode.cn/problems/traffic-light-controlled-intersection/) · [EN](https://leetcode.com/problems/traffic-light-controlled-intersection/) · Easy

**题意**：两条路（1 是南北向、2 是东西向）交汇，初始路 1 绿灯。车辆到达时调用 `carArrived`，如果自己这条路是红灯就先 `turnGreen()` 切换，再 `crossCar()`。保证不会两条路同时有车通过、也不会无意义地切换。

**思路**：一把锁让“检查灯、切换、通过”成为原子操作，再记录当前绿灯的道路。

```python
from threading import Lock

class TrafficLight:
    def __init__(self):
        self.green = 1      # 1. 当前绿灯的道路
        self.lock = Lock()

    def carArrived(self, carId: int, roadId: int, direction: int,
                   turnGreen: 'Callable[[], None]', crossCar: 'Callable[[], None]') -> None:
        with self.lock:
            # 2. 自己这条路是红灯才切换
            if self.green != roadId:
                turnGreen()
                self.green = roadId
            crossCar()
```

<br>
<br>

## 附录

正文之外的补充：开放式系统设计（LeetCode 上没有原题）、北美高频题索引、安防场景变形题，以及按优先级排好的刷题清单。

### 附录 A 监控类系统设计

这些是开放式的系统设计题，LeetCode 上没有对应题号。右列的 LeetCode 题只能练到其中的核心数据结构，采集、存储选型、降采样、分布式计数等考点要另外准备。

| 题目 | 核心考点 | 可练的 LeetCode 题 |
|---|---|---|
| Design a Metrics Monitoring & Alerting System（类似 Datadog、Prometheus） | 最经典的一道：时序数据库、数据采集（push 还是 pull）、聚合、告警规则、降采样 | [362. 敲击计数器](https://leetcode.cn/problems/design-hit-counter/)<br>[1348. 推文计数](https://leetcode.cn/problems/tweet-counts-per-frequency/)<br>[981. 基于时间的键值存储](https://leetcode.cn/problems/time-based-key-value-store/) |
| Design a Rate Limiter | 令牌桶、滑动窗口计数，在分布式环境下如何计数 | [359. 日志速率限制器](https://leetcode.cn/problems/logger-rate-limiter/)<br>[933. 最近的请求次数](https://leetcode.cn/problems/number-of-recent-calls/) |
| Design a Distributed Logging System（类似 ELK、Splunk） | 日志采集、消息队列（Kafka）、索引和检索 | [635. 设计日志存储系统](https://leetcode.cn/problems/design-log-storage-system/) |
| Design a Health Check / Heartbeat Service | 心跳机制、超时判定、怎么避免误报 | [1797. 设计一个验证系统](https://leetcode.cn/problems/design-authentication-manager/) |
| Design Top-K / Heavy Hitters（比如热门搜索词、Spotify 播放榜单） | 流式统计、Count-Min Sketch、分时间窗口聚合 | [1244. 力扣排行榜](https://leetcode.cn/problems/design-a-leaderboard/) |
| Design an Ad Click Aggregator | 实时计数、去重、Flink 之类的流处理 | [362. 敲击计数器](https://leetcode.cn/problems/design-hit-counter/)<br>[933. 最近的请求次数](https://leetcode.cn/problems/number-of-recent-calls/) |
| Design a Price Alert System（类似 camelcamelcamel） | 定时抓取价格、条件触发、通知推送 | [2034. 股票价格波动](https://leetcode.cn/problems/stock-price-fluctuation/) |
| Design a Web Crawler | 调度、去重、礼貌爬取（控制访问频率） | [1236. 网络爬虫](https://leetcode.cn/problems/web-crawler/)<br>[1242. 多线程网络爬虫](https://leetcode.cn/problems/web-crawler-multithreaded/) |

1236 和 1242 是会员题，属于 BFS / 并发标签，不在 Design 标签里。

如果只准备一道，就准备 Metrics Monitoring & Alerting。Alex Xu 的《System Design Interview Vol. 2》有专门的一章，同一卷还有 Ad Click Event Aggregation；Rate Limiter 和 Web Crawler 在 Vol. 1。

<br>

#### Design a Metrics Monitoring & Alerting System

类型：开放式系统设计（类似 Datadog、Prometheus）

设计一个指标监控与告警系统：收集大量服务器和服务上报的运行指标（CPU、内存、QPS、延迟），存成时间序列，支持按标签查询和聚合；指标满足告警规则时通知值班人员。

**需要支持：**

- 采集：每条数据是「指标名 + 标签 + 值 + 时间戳」，由机器主动上报（push）或系统定期拉取（pull）
- 查询：按指标名和标签筛选，在时间范围内按固定步长聚合（avg、max、sum、p99）
- 告警：规则形如「表达式 > 阈值并持续 N 分钟」，触发后去重，再通过邮件、短信、PagerDuty 等渠道通知
- 可视化：仪表盘按查询结果画图
- 不包括：日志和链路追踪

**示例：**

```text
写入  cpu.load{host=web-1, pool=web} 0.72 @ 1700000000
查询  avg(cpu.load{pool=web})，最近 1 小时，每 1 分钟一个点
告警  avg(cpu.load{pool=web}) > 0.9 持续 5 分钟 → 通知值班人员
```

**提示（Alex Xu 书中的规模假设）：**

- 1 亿日活用户
- 1000 个服务器池，每池 100 台机器，每台 100 个指标，约 1000 万个时间序列
- 数据保留 1 年：原始数据保留 7 天，之后降采样为 1 分钟粒度保留 30 天，再降为 1 小时粒度保留到 1 年
- 写多读少；查询延迟要低，告警要及时

```python
class MetricsSystem:

    def __init__(self):


    def write(self, name: str, tags: Dict[str, str], value: float, timestamp: int) -> None:


    def query(self, name: str, tags: Dict[str, str], start: int, end: int, step: int, agg: str) -> List[Tuple[int, float]]:


    def addAlertRule(self, ruleId: str, name: str, tags: Dict[str, str], threshold: float, duration: int) -> None:


    def evaluateAlerts(self, now: int) -> List[str]:



# Your MetricsSystem object will be instantiated and called as such:
# obj = MetricsSystem()
# obj.write(name,tags,value,timestamp)
# param_2 = obj.query(name,tags,start,end,step,agg)
# obj.addAlertRule(ruleId,name,tags,threshold,duration)
# param_4 = obj.evaluateAlerts(now)
```

<br>

#### Design a Rate Limiter

类型：开放式系统设计

设计一个限流器：限制每个客户端（按用户 ID、IP 或 API key 区分）在一个时间窗口内的请求数，超出的请求直接拒绝，返回 HTTP 429。

**需要支持：**

- 规则可配置，例如每个用户每秒最多 10 次、每个 IP 每分钟最多 100 次
- 判断要快，不能明显增加请求延迟
- 分布式：多台网关服务器共享同一份计数
- 被限流时告诉客户端还剩多少次、多久后可以重试（如 `X-Ratelimit-Remaining`、`X-Ratelimit-Retry-After` 响应头）
- 限流器自身故障时不能拖垮整个服务

**示例：**

```text
规则  每个 user_id 每秒最多 2 次
t=1.0  allow("u1") → true
t=1.3  allow("u1") → true
t=1.6  allow("u1") → false（返回 429）
t=2.1  allow("u1") → true
```

**提示：**

- 常见算法：令牌桶、漏桶、固定窗口计数、滑动窗口日志、滑动窗口计数
- 计数一般放在 Redis 等共享缓存里，用 Lua 脚本或原子操作避免并发下的竞态
- 固定窗口在窗口边界附近可能放过两倍流量，滑动窗口可以避免

```python
class RateLimiter:

    def __init__(self, limit: int, window: int):


    def allow(self, clientId: str, timestamp: float) -> bool:



# Your RateLimiter object will be instantiated and called as such:
# obj = RateLimiter(limit,window)
# param_1 = obj.allow(clientId,timestamp)
```

<br>

#### Design a Distributed Logging System

类型：开放式系统设计（类似 ELK、Splunk）

设计一个集中式日志系统：成千上万台机器上的服务不断产生日志，系统负责采集、传输、存储，并支持按服务、级别、时间范围和关键词检索。

**需要支持：**

- 采集：每台机器上的 agent 读取日志文件并发送
- 缓冲：用消息队列（如 Kafka）削峰，并把采集和存储解耦
- 存储与索引：按时间分片存储，建倒排索引支持全文检索
- 查询：按服务、级别、时间范围和关键词搜索，结果按时间排序
- 保留策略：近期数据放快存储，过期数据转到冷存储或删除

**示例：**

```text
写入  2026-11-15T10:00:01Z service=order level=ERROR "payment timeout uid=42"
查询  service=order AND level=ERROR AND "timeout"，时间 10:00–10:05
返回  按时间倒序排列的匹配日志
```

**提示：**

- 写入量远大于查询量
- 日志写入后几秒内可查即可，但不能丢
- 搜索要支持全文关键词，不只是字段精确匹配

```python
class LoggingService:

    def __init__(self):


    def ingest(self, service: str, level: str, message: str, timestamp: int) -> None:


    def search(self, query: str, start: int, end: int, limit: int) -> List[str]:



# Your LoggingService object will be instantiated and called as such:
# obj = LoggingService()
# obj.ingest(service,level,message,timestamp)
# param_2 = obj.search(query,start,end,limit)
```

<br>

#### Design a Health Check / Heartbeat Service

类型：开放式系统设计

设计一个健康检查服务：持续判断集群里每个节点（或服务实例）是否存活，把故障节点从负载均衡里摘掉，恢复后再加回来。

**需要支持：**

- 心跳：节点每隔固定时间上报一次（push），或检查服务定期探测节点（pull）
- 超时判定：连续多个周期收不到心跳才判定为故障
- 避免误报：网络抖动、GC 停顿、检查服务自身出故障，都不应导致大批节点被误摘
- 状态变化时通知订阅方（负载均衡、服务发现、告警）

**示例：**

```text
心跳间隔 5 秒，超时 15 秒
t=0   heartbeat("node-1")
t=5   heartbeat("node-1")
t=12  status("node-1") → ALIVE
t=21  status("node-1") → DEAD（距上次心跳 16 秒）
```

**提示：**

- 检查服务本身要多副本部署，否则它就是单点
- 节点数可达数万，找超时节点要按时间分桶或用有序结构，不能每次全量扫描

```python
class HeartbeatMonitor:

    def __init__(self, timeout: int):


    def heartbeat(self, nodeId: str, timestamp: int) -> None:


    def status(self, nodeId: str, timestamp: int) -> str:


    def deadNodes(self, timestamp: int) -> List[str]:



# Your HeartbeatMonitor object will be instantiated and called as such:
# obj = HeartbeatMonitor(timeout)
# obj.heartbeat(nodeId,timestamp)
# param_2 = obj.status(nodeId,timestamp)
# param_3 = obj.deadNodes(timestamp)
```

<br>

#### Design Top-K / Heavy Hitters

类型：开放式系统设计（如热门搜索词、Spotify 播放榜单）

设计一个实时热门榜单：从高速事件流（搜索、播放、观看）中，统计最近一段时间内出现次数最多的 K 个元素。

**需要支持：**

- 按不同时间窗口查询 Top K，如最近 1 分钟、1 小时、1 天
- 结果允许少量误差，但真正的热门元素不能漏
- 事件量极大，不能给每个元素都存一个精确计数

**示例：**

```text
事件流  play(A) ×5, play(B) ×3, play(C) ×1（都在最近 1 分钟内）
topK(k=2, window=1min) → [A, B]
```

**提示：**

- 近似计数用 Count-Min Sketch，再用大小为 K 的最小堆维护候选
- 需要精确结果时，用离线批处理（如 Spark）定期重算
- 每个分片各算 Top K，再合并各分片的结果

```python
class HeavyHitters:

    def __init__(self, k: int):


    def add(self, item: str, timestamp: int) -> None:


    def topK(self, window: int, timestamp: int) -> List[str]:



# Your HeavyHitters object will be instantiated and called as such:
# obj = HeavyHitters(k)
# obj.add(item,timestamp)
# param_2 = obj.topK(window,timestamp)
```

<br>

#### Design an Ad Click Aggregator

类型：开放式系统设计

设计一个广告点击聚合系统：实时接收点击事件，按广告 ID 统计点击次数，供广告主看报表和计费。

**需要支持：**

- 查询某个广告最近 M 分钟的点击数
- 查询每分钟点击最多的 N 个广告
- 按国家、IP、用户 ID 等维度过滤
- 结果用于计费，必须准确：同一次点击不能重复计数，也不能丢
- 能处理迟到的事件，发现错误后能重算

**示例：**

```text
click(ad=1, user=u1, t=10:00:05)
click(ad=1, user=u2, t=10:00:20)
click(ad=2, user=u1, t=10:00:40)
count(ad=1, 10:00–10:01) → 2
topN(n=1, minute=10:00) → [1]
```

**提示（Alex Xu 书中的规模假设）：**

- 每天 10 亿次点击，共 200 万个广告，点击量每年增长 30%
- 平均约 1 万 QPS，峰值按 5 倍估算为 5 万 QPS
- 流处理（如 Flink）按 1 分钟窗口聚合；原始事件也要保存，用来对账和重算
- 去重靠事件 ID 加精确一次（exactly-once）处理

```python
class AdClickAggregator:

    def __init__(self):


    def click(self, adId: int, userId: str, timestamp: int) -> None:


    def count(self, adId: int, start: int, end: int) -> int:


    def topN(self, n: int, minute: int) -> List[int]:



# Your AdClickAggregator object will be instantiated and called as such:
# obj = AdClickAggregator()
# obj.click(adId,userId,timestamp)
# param_2 = obj.count(adId,start,end)
# param_3 = obj.topN(n,minute)
```

<br>

#### Design a Price Alert System

类型：开放式系统设计（类似 camelcamelcamel）

设计一个降价提醒系统：用户订阅某个商品并设定目标价，商品价格降到目标价或以下时通知用户。

**需要支持：**

- 用户添加、删除提醒：（用户, 商品, 目标价）
- 定期抓取商品价格，热门商品抓得更勤
- 价格达到目标价时通知用户（邮件、推送），同一个提醒只通知一次
- 展示商品的历史价格曲线

**示例：**

```text
subscribe(user=u1, item=X, target=80)
subscribe(user=u2, item=X, target=95)
updatePrice(item=X, price=100) → []
updatePrice(item=X, price=90)  → [u2]
updatePrice(item=X, price=75)  → [u1]（u2 已通知过）
```

**提示：**

- 商品和订阅都很多。每个商品的订阅按目标价从高到低排好，价格更新时从头扫到目标价低于新价格为止
- 抓取要遵守目标网站的访问频率限制
- 通知走消息队列异步发送，失败重试

```python
class PriceAlert:

    def __init__(self):


    def subscribe(self, userId: str, itemId: str, target: int) -> None:


    def unsubscribe(self, userId: str, itemId: str) -> None:


    def updatePrice(self, itemId: str, price: int) -> List[str]:



# Your PriceAlert object will be instantiated and called as such:
# obj = PriceAlert()
# obj.subscribe(userId,itemId,target)
# obj.unsubscribe(userId,itemId)
# param_3 = obj.updatePrice(itemId,price)
```

<br>

#### Design a Web Crawler

类型：开放式系统设计

设计一个网页爬虫：从一组种子 URL 出发，下载网页、解析出新链接、继续抓取，为搜索引擎收集大量网页。

**需要支持：**

- 调度（URL Frontier）：决定下一个抓哪个 URL，按优先级排队
- 去重：同一个 URL 不抓两次，内容相同的页面不重复存
- 礼貌爬取：同一个域名同一时间只开一个连接，访问间隔受控，遵守 robots.txt
- 健壮性：处理超时、重定向、无限生成链接的陷阱页面（spider trap）
- 可扩展：多台机器并行抓取

**示例：**

```text
种子  https://a.com
抓取 a.com    → 发现 a.com/1、a.com/2、b.com
抓取 a.com/1  → 发现 a.com（已抓过，跳过）
抓取 b.com    → 与 a.com 不同域名，可以并行
```

**提示（Alex Xu 书中的规模假设）：**

- 每月抓取 10 亿个网页，平均约 400 QPS，峰值 800 QPS
- 平均网页 500 KB，每月约 500 TB，保存 5 年约 30 PB
- URL 去重用哈希集合或 Bloom Filter；内容去重比较页面哈希
- Frontier 按域名拆成多个队列，每个队列只由一个工作线程消费，保证礼貌

```python
class WebCrawler:

    def __init__(self, seeds: List[str]):


    def nextUrl(self) -> str:


    def report(self, url: str, links: List[str], contentHash: str) -> None:



# Your WebCrawler object will be instantiated and called as such:
# obj = WebCrawler(seeds)
# param_1 = obj.nextUrl()
# obj.report(url,links,contentHash)
```

<br>
<br>

### 附录 B 北美高频设计题

北美大厂（Amazon、Meta、Google、Microsoft、Bloomberg 等）面试中出现最多的设计题。频率参考 LeetCode 公司标签和一亩三分地面经，具体数字需要 LeetCode 会员才能看到，排序只作大致参考。题解都在正文对应分类里。

#### 第一梯队：几乎每家都考

| 题目 | 常见公司 | 核心结构 | 本文位置 |
|---|---|---|---|
| 146. LRU 缓存 | Amazon, Meta, Microsoft, Bloomberg, Google | 哈希表 + 双向链表 | [1. 缓存类](#1-缓存类) |
| 380. O(1) 时间插入、删除和获取随机元素 | Meta, Amazon, LinkedIn, Google | 数组 + 值到下标的哈希表 | [3. 随机化](#3-随机化--o1-操作) |
| 981. 基于时间的键值存储 | Google, Amazon, Uber, Netflix | 哈希表 + 二分 | [6. 区间与范围查询](#6-区间与范围查询) |
| 295. 数据流的中位数 | Amazon, Google, Microsoft | 大顶堆 + 小顶堆 | [5. 数据流与统计](#5-数据流与统计) |
| 155. 最小栈 | Amazon, Bloomberg, Microsoft | 辅助栈 | [2. 基础数据结构实现](#2-基础数据结构实现) |
| 208. 实现 Trie | Amazon, Google, Microsoft | Trie | [7. 字符串 / 字典树](#7-字符串--字典树) |
| 1396. 设计地铁系统 | Bloomberg（非常高频）, Amazon | 两个哈希表 | [8. 业务系统模拟](#8-业务系统模拟) |
| 460. LFU 缓存 | Amazon, Google | 哈希表 + 每个频率一条链表 | [1. 缓存类](#1-缓存类) |

#### 第二梯队：特定公司高频

| 题目 | 常见公司 | 核心结构 | 本文位置 |
|---|---|---|---|
| 1570. 两个稀疏向量的点积 🔒 | Meta（排名靠前） | 哈希表 / 下标值对 + 双指针 | [2. 基础数据结构实现](#2-基础数据结构实现) |
| 341. 扁平化嵌套列表迭代器 | Meta, LinkedIn, Airbnb | 栈 | [4. 迭代器](#4-迭代器) |
| 173. 二叉搜索树迭代器 | Meta, Microsoft | 栈模拟中序遍历 | [4. 迭代器](#4-迭代器) |
| 642. 设计搜索自动补全系统 🔒 | Amazon, Microsoft, Google | Trie + 热度 | [7. 字符串 / 字典树](#7-字符串--字典树) |
| 588. 设计内存文件系统 🔒 | Amazon, Microsoft | 按路径的 Trie | [7. 字符串 / 字典树](#7-字符串--字典树) |
| 1166. 设计文件系统 🔒 | Amazon, Airbnb | 哈希表 / Trie | [7. 字符串 / 字典树](#7-字符串--字典树) |
| 355. 设计推特 | Amazon, X (Twitter) | 哈希表 + 堆合并 | [8. 业务系统模拟](#8-业务系统模拟) |
| 362. 敲击计数器 🔒 | Dropbox, Google, Uber | 队列 / 分桶 | [5. 数据流与统计](#5-数据流与统计) |
| 359. 日志速率限制器 🔒 | Google, Microsoft | 哈希表 | [5. 数据流与统计](#5-数据流与统计) |
| 348. 设计井字棋 🔒 | Google, Microsoft, Meta | 行、列、对角线计数 | [8. 业务系统模拟](#8-业务系统模拟) |
| 353. 设计贪吃蛇游戏 🔒 | Google, Amazon | 双端队列 + 集合 | [8. 业务系统模拟](#8-业务系统模拟) |
| 1146. 快照数组 | Google | 每个下标一个列表 + 二分 | [6. 区间与范围查询](#6-区间与范围查询) |
| 2034. 股票价格波动 | Google, Atlassian | 有序集合 / 堆 + 懒删除 | [5. 数据流与统计](#5-数据流与统计) |
| 1244. 力扣排行榜 🔒 | Bloomberg, Wayfair | 哈希表 + 有序集合 | [8. 业务系统模拟](#8-业务系统模拟) |
| 1472. 设计浏览器历史记录 | Bloomberg, Roblox | 数组 + 指针 | [8. 业务系统模拟](#8-业务系统模拟) |
| 716. 最大栈 🔒 | LinkedIn, Amazon | 栈 + 堆 + 懒删除 | [2. 基础数据结构实现](#2-基础数据结构实现) |
| 895. 最大频率栈 | Amazon | 每个频率一个栈 | [2. 基础数据结构实现](#2-基础数据结构实现) |
| 284. 窥视迭代器 | Google, Apple | 预取一个元素 | [4. 迭代器](#4-迭代器) |
| 729. 我的日程安排表 I | Google | 有序集合 + 二分 | [6. 区间与范围查询](#6-区间与范围查询) |
| 1188. 设计有限阻塞队列 🔒 | Microsoft, LinkedIn | Condition | [9. 多线程 / 并发](#9-多线程--并发) |
| 1242. 多线程网络爬虫 🔒 | Dropbox, Databricks | 线程池 + 加锁的 visited | [9. 多线程 / 并发](#9-多线程--并发) |

<br>

### 附录 C 安防场景变形题

准备 Ambient.ai 面试时整理的三道变形题：把常见的数据结构题套进安防摄像头的场景。LeetCode 上没有原题，下面的题面、示例和约束是自己定的。

| 题目 | 考点 | 可练的 LeetCode 题 |
|---|---|---|
| Event Deduplication（事件去重） | 哈希表、排序、滑动窗口 | [359. 日志速率限制器](https://leetcode.cn/problems/logger-rate-limiter/)（会员题）<br>[933. 最近的请求次数](https://leetcode.cn/problems/number-of-recent-calls/) |
| Top K Alerts（最高频告警） | 哈希表、堆、队列、过期数据清理 | [362. 敲击计数器](https://leetcode.cn/problems/design-hit-counter/)（会员题）<br>[347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/) |
| Group Nearby Cameras（摄像头分组） | 图、并查集、DFS / BFS | [547. 省份数量](https://leetcode.cn/problems/number-of-provinces/)<br>[721. 账户合并](https://leetcode.cn/problems/accounts-merge/) |

第三道尤其值得准备：Glassdoor 上 2026 年 6 月一份 Ambient.ai Applied Scientist 面经里出现过按距离给点分组的题。岗位不同，但这比只根据简历猜题更直接。

<br>

#### Event Deduplication：事件去重

类型：变形题（哈希表 + 排序 + 滑动窗口）

摄像头会产生大量重复事件。给定一批事件 `events[i] = (cameraId, timestamp, eventType)`，对同一个摄像头的同一类事件去重：某条事件被保留后，同一摄像头、同一类型、时间戳在 `[t, t + window)` 内的后续事件都丢弃；时间戳到达 `t + window` 的事件算新事件，重新开始计时。

输入不保证按时间排序。返回被保留的事件，按时间戳从小到大排列；时间戳相同时保持输入中的先后顺序。

**示例 1：**

```text
输入
events = [("cam1",1,"person"), ("cam1",5,"person"), ("cam1",12,"person"),
          ("cam2",3,"person"), ("cam1",8,"vehicle"), ("cam1",11,"person")]
window = 10
输出
[("cam1",1,"person"), ("cam2",3,"person"), ("cam1",8,"vehicle"), ("cam1",11,"person")]
解释
cam1 的 person 在 1 被保留，5 落在 [1, 11) 内丢弃；11 正好到达 1 + 10，保留并重新计时；12 落在 [11, 21) 内丢弃。
cam2 的 person 和 cam1 的 vehicle 是不同的键，各自保留。
```

**示例 2：**

```text
输入
events = [("cam1",20,"person"), ("cam1",0,"person"), ("cam1",9,"person"), ("cam1",10,"person")]
window = 10
输出
[("cam1",0,"person"), ("cam1",10,"person"), ("cam1",20,"person")]
解释
先按时间排序：0 保留，9 丢弃，10 保留，20 保留。
```

**提示：**

- `1 <= events.length <= 10^5`
- `0 <= timestamp <= 10^9`
- `1 <= window <= 10^6`
- `cameraId` 和 `eventType` 是非空字符串

**进阶：** 如果事件是实时流入、并且可能乱序到达（最多晚到 5 秒），怎么做？

```python
class Solution:
    def dedupEvents(self, events: List[Tuple[str, int, str]], window: int) -> List[Tuple[str, int, str]]:
```

<br>

#### Top K Alerts：最高频告警

类型：变形题（哈希表 + 堆 + 队列）

实时接收安防告警，随时查询最近 5 分钟（300 秒）内告警最多的 `k` 个摄像头。

实现 `AlertTracker` 类：

- `AlertTracker(int window)`：用时间窗口长度 `window`（秒）初始化。
- `void record(String cameraId, int timestamp)`：摄像头 `cameraId` 在 `timestamp` 产生一条告警。
- `List<String> topK(int k, int timestamp)`：返回时间范围 `(timestamp - window, timestamp]` 内告警数最多的 `k` 个摄像头，按告警数从多到少排列；告警数相同时按 `cameraId` 字典序从小到大。有告警的摄像头不足 `k` 个时，全部返回。

所有调用按时间顺序发生，`timestamp` 单调不减。

**示例：**

```text
输入
["AlertTracker","record","record","record","record","record","record","topK","topK","topK"]
[[300],["A",1],["B",2],["A",3],["C",100],["C",200],["C",250],[2,250],[2,302],[3,400]]
输出
[null,null,null,null,null,null,null,["C","A"],["C","A"],["C"]]

解释
topK(2, 250)：窗口 (-50, 250]，C 3 条、A 2 条、B 1 条，返回 ["C","A"]
topK(2, 302)：窗口 (2, 302]，A 在 1 和 B 在 2 的告警过期，C 3 条、A 1 条，返回 ["C","A"]
topK(3, 400)：窗口 (100, 400]，只剩 C 在 200 和 250 的 2 条，返回 ["C"]
```

**提示：**

- `1 <= window <= 3600`
- `1 <= k <= 1000`
- `0 <= timestamp <= 10^9`
- `record` 和 `topK` 最多共调用 `10^5` 次

**进阶：** 摄像头有上百万个、`topK` 调用非常频繁时，怎么避免每次查询都对所有摄像头排序？

```python
class AlertTracker:

    def __init__(self, window: int):


    def record(self, cameraId: str, timestamp: int) -> None:


    def topK(self, k: int, timestamp: int) -> List[str]:



# Your AlertTracker object will be instantiated and called as such:
# obj = AlertTracker(window)
# obj.record(cameraId,timestamp)
# param_2 = obj.topK(k,timestamp)
```

<br>

#### Group Nearby Cameras：摄像头分组

类型：变形题（图 + 并查集）

给定 `n` 个摄像头的平面坐标 `points[i] = [x_i, y_i]` 和距离阈值 `d`。两个摄像头的欧几里得距离不超过 `d` 时，认为它们直接相连；相连关系可以传递（A 连 B、B 连 C，则 A、B、C 在同一组）。返回分组的数量。

**示例 1：**

```text
输入
points = [[0,0],[3,4],[10,0],[13,4],[50,50]], d = 5
输出
3
解释
(0,0) 和 (3,4) 距离 5，相连；(10,0) 和 (13,4) 距离 5，相连；(3,4) 和 (10,0) 距离约 8.06，不相连。
分组为 {(0,0),(3,4)}、{(10,0),(13,4)}、{(50,50)}。
```

**示例 2：**

```text
输入
points = [[0,0],[3,4],[10,0],[13,4],[50,50]], d = 9
输出
2
解释
(3,4) 和 (10,0) 距离约 8.06 <= 9，把前两组连成一组。
```

**示例 3：**

```text
输入
points = [[1,1]], d = 0
输出
1
```

**提示：**

- `1 <= n <= 2000`
- `-10^4 <= x_i, y_i <= 10^4`
- `0 <= d <= 10^4`
- 比较距离时用平方，避免浮点误差：`dx*dx + dy*dy <= d*d`

**进阶：** 如果 `n` 到 `10^5`，O(n²) 两两比较太慢，怎么做？（提示：按边长为 `d` 的网格分桶，每个点只和自己及相邻 8 个格子里的点比较。）

```python
class Solution:
    def countGroups(self, points: List[List[int]], d: int) -> int:
```

<br>

#### 3 小时冲刺

只有 3 小时时，按这个顺序练：

| 时间 | 题目 | 本文位置 |
|---|---|---|
| 第一小时 | [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/) | [1. 缓存类](#1-缓存类) |
| 第一小时 | [981. 基于时间的键值存储](https://leetcode.cn/problems/time-based-key-value-store/) | [6. 区间与范围查询](#6-区间与范围查询) |
| 第二小时 | [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) | [附录 D](#附录-d-按优先级刷题) |
| 第二小时 | [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/) | 不在本文，哈希表 + 堆 |
| 第三小时 | [721. 账户合并](https://leetcode.cn/problems/accounts-merge/) | [附录 D](#附录-d-按优先级刷题) |
| 第三小时 | Group Nearby Cameras | 本章 |

<br>
<br>

### 附录 D 按优先级刷题

按优先级排好的 8 道题，覆盖缓存、堆、在线数据处理、按时间查询、索引结构、窗口计算和并查集。已在前面章节出现的题，「本文位置」链接到对应章节；其余题目的完整题面放在本章。

| 优先级 | 题目 | 对应能力 | 本文位置 |
|---|---|---|---|
| 1 | [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/) | 内存管理、缓存结构 | [1. 缓存类](#1-缓存类) |
| 2 | [460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/) | 高级数据结构设计 | [1. 缓存类](#1-缓存类) |
| 3 | [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/) | 堆、多路归并 | 本章 |
| 4 | [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/) | 在线数据处理 | [5. 数据流与统计](#5-数据流与统计) |
| 5 | [981. 基于时间的键值存储](https://leetcode.cn/problems/time-based-key-value-store/) | 存储、按时间查询 | [6. 区间与范围查询](#6-区间与范围查询) |
| 6 | [1206. 设计跳表](https://leetcode.cn/problems/design-skiplist/) | 索引数据结构 | 本章 |
| 7 | [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) | 高性能窗口计算 | 本章 |
| 8 | [721. 账户合并](https://leetcode.cn/problems/accounts-merge/) | 并查集 | 本章 |

<br>

#### 23. 合并 K 个升序链表

[CN](https://leetcode.cn/problems/merge-k-sorted-lists/) · [EN](https://leetcode.com/problems/merge-k-sorted-lists/)

难度：Hard

给你 `k` 个链表，每个链表都已经按升序排好。把它们合并成一个升序链表并返回。

**示例 1：**

```text
输入
lists = [[1,4,5],[1,3,4],[2,6]]
输出
[1,1,2,3,4,4,5,6]
解释
三个链表分别是 1->4->5、1->3->4、2->6，合并后得到 1->1->2->3->4->4->5->6。
```

**示例 2：**

```text
输入
lists = []
输出
[]
```

**示例 3：**

```text
输入
lists = [[]]
输出
[]
```

**提示：**

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- `lists[i]` 按升序排列
- 所有 `lists[i].length` 之和不超过 `10^4`

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: list[ListNode | None]) -> ListNode | None:
```

<br>

#### 1206. 设计跳表

[CN](https://leetcode.cn/problems/design-skiplist/) · [EN](https://leetcode.com/problems/design-skiplist/)

难度：Hard

不使用任何库函数，设计一个跳表。

跳表能在 O(log n) 的平均时间内完成插入、删除和查找。它由多层有序链表组成：最底层包含所有元素，每往上一层，元素大约少一半，查找时从最高层开始，像二分一样快速跳过大段元素。

实现 `Skiplist` 类：

- `Skiplist()`：初始化跳表。
- `bool search(int target)`：返回 `target` 是否存在于跳表中。
- `void add(int num)`：插入一个元素 `num`。
- `bool erase(int num)`：删除一个值为 `num` 的元素。如果 `num` 不存在，返回 `false`；如果有多个，删除任意一个即可。

跳表中可能有重复的值，需要正确处理。

**示例：**

```text
输入
["Skiplist", "add", "add", "add", "search", "add", "search", "erase", "erase", "search"]
[[], [1], [2], [3], [0], [4], [1], [0], [1], [1]]
输出
[null, null, null, null, false, null, true, false, true, false]
```

**提示：**

- `0 <= num, target <= 2 * 10^4`
- `search`、`add`、`erase` 最多共调用 `5 * 10^4` 次

```python
class Skiplist:

    def __init__(self):


    def search(self, target: int) -> bool:


    def add(self, num: int) -> None:


    def erase(self, num: int) -> bool:



# Your Skiplist object will be instantiated and called as such:
# obj = Skiplist()
# param_1 = obj.search(target)
# obj.add(num)
# param_3 = obj.erase(num)
```

<br>

#### 239. 滑动窗口最大值

[CN](https://leetcode.cn/problems/sliding-window-maximum/) · [EN](https://leetcode.com/problems/sliding-window-maximum/)

难度：Hard

给你一个整数数组 `nums` 和一个大小为 `k` 的滑动窗口。窗口从数组最左边开始，每次向右移动一位，直到到达最右边。返回每个窗口位置上，窗口内 `k` 个数的最大值。

**示例 1：**

```text
输入
nums = [1,3,-1,-3,5,3,6,7], k = 3
输出
[3,3,5,5,6,7]
解释
窗口位置                      最大值
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```text
输入
nums = [1], k = 1
输出
[1]
```

**提示：**

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`
- `1 <= k <= nums.length`

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
```

<br>

#### 721. 账户合并

[CN](https://leetcode.cn/problems/accounts-merge/) · [EN](https://leetcode.com/problems/accounts-merge/)

难度：Medium

给定一个账户列表 `accounts`，`accounts[i][0]` 是账户的名称，其余元素是这个账户的邮箱地址。

两个账户只要有一个相同的邮箱，就属于同一个人；名称相同不代表是同一个人，因为不同的人可能重名。同一个人的所有账户名称都相同。

合并属于同一个人的账户。返回结果中每个账户的第一个元素是名称，其余是按 ASCII 顺序排序的邮箱。账户之间可以按任意顺序返回。

**示例 1：**

```text
输入
accounts = [["John","johnsmith@mail.com","john00@mail.com"],["John","johnnybravo@mail.com"],["John","johnsmith@mail.com","john_newyork@mail.com"],["Mary","mary@mail.com"]]
输出
[["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["John","johnnybravo@mail.com"],["Mary","mary@mail.com"]]
解释
第一个和第三个 John 都有 "johnsmith@mail.com"，是同一个人，合并。
第二个 John 和 Mary 的邮箱没有和其他账户重合，各自单独成组。
```

**示例 2：**

```text
输入
accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
输出
[["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]
```

**提示：**

- `1 <= accounts.length <= 1000`
- `2 <= accounts[i].length <= 10`
- `1 <= accounts[i][j].length <= 30`
- `accounts[i][0]` 由英文字母组成
- `accounts[i][j]`（`j > 0`）是有效的邮箱地址

```python
class Solution:
    def accountsMerge(self, accounts: list[list[str]]) -> list[list[str]]:
```
