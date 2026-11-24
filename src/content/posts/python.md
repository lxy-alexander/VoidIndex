---
title: vLLM 中的 Python 语言特性
description: 按 vLLM 源码中的出现频率整理 Python 语言特性：类与对象、数据容器、pydantic、类型系统、并发、标准库等，每项先讲何时使用，再给实战代码。
pubDatetime: 2026-11-14T08:30:00Z
tags:
  - python
  - vllm
draft: false
---

- [1. 类与对象](#1-类与对象)
  - `@classmethod`（2041）、`@staticmethod`（1587）、`@property`（1398）、`@cached_property`（128）
  - 魔法方法：`__call__`（276）；`__len__`、`__iter__`、`__getitem__`、`__eq__`、`__hash__` 等（115）
  - `__slots__`（14）、`__init_subclass__`（11）、`metaclass=`（4）
  - 自定义异常（49）
  - `copy` / `deepcopy`（85）
- [2. 数据容器](#2-数据容器)
  - `@dataclass`（912）、`field(default_factory=...)`（233）、`__post_init__`（95）
  - `NamedTuple`（197）、`TypedDict`（80）
  - `Enum`（98）+ `auto()`（113）
  - `types.MappingProxyType`、`SimpleNamespace`（`types` 模块共 36）
- [3. pydantic](#3-pydantic)
  - `BaseModel` 子类（183，几乎全在 `entrypoints/`）
  - `Field`、`Annotated[int, Field(ge=0)]`
  - `@model_validator`（88）、`@field_validator`（55）
  - `ConfigDict(extra="allow")`
  - `pydantic.dataclasses.dataclass` + `@config`（45，所有配置类）
- [4. 类型系统](#4-类型系统)
  - `from __future__ import annotations`（265）
  - `TYPE_CHECKING`（1051）
  - `collections.abc`：`Iterable`、`Callable`、`Sequence`、`Mapping`、`Iterator`、`Generator`（986）
  - `Any`（760）、`cast`（191）
  - `Literal`（552）、`ClassVar`（545）、`TypeAlias`（329）、`Annotated`（318）
  - `@override`（163）、`@overload`（150）、`Final` / `@final`（27）
  - `TypeVar`（132）、`Generic`（40）、`ParamSpec`（24）、`Self`（21）
  - `Protocol`（119）+ `@runtime_checkable`（37）
  - `TypeIs` / `TypeGuard`（12）、`assert_never`（9）、`NotRequired` / `Required`（5）
  - `get_args`、`get_origin`、`get_type_hints`（30）
- [5. 抽象与扩展](#5-抽象与扩展)
  - `ABC` + `@abstractmethod`（616）
  - 装饰器 + `functools.wraps`（45）
  - `functools`：`cache` / `lru_cache`（427）、`partial`（229）
  - 装饰器 + 注册表模式：`@CustomOp.register`、`@MULTIMODAL_REGISTRY.register_processor`
  - `importlib`（200）、模块级 `__getattr__`（15）、`lambda` 字典延迟求值（`envs.py`，505）
  - `inspect`（112）、`ast`（21）
  - deprecated 标记（126）
- [6. 上下文与流程控制](#6-上下文与流程控制)
  - `@contextmanager`（149）、`ExitStack`（37）、`@asynccontextmanager`（10）
  - `ContextVar`（24）
  - 生成器 `yield`、`yield from`（58）
  - 海象运算符 `:=`（653）、`match` / `case`（13）
- [7. 并发](#7-并发)
  - `async def`（631）、`AsyncGenerator`（125）
  - `asyncio.create_task`（44）、`gather`（44）、`Queue`（12）、`Event`（12）
  - `run_in_executor` / `to_thread`（28）
  - `threading.Thread`（60）、`Lock` / `Event` / `Condition`（109）、`queue.Queue`（28）
  - `ThreadPoolExecutor`（78）
  - `multiprocessing.get_context("spawn")`（15）
  - `signal`（20）、`atexit`（20）、`weakref`（79）
- [8. 标准库数据结构](#8-标准库数据结构)
  - `itertools`（219）、`defaultdict`（124）、`OrderedDict`（49）
  - `deque`（29）、`Counter`（23）、`bisect`（22）、`heapq`（14）
- [9. 常用标准库](#9-常用标准库)
  - `math`（251）、`os`（184）、`pathlib`（75）、`json`（171）、`regex`（145）
  - `time`（149）、`logging`（58）、`argparse`（65）、`sys`（59）
  - `hashlib`（40）、`uuid`（24）、`tempfile`（22）
- [10. 底层与系统](#10-底层与系统)
  - `pickle`（163）、`__reduce__` / `__getstate__`（12）
  - `ctypes`（382）、`struct`（108）
  - `mmap` / 共享内存（统计含误匹配）
  - `gc`（47）
- [11. 框架装饰器（只需认得）](#11-框架装饰器只需认得)
  - PyTorch：`@torch.inference_mode`（91）、`@torch.no_grad`（54）、`@torch.compile`（28）、`@register_fake`（44）
  - vLLM：`@support_torch_compile`（156）、`@traced`（293）
  - FastAPI：`@router.post`（61）
  - kernel DSL：`@triton.jit`（866）、`@triton.autotune`（52）、`@triton.heuristics`（78）、`@cute.jit`（394）、`@cute.kernel`（86）
- [12. Python 内置容器与函数（LeetCode 常考）](#12-python-内置容器与函数leetcode-常考)
- [13. 线程同步工具](#13-线程同步工具)
  - `Lock`、`RLock`
  - `Semaphore`、`BoundedSemaphore`
  - `Event`、`Condition`、`Barrier`

<br>
<br>

## 1. 类与对象

Python 的类比看上去更灵活：方法可以绑定到实例或类，属性读取可以被拦截，`+`、`len()`、`for` 这些语法都会转成对魔法方法的调用。这一章讲清这些机制。

<br>

### @classmethod

普通方法的第一个参数是实例 `self`，`@classmethod` 的第一个参数是类本身 `cls`。它不需要实例就能调用，最常见的用途是做「替代构造函数」。因为拿到的是实际调用的类，所以子类调用时会创建子类对象。

**实战：从不同来源创建配置**

```python
class ModelConfig:
    def __init__(self, name: str, max_len: int):
        self.name = name
        self.max_len = max_len

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["name"], d["max_len"])


class QwenConfig(ModelConfig):
    pass


cfg = QwenConfig.from_dict({"name": "qwen", "max_len": 4096})
assert type(cfg) is QwenConfig   # cls 是 QwenConfig，不是 ModelConfig
```

<br>

### @staticmethod

`@staticmethod` 既不接收 `self` 也不接收 `cls`，就是放在类命名空间里的普通函数。当一个工具函数和类逻辑相关、但不需要访问类或实例状态时使用。

**实战**

```python
class Tokenizer:
    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.split()).lower()


assert Tokenizer.normalize("  Hello   World ") == "hello world"
```

**怎么选：** 要访问实例状态用普通方法，要访问类或创建实例用 `@classmethod`，两者都不要用 `@staticmethod`。

<br>

### @property

`@property` 把方法变成「看起来像属性」的访问：写 `obj.x` 时实际调用一个方法。好处是接口保持简单，内部可以计算或校验。配合 `@x.setter` 可以拦截赋值；没有 setter 的属性是只读的。适合需要计算、校验或只读的属性，或者想把普通字段改成计算逻辑又不改调用方代码时。

**实战：计算属性与赋值校验**

```python
class Request:
    def __init__(self, prompt_len: int, output_len: int):
        self.prompt_len = prompt_len
        self._output_len = output_len

    @property
    def total_len(self) -> int:
        return self.prompt_len + self._output_len

    @property
    def output_len(self) -> int:
        return self._output_len

    @output_len.setter
    def output_len(self, value: int) -> None:
        if value < 0:
            raise ValueError("output_len must be >= 0")
        self._output_len = value


r = Request(10, 5)
assert r.total_len == 15      # 不用写括号
r.output_len = 7
assert r.total_len == 17
try:
    r.total_len = 1           # 没有 setter，只读
except AttributeError:
    pass
```

<br>

### @cached_property

`functools.cached_property` 第一次访问时计算一次，结果写进实例的 `__dict__`，之后直接读缓存。适合计算昂贵、且对象生命周期内结果不变的值。

**实战：延迟加载**

```python
from functools import cached_property


class Model:
    load_count = 0

    @cached_property
    def weights(self) -> list[int]:
        Model.load_count += 1
        return [1, 2, 3]


m = Model()
m.weights
m.weights
assert Model.load_count == 1      # 只计算一次
del m.weights                     # 删除缓存，下次重新计算
m.weights
assert Model.load_count == 2
```

**坑：** 依赖的数据变了，缓存不会自动失效；并且它需要实例有 `__dict__`，与 `__slots__` 不兼容。

<br>

### 魔法方法

魔法方法（dunder method）是 Python 语法和对象之间的协议。解释器把语法转成方法调用：`len(x)` 调 `x.__len__()`，`x[i]` 调 `x.__getitem__(i)`，`for` 调 `__iter__`，`x()` 调 `__call__`。实现这些方法，自定义对象就能像内置类型一样使用。当希望自定义对象支持 `len`、下标、`for` 循环、函数调用或作为字典键时实现对应方法。

| 语法 | 方法 |
|---|---|
| `obj(...)` | `__call__` |
| `len(obj)` | `__len__` |
| `iter(obj)` / `for` | `__iter__` |
| `obj[k]` | `__getitem__` |
| `a == b` | `__eq__` |
| `hash(obj)` / 作字典键 | `__hash__` |

**实战：`__call__` 让对象像函数一样调用**

对象既有状态又能调用。PyTorch 的 `model(x)` 就是靠 `nn.Module.__call__` 转到 `forward`。

```python
class Scaler:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, x: float) -> float:
        return x * self.factor


double = Scaler(2)
assert double(3) == 6
assert callable(double)
```

**实战：自定义序列与可哈希的键**

```python
class Batch:
    def __init__(self, items: list[str]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)


b = Batch(["a", "b"])
assert len(b) == 2 and b[0] == "a" and list(b) == ["a", "b"]


class CacheKey:
    def __init__(self, model: str, tokens: tuple[int, ...]):
        self.model = model
        self.tokens = tokens

    def __eq__(self, other) -> bool:
        if not isinstance(other, CacheKey):
            return NotImplemented
        return (self.model, self.tokens) == (other.model, other.tokens)

    def __hash__(self) -> int:
        return hash((self.model, self.tokens))


cache = {CacheKey("m", (1, 2)): "hit"}
assert cache[CacheKey("m", (1, 2))] == "hit"
```

**坑：** 定义 `__eq__` 而不定义 `__hash__`，类会变成不可哈希。相等的对象哈希值必须相同，且作键的字段不应再被修改。

<br>

### __slots__

普通实例用一个 `__dict__` 存属性，灵活但费内存。`__slots__` 声明固定的属性名，取消 `__dict__`，属性存到固定位置。对象数量很大时能省内存，同时禁止添加未声明的属性。

**实战：大量小对象**

```python
class Token:
    __slots__ = ("id", "logprob")

    def __init__(self, id: int, logprob: float):
        self.id = id
        self.logprob = logprob


t = Token(1, -0.5)
assert not hasattr(t, "__dict__")
try:
    t.text = "hi"
except AttributeError:
    pass
```

**坑：** 子类如果不声明 `__slots__`，实例会重新拥有 `__dict__`。

<br>

### __init_subclass__

每当定义一个子类，父类的 `__init_subclass__` 就会自动运行。它能在不写元类的情况下，在定义子类时做注册或检查。

**实战：子类自动注册并校验**

```python
class Backend:
    registry: dict[str, type] = {}

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.backend_name = name
        Backend.registry[name] = cls


class CudaBackend(Backend, name="cuda"):
    pass


assert Backend.registry["cuda"] is CudaBackend
assert CudaBackend.backend_name == "cuda"
```

<br>

### metaclass

对象由类创建，类由元类创建，默认元类是 `type`。`metaclass=` 可以自定义「类的创建过程」和类的行为。它功能强但难读，能用 `__init_subclass__` 或装饰器解决时不要用它。

**实战：单例**

```python
class Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        # Registry() 会先调到元类的 __call__
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Registry(metaclass=Singleton):
    pass


assert Registry() is Registry()
assert type(Registry) is Singleton
```

<br>

### 自定义异常

继承 `Exception` 就能定义自己的异常。调用方可以按类型精确捕获，而不用解析错误字符串。通常为项目定义一个基类，再按错误类别派生；异常还可以带结构化字段。

**实战：异常层次，以及保留原始原因**

```python
class EngineError(Exception):
    """项目内所有异常的基类"""


class PromptTooLongError(EngineError):
    def __init__(self, length: int, limit: int):
        super().__init__(f"prompt length {length} exceeds {limit}")
        self.length = length
        self.limit = limit


def check(length: int) -> None:
    if length > 100:
        raise PromptTooLongError(length, 100)


def load(path: str) -> None:
    try:
        open(path)
    except OSError as e:
        raise EngineError(f"cannot load {path}") from e  # 保留原因到 __cause__


try:
    check(200)
except EngineError as e:              # 用基类也能捕获
    assert e.limit == 100

try:
    load("/no/such/file")
except EngineError as e:
    assert isinstance(e.__cause__, FileNotFoundError)
```

<br>

### copy / deepcopy

赋值 `b = a` 不复制，两个名字指向同一对象。`copy.copy` 是浅拷贝：新建外层对象，内部元素仍然共享。`copy.deepcopy` 递归复制所有内容，互不影响，但更慢。需要改一份数据又不想影响原对象时使用：只改外层用浅拷贝，内层可变对象也要改就用深拷贝。

**实战：修改配置时不污染模板**

```python
import copy

template = {"model": "qwen", "stop": ["</s>"]}

shallow = copy.copy(template)
shallow["stop"].append("<eos>")
assert len(template["stop"]) == 2     # 内层列表是共享的，模板被污染

deep = copy.deepcopy(template)
deep["stop"].append("<x>")
assert len(template["stop"]) == 2     # 互不影响
```

<br>
<br>

## 2. 数据容器

这一章的工具都用来「给一组数据起名字、定结构」，代替到处传裸 `dict` 和裸 `tuple`。

| 工具 | 运行时是什么 | 适合 |
|---|---|---|
| `@dataclass` | 普通类 | 通用的可变数据对象 |
| `NamedTuple` | tuple | 不可变、轻量、可解包 |
| `TypedDict` | dict | 给现有 dict 标注键和类型 |
| `Enum` | 枚举对象 | 一组固定选项 |

<br>

### @dataclass

`@dataclass` 根据类型标注的字段自动生成 `__init__`、`__repr__`、`__eq__`，省去样板代码。它**不会在运行时校验类型**，标注只用于生成代码和静态检查。`frozen=True` 让实例不可修改（并且可哈希），`slots=True` 自动使用 `__slots__`。适合主要用来存数据的类，如配置、请求、结果；需要运行时校验时换 pydantic。

**实战**

```python
from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 16


a = SamplingParams(temperature=0.7)
assert a == SamplingParams(0.7, 16)            # 自动生成 __eq__
b = replace(a, max_tokens=32)                  # frozen 对象的「修改」：生成新对象
assert (a.max_tokens, b.max_tokens) == (16, 32)
assert asdict(b) == {"temperature": 0.7, "max_tokens": 32}
assert len({a, a}) == 1                        # frozen 后可哈希

try:
    a.max_tokens = 1
except AttributeError:                         # FrozenInstanceError 是其子类
    pass
```

<br>

### field(default_factory=...)

默认值只在定义类时求值一次。如果用 `[]` 作默认值，所有实例会共享同一个列表，所以 dataclass 直接禁止这样写。`field(default_factory=list)` 表示每次创建实例时调用工厂函数生成新值。`field` 还能控制字段是否进入 `__init__`、`__repr__`。字段默认值是 list、dict 等可变对象时必须用它。

**实战**

```python
from dataclasses import dataclass, field


@dataclass
class Sequence:
    request_id: str
    tokens: list[int] = field(default_factory=list)
    secret: str = field(default="", repr=False)    # 不打印到日志


a, b = Sequence("a"), Sequence("b")
a.tokens.append(1)
assert b.tokens == []
assert "secret" not in repr(a)
```

<br>

### __post_init__

自动生成的 `__init__` 赋值完后会调用 `__post_init__`。用于校验字段、计算派生字段。派生字段用 `field(init=False)` 从构造参数中排除。

**实战**

```python
from dataclasses import dataclass, field


@dataclass
class ParallelConfig:
    world_size: int
    tensor_parallel: int
    pipeline_parallel: int = field(init=False)

    def __post_init__(self):
        if self.world_size % self.tensor_parallel:
            raise ValueError("world_size must be divisible by tensor_parallel")
        self.pipeline_parallel = self.world_size // self.tensor_parallel


assert ParallelConfig(8, 2).pipeline_parallel == 4
```

**坑：** `frozen=True` 时不能直接赋值，要写 `object.__setattr__(self, "x", v)`。

<br>

### NamedTuple

`typing.NamedTuple` 生成一个 tuple 子类：既能用名字访问，也能下标访问和解包。不可变、内存开销小，适合函数返回多个值，或作为字典键。

**实战**

```python
from typing import NamedTuple


class TopK(NamedTuple):
    values: list[float]
    indices: list[int]


def top1(xs: list[float]) -> TopK:
    i = max(range(len(xs)), key=xs.__getitem__)
    return TopK([xs[i]], [i])


result = top1([0.1, 0.9])
assert result.indices == [1]       # 按名字
values, indices = result           # 解包
assert result[0] == values         # 下标
```

<br>

### TypedDict

`TypedDict` 只给类型检查器描述 dict 有哪些键、各是什么类型。运行时它就是普通 `dict`，没有校验。适合 JSON、kwargs 这类本来就是 dict 的数据。

**实战**

```python
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


m: ChatMessage = {"role": "user", "content": "hi"}
assert type(m) is dict                # 运行时只是 dict
assert ChatMessage.__required_keys__ == {"role", "content"}
```

<br>

### Enum + auto()

`Enum` 定义一组有名字的固定选项，防止魔法字符串和拼写错误，比较用 `is`。`auto()` 自动赋值。`StrEnum`（3.11+）的成员同时是字符串，`auto()` 的值是小写的成员名，便于 JSON 序列化。当一个值只能取几个固定选项（状态、模式、设备类型）时使用。

**实战**

```python
from enum import Enum, StrEnum, auto


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

    def is_finished(self) -> bool:
        return self is Status.FINISHED


class Task(StrEnum):
    GENERATE = auto()
    EMBED = auto()


assert Status.WAITING.value == 1
assert Status.FINISHED.is_finished()
assert Status["RUNNING"] is Status.RUNNING     # 按名字查
assert Task("embed") is Task.EMBED             # 按值查
assert Task.GENERATE == "generate"
```

<br>

### types.MappingProxyType

给 dict 包一层**只读视图**。不能通过它修改，但原字典变化时视图也会变。不用复制，适合暴露内部字典。

**实战**

```python
from types import MappingProxyType

_registry = {"llama": "LlamaForCausalLM"}
REGISTRY = MappingProxyType(_registry)

assert REGISTRY["llama"] == "LlamaForCausalLM"
try:
    REGISTRY["x"] = "y"
except TypeError:
    pass
_registry["qwen"] = "Qwen2ForCausalLM"
assert "qwen" in REGISTRY          # 视图能看到底层变化
```

<br>

### SimpleNamespace

一个可以随意添加属性的对象，用 `ns.x` 代替 `d["x"]`。不用定义类，适合临时对象、测试替身。

**实战：测试里伪造配置**

```python
from types import SimpleNamespace


def describe(config) -> str:
    return f"{config.model_type}:{config.hidden_size}"


fake = SimpleNamespace(model_type="llama", hidden_size=128)
assert describe(fake) == "llama:128"
fake.num_layers = 2
assert vars(fake)["num_layers"] == 2
```

<br>
<br>

## 3. pydantic

`dataclass` 不校验类型。pydantic 是第三方库，它**在运行时**按类型标注校验并转换数据，失败时给出结构化错误。所以它用在系统边界：解析 HTTP 请求、校验配置。本章基于 pydantic v2（`pip install pydantic`）。

<br>

### BaseModel

继承 `BaseModel` 并写类型标注，创建实例时就会校验。默认模式会做合理的转换，比如把 `"3"` 转成 `3`。`model_validate` 从 dict 构造，`model_dump` / `model_dump_json` 导出，校验失败抛 `ValidationError`。适合处理外部输入：API 请求体、配置文件、模型返回的 JSON。

**实战：解析 API 请求**

```python
from pydantic import BaseModel, ValidationError


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]          # 嵌套模型会递归校验
    max_tokens: int | None = None


req = ChatRequest.model_validate({
    "model": "qwen",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": "16",              # 字符串被转成 int
})
assert req.max_tokens == 16 and req.messages[0].role == "user"
assert req.model_dump()["messages"][0]["content"] == "hi"

try:
    ChatRequest(model="qwen", messages=[{"role": "user"}])
except ValidationError as e:
    err = e.errors()[0]
    assert err["loc"] == ("messages", 0, "content")
    assert err["type"] == "missing"
```

<br>

### Field 与 Annotated

`Field` 给字段加默认值、约束和描述。约束有两种写法：`x: int = Field(ge=0)`，或把约束放进类型 `Annotated[int, Field(ge=0)]`。后者可以把带约束的类型提取成别名复用。`default_factory` 的作用与 dataclass 相同。需要给字段加范围约束、默认值或描述时使用。

**实战**

```python
from typing import Annotated
from pydantic import BaseModel, Field, ValidationError

PositiveInt = Annotated[int, Field(gt=0)]    # 可复用


class Sampling(BaseModel):
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: PositiveInt = 16
    stop: list[str] = Field(default_factory=list)


assert Sampling(temperature=0.5).temperature == 0.5
for bad in ({"temperature": 3}, {"max_tokens": 0}):
    try:
        Sampling(**bad)
        raise AssertionError
    except ValidationError:
        pass
```

<br>

### @field_validator

校验单个字段的自定义逻辑。写成类方法，接收值，返回（可能修改过的）值，不合法时抛 `ValueError`。默认在类型校验之后运行，`mode="before"` 在之前运行，可用于预处理原始输入。当类型和 `Field` 约束表达不了规则（如格式转换、自定义检查）时使用。

**实战：把字符串统一成列表**

```python
from pydantic import BaseModel, field_validator


class Req(BaseModel):
    stop: list[str]

    @field_validator("stop", mode="before")
    @classmethod
    def to_list(cls, v):
        return [v] if isinstance(v, str) else v


assert Req(stop="\n").stop == ["\n"]
assert Req(stop=["a", "b"]).stop == ["a", "b"]
```

<br>

### @model_validator

校验多个字段之间的关系。`mode="after"` 拿到已构造好的实例，写成普通方法，必须返回 `self`；`mode="before"` 拿到原始输入。

**实战：互斥字段**

```python
from pydantic import BaseModel, ValidationError, model_validator


class Completion(BaseModel):
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.prompt is None) == (self.prompt_token_ids is None):
            raise ValueError("provide exactly one of prompt / prompt_token_ids")
        return self


Completion(prompt="hi")
try:
    Completion()
    raise AssertionError
except ValidationError:
    pass
```

<br>

### ConfigDict(extra="allow")

`model_config = ConfigDict(...)` 配置模型行为。`extra` 决定如何处理未声明的字段：`"ignore"`（默认，丢弃）、`"forbid"`（报错）、`"allow"`（保留）。OpenAI 兼容接口常用 `allow` 接受额外参数；要严格防止拼写错误时用 `forbid`。

**实战**

```python
from pydantic import BaseModel, ConfigDict, ValidationError


class Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


r = Lenient(model="m", top_k=5)
assert r.top_k == 5 and r.model_extra == {"top_k": 5}
try:
    Strict(model="m", tpo_k=5)      # 拼写错误被发现
    raise AssertionError
except ValidationError:
    pass
```

<br>

### pydantic.dataclasses.dataclass

用法和标准库 `@dataclass` 一样，但增加运行时校验，也支持 `Field` 和校验器。可以传 `config=ConfigDict(...)`。vLLM 的 `@config` 是另一个装饰器：按 `vllm/config/utils.py` 的说明，它用于确保配置类的字段都有默认值和文档字符串。下面只展示 pydantic 自身的校验机制。想保留 dataclass 写法、又需要运行时校验时使用，比如配置类。

**实战：配置类**

```python
from pydantic import ConfigDict, Field, ValidationError
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="forbid"))
class SchedulerConfig:
    max_num_seqs: int = Field(default=128, gt=0)
    policy: str = "fcfs"


assert SchedulerConfig(max_num_seqs="64").max_num_seqs == 64
try:
    SchedulerConfig(max_num_seqs=0)
    raise AssertionError
except ValidationError:
    pass
```

<br>
<br>

## 4. 类型系统

类型标注主要服务于 mypy、pyright 这类静态检查器和 IDE。**Python 运行时不会强制检查它们**，`x: int = "a"` 能正常运行。本章的实战里，有些断言只验证运行时行为，类型错误需要用检查器才能发现。

<br>

### from __future__ import annotations

让模块内的标注不在定义时求值，而是保存为字符串。这样可以直接引用尚未定义的类，也不用为只在标注中出现的名字付出导入成本。Python 3.14 起标注默认延迟求值，但这个导入在老版本中仍是常见写法。

**实战：引用自身**

```python
from __future__ import annotations


class Node:
    def child(self) -> Node:      # 没有上面的导入，3.14 之前这里会 NameError
        return Node()


assert Node.child.__annotations__["return"] == "Node"
```

<br>

### TYPE_CHECKING

`typing.TYPE_CHECKING` 运行时是 `False`，类型检查器把它视为 `True`。把只用于标注的导入放进去，可以打破循环导入，也能避免启动时导入 torch 这类重模块。

**实战**

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch               # 运行时不执行


def device_of(x: torch.Tensor) -> str:   # 标注不求值，所以没有 torch 也能运行
    return str(x.device)


assert TYPE_CHECKING is False
assert "torch" not in globals()
```

<br>

### collections.abc

标注参数时应该写「需要什么能力」，而不是具体类型。接收 `Sequence[int]` 的函数，list 和 tuple 都能传。参数类型宽、返回类型具体，是好的接口习惯。

| 类型 | 能力 |
|---|---|
| `Iterable` | 能 `for` 遍历 |
| `Iterator` | 能 `next()` |
| `Sequence` | 能下标访问，有长度 |
| `Mapping` | 只读字典接口 |
| `Callable[[A], R]` | 可调用 |
| `Generator[Y, S, R]` | 生成器 |

**实战**

```python
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence


def total(xs: Iterable[int]) -> int:
    return sum(xs)


def last(xs: Sequence[int]) -> int:
    return xs[-1]


def lookup(m: Mapping[str, int], k: str) -> int:
    return m.get(k, 0)


def apply(f: Callable[[int], int], x: int) -> int:
    return f(x)


def counter(n: int) -> Iterator[int]:
    yield from range(n)


assert total(x for x in [1, 2]) == 3
assert last((1, 2)) == 2
assert lookup({"a": 1}, "a") == 1
assert apply(lambda x: x + 1, 1) == 2
assert isinstance([], Sequence) and isinstance({}, Mapping)   # 也能运行时判断
```

<br>

### Any 与 cast

`Any` 关闭对这个值的类型检查，它与任何类型互相兼容，是逃生口。`cast(T, x)` 告诉检查器「相信我，这是 T」，**运行时什么也不做**，直接返回 x。如果说错了，检查器也发现不了。只在类型系统表达不了、或第三方库缺少类型信息时使用，能不用就不用。

**实战**

```python
from typing import Any, cast


def parse(raw: Any) -> str:
    return cast(str, raw["name"])      # 从无类型的 JSON 收窄


assert parse({"name": "x"}) == "x"
assert cast(int, "not int") == "not int"   # 运行时不校验
```

<br>

### Literal

限定值只能是几个字面量。比 `str` 精确，比 `Enum` 轻量，常用于字符串选项。

**实战**

```python
from typing import Literal, get_args

Task = Literal["generate", "embed"]


def run(task: Task) -> str:
    return task.upper()


assert run("embed") == "EMBED"
assert get_args(Task) == ("generate", "embed")   # 运行时也能取出选项
```

<br>

### ClassVar

标记属性属于类而不是实例。在 dataclass 中，`ClassVar` 不会变成字段，也不进入 `__init__`。类上共享的常量、计数器或注册表用它标注，在 dataclass 中尤其需要。

**实战**

```python
from dataclasses import dataclass, fields
from typing import ClassVar


@dataclass
class Model:
    supports_vision: ClassVar[bool] = False
    name: str = "m"


assert [f.name for f in fields(Model)] == ["name"]
```

<br>

### TypeAlias

给复杂类型起名字，提高可读性。3.12+ 也可以用 `type X = ...` 语法。

**实战**

```python
from typing import TypeAlias

TokenIds: TypeAlias = list[int]
PromptType: TypeAlias = str | TokenIds | dict[str, str]


def length(ids: TokenIds) -> int:
    return len(ids)


assert length([1, 2]) == 2
```

<br>

### Annotated

`Annotated[T, meta...]` 给类型附加元数据。类型仍然是 T，元数据由读取它的库解释，pydantic 用它存约束，FastAPI 用它存依赖。

**实战：读取元数据**

```python
from typing import Annotated, get_args, get_type_hints


class Doc:
    def __init__(self, text: str):
        self.text = text


class Config:
    size: Annotated[int, Doc("hidden size")] = 128


hint = get_type_hints(Config, include_extras=True)["size"]
base, meta = get_args(hint)
assert base is int and meta.text == "hidden size"
```

<br>

### @override、@overload、Final

- `@override`：声明方法重写了父类方法。父类改名后，检查器能报错。
- `@overload`：只用于声明同一函数的多种调用签名，让返回类型随参数变化；真正的实现只有最后一个。
- `Final` / `@final`：名字不可重新赋值，类或方法不可继承或重写。

**实战**

```python
from typing import Final, final, overload, override

MAX_RETRIES: Final = 3


class Base:
    def run(self) -> str:
        return "base"

    @final
    def id(self) -> int:
        return 1


class Child(Base):
    @override
    def run(self) -> str:
        return "child"


@overload
def first(x: str) -> str: ...
@overload
def first(x: list[int]) -> int: ...
def first(x):
    return x[0]


assert Child().run() == "child"
assert first("ab") == "a" and first([1]) == 1
```

<br>

### TypeVar、Generic、ParamSpec、Self

泛型表示「输入和输出类型之间的关系」。
- `TypeVar`：类型变量。`first(xs: list[T]) -> T`，传 `list[int]` 就返回 `int`。
- `Generic[T]`：让类支持类型参数。3.12+ 可写成 `class Stack[T]:`。
- `ParamSpec`：捕获函数的整组参数，写装饰器时保留原函数签名。
- `Self`：表示当前类的类型，子类调用时就是子类。

**实战**

```python
from collections.abc import Callable
from typing import Generic, ParamSpec, Self, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: list[T] = []

    def push(self, x: T) -> Self:
        self.items.append(x)
        return self                 # 支持链式调用，子类也能保留类型

    def pop(self) -> T:
        return self.items.pop()


def logged(f: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(*args, **kwargs)
    return wrapper


@logged
def add(a: int, b: int) -> int:
    return a + b


assert Stack[int]().push(1).push(2).pop() == 2
assert add(1, 2) == 3
```

<br>

### Protocol + @runtime_checkable

`Protocol` 是结构化类型（静态的鸭子类型）：只要对象有所需的方法，就算实现了接口，不用继承。`ABC` 是名义类型，必须显式继承。`@runtime_checkable` 让 `isinstance` 可用，但只检查方法名是否存在，不检查签名。只关心对象有没有某些方法、不想强制继承时用 `Protocol`；需要共享实现或强制子类实现时用 `ABC`。

**实战**

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsVision(Protocol):
    def embed_image(self, path: str) -> list[float]: ...


class VisionModel:                     # 没有继承 Protocol
    def embed_image(self, path: str) -> list[float]:
        return [0.0]


class TextModel:
    pass


assert isinstance(VisionModel(), SupportsVision)
assert not isinstance(TextModel(), SupportsVision)
```

<br>

### TypeIs、TypeGuard、assert_never

- 自定义判断函数返回 `TypeIs[T]` 时，检查器在 `if` 分支内外都能收窄类型；`TypeGuard[T]` 只收窄真分支，而且不要求 T 是输入的子类型。
- `assert_never(x)`：参数类型是 `Never`。如果 `match` 或 `if` 漏了分支，检查器会报错；运行时走到这里会抛异常。

**实战**

```python
from typing import Literal, TypeIs, assert_never

Prompt = str | list[int]


def is_text(p: Prompt) -> TypeIs[str]:
    return isinstance(p, str)


def size(p: Prompt) -> int:
    if is_text(p):
        return len(p.split())     # 这里是 str
    return len(p)                 # 这里是 list[int]


def label(mode: Literal["a", "b"]) -> int:
    if mode == "a":
        return 1
    if mode == "b":
        return 2
    assert_never(mode)            # 新增选项却忘了处理时，检查器报错


assert size("a b") == 2 and size([1]) == 1
assert label("b") == 2
try:
    label("c")                    # type: ignore
except AssertionError:
    pass
```

<br>

### NotRequired / Required

控制 `TypedDict` 中单个键是否必须存在。`NotRequired` 表示可省略；注意它和 `Optional` 不同，后者表示键存在但值可能是 `None`。

**实战**

```python
from typing import NotRequired, TypedDict


class Item(TypedDict):
    id: int
    score: NotRequired[float]


a: Item = {"id": 1}
assert Item.__required_keys__ == {"id"}
assert Item.__optional_keys__ == {"score"}
```

<br>

### get_args、get_origin、get_type_hints

运行时读取类型信息。`get_type_hints` 解析标注（包括字符串形式），`get_origin` 取外层类型，`get_args` 取类型参数。框架靠它们根据标注生成 CLI 参数或校验逻辑。

**实战：从 dataclass 标注提取选项**

```python
from dataclasses import dataclass
from typing import Literal, get_args, get_origin, get_type_hints


@dataclass
class Args:
    device: Literal["cpu", "cuda"] = "cpu"
    layers: list[int] | None = None


hints = get_type_hints(Args)
assert get_origin(hints["device"]) is Literal
assert get_args(hints["device"]) == ("cpu", "cuda")
assert get_origin(list[int]) is list
```

<br>
<br>

## 5. 抽象与扩展

大型框架要让新模型、新后端能「插进来」而不改核心代码。这一章的工具都是实现这件事的：定义接口、包装函数、注册实现、延迟加载。

<br>

### ABC + @abstractmethod

`ABC` 定义抽象基类，`@abstractmethod` 声明子类必须实现的方法。只要还有抽象方法没实现，实例化时就抛 `TypeError`。错误提前到创建对象时，而不是等到调用时。适合定义插件接口：多个后端或调度器实现同一组方法，漏实现时尽早报错。

**实战：调度器接口**

```python
from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def schedule(self) -> list[str]:
        ...

    def step(self) -> int:            # 基类可以提供公共逻辑（模板方法）
        return len(self.schedule())


class Fcfs(Scheduler):
    def schedule(self) -> list[str]:
        return ["req-1"]


assert Fcfs().step() == 1
try:
    Scheduler()
except TypeError:
    pass
```

<br>

### 装饰器 + functools.wraps

函数在 Python 中是对象，装饰器就是「接收函数、返回函数」的函数。`@deco` 放在 `def f` 上等价于 `f = deco(f)`。带参数的装饰器多一层：先调用得到装饰器。`functools.wraps` 把原函数的名字和文档复制到 wrapper 上，否则调试时看到的都是 `wrapper`。

**实战：带参数的重试装饰器**

```python
from functools import wraps


def retry(times: int):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return f(*args, **kwargs)
                except ConnectionError:
                    if attempt == times - 1:
                        raise
        return wrapper
    return decorator


calls = []


@retry(times=3)
def fetch() -> str:
    """Fetch weights."""
    calls.append(1)
    if len(calls) < 3:
        raise ConnectionError
    return "ok"


assert fetch() == "ok" and len(calls) == 3
assert fetch.__name__ == "fetch" and fetch.__doc__ == "Fetch weights."
```

<br>

### functools.cache / lru_cache

以参数为键，缓存函数返回值。`cache` 无容量上限，`lru_cache(maxsize=n)` 超过容量后淘汰最久未使用的条目。参数必须可哈希。适合纯函数、参数可哈希且会被重复调用的场景，比如解析配置、递归 DP。

**实战**

```python
from functools import cache, lru_cache


@cache
def fib(n: int) -> int:
    return n if n < 2 else fib(n - 1) + fib(n - 2)


@lru_cache(maxsize=2)
def tokenizer(name: str) -> str:
    return f"tokenizer:{name}"


assert fib(80) == 23416728348467685
tokenizer("a"); tokenizer("a")
assert tokenizer.cache_info().hits == 1
```

**坑：** 不要直接装饰实例方法：`self` 会成为缓存键，缓存持有实例，导致对象无法被回收。

<br>

### functools.partial

固定函数的部分参数，得到一个新的可调用对象。比 `lambda` 更明确，也能被 pickle（前提是原函数可 pickle），所以适合传给进程池或回调。

**实战**

```python
from functools import partial


def tokenize(text: str, lower: bool, max_len: int) -> list[str]:
    words = text.lower().split() if lower else text.split()
    return words[:max_len]


short = partial(tokenize, lower=True, max_len=2)
assert short("A B C") == ["a", "b"]
assert short.func is tokenize and short.keywords["max_len"] == 2
```

<br>

### 装饰器 + 注册表模式

用字典保存「名字 → 实现」，装饰器在定义时把类写进字典，并原样返回。核心代码只查表，新增实现不用改 `if/else`。vLLM 的 `@CustomOp.register`、`@MULTIMODAL_REGISTRY.register_processor` 都是这个模式。

**实战：可扩展的算子**

```python
class Op:
    _registry: dict[str, type["Op"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(sub):
            cls._registry[name] = sub
            return sub                 # 一定要返回，否则类名变成 None
        return decorator

    @classmethod
    def create(cls, name: str) -> "Op":
        return cls._registry[name]()


@Op.register("relu")
class Relu(Op):
    def __call__(self, x: float) -> float:
        return max(x, 0.0)


assert Op.create("relu")(-1.0) == 0.0
```

<br>

### importlib

`importlib.import_module("a.b")` 用字符串在运行时导入模块。配合 `getattr` 就能把配置里的 `"模块:类"` 变成真正的类，常用于插件加载，以及只在需要时才导入重依赖。

**实战：按字符串加载类**

```python
import importlib


def resolve(path: str):
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


assert resolve("collections:OrderedDict").__name__ == "OrderedDict"
```

<br>

### 模块级 __getattr__

模块里定义函数 `__getattr__(name)` 后，访问不存在的模块属性时会调用它。可用于延迟导入、给旧名字加兼容提示，而不影响 `import` 的速度。

**实战：延迟导入**

```python
import importlib
import sys

_LAZY = {"json_mod": "json"}


def __getattr__(name: str):
    if name in _LAZY:
        return importlib.import_module(_LAZY[name])
    raise AttributeError(f"module has no attribute {name}")


this = sys.modules[__name__]
assert this.json_mod.dumps(1) == "1"
assert not hasattr(this, "missing")
```

<br>

### lambda 字典延迟求值

字典的值如果直接写表达式，定义时就会求值。写成 `lambda` 后，读取时才执行。环境变量表用这种写法，就能在每次访问时读到最新值，不受模块导入时间影响。

**实战：结合模块 `__getattr__` 的环境变量表**

```python
import os

variables = {
    "APP_LOG_LEVEL": lambda: os.getenv("APP_LOG_LEVEL", "INFO"),
    "APP_WORKERS": lambda: int(os.getenv("APP_WORKERS", "1")),
}

assert variables["APP_WORKERS"]() == 1
os.environ["APP_WORKERS"] = "4"        # 导入之后才设置
assert variables["APP_WORKERS"]() == 4
```

<br>

### inspect

运行时查看对象：函数签名、源码、类成员、调用栈。框架用它做依赖注入或检查函数是否支持某个参数。

**实战：只传函数接受的参数**

```python
import inspect


def filter_kwargs(f, kwargs: dict) -> dict:
    params = inspect.signature(f).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def forward(input_ids, attention_mask=None):
    return attention_mask


kw = filter_kwargs(forward, {"attention_mask": 1, "unknown": 2})
assert kw == {"attention_mask": 1}
assert inspect.iscoroutinefunction(forward) is False
```

<br>

### ast

把源码解析成抽象语法树，**不执行代码**。可用于静态分析，例如不导入模块就找出里面定义的类。`ast.literal_eval` 只解析字面量，是 `eval` 的安全替代。

**实战**

```python
import ast

source = """
class LlamaForCausalLM:
    pass
def helper(): pass
"""
tree = ast.parse(source)
assert [n.name for n in tree.body if isinstance(n, ast.ClassDef)] == ["LlamaForCausalLM"]

assert ast.literal_eval("[1, {'a': 2}]") == [1, {"a": 2}]
try:
    ast.literal_eval("__import__('os')")      # 拒绝执行
except ValueError:
    pass
```

<br>

### deprecated 标记

`warnings.deprecated`（3.13+，旧版本用 `typing_extensions`）标记 API 已废弃：调用时发出 `DeprecationWarning`，检查器和 IDE 也能提示。重命名或准备移除 API 时使用，给调用方一个过渡期。

**实战**

```python
import warnings
from warnings import deprecated


@deprecated("Use generate() instead")
def complete() -> str:
    return "ok"


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    complete()
assert issubclass(caught[0].category, DeprecationWarning)
assert "generate" in str(caught[0].message)
```

<br>
<br>

## 6. 上下文与流程控制

<br>

### with 语句与 @contextmanager

`with` 保证资源一定被清理：进入时调 `__enter__`，退出时调 `__exit__`，出异常也一样。`contextlib.@contextmanager` 用生成器写上下文管理器，`yield` 之前是进入逻辑，之后是退出逻辑。**清理代码要放进 `finally`**，否则异常时不会执行。打开文件、加锁、临时修改状态这类成对的「获取 / 释放」操作都应该用 `with`。

**实战：临时修改环境变量**

```python
import os
from contextlib import contextmanager


@contextmanager
def set_env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield value                       # 这个值赋给 as 后面的变量
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old


try:
    with set_env("MODE", "test") as v:
        assert v == "test" and os.environ["MODE"] == "test"
        raise RuntimeError
except RuntimeError:
    pass
assert "MODE" not in os.environ           # 异常时也恢复了
```

<br>

### ExitStack

`contextlib.ExitStack` 动态管理多个清理动作，退出时按**后进先出**执行。当需要进入的上下文数量运行时才知道时，无法写嵌套 `with`，就用它。`callback` 注册普通函数，`enter_context` 进入上下文管理器。

**实战：部分失败也能全部清理**

```python
from contextlib import ExitStack

log = []
try:
    with ExitStack() as stack:
        for i in range(3):
            stack.callback(log.append, f"release-{i}")
            if i == 1:
                raise RuntimeError("worker 2 failed")
except RuntimeError:
    pass

assert log == ["release-1", "release-0"]   # 只清理已获取的资源，顺序相反
```

<br>

### @asynccontextmanager

`@contextmanager` 的异步版本，用 `async with`，进入和退出时都能 `await`。Web 框架常用它管理应用生命周期：启动时加载，关闭时释放。

**实战：FastAPI lifespan 风格**

```python
import asyncio
from contextlib import asynccontextmanager

events = []


@asynccontextmanager
async def lifespan():
    await asyncio.sleep(0)
    events.append("startup")
    try:
        yield {"engine": "ready"}
    finally:
        await asyncio.sleep(0)
        events.append("shutdown")


async def main():
    async with lifespan() as state:
        assert state["engine"] == "ready"


asyncio.run(main())
assert events == ["startup", "shutdown"]
```

<br>

### ContextVar

全局变量在并发中会串数据。`threading.local` 按线程隔离，但一个线程上的所有协程共用它。`ContextVar` 按异步任务隔离，每个 `asyncio.Task` 都有自己的上下文副本。适合 request id、trace id 这种不想层层传参的值。`set` 返回 token，用 `reset(token)` 恢复。

**实战：每个请求独立的 request id**

```python
import asyncio
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="-")


async def log_line() -> str:
    await asyncio.sleep(0)
    return request_id.get()


async def handle(rid: str) -> str:
    request_id.set(rid)
    await asyncio.sleep(0)            # 中间切到其他任务也不会串
    return await log_line()


async def main():
    results = await asyncio.gather(handle("a"), handle("b"))
    assert results == ["a", "b"]
    assert request_id.get() == "-"    # 主任务不受影响


asyncio.run(main())
```

<br>

### 生成器 yield / yield from

在 Python 中，`yield` 将函数变成生成器，每次产生一个值，并在迭代之间保留执行状态，下次从暂停的位置继续执行。数据量大或需要流式处理时使用，比如逐行读文件、流式输出 token、分批处理。

含 `yield` 的函数调用时不执行，而是返回生成器。每次 `next` 运行到下一个 `yield` 并暂停，保留局部变量。它**惰性**地产生数据，不用一次把所有结果放进内存。`yield from it` 把子迭代器的值逐个转发。

**实战：流式输出与分批**

```python
from collections.abc import Iterable, Iterator


def batched(items: Iterable[int], n: int) -> Iterator[list[int]]:
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def flatten(nested) -> Iterator[int]:
    for x in nested:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x


def infinite():
    i = 0
    while True:
        yield i
        i += 1


assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
assert list(flatten([1, [2, [3]]])) == [1, 2, 3]
g = infinite()
assert next(g) == 0 and next(g) == 1        # 无限序列也没问题
```

**坑：** 生成器只能遍历一次，耗尽后再遍历什么也得不到。

<br>

### 海象运算符 :=

在表达式内部赋值并返回该值。避免重复计算，或把「先赋值再判断」合成一行。

**实战**

```python
import re

if (m := re.match(r"layers\.(\d+)", "layers.12.mlp")) is not None:
    assert m.group(1) == "12"

data = iter([b"ab", b"c", b""])
chunks = []
while chunk := next(data):                     # 读到空为止
    chunks.append(chunk)
assert chunks == [b"ab", b"c"]

values = [1, 2, 3, 4]
assert [y for x in values if (y := x * x) > 4] == [9, 16]
```

<br>

### match / case

结构化模式匹配（3.10+），不只是 switch：它能按类型、结构解构数据并绑定变量。`case _` 是默认分支，`if` 可加守卫条件。适合按数据结构分发，例如处理不同形状的消息；只比较几个值时用 `if` 或字典就够了。

**实战：按消息结构分发**

```python
from dataclasses import dataclass


@dataclass
class Text:
    content: str


@dataclass
class Image:
    url: str


def handle(msg) -> str:
    match msg:
        case Text(content=""):
            return "empty"
        case Text(content=c):                    # 匹配类并绑定字段
            return f"text:{c}"
        case Image(url=u) if u.startswith("http"):
            return "remote"
        case {"type": "tool", "name": name}:     # 匹配字典结构
            return f"tool:{name}"
        case [first, *_]:                        # 匹配序列
            return f"first:{first}"
        case _:
            return "unknown"


assert handle(Text("hi")) == "text:hi"
assert handle(Image("http://x")) == "remote"
assert handle({"type": "tool", "name": "search"}) == "tool:search"
assert handle([1, 2]) == "first:1"
assert handle(3) == "unknown"
```

**坑：** `case FOO:` 里的裸名字是**捕获变量**，不是比较常量。要比较常量，写成带点的名字，比如 `case Status.RUNNING:`。

<br>
<br>

## 7. 并发

先分清三种模型：

| 模型 | 适合 | 特点 |
|---|---|---|
| `asyncio` 协程 | 大量 IO 等待（HTTP、网络） | 单线程，在 `await` 处主动切换 |
| `threading` 线程 | IO、调用释放 GIL 的 C 库 | 抢占式切换，要加锁；普通构建下 GIL 使同一时刻只有一个线程跑 Python 字节码 |
| `multiprocessing` 进程 | CPU 密集、GPU 隔离 | 独立内存，通信要序列化 |

vLLM 这类服务通常都用到：API 层用 asyncio，引擎和 GPU worker 用进程。

<br>

### async def 与 await

`async def` 定义协程函数，调用时不执行，只返回协程对象。`await` 暂停当前协程，把控制权还给事件循环，等结果就绪再恢复。事件循环在单线程里调度协程，所以并发来自「等待时让出」。适合大量 IO 等待的场景，如 HTTP 服务、调用外部 API；CPU 密集任务用它没有收益。

**实战**

```python
import asyncio


async def fetch(x: int) -> int:
    await asyncio.sleep(0)          # 让出控制权
    return x * 2


coro = fetch(1)                     # 这里函数体还没运行
assert asyncio.run(coro) == 2       # asyncio.run 启动事件循环
```

**坑：** 在协程里调用阻塞函数，比如 `time.sleep`，会卡住**整个**事件循环。

<br>

### AsyncGenerator

`async def` 中用 `yield` 得到异步生成器，用 `async for` 遍历，每次产出之间都可以 `await`。最典型的用途是流式返回 token。

**实战：流式响应**

```python
import asyncio
from collections.abc import AsyncGenerator


async def stream(tokens: list[str]) -> AsyncGenerator[str, None]:
    for t in tokens:
        await asyncio.sleep(0)      # 等模型输出
        yield t


async def main() -> str:
    return "".join([t async for t in stream(["he", "llo"])])


assert asyncio.run(main()) == "hello"
```

<br>

### create_task 与 gather

直接 `await f()` 是顺序执行。`create_task` 把协程放进事件循环后台运行，并返回 Task；`gather` 并发运行多个协程，按传入顺序返回结果。`return_exceptions=True` 时，异常作为结果返回，不会立刻向外抛。多个互不依赖的 IO 操作需要同时进行时使用。

**实战：并发请求**

```python
import asyncio
import time


async def call(i: int) -> int:
    await asyncio.sleep(0.05)
    if i == 2:
        raise ValueError("bad")
    return i


async def main():
    start = time.perf_counter()
    results = await asyncio.gather(call(0), call(1))
    assert results == [0, 1]
    assert time.perf_counter() - start < 0.09        # 并发，不是 0.1 秒

    mixed = await asyncio.gather(call(1), call(2), return_exceptions=True)
    assert mixed[0] == 1 and isinstance(mixed[1], ValueError)

    task = asyncio.create_task(call(1))              # 已经在后台运行
    assert await task == 1


asyncio.run(main())
```

**坑：** 事件循环只保留 Task 的弱引用，后台任务要自己保存引用，否则可能被回收。

<br>

### asyncio.Queue 与 Event

`asyncio.Queue` 让协程之间传数据，`await get()` 在空时等待，`maxsize` 提供背压。`asyncio.Event` 是一个信号，`wait()` 等到 `set()`。它们不是线程安全的，只能用在同一事件循环。

**实战：生产者 / 消费者**

```python
import asyncio


async def producer(q: asyncio.Queue) -> None:
    for i in range(3):
        await q.put(i)
    await q.put(None)              # 结束标记


async def consumer(q: asyncio.Queue, done: asyncio.Event, out: list) -> None:
    while (item := await q.get()) is not None:
        out.append(item)
    done.set()


async def main():
    q, done, out = asyncio.Queue(maxsize=1), asyncio.Event(), []
    await asyncio.gather(producer(q), consumer(q, done, out))
    await done.wait()
    assert out == [0, 1, 2]


asyncio.run(main())
```

<br>

### run_in_executor / to_thread

不能修改的阻塞函数不能直接放进协程。`asyncio.to_thread(f, ...)` 把它放到线程池运行，协程在等待时不阻塞事件循环。`loop.run_in_executor` 是更底层的版本，可以指定 executor。

**实战**

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


def blocking(x: int) -> int:
    time.sleep(0.05)
    return x


async def main():
    start = time.perf_counter()
    a, b = await asyncio.gather(asyncio.to_thread(blocking, 1), asyncio.to_thread(blocking, 2))
    assert (a, b) == (1, 2)
    assert time.perf_counter() - start < 0.09

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(1) as pool:
        assert await loop.run_in_executor(pool, blocking, 3) == 3


asyncio.run(main())
```

<br>

### threading.Thread

`Thread(target=f)` 创建线程，`start()` 启动，`join()` 等待结束。`daemon=True` 的线程不会阻止进程退出。线程的返回值拿不到，要用共享对象或队列。适合 IO 型后台任务，如心跳、日志上传；CPU 密集任务受 GIL 限制，应换进程。

**实战**

```python
import threading

results = []


def worker(i: int) -> None:
    results.append(i * i)


threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert sorted(results) == [0, 1, 4]
```

<br>

### Lock、Event、Condition

线程可能在任意字节码之间切换，`count += 1` 是「读、加、写」，并非原子操作，要用 `Lock` 保护。`Event` 是一次性信号。`Condition` 让线程等待某个条件成立，要在持锁时 `wait`，并用循环或 `wait_for` 重新检查条件。

**实战：Condition 实现有界缓冲区**

```python
import threading


class BoundedBuffer:
    def __init__(self, capacity: int):
        self.items: list[int] = []
        self.capacity = capacity
        self.cond = threading.Condition()

    def put(self, x: int) -> None:
        with self.cond:
            self.cond.wait_for(lambda: len(self.items) < self.capacity)
            self.items.append(x)
            self.cond.notify_all()

    def get(self) -> int:
        with self.cond:
            self.cond.wait_for(lambda: self.items)
            x = self.items.pop(0)
            self.cond.notify_all()
            return x


buf = BoundedBuffer(1)
out = []
consumer = threading.Thread(target=lambda: out.extend(buf.get() for _ in range(3)))
consumer.start()
for i in range(3):
    buf.put(i)
consumer.join()
assert out == [0, 1, 2]

count = 0
lock = threading.Lock()


def inc():
    global count
    for _ in range(10000):
        with lock:
            count += 1


ts = [threading.Thread(target=inc) for _ in range(4)]
[t.start() for t in ts]
[t.join() for t in ts]
assert count == 40000
```

<br>

### queue.Queue

线程安全的 FIFO 队列，内部已加锁，是线程通信的首选。`get()` 在空时阻塞，`task_done()` 配合 `join()` 等待全部处理完成。它不能跨进程，也不能在协程里直接 `await`。

**实战**

```python
import queue
import threading

q: queue.Queue[int] = queue.Queue()
out = []


def worker() -> None:
    while True:
        item = q.get()
        out.append(item)
        q.task_done()


threading.Thread(target=worker, daemon=True).start()
for i in range(3):
    q.put(i)
q.join()                  # 等所有任务 task_done
assert out == [0, 1, 2]
```

<br>

### ThreadPoolExecutor

线程池复用固定数量的线程。`submit` 返回 `Future`，`map` 保持输入顺序，`as_completed` 按完成顺序产出。比手动管理 Thread 简单，还能拿到返回值和异常。需要并发执行一批阻塞任务并收集结果时使用。

**实战**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


def load(i: int) -> int:
    if i == 3:
        raise ValueError("corrupt")
    return i * 10


with ThreadPoolExecutor(max_workers=2) as pool:
    assert list(pool.map(load, [0, 1, 2])) == [0, 10, 20]
    futures = {pool.submit(load, i): i for i in (1, 3)}
    outcomes = {}
    for f in as_completed(futures):
        outcomes[futures[f]] = f.exception() or f.result()

assert outcomes[1] == 10 and isinstance(outcomes[3], ValueError)
```

<br>

### multiprocessing.get_context("spawn")

启动子进程有三种主要方式：`fork` 复制父进程（快，但在多线程或已初始化 CUDA 后不安全）；`spawn` 启动全新解释器（慢，但干净）；`forkserver` 介于两者之间。GPU 程序应使用 `spawn`。因为子进程会重新导入主模块，所以入口必须放在 `if __name__ == "__main__":` 下，而且目标函数和参数必须能 pickle。

**实战**

```python
import multiprocessing as mp


def worker(rank: int, q) -> None:
    q.put(rank * 2)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, q)) for i in range(2)]
    for p in procs:
        p.start()
    results = sorted(q.get() for _ in procs)
    for p in procs:
        p.join()
    assert results == [0, 2]
```

**坑：** 先取完队列数据再 `join`。子进程在数据写完前不会退出，顺序反过来可能死锁。

<br>

### signal

信号是操作系统发给进程的异步通知，如 Ctrl+C 的 `SIGINT`、`kill` 的 `SIGTERM`。`signal.signal` 注册处理函数，让服务能优雅退出。在 Python 中，处理函数只在**主线程**运行，也只能在主线程注册。

**实战：优雅退出**

```python
import os
import signal

stopping = []


def handler(signum, frame):
    stopping.append(signal.Signals(signum).name)


previous = signal.signal(signal.SIGTERM, handler)
os.kill(os.getpid(), signal.SIGTERM)       # 给自己发信号
signal.signal(signal.SIGTERM, previous)
assert stopping == ["SIGTERM"]
```

<br>

### atexit

`atexit.register` 注册解释器正常退出时运行的函数，按注册的相反顺序执行。被 `SIGKILL` 杀死、调用 `os._exit` 或未处理的致命信号时不会运行，所以不能依赖它做关键持久化。适合退出时做收尾：删除临时文件、关闭子进程、打印统计。

**实战**

```python
import atexit
import subprocess
import sys

code = """
import atexit
atexit.register(print, "first registered")
atexit.register(print, "second registered")
"""
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
assert out.split("\n")[:2] == ["second registered", "first registered"]
```

<br>

### weakref

普通引用让对象存活，弱引用不增加引用计数：对象被回收后，弱引用变成 `None`。用于缓存、回调、父子互引，避免内存泄漏。`weakref.finalize` 在对象回收时运行清理，比 `__del__` 可靠。

**实战**

```python
import gc
import weakref


class Model:
    pass


released = []
m = Model()
ref = weakref.ref(m)
weakref.finalize(m, released.append, "freed")
assert ref() is m

del m
gc.collect()
assert ref() is None and released == ["freed"]

cache = weakref.WeakValueDictionary()
held = Model()
cache["a"] = held
assert "a" in cache
del held
gc.collect()
assert "a" not in cache                  # 没有强引用后自动移除
```

<br>
<br>

## 8. 标准库数据结构

`defaultdict`、`Counter`、`deque`、`heapq`、`bisect` 的基础接口和复杂度见[内置容器与函数](#12-python-内置容器与函数leetcode-常考)。这一章解释它们的原理，实战用工程场景。

<br>

### itertools

`itertools` 提供惰性迭代器工具，按需生成元素而不是构建列表，大数据量时省内存。输入输出都是迭代器，可以组合。

| 函数 | 作用 |
|---|---|
| `chain` / `chain.from_iterable` | 串联多个可迭代对象 |
| `islice` | 切片迭代器 |
| `product` | 笛卡尔积，代替嵌套循环 |
| `accumulate` | 累积（前缀和） |
| `groupby` | 把**连续**相同键分组 |
| `batched`（3.12+） | 分批 |
| `pairwise` | 相邻两两配对 |
| `count` / `repeat` / `cycle` | 无限序列 |

**实战**

```python
from itertools import accumulate, batched, chain, count, groupby, islice, pairwise, product

assert list(chain([1], [2, 3])) == [1, 2, 3]
assert list(islice(count(10), 2)) == [10, 11]
assert list(accumulate([1, 2, 3])) == [1, 3, 6]
assert list(pairwise([1, 2, 3])) == [(1, 2), (2, 3)]
assert [list(b) for b in batched(range(5), 2)] == [[0, 1], [2, 3], [4]]

# 枚举超参组合
grid = list(product([1, 2], ["cpu", "cuda"]))
assert len(grid) == 4

# groupby 只合并连续元素，所以要先排序
reqs = [("a", 1), ("b", 2), ("a", 3)]
key = lambda r: r[0]
grouped = {k: [v for _, v in g] for k, g in groupby(sorted(reqs, key=key), key)}
assert grouped == {"a": [1, 3], "b": [2]}
```

<br>

### defaultdict

`defaultdict(factory)` 访问不存在的键时，调用 factory 创建默认值并**写入字典**。省掉「先判断键是否存在」的代码。分组、建邻接表、累加计数时使用。

**实战：按键分组**

```python
from collections import defaultdict

by_model = defaultdict(list)
for model, req in [("a", 1), ("b", 2), ("a", 3)]:
    by_model[model].append(req)
assert dict(by_model) == {"a": [1, 3], "b": [2]}

counts = defaultdict(int)
_ = counts["unseen"]                # 只是读取，也会写入键
assert "unseen" in counts
```

<br>

### OrderedDict

普通 dict 从 3.7 起已保持插入顺序。`OrderedDict` 仍有用，是因为提供 `move_to_end` 和 `popitem(last=False)`，可以 O(1) 调整顺序，很适合实现 LRU。

**实战：LRU 缓存**

```python
from collections import OrderedDict


class LRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)     # 弹出最旧的


c = LRU(2)
c.put("a", 1)
c.put("b", 2)
c.get("a")
c.put("c", 3)
assert list(c.data) == ["a", "c"]
```

<br>

### deque

双端队列，两端增删都是 O(1)，中间访问是 O(n)。`maxlen` 让它成为固定长度窗口，满了自动丢弃另一端的旧元素。调度器的等待队列就很适合用它。

**实战：FCFS 队列与滑动平均**

```python
from collections import deque

waiting = deque(["r1", "r2"])
running = waiting.popleft()
waiting.appendleft(running)             # 被抢占后放回队首
assert waiting[0] == "r1"

latencies = deque(maxlen=3)
for x in [10, 20, 30, 40]:
    latencies.append(x)
assert sum(latencies) / len(latencies) == 30
```

<br>

### Counter

dict 的子类，用于计数。缺失键返回 0 且不写入，支持 `most_common` 和计数加减。统计频率、求 Top K、比较两个集合的元素组成时使用。

**实战**

```python
from collections import Counter

c = Counter(["cuda:0", "cuda:1", "cuda:0"])
assert c.most_common(1) == [("cuda:0", 2)]
assert c["cuda:9"] == 0 and "cuda:9" not in c
assert Counter("aab") - Counter("a") == Counter("ab")
```

<br>

### bisect

在**有序**列表上二分查找。`bisect_left` 返回第一个 `>= x` 的位置，`bisect_right` 返回第一个 `> x` 的位置。`insort` 插入并保持有序；查找是 O(log n)，但 list 插入要移动元素，所以是 O(n)。在有序数据里按阈值找位置时使用，比如分桶、选最接近的配置。

**实战：根据 token 数选择预热的 batch 大小**

```python
from bisect import bisect_left, insort

buckets = [1, 2, 4, 8, 16]


def round_up(n: int) -> int:
    return buckets[bisect_left(buckets, n)]


assert round_up(3) == 4 and round_up(4) == 4
xs = [1, 3]
insort(xs, 2)
assert xs == [1, 2, 3]
```

<br>

### heapq

在普通 list 上维护小顶堆：`h[0]` 永远最小，`push` / `pop` 为 O(log n)。没有大顶堆，通常对值取负。比较相同时会继续比较元组下一项，所以要加递增序号，既避免比较对象报错，也保证同优先级先进先出。需要反复取最小（或最大）值时使用：优先级队列、Top K、合并多个有序序列。

**实战：优先级调度**

```python
import heapq
from itertools import count

heap = []
seq = count()
for priority, req in [(1, "b"), (0, "a"), (1, "c")]:
    heapq.heappush(heap, (priority, next(seq), req))

assert [heapq.heappop(heap)[2] for _ in range(3)] == ["a", "b", "c"]
assert heapq.nlargest(2, [1, 5, 3]) == [5, 3]
```

<br>
<br>

## 9. 常用标准库

<br>

### math

浮点和整数数学函数。工程中最常见的是向上取整、对数、最大公约数，以及判断 `nan` / `inf`。

**实战：计算页数与对齐**

```python
import math

assert math.ceil(10 / 4) == 3                  # 需要多少页
assert -(-10 // 4) == 3                        # 大整数用这个，避免浮点误差
assert math.prod([2, 3, 4]) == 24
assert math.gcd(12, 18) == 6
assert math.isclose(0.1 + 0.2, 0.3)            # 浮点不要用 ==
assert math.isnan(float("nan"))
assert math.log2(1024) == 10
```

<br>

### os

与操作系统交互：环境变量、进程信息、CPU 数量。文件路径优先用 `pathlib`。

**实战**

```python
import os

os.environ["APP_DEBUG"] = "1"
assert os.getenv("APP_DEBUG") == "1"
assert os.getenv("NOT_SET", "fallback") == "fallback"
assert os.getpid() > 0
assert os.cpu_count() >= 1
```

**坑：** 环境变量都是字符串，`bool(os.getenv("X"))` 对 `"0"` 也是 `True`。

<br>

### pathlib

用对象表示路径，`/` 拼接路径，比字符串拼接和 `os.path` 更清晰，还能跨平台。

**实战：查找模型文件**

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "model"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    (root / "a.safetensors").touch()

    assert [p.name for p in root.glob("*.safetensors")] == ["a.safetensors"]
    p = root / "config.json"
    assert (p.stem, p.suffix, p.parent.name) == ("config", ".json", "model")
    assert p.exists() and p.read_text() == "{}"
```

<br>

### json

`dumps` / `loads` 处理字符串，`dump` / `load` 处理文件。JSON 没有 tuple、set、datetime，自定义类型通过 `default` 转换。读写配置文件、处理 API 数据时使用。

**实战**

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Result:
    id: int
    tags: set


def encode(o):
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


text = json.dumps(asdict(Result(1, {"b", "a"})), default=encode, ensure_ascii=False)
assert json.loads(text) == {"id": 1, "tags": ["a", "b"]}
assert json.loads(json.dumps((1, 2))) == [1, 2]       # tuple 变成 list
assert json.loads(json.dumps({1: "a"})) == {"1": "a"}  # 键变成字符串
```

<br>

### regex / re

标准库 `re` 用模式匹配字符串。第三方 `regex` 与它接口兼容，功能更多，vLLM 里常见 `import regex as re`。`match` 从开头匹配，`search` 找第一处，`findall` 找全部，`sub` 替换。写模式用原始字符串 `r"..."`，高频使用时先 `compile`。

**实战：重写权重名**

```python
import re

LAYER = re.compile(r"^model\.layers\.(?P<idx>\d+)\.(?P<rest>.+)$")

m = LAYER.match("model.layers.12.self_attn.q_proj")
assert m["idx"] == "12" and m["rest"] == "self_attn.q_proj"

assert re.sub(r"\.layers\.(\d+)", r".h.\1", "m.layers.3.mlp") == "m.h.3.mlp"
assert re.findall(r"\d+", "a1b22") == ["1", "22"]
assert re.findall(r"<.+>", "<a><b>") == ["<a><b>"]    # 贪婪
assert re.findall(r"<.+?>", "<a><b>") == ["<a>", "<b>"]  # 非贪婪
```

<br>

### time

`time.time()` 是墙上时钟，可能被系统调整，用于时间戳。`time.perf_counter()` 单调且精度高，用于测耗时。`monotonic` 也是单调时钟，常用于超时判断。

**实战：耗时统计**

```python
import time
from contextlib import contextmanager


@contextmanager
def timer(out: dict, name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        out[name] = time.perf_counter() - start


stats = {}
with timer(stats, "sleep"):
    time.sleep(0.01)
assert stats["sleep"] >= 0.009
```

<br>

### logging

logger 按模块命名成树，日志向上传播给父 logger 的 handler。库代码只用 `logging.getLogger(__name__)`，配置留给应用。参数用 `%s` 惰性格式化：级别被过滤时不会拼接字符串。

**实战**

```python
import logging
import sys

logger = logging.getLogger("app.engine")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

logger.debug("hidden %s", "value")            # 低于 INFO，被过滤
logger.info("loaded %d layers", 12)
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("step failed")           # 自动附带堆栈
```

<br>

### argparse

解析命令行参数，自动生成 `--help`。`type` 转换类型，`choices` 限定值，`action="store_true"` 定义开关，子命令用 `add_subparsers`。写命令行脚本或服务启动入口时使用。

**实战**

```python
import argparse

parser = argparse.ArgumentParser(prog="serve")
parser.add_argument("model")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--dtype", choices=["auto", "float16"], default="auto")
parser.add_argument("--trust-remote-code", action="store_true")

args = parser.parse_args(["qwen", "--port", "9000", "--trust-remote-code"])
assert args.model == "qwen" and args.port == 9000
assert args.trust_remote_code is True             # 横线变下划线
```

<br>

### sys

访问解释器状态：命令行参数、模块表、标准流、版本、退出。

**实战**

```python
import sys

assert sys.version_info >= (3, 8)
assert "sys" in sys.modules                 # 已导入模块的缓存
assert isinstance(sys.argv, list)
print("warning", file=sys.stderr)

try:
    sys.exit(2)                             # 本质是抛 SystemExit
except SystemExit as e:
    assert e.code == 2
```

<br>

### hashlib

计算 sha256 等密码学哈希。输入是 bytes，相同输入得到相同结果。内置 `hash()` 对字符串按进程随机化，**跨进程不稳定**，所以缓存键要用 hashlib。

**实战：prefix cache 的块哈希**

```python
import hashlib


def block_hash(parent: str, tokens: list[int]) -> str:
    h = hashlib.sha256()
    h.update(parent.encode())
    h.update(repr(tokens).encode())
    return h.hexdigest()


a = block_hash("", [1, 2])
assert a == block_hash("", [1, 2])
assert block_hash(a, [3]) != block_hash("", [3])   # 哈希包含前缀，形成链
```

<br>

### uuid

生成全局唯一 ID。`uuid4` 随机生成，最常用；`uuid5` 由命名空间和名称确定性生成。给请求、任务、文件生成不会冲突的 ID 时使用。

**实战**

```python
import uuid

request_id = f"chatcmpl-{uuid.uuid4().hex}"
assert len(uuid.uuid4().hex) == 32
assert request_id.startswith("chatcmpl-")
a = uuid.uuid5(uuid.NAMESPACE_URL, "model://qwen")
assert a == uuid.uuid5(uuid.NAMESPACE_URL, "model://qwen")
```

<br>

### tempfile

安全创建临时文件和目录，避免名字冲突。`TemporaryDirectory` 退出 `with` 时自动删除。

**实战：原子写入**

先写到同目录的临时文件，再 `os.replace`，读者不会看到写了一半的文件。

```python
import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


with tempfile.TemporaryDirectory() as d:
    target = Path(d) / "state.json"
    atomic_write(target, "{}")
    assert target.read_text() == "{}"
    assert len(list(Path(d).iterdir())) == 1
assert not Path(d).exists()
```

<br>
<br>

## 10. 底层与系统

<br>

### pickle

把 Python 对象序列化成字节，再还原。多进程通信默认使用它。类和函数只保存「模块路径 + 名字」，还原时要能导入到同名对象，所以 `lambda`、局部函数不能 pickle。

**实战**

```python
import pickle
from dataclasses import dataclass


@dataclass
class Request:
    id: int


data = pickle.dumps(Request(1))
assert pickle.loads(data) == Request(1)
try:
    pickle.dumps(lambda x: x)
except (pickle.PicklingError, AttributeError):
    pass
```

**坑：** **绝不要 unpickle 不可信数据**。pickle 可以在加载时执行任意代码。

<br>

### __getstate__ / __setstate__ / __reduce__

自定义 pickle 行为。`__getstate__` 返回要保存的状态，`__setstate__` 用它恢复，常用于跳过锁、连接等不能序列化的成员。`__reduce__` 更底层，返回「用什么函数和什么参数重建对象」。

**实战：跳过不可序列化的锁，以及用参数重建**

```python
import pickle
import threading


class Stats:
    def __init__(self):
        self.count = 1
        self.lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()


class Conn:
    def __init__(self, url: str):
        self.url = url
        self.socket = object()              # 模拟不能序列化的资源

    def __reduce__(self):
        return (Conn, (self.url,))          # 重建时调用 Conn(url)


s = pickle.loads(pickle.dumps(Stats()))
assert s.count == 1 and s.lock.acquire(False)
assert pickle.loads(pickle.dumps(Conn("x"))).url == "x"
```

<br>

### ctypes

不写 C 扩展就能调用动态库。`ctypes.CDLL` 加载库，要声明 `argtypes` 和 `restype`，否则默认按 int 处理，可能截断指针。`Structure` 描述 C 结构体的内存布局，常用于 FFI、NVML 这类系统库绑定。

**实战：调用 libc 与定义结构体**

```python
import ctypes

libc = ctypes.CDLL(None)                     # 当前进程已加载的 C 库
libc.strlen.argtypes = [ctypes.c_char_p]
libc.strlen.restype = ctypes.c_size_t
assert libc.strlen(b"hello") == 5


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]


p = Point(1, 2)
assert ctypes.sizeof(p) == 8
ptr = ctypes.pointer(p)
ptr.contents.x = 9                           # 通过指针修改同一块内存
assert p.x == 9
assert bytes(p)[:4] == (9).to_bytes(4, "little")
```

**坑：** 这里没有类型安全，声明错误可能直接导致段错误，不是 Python 异常。

<br>

### struct

在 Python 值和 C 布局的二进制数据之间转换。格式字符串描述布局：`<` 小端，`>` 大端，`I` 是 4 字节无符号整数，`d` 是 8 字节浮点，`B` 是 1 字节。解析网络协议或文件头时使用。

**实战：解析 safetensors 风格的文件头**

文件开头 8 字节是小端长度，后面是 JSON 头。

```python
import json
import struct

header = json.dumps({"w": {"dtype": "F32"}}).encode()
blob = struct.pack("<Q", len(header)) + header + b"\x00" * 4

size = struct.unpack_from("<Q", blob, 0)[0]
meta = json.loads(blob[8:8 + size])
assert meta["w"]["dtype"] == "F32"
assert struct.calcsize("<IB") == 5
assert struct.calcsize("@IB") >= 5           # 原生模式可能有填充
```

<br>

### mmap / 共享内存

`mmap` 把文件映射到虚拟内存，按页惰性读取，多个进程映射同一文件时可以共享物理页。加载大模型权重很适合。`multiprocessing.shared_memory` 让进程共享一块命名内存，避免序列化大数据。

**实战：mmap 随机访问，以及共享内存**

```python
import mmap
import multiprocessing.shared_memory as sm
import tempfile

with tempfile.TemporaryFile() as f:
    f.write(b"hello world")
    f.flush()
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        assert mm[6:11] == b"world"           # 不用读整个文件

owner = sm.SharedMemory(create=True, size=8)
try:
    owner.buf[:2] = b"ok"
    other = sm.SharedMemory(name=owner.name)  # 其他进程用名字打开
    assert bytes(other.buf[:2]) == b"ok"
    other.close()
finally:
    owner.close()
    owner.unlink()                            # 由创建方释放
```

**坑：** `close` 只断开当前进程的映射，`unlink` 才销毁内存块。忘记 `unlink` 会泄漏。

<br>

### gc

CPython 主要靠引用计数回收，计数归零时立即释放。循环引用的对象计数不会归零，由分代循环垃圾回收器处理。`gc.collect()` 手动触发。释放大模型后，常先 `del`、`gc.collect()`，再调用 `torch.cuda.empty_cache()`。

**实战：循环引用的回收**

```python
import gc
import weakref


class Node:
    pass


a, b = Node(), Node()
a.other, b.other = b, a          # 循环引用
ref = weakref.ref(a)
gc.disable()                     # 排除自动回收的干扰
del a, b
assert ref() is not None         # 引用计数无法回收
gc.collect()
gc.enable()
assert ref() is None             # 循环回收器处理掉了
```

<br>
<br>

## 11. 框架装饰器（只需认得）

这些装饰器依赖框架，用法都建立在第 5 章的装饰器机制上。目标是读源码时知道它们做什么。**本章代码是示意，依赖 torch、FastAPI 等环境，未运行验证。**

<br>

### @torch.inference_mode / @torch.no_grad

两者都关闭梯度记录，用于推理。`no_grad` 不构建计算图；`inference_mode` 限制更严，还跳过视图和版本计数跟踪，所以更快。但它产生的张量之后不能用于反向传播。

**实战**

```python
import torch


@torch.inference_mode()
def generate(model, input_ids):
    return model(input_ids)
```

<br>

### @torch.compile

`torch.compile` 通过 TorchDynamo 捕获计算图，再用后端优化，默认后端是 TorchInductor。首次运行编译较慢，之后更快。输入形状变化可能触发重新编译，所以常要声明动态维度。

**实战**

```python
import torch


@torch.compile
def add_scaled(x, y, a):
    return x + a * y
```

<br>

### @register_fake

为自定义算子注册 fake 实现，只根据输入计算输出的 shape 和 dtype，不做真实计算，也不申请真实内存。`torch.compile` 追踪时靠它推导形状。没有它，自定义算子无法被正常追踪。

**实战**

```python
import torch
from torch.library import custom_op, register_fake


@custom_op("mylib::scale", mutates_args=())
def scale(x: torch.Tensor, a: float) -> torch.Tensor:
    return x * a


@register_fake("mylib::scale")
def _(x, a):
    return torch.empty_like(x)
```

<br>

### @support_torch_compile

vLLM 用于标记模型类可以被编译，通过 `dynamic_arg_dims` 指明哪些参数的哪一维是动态的。编译流程由框架处理，模型类只需声明。

**实战**

```python
@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})
class LlamaModel(nn.Module):
    def forward(self, input_ids, positions):
        ...
```

<br>

### @traced

按名字理解是追踪装饰器，为函数或方法记录执行信息。原理就是第 5 章的 wrapper：调用前后加上记录逻辑。本机的 vLLM 副本比 README 的统计版本旧，里面搜不到它的定义，这段描述未对照源码，读到时以实际实现为准。

**实战**

```python
@traced
def schedule(self):
    ...
```

<br>

### @router.post

FastAPI 路由装饰器，把函数注册到路径上，属于注册表模式。FastAPI 读取参数的类型标注：pydantic 模型的参数从请求体解析并校验，`async def` 在事件循环运行。

**实战**

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Req(BaseModel):
    prompt: str


@router.post("/v1/completions")
async def create(req: Req) -> dict:
    return {"text": req.prompt}
```

<br>

### @triton.jit / @triton.autotune / @triton.heuristics

- `@triton.jit`：把 Python 语法的函数编译成 GPU kernel。函数体不是普通 Python，只能使用 Triton 支持的语法。
- `@triton.autotune`：首次运行时尝试多组配置，缓存最快的结果。
- `@triton.heuristics`：根据输入用函数决定配置值，不用实测。

**实战**：见 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)。

```python
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})], key=["n"])
@triton.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1, mask=mask)
```

<br>

### @cute.jit / @cute.kernel

CuTe DSL（CUTLASS 的 Python 接口）的装饰器，是 NVIDIA 用于编写高性能 GPU kernel 的 DSL。`@cute.kernel` 标记 GPU 上运行的入口函数，`@cute.jit` 标记由 kernel 调用的、会被编译的辅助函数。相比 Triton，它更底层，可以更细致地控制硬件。

**实战**

```python
import cutlass.cute as cute


@cute.jit
def helper(x):
    return x + 1


class AddOp:
    @cute.kernel
    def kernel(self, x):
        y = helper(x)
```

<br>
<br>

## 12. Python 内置容器与函数（LeetCode 常考）

刷题时 90% 的 Python 代码只用到下面这些：4 个内置容器、`collections` 里的 4 个、`heapq`、`bisect`，再加一组内置函数。先记住复杂度表，再看每个容器的常用写法。

<br>

### 复杂度速查

| 操作 | `list` | `dict` / `set` | `deque` | `heapq` |
|---|---|---|---|---|
| 按下标访问 `a[i]` | O(1) | — | O(n) | — |
| 末尾增删 `append` / `pop()` | O(1) | — | O(1) | — |
| 头部增删 `insert(0)` / `pop(0)` | **O(n)** | — | O(1) `appendleft` / `popleft` | — |
| 查找 `x in` | **O(n)** | O(1) | O(n) | O(n) |
| 插入 / 删除键 | — | O(1) | — | — |
| 取最小值 | O(n) | — | — | O(1) `h[0]` |
| 插入 / 弹出最小值 | — | — | — | O(log n) |
| 排序 `sort()` | O(n log n) | — | — | — |

记住两条：**在 list 里查找用 `in` 是 O(n)，要查找就换 set/dict；从 list 头部弹出是 O(n)，BFS 要用 deque。**

<br>

### list

list 是动态数组，按下标访问和在末尾增删都是 O(1)。需要按顺序存数据、按下标访问或当栈用时使用。

| 类型        | 合并写法  | 加一个元素                 |
| ----------- | --------- | -------------------------- |
| 集合 `set`  | `a | {x}` | `a | {x}` 或 `a.add(x)`    |
| 列表 `list` | `a + [x]` | `a + [x]` 或 `a.append(x)` |

```python
a = [3, 1, 2]
a.append(4)          # 末尾加      O(1)
a.pop()              # 末尾弹出    O(1)
a[-1]                # 最后一个元素
a[::-1]              # 反转后的新列表
a.sort()             # 原地排序，返回 None
b = sorted(a)        # 返回新列表，原列表不变

grid = [[0] * 3 for _ in range(2)]   # 二维数组，正确写法
stack = []                           # list 直接当栈用
stack.append(1); stack.pop()
```

| 接口 | 作用 |
|---|---|
| `append(x)` / `pop()` | 末尾增删，O(1)，当栈用 |
| `pop(i)` / `insert(i, x)` | 任意位置增删，O(n) |
| `a[i:j]` | 切片，**会复制**，O(j-i) |
| `sort(key=..., reverse=...)` | 原地稳定排序 |
| `index(x)` / `count(x)` | 线性查找、计数，O(n) |

**坑：** `[[0] * 3] * 2` 里两行是**同一个列表对象**，改一行，另一行也跟着变。二维数组要用列表推导式创建。

#### stack

在 Python 里直接用 `list` 当栈就行，这是标准做法：

python

```python
stack = []
stack.append(1)    # 入栈
stack.append(2)
stack[-1]          # 看栈顶：2
stack.pop()        # 出栈：2
not stack          # 判断栈是否为空
```

`append` 和 `pop()` 都在尾部操作，复杂度 O(1)。

**注意**：

-   对空列表 `pop()` 或取 `[-1]` 会报 `IndexError`，用之前先检查 `if stack:`。
-   别拿 `list` 当队列用。`pop(0)` 从头部删除，后面的元素都要往前挪，是 O(n)。队列要用 `collections.deque`，配合 `popleft()`，复杂度是 O(1)。



#### slice

默认值取决于**步长 step 是正数还是负数**：

`a[start:stop]` 取的是 `start` 到 `stop - 1`，不包括 `stop`。

-   ==想一直取到末尾，就**省略 stop**：`a[start:]`==
-   写 `-1` 的意思是"到倒数第二个为止"，最后一个不包含在内

|                             | start 默认                       | stop 默认                         | step 默认 |
| --------------------------- | -------------------------------- | --------------------------------- | --------- |
| step 为正（包括省略时的 1） | `0`：从**开头**开始              | `len(a)`：走到**末尾**为止        | `1`       |
| step 为负（比如 -1）        | `len(a) - 1`：从**最后一个**开始 | 走到**开头之前**为止，包含第 0 个 |           |

可以这样理解：省略 start 和 stop，就是**”沿着 step 的方向，从这一头一直走到另一头"**。

python

```python
a = [0, 1, 2, 3, 4]
a[-2:]     # -2 指倒数第 2 个元素，也就是 3。省略终点表示一直取到末尾，所以结果是最后两个元素 →  [3, 4]
a[::1]     # 从 0 走到末尾       → [0, 1, 2, 3, 4]
a[::-1]    # 从 4 走到开头之前   → [4, 3, 2, 1, 0]
a[2:]      # 从 2 走到末尾       → [2, 3, 4]
a[2::-1]   # 从 2 往前走到开头   → [2, 1, 0]
a[:2]      # 从 0 走到 2（不含） → [0, 1]
a[:2:-1]   # 从 4 往前走到 2（不含） → [4, 3]
```



#### extend

`list.extend(str)` 会把字符串拆成单个字符，逐个加进列表：

```python
a = []
a.extend("abc")   # ['a', 'b', 'c']
a.append("abc")   # ['a', 'b', 'c', 'abc']
```

要整体加进列表就用 `append`。

```python
a = []
a.extend("google")      # ['g', 'o', 'o', 'g', 'l', 'e']
a.extend(["google"])    # 再加一个整体 'google'
a.extend("")            # 空字符串，什么都不加
a.extend(123)           # TypeError: 'int' object is not iterable
```



<br>

### dict

dict 是哈希表，按键增删查都是平均 O(1)。从 Python 3.7 起，遍历顺序就是插入顺序。需要按键快速查找、计数或做映射时使用。

| 类型   | 合并    | 加一个元素（生成新对象） | 加一个元素（修改原对象） |
| ------ | ------- | ------------------------ | ------------------------ |
| `set`  | `a | b` | `a | {x}`                | `a.add(x)`               |
| `list` | `a + b` | `a + [x]`                | `a.append(x)`            |
| `dict` | `a | b` | `a | {k: v}`             | `a[k] = v`               |

**常考写法**

```python
d = {}
d["a"] = 1
d.get("b", 0)                 # 键不存在时返回默认值，不报错
d.setdefault("c", []).append(1)
for k, v in d.items():        # 同时遍历键和值
    pass
"a" in d                      # 判断键是否存在，O(1)

# 两数之和：存 值 -> 下标
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```

| 接口 | 作用 |
|---|---|
| `d[k]` | 取值，键不存在时抛 `KeyError` |
| `d.get(k, default)` | 取值，键不存在时返回默认值 |
| `d.setdefault(k, v)` | 键不存在时先放入 v，然后返回 `d[k]` |
| `d.pop(k, default)` | 删除并返回值 |
| `keys()` / `values()` / `items()` | 遍历 |

| 写法                | 作用                                                  | key 不存在时             |
| ------------------- | ----------------------------------------------------- | ------------------------ |
| `del d[k]`          | 删除 k，不返回值, O(1)                                | 报 `KeyError`            |
| `d.pop(k)`          | 删除 k，`dict.pop(key)` **只返回 value**。            | 报 `KeyError`            |
| `d.pop(k, default)` | 同上, 返回默认值                                      | 返回 `default`，不报错   |
| `d.popitem()`       | 删除并返回**最后插入**的 `(k, v)`, 不能指定删哪个 key | dict 为空时报 `KeyError` |
| `d.clear()`         | 清空整个 dict                                         | —                        |

**坑：** 不能在遍历 dict 的同时增删键，会报 `RuntimeError`。要删就先遍历 `list(d)`。

<br>

### set

set 是只有键、没有值的哈希表，用来去重和做 O(1) 的存在性判断。

| 类型       | 合并写法  | 加一个元素              |
| ---------- | --------- | ----------------------- |
| 集合 `set` | `a | {x}` | `a | {x}` 或 `a.add(x)` |

```python
s = set()
s.add(1); s.discard(2)        # discard：元素不存在也不报错
1 in s                        # O(1)
a, b = {1, 2, 3}, {2, 3, 4}
a & b, a | b, a - b           # 交集、并集、差集

visited = set()               # BFS / DFS 的访问标记
```

| 接口 | 作用 |
|---|---|
| `add(x)` | 加入 |
| `remove(x)` / `discard(x)` | 删除；元素不存在时，remove 报错，==discard 不报错== |
| `&` `\|` `-` `^` | 交、并、差、对称差 |

**坑：** 元素必须可哈希。`list` 不能放进 set，要先转成 `tuple`，比如 `visited.add((r, c))`。

<br>

### tuple

tuple 是不可变的 list，可以哈希，所以能当 dict 的键或放进 set。

| 类型    | 合并    | 加一个元素                 |
| ------- | ------- | -------------------------- |
| `set`   | `a | b` | `a | {x}` / `a.add(x)`     |
| `list`  | `a + b` | `a + [x]` / `a.append(x)`  |
| `tuple` | `a + b` | `a + (x,)`（只能生成新的） |
| `dict`  | `a | b` | `a | {k: v}` / `a[k] = v`  |
| `deque` | `a + b` | `append` / `appendleft`    |

```python
point = (1, 2)
r, c = point                  # 解包
memo = {(r, c): 0}            # 多维状态当键
a, b = 1, 2
a, b = b, a                   # 交换两个变量，本质就是 tuple 打包再解包
```

<br>

### str

str 是不可变的字符序列，任何"修改"都会生成新字符串。

```python
s = "hello"

s[::-1]                       # 反转

"".join(["a", "b"])           # 拼接：先放进 list，最后 join

s.split(), s.strip()

s.isdigit(), s.isalpha(), s.isalnum()

ord("a"), chr(97)             # 字符和编码互转, ord 返回单个字符对应的 Unicode 编码（一个整数），chr 是它的反向操作. ord 只接受一个字符，ord("ab") 会报 TypeError。

ord(ch) - ord("a")            # 字母映射到 0-25

a.extend(int(c) for c in str(123))   # [1, 2, 3]
```

**坑：** 在循环里写 `s += ch`，每次都会复制整个字符串，最坏 O(n²)。应该先 append 到 list，最后 `"".join(...)`。

<br>

### collections.deque

deque 是双端队列，两头增删都是 O(1)。BFS 用它。

`list.pop(0)` 要把后面所有元素前移一位，是 O(n)。BFS 每轮都要从头部弹出，用 list 整体会退化成 O(n²)。

```python
from collections import deque

def bfs(start, graph):
    q = deque([start])
    seen = {start}
    while q:
        node = q.popleft()          # O(1)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen

window = deque(maxlen=3)            # 定长窗口，满了自动挤掉最旧的元素
```

| 接口 | 作用 |
|---|---|
| `append` / `pop` | 右端增删 |
| `appendleft` / `popleft` | 左端增删 |
| `deque(maxlen=k)` | 固定长度 |

**单调队列（滑动窗口最大值）：** deque 里存下标，并保持对应的值单调递减。

```python
def max_sliding_window(nums, k):
    q, res = deque(), []
    for i, x in enumerate(nums):
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        if q[0] <= i - k:
            q.popleft()
        if i >= k - 1:
            res.append(nums[q[0]])
    return res
```

<br>

### collections.Counter

Counter 是专门用来计数的 dict：值是出现次数，访问不存在的键返回 0。计数、比较字母组成（如异位词）时使用。

```python
from collections import Counter

c = Counter("aabbbc")          # Counter({'b': 3, 'a': 2, 'c': 1})
c["z"]                         # 0，不报错
c.most_common(2)               # [('b', 3), ('a', 2)]，Top K
Counter("abc") == Counter("cba")   # 判断字母异位词
```

<br>

### collections.defaultdict

defaultdict 在访问不存在的键时，会用工厂函数自动创建默认值。分组、建图的邻接表时使用。

```python
from collections import defaultdict

graph = defaultdict(list)      # 邻接表
for u, v in [(1, 2), (1, 3)]:
    graph[u].append(v)

groups = defaultdict(list)     # 字母异位词分组
for w in ["eat", "tea", "tan"]:
    groups["".join(sorted(w))].append(w)
```

| 工厂函数 | 默认值 | 典型用途 |
|---|---|---|
| `list` | `[]` | 邻接表、分组 |
| `int` | `0` | 计数 |
| `set` | `set()` | 去重的邻接表 |

**坑：** 只是读一下 `d[k]`，也会把 k 插进字典。只想判断键在不在，要用 `k in d`。

#### 无参 lambda

区别只在每次生成的默认值不一样。

| 写法                                    | 新 key 的默认值         | 用法             | 适合场景                                      |
| --------------------------------------- | ----------------------- | ---------------- | --------------------------------------------- |
| `defaultdict(lambda: [0, 0])`           | 新列表 `[0, 0]`         | `d[key][0] += x` | 一个 key 要同时存几个固定的数（如总和、次数） |
| `defaultdict(lambda: defaultdict(int))` | 新的 `defaultdict(int)` | `d[a][b] += x`   | 按两层 key 计数或累加（如起点 → 终点）        |

```python
# 每个 key 存固定几个数
d = defaultdict(lambda: [0, 0])
d["小明"][0] += 90    # 总分
d["小明"][1] += 1     # 次数
# {"小明": [90, 1]}

# 每个 key 下还有子 key
d = defaultdict(lambda: defaultdict(int))
d["小明"]["苹果"] += 3
d["小明"]["香蕉"] += 2
# {"小明": {"苹果": 3, "香蕉": 2}}
```

两者的共同点：

-   必须用 `lambda:` 包一层。直接写 `[0, 0]` 或 `defaultdict(int)` 传的是对象，不是工厂，会报错或者让所有 key 共用同一份数据。
-   每遇到一个新 key，lambda 都会被调用一次，生成**新的**对象，各个 key 之间互不影响。

<br>

### collections.OrderedDict

OrderedDict 是带顺序操作的 dict，可以 O(1) 把键移到末尾，或从两端弹出。LRU 缓存就是它的经典题。

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap, self.d = capacity, OrderedDict()

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)          # 标记为最近使用
        return self.d[key]

    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)   # 弹出最久未使用的
```

<br>

### heapq

heapq 在普通 list 上维护**最小堆**：==`h[0]` 永远是最小值，插入和弹出都是 O(log n)==。需要反复取最小值时使用：Top K、Dijkstra、合并 K 个有序链表。

-   小顶堆 `large`：`self.large[0]` 是最小值
-   大顶堆 `small`（存负数）：`-self.small[0]` 是最大值

| 类型         | 合并                        | 加一个元素              |
| ------------ | --------------------------- | ----------------------- |
| `set`        | `a | b`                     | `a.add(x)`              |
| `list`       | `a + b`                     | `a.append(x)`           |
| `dict`       | `a | b`                     | `a[k] = v`              |
| heap（list） | `a + b` 再 `heapify`        | `heappush(h, x)`        |
| `deque`      | `a + b`（两边都要是 deque） | `append` / `appendleft` |

```python
import heapq

h = []
heapq.heappush(h, 3)
heapq.heappush(h, 1)
heapq.heappop(h)               # 1
h[0]                           # 查看最小值，不弹出
heapq.heapify(nums := [5, 2, 8])   # 原地建堆，O(n)

# 最大堆：存负数
mx = []
for x in [3, 1, 2]:
    heapq.heappush(mx, -x)
-mx[0]                         # 3，最大值

# 第 K 大：维护大小为 k 的最小堆
def kth_largest(nums, k):
    h = []
    for x in nums:
        heapq.heappush(h, x)
        if len(h) > k:
            heapq.heappop(h)
    return h[0]

# 按优先级排序的元组：(距离, 节点)，Dijkstra 常用
pq = []                        # 新开一个堆，不要和上面存 int 的堆混用
heapq.heappush(pq, (0, "A"))
```

| 接口 | 作用 |
|---|---|
| `heappush(h, x)` / `heappop(h)` | 插入 / 弹出最小值，O(log n) |
| `heapify(list)` | 原地建堆，O(n) |
| `heappushpop(h, x)` | 先 push 再 pop，比分开调用更快 |
| `nlargest(k, it)` / `nsmallest(k, it)` | 直接取 Top K |

**坑：** 只有最小堆，最大堆要存负数。元组的第一个元素相等时会去比较第二个元素，如果第二个元素不可比较（比如节点对象），就会报错。这时在中间加一个计数器：`(dist, i, node)`。

<br>

### bisect

bisect 在**已排序**的 list 上做二分查找，O(log n) 找到插入位置。

```python
import bisect

a = [1, 3, 3, 5]
bisect.bisect_left(a, 3)       # 1：第一个 >= 3 的位置
bisect.bisect_right(a, 3)      # 3：第一个 > 3 的位置
bisect.insort(a, 4)            # 插入并保持有序（因为要移动元素，整体是 O(n)）

# 最长递增子序列 O(n log n)
def length_of_lis(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

<br>

### 内置函数

这些函数让循环写得更短，也更不容易错。

```python
nums = [3, 1, 2]
for i, x in enumerate(nums): ...          # 同时拿下标和值
for a, b in zip(nums, nums[1:]): ...      # 相邻元素两两配对
sum(nums), max(nums), min(nums)
max(nums, key=abs)                        # 按规则取最大
any(x > 2 for x in nums), all(x > 0 for x in nums)
list(map(int, "123"))                     # [1, 2, 3]
divmod(7, 3)                              # (2, 1)
float("inf"), -float("inf")               # 正负无穷，用作初始值

words = ["bb", "a", "ccc"]
sorted(words, key=len)                    # 按长度排
sorted(words, key=lambda w: (-len(w), w)) # 先按长度降序，再按字典序
intervals = [[1, 3], [0, 2]]
intervals.sort(key=lambda x: x[0])        # 区间题：按起点排序
```

| 函数 | 作用 |
|---|---|
| `enumerate(it, start=0)` | 同时产出 (下标, 值) |
| `zip(a, b)` | 并行遍历，按最短的那个截断 |
| `sorted(it, key=, reverse=)` | 返回新的有序列表，稳定排序 |
| `max` / `min(it, key=)` | 按 key 取最值 |
| `any` / `all` | 有一个为真 / 全部为真 |
| `range(a, b, step)` | 整数序列，不占内存 |
| `divmod(a, b)` | 同时得到商和余数 |

<br>

### itertools 与 functools

itertools 生成组合、排列和累加序列；`functools.cache` 一行实现记忆化搜索。

```python
from itertools import permutations, combinations, product, accumulate, pairwise
from functools import cache

list(permutations([1, 2, 3], 2))      # 排列
list(combinations([1, 2, 3], 2))      # 组合
list(product([0, 1], repeat=2))       # 笛卡尔积
list(accumulate([1, 2, 3]))           # [1, 3, 6] 前缀和
list(accumulate([1, 2, 3], initial=0))# [0, 1, 3, 6]，区间和更方便
list(pairwise([1, 2, 3]))             # [(1, 2), (2, 3)]，Python 3.10+

@cache                                # 记忆化：把 DFS 变成 DP
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
```

**坑：** 递归深的题要提高递归上限：`sys.setrecursionlimit(10**6)`。`@cache` 的参数必须可哈希，list 要先转成 tuple。

<br>

### 面试题选型

| 题型 | 用什么 |
|---|---|
| 两数之和、存在性判断 | `dict` / `set` |
| 计数、字母异位词、Top K 频率 | `Counter` |
| 图的邻接表、分组 | `defaultdict(list)` |
| BFS、层序遍历 | `deque` |
| 滑动窗口最大值 | `deque` 单调队列 |
| 括号匹配、单调栈 | `list` 当栈 |
| 第 K 大、合并 K 个有序链表、Dijkstra | `heapq` |
| 有序数组查找、LIS | `bisect` |
| LRU 缓存 | `OrderedDict` |
| 前缀和 | `accumulate` |
| 记忆化搜索 | `@cache` |
| 网格访问标记 | `set` 存 `(r, c)` |

<br>
<br>

## 13. 线程同步工具

保护数据用 `Lock`，限制数量用 `Semaphore`，等信号用 `Event`，等条件用 `Condition`，凑齐一组用 `Barrier`。以下工具都在 `threading` 模块里。

| 工具 | 作用 | 典型场景 |
|---|---|---|
| `Lock` | 互斥锁，同一时刻只有一个线程能进入 | 保护共享数据，比如停车场的计数 |
| `RLock` | 可重入锁，同一线程可以多次获取 | 加锁的方法里又调用了加锁的方法 |
| `Semaphore(n)` | 计数器，最多允许 n 个线程同时进入 | 限流、车位、连接池 |
| `BoundedSemaphore(n)` | 同上，但 `release` 次数多于 `acquire` 时会报错 | 防止误多还资源 |
| `Event` | 一个开关，`set()` 之后所有在 `wait()` 的线程一起放行 | 一个线程通知其他线程“可以开始了”（1114） |
| `Condition` | 锁加上等待和通知，满足条件才继续 | 生产者-消费者、按条件轮流执行（1188、1195） |
| `Barrier(n)` | 凑齐 n 个线程后一起放行 | 分组执行（1117 H2O） |

<br>

### Lock：互斥锁

`count += 1` 分“读、算、写”三步，不加锁可能在中间被打断，导致结果偏小。

**实战：4 个线程各加 10 万次**

加锁后结果稳定是 400000。

```python
import threading

count = 0
lock = threading.Lock()

def add():
    global count
    for _ in range(100000):
        with lock:          # 同一时刻只有一个线程能改 count
            count += 1

threads = [threading.Thread(target=add) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(count)   # 400000
```

<br>

### RLock：可重入锁

同一个线程可以多次获取同一把 `RLock`。获取几次就要释放几次，计数归零后其他线程才能拿到锁；`with` 会自动配对。

**实战：加锁的方法里调用加锁的方法**

`inc_twice` 已经持有锁，又调用了也要加锁的 `inc`。用 `RLock` 正常输出 2；换成 `Lock` 会死锁。

```python
import threading

class Counter:
    def __init__(self):
        self.n = 0
        self.lock = threading.RLock()

    def inc(self):
        with self.lock:
            self.n += 1

    def inc_twice(self):
        with self.lock:
            self.inc()      # 同一线程再次获取同一把锁
            self.inc()

c = Counter()
c.inc_twice()
print(c.n)   # 2
```

<br>

### Semaphore：计数器

内部是一个计数器，`acquire` 减一，`release` 加一，减到 0 时后来的线程排队等。

**实战：2 个车位、5 辆车**

任何时刻最多只有 2 辆停在里面，其余的排队等。

```python
import threading, time

spots = threading.Semaphore(2)    # 2 个车位

def park(car):
    with spots:                   # acquire：占一个车位，满了就等
        print(f"{car} 停入")
        time.sleep(0.1)
        print(f"{car} 离开")      # 离开 with 时自动 release

threads = [threading.Thread(target=park, args=(f"车{i}",)) for i in range(1, 6)]
for t in threads: t.start()
for t in threads: t.join()
```

运行结果（顺序可能略有不同）：

```text
车1 停入
车2 停入
车1 离开
车2 离开
车3 停入
车4 停入
...
```

只想试一下、拿不到就放弃，用 `spots.acquire(blocking=False)`，它返回 `True` 或 `False`（停车场 1603 的写法）。

<br>

### BoundedSemaphore：有上限的计数器

资源数量固定的场景（车位、连接池）优先用 `BoundedSemaphore`，可以提前发现 bug。

**实战：多还一次资源**

`BoundedSemaphore` 会报 `ValueError`；普通 `Semaphore` 不报错，计数会悄悄变成 3，车位凭空多出一个。

```python
import threading

sem = threading.BoundedSemaphore(2)
sem.acquire()
sem.release()
sem.release()   # ValueError: Semaphore released too many times

sem2 = threading.Semaphore(2)
sem2.release()  # 不报错，计数变成 3
```

<br>

### Event：开关信号

`set()` 之后开关一直是开的，后来的 `wait()` 会直接通过；要重新关上用 `clear()`。

**实战：主线程发开始信号**

工作线程在 `wait()` 处停住，直到主线程调用 `set()` 才继续。

```python
import threading, time

ready = threading.Event()

def worker():
    print("工作线程：等待开始信号")
    ready.wait()                 # 阻塞，直到 set()
    print("工作线程：收到信号，开始工作")

t = threading.Thread(target=worker)
t.start()
time.sleep(0.2)
print("主线程：发出信号")
ready.set()                      # 放行所有在等的线程
t.join()
```

输出：

```text
工作线程：等待开始信号
主线程：发出信号
工作线程：收到信号，开始工作
```

<br>

### Condition：等条件成立

`Condition` 是一把锁加上等待和通知。线程持锁检查条件，不满足就 `wait`，别的线程改完数据后 `notify` 把它唤醒。

**实战：生产者-消费者**

消费者先启动，等到列表非空才取商品；生产者放入商品后用 `notify()` 把它唤醒。

```python
import threading, time

items = []
cond = threading.Condition()

def consumer():
    with cond:
        print("消费者：等待商品")
        cond.wait_for(lambda: len(items) > 0)   # 等待时会暂时释放锁
        print("消费者：取到", items.pop(0))

def producer():
    with cond:
        items.append("面包")
        print("生产者：放入 面包")
        cond.notify()                           # 唤醒一个在等的线程

c = threading.Thread(target=consumer); c.start()
time.sleep(0.1)
p = threading.Thread(target=producer); p.start()
c.join(); p.join()
```

输出：

```text
消费者：等待商品
生产者：放入 面包
消费者：取到 面包
```

- `wait_for` 等待时会释放锁，所以生产者才能进入 `with cond`。
- 被唤醒后会重新检查条件，不满足就继续等。
- `notify_all()` 唤醒所有在等的线程。

<br>

### Barrier：凑齐再走

放行一组后，`Barrier` 会自动重置，可以接着凑下一组。

**实战：3 名选手一起出发**

3 名选手先后就位，先到的在 `wait()` 处等，第 3 个到齐后三人一起出发。

```python
import threading, time

barrier = threading.Barrier(3)

def runner(name, delay):
    time.sleep(delay)
    print(f"{name} 就位")
    barrier.wait()               # 凑齐 3 个才放行
    print(f"{name} 出发")

threads = [threading.Thread(target=runner, args=(n, d))
           for n, d in [("A", 0.1), ("B", 0.2), ("C", 0.3)]]
for t in threads: t.start()
for t in threads: t.join()
```

输出（出发顺序不固定）：

```text
A 就位
B 就位
C 就位
C 出发
B 出发
A 出发
```

<br>

### 补充

默认用 `Lock`，确实需要时再换其他工具。

- **用 `with` 加锁**：出异常或中途 `return` 也会自动释放，避免忘记 `release()` 造成死锁。`Lock`、`RLock`、`Semaphore`、`Condition` 都支持。
- **`queue.Queue`**：线程安全的阻塞队列，生产者-消费者一般直接用它：`q.put(x)` 放入，`q.get()` 取出，队列为空时自动等待。
- **`Lock` 还是 `RLock`**：只有同一线程嵌套获取同一把锁时才用 `RLock`；也可以把内层方法拆成不加锁的 `_inc`，继续用 `Lock`。
