# TRE v2 Console 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans，按任务逐个实现。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 把 console 重构为四页（Overview / Signals / Fleet / Ops）、以全宽时序大图为核心的操作台，并补齐 07-15 恢复架构的可观测面，全程不影响扩缩容性能。

**Architecture:** 后端只改 `ui/` —— sampler 增加廉价只读源的 2s 轮询与 signal_log 增量环，新增三个端点；controller 与 service-manager 代码零改动。前端由单文件 app.js 拆为 StaticFiles 挂载的 ES modules，图表用内联 SVG 自绘。

**Tech Stack:** Python 3 / FastAPI / Redis / 原生 JS ES modules / 内联 SVG。无 CDN、无前端框架（UI pod 除 docker.io 外无外网）。

## Global Constraints

- 设计依据：`tre/docs/design/20260725-console-redesign.md`。
- **绝不改动** `tre/controller/`、`tre/service-manager/`、`aibrix-system` 命名空间。
- **`/v2/audit` 绝不进入任何自动轮询路径**（它会 list k8s pods 并逐个 HTTP 探测 20 个 vLLM pod 的 `/is_sleeping`）。仅按钮触发。
- SSE 推送频率保持 0.5s 不变；历史曲线不得塞进 SSE 通道。
- 浏览器只读 sampler 内存快照，永不触发上游读（sampler 模块文档中已声明的架构不变式）。
- 读 `tre:v2:controller:signal_log` **只能用 XREVRANGE/XRANGE + COUNT 上限**，禁止全量扫描（该 stream 有 20 万条）。
- 权威检查：`cd /data/nfs_shared_data/xxy/aibrix/tre && make check`，当前基线 **580 passed**。
- 单测临时跑：`PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest <path> -x`
- 空载时 `z_m` 为 `nan`：前端必须显示「idle · 无负载」，不得渲染为 `—`。
- 提交在 76 上的 `/data/nfs_shared_data/xxy/aibrix`，branch `main`。

## 文件结构

| 文件 | 职责 |
|---|---|
| `ui/tre_ui/sampler.py`（改） | 新增 fleet/ops/leases 轮询源与 signal_log 增量环；audit 不得出现 |
| `ui/tre_ui/app.py`（改） | 新增 `/api/signal/timeline`、`GET+POST /api/ops/audit`；静态资源改 StaticFiles 挂载 |
| `ui/tre_ui/static/index.html`（重写） | 外壳 + 四个 view 容器 |
| `ui/tre_ui/static/style.css`（重写） | 布局与主题 |
| `ui/tre_ui/static/js/core.js`（新） | 全局状态、SSE、格式化助手、导航 |
| `ui/tre_ui/static/js/chart.js`（新） | 时序图组件：纯标度函数 + SVG 渲染 + 十字准星联动 |
| `ui/tre_ui/static/js/view_overview.js`（新） | Overview 页 |
| `ui/tre_ui/static/js/view_signals.js`（新） | Signals 页 |
| `ui/tre_ui/static/js/view_fleet.js`（新） | Fleet 页 |
| `ui/tre_ui/static/js/view_ops.js`（新） | Ops 页 |
| `ui/tre_ui/static/js/main.js`（新） | 入口装配 |
| `ui/tests/test_sampler.py`（改） | 新源测试 + audit 守卫测试 |
| `ui/tests/test_ui_app.py`（改） | 新端点与静态挂载测试 |

**已知取舍：** 本仓库没有 JS 测试运行环境，引入一套 node 测试栈超出本次范围。因此 chart.js 的标度计算写成不依赖 DOM 的纯函数并集中在文件顶部，正确性由 Task 9 的在线验证覆盖；如需要，后续可单独补 JS 测试栈。

---

### Task 1: sampler 接入廉价只读源（fleet / supervisor / operations / gpu leases）

**Files:**
- Modify: `ui/tre_ui/sampler.py`
- Modify: `ui/tre_ui/app.py`（Sampler 构造改传 client）
- Test: `ui/tests/test_sampler.py`

**Interfaces:**
- Consumes: `ServiceManagerClient.request(method, path, payload=None) -> dict`（`ui/tre_ui/server.py:21` 已实现）
- Produces: snapshot 新增顶层 key `fleet`（`{"state": dict, "supervisor": dict, "age_ms": int}`）、`operations`（`{"items": list, "age_ms": int}`）、`leases`（`{"gpus": dict, "age_ms": int}`）

- [ ] **Step 1: 写失败测试**

在 `ui/tests/test_sampler.py` 追加：

```python
class FakeSmClient:
    def __init__(self):
        self.calls = []

    def get_state(self):
        self.calls.append("/v2/state")
        return {"version": 1, "bindings": []}

    def request(self, method, path, payload=None):
        self.calls.append(path)
        if path == "/v2/fleet/state":
            return {"desired_version": 3, "observed_version": 3,
                    "desired": [], "observed": [], "mismatches": []}
        if path == "/v2/supervisor":
            return {"enabled": True, "running": True, "drift_observations": 0}
        if path.startswith("/v2/operations"):
            return {"operations": [{"id": "op-1", "status": "succeeded"}]}
        raise AssertionError(f"unexpected path {path}")


def test_sampler_publishes_fleet_supervisor_and_operations():
    sampler = Sampler(FakeRedis(), FakeSmClient(), model_names=["m1"])
    sampler.sample_once()
    snap = sampler.snapshot()

    assert snap["fleet"]["state"]["desired_version"] == 3
    assert snap["fleet"]["supervisor"]["running"] is True
    assert snap["operations"]["items"][0]["id"] == "op-1"


def test_sampler_never_polls_the_expensive_audit_endpoint():
    """Guard: /v2/audit lists k8s pods and probes every vLLM pod over HTTP.

    Polling it would put load on the model pods and perturb scaling
    experiments, so it must stay button-triggered only.
    """
    client = FakeSmClient()
    sampler = Sampler(FakeRedis(), client, model_names=["m1"])
    for _ in range(20):
        sampler.sample_once()

    assert not any("audit" in path for path in client.calls)
    assert not any("audit" in source for source in _RATES)
```

文件顶部 import 补上 `_RATES`（与既有 `from tre_ui.sampler import ...` 行合并）。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/nfs_shared_data/xxy/aibrix/tre && PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests/test_sampler.py -x -q`
Expected: FAIL —— `TypeError`（Sampler 第二参数目前是 `sm_get_state` 可调用对象）或 `KeyError: 'fleet'`

- [ ] **Step 3: 实现**

`sampler.py` 速率表加两个源（gpu leases 复用 `gpu` 节拍，不新增源）：

```python
_RATES = {"decision": 1.0, "hist": 2.0, "sm": 2.0, "gpu": 5.0, "probes": 2.0,
          "fleet": 2.0, "ops": 2.0}
_OPS_LIMIT = 50
```

构造函数第二参数由 `sm_get_state: Callable[[], dict]` 改为 `sm_client: Any`，保存为 `self._sm = sm_client`；`_read_sm` 内改为 `self._parts["sm"] = self._sm.get_state()`。

新增两个 reader：

```python
    def _read_fleet(self) -> bool:
        # Cheap: fleet state is two Redis reads, supervisor is in-memory.
        # NEVER add /v2/audit here -- it lists k8s pods and HTTP-probes every
        # vLLM pod, which would perturb the models under load.
        try:
            state = self._sm.request("GET", "/v2/fleet/state")
        except Exception as exc:  # noqa: BLE001
            state = {"error": str(exc)}
        try:
            supervisor = self._sm.request("GET", "/v2/supervisor")
        except Exception as exc:  # noqa: BLE001
            supervisor = {"error": str(exc)}
        self._parts["fleet"] = {"state": state, "supervisor": supervisor}
        self._ages["fleet"] = self._now_ms()
        return True

    def _read_ops(self) -> bool:
        try:
            payload = self._sm.request("GET", f"/v2/operations?limit={_OPS_LIMIT}")
            items = payload.get("operations", [])
        except Exception:  # noqa: BLE001
            items = []
        self._parts["operations"] = items
        self._ages["ops"] = self._now_ms()
        return True
```

`sample_once` 的 reader 元组追加 `("fleet", self._read_fleet), ("ops", self._read_ops)`。

`_read_gpu` 末尾追加 lease 读取（Redis hash，廉价）：

```python
        try:
            raw = self._redis.hgetall("tre:v2:sm:gpu_leases") or {}
            self._parts["leases"] = {_text(k): _loads(v) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            self._parts["leases"] = {}
```

`_rebuild` 的 snapshot 字典追加：

```python
            "fleet": {**(self._parts.get("fleet") or {"state": {}, "supervisor": {}}),
                      "age_ms": self._age(now, "fleet")},
            "operations": {"items": self._parts.get("operations", []),
                           "age_ms": self._age(now, "ops")},
            "leases": {"gpus": self._parts.get("leases", {}),
                       "age_ms": self._age(now, "gpu")},
```

`app.py` 构造改为 `sampler = Sampler(redis_client, service_manager_client, model_names=model_names)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests -q`
Expected: PASS（既有 ui 测试若因构造签名变化失败，同步改为传 client 对象）

- [ ] **Step 5: 提交**

```bash
cd /data/nfs_shared_data/xxy/aibrix
git add tre/ui && git commit -m "feat(ui): sample fleet, supervisor and operations state"
```

---

### Task 2: signal_log 增量环与 /api/signal/timeline

**Files:**
- Modify: `ui/tre_ui/sampler.py`, `ui/tre_ui/app.py`
- Test: `ui/tests/test_sampler.py`, `ui/tests/test_ui_app.py`

**Interfaces:**
- Consumes: Redis stream `tre:v2:controller:signal_log`，字段 `ts, window_id, model, z_m, tss, queue_len, decode_tps, prefill_tps, replicas_awake, replicas_target, tier, action, theta_m`
- Produces: `Sampler.timeline(model: str, since_ms: int = 0) -> list[dict]`；`GET /api/signal/timeline?model=<name>&since_ms=<int>` → `{"model": str, "points": list[dict]}`

- [ ] **Step 1: 写失败测试**

`ui/tests/test_sampler.py` 追加：

```python
class FakeStreamRedis(FakeRedis):
    def __init__(self, entries):
        super().__init__()
        self.entries = list(entries)  # [(id, {field: value})]
        self.stream_calls = []

    def xrevrange(self, key, max="+", min="-", count=None):
        self.stream_calls.append(("xrevrange", count))
        out = list(reversed(self.entries))
        return out[:count] if count else out

    def xrange(self, key, min="-", max="+", count=None):
        self.stream_calls.append(("xrange", count))
        start = str(min).lstrip("(")
        out = [e for e in self.entries if e[0] > start]
        return out[:count] if count else out


def _entry(entry_id, model, ts_ms, z, action="none"):
    return (entry_id, {
        b"ts": str(ts_ms / 1000).encode(), b"window_id": str(ts_ms).encode(),
        b"model": model.encode(), b"z_m": str(z).encode(), b"queue_len": b"3",
        b"decode_tps": b"120", b"prefill_tps": b"400", b"replicas_awake": b"1",
        b"replicas_target": b"2", b"tier": b"healthy", b"action": action.encode(),
    })


def test_timeline_backfills_then_reads_incrementally():
    redis = FakeStreamRedis([_entry("1-0", "m1", 1000, 0.5)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])
    sampler.sample_once()

    assert [p["z_m"] for p in sampler.timeline("m1")] == [0.5]
    assert redis.stream_calls[0][0] == "xrevrange"

    redis.entries.append(_entry("2-0", "m1", 2000, 0.9, action="scale_up"))
    sampler._next["timeline"] = 0.0
    sampler.sample_once()

    points = sampler.timeline("m1")
    assert [p["z_m"] for p in points] == [0.5, 0.9]
    assert points[1]["action"] == "scale_up"
    assert redis.stream_calls[-1][0] == "xrange"


def test_timeline_reads_are_always_count_bounded():
    redis = FakeStreamRedis([_entry(f"{i}-0", "m1", i * 1000, 0.1) for i in range(1, 50)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])
    sampler.sample_once()

    assert all(call[1] is not None and call[1] > 0 for call in redis.stream_calls)


def test_timeline_decodes_nan_as_none():
    redis = FakeStreamRedis([_entry("1-0", "m1", 1000, float("nan"))])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])
    sampler.sample_once()

    assert sampler.timeline("m1")[0]["z_m"] is None


def test_timeline_filters_by_since_ms():
    redis = FakeStreamRedis([_entry("1-0", "m1", 1000, 0.5), _entry("2-0", "m1", 5000, 0.7)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])
    sampler.sample_once()

    assert [p["z_m"] for p in sampler.timeline("m1", since_ms=3000)] == [0.7]
```

`ui/tests/test_ui_app.py` 追加（沿用该文件既有的 app 装配 fixture）：

```python
def test_signal_timeline_endpoint_returns_points_for_known_model(client):
    response = client.get("/api/signal/timeline?model=m1")

    assert response.status_code == 200
    assert response.json()["model"] == "m1"
    assert isinstance(response.json()["points"], list)


def test_signal_timeline_endpoint_rejects_unknown_model(client):
    assert client.get("/api/signal/timeline?model=nope").status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests -x -q`
Expected: FAIL —— `AttributeError: 'Sampler' object has no attribute 'timeline'`

- [ ] **Step 3: 实现**

`sampler.py` import 与常量：

```python
from tre_common.rediskeys import (
    CONTROLLER_SAFESCALE_PROBES_KEY, CONTROLLER_SIGNAL_LOG_KEY,
    DECISION_LATEST_KEY, decision_hist_key,
)

_TIMELINE_RING = 1200      # points per model (~1h at one 5s window per model)
_TIMELINE_BACKFILL = 4000  # bounded first read; the stream holds ~200k entries
_TIMELINE_INCREMENT = 500  # bounded per-tick read
_TIMELINE_NUMERIC = ("z_m", "tss", "queue_len", "decode_tps", "prefill_tps",
                     "replicas_awake", "replicas_target", "theta_m")
```

`_RATES` 追加 `"timeline": 2.0`。构造函数追加：

```python
        self._timeline: dict[str, list[dict]] = {m: [] for m in model_names}
        self._timeline_last_id: str | None = None
```

模块级解码函数：

```python
def decode_signal_row(fields: dict[Any, Any]) -> dict[str, Any]:
    """Decode one signal_log entry; nan and absent numerics both become None."""
    row = {_text(k): _text(v) for k, v in (fields or {}).items()}
    out: dict[str, Any] = {
        "model": row.get("model"),
        "tier": row.get("tier"),
        "action": row.get("action"),
        "signal_source": row.get("signal_source"),
    }
    window = row.get("window_id")
    out["ts_ms"] = int(window) if window and window.lstrip("-").isdigit() else None
    for field in _TIMELINE_NUMERIC:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            value = float("nan")
        out[field] = None if value != value else value  # nan -> None
    return out
```

reader：

```python
    def _read_timeline(self) -> bool:
        # Bounded reads only: this stream is capped at 200k entries, so a full
        # XRANGE would be a multi-second scan. The first tick backfills, later
        # ticks read forward from the last seen id.
        try:
            if self._timeline_last_id is None:
                entries = list(reversed(
                    self._redis.xrevrange(CONTROLLER_SIGNAL_LOG_KEY, count=_TIMELINE_BACKFILL)
                ))
            else:
                entries = self._redis.xrange(
                    CONTROLLER_SIGNAL_LOG_KEY,
                    min=f"({self._timeline_last_id}",
                    count=_TIMELINE_INCREMENT,
                )
        except Exception:  # noqa: BLE001
            return False
        for entry_id, fields in entries or []:
            self._timeline_last_id = _text(entry_id)
            row = decode_signal_row(fields)
            model = row.get("model")
            if model in self._timeline and row.get("ts_ms") is not None:
                ring = self._timeline[model]
                ring.append(row)
                if len(ring) > _TIMELINE_RING:
                    del ring[: len(ring) - _TIMELINE_RING]
        return bool(entries)
```

`sample_once` reader 元组追加 `("timeline", self._read_timeline)`。消费者方法：

```python
    def timeline(self, model: str, since_ms: int = 0) -> list[dict]:
        return [p for p in self._timeline.get(model, []) if (p.get("ts_ms") or 0) >= since_ms]
```

`app.py` 在既有 `/api/signal/history` 之后新增：

```python
    @app.get("/api/signal/timeline")
    def signal_timeline(model: str, since_ms: int = 0) -> dict[str, Any]:
        if model not in model_names:
            raise HTTPException(status_code=404, detail="unknown model")
        return {"model": model, "points": sampler.timeline(model, since_ms=since_ms)}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tre/ui && git commit -m "feat(ui): serve a bounded signal_log timeline"
```

---

### Task 3: 手动 audit 端点

**Files:**
- Modify: `ui/tre_ui/app.py`
- Test: `ui/tests/test_ui_app.py`

**Interfaces:**
- Produces: `POST /api/ops/audit` → `{"ran_at_ms": int, "result": dict}`；`GET /api/ops/audit` → 上次结果，未跑过时为 `{"ran_at_ms": None, "result": None}`

- [ ] **Step 1: 写失败测试**

测试用的 SM fake 需为 `/v2/audit` 返回 `{"healthy": True, "version": 1, "issues": []}`。

```python
def test_audit_runs_only_when_explicitly_posted(client, sm_client):
    assert client.get("/api/ops/audit").json()["ran_at_ms"] is None
    assert not any("audit" in path for path in sm_client.calls)

    posted = client.post("/api/ops/audit")

    assert posted.status_code == 200
    assert posted.json()["result"]["healthy"] is True
    assert client.get("/api/ops/audit").json()["ran_at_ms"] is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests/test_ui_app.py -x -q`
Expected: FAIL —— 404 Not Found

- [ ] **Step 3: 实现**

`app.py` 在 `create_ui_app` 内、其他端点之前加缓存与两个端点（文件需 `import time`）：

```python
    audit_cache: dict[str, Any] = {"ran_at_ms": None, "result": None}

    @app.get("/api/ops/audit")
    def get_audit() -> dict[str, Any]:
        return dict(audit_cache)

    @app.post("/api/ops/audit")
    def run_audit() -> dict[str, Any]:
        # Deliberately manual: /v2/audit lists k8s pods and HTTP-probes every
        # vLLM pod, so it must never sit on a timer.
        try:
            result = service_manager_client.request("GET", "/v2/audit")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        audit_cache["ran_at_ms"] = int(time.time() * 1000)
        audit_cache["result"] = result
        return dict(audit_cache)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tre/ui && git commit -m "feat(ui): add button-triggered fleet audit"
```

---

### Task 4: 静态资源改为 StaticFiles 挂载

**Files:**
- Modify: `ui/tre_ui/app.py`, `ui/tre_ui/static/index.html`
- Create: `ui/tre_ui/static/js/main.js`
- Test: `ui/tests/test_ui_app.py`

**Interfaces:**
- Produces: `/static/<path>` 提供静态文件；`/` 仍返回 index.html

- [ ] **Step 1: 写失败测试**

```python
def test_static_assets_are_served_from_the_static_mount(client):
    assert client.get("/static/js/main.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
```

同时删除既有断言 `/app.js` 与 `/style.css` 旧路由的测试。

- [ ] **Step 2: 跑测试确认失败**

Expected: FAIL —— 404（`/static` 未挂载，且 `js/main.js` 不存在）

- [ ] **Step 3: 实现**

先建占位文件 `ui/tre_ui/static/js/main.js`，内容 `// entry point`（Task 5-8 填充）。

`app.py` 删除 `/app.js` 与 `/style.css` 两个路由，改为：

```python
from pathlib import Path

from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"
```

在 `app = FastAPI(...)` 之后：

```python
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
```

`index.html` 引用改为 `/static/style.css` 与 `<script type="module" src="/static/js/main.js"></script>`。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tre/ui && git commit -m "refactor(ui): serve the console from a static mount"
```

---

### Task 5: 前端骨架与 core.js

**Files:**
- Rewrite: `ui/tre_ui/static/index.html`, `ui/tre_ui/static/style.css`
- Create: `ui/tre_ui/static/js/core.js`；填充 `ui/tre_ui/static/js/main.js`

**Interfaces:**
- Produces: `core.js` 导出 `S`（全局状态）、`$`、`el`、`fmt`、`fmtInt`、`ageText`、`clockText`、`toast`、`api`、`zText`、`freshness`、`registerView(id, view)`、`renderAll()`、`switchView(id)`、`openStream()`

- [ ] **Step 1: 写 index.html 骨架**

四个 view 容器 `#view-overview`、`#view-signals`、`#view-fleet`、`#view-ops`；顶栏五个 pill：mode、supervisor、audit、gpu-truth、conn；左栏四项导航，快捷键 1-4。

- [ ] **Step 2: 写 core.js**

从旧 `app.js` 移植并保留：`$`、`el`、`fmt`、`fmtInt`、`ageText`、`clockText`、`toast`、`api`、SSE 的 `openStream`/`setConn`。新增两个助手：

```js
// Idle (no load) and a missing signal are different conditions; the old
// console rendered both as an em dash, which reads as a broken system.
export function zText(z, state) {
  if (z == null) return state === 'idle' ? 'idle · 无负载' : '信号缺失';
  return Number(z).toFixed(3);
}

export function freshness(ageMs, staleMs = 15000) {
  if (ageMs == null) return 'unknown';
  return ageMs > staleMs ? 'stale' : 'fresh';
}
```

view 注册表：

```js
const VIEWS = new Map();
export function registerView(id, view) { VIEWS.set(id, view); }
export function renderAll() {
  VIEWS.forEach((view, id) => {
    try { view.render(); } catch (err) { console.error(id, err); }
  });
}
```

`openStream()` 的 `onmessage` 内改为 `S.snap = JSON.parse(ev.data); renderAll(); setConn(true);`

- [ ] **Step 3: main.js 装配**

import 四个 view 模块（副作用注册）后调用 `openStream()`。

- [ ] **Step 4: 验证**

Run: `PYTHONPATH=common:deploy:ui:controller:service-manager:replayer python3 -m pytest ui/tests -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tre/ui && git commit -m "feat(ui): four-view console shell"
```

---

### Task 6: 时序图组件 chart.js

**Files:**
- Create: `ui/tre_ui/static/js/chart.js`

**Interfaces:**
- Produces: `makeScale(lo, hi, px0, px1) -> (v) => number`、`niceBounds(values, extra) -> {lo, hi}`、`createChart(container, options) -> {update(points, taus), showCursorAt(tsMs), onHover}`、`linkCrosshair(charts)`

- [ ] **Step 1: 纯标度函数**

```js
// Pure, DOM-free helpers kept at the top of the file so the scale maths can be
// reasoned about (and later unit-tested) without a browser.
export function makeScale(lo, hi, px0, px1) {
  const span = (hi - lo) || 1;
  return (v) => px0 + ((v - lo) / span) * (px1 - px0);
}

export function niceBounds(values, extra = []) {
  const all = values.concat(extra).filter((v) => v != null && !Number.isNaN(v));
  if (!all.length) return { lo: 0, hi: 1 };
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.12;
  return { lo: lo - pad, hi: hi + pad };
}
```

- [ ] **Step 2: createChart 渲染**

绘图区高 300px、宽度撑满容器，`<svg viewBox="0 0 1000 340" preserveAspectRatio="none">`。要素，按 z 序绘制：
1. τ_high / τ_crit 阈值带：半透明 `<rect>` + 虚线 `<line>`，并在右侧标注 `τ_high` / `τ_crit`
2. Z_m 主曲线：`<path>`，`z_m` 为 null 处断开（生成新的 `M` 而非 `L`）
3. queue_len 次轴曲线（右侧刻度，颜色区分）
4. `replicas_awake` 与 `replicas_target` 阶梯线：用 `H`/`V` 指令；两条分叉即"想扩但扩不上"
5. action 竖线：对 `points.filter(p => p.action && p.action !== 'none')` 每点画 `<line>` + `<title>` 显示 action 名与时刻
6. 十字准星层：容器 `mousemove` 将像素 x 换算为时间戳后调用 `this.onHover(tsMs)`

`decode_tps`/`prefill_tps` 由 `options.series` 控制，默认不绘制。

- [ ] **Step 3: linkCrosshair**

```js
export function linkCrosshair(charts) {
  charts.forEach((chart) => {
    chart.onHover = (tsMs) => charts.forEach((other) => other.showCursorAt(tsMs));
  });
}
```

- [ ] **Step 4: 验证**

浏览器打开 console 的 Signals 页，确认三张图渲染、悬停任一张时三张同时出现游标。

- [ ] **Step 5: 提交**

```bash
git add tre/ui && git commit -m "feat(ui): full-width linked time-series chart"
```

---

### Task 7: Overview 与 Signals 页

**Files:**
- Create: `ui/tre_ui/static/js/view_overview.js`, `ui/tre_ui/static/js/view_signals.js`

- [ ] **Step 1: view_overview.js**

渲染四块：
1. 状态灯行：controller mode（observe 时显示是人工还是 supervisor 自动，取自 `fleet.supervisor.last_recovery_operation_id` 是否非空）、supervisor running/drift_observations、audit 上次结果与时间（`GET /api/ops/audit`）、每节点 gpu truth 新鲜度、SSE 连接
2. desired/observed/mismatch 计数摘要，mismatch 非空时逐条列出
3. 每模型一行：tier、`zText(z_m, state)`、awake/target 副本、routable_pods
4. 最近 5 条事件（取 `events_head`）

- [ ] **Step 2: view_signals.js**

为每个模型建一个 chart 容器；`fetch('/api/signal/timeline?model=' + name + '&since_ms=' + since)` 拉取并 `chart.update(points, taus)`；窗口按钮 5m / 15m / 1h（默认 15m）；暂停按钮停止刷新；`linkCrosshair(charts)` 联动；decode/prefill 序列开关默认关闭。τ 值取自 `/api/meta` 的每模型 `trs.tau_high` / `trs.tau_crit`。

- [ ] **Step 3: 验证**

浏览器打开 `http://192.168.223.76:30812`，确认两页渲染且数值与 `/api/snapshot` 一致；空载模型显示「idle · 无负载」而非 `—`。

- [ ] **Step 4: 提交**

```bash
git add tre/ui && git commit -m "feat(ui): overview and signals views"
```

---

### Task 8: Fleet 与 Ops 页

**Files:**
- Create: `ui/tre_ui/static/js/view_fleet.js`, `ui/tre_ui/static/js/view_ops.js`

- [ ] **Step 1: view_fleet.js**

节点 × GPU 矩阵。每格显示：驻留模型列表（awake 高亮）、显存条 `used_mib/total_mib` 带百分比、lease 的 `phase`/`fencing_token`/`owner`（取自 `leases.gpus["<node>/<gpu_id>"]`）、pod 名。与 `fleet.state.observed` 不一致处加 `.mismatch` 类标红。

- [ ] **Step 2: view_ops.js**

四块：
1. operation journal 时间线：id、类型、状态、耗时、失败原因（失败标红），取自 `operations.items`
2. supervisor 面板：running、drift_observations、last_drift、last_recovery_operation_id
3. audit 区：按钮调 `POST /api/ops/audit`，展示 issues 列表与上次运行时间，按钮旁注明"会探测全部模型 pod，勿在实验负载期间频繁点击"
4. 控制区：mode 切换（带确认对话框）、reconcile、defrag、params 编辑，并显示 `params_hash` vs `applied_hash` 与 `pending_restart`

- [ ] **Step 3: 验证**

点击 audit 按钮确认返回结果并渲染；确认 mode 切换仍工作且需要确认；确认 params 编辑保存后 `pending_restart` 正确显示。

- [ ] **Step 4: 提交**

```bash
git add tre/ui && git commit -m "feat(ui): fleet and ops views"
```

---

### Task 9: 构建、部署与在线验证

**Files:**
- Modify: `deploy/overlays/tre-v2/ui.yaml`, `deploy/tests/test_kustomize_overlays.py`

- [ ] **Step 1: 全量测试**

Run: `cd /data/nfs_shared_data/xxy/aibrix/tre && make check`
Expected: PASS，数量 ≥ 580 加上新增测试数

- [ ] **Step 2: 构建镜像**

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre
SHA=$(git rev-parse --short HEAD)
nohup docker build -f ui/Dockerfile -t tre-v2-ui:20260725-${SHA} . > /tmp/ui_build.log 2>&1 &
```
build context 必须是 `tre/`。轮询 `docker images | grep tre-v2-ui`。

- [ ] **Step 3: 双处改 tag**

`deploy/overlays/tre-v2/ui.yaml` 与守卫测试 `deploy/tests/test_kustomize_overlays.py` 的 ui 镜像 tag 同时改为新值，再跑 `make check` 确认守卫测试通过。

- [ ] **Step 4: 部署**

```bash
kubectl -n tre-v2 apply -f tre/deploy/overlays/tre-v2/ui.yaml
kubectl -n tre-v2 rollout status deploy/tre-v2-ui --timeout=180s
```

- [ ] **Step 5: 在线验证**

- `curl -s http://192.168.223.76:30812/api/snapshot | python3 -c "import json,sys; print(sorted(json.load(sys.stdin)))"` 含 fleet / operations / leases
- `curl -s 'http://192.168.223.76:30812/api/signal/timeline?model=dsqwen-7b' | head -c 300` 返回点位
- **性能守卫在线确认**：`kubectl -n tre-v2 logs deploy/tre-v2-service-manager --since=5m | grep -c audit` 为 0
- 浏览器逐页检查四个视图，确认三图联动与动作标注
- 确认 controller mode 未被改动、SM audit 仍 healthy、20/20 pod Running、restarts=0

- [ ] **Step 6: 提交并更新 HANDOFF**

```bash
git add -A && git commit -m "deploy(tre): roll console <sha>"
```
