# 任务 4B：三线程批量检测流水线

- 完成日期：2026-11-18
- 对应 SRS：FR-009、FR-010、NFR-002、NFR-003；AT-007、AT-008
- 入口：`edge_ai --model ... --image-dir ... [--queue-capacity N]`
- 实现：`include/threaded_pipeline.hpp`、`src/threaded_pipeline.cpp`

## 1. 结构

```text
ImageSource
    │  reader thread
    ▼
input BoundedQueue<InputItem>   (单生产者 / 单消费者)
    │  inference thread
    ▼
result BoundedQueue<FrameResult> (单生产者 / 单消费者)
    │  result thread
    ▼
按输入顺序输出 + 唯一汇总
```

- 读取线程调用 `ImageSource::next()`，把序号、路径、时间戳、图像或错误记录
  送入第一个队列。坏图仍占用一个位置，保证最终报告保留其输入序号。
- 推理线程从第一个队列取项；坏图跳过预处理/推理，生成 `ERROR` 且
  `quality_available=false` 的结果；好图完成预处理、推理、质量判定后送入
  第二个队列。
- 结果线程是唯一修改 `BatchResult` 和 total/ok/ng/errors 的线程；回调也只在
  该线程按输入顺序调用。
- 两个队列均为 `BoundedQueue`；容量由 `--queue-capacity` 同时设置（默认 4，
  必须 >0）。满载时对应生产者阻塞，形成背压。负值、0 和超出 `size_t` 范围
  的值在 CLI 解析阶段返回 2。

## 2. 正常排空与错误策略

- 图片源返回 `End`：读取线程关闭第一个队列；推理线程排空后关闭第二个队列；
  结果线程排空后退出。
- 单帧坏图/解码失败用 `SourceStatus::Error` 传递，作为普通 `ERROR` 结果继续；
  后续好图仍会处理。
- 逃逸出 `ImageSource::next()` 的异常、或结果回调返回 `false`/抛异常，视为
  致命错误：关闭两个队列，唤醒阻塞的读取/推理/结果线程，join 全部线程，
  `run()` 返回 `false` 和诊断信息。
- 创建第 2/3 个线程时若抛出 `std::system_error`，不会让仍可 join 的线程在
  栈展开中析构；代码关闭两个队列、join 已启动线程，再返回带阶段名称的
  启动失败错误。
- 结果回调由结果线程调用；正常 CLI 回调只打印每帧状态并返回 `true`。

## 3. 验收场景实测

### 场景 1：正常排空（六图 + 真实模型）

命令：

```bash
./build/bin/edge_ai \
  --model "$MODEL_DIR/neu_det_yolov8n_best.onnx" \
  --image-dir <six-image-dir> \
  --queue-capacity 1 --conf 0.25 --iou 0.45 \
  --json-out build/threaded-batch.json
```

实际输出（线程标识为本次运行值）：

```text
threads: reader=125537458542208 inference=125537450149504 result=125537441756800
queues: input capacity=1 max_depth=1 producer_waits=4 consumer_waits=1;
        result capacity=1 max_depth=1 producer_waits=0 consumer_waits=7
summary: total=6 ok=0 ng=6 errors=0 skipped_unsupported=0
```

顺序与检测对照工具 `tools/compare_batch_reference.py`：

```text
batch comparison: frames=6 ok=0 ng=6 errors=0
max_conf_diff=1.192e-06 max_bbox_diff=3.436e-05
```

帧按文件名顺序处理，序号 0..5；检测数量/类别/置信度/bbox 与 2A/3 参考
一致，全部为 NG。

### 场景 2：容量 1 受控积压

`threaded_full_backpressure` 使用两个容量 1 队列。结果回调在收到第一条结果
后暂停；测试通过队列的进入/退出等待钩子和实时深度确认**两个队列当前都满、
两个生产者当前都在等待**，再放行回调；只有旧等待标志但队列已腾出位置时
门槛不会打开：

```text
backpressure: threads(reader=..., inference=..., result=...)
queues(input cap=1 max=1 waits=4, result cap=1 max=1 waits=1)
counts(total=5 ok=5 ng=0 errors=0)
```

- 两个队列最大深度均为 1，不超过容量；
- 两个队列都记录到生产者等待；
- 结果回调按序号收到 0..4，每个序号恰好一次，无丢失/重复/乱序。

### 场景 3：下游致命错误唤醒退出

`threaded_fatal_wakes` 先暂停结果回调，并通过两个队列的实时等待状态确认
读取和推理线程**此刻同时**阻塞，再注入失败（返回 `false`）：

```text
fatal exit: threads(reader=..., inference=..., result=...)
queues(input cap=1 max=1, result cap=1 max=1)
error="result callback failed: injected result failure"
```

`run()` 关闭队列、唤醒所有等待者、join 三个线程后返回 `false`；测试设置
`TIMEOUT 10`，不会因死锁挂住 CTest。

## 4. 测试

```bash
ctest --test-dir build --output-on-failure
```

任务 4B 新增：

| 测试 | 内容 |
| --- | --- |
| `threaded_normal_drain` | 三线程正常排空、顺序、线程 ID、队列深度 |
| `threaded_full_backpressure` | 两队列容量 1、受控暂停、满载等待、序号唯一 |
| `threaded_fatal_wakes` | 等两个生产者都进入等待后注入下游失败，验证唤醒、join、错误返回 |
| `threaded_start_failure_inference` | 第 2 个线程启动失败：关闭队列、join reader、返回错误 |
| `threaded_start_failure_result` | 第 3 个线程启动失败：关闭队列、join reader/inference、返回错误 |
| `threaded_wait_gate_stale_flag` | 队列腾出位置但旧等待标志未清除时门槛不通过 |
| `threaded_external_stop_drains` | `max_input_frames=0` 持续运行，外部停止后排空已接收项 |
| `threaded_batch_reference` | 六图真实模型批量顺序/检测结果与参考一致（可选） |

本次配置总计 68/68 通过；不提供真实模型/六图时 65 个测试通过。

## 5. 限制

- 每个队列仍是单生产者/单消费者；多采集源/多推理线程未验证。
- 队列容量由单一 `--queue-capacity` 同时设置；如需分别调优，可后续扩展。
- 回调在主流程中同步执行；生产代码中的回调必须及时返回，测试中的暂停由
  CTest 超时保护。
- 单图模式仍走同步路径，不使用三线程流水线。
- 源若在下一次调用中长期阻塞，致命停止无法中断 `source.next()`；当前批量
  源是有限文件枚举。4A 的关闭语义只唤醒队列等待者。
- 模型/数据集许可、彩色输入、精度等限制与之前任务相同。
