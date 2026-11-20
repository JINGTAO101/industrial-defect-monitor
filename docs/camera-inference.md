# 任务 5B：摄像头三线程检测与有限重连

- 完成日期：2026-11-19
- 对应 SRS：FR-002、FR-003、FR-009、FR-010、FR-021；NFR-002
- 入口：`edge_ai --model model.onnx --camera /dev/videoN --max-frames N`
- 实现：`ReconnectingCameraSource` + 现有 `ThreadedPipeline`

## 1. 结构

```text
CameraSource (V4L2)
    │  ReconnectingCameraSource（坏帧/失联分类、有限重连、序号保持）
    ▼
读取线程 → 输入队列 → 推理线程 → 结果队列 → 结果线程
```

- 复用任务 4B 的读取/推理/结果三线程和两个有界队列。
- 摄像头结果逐帧由结果线程回调输出；`retain_frames=false`，不把每帧追加到
  `BatchResult.frames`，长期运行内存不随帧数增长，只维护计数。
- `max_input_frames` 限制本次运行接收的输入项数（坏帧也占一项），到达后
  读取线程关闭第一个队列，推理/结果线程排空后退出。
- Ctrl+C：SIGINT 设置外部停止标志；读取线程停止接收并关闭队列，已接收的帧
  继续经过推理和结果线程排空；线程全部 join 后正常返回。

## 2. 坏帧与设备失联

- **坏帧**（`V4L2_BUF_FLAG_ERROR`、空帧、超长帧）：记录为 `ERROR`，
  `quality` 不可用，不做 OK/NG 判定；不触发重连。
- **设备失联/读取超时**：关闭当前 `CameraSource`，按
  `--camera-reconnect-attempts`（默认 3）和
  `--camera-reconnect-interval-ms`（默认 1000）重开设备。
- 工厂在打开失败时会返回**非空对象 + 错误字符串**，因此包装器以
  `open_error.empty()` 作为成功条件，而不只是检查指针是否为空。
- 重试预算是**一次失联事件**的累计值：每次重开尝试（包括“打开成功但一直
  读不到帧”）都消耗一次，只有收到第一帧有效画面才判定恢复并清零。这样
  “每次都能打开、但读帧始终超时”不会无限重连。
- 失联、每次重开成功/失败、恢复和预算耗尽都输出
  `camera reconnect: ...` 日志。
- 重连成功后帧序号**不重置**；包装器维护全局递增序号。
- 重试预算耗尽：返回 `Fatal`，流水线关闭两个队列、唤醒并 join 所有线程，
  CLI 输出 `camera pipeline failed` 并以 1 退出。
- 重连等待按小片（≤50ms）检查外部停止标志；读取中的 `poll` 也按 50ms
  分片，因此设备无帧时 Ctrl+C 可在约 50ms 内结束等待，而不是等满读超时。
- 工厂调用返回后、判定重开结果前再检查一次停止标志：若 Ctrl+C 在打开过程
  中到达，即使该次打开随后失败，也按正常停止返回 End 并排空已入队帧，不会
  误报重连耗尽。

## 3. CLI

```bash
./build/bin/edge_ai \
  --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --camera /dev/video0 \
  --max-frames 30 \
  --queue-capacity 1 \
  --camera-format YUYV --camera-width 640 --camera-height 480 --camera-fps 30 \
  --camera-reconnect-attempts 3 --camera-reconnect-interval-ms 1000
```

- `--camera` 与 `--model` 组合启用摄像头检测；`--capture-only` 仍是无模型的
  5A 采集模式。
- 摄像头检测**省略 `--max-frames` 时连续运行**，直到 Ctrl+C；纯采集模式
  省略时仍默认 30 帧。
- `--queue-capacity` 同时设置两个队列，本验收使用 1。
- 摄像头模式暂不支持 `--json-out`/`--dump-input`。

## 4. vivid 30 帧检测证据

完整日志：`docs/camera-inference-run.txt`。

```text
$ edge_ai --model ... --camera /dev/video0 --max-frames 30 --queue-capacity 1
...
threads: reader=138186668463744 inference=138186660071040 result=138186651678336
queues: input capacity=1 max_depth=1 producer_waits=28;
        result capacity=1 max_depth=1 producer_waits=0
summary: total=30 ok=0 ng=30 errors=0 reconnect_count=0 stopped=0
```

- 30 帧完成，序号 0..29；
- 两个队列容量均为 1，最大深度均为 1，未越界；
- OK/NG/错误计数由结果线程独立维护，`BatchResult.frames` 不增长；
- vivid 虚拟画面被真实模型判为 NG（仅验证流水线，不代表精度）。

### Ctrl+C 排空

```text
$ timeout --preserve-status -s INT 2s edge_ai --model ... --camera /dev/video0 \
    --max-frames 1000 --queue-capacity 1
...
summary: total=29 ok=0 ng=29 errors=0 reconnect_count=0 stopped=1
camera stopped by SIGINT; accepted frames drained
# exit code: 0
```

已接收的 29 帧全部完成推理并输出，三个线程回收后正常退出。

### 连续模式（省略 `--max-frames`）

```text
$ timeout --preserve-status -s INT 6s edge_ai --model ... --camera /dev/video0 \
    --queue-capacity 1 --camera-timeout-ms 3000
...
camera reconnect: attempts=3 interval_ms=1000
frame limit: continuous (Ctrl+C to stop)
...
summary: total=69 ok=0 ng=69 errors=0 reconnect_count=0 stopped=1
camera stopped by SIGINT; accepted frames drained
# wall=6.25s, exit code 0
```

6 秒内处理 69 帧（明显超过旧的固定 30 帧），Ctrl+C 后读取线程停止接收，
已接收帧排空，三个线程 join 后退出。

### 坏帧

vivid 注入 `V4L2_BUF_FLAG_ERROR` 后运行 5 帧：

```text
[ERROR] index=0 source=/dev/video0: camera frame marked with V4L2_BUF_FLAG_ERROR
[NG] index=1 ... [NG] index=4 ...
summary: total=5 ok=0 ng=4 errors=1 reconnect_count=0 stopped=0
```

坏帧只计为错误、没有 OK/NG，后续帧继续检测。

## 5. 模拟设备重连测试

| 测试 | 内容 |
| --- | --- |
| `unit_camera_reconnect_success` | 首设备帧后失联，第二设备恢复；帧序号 0、1 连续，reconnect_count=1 |
| `unit_camera_reconnect_exhausted` | 重开持续失败，按配置次数耗尽后返回 Fatal 错误 |
| `unit_camera_reconnect_open_error` | 工厂返回“非空对象 + 打开错误”时仍计为失败，耗尽后 Fatal |
| `unit_camera_reconnect_no_frame_budget` | 每次重开成功但读帧始终超时；预算 3 时只重开 3 次即耗尽 |
| `unit_camera_reconnect_stop` | 重连等待期间设置停止标志，返回 End 且不再尝试重开 |
| `unit_camera_reconnect_stop_during_open` | 最后一次重开在返回错误前收到停止；按正常停止返回 End 且错误为空，不报耗尽 |
| `unit_camera_bad_frame_sequence` | 坏帧计为 DataError，不做判定；后续好帧序号继续递增 |
| `unit_camera_read_stop` | 3 秒读超时中设置停止，poll 分片使等待在 1 秒内结束 |
| `unit_camera_source_external_stop` | 停止标志传递到后端，阻塞读返回 End 而非错误 |
| `threaded_external_stop_drains` | `max_input_frames=0` 时持续运行，停止后排空已接收帧 |
| `cli_signal` | 图片批量模式收到 SIGINT 以 130 终止，不再被摄像头处理器吞掉 |

## 6. 验证

```bash
ctest --test-dir build --output-on-failure
```

本次完整套件 **73/73 通过**；无模型/图片/摄像头时 70 个测试通过。

## 7. 限制与待确认

1. **AT-009 仍待真实 USB 摄像头验证**：VMware 无摄像头直通，vivid 只用于
   实现验证。
2. 队列满时等待只证明应用层背压，**不能证明 V4L2 驱动没有丢帧**；驱动丢帧
   需要实机统计/驱动计数。
3. 重连只覆盖运行中的设备失联；首次打开失败直接报错退出，不重试。
4. `--max-frames` 统计输入项，坏帧也占一项，因此 OK/NG/错误之和等于总项数；
   省略时为连续模式，只能由 Ctrl+C 或致命源错误结束。
5. 重连和故障测试均为脚本化模拟；真实设备上的重连、MJPG、占用等仍需实机。
6. SIGINT 仅由摄像头模式接管；图片/单图模式保持系统默认的中断行为。
7. 可视化留到 5C；当前只打印每帧状态与汇总。
