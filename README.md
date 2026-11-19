# Edge AI 工业缺陷检测系统

面向边缘设备的工业表面缺陷检测系统。当前已完成：

- **任务 1 / 2A**：C++17 工程骨架、YOLOv8n-NEU-DET 模型来源/许可/接口核验、
  Python 参考推理；
- **任务 2B**：基于 ONNX Runtime C++ API 和 CPU ExecutionProvider 的单图推理，
  包括模型契约校验、letterbox 预处理、类别感知 NMS、坐标还原、失败路径测试，
  以及与 2A 参考结果的逐项对照；
- **任务 3**：统一 `Frame`/`ImageSource` 图片输入抽象，支持单图和目录批量，
  实现可配置阈值与 OK/NG 判定，覆盖坏文件继续策略和输入/阈值错误测试；
- **任务 4A**：可关闭的线程安全有界队列，支持阻塞生产/消费、满载等待、
  关闭排空和并发无丢失验证；
- **任务 4B**：`--image-dir` 批量检测接入读取/推理/结果三线程，两个容量 1
  队列形成背压；覆盖六图正常排空、受控积压、下游致命错误唤醒退出；
- **任务 5A**：V4L2 `CameraSource` 单摄像头采集，支持 YUYV/MJPG 转 BGR、
  序号/时间戳、读取超时和 `--capture-only --max-frames`；真实 USB 直通待实机；
- **任务 5B**：摄像头接入三线程检测流水线，支持逐帧结果、内存有界、Ctrl+C
  排空、坏帧/设备失联区分和有限重连；真实摄像头 AT-009 仍待实机；
- **任务 5C**：单图和摄像头共用检测框/类别/置信度/OK-NG-ERROR 标注；支持
  单图无桌面保存、单图/摄像头窗口预览，预览缓存容量 1 且慢窗口只丢预览帧；
  关闭窗口沿用 5B 停止排空。

SQLite、MQTT 不在当前阶段范围内。

## 已验证环境

以下为实际执行验证的环境（2026-11-18）：

| 项目 | 实际值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.5 LTS (Noble Numbat) |
| 架构 | x86_64 |
| CPU | AMD Ryzen 5 6600U（VMware 虚拟机，4 vCPU） |
| 内存 | 5.7 GiB（另有 4.0 GiB swap） |
| GCC / G++ | 13.3.0 |
| CMake / Ninja | 3.28.3 / 1.11.1 |
| GDB / Git | 15.1 / 2.43.0 |
| OpenCV | 4.6.0（`libopencv-dev`） |
| OpenSSL | 3.0.13（`libssl-dev`） |
| ONNX Runtime C++ | 1.32.0 CPU（`libonnxruntime.so.1.32.0`） |
| Python（参考推理） | 3.12.3，ONNX Runtime 1.32.0，OpenCV 4.10.0 |

## 依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake gdb git ninja-build pkg-config \
  libopencv-dev libssl-dev
```

ONNX Runtime C++ CPU 包（必需）：

```bash
ORT_DIR="$HOME/edge-ai-models/onnxruntime-1.32.0"
mkdir -p "$ORT_DIR" && cd "$ORT_DIR"
curl -fL \
  https://github.com/microsoft/onnxruntime/releases/download/v1.32.0/onnxruntime-linux-x64-1.32.0.tgz \
  -o onnxruntime-linux-x64-1.32.0.tgz
sha256sum onnxruntime-linux-x64-1.32.0.tgz
# bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503
tar -xzf onnxruntime-linux-x64-1.32.0.tgz
```

本工程通过 `-DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.32.0` 查找
头文件和共享库；不使用该 tarball 自带的 CMake config（其路径布局与实际包
不一致）。

模型和六张测试图片不提交到仓库。来源、SHA256、许可状态和下载/校验脚本见
[docs/model-verification.md](docs/model-verification.md) 与
`tools/reproduce_reference_verification.sh`。

## 构建

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$HOME/edge-ai-models/onnxruntime-1.32.0/onnxruntime-linux-x64-1.32.0"
cmake --build build
```

启用真实模型集成测试时，额外传入（均为可选）：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$HOME/edge-ai-models/onnxruntime-1.32.0/onnxruntime-linux-x64-1.32.0" \
  -DEDGE_AI_MODEL_PATH="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  -DEDGE_AI_TEST_IMAGES_DIR="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/test-images" \
  -DEDGE_AI_PYTHON="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/venv/bin/python"
```

生成的可执行文件为 `build/bin/edge_ai`，测试程序为
`build/bin/edge_ai_tests`。

## 运行

```bash
# 帮助 / 版本
./build/bin/edge_ai --help
./build/bin/edge_ai --version

# 无参数：打印启动横幅
./build/bin/edge_ai

# 单图推理
./build/bin/edge_ai \
  --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --image "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/test-images/crazing_101.jpg" \
  --conf 0.25 --iou 0.45 \
  --json-out build/verification/crazing_101.cpp.json

# 目录批量推理（三线程流水线；queue-capacity 同时设置两个队列）
./build/bin/edge_ai \
  --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --image-dir "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/test-images" \
  --queue-capacity 1 --conf 0.25 --iou 0.45 \
  --json-out build/verification/batch.json

# V4L2 纯采集（无需模型）
./build/bin/edge_ai \
  --camera /dev/video0 --capture-only --max-frames 30 \
  --camera-format YUYV --camera-width 640 --camera-height 480 --camera-fps 30

# 摄像头 + 模型三线程检测（省略 --max-frames 为连续模式，Ctrl+C 停止）
./build/bin/edge_ai \
  --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --camera /dev/video0 --queue-capacity 1 \
  --camera-reconnect-attempts 3 --camera-reconnect-interval-ms 1000

# 单图标注：无桌面保存 / 窗口预览
./build/bin/edge_ai --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --image part.jpg --save-annotated build/part-annotated.png
./build/bin/edge_ai --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --image part.jpg --display

# 摄像头窗口预览（关闭窗口按 5B 流程停止并排空）
./build/bin/edge_ai \
  --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --camera /dev/video0 --display --queue-capacity 1

# 非法参数：stderr 报错，退出码 2
./build/bin/edge_ai --bogus
```

`--image` 与 `--image-dir` 互斥；`--dump-input` 仅支持单图。支持
`jpg/jpeg/png/bmp/tif/tiff`。批量目录中的其他扩展名文件会被跳过并在摘要中
计数；损坏图片会记为 `[ERROR]` 后继续处理，最终退出码为 1。若
`ImageSource::next()` 抛出无法确认已推进的异常，则视为致命错误立即停止
（不重试）。判定规则为：
至少一个检测的置信度严格大于 `--conf` 时 NG，否则 OK。非法 `--conf`/`--iou`
（0、负数、>1、NaN、Inf）明确返回 2，不会静默使用默认值。

主要参数：`--model`、`--image`、`--conf`（默认 0.25）、`--iou`（默认 0.45）、
`--json-out`、`--dump-input`。参数会先完整校验；`--help --bogus` 等组合仍会
返回非零。缺值的选项不会吞掉后续选项；以 `-` 开头的值必须写成
`--option=value`（例如 `--model=-model.onnx`）。模型/图片/推理失败返回 1。

`--json-out` 和 `--dump-input` 在写入前会与模型、输入图片以及彼此做路径
冲突检查，覆盖相对路径、`..`、符号链接（包括悬空符号链接目标及其链）和
硬链接；冲突时返回 2，且不写入任何文件。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

不提供真实模型路径时运行 68 个测试（单元测试 + CLI 失败路径 + 空检测 +
路径冲突 + 批量输入 + 队列 + 三线程流水线 + 摄像头错误/超时/坏帧/色彩/
重连/停止 + 标注/预览/无桌面保存）；提供 `EDGE_AI_MODEL_PATH`、
`EDGE_AI_TEST_IMAGES_DIR` 和 `EDGE_AI_PYTHON` 时额外运行六图对照、对照
守卫和线程化批量对照测试，共 71 个。

`cli_inference_failures` 覆盖模型缺失/损坏、输入输出契约错误、缺失/不支持/
损坏图片。单元测试覆盖非方形 letterbox、空候选、同类/跨类 NMS、坐标还原、
非有限输出、类别分数越界（>1/<0）与边界 0/1、输出尺寸错误、未知类别
`Unknown`+anomaly、契约校验和类别 metadata 解析。

任务 3 新增测试覆盖：

- 无缺陷 → OK、单缺陷 → NG、低于阈值 → OK、多缺陷 → NG；
- 阈值边界（严格 `>`）和非法阈值（0/负数/>1/NaN/Inf）；
- Frame 读取时间戳；失败帧 `quality:null` 且 `quality_available=false`；
- 正常 `SourceStatus::Error` 后继续；持续抛异常的源在第一次异常后致命停止
  （不重试、不会无限循环）；
- 单图正常/缺失/不支持/损坏；目录批量正常、坏文件继续、扩展名跳过、缺失/空目录；
- 批量全 OK、坏文件继续并汇总、批量 JSON、批量输出路径冲突、`--dump-input`
  与 `--image-dir` 组合拒绝；
- 有真实模型时六张图批量输出 `ok=0 ng=6 errors=0`。

任务 4A 新增 6 个独立队列测试：

- 容量 0 拒绝；
- 容量 2 满载阻塞、取走 A 后 C 入队、剩余顺序 B/C；
- 空队列消费者等待并被生产者唤醒；
- close 幂等、唤醒等待者、拒绝新入队、排空后返回 Closed；
- 递增序号 + 慢消费者的 2000 项并发无丢失/重复；
- 所有队列测试通过 CTest `TIMEOUT 10` 防止死锁挂起。

任务 4B 新增 4 个测试：

- `threaded_normal_drain`：读取/推理/结果三线程正常排空，序号顺序与队列深度；
- `threaded_full_backpressure`：两个容量 1 队列、受控暂停、满载等待、序号唯一；
- `threaded_fatal_wakes`：用进入/退出等待钩子确认两个生产者当前同时等待后，
  再注入失败，验证唤醒、join 和错误返回；
- `threaded_start_failure_inference` / `threaded_start_failure_result`：
  注入第 2/3 个线程启动失败，验证已启动线程被回收且进程不终止；
- `threaded_wait_gate_stale_flag`：输入队列已腾出位置但旧等待标志尚未清除时，
  门槛必须保持关闭；
- `threaded_batch_reference`：六图真实模型批量顺序/检测结果与参考对照（可选）。
- `--queue-capacity=-1` 与超出 `size_t` 的容量值会在 CLI 解析阶段返回 2。

任务 5A 新增 9 个不需要摄像头的测试：

- `camera_missing_device`：设备不存在时明确报错；
- `camera_timeout`：注入超时 backend，验证 `next()` 返回 timeout；
- `camera_invalid_options`：不支持的像素格式被拒绝；
- `camera_frame_conversion`：2×2 YUYV 注入帧转 BGR、序号和时间戳；
- `camera_buffer_validation`：`V4L2_BUF_FLAG_ERROR`、空帧、超长帧、非法索引；
- `camera_busy_error`：初始化 EBUSY 统一归类为 device busy；
- `camera_read_deadline`：poll EINTR / DQBUF EAGAIN / 连续 DQBUF EINTR 共用截止时间；
- `camera_colorimetry`：仅接受 BT.601/limited（含 default 映射），拒绝其他；
- `cli_camera`：无设备、缺少参数/非法帧数/格式/尺寸及错误组合。

任务 5B 新增模拟重连与中断测试：

- `camera_reconnect_success`：设备失联后重开成功，序号不重置；
- `camera_reconnect_exhausted`：重试预算耗尽返回 Fatal；
- `camera_reconnect_open_error`：工厂返回“非空对象 + 错误”仍按失败处理；
- `camera_reconnect_no_frame_budget`：重开成功但始终无帧，累计预算只重开 3 次；
- `camera_reconnect_stop`：重连等待期间停止，返回 End 且不再重开；
- `camera_reconnect_stop_during_open`：最后一次重开途中停止，按正常停止返回
  End 且错误为空，不报重连耗尽；
- `camera_bad_frame_sequence`：坏帧只记 DataError，后续好帧序号继续；
- `camera_read_stop` / `camera_source_external_stop`：3 秒读超时期间停止可在
  1 秒内结束并返回 End；
- `threaded_external_stop_drains`：省略帧数时连续运行，停止后只排空已接收帧；
- `cli_signal`：图片批量模式收到 SIGINT 以 130 终止，摄像头处理器不接管。

任务 5C 新增展示测试：

- `annotation_ng_boxes` / `annotation_error_frame`：非正方形原图坐标不缩放，
  ERROR 帧只画状态条不画框；
- `annotation_label_avoids_banner` / `annotation_label_right_edge`：左上角标签
  不压状态条，右侧标签不越出图像；
- `annotation_save_png`：无 GUI 写 PNG 并回读尺寸/框像素；
- `preview_drops_frames`：慢窗口只保留最新预览帧，submit 不阻塞；
- `preview_close_stops` / `preview_error_stops`：关闭或显示失败置停止标志；
- `preview_closed_without_frames`：首帧后断流，仍能通过窗口 poll 检测关闭；
- `threaded_preview_image_included`：预览开启时结果携带源图；
- `cli_annotation`：无桌面保存、尺寸检查、路径冲突、headless 和失效 DISPLAY 守卫。

`cli_empty_detections` 使用合法契约、输出全零的合成 ONNX 模型，端到端确认
无候选时返回 `detections: 0` 且退出码为 0。`cli_path_conflicts` 检查输出路径
指向模型、图片或彼此（含相对路径/悬空符号链接/硬链接）时被拒绝，且原文件
SHA256 不变。真实模型对照守卫检查六张图必须齐全、逐图 SHA256 与参考一致、
运行阈值与参考结果一致。详细说明见
[docs/input-and-quality.md](docs/input-and-quality.md)。

## 目录结构

```text
.
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/
│   ├── batch_processor.hpp
│   ├── bounded_queue.hpp
│   ├── camera_source.hpp
│   ├── image_preprocess.hpp
│   ├── image_source.hpp
│   ├── model_contract.hpp
│   ├── model_runner.hpp
│   ├── postprocess.hpp
│   ├── quality.hpp
│   ├── sha256.hpp
│   ├── system_info.hpp
│   └── threaded_pipeline.hpp
├── src/
│   ├── batch_processor.cpp
│   ├── camera_source.cpp
│   ├── image_preprocess.cpp
│   ├── image_source.cpp
│   ├── main.cpp
│   ├── model_contract.cpp
│   ├── model_runner.cpp
│   ├── postprocess.cpp
│   ├── quality.cpp
│   ├── sha256.cpp
│   ├── system_info.cpp
│   └── threaded_pipeline.cpp
├── docs/
│   ├── model-verification.md          # 2A 模型来源/许可/接口/参考推理
│   ├── cpp-inference.md               # 2B C++ 推理与对照记录
│   ├── cpp-inference-run.txt          # 2B/3/4A/4B 干净构建/CTest/推理/失败路径运行日志
│   ├── input-and-quality.md           # 任务 3 输入抽象与 OK/NG 判定
│   ├── bounded-queue.md               # 任务 4A 队列语义与满载/并发指标
│   ├── threaded-pipeline.md           # 任务 4B 三线程流水线与故障/积压记录
│   ├── v4l2-camera.md                 # 任务 5A V4L2 采集与虚拟设备证据
│   ├── v4l2-capture-run.txt           # vivid 30 帧采集日志
│   ├── v4l2-verification.txt          # 5A 干净构建、49 项测试和故障回归
│   ├── camera-inference.md            # 任务 5B 三线程摄像头检测与重连
│   ├── camera-inference-run.txt       # 5B vivid 30 帧、连续模式、Ctrl+C、坏帧日志
│   ├── visualization.md               # 任务 5C 标注、预览缓存与退出语义
│   ├── visualization-run.txt          # 5C 无桌面保存、非正方形核对、vivid 预览日志
│   ├── evidence/
│   │   ├── synthetic-non-square.png   # 本地生成的 480×200 合成输入
│   │   ├── annotated-non-square.png   # 左上角标签避让 NG 状态条
│   │   ├── synthetic-right-edge.png   # 右侧缺陷合成输入
│   │   └── annotated-right-edge.png   # 右侧标签不越出图像
│   ├── reference-inference-results.json
│   └── reference-inference-run.txt
├── tools/
│   ├── neu_det_reference.py
│   ├── compare_author_predict.py
│   ├── compare_cpp_reference.py       # C++ 与 2A 参考结果对照
│   ├── compare_batch_reference.py     # 线程化六图批量与参考对照
│   ├── reproduce_reference_verification.sh
│   ├── generate_test_fixtures.py
│   └── requirements-reference.txt
└── tests/
    ├── cli_tests.cmake
    ├── cli_inference_tests.cmake
    ├── cli_empty_detections.cmake
    ├── cli_path_conflict_tests.cmake
    ├── cli_batch_tests.cmake
    ├── cli_camera_tests.cmake
    ├── cli_signal_tests.cmake
    ├── cli_annotation_tests.cmake
    ├── compare_reference_guard_tests.cmake
    ├── queue_tests.cpp
    ├── threaded_pipeline_tests.cpp
    ├── threaded_batch_reference.cmake
    ├── unit_tests.cpp
    └── fixtures/                      # 合成图片/契约错误 ONNX/空输出模型/损坏模型
```

构建目录 `build/`、模型权重、测试图片和本地环境文件均不纳入版本库。

## 阶段 2A：模型核验

候选模型为 YOLOv8n-NEU-DET（`winthegame259/steel-surface-defect-detection`
release v1.0.0）。详细记录见 [docs/model-verification.md](docs/model-verification.md)：

- ONNX 文件 SHA256 与 release digest 一致；
- 输入 `images [1,3,640,640] float32`，BGR、letterbox、`/255`、NCHW；
- 输出 `output0 [1,10,8400] float32`，通道优先、未内置 NMS；
- 类别顺序来自 ONNX metadata：`crazing, inclusion, patches, pitted_surface,
  rolled-in_scale, scratches`；
- 模型的 AGPL-3.0 与 NEU-DET 学术研究限制仍是待确认事项。

## 阶段 2B：C++ 单图推理

完整记录、失败输出和限制见 [docs/cpp-inference.md](docs/cpp-inference.md)。
关键结果：

- ONNX Runtime 1.32.0 CPU ExecutionProvider 加载成功，模型 SHA256 与 2A 一致；
- 六张固定图的检测数量、类别、置信度和 bbox 与 2A Python 参考一致：
  `max_conf_diff=1.192e-06`、`max_bbox_diff=3.436e-05`、
  `max_blob_diff=5.960e-08`；
- 320×200 非方形图验证 letterbox pad=(0,120)，Python/C++ 预处理 Tensor
  最大差 `5.960e-08`；
- FR-007：未知 ID 返回 `Unknown` 并记录 anomaly，不会静默映射到已有类别。

## 阶段 3：图片输入与 OK/NG 判定

完整设计、错误策略和测试记录见
[docs/input-and-quality.md](docs/input-and-quality.md)。关键点：

- `Frame` + `ImageSource` 统一接口，`SingleImageSource`/`DirectoryImageSource`
  两种实现；批处理只依赖接口；
- `Detection` + `QualityResult`，`--conf` 可配置；至少一个置信度严格大于
  阈值的检测 → NG，否则 OK；
- 单图缺失/不支持/损坏、目录缺失/空/含坏文件均有明确行为；批量坏文件继续
  处理、汇总错误并以退出码 1 结束；
- 真实模型六图批量结果：`ok=0 ng=6 errors=0`。

## 任务 4A：有界队列

实现与详细测试见 [docs/bounded-queue.md](docs/bounded-queue.md)。
`include/bounded_queue.hpp` 提供容量可配置的阻塞队列：满时生产者等待、空时
消费者等待、`close()` 幂等并唤醒所有等待者、关闭后拒绝新项但允许排空。
队列测试仅使用整数序号，不依赖模型或图片。

满载实测：`capacity=2 enqueued=3 dequeued=3 max_depth=2 producer_wait_count=1`。
并发实测（本次运行）：`capacity=4 enqueued=2000 dequeued=2000 max_depth=4
producer_wait_count=1973`，序号无丢失、无重复、无乱序。队列尚未接入主程序，
4B 再完成三线程接线。

## 任务 4B：三线程批量流水线

详细设计、线程/队列指标与故障记录见
[docs/threaded-pipeline.md](docs/threaded-pipeline.md)。`--image-dir` 现在使用：

```text
读取线程 -> 输入队列 -> 推理线程 -> 结果队列 -> 结果线程
```

两个队列容量由 `--queue-capacity` 设置（默认 4，必须 >0）；结果线程唯一维护
顺序、total/ok/ng/errors 和汇总。六图真实模型容量 1 的正常排空结果为
`total=6 ok=0 ng=6 errors=0`，检测结果与 2A/3 参考一致；容量 1 受控积压
测试中两个队列最大深度均为 1；下游回调致命失败会关闭队列、唤醒并 join 全部
线程后返回非零。

## 任务 5A：V4L2 摄像头采集

详细设备信息、请求/实际格式、错误分类和 30 帧日志见
[docs/v4l2-camera.md](docs/v4l2-camera.md)。

- `CameraSource` 通过 V4L2 MMAP 连续采集，支持 YUYV/MJPG 转 BGR；返回前复制
  数据并交还驱动缓冲区，`Frame` 可独立使用。
- `--capture-only --max-frames N` 不需要模型；设备不存在、占用、格式/帧率
  不支持、读取超时都有可区分错误并以非零退出。
- 30 帧采集使用 vivid 虚拟设备验证：`/dev/video0`、YUYV 640x480 @30，
  序号 0..29，退出码 0；坏帧标志被拒绝、缓冲区归还后恢复正常采集，设备
  占用明确报告 busy。
- **AT-009 真实 USB 摄像头直通待实机验证**：当前 `lsusb` 无摄像头，VMware
  未直通；vivid 日志仅用于实现验证。
- 干净构建与回归：`docs/v4l2-verification.txt`；30 帧采集：
  `docs/v4l2-capture-run.txt`。
- 5A 自动测试 9 个，均不依赖摄像头；当前完整套件 71/71（含 5B/5C 新增测试），通过。

## 任务 5B：摄像头三线程检测与有限重连

详细设计与证据见 [docs/camera-inference.md](docs/camera-inference.md)。

- `--model ... --camera /dev/videoN` 复用读取/推理/结果三线程；省略
  `--max-frames` 时连续运行，指定时按输入项数停止；
- 摄像头结果逐帧回调，`retain_frames=false`，长期运行不使用
  `BatchResult.frames` 累积内存；
- 坏帧记 ERROR、不做 OK/NG；设备失联按可配置次数/间隔重开，序号不重置；
- 重开以“对象非空且错误为空”为成功；重试预算在一次失联内累计，只有收到
  有效帧才恢复并清零，避免“打开成功但一直无帧”无限重连；
- 失联、重开、恢复、耗尽均输出 `camera reconnect: ...` 日志；
- 重试耗尽或源 Fatal 时关闭队列、join 全部线程并返回非零；
- 摄像头模式才接管 SIGINT；停止接收并排空已接收帧后正常退出，无帧等待按
  50ms 分片检查停止；图片模式保持默认中断行为；
- 模拟测试覆盖打开失败（非空对象+错误）、无帧预算耗尽、重连等待中停止、
  坏帧序号和停止可打断读等待；
- 队列容量 1 的 30 帧 vivid 检测：`total=30 ok=0 ng=30 errors=0`，两个队列
  峰值均为 1；连续模式 6 秒处理 69 帧，Ctrl+C 后 `stopped=1` 正常退出。

## 任务 5C：检测结果展示

详细设计、坐标核对和预览统计见
[docs/visualization.md](docs/visualization.md)。

- 单图和摄像头共用 `annotateFrame()`：原图坐标画框、类别、置信度，并显示
  `OK`/`NG`/`ERROR`；ERROR 帧不画框、不产生质量判定；
- 单图支持 `--save-annotated`（无桌面）和 `--display`（窗口）；
- 摄像头 `--display` 的 highgui 全在主线程执行；结果线程只做非阻塞
  submit，待显示缓存只有 1 帧；窗口慢时覆盖旧预览帧并计数 `dropped`；
- 主线程每约 15ms pump 一次，**无新帧时也处理窗口事件**；断流/重连期间
  关窗或按 Esc 仍会设置 5B 外部停止标志并正常排空；
- 无 `DISPLAY` 时 `--display` 返回 1；`DISPLAY` 已设置但连接失效时先在子
  进程探测 highgui，失败返回可读错误而不是 Qt SIGABRT；
- 状态条与类别标签先布局后绘制，标签避开状态条；靠右边缘的标签限制在
  图像内；
- 480×200 非正方形合成图实测：2 个检测框与 JSON 坐标逐边核对一致，保存图
  仍为 480×200；左上角标签已避开 NG 状态条，右侧样例的标签不越界；
- 上一轮 vivid 预览 53 帧，`submitted=53 shown=47 dropped=6`，`stopped=1`
  退出码 0；本轮复验机无 `/dev/video*`，摄像头预览未重跑，断流关窗由
  `preview_closed_without_frames` 单元测试覆盖；
- 5C 自动测试 11 个；完整套件 71/71，无外部资产 68 个。

## 验收记录（2B/3）

```text
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=... -DEDGE_AI_MODEL_PATH=... \
    -DEDGE_AI_TEST_IMAGES_DIR=... -DEDGE_AI_PYTHON=...
-- ONNX Runtime library: .../libonnxruntime.so
-- Configuring done
-- Generating done

$ cmake --build build
# 构建完成：build/bin/edge_ai、build/bin/edge_ai_tests

$ ctest --test-dir build --output-on-failure
...
100% tests passed, 0 tests failed out of 53

$ ./build/bin/edge_ai --model ... --image .../crazing_101.jpg \
    --conf 0.25 --iou 0.45
ONNX Runtime 1.32.0 providers=CPUExecutionProvider
input:  images [1, 3, 640, 640] tensor(float)
output: output0 [1, 10, 8400] tensor(float)
...
detections: 2
  class_id=0 crazing conf=0.525822 bbox_xyxy=[48.0117, 37.2517, 195.252, 94.7151]
  class_id=0 crazing conf=0.484310 bbox_xyxy=[4.31551, 108.104, 180.438, 199.272]
quality: NG (confidence>0.25, qualifying=2/2)

$ ./build/bin/edge_ai --model ... --image-dir build/cli-batch-tests/real-batch \
    --conf 0.25 --iou 0.45
...
summary: total=6 ok=0 ng=6 errors=0 skipped_unsupported=0

$ ./build/bin/edge_ai --model tests/fixtures/zero_output.onnx \
    --image-dir build/cli-batch-tests/mixed
...
[ERROR] .../bad.jpg: failed to read image: .../bad.jpg
[OK] .../valid.png detections=0
summary: total=2 ok=1 ng=0 errors=1 skipped_unsupported=1

$ ./build/bin/edge_ai --model /tmp/does-not-exist.onnx \
    --image tests/fixtures/non_square_320x200.png
edge_ai: model file not found: /tmp/does-not-exist.onnx
# exit code: 1

$ ctest --test-dir build -R cpp_reference_comparison --output-on-failure
SUMMARY: images=6 failures=0 max_conf_diff=1.192e-06 \
  max_bbox_diff=3.436e-05 max_blob_diff=5.960e-08
```

## 许可与仓库策略

模型代码/权重按仓库与 ONNX metadata 记录为 AGPL-3.0；NEU-DET 只有学术研究
用途线索，官方没有明确许可证文本。商业使用、闭源分发和测试图再分发仍为
**待确认**。本仓库不提交模型权重、测试图片、密码或令牌。
