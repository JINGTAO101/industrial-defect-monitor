# 任务 5C：检测结果展示（单图与摄像头预览）

- 完成日期：2026-11-19
- 对应 SRS：FR-011、FR-017；AT-015
- 入口：
  - 单图：`edge_ai --model ... --image ... [--save-annotated out.png] [--display]`
  - 摄像头：`edge_ai --model ... --camera /dev/videoN [--display]`

## 1. 共用标注逻辑

实现：`include/annotation.hpp`、`src/annotation.cpp`。

- `annotateFrame()` 接收原始 BGR 图像和 `AnnotationInput`，返回新的标注图，
  不修改调用方图像。
- 检测框直接使用 `Detection.x1/y1/x2/y2`。这些坐标已由后处理从 640×640
  letterbox 还原到原图，因此标注层**不再做任何缩放**；非正方形图片的框
  保持原图坐标。
- 每个框绘制类别和两位小数置信度；置信度来自模型后处理。
- 左上角绘制 `OK`、`NG` 或 `ERROR` 状态条：
  - `OK`/`NG` 使用质量判定结果；
  - `has_error=true` 时只绘制 `ERROR`，**不绘制任何检测框**，避免错误帧被
    误解为完成的质量判定；
  - 未完成判定的帧显示 `N/A`（正常路径不会出现）。
- 非法/非有限、宽高倒置、越界的框会被跳过或裁剪，不越界绘制。
- 单图保存和摄像头预览都调用同一个 `annotateFrame()`。

## 2. CLI

```bash
# 单图：无桌面保存标注图
./build/bin/edge_ai --model model.onnx --image part.jpg \
  --save-annotated build/part-annotated.png

# 单图：窗口预览（可选择同时保存）
./build/bin/edge_ai --model model.onnx --image part.jpg --display

# 摄像头：连续检测 + 窗口预览
./build/bin/edge_ai --model model.onnx --camera /dev/video0 --display
```

- `--save-annotated <path>` 仅支持 `--image`；会创建父目录，并与模型、输入
  图片、`--json-out`、`--dump-input` 做路径冲突检查。
- `--display` 支持单图和摄像头；`--image-dir`、`--capture-only` 明确拒绝。
- 无桌面环境（无 `DISPLAY`/`WAYLAND_DISPLAY` 且未设置
  `QT_QPA_PLATFORM=offscreen`）时，`--display` 在调用 highgui 前返回 1 并
  提示使用 `--save-annotated`，不会让 Qt 直接 abort。
- 单图推理失败时，如果请求了保存/预览，也会生成 `ERROR` 标注图，但退出码
  仍为 1，且不生成质量判定。

## 3. 摄像头预览结构

```text
推理线程 ——> 结果队列 ——> 结果线程（打印结果 + 非阻塞 submit）
                                      │
                                      ▼
                          pending（容量 1，只保留最新帧）
                                      │
                                      ▼
                             预览线程：annotate + imshow
```

- `PipelineOptions::include_frame_image` 仅在启用预览时打开；推理线程把
  该帧图像放入 `FrameResult`。结果队列容量 1 时，图像内存有界。
- `PreviewWindow::submit()` 只持锁替换“最新待显示帧”，当窗口慢时覆盖旧帧
  并累计 `dropped`；它**不等待窗口**，因此检测输出不被预览拖慢。
- 预览线程负责标注、`imshow` 和 `waitKey`，与推理/结果线程分离。
- 用户关闭窗口或按 Esc/Q 时，预览线程设置 5B 的外部停止标志；读取线程停止
  接收，已入队帧继续经过推理和结果线程排空，然后正常退出。
- 显示后端异常（无法创建窗口等）同样设置停止标志；`run()` 结束后 CLI 输出
  失败原因并返回 1。

## 4. 非正方形单图证据

输入为本地生成的 480×200 合成钢表面纹理
`docs/evidence/synthetic-non-square.png`（不包含 NEU-DET 图像，避免数据集
再分发问题）。真实模型检测到 2 个框：

```text
image: docs/evidence/synthetic-non-square.png 480x200
preprocess: letterbox ratio=1.33333 pad=(0,187)
detections: 2
  class_id=3 pitted_surface conf=0.621675 bbox_xyxy=[0, 0, 190.117, 199.114]
  class_id=5 scratches conf=0.264694 bbox_xyxy=[315.195, 166.302, 376.593, 185.851]
quality: NG
annotated image saved: docs/evidence/annotated-non-square.png
```

标注结果：[docs/evidence/annotated-non-square.png](evidence/annotated-non-square.png)。

自动核对 JSON 框坐标与保存后的 PNG 像素：对每个框取左/右/上/下边中点，
像素均为 NG 红色 `(B,G,R)=(0,0,255)`；框外左侧像素保持背景色。输出：

```text
pitted_surface (0, 0, 190, 199) edge (0, 99) (0, 0, 255)
pitted_surface (0, 0, 190, 199) edge (189, 99) (0, 0, 255)
...
scratches (315, 166, 377, 186) edge (315, 176) (0, 0, 255)
...
all borders at JSON coordinates: True
```

这验证了标注框与检测 JSON 的原图坐标一致，没有 letterbox 偏移。

## 5. 摄像头预览证据（vivid）

环境无桌面，使用 OpenCV Qt highgui 的 offscreen 平台验证展示链路：

```bash
QT_QPA_PLATFORM=offscreen timeout --preserve-status -s INT 5s \
  ./build/bin/edge_ai --model ... --camera /dev/video0 --display --queue-capacity 1
```

```text
...
summary: total=53 ok=0 ng=53 errors=0 reconnect_count=0 stopped=1
preview: submitted=53 shown=47 dropped=6 closed_by_user=0
camera stopped; accepted frames drained
# exit code: 0
```

- 53 帧结果全部输出，队列峰值仍为 1；
- 预览线程显示 47 帧、跳过 6 帧（`dropped=6`），说明慢窗口只丢预览，
  不影响检测；
- SIGINT 后按 5B 流程排空并正常退出。

窗口关闭路径由 `preview_close_stops` 覆盖：模拟 sink 返回 Closed 后，外部
停止标志被置位，`closed_by_user=true`，不记录为失败。

## 6. 无桌面保存测试

`cli_annotation` 不依赖图形环境：

1. 用 `zero_output.onnx` + `non_square_320x200.png` 运行
   `--save-annotated`，要求退出 0、文件存在且非空；
2. 用 `file` 检查保存后的 PNG 仍为 `320 x 200`；
3. 验证标注输出与输入、JSON 输出路径冲突时返回 2；
4. 验证 `--save-annotated`/`--display` 与 `--image-dir` 的组合被拒绝；
5. 在无 `DISPLAY`、无 `QT_QPA_PLATFORM` 的环境中，`--display` 返回 1 并
   给出明确提示，而不是 Qt abort。

## 7. 测试

```bash
ctest --test-dir build --output-on-failure
```

本次完整套件 **68/68 通过**；无模型/图片/摄像头时 65 个测试通过。

任务 5C 新增：

| 测试 | 内容 |
| --- | --- |
| `annotation_ng_boxes` | 非正方形原图、框坐标不缩放、不修改源图 |
| `annotation_error_frame` | ERROR 帧只画状态条，不画检测框 |
| `annotation_save_png` | 无 GUI 保存/回读 PNG，尺寸与框像素保持 |
| `preview_drops_frames` | 慢速预览只保留最新帧，`dropped>0`，submit 不阻塞 |
| `preview_close_stops` | 关闭窗口置外部停止标志，记录 user close |
| `preview_error_stops` | 显示失败置停止标志并保留诊断 |
| `threaded_preview_image_included` | 预览开启时结果携带源图；关闭时不保留 |
| `cli_annotation` | 无桌面保存、尺寸、路径冲突和 headless 显示守卫 |

## 8. 限制与待确认

1. **AT-009 仍待真实 USB 摄像头**；vivid 仅验证软件链路。
2. **AT-015 的窗口交互**在无桌面环境用 offscreen + 模拟 sink 验证；真实
   桌面上的鼠标关闭/按键行为和缩放窗口未实机验证。
3. 摄像头预览只用于显示，不录制视频；单图保存一次一张。
4. 预览只保留最新一帧，慢窗口会跳过中间帧；检测结果不受影响。
5. `--save-annotated` 目前只支持单图；摄像头逐帧落盘未实现。
6. 窗口关闭后的排空依赖 5B 的外部停止语义；实时画面延迟取决于窗口本身。
7. 没有成功解码画面的坏帧（例如摄像头 V4L2 错误帧）只输出控制台 ERROR，
   没有图像可标注；有图像但推理失败的帧会显示 ERROR 标注。
8. `DISPLAY` 存在但连接失效时，Qt 仍可能在插件初始化阶段终止进程；当前只
   对“无显示环境变量”做前置拒绝，真实桌面/远程 X 的有效性需实机确认。
