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
- **状态条与标签避让**：先计算状态条矩形，再为每个标签尝试框上方、框内
  顶部、框下方、框内底部、框右侧、框左侧和状态条下方等位置；选择第一个
  完整位于图像内且不与状态条相交的位置。靠右边缘的标签会水平裁剪/回退到
  图像内。`computeAnnotationLayout()` 暴露状态条和标签矩形，测试直接断言
  二者不相交。
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
  `QT_QPA_PLATFORM=offscreen`）时，`--display` 直接返回 1 并提示使用
  `--save-annotated`。
- `DISPLAY`/`WAYLAND_DISPLAY` 已设置但连接失效时，先由子进程探测 highgui；
  探测失败（包括 Qt 的 SIGABRT）会转换成可读错误并返回 1。`DISPLAY=:9999`
  已验证不会再把主进程以 134 中止。
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
                 主线程 pump：annotate + imshow + waitKey(1)
                 无新帧时也 poll 窗口事件
```

- `PipelineOptions::include_frame_image` 仅在启用预览时打开；推理线程把
  该帧图像放入 `FrameResult`。结果队列容量 1 时，图像内存有界。
- `PreviewWindow::submit()` 只持锁替换“最新待显示帧”，当窗口慢时覆盖旧帧
  并累计 `dropped`；它**不等待窗口**，因此检测输出不被预览拖慢。
- 所有 highgui 调用（`namedWindow`/`imshow`/`waitKey`/`destroyWindow`）都在
  进程主线程执行。启用预览时三线程流水线在 worker 线程运行，主线程每约
  15ms 调用一次 `pump()`。
- **没有新帧时** `pump()` 仍调用 sink 的 `poll()`（`waitKey(1)` + 窗口可见性
  检查）。因此摄像头断流/重连期间，关窗或按 Esc 仍会立即设置 5B 外部停止
  标志，而不是等到下一帧。
- 用户关闭窗口或按 Esc/Q 时，主线程设置外部停止标志；读取线程停止接收，
  已入队帧继续经过推理和结果线程排空，然后正常退出。
- 显示后端异常同样设置停止标志；worker 结束后 CLI 输出失败原因并返回 1。
- `--display` 启动前会在**子进程**里探测一次 highgui：Qt 无法加载平台插件或
  连接 DISPLAY 时子进程会 abort，父进程看到子进程异常退出后返回可读错误，
  不会让检测进程收到 SIGABRT。子进程禁用 core dump，并把 Qt 诊断写入
  `/dev/null`。

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

这验证了标注框与检测 JSON 的原图坐标一致，没有 letterbox 偏移；新布局下
左上角 `pitted_surface 0.62` 标签位于 NG 状态条下方，不再被遮挡。

### 靠右边缘样例

第二个本地合成输入 `docs/evidence/synthetic-right-edge.png` 在右边缘产生
两个 `scratches` 检测（其中一个框到 x=480）：

```text
detections: 3
  pitted_surface conf=0.862909 bbox=[17.3074, 0, 396.557, 200]
  scratches      conf=0.642606 bbox=[412.836, 22.2368, 480, 62.8815]
  scratches      conf=0.556563 bbox=[399.962, 127.814, 479.931, 156.754]
```

标注结果：[docs/evidence/annotated-right-edge.png](evidence/annotated-right-edge.png)。
`scratches 0.64`/`0.56` 标签被限制在图像右边界内，没有越界；框边像素仍与
JSON 坐标一致。

## 5. 摄像头预览证据（vivid）

> 本轮复验的虚拟机没有 `/dev/video*` 采集节点，因此没有重新运行 vivid
> 摄像头预览。以下统计来自 5C 上一轮（预览线程版本）的 vivid + offscreen
> 运行；主线程 pump 后的窗口/断流关闭路径由本轮单元测试覆盖。

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
- 预览显示 47 帧、跳过 6 帧（`dropped=6`），说明慢窗口只丢预览，
  不影响检测；
- SIGINT 后按 5B 流程排空并正常退出；
- 本轮新增 `preview_closed_without_frames` 覆盖“首帧后断流仍能关窗”，
  以及 `DISPLAY=:9999` 返回 1 而不 abort 的 CLI 守卫。

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

本次完整套件 **73/73 通过**；无模型/图片/摄像头时 70 个测试通过。

任务 5C 新增：

| 测试 | 内容 |
| --- | --- |
| `annotation_ng_boxes` | 非正方形原图、框坐标不缩放、不修改源图 |
| `annotation_error_frame` | ERROR 帧只画状态条，不画检测框 |
| `annotation_label_avoids_banner` | 左上角框的标签不与状态条相交 |
| `annotation_labels_do_not_overlap` | 相邻框的标签互相避让，不互相覆盖 |
| `annotation_label_right_edge` | 靠右边缘的标签不越出图像 |
| `annotation_save_png` | 无 GUI 保存/回读 PNG，尺寸与框像素保持 |
| `preview_drops_frames` | 慢速预览只保留最新帧，`dropped>0`，submit 不阻塞 |
| `preview_close_stops` | 关闭窗口置外部停止标志，记录 user close |
| `preview_drains_final_frame` | 流水线结束时仍先显示最后一帧（`--max-frames 1`） |
| `preview_closed_without_frames` | 首帧后停止送帧，仍能通过 poll 检测关窗并退出 |
| `preview_error_stops` | 显示失败置停止标志并保留诊断 |
| `threaded_preview_image_included` | 预览开启时结果携带源图；关闭时不保留 |
| `cli_annotation` | 无桌面保存、尺寸、路径冲突、headless/失效 DISPLAY/offscreen 守卫 |

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
8. 子进程探测覆盖“无法加载 Qt 插件/无法连接 DISPLAY”，但真实桌面上的窗口
   管理器行为、缩放和鼠标关闭仍按 AT-015 实机验收。
9. 主线程每约 15ms pump 一次；预览关闭延迟上限约为一次 pump + `waitKey(1)`。
