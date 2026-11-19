# 任务 5A：V4L2 摄像头采集

- 完成日期：2026-11-19
- 对应 SRS：FR-002、FR-003、§7.4；AT-009
- 实现：`include/camera_source.hpp`、`src/camera_source.cpp`
- 入口：`edge_ai --camera /dev/videoN --capture-only --max-frames N`

## 1. 设备直通与实测环境

VMware 虚拟机中**没有 USB 摄像头直通**：`lsusb` 只显示虚拟 hub 和 VMware
Virtual Mouse，初始没有 `/dev/video*` 节点。

为了验证 V4L2 实现，加载了 Ubuntu 内核自带的 `vivid` 虚拟视频驱动：

```bash
sudo modprobe vivid
v4l2-ctl --list-devices
# vivid (platform:vivid-000):
#   /dev/video0 ... /dev/video3
```

因此本文的 30 帧采集日志来自 **vivid 虚拟设备**，不是真实 USB 摄像头。
**AT-009 真实摄像头验收仍待实机验证。**

真实摄像头接入时，当前用户需在 `video` 组中（并重新登录），或由管理员配置
相应 udev 权限。

## 2. 请求值与设备实际值

实测节点：`/dev/video0`（vivid）。

请求：

```bash
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV \
  --set-parm=30
```

实际（v4l2-ctl 回读）：

| 项目 | 请求值 | 实际值 |
| --- | --- | --- |
| 像素格式 | YUYV | YUYV 4:2:2 |
| 尺寸 | 640×480 | 640×480 |
| 帧率 | 30 fps | 30.000 fps |
| bytes per line | — | 1280 |
| 色彩 | — | sRGB / ycbcr default / quantization default |
| 驱动 | — | vivid (platform:vivid-000) |

程序自身打印的请求/实际值也一致：

```text
requested: YUYV 640x480 @ 30 fps timeout=2000 ms
actual: YUYV 640x480 @ 30.000 fps bytes_per_line=1280
```

## 3. CameraSource 设计

- 复用 `ImageSource` / `Frame`；`next()` 返回 `Frame` 或 `Error`。
- V4L2 MMAP 采集，4 个缓冲区；`poll()` 读取超时（默认 2000 ms）。
- 支持 `YUYV` 与 `MJPG`：
  - YUYV → `cv::cvtColor(..., COLOR_YUV2BGR_YUYV)`；
  - MJPG → `cv::imdecode(..., IMREAD_COLOR)`。
- 出队后先把驱动缓冲区内容复制到 `std::vector`，在返回 `Frame` 前
  `VIDIOC_QBUF` 交还驱动；`Frame.image` 是自己拥有的 BGR `cv::Mat`，之后不会
  被驱动复用覆盖。
- 出队缓冲区带 `V4L2_BUF_FLAG_ERROR`、空帧、超长帧或非法索引时拒绝该帧；
  只要索引有效，即使帧损坏/长度非法也会先 `VIDIOC_QBUF` 归还缓冲区。
- 整个读取使用同一个 `steady_clock` 截止时间：`poll` 的 EINTR 和 `DQBUF` 的
  EAGAIN/EINTR 都按剩余时间重试，不会重新获得完整超时，也不会在 EAGAIN 时
  立即误报超时。`VIDIOC_DQBUF` 只执行**一次原始 ioctl**，不内部重试
  EINTR，否则外层的截止时间循环看不到中断、无法按时退出。
- YUYV 只接受 BT.601/default 编码 + limited/default 量化（OpenCV
  `COLOR_YUV2BGR_YUYV` 的固定转换）；其他 colorspace/encoding/quantization
  明确拒绝。实测 vivid 为 `colorspace=sRGB ycbcr=default quantization=default`。
- 实际帧率来自 `VIDIOC_G_PARM`；若无法确认则记录为 0，CLI 打印 `unknown`，
  不再用请求值冒充实际值。
- `Frame.source` 为设备节点；`Frame.index` 从 0 开始递增；`Frame.timestamp`
  为成功解码后的 `system_clock` 墙钟时间。
- 析构时 `STREAMOFF`、`munmap`、`close`；构造/读取失败路径都会释放资源。

错误诊断（非零退出）：

- 设备不存在：`camera device not found: ...`
- 设备被占用：`camera device busy: ...`
- 权限不足：`camera permission denied: ...`
- 格式/尺寸不支持：`camera format not supported: ...`
- 帧率不支持：`camera frame rate not supported: ...`
- 读取超时：`camera read timeout after N ms`
- 驱动返回无效帧/队列错误：明确的 `camera ... failed` 信息

## 4. CLI 与 30 帧采集证据

```bash
./build/bin/edge_ai \
  --camera /dev/video0 \
  --capture-only \
  --max-frames 30 \
  --camera-format YUYV \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30
```

`--capture-only` 不需要模型，也不能与 `--model`/`--image`/`--image-dir`/
`--json-out`/`--dump-input` 组合；非法参数返回 2。

运行摘要（完整日志见 `docs/v4l2-capture-run.txt`；干净构建与故障回归见
`docs/v4l2-verification.txt`）：

```text
camera: /dev/video0
requested: YUYV 640x480 @ 30 fps timeout=2000 ms
actual: YUYV 640x480 @ 30.000 fps bytes_per_line=1280
frame 0: 640x480 channels=3 timestamp_ms=...
...
frame 29: 640x480 channels=3 timestamp_ms=...
camera capture complete: device=/dev/video0 format=YUYV size=640x480
  fps=30 frames=30 index_range=0..29
# exit code: 0
```

这是 **vivid 虚拟设备** 的连续 30 帧采集，序号 0..29 完整、无空帧；
真实摄像头日志和 AT-009 仍需在具备 USB 直通的实机补测。

补充的 vivid 故障回归：

```text
$ v4l2-ctl -d /dev/video0 --set-ctrl inject_v4l2_buf_flag_error=1
$ edge_ai --camera /dev/video0 --capture-only --max-frames 1
edge_ai: camera capture failed: camera frame marked with V4L2_BUF_FLAG_ERROR
# exit code: 1

# 随后正常采集 2 帧仍成功，说明坏帧缓冲区已归还
camera capture complete: ... frames=2 index_range=0..1
# exit code: 0

# 另一进程占用设备时
edge_ai: camera device busy: /dev/video0: Device or resource busy
# exit code: 1
```

## 5. 自动测试（无摄像头可运行）

```bash
ctest --test-dir build --output-on-failure
```

| 测试 | 内容 |
| --- | --- |
| `unit_camera_missing_device` | 不存在的设备返回 `camera device not found` |
| `unit_camera_timeout` | 注入超时 backend，`next()` 返回明确 timeout 错误 |
| `unit_camera_invalid_options` | 不支持的像素格式被拒绝 |
| `unit_camera_frame_conversion` | 注入 2×2 YUYV 帧，验证 BGR、序号、时间戳 |
| `unit_camera_buffer_validation` | 坏帧标志、空帧、超长帧、非法索引被拒绝 |
| `unit_camera_busy_error` | EBUSY 归类为 device busy，EACCES 归类为权限问题 |
| `unit_camera_read_deadline` | poll EINTR、DQBUF EAGAIN 与连续 DQBUF EINTR 共用同一截止时间 |
| `unit_camera_colorimetry` | 仅 BT.601/limited（含 default）可用，其他被拒绝 |
| `cli_camera` | 无设备、缺少 `--capture-only`/`--camera`、非法帧数/格式/尺寸、错误组合 |

当前完整套件 **59/59 通过**；无模型/图片/摄像头时 56 个测试通过。干净
构建日志见 `docs/v4l2-verification.txt`。

## 6. 待确认 / 限制

1. **AT-009 待实机**：VMware 当前没有 USB 摄像头直通，只有 vivid 虚拟设备；
   真实摄像头需单独补 30 帧日志。
2. 设备被占用（`EBUSY`）和“驱动返回格式与请求不一致”的路径已实现，但无真实
   硬件触发记录；自动测试未覆盖占用场景。
3. MJPG 转换已实现，但本次实测使用 YUYV；MJPG 真实设备待验证。
4. 仅支持单平面视频采集；不支持多平面或格式自动协商。
5. 断线重连由 5B 的 `ReconnectingCameraSource` 提供；采集模式本身仍是
   有限帧数。
6. 采集帧目前只打印元数据，不落盘；后续可视化/推理接入时再处理。

## 7. 相关文件

- `include/camera_source.hpp`、`src/camera_source.cpp`
- `src/main.cpp`（`--camera` / `--capture-only` 入口）
- `tests/unit_tests.cpp`、`tests/cli_camera_tests.cmake`
- `docs/v4l2-capture-run.txt`
