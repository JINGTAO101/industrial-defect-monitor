# 任务 3：图片输入抽象与 OK/NG 判定

- 完成日期：2026-11-18
- 对应 SRS：FR-001、FR-003、FR-008；AT-002、AT-006
- 结论：单图与目录批量输入都通过统一的 `ImageSource` 接口提供 `Frame`；推理
  层不依赖具体图片源。判定规则为“至少一个检测的置信度严格大于阈值则 NG，
  否则 OK”。非法阈值、缺失/损坏/不支持的图片都有明确错误路径。

## 1. 抽象接口

### Frame

`include/image_source.hpp`：

```cpp
using FrameTimestamp = std::chrono::system_clock::time_point;

struct Frame {
    cv::Mat image;       // 成功加载后的 BGR 8UC3 图像
    std::string source;  // 文件路径
    std::size_t index;   // 输入序号
    FrameTimestamp timestamp;  // 成功解码时的墙钟时间
};
```

### ImageSource

```cpp
enum class SourceStatus { Frame, End, Error };

class ImageSource {
public:
    virtual SourceStatus next(Frame& frame, std::string& error) = 0;
    virtual std::size_t skippedUnsupported() const { return 0; }
};
```

具体实现：

| 类 | 用途 | 行为 |
| --- | --- | --- |
| `SingleImageSource` | 单张图片 | 构造时检查存在性、普通文件、扩展名；`next` 加载一次后返回 `End` |
| `DirectoryImageSource` | 目录批量 | 字典序枚举支持扩展名的普通文件；逐个加载，坏文件返回 `Error`，下一次调用继续 |

`timestamp` 使用 `std::chrono::system_clock`，语义为“该图片成功解码、进入
软件处理边界时的墙钟时间”，不是 EXIF 拍摄时间。单图和批量源都会在成功
加载时填充；失败帧没有可交付的 `Frame`。宽、高、通道数不重复存放，调用方
使用 `frame.image.cols`、`rows`、`channels()` 获取。

`BatchProcessor`（`include/batch_processor.hpp`）只接收 `ImageSource&` 和
`ModelRunner&`，因此推理/批处理代码不依赖 `SingleImageSource` 或
`DirectoryImageSource` 的具体实现。

## 2. 支持格式与错误行为

支持扩展名：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`（大小写不敏感，
与 2A 参考一致）。

### 单图

| 情况 | 行为 |
| --- | --- |
| 路径不存在 | 构造 `SingleImageSource` 失败，stderr 报 `image file not found: ...`，退出码 1 |
| 扩展名不支持 | 报 `unsupported image extension: ...`，退出码 1 |
| OpenCV 无法解码（损坏） | `next` 返回 `Error`，报 `failed to read image: ...`，退出码 1 |
| 解码过程抛出 C++/OpenCV 异常 | 在图片源内捕获并转为 `Error`，不终止进程 |
| 正常 | 返回一个 `Frame`，继续预处理/推理 |

### 批量目录

- 只枚举支持扩展名的普通文件，按路径字典序处理；其他文件（如 `.xml`、
  `.txt`）被跳过，并在 `skippedUnsupported()` / 批量摘要中计数。
- 目录不存在、不是目录或没有支持图片：明确报错，退出码 1。
- 目录迭代初始化/递增失败或条目元数据读取失败：报明确目录枚举错误，不会
  误报为“没有支持的图片”。
- 单个文件损坏（`SourceStatus::Error`）：记录该帧 `[ERROR]`，**继续处理
  后续文件**；最终退出码为 1。
- 解码/OpenCV 异常在图片源内部被捕获并转换为 `SourceStatus::Error`，因此
  正常坏图仍按“继续”处理。
- 如果异常逃逸出 `ImageSource::next()`，说明无法确认源是否已推进：批处理
  立即视为**致命错误**停止，不重试、不继续，也不会无限追加错误。
- 所有文件处理成功（包括 OK 和 NG）：退出码 0。
- 默认策略是“坏图继续并汇报”；源级异常是 fatal；这一行为已写入 README 和测试。

## 3. Detection 与 QualityResult

`Detection` 沿用 2B 后处理定义（`include/postprocess.hpp`）：

```cpp
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    float x1, y1, x2, y2;
};
```

`include/quality.hpp`：

```cpp
struct QualityResult {
    bool is_ng = false;
    float confidence_threshold = 0.0f;
    std::size_t total_detections = 0;
    std::vector<Detection> defects;  // confidence > threshold
};
```

批处理结果中的 `FrameResult` 还有 `quality_available` 标志：只有完成推理和
判定的帧才为 `true`；坏图/推理失败的帧为 `false`，其 `quality` 不应被读取。
批量 JSON 对应输出 `quality: null`。

判定规则：

- 阈值由 `--conf` 配置，默认 `0.25`；
- 对每个检测检查 `confidence > confidence_threshold`（严格大于，与 2B
  后处理一致）；
- 至少一个满足条件的检测 → `is_ng=true`，状态 `NG`；
- 没有满足条件的检测（包括无检测、全部低于/等于阈值）→ `OK`；
- 阈值必须是有限值且位于 `(0, 1]`；`0`、负数、`>1`、NaN、Inf 均返回明确
  错误，不产生判定结果，不会静默使用默认值。

## 4. CLI

```bash
# 单图
./build/bin/edge_ai --model model.onnx --image image.jpg \
  --conf 0.25 --iou 0.45 --json-out single.json

# 批量目录
./build/bin/edge_ai --model model.onnx --image-dir images/ \
  --conf 0.25 --iou 0.45 --json-out batch.json
```

- `--image` 与 `--image-dir` 互斥，必须二选一。
- `--dump-input` 仅支持 `--image`；与 `--image-dir` 组合会返回 2。
- `--json-out`：单图写单图结果 JSON；批量写汇总 JSON（每帧状态、错误、
  检测和 OK/NG）。
- 批量输出示例：

```text
[NG] images/a.jpg detections=2
[OK] images/b.jpg detections=0
[ERROR] images/bad.jpg: failed to read image: images/bad.jpg
summary: total=3 ok=1 ng=1 errors=1 skipped_unsupported=2
```

单图输出新增：

```text
quality: NG (confidence>0.25, qualifying=2/2)
```

JSON 中对应字段：

```json
"quality": {
  "status": "NG",
  "is_ng": true,
  "confidence_threshold": 0.25,
  "total_detections": 2,
  "qualifying_defects": 2
}
```

批量 JSON 的顶层结构包含 `model`、`thresholds`、`quality_rule`、`frames`
和 `summary`；每个 frame 含 `source`、`status`（`OK`/`NG`/`ERROR`）、
`error`、`width`、`height`、`letterbox`、`detections`、`anomalies`、`quality`。

## 5. 测试

```bash
ctest --test-dir build --output-on-failure
```

质量与输入相关的构造测试：

| 测试 | 内容 |
| --- | --- |
| `unit_quality_no_defects` | 无检测 → OK |
| `unit_quality_single_defect` | 单个高于阈值 → NG |
| `unit_quality_below_threshold` | 单个低于阈值 → OK |
| `unit_quality_multiple_defects` | 多个高于阈值 → NG |
| `unit_quality_boundary` | 恰好等于阈值 → OK（严格 `>`） |
| `unit_quality_invalid_threshold` | 0、负数、>1、NaN、Inf → 明确错误 |
| `unit_image_source_single` | 单图正常/缺失/不支持/损坏 |
| `unit_image_source_directory` | 批量正常、坏文件继续、扩展名跳过、缺失/空目录、读取时间戳 |
| `batch_source_exception` | 正常 `SourceStatus::Error` 后继续下一帧；持续抛异常的源在第一次异常后致命停止（不重试） |
| `cli_batch` | 全 OK、坏文件继续、跳过不支持文件、批量 JSON（失败帧 `quality: null`）、输出路径冲突、`--dump-input` 拒绝；有真实模型时验证六图全 NG |
| `cli_behavior` | 单图缺失/不支持/损坏、非法阈值、单图 OK |

本次配置（真实模型 + 六张图片）**82/82 通过**；不提供真实模型/图片时，
79 个不依赖外部资产的测试通过。完整运行记录见
`docs/cpp-inference-run.txt`。

## 6. 待确认 / 限制

1. OK/NG 规则是 SRS 的暂定规则；默认阈值 `0.25` 是 2A 参考默认，不代表
   经过生产验证的质量阈值。
2. 批量对正常坏图继续并汇总；对无法确认已推进的源异常立即停止；没有自动
   重试、断点续跑或并发处理。
3. 批量只处理固定扩展名；其他文件跳过并计数，不把它们当作图片错误。
4. 彩色输入 BGR/RGB、模型精度、AGPL-3.0/NEU-DET 许可等限制与 2A/2B 相同，
   仍未解决。
5. 摄像头、多线程、SQLite、MQTT 不在本任务范围。

## 7. 相关文件

- 接口与源：`include/image_source.hpp`、`src/image_source.cpp`
- 质量判定：`include/quality.hpp`、`src/quality.cpp`
- 批处理：`include/batch_processor.hpp`、`src/batch_processor.cpp`
- CLI：`src/main.cpp`
- 测试：`tests/unit_tests.cpp`、`tests/cli_tests.cmake`、
  `tests/cli_batch_tests.cmake`
