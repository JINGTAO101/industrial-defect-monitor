# 任务 2B：C++ 单图推理

- 完成日期：2026-11-18
- 对应 SRS：FR-004～FR-007、AT-003～AT-005
- 模型：`neu_det_yolov8n_best.onnx`（YOLOv8n-NEU-DET，SHA256
  `495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1`）
- 结论：C++ 使用 ONNX Runtime 1.32.0 CPU ExecutionProvider 加载模型，完成
  letterbox 预处理、输出解析、类别感知 NMS 和原图坐标还原；六张固定 NEU-DET
  样例与 2A Python 参考结果一致（差异远小于记录容差）。模型/数据集许可仍为
  待确认事项，详见 `docs/model-verification.md`。

## 1. 环境与 ONNX Runtime C++ 安装

| 项目 | 值 |
| --- | --- |
| OS / 架构 | Ubuntu 24.04.5 LTS，x86_64 |
| CPU | AMD Ryzen 5 6600U，4 vCPU（VMware） |
| 编译器 | GCC/G++ 13.3.0 |
| CMake / Ninja | 3.28.3 / 1.11.1 |
| OpenCV | 4.6.0（`libopencv-dev`） |
| OpenSSL | 3.0.13（`libssl-dev`） |
| ONNX Runtime C++ | 1.32.0 CPU，`libonnxruntime.so.1.32.0` |

ONNX Runtime C++ CPU 包来源：

- Release：<https://github.com/microsoft/onnxruntime/releases/tag/v1.32.0>
- 文件：`onnxruntime-linux-x64-1.32.0.tgz`（13,425,244 bytes）
- SHA256：`bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503`
  （GitHub release digest 与实际下载一致）
- 本机解压路径：
  `$HOME/edge-ai-models/onnxruntime-1.32.0/onnxruntime-linux-x64-1.32.0`

安装命令：

```bash
mkdir -p "$HOME/edge-ai-models/onnxruntime-1.32.0"
cd "$HOME/edge-ai-models/onnxruntime-1.32.0"
curl -fL \
  https://github.com/microsoft/onnxruntime/releases/download/v1.32.0/onnxruntime-linux-x64-1.32.0.tgz \
  -o onnxruntime-linux-x64-1.32.0.tgz
sha256sum onnxruntime-linux-x64-1.32.0.tgz
# bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503
tar -xzf onnxruntime-linux-x64-1.32.0.tgz
```

> 该 tarball 自带的 CMake config 将 include 写为 `include/onnxruntime`、
> lib 写为 `lib64/`，与实际解压布局（`include/`、`lib/`）不一致。本工程
> 不使用 `find_package(onnxruntime)`，而是通过 `ONNXRUNTIME_ROOT` 手工查找
> `onnxruntime_cxx_api.h` 和 `libonnxruntime.so`。

其他依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake gdb git ninja-build pkg-config \
  libopencv-dev libssl-dev
```

## 2. 构建

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$HOME/edge-ai-models/onnxruntime-1.32.0/onnxruntime-linux-x64-1.32.0" \
  -DEDGE_AI_MODEL_PATH="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  -DEDGE_AI_TEST_IMAGES_DIR="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/test-images" \
  -DEDGE_AI_PYTHON="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/venv/bin/python"
cmake --build build
```

前三个 `EDGE_AI_*` 变量只用于启用可选集成测试；不设置时仍可构建和运行
单元/失败路径测试。`ONNXRUNTIME_ROOT` 是必需项。构建产物：

- `build/bin/edge_ai`
- `build/bin/edge_ai_tests`
- 编译选项：C++17、`-Wall -Wextra -Wpedantic`
- `edge_ai` 的 RUNPATH 只包含 ONNX Runtime `lib` 目录，并由 CMake 的
  `CMAKE_SKIP_RPATH` + 显式 `-rpath` 控制。

## 3. 单图运行入口

```bash
./build/bin/edge_ai \
  --model "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/neu_det_yolov8n_best.onnx" \
  --image "$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/test-images/crazing_101.jpg" \
  --conf 0.25 --iou 0.45 \
  --json-out build/verification/crazing_101.cpp.json \
  --dump-input build/verification/crazing_101.cpp.bin
```

`--help` 输出完整参数说明；`--dump-input` 写出 float32 NCHW 原始 Tensor，
供预处理对照。无 `--model/--image` 时仍打印启动横幅。参数错误返回 2，
模型/图片/推理错误返回 1。

参数解析规则：

- 缺值不会吞掉后续选项；例如 `--help --model --bogus` 返回 2，而不是把
  `--bogus` 当作模型路径后显示帮助。
- 以 `-` 开头的值必须写成 `--option=value`，例如
  `--model=-model.onnx`；空格分隔形式只接受不以 `-` 开头的值。
- `--json-out` 和 `--dump-input` 在写入前与模型、输入图片及彼此做路径冲突
  检查；相对路径、`..`、符号链接、悬空符号链接的目标及其链、硬链接都按
  同一文件处理。冲突时返回 2，且不会创建或截断任何文件。

实测输出（`crazing_101.jpg`）：

```text
ONNX Runtime 1.32.0 providers=CPUExecutionProvider
model sha256: 495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1
input:  images [1, 3, 640, 640] tensor(float)
output: output0 [1, 10, 8400] tensor(float)
classes: 0=crazing, 1=inclusion, 2=patches, 3=pitted_surface, 4=rolled-in_scale, 5=scratches
image: .../crazing_101.jpg 200x200
preprocess: letterbox ratio=3.2 pad=(0,0) BGR /255 NCHW
input tensor: [1, 3, 640, 640] float32 min=0.172549 max=0.996078
thresholds: conf>0.25 iou=0.45
detections: 2
  class_id=0 crazing conf=0.525822 bbox_xyxy=[48.0117, 37.2517, 195.252, 94.7151]
  class_id=0 crazing conf=0.484310 bbox_xyxy=[4.31551, 108.104, 180.438, 199.272]
```

## 4. 模型加载与接口校验

`ModelRunner::load` 使用 RAII 管理 `Ort::Env`、`Ort::SessionOptions`、
`Ort::Session` 和每次推理的 `Ort::Value`；会话只追加 CPU ExecutionProvider。

加载时执行以下校验，任何不匹配都拒绝推理并返回具体差异：

| 项目 | 期望值 | 检查位置 |
| --- | --- | --- |
| 输入/输出数量 | 1 / 1 | `model_contract.cpp` |
| 输入名称/形状/类型 | `images` / `[1,3,640,640]` / `tensor(float)` | 同上 |
| 输出名称/形状/类型 | `output0` / `[1,10,8400]` / `tensor(float)` | 同上 |
| 类别 metadata | 6 类，ID 0–5，名称与 2A 一致 | 同上 + 类别 metadata 解析 |
| SHA256 | 加载时计算并在输出/JSON 中记录 | `sha256.cpp` |

错误示例（原始 stderr，退出码均为 1）：

```text
edge_ai: model file not found: /tmp/does-not-exist.onnx
edge_ai: failed to load ONNX model 'tests/fixtures/corrupt_model.bin': Load model from tests/fixtures/corrupt_model.bin failed:Protobuf parsing failed.
edge_ai: model contract mismatch: input shape mismatch: expected [1, 3, 640, 640], got [1, 3, 320, 320]
edge_ai: model contract mismatch: output shape mismatch: expected [1, 10, 8400], got [1, 10, 100]
edge_ai: failed to read image: tests/fixtures/corrupt_model.bin
```

输出路径冲突（退出码 2，冲突文件不会被创建或截断）：

```text
edge_ai: output path '.../zero_output.onnx' conflicts with model path '.../zero_output.onnx'
edge_ai: output paths '.../same-output.bin' and '.../same-output.bin' refer to the same file
```

`cli_path_conflicts` 覆盖相对路径、`..`、符号链接、硬链接，以及
`alias -> 尚不存在的 result.json`（包括符号链接链）这种悬空链接场景；还覆盖
“父目录是符号链接 + 相对目标”的情况（`link -> real/sub`、
`real/sub/alias -> ../result.json`），确保先解析父目录再处理 `..`。测试确认
程序拒绝组合、目标文件没有被创建，并在测试后比较模型/图片的 SHA256，确认
原文件未被修改。

前两条和契约错误使用仓库内合成夹具
`tests/fixtures/{corrupt_model.bin,wrong_input.onnx,wrong_output.onnx}`；图片
读取失败使用真实模型 + 损坏图片。夹具由
`tools/generate_test_fixtures.py` 生成，不含第三方权重。

## 5. 预处理

严格遵循 2A 参考行为：

- OpenCV `IMREAD_COLOR` 读取为 **BGR**（不做 BGR→RGB 转换）；
- `ratio = 640 / max(W,H)`，`new_w=int(W*ratio)`、`new_h=int(H*ratio)`；
- `cv::resize(..., INTER_LINEAR)`；
- 用像素值 114 居中填充到 640×640；左右/上下分配与参考代码一致
  （左侧/上侧取整除，右侧/下侧取余数）；
- `float32 / 255.0`，无 mean/std；
- HWC → CHW → NCHW。

本次固定样例的输入 Tensor：

| 图片 | 尺寸 | ratio | pad (left, top) | Tensor 形状/类型 | 值域 |
| --- | --- | --- | --- | --- | --- |
| `crazing_101.jpg` | 200×200 | 3.2 | (0, 0) | `[1,3,640,640]` float32 | 0.172549–0.996078 |
| `non_square_320x200.png` | 320×200 | 2.0 | (0, 120) | `[1,3,640,640]` float32 | 0–1.0 |

非方形测试使用合成图 `tests/fixtures/non_square_320x200.png`：

- 320×200 → 640×400 内容 + 上下各 120 像素 114 填充；单元测试还检查
  200×320 → 左右各 120 填充；
- 单元测试验证填充位置、填充值 114/255、内容起始像素值和坐标还原；
- 与 Python 参考预处理的 Tensor 最大绝对差为 `5.960e-08`（仅 float32
  舍入），允许误差设为 `1e-6`。

比较方法：

```bash
# C++ 侧
./build/bin/edge_ai ... --image <image> --dump-input build/verification/x.bin

# Python 侧重算同样 letterbox+归一化，读取 raw float32 后逐元素比较
# 实际使用 tools/compare_cpp_reference.py；它同时比较六张图。
```

### 颜色顺序限制

当前明确遵循 2A 参考实现的 **BGR 直送**。NEU-DET 图片是灰度图（实测
B=G=R），所以本数据集上 BGR/RGB 无法区分。对真正的彩色输入，训练侧通常
使用 RGB，而本实现/参考代码按 BGR 输入，**通用彩色场景仍未验证**。2B 不据此
宣称彩色处理正确。

## 6. 输出解析、NMS 与类别映射

- 输出 `[1,10,8400]`，通道优先；每个预测前 4 个是 `cx,cy,w,h`（640 letterbox
  像素），后 6 个是 Sigmoid 类别分数。
- `confidence = max(scores)`，`class_id = argmax(scores)`；严格 `> conf`
  筛选；无 objectness。
- 类别分数必须是模型的 Sigmoid 概率 `[0, 1]`。任何 NaN/Inf 或超出此范围的
  值都会返回明确错误，且不产生任何检测；边界值 0 和 1 是合法的（0 通常低于
  置信度阈值，1 可以正常产生检测）。
- `cxcywh → xyxy` 后减去 padding、除以 ratio，再裁剪到原图尺寸。
- 按类别分别做 NMS（`cv::dnn::NMSBoxes`，IoU=0.45），类别 ID 升序、
  组内按分数降序。NMS 不在 ONNX 图内。
- 类别名从 ONNX metadata `names` 读取，不按名称顺序猜测。
- **FR-007**：`classNameFor` 对未知 ID 返回 `Unknown` 并追加 anomaly；
  不会静默映射到已有类别。当前模型契约校验会先拒绝类别数/输出通道不一致
  的模型，因此真实推理中未知 ID 是防御性路径；单元测试直接覆盖该行为。

## 7. 与 2A 参考结果对照

同一模型、六张图、`conf=0.25`、`IoU=0.45`，比较检测数量、类别、置信度和
原图 `bbox_xyxy`；同时比较 `--dump-input` 与 Python 预处理 Tensor。
`tools/compare_cpp_reference.py` 还会在开始前执行完整性守卫：

- 图片目录必须与参考结果中的六张图**完全一致**；缺图或多出 JPG 都会失败；
- 每张图片的实际 SHA256 必须与参考记录一致；
- `--conf`/`--iou` 必须与参考结果阈值一致；
- C++ JSON 中的模型 SHA256、CPU provider 和阈值也会再次核对。

因此“只放一张图”或“阈值不同”的情况不会被当作六图对照通过；
`cpp_reference_rejects_incomplete_inputs` 专门覆盖这两种拒绝路径。

| 图片 | 检测数 C++/参考 | 类别一致 | 最大置信度差 | 最大 bbox 差 | 最大预处理差 |
| --- | ---: | --- | ---: | ---: | ---: |
| `crazing_101.jpg` | 2/2 | 是 | 1.192e-06 | 1.907e-05 | 5.960e-08 |
| `inclusion_1.jpg` | 4/4 | 是 | 1.196e-07 | 1.146e-05 | 2.980e-08 |
| `patches_1.jpg` | 3/3 | 是 | 2.381e-07 | 9.555e-06 | 5.960e-08 |
| `pitted_surface_1.jpg` | 2/2 | 是 | 2.981e-07 | 3.436e-05 | 2.980e-08 |
| `rolled-in_scale_1.jpg` | 2/2 | 是 | 4.175e-07 | 7.630e-06 | 5.960e-08 |
| `scratches_1.jpg` | 2/2 | 是 | 1.162e-06 | 4.760e-06 | 5.960e-08 |

汇总：`max_conf_diff=1.192e-06`、`max_bbox_diff=3.436e-05`、
`max_blob_diff=5.960e-08`；记录容差分别为 `1e-5`、`1e-4`、`1e-6`，全部通过。

差异来源：参考结果由 Python/NumPy 生成，C++ 在 `bbox` 还原中使用 float32
运算；两边的 OpenCV 版本也不同（Python 4.10.0，C++ 4.6.0）。差异远小于
一个像素，属于可接受的数值误差。

运行对照：

```bash
"$VENV/bin/python" -I tools/compare_cpp_reference.py \
  --edge-ai build/bin/edge_ai \
  --model "$MODEL_DIR/neu_det_yolov8n_best.onnx" \
  --image-dir "$MODEL_DIR/test-images" \
  --reference-results docs/reference-inference-results.json \
  --out-dir build/reference-comparison
```

实际输出：

```text
crazing_101.jpg: OK: detections cpp/ref=2/2 class_ok=True conf_max_diff=1.192e-06 bbox_max_diff=1.907e-05 blob_max_diff=5.960e-08 (tolerances conf=1e-05 bbox=0.0001 blob=1e-06)
...
SUMMARY: images=6 failures=0 max_conf_diff=1.192e-06 max_bbox_diff=3.436e-05 max_blob_diff=5.960e-08
```

## 8. 测试

```bash
ctest --test-dir build --output-on-failure
```

当前配置（提供真实模型和图片路径）共 60 个测试全部通过：

| 测试 | 内容 |
| --- | --- |
| `unit_preprocess_non_square` | 横向/纵向非方形 letterbox、填充位置和值 |
| `unit_postprocess_empty` | 无候选返回空列表，不报错、不伪造 |
| `unit_postprocess_nms_same_class` | 同类重叠框抑制，保留高分和远距离框 |
| `unit_postprocess_nms_cross_class` | 不同类别重叠框不被互相抑制 |
| `unit_postprocess_coordinate_restore` | pad/ratio 坐标还原 |
| `unit_postprocess_nonfinite` | NaN 输出被拒绝并给出诊断 |
| `unit_postprocess_score_range` | 分数 >1/<0 被拒绝；边界 0/1 合法 |
| `unit_postprocess_shape_mismatch` | 输出尺寸错误被拒绝 |
| `unit_class_mapping_unknown` | 未知 ID → `Unknown` + anomaly |
| `unit_contract_validation` | 名称/形状/类型/数量/类别不匹配诊断 |
| `unit_class_metadata_parser` | names metadata 解析与重复/断号拒绝 |
| `cli_behavior` | 帮助、非法参数、无参数横幅 |
| `cli_inference_failures` | 缺失/损坏模型、错误输入/输出契约、损坏图片 |
| `cli_empty_detections` | 合法契约、全零输出模型端到端返回 0 个检测 |
| `cli_path_conflicts` | 输出路径指向模型/图片/彼此（含相对路径/悬空符号链接/硬链接）被拒绝，原文件 SHA256 不变且悬空目标未被创建 |
| `cpp_reference_comparison` | 六图 C++ 与 2A 参考对照（可选配置） |
| `cpp_reference_rejects_incomplete_inputs` | 只有一张图时必须失败；阈值不一致时必须失败（可选配置） |

不提供 `EDGE_AI_MODEL_PATH` 等变量时，两个 `cpp_reference_*` 测试不会注册；
其余 57 个测试仍可运行。失败路径的原始输出见第 4 节。任务 3 新增的质量判定、图片源和批量测试见
`docs/input-and-quality.md`。

测试夹具：

- `tests/fixtures/non_square_320x200.png`：合成 320×200 RGB 图；
- `tests/fixtures/wrong_input.onnx`：可加载但输入形状错误（合成）；
- `tests/fixtures/wrong_output.onnx`：可加载但输出形状错误（合成）；
- `tests/fixtures/zero_output.onnx`：合法契约、输出全零（合成），用于空检测；
- `tests/fixtures/corrupt_model.bin`：故意损坏的字节；
- 全部由 `tools/generate_test_fixtures.py` 生成，不包含预训练权重或第三方图片。

## 9. 尚未完成 / 待验证

1. **许可**：模型/权重 AGPL-3.0 与 NEU-DET 学术研究限制仍未解决；商业使用、
   闭源分发和测试图再分发未获授权。详见 `docs/model-verification.md`。
2. **彩色输入**：BGR 直送是参考行为；真实彩色场景的 RGB/BGR 正确性未验证。
3. **精度**：没有复测 mAP/误检率；本任务只验证接口、固定样例结果和失败路径。
4. **退化框**：C++ 在坐标还原后跳过 `x2<=x1` 或 `y2<=y1` 的框并记录 anomaly；
   参考 Python 代码没有该显式跳过。六张样例未触发该差异，密集/异常输出场景
   尚未系统测试。
5. **未知类别**：实现按 FR-007 返回 `Unknown` + anomaly，但当前严格契约会先
   拒绝类别数/通道不匹配的模型；真实模型上的未知 ID 路径未出现。
6. **NMS 覆盖**：只验证了六张固定图和构造的重叠框；不同阈值、极端密集框、
   `top_k/eta` 等行为未测。
7. **性能**：未设置性能目标，也未做系统基准；仅记录功能验证。
8. **未包含**：摄像头、多线程、SQLite、MQTT。
9. **可选集成测试**：需要外部模型、六张测试图和 Python venv；这些资产不提交，
   干净克隆只能默认运行 57 个不依赖真实模型的测试。

## 10. 相关文件

- 推理封装：`include/model_runner.hpp`、`src/model_runner.cpp`
- 契约校验：`include/model_contract.hpp`、`src/model_contract.cpp`
- 预处理：`include/image_preprocess.hpp`、`src/image_preprocess.cpp`
- 后处理/NMS：`include/postprocess.hpp`、`src/postprocess.cpp`
- SHA256：`include/sha256.hpp`、`src/sha256.cpp`
- 单图 CLI：`src/main.cpp`
- 测试：`tests/unit_tests.cpp`、`tests/cli_inference_tests.cmake`、
  `tests/cli_empty_detections.cmake`、`tests/cli_path_conflict_tests.cmake`、
  `tests/compare_reference_guard_tests.cmake`、`tools/compare_cpp_reference.py`
- 2A 参考记录：`docs/model-verification.md`
- 本次干净构建、CTest、单图推理、失败路径和六图对照的原始运行日志：
  `docs/cpp-inference-run.txt`（路径已脱敏为 `$MODEL_DIR` / `$ORT_DIR`）
