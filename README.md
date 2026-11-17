# Edge AI 工业缺陷检测系统

面向边缘设备的工业表面缺陷检测系统。当前已完成：

- **任务 1 / 2A**：C++17 工程骨架、YOLOv8n-NEU-DET 模型来源/许可/接口核验、
  Python 参考推理；
- **任务 2B**：基于 ONNX Runtime C++ API 和 CPU ExecutionProvider 的单图推理，
  包括模型契约校验、letterbox 预处理、类别感知 NMS、坐标还原、失败路径测试，
  以及与 2A 参考结果的逐项对照。

摄像头、多线程、SQLite、MQTT 不在当前阶段范围内。

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

# 非法参数：stderr 报错，退出码 2
./build/bin/edge_ai --bogus
```

主要参数：`--model`、`--image`、`--conf`（默认 0.25）、`--iou`（默认 0.45）、
`--json-out`、`--dump-input`。参数会先完整校验；`--help --bogus` 等组合仍会
返回非零。模型/图片/推理失败返回 1。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

不提供真实模型路径时运行 13 个测试（单元测试 + CLI 失败路径 + 空检测）；
提供 `EDGE_AI_MODEL_PATH`、`EDGE_AI_TEST_IMAGES_DIR` 和 `EDGE_AI_PYTHON` 时
额外运行 `cpp_reference_comparison`，共 14 个。本次验证 14/14 通过。

`cli_inference_failures` 覆盖：

- 模型文件不存在；
- 模型损坏/无法加载；
- 输入形状不符合已核验契约；
- 输出形状不符合已核验契约；
- 图片无法读取（需要真实模型）。

单元测试覆盖非方形 letterbox、空候选、同类/跨类 NMS、坐标还原、非有限输出、
输出尺寸错误、未知类别 `Unknown`+anomaly、契约校验和类别 metadata 解析。
`cli_empty_detections` 使用合法契约、输出全零的合成 ONNX 模型，端到端确认
无候选时返回 `detections: 0` 且退出码为 0。

## 目录结构

```text
.
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/
│   ├── image_preprocess.hpp
│   ├── model_contract.hpp
│   ├── model_runner.hpp
│   ├── postprocess.hpp
│   ├── sha256.hpp
│   └── system_info.hpp
├── src/
│   ├── image_preprocess.cpp
│   ├── main.cpp
│   ├── model_contract.cpp
│   ├── model_runner.cpp
│   ├── postprocess.cpp
│   ├── sha256.cpp
│   └── system_info.cpp
├── docs/
│   ├── model-verification.md          # 2A 模型来源/许可/接口/参考推理
│   ├── cpp-inference.md               # 2B C++ 推理与对照记录
│   ├── cpp-inference-run.txt          # 2B 干净构建/CTest/推理/失败路径运行日志
│   ├── reference-inference-results.json
│   └── reference-inference-run.txt
├── tools/
│   ├── neu_det_reference.py
│   ├── compare_author_predict.py
│   ├── compare_cpp_reference.py       # C++ 与 2A 参考结果对照
│   ├── reproduce_reference_verification.sh
│   ├── generate_test_fixtures.py
│   └── requirements-reference.txt
└── tests/
    ├── cli_tests.cmake
    ├── cli_inference_tests.cmake
    ├── cli_empty_detections.cmake
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

## 验收记录（2B）

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
100% tests passed, 0 tests failed out of 14

$ ./build/bin/edge_ai --model ... --image .../crazing_101.jpg \
    --conf 0.25 --iou 0.45
ONNX Runtime 1.32.0 providers=CPUExecutionProvider
input:  images [1, 3, 640, 640] tensor(float)
output: output0 [1, 10, 8400] tensor(float)
...
detections: 2
  class_id=0 crazing conf=0.525822 bbox_xyxy=[48.0117, 37.2517, 195.252, 94.7151]
  class_id=0 crazing conf=0.484310 bbox_xyxy=[4.31551, 108.104, 180.438, 199.272]

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
