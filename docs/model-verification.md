# 任务 2A 模型核验记录：YOLOv8n-NEU-DET

- 核验日期：2026-11-18
- 核验对象：`neu_det_yolov8n_best.onnx`（YOLOv8n，NEU-DET 六类缺陷检测）
- 结论：模型可下载、SHA256 可核对、可在 Ubuntu x86-64 的 ONNX Runtime CPU
  ExecutionProvider 上加载并完成单图推理；参考推理与作者原始后处理逐项一致。
  许可方面仅确认了研究/学术使用条件，**商业使用与再分发尚未确认**（见第 2 节）。

## 1. 来源与版本

| 项目 | 记录 |
| --- | --- |
| 模型仓库 | <https://github.com/winthegame259/steel-surface-defect-detection> |
| 版本 | tag `v1.0.0`，release 发布 2026-08-31T10:44:37Z |
| 对应 commit | `1226b2d8a241b3ecacb61d9e727051ccb8baa64e` |
| ONNX 下载地址 | <https://github.com/winthegame259/steel-surface-defect-detection/releases/download/v1.0.0/neu_det_yolov8n_best.onnx> |
| ONNX 文件名 | `neu_det_yolov8n_best.onnx` |
| 文件大小 | 12,242,104 bytes |
| 本地 SHA256 | `495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1` |
| GitHub release 记录 digest | `sha256:495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1`（一致） |
| PyTorch 权重（未使用） | `neu_det_yolov8n_best.pt`，6,281,635 bytes，release 记录 SHA256 `80450b10c9dc9dfbfa716f3c72073319f7de4b0419544e57d089c51ac3e90a11`；仅记录 release 元数据，未下载核验 |
| ONNX 导出方 | 仓库作者在 release 中直接提供；**不是本任务导出的**，因此没有本地的 `ultralytics export` 命令记录 |

下载与校验命令：

```bash
MODEL_DIR="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0"
mkdir -p "$MODEL_DIR"
curl -fL \
  https://github.com/winthegame259/steel-surface-defect-detection/releases/download/v1.0.0/neu_det_yolov8n_best.onnx \
  -o "$MODEL_DIR/neu_det_yolov8n_best.onnx"
sha256sum "$MODEL_DIR/neu_det_yolov8n_best.onnx"
# 495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1
```

ONNX 内部元数据（ONNX Runtime 1.32.0 实际读取）：

| 键 | 值 |
| --- | --- |
| producer | `pytorch` |
| author | `Ultralytics` |
| version | `8.2.98` |
| task | `detect` |
| imgsz | `[640, 640]` |
| batch | `1` |
| stride | `32` |
| license | `AGPL-3.0 License (https://ultralytics.com/license)` |
| description | `Ultralytics best model trained on data/processed/neu_det_yolo/data.yaml` |
| date | `2026-08-31T17:31:47.455530` |
| opset | `ai.onnx` 12（IR version 7） |
| 图节点数 | 259 |
| 参数元素数 | 3,006,818（全部 float32，按 ONNX initializer 统计） |

> 溯源待确认：仓库 `docs/experiment_log.md` 的 EXP-006 记录 ONNX 大小为
> 42.7 MB（非简化版），而 v1.0.0 release 的实际文件为 12.2 MB。两者可能对应
> 不同模型（v8s / v8n）或不同导出批次；作者未给出 release 文件对应的精确
> 导出命令。本记录的全部接口与推理结论均基于上述 SHA256 的 release 文件，
> 不依赖日志中的 42.7 MB 文件。

本次用于阅读/对照的参考源码文件（均取自 commit `1226b2d8...`）：

| 文件 | SHA256 |
| --- | --- |
| `src/predict.py` | `84dfc96d0a8bfe9a1b4474b32a2e53c7e94b4ebba2f615269bc5f938b930c130` |
| `cpp_inference/inference.cpp` | `a2282c5ab191495b6f8b4f670204582ba36714e4cc541e9856c647b7301a23da` |
| `configs/neu_det.yaml` | `e1ffe1289d4e4fffba798a2d727cf2ca94602a5966da9b5323c3a8b1b0e49e12` |
| `README.md` | `106ce2cabe37b75e8a5241fced68ec1a83ceea91035e8b98d42f0e4ee8dd138e` |

## 2. 许可证和使用范围

| 对象 | 来源与证据 | 记录 | 结论 |
| --- | --- | --- | --- |
| 仓库代码 | 仓库根目录 `LICENSE`；README「License / 引用」第 222 行 | AGPL-3.0 | 代码分发/修改需遵守 AGPL-3.0 |
| ONNX/PyTorch 权重 | 仓库 `LICENSE`；ONNX metadata `license` 字段 | AGPL-3.0（未发现单独的权重许可文件） | 按仓库与模型 metadata 记录为 AGPL-3.0；商业闭源分发未获确认 |
| NEU-DET 数据集 | README 第 223 行写明“仅用于学术研究”；东北大学数据集主页只提出引用请求，未给出明确许可证 | 学术研究使用；商业使用/再分发**待确认** | 不得当作已获商业授权 |
| 测试图片 | 来自公开 GitHub 镜像 `siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`（无 license 文件）；原始数据版权仍属东北大学 | 仅用于本次非商业核验；图片不提交到仓库 | 再分发/商用**待确认** |

许可证原文位置：

- 仓库 LICENSE：<https://github.com/winthegame259/steel-surface-defect-detection/blob/1226b2d8a241b3ecacb61d9e727051ccb8baa64e/LICENSE>
- README「License / 引用」：<https://github.com/winthegame259/steel-surface-defect-detection/blob/1226b2d8a241b3ecacb61d9e727051ccb8baa64e/README.md#L220-L224>
- 东北大学数据集主页（引用要求）：<http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm>
- 数据集引用：K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects recognition,” *Applied Surface Science*, vol. 285, pp. 858–864, 2013.

> 补充调查：有二次来源（例如 PLOS ONE 2025 的图注）将 NEU-DET 描述为
> CC BY 4.0 并称获得作者许可，但东北大学官方页面没有许可证文本，镜像也没有
> license 文件。本记录不据此认定为 CC BY 4.0，仍按“商用/再分发待确认”处理。
> 参考：<https://journals.plos.org/plosone/article/figure?id=10.1371/journal.pone.0359400.g002>

**使用边界（当前结论）**：本阶段仅将模型和图片用于学术/内部技术核验。若项目需要
商业部署、闭源分发或对外提供服务，必须先解决 AGPL-3.0 义务（或取得
Ultralytics 商业许可）以及 NEU-DET 数据集的商用授权；这两项均**未完成**。

## 3. 模型基本信息

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| 模型类型 | YOLOv8n 目标检测模型 | 仓库 README、ONNX metadata `version: 8.2.98` / `task: detect` |
| 检测对象 | 热轧钢带表面六类缺陷，输入为单张图像 | 仓库 README；NEU-DET 数据集说明 |
| 训练数据集 | NEU-DET（东北大学），1800 张 200×200 灰度图，6 类 | 仓库 README 第 42–57 行；ONNX metadata `description` |
| 类别数 | 6 | ONNX metadata `names`（见第 5 节） |
| 参数量 | 3,006,818 个 float32 元素 | 本次用 `onnx` 1.23.2 统计 initializer |
| 输入尺寸 | 固定 640×640 | 本次 ONNX Runtime 检查；metadata `imgsz` |
| ONNX 内是否包含 NMS | **否** | 图中无 `NonMaxSuppression` 节点；`onnx` 图检查结果 |
| 作者报告的精度/速度 | README：test mAP50 = 0.774、CPU 前向 14.6 ms | 作者声明；本次未复测精度，速度见第 6 节（硬件不同） |

## 4. 输入契约

以下全部来自实际模型检查与作者在 tag `v1.0.0` 的参考代码，不是按“YOLO
通常如此”推断。

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| 输入 Tensor 名称 | `images`，共 1 个输入 | ONNX Runtime `session.get_inputs()` |
| 形状 | `[1, 3, 640, 640]`，固定尺寸（无动态维） | ONNX Runtime 报告 |
| 数据类型 | `tensor(float)`（float32） | ONNX Runtime 报告 |
| 布局 | NCHW | `src/predict.py` 第 81–83 行；`cpp_inference/inference.cpp` 第 121–136 行 |
| 颜色顺序 | 参考实现按 **BGR** 输入：`cv2.imread` 后直接 letterbox，没有 `cvtColor` | `src/predict.py` 第 52、67–84 行；`cpp_inference/inference.cpp` 第 119–131 行 |
| 尺寸调整 | 等比例缩放 + 居中填充（letterbox）：`ratio = 640 / max(W,H)`；`new_w = int(W*ratio)`、`new_h = int(H*ratio)`；`INTER_LINEAR`；用像素值 114 填充到 640×640；`left = (640-new_w)//2`、`top = (640-new_h)//2`，右侧/下侧取余数 | `src/predict.py` 第 67–84 行；`cpp_inference/inference.cpp` 第 54–73 行 |
| 归一化 | 仅 `float32 / 255.0`；**没有**减均值或除标准差 | `src/predict.py` 第 81 行；`cpp_inference/inference.cpp` 第 123 行 |
| Batch | 固定为 1 | ONNX 输入形状第 0 维为 1；metadata `batch: 1` |

### 颜色顺序说明

参考代码确实把 OpenCV 读出的 BGR 三通道直接送入模型，没有转换成 RGB。
NEU-DET 原图是灰度图（虽然 JPEG 存为三通道），本次实测六张图均满足
`B == G == R`（最大通道差为 0），因此对这些图片 BGR/RGB 的数值完全相同，
无法从本数据集区分二者的影响。

- 对 NEU-DET 灰度图：按参考实现使用 BGR 即可。
- 对真正的彩色输入：参考实现仍是 BGR 直送，但 Ultralytics 训练侧通常使用
  RGB；颜色顺序在彩色图像上的正确性**待验证**，2B 集成时不应默认二者等价。

## 5. 输出契约与六类标签

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| 输出 Tensor 名称 | `output0`，共 1 个输出 | ONNX Runtime 报告 |
| 形状 | `[1, 10, 8400]`（float32） | ONNX Runtime 报告；`cpp_inference/inference.cpp` 第 145–149 行 |
| 布局 | 通道优先：`[batch, 4+nc, num_preds]` | 参考代码先 `output[0].T`；C++ 用 `raw[c*num_preds + i]` |
| 10 的含义 | 4 个边界框参数 + 6 个类别分数 | 6 类来自 metadata `names` |
| 8400 的含义 | 640 输入下三个检测头的锚点数：80×80 + 40×40 + 20×20 = 8400 | metadata `stride: 32`；YOLOv8 三尺度检测头 |
| 边界框表示 | 每个预测为 `cx, cy, w, h`，单位是 **640×640 letterbox 输入图像的像素**，不是归一化坐标 | 原始输出 `cx` 范围 4.46–634.42、`w` 范围 8.92–625.73；参考代码直接按像素处理 |
| 类别分数 | 6 个 Sigmoid 概率，每类一个；**没有单独的 objectness 分数** | 图末尾算子为 `Sigmoid` + `Concat`；参考代码 `scores = output[:, 4:]` |
| 置信度计算 | `confidence = max(scores)`，`class_id = argmax(scores)` | `src/predict.py` 第 99–108 行 |
| NMS 是否已包含 | **否** | ONNX 图无 `NonMaxSuppression`；图末尾只有 `Sigmoid`/`Concat` |
| 参考 NMS | 类别感知 NMS：按类别分别调用 `cv2.dnn.NMSBoxes`；默认 `conf=0.25`、`IoU=0.45` | `src/predict.py` 第 136–151 行；函数默认参数第 21–22 行 |
| 先验筛选 | `max_score > conf_thres`（严格大于） | `src/predict.py` 第 105 行 |
| NMS 顺序 | 按类别 ID 升序分组，组内按置信度降序保留 | `src/predict.py` 第 138–151 行 |
| 原图坐标还原 | `x1=(cx-w/2-pad_left)/ratio`，`y1=(cy-h/2-pad_top)/ratio`，`x2=(cx+w/2-pad_left)/ratio`，`y2=(cy+h/2-pad_top)/ratio`；再裁剪到 `[0,W]×[0,H]` | `src/predict.py` 第 113–131 行；`cpp_inference/inference.cpp` 第 170–179 行 |

### 一个输出 Tensor 如何变成“类别 + 置信度 + 原图边界框”

1. 取 `output0[0]`，得到 `[10, 8400]`；转置为 `[8400, 10]`。
2. 每行前 4 个值是 `cx, cy, w, h`（640 输入像素坐标），后 6 个值是六类
   Sigmoid 概率。
3. 对每行取最大类别分数和对应类别 ID；保留 `max_score > 0.25` 的候选。
4. 由 `cx,cy,w,h` 计算 `xyxy`，减去 letterbox 的 `pad_left/pad_top`，再除以
   `ratio`，得到原图坐标，并裁剪到图像边界。
5. 按类别分别做 IoU=0.45 的 NMS。
6. 用模型 metadata 的 `names` 把类别 ID 映射为名称。核验脚本
   `tools/neu_det_reference.py` 对未知 ID 采用 fail-fast（抛出异常），
   以确保核验不会静默错映射；**产品侧 C++ 必须按 SRS FR-007 处理：
   显示 `Unknown` 并记录异常**，或先完成需求变更再调整。两种方式都
   不得把未知 ID 静默映射成已有类别。

### 六类标签映射（ID 顺序来自模型 metadata）

| 类别 ID | 模型原始名称 | 中文含义（本次翻译，非模型字段） | 证据 |
| --- | --- | --- | --- |
| 0 | `crazing` | 龟裂/裂纹 | ONNX metadata `names`；`configs/neu_det.yaml`；`src/predict.py` 第 189 行 |
| 1 | `inclusion` | 夹杂物 | 同上 |
| 2 | `patches` | 斑块 | 同上 |
| 3 | `pitted_surface` | 麻点表面 | 同上 |
| 4 | `rolled-in_scale` | 氧化皮压入 | 同上 |
| 5 | `scratches` | 划痕 | 同上 |

注意：SRS 中的 `Crazing`、`Inclusion`、`Patches`、`Pitted Surface`、
`Rolled-in Scale`、`Scratches` 与以上名称一一对应，但大小写/下划线与模型
原始字段不同；集成时以模型 metadata 的 ID 顺序为准，不能只按名称顺序猜测。

## 6. 参考推理环境与固定图片

### 环境

| 项目 | 实际值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.5 LTS，x86_64 |
| CPU | AMD Ryzen 5 6600U，4 vCPU（VMware 虚拟机） |
| 内存 | 5.7 GiB |
| 推理后端 | ONNX Runtime **1.32.0**，`CPUExecutionProvider`（无 CUDA） |
| Python | 3.12.3 |
| ONNX | 1.23.2 |
| NumPy | 2.5.5 |
| OpenCV | opencv-python-headless 4.10.0.84 |
| 模型加载 | 成功，输入/输出形状与 metadata 均按上述结果读取 |

模型加载与 CPU provider 的实际输出：

```text
ONNX Runtime 1.32.0
available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
session providers: ['CPUExecutionProvider']
input:  images [1, 3, 640, 640] tensor(float)
output: output0 [1, 10, 8400] tensor(float)
```

### 固定测试图片

主参考图选用 `crazing_101.jpg`（与作者实验日志中 C++/Python 对照使用的图片
同名），来源为公开 GitHub 镜像上的 NEU-DET 副本：

| 项目 | 值 |
| --- | --- |
| 文件 | `crazing_101.jpg` |
| 尺寸 / 大小 | 200×200，22,537 bytes |
| SHA256 | `0c4ccbfd539aa9d0a151f25b3e798fbad36d2a3532218bcc5c313f06c2c1dda4` |
| 镜像仓库 | <https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection> |
| 镜像 commit | `351bee21b56bbf97a1e1d18262bceb202eb7ef41` |
| 图片 URL | <https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/blob/351bee21b56bbf97a1e1d18262bceb202eb7ef41/IMAGES/crazing_101.jpg> |
| 原始来源/引用 | NEU-DET，东北大学；引用同第 2 节 |
| 许可状态 | 镜像无 license；原始数据仅确认学术研究用途，再分发/商用待确认 |
| 仓库策略 | 图片仅保存在仓库外，**不提交** |

补充用图（同一镜像与 commit）：

| 文件 | SHA256 |
| --- | --- |
| `scratches_1.jpg` | `500c46e2f5d7065d8c492b0cfb36c3ba77a85a744079bf6453dfc85e02b97a3c` |
| `patches_1.jpg` | `0ea4843aa7801ce5a5e672ec8c0266d4910aa7d740955f38d087b38154d0fe67` |
| `inclusion_1.jpg` | `2b0a6eff755d5c5867d3e3ca6b26af9548813e50c9b12b0f3381a49560a42634` |
| `pitted_surface_1.jpg` | `38a0ece141350f5bed53c99da4d96ce25d42b3ed4248e536a579ab823abfb318` |
| `rolled-in_scale_1.jpg` | `e9d0358a2859da86a7a9cc75deb5c235fcb8b14e8773cd92e94593b0289736fd` |

## 7. 复现命令

以下命令使用仓库内已提交的 `tools/neu_det_reference.py`。该脚本按作者
`src/predict.py` 的预处理/后处理逻辑实现，类别名直接从 ONNX metadata 读取；
并已用 `tools/compare_author_predict.py` 与作者原函数逐项对比（见第 8 节）。

### 7.0 一键复现（推荐）

`tools/reproduce_reference_verification.sh` 会从空的 `MODEL_DIR` 开始：

1. 下载模型、六张测试图和作者 `src/predict.py`；
2. 对每个文件校验本文记录的 SHA256；
3. 为六张图逐张生成结果 JSON 和标注图；
4. 运行作者实现对照，成功时输出 `REFERENCE_VERIFICATION_OK`。

```bash
VENV="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/venv" \
MODEL_DIR="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0" \
GH="$HOME/.local/bin/gh" \
tools/reproduce_reference_verification.sh
```

`GH` 仅在 `gh` 不在 `PATH` 中时需要设置。2026-11-18 已在一个空的
`/tmp/edge-ai-clean-repro` 下实际执行通过：模型、作者脚本和六张图片的
SHA256 全部校验通过，六图推理完成，最后输出 `PARITY: IDENTICAL` 和
`REFERENCE_VERIFICATION_OK`。下面的 7.2–7.4 是脚本对应的手工步骤。

### 7.1 Python 环境

本次虚拟环境没有系统 pip，实际采用的命令如下：

```bash
VENV="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/venv"
python3 -m venv --without-pip "$VENV"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$VENV/bin/python" /tmp/get-pip.py
"$VENV/bin/python" -m pip install -r tools/requirements-reference.txt
```

如果系统已安装 `python3-venv`（`ensurepip` 可用），也可以直接：

```bash
python3 -m venv "$VENV"
"$VENV/bin/pip" install -r tools/requirements-reference.txt
```

### 7.2 下载模型与全部六张测试图

```bash
MODEL_DIR="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0"
MODEL_REPO="winthegame259/steel-surface-defect-detection"
IMAGE_REPO="siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection"
IMAGE_COMMIT="351bee21b56bbf97a1e1d18262bceb202eb7ef41"

# 参考实现、结果 JSON 和标注图目录必须先创建（含作者对照用的 reference-src）。
mkdir -p "$MODEL_DIR/test-images" "$MODEL_DIR/reference-src" \
         "$MODEL_DIR/verification-results/json" \
         "$MODEL_DIR/verification-results/annotated"

# 本环境 github.com 与 raw.githubusercontent.com 不稳定，统一走 api.github.com；
# 需要已登录的 gh。也可改用 release/raw 的 curl 地址下载同一文件。
ASSET_ID=$(gh api "repos/$MODEL_REPO/releases/tags/v1.0.0" \
  --jq '.assets[] | select(.name=="neu_det_yolov8n_best.onnx") | .id')
gh api "repos/$MODEL_REPO/releases/assets/$ASSET_ID" \
  -H "Accept: application/octet-stream" > "$MODEL_DIR/neu_det_yolov8n_best.onnx"
echo "495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1  $MODEL_DIR/neu_det_yolov8n_best.onnx" \
  | sha256sum -c -

# 六张图逐张下载并校验；每张图都必须有对应 JSON，才能做六图 author 对照。
while read -r name sha; do
  gh api "repos/$IMAGE_REPO/contents/IMAGES/$name.jpg?ref=$IMAGE_COMMIT" \
    -H "Accept: application/vnd.github.raw" > "$MODEL_DIR/test-images/$name.jpg"
  echo "$sha  $MODEL_DIR/test-images/$name.jpg" | sha256sum -c -
done <<'EOF'
crazing_101 0c4ccbfd539aa9d0a151f25b3e798fbad36d2a3532218bcc5c313f06c2c1dda4
scratches_1 500c46e2f5d7065d8c492b0cfb36c3ba77a85a744079bf6453dfc85e02b97a3c
patches_1 0ea4843aa7801ce5a5e672ec8c0266d4910aa7d740955f38d087b38154d0fe67
inclusion_1 2b0a6eff755d5c5867d3e3ca6b26af9548813e50c9b12b0f3381a49560a42634
pitted_surface_1 38a0ece141350f5bed53c99da4d96ce25d42b3ed4248e536a579ab823abfb318
rolled-in_scale_1 e9d0358a2859da86a7a9cc75deb5c235fcb8b14e8773cd92e94593b0289736fd
EOF
```

### 7.3 六图参考推理（逐图生成 JSON）

```bash
cd /path/to/industrial-defect-monitor
for name in crazing_101 scratches_1 patches_1 inclusion_1 pitted_surface_1 rolled-in_scale_1; do
  "$VENV/bin/python" -I tools/neu_det_reference.py \
    --model "$MODEL_DIR/neu_det_yolov8n_best.onnx" \
    --image "$MODEL_DIR/test-images/$name.jpg" \
    --conf 0.25 --iou 0.45 --imgsz 640 \
    --expected-sha256 495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1 \
    --json-out "$MODEL_DIR/verification-results/json/$name.json" \
    --image-out "$MODEL_DIR/verification-results/annotated/$name.jpg"
done
```

### 7.4 与作者原实现对比

```bash
# 作者脚本目录已在 7.2 创建；这里再次显式创建，保证本步骤可单独执行。
mkdir -p "$MODEL_DIR/reference-src"
gh api \
  "repos/winthegame259/steel-surface-defect-detection/contents/src/predict.py?ref=1226b2d8a241b3ecacb61d9e727051ccb8baa64e" \
  -H "Accept: application/vnd.github.raw" > "$MODEL_DIR/reference-src/src_predict.py"
echo "84dfc96d0a8bfe9a1b4474b32a2e53c7e94b4ebba2f615269bc5f938b930c130  $MODEL_DIR/reference-src/src_predict.py" \
  | sha256sum -c -

"$VENV/bin/python" -I tools/compare_author_predict.py \
  --author-predict "$MODEL_DIR/reference-src/src_predict.py" \
  --model "$MODEL_DIR/neu_det_yolov8n_best.onnx" \
  --image-dir "$MODEL_DIR/test-images" \
  --results-dir "$MODEL_DIR/verification-results/json"
```

## 8. 参考推理结果

### 主图结果（`crazing_101.jpg`，conf>0.25，IoU=0.45）

原始输出中有 14 个候选满足 `max_score > 0.25`，经类别感知 NMS 后保留 2 个：

| 类别 ID | 类别名 | 置信度 | 原图 `bbox_xyxy` |
| --- | --- | --- | --- |
| 0 | `crazing` | 0.525821 | `[48.01, 37.25, 195.25, 94.72]` |
| 0 | `crazing` | 0.484310 | `[4.32, 108.10, 180.44, 199.27]` |

六张图的汇总（完整框坐标见 `docs/reference-inference-results.json`）：

| 图片 | 检测数 | 类别 |
| --- | --- | --- |
| `crazing_101.jpg` | 2 | `crazing` |
| `scratches_1.jpg` | 2 | `scratches` |
| `patches_1.jpg` | 3 | `patches` |
| `inclusion_1.jpg` | 4 | `inclusion` |
| `pitted_surface_1.jpg` | 2 | `pitted_surface` |
| `rolled-in_scale_1.jpg` | 2 | `rolled-in_scale` |

没有出现空结果；该结果只说明模型能在这些固定样例上输出检测，不代表精度
指标或泛化能力已经过独立评估。

### 与作者实现的一致性

`tools/compare_author_predict.py` 直接调用作者 `src/predict.py` 中的
`preprocess_onnx` / `postprocess_onnx`，与本地脚本结果逐项比较：

```text
crazing_101.jpg: author=2 ours=2 class_ids_match=True max_abs_diff=0.000e+00 -> IDENTICAL
inclusion_1.jpg: author=4 ours=4 class_ids_match=True max_abs_diff=0.000e+00 -> IDENTICAL
patches_1.jpg: author=3 ours=3 class_ids_match=True max_abs_diff=0.000e+00 -> IDENTICAL
pitted_surface_1.jpg: author=2 ours=2 class_ids_match=True max_abs_diff=0.000e+00 -> IDENTICAL
rolled-in_scale_1.jpg: author=2 ours=2 class_ids_match=True max_abs_diff=0.000e+00 -> IDENTICAL
scratches_1.jpg: author=2 ours=2 class_ids_match=True max_abs_diff=0.000e+00 -> IDENTICAL
PARITY: IDENTICAL
```

### CPU 延迟（参考，不作为验收指标）

同一台 4 vCPU 虚拟机、CPUExecutionProvider、640×640 输入，预热 10 次后
统计 30 次纯 `session.run`（不含预处理/后处理）：

```text
mean=80.76 ms  p50=75.52 ms  p95=112.46 ms  min=63.83 ms  max=140.85 ms
```

作者 README 报告的 14.6 ms 是在不同硬件/线程配置下得到的，不能直接与本次
虚拟机数据比较。

### 结果文件

| 文件 | 是否提交 | 说明 |
| --- | --- | --- |
| `docs/reference-inference-results.json` | 是 | 六张图的完整检测结果、模型 metadata、环境版本 |
| `docs/reference-inference-run.txt` | 是 | 实际运行命令与 stdout（路径已脱敏为 `$MODEL_DIR`） |
| `$MODEL_DIR/verification-results/annotated/*.jpg` | 否 | 标注图，受数据集许可限制，仅保存在仓库外 |
| `$MODEL_DIR/neu_det_yolov8n_best.onnx` | 否 | 模型权重，仅保存在仓库外 |

## 9. 尚未核实 / 待验证事项

1. **NEU-DET 数据集许可**：原始发布页只有引用请求，仓库声明“仅用于学术
   研究”，没有明确许可证文本。商用与再分发权利待确认。
2. **权重许可**：仓库和 ONNX metadata 均标为 AGPL-3.0，但没有单独权重
   许可文件；商业闭源部署是否可行待法务/授权确认。
3. **彩色图像的颜色顺序**：参考实现 BGR 直送；本次六张 NEU-DET 图是灰度
   （B=G=R），无法验证 RGB/BGR 在彩色输入上的差异。2B 集成前需决定并测试。
4. **精度指标**：作者报告的 mAP50=0.774 等未在本次核验中复测；本次只验证
   加载、接口和固定图输出。
5. **ONNX 溯源细节**：release 文件 12.2 MB 与实验日志 42.7 MB 不一致；
   作者未提供 release 对应的精确导出命令。当前以实际 SHA256 文件为准。
6. **NMS 只覆盖 6 张图**：阈值 0.25/0.45 是作者参考默认值，不是模型
   metadata；不同阈值下的行为、密集/重叠缺陷场景尚未系统测试。
7. **C++ 集成**：任务 2B 尚未开始，ONNX Runtime C++ API 的加载失败路径、
   错误码和资源释放仍需实现与测试。
8. **模型适用域**：仅 NEU-DET 热轧钢带表面；其他材质、光照、相机和缺陷
   类型的泛化能力未知。
9. **测试图片再分发权**：镜像仓库无 license，图片未提交；若后续需要把
   测试图纳入 CI，必须先取得明确授权或改用可再分发的替代图片。
10. **SRS FR-007 对齐**：核验脚本对未知类别 ID 采用 fail-fast；SRS 要求
    显示 `Unknown` 并记录异常。2B 的 C++ 实现应按 SRS 执行，或先记录需求
    变更后调整，不能静默映射。

## 10. 参考链接

- 模型仓库（tag 对应 commit）：<https://github.com/winthegame259/steel-surface-defect-detection/tree/1226b2d8a241b3ecacb61d9e727051ccb8baa64e>
- Release v1.0.0：<https://github.com/winthegame259/steel-surface-defect-detection/releases/tag/v1.0.0>
- 参考预处理/后处理：<https://github.com/winthegame259/steel-surface-defect-detection/blob/1226b2d8a241b3ecacb61d9e727051ccb8baa64e/src/predict.py#L67-L160>
- 参考 C++ 推理：<https://github.com/winthegame259/steel-surface-defect-detection/blob/1226b2d8a241b3ecacb61d9e727051ccb8baa64e/cpp_inference/inference.cpp>
- 训练/导出配置：<https://github.com/winthegame259/steel-surface-defect-detection/blob/1226b2d8a241b3ecacb61d9e727051ccb8baa64e/configs/neu_det.yaml>
- 实验日志（ONNX 导出记录）：<https://github.com/winthegame259/steel-surface-defect-detection/blob/1226b2d8a241b3ecacb61d9e727051ccb8baa64e/docs/experiment_log.md>
- NEU-DET 官方主页：<http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm>
- 测试图片镜像：<https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/tree/351bee21b56bbf97a1e1d18262bceb202eb7ef41>
- ONNX Runtime CPU 安装/文档：<https://onnxruntime.ai/docs/execution-providers/CPU-Execution-Provider.html>
