# Edge AI 工业缺陷检测系统 —— 阶段一：工程骨架

面向边缘设备的工业缺陷检测系统。当前阶段只建立可独立构建、运行和测试的
C++17 工程骨架，包含可执行文件 `edge_ai` 的基础命令行接口与 CTest 测试；
摄像头、模型推理和网络模块将在后续阶段加入。

## 已验证环境

以下为实际执行验证的环境（2026-11-18）：

| 项目 | 实际值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.5 LTS (Noble Numbat) |
| 架构 | x86_64 |
| CPU | AMD Ryzen 5 6600U（VMware 虚拟机，4 vCPU） |
| 内存 | 5.7 GiB（另有 4.0 GiB swap） |
| GCC / G++ | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| CMake | 3.28.3 |
| GDB | 15.1 |
| Git | 2.43.0 |
| Ninja | 1.11.1 |
| OpenCV | 本机尚未安装；当前代码不链接 OpenCV，安装方式见下 |

> OpenCV 在本阶段被 CMake 以可选依赖方式探测，不参与编译。摄像头与图像
> 预处理模块接入时需要安装 OpenCV 4.x（Ubuntu 24.04 官方源版本为 4.6.0）。

## 依赖

必需：

- GCC/G++，支持 C++17
- CMake >= 3.20
- Git

开发与后续阶段：

- GDB（调试）
- Ninja（可选，更快的构建后端；也可使用默认的 Makefiles）
- OpenCV 4.x（可选，后续图像/摄像头模块使用）

在 Ubuntu 上安装：

```bash
sudo apt update
sudo apt install -y build-essential cmake gdb git ninja-build pkg-config libopencv-dev
```

## 构建

从干净构建目录配置并编译：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

如果没有安装 Ninja，去掉 `-G Ninja` 即可使用默认生成器：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成的可执行文件位于 `build/bin/edge_ai`。工程启用 C++17
（`CMAKE_CXX_STANDARD 17`，禁用编译器扩展），并为 GCC/Clang 开启
`-Wall -Wextra -Wpedantic`。

## 运行

```bash
# 查看帮助
./build/bin/edge_ai --help

# 查看版本
./build/bin/edge_ai --version

# 无参数：打印启动横幅
./build/bin/edge_ai

# 非法参数：向 stderr 输出错误并以非零码（2）退出
./build/bin/edge_ai --bogus
```

`--help` / `-h` 退出码为 0；未知选项或多余位置参数会输出明确错误并返回
退出码 2。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

CTest 中的 `cli_behavior` 测试会真实执行 `edge_ai`，检查：

1. `--help` 退出码为 0，且输出包含用法说明；
2. 未知选项退出码非 0，stderr 包含错误原因与 `--help` 提示；
3. 多余位置参数退出码非 0；
4. 无参数运行时退出码为 0，并打印启动横幅。

## 目录结构

```text
.
├── CMakeLists.txt          # CMake 构建定义（C++17、目标 edge_ai、CTest）
├── README.md
├── .gitignore              # 忽略构建产物、本地配置与模型权重
├── include/
│   └── system_info.hpp     # 系统信息模块接口
├── src/
│   ├── main.cpp            # edge_ai 入口与命令行参数处理
│   └── system_info.cpp     # 系统信息模块实现
└── tests/
    └── cli_tests.cmake     # CTest 调用的 CLI 行为测试脚本
```

构建目录 `build/` 完全由 CMake 生成，不纳入版本库。模型权重、密钥和本地
环境文件同样被 `.gitignore` 排除。

## 验收记录

```text
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
-- OpenCV not found (optional at this stage)
-- Configuring done
-- Generating done

$ cmake --build build
[4/4] Linking CXX executable bin/edge_ai

$ ./build/bin/edge_ai --help
edge_ai - Industrial defect monitoring system (edge AI)

Usage:
  edge_ai [OPTION]

Options:
  -h, --help     Show this help message and exit
  -V, --version  Show version information and exit

With no options, edge_ai prints its startup banner.
# exit code: 0

$ ./build/bin/edge_ai --bogus
edge_ai: unknown option '--bogus'
Try 'edge_ai --help' for more information.
# exit code: 2

$ ctest --test-dir build --output-on-failure
1/1 Test #1: cli_behavior .....................   Passed
100% tests passed, 0 tests failed out of 1
```
